"""Anthropic server-side context editing (``clear_tool_uses_20250919``).

Every other lever in Headroom rewrites bytes *before* they leave the proxy.
This one is different: it asks the provider to drop stale tool-use/tool-result
pairs on its own side, replacing each with a short placeholder. That matters
for two reasons the client-side compressors cannot match:

1. **It is free.** The cleared bytes are never billed as input tokens, and the
   proxy spends no model call, no embedding and no CPU to remove them.
2. **It runs after the prompt-cache lookup.** Anthropic evaluates cache hits
   against the request as sent, then applies the edits. Stripping the same
   history client-side would change the cached prefix and cost a full cache
   miss; letting the provider do it does not.

The trade Anthropic does *not* advertise: once an edit actually fires, the
suffix after the clear point differs from what was cached, so subsequent turns
re-cache from there. That is why :data:`DEFAULT_CLEAR_AT_LEAST_TOKENS` is
non-zero — a clear that frees a handful of tokens is strictly worse than no
clear at all, because it pays a re-cache to save nothing. ``clear_at_least``
is the provider-side guard against exactly that.

Scope: first-party Anthropic only. Custom Anthropic-compatible gateways reject
the unknown ``context_management`` body field, so callers must gate on
:func:`headroom.proxy.helpers.anthropic_first_party_tool_search_supported`
(same first-party test the tool-search deferral uses).

Client requests win. When the caller already sent a ``context_management``
block containing a ``clear_tool_uses`` edit, Headroom does not touch it — a
harness that configured its own retention policy knows more about its
transcript than we do.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Beta token required for the ``context_management`` request field. Same token
# the native memory tool already needs (see ``proxy/memory_tool_adapter.py``),
# so on a memory-enabled turn the merge is a no-op.
CONTEXT_EDITING_BETA = "context-management-2025-06-27"

# Versioned edit type. Anthropic pins behavior to the date suffix, so this is a
# constant rather than a knob: a different date is a different contract.
CLEAR_TOOL_USES_EDIT_TYPE = "clear_tool_uses_20250919"

# Any edit whose type starts with this prefix is "a clear_tool_uses edit" for
# the purpose of detecting a client-supplied one, whatever version it pins.
_CLEAR_TOOL_USES_PREFIX = "clear_tool_uses"

ENV_ENABLED = "HEADROOM_CONTEXT_EDITING"
ENV_TRIGGER_TOKENS = "HEADROOM_CONTEXT_EDITING_TRIGGER_TOKENS"
ENV_KEEP_TOOL_USES = "HEADROOM_CONTEXT_EDITING_KEEP_TOOL_USES"
ENV_CLEAR_AT_LEAST_TOKENS = "HEADROOM_CONTEXT_EDITING_CLEAR_AT_LEAST_TOKENS"
ENV_CLEAR_TOOL_INPUTS = "HEADROOM_CONTEXT_EDITING_CLEAR_TOOL_INPUTS"
ENV_EXCLUDE_TOOLS = "HEADROOM_CONTEXT_EDITING_EXCLUDE_TOOLS"

# Anthropic's own default trigger. Left as-is rather than lowered: below this,
# most coding turns still fit comfortably and the re-cache is not yet worth it.
DEFAULT_TRIGGER_TOKENS = 100_000

# Anthropic's default retention. Three tool uses is enough for the model to see
# what it just did without carrying the whole session.
DEFAULT_KEEP_TOOL_USES = 3

# Headroom's addition. Anthropic defaults this to 0, which permits a clear that
# frees almost nothing while still invalidating the cached suffix.
DEFAULT_CLEAR_AT_LEAST_TOKENS = 5_000

# Tools whose results are never worth clearing. ``headroom_retrieve`` is the
# CCR expansion path: its result IS the recovered original the model asked for,
# so clearing it re-creates the gap the model just paid a turn to close.
DEFAULT_EXCLUDE_TOOLS: tuple[str, ...] = ("headroom_retrieve",)

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Rollout feature that gates the lever (see headroom.rollout.FEATURES).
FEATURE_CONTEXT_EDITING = "anthropic_context_editing"

# Resolved once per process, like the Paritok tool filter: re-reading the
# rollout per request would let one conversation change policy mid-session.
_enabled: bool | None = None


def context_editing_enabled() -> bool:
    """Whether the proxy should request server-side context editing."""
    global _enabled

    if _enabled is None:
        from headroom.rollout import feature_enabled

        _enabled = feature_enabled(FEATURE_CONTEXT_EDITING)
    return _enabled


def reset_context_editing() -> None:
    """Drop the cached rollout resolution (tests, ``POST /admin/runtime-env``)."""
    global _enabled

    _enabled = None


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Invalid %s=%r (min %d); using %d", name, raw, minimum, default)
        return default
    return value


def _env_tools(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    names = tuple(part.strip() for part in raw.replace(";", ",").split(",") if part.strip())
    return names


@dataclass(frozen=True)
class ContextEditingSettings:
    """Resolved tuning for the ``clear_tool_uses`` edit.

    ``enabled`` here is only the *tuning-level* switch. Whether the lever is
    live is decided by the rollout snapshot (feature
    ``anthropic_context_editing``); this flag exists so the same env name can
    act as the legacy alias without a second parser.
    """

    enabled: bool = False
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS
    keep_tool_uses: int = DEFAULT_KEEP_TOOL_USES
    clear_at_least_tokens: int = DEFAULT_CLEAR_AT_LEAST_TOKENS
    clear_tool_inputs: bool = False
    exclude_tools: tuple[str, ...] = DEFAULT_EXCLUDE_TOOLS

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> ContextEditingSettings:
        """Build settings from ``HEADROOM_CONTEXT_EDITING*``.

        ``enabled`` overrides the env switch; callers that already resolved the
        rollout feature pass the resolved value so the two cannot disagree.
        """
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            trigger_tokens=_env_int(ENV_TRIGGER_TOKENS, DEFAULT_TRIGGER_TOKENS, minimum=1),
            keep_tool_uses=_env_int(ENV_KEEP_TOOL_USES, DEFAULT_KEEP_TOOL_USES, minimum=0),
            clear_at_least_tokens=_env_int(
                ENV_CLEAR_AT_LEAST_TOKENS, DEFAULT_CLEAR_AT_LEAST_TOKENS, minimum=0
            ),
            clear_tool_inputs=_env_bool(ENV_CLEAR_TOOL_INPUTS, False),
            exclude_tools=_env_tools(ENV_EXCLUDE_TOOLS, DEFAULT_EXCLUDE_TOOLS),
        )

    def build_edit(self) -> dict[str, Any]:
        """Render the settings as one Anthropic ``edits[]`` entry.

        Optional members are omitted when they carry Anthropic's own default so
        the body stays as close to what the client would have sent as possible.
        """
        edit: dict[str, Any] = {
            "type": CLEAR_TOOL_USES_EDIT_TYPE,
            "trigger": {"type": "input_tokens", "value": self.trigger_tokens},
            "keep": {"type": "tool_uses", "value": self.keep_tool_uses},
        }
        if self.clear_at_least_tokens > 0:
            edit["clear_at_least"] = {
                "type": "input_tokens",
                "value": self.clear_at_least_tokens,
            }
        if self.clear_tool_inputs:
            edit["clear_tool_inputs"] = True
        if self.exclude_tools:
            edit["exclude_tools"] = list(self.exclude_tools)
        return edit


@dataclass(frozen=True)
class ContextEditingResult:
    """Outcome of one :func:`apply_context_editing` call."""

    applied: bool
    reason: str
    edit: dict[str, Any] | None = None
    tool_use_count: int = 0


def count_tool_uses(messages: Any) -> int:
    """Count ``tool_use`` blocks across an Anthropic ``messages`` array.

    Counts the *uses*, not the results: ``keep`` is denominated in tool uses,
    so this is the quantity that decides whether an edit could ever fire.
    Tolerates the string-content shape and any non-dict junk in the array.
    """
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                total += 1
    return total


def _has_clear_tool_uses_edit(edits: Any) -> bool:
    if not isinstance(edits, list):
        return False
    for edit in edits:
        if isinstance(edit, dict) and str(edit.get("type", "")).startswith(
            _CLEAR_TOOL_USES_PREFIX
        ):
            return True
    return False


def merge_edit(body: dict[str, Any], edit: dict[str, Any], type_prefix: str) -> str:
    """Append one edit to ``body["context_management"]["edits"]`` in place.

    Shared by every provider-side edit Headroom knows how to request — the
    ``clear_tool_uses`` lever here and the compaction lever in
    :mod:`headroom.proxy.context_compaction` — because they write the same
    array and must not clobber each other or the client.

    Returns the reason string the caller reports:

    ``client_malformed``
        ``context_management`` is present but not an object. Not ours to
        repair; forwarding it unchanged keeps the provider's error accurate.
    ``client_configured``
        The caller already sent an edit whose type starts with ``type_prefix``.
        A harness that configured its own policy knows more than we do.
    ``applied``
        Merged. Any unrelated client edit in the array is preserved.
    """
    existing = body.get("context_management")
    if existing is not None and not isinstance(existing, dict):
        return "client_malformed"

    edits = existing.get("edits") if isinstance(existing, dict) else None
    if isinstance(edits, list):
        for entry in edits:
            if isinstance(entry, dict) and str(entry.get("type", "")).startswith(type_prefix):
                return "client_configured"

    if isinstance(existing, dict):
        merged = list(edits) if isinstance(edits, list) else []
        merged.append(edit)
        body["context_management"] = {**existing, "edits": merged}
    else:
        body["context_management"] = {"edits": [edit]}
    return "applied"


def apply_context_editing(
    body: dict[str, Any],
    settings: ContextEditingSettings,
) -> ContextEditingResult:
    """Merge Headroom's ``clear_tool_uses`` edit into ``body`` in place.

    Returns without touching ``body`` when the lever is off, when the client
    already configured its own ``clear_tool_uses`` edit, or when the transcript
    holds no more tool uses than ``keep`` would retain anyway — in that last
    case the edit provably cannot fire, and adding a body field that changes
    nothing only widens the surface for a gateway to reject the request.
    """
    if not settings.enabled:
        return ContextEditingResult(applied=False, reason="disabled")

    tool_use_count = count_tool_uses(body.get("messages"))
    if tool_use_count <= settings.keep_tool_uses:
        return ContextEditingResult(
            applied=False, reason="below_keep_threshold", tool_use_count=tool_use_count
        )

    edit = settings.build_edit()
    reason = merge_edit(body, edit, _CLEAR_TOOL_USES_PREFIX)
    if reason != "applied":
        return ContextEditingResult(
            applied=False, reason=reason, tool_use_count=tool_use_count
        )

    return ContextEditingResult(
        applied=True, reason="applied", edit=edit, tool_use_count=tool_use_count
    )


def cleared_input_tokens(usage: Any) -> int:
    """Total input tokens the provider reported clearing on this response.

    Anthropic echoes what it actually did under
    ``usage.context_management.applied_edits[]``; each entry carries
    ``cleared_input_tokens``. Absent or unparseable usage reads as zero — this
    feeds a savings tag, never a control decision.
    """
    if not isinstance(usage, dict):
        return 0
    management = usage.get("context_management")
    if not isinstance(management, dict):
        return 0
    applied = management.get("applied_edits")
    if not isinstance(applied, list):
        return 0
    total = 0
    for entry in applied:
        if not isinstance(entry, dict):
            continue
        value = entry.get("cleared_input_tokens")
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total


def cleared_tool_uses(usage: Any) -> int:
    """Total tool uses the provider reported clearing on this response."""
    if not isinstance(usage, dict):
        return 0
    management = usage.get("context_management")
    if not isinstance(management, dict):
        return 0
    applied = management.get("applied_edits")
    if not isinstance(applied, list):
        return 0
    total = 0
    for entry in applied:
        if not isinstance(entry, dict):
            continue
        value = entry.get("cleared_tool_uses")
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total
