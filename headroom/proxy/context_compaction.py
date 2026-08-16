"""Anthropic server-side compaction (``compact_20260112``).

The sibling of :mod:`headroom.proxy.context_editing`, and the second lever in
Headroom that spends the provider's compute instead of its own. Where
``clear_tool_uses`` *deletes* stale tool results, compaction *summarises* the
older part of the conversation into a compaction block and continues from
there, so the transcript keeps its meaning rather than acquiring holes.

Two reasons this is worth having on top of the client-side history summary the
Paritok stage already performs:

1. **Tool pairing is the provider's problem, not ours.** The single most common
   way a home-grown compactor breaks a session is by summarising away a
   ``tool_use`` block whose ``tool_result`` survives; the next request 400s on
   an orphaned ``tool_use_id``. Anthropic's implementation maintains that
   pairing and the user/assistant alternation by construction.
2. **It runs after the prompt-cache lookup**, exactly like context editing, so
   asking for it does not itself cost a cache miss.

The cost this lever has and context editing does not
----------------------------------------------------
Compaction puts a **compaction block into the assistant response**, and the
conversation only continues from the compacted state if that block is echoed
back on the next request. Headroom does not own the client's transcript — the
client rebuilds the message array from its own history every turn. So:

* A client that passes unknown assistant content blocks through unchanged
  (Claude Code does) gets the full benefit.
* A client that drops them loses the benefit and re-sends the raw history.
  That is wasted provider work, not a broken session — the next turn simply
  compacts again.
* A client that *validates* assistant content block types strictly could
  reject the response outright.

That third case is why this lever is off by default and gated on the model
family rather than merely on the endpoint. Enable it when you know the client.

Scope: first-party Anthropic only, same test the tool-search deferral and the
context-editing lever use, because ``context_management`` is an unknown body
field to custom Anthropic-compatible gateways and they 400 on it.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from headroom.proxy.context_editing import merge_edit

logger = logging.getLogger(__name__)

__all__ = [
    "COMPACTION_BETA",
    "COMPACT_EDIT_TYPE",
    "CompactionResult",
    "CompactionSettings",
    "apply_compaction",
    "compaction_enabled",
    "model_supports_compaction",
    "reset_compaction",
]

# Beta token required for the compaction edit. Distinct from the context-editing
# token: a request may legitimately carry both, which is why the handler merges
# rather than assigns.
COMPACTION_BETA = "compact-2026-01-12"

# Versioned edit type. Anthropic pins behaviour to the date suffix, so this is a
# constant rather than a knob — a different date is a different contract.
COMPACT_EDIT_TYPE = "compact_20260112"

# Any edit whose type starts with this prefix counts as "the client already
# configured compaction", whatever version it pinned.
_COMPACT_PREFIX = "compact"

ENV_ENABLED = "HEADROOM_CONTEXT_COMPACTION"
ENV_TRIGGER_TOKENS = "HEADROOM_CONTEXT_COMPACTION_TRIGGER_TOKENS"

# Anthropic triggers compaction near the context limit by default. Headroom
# sets an explicit trigger instead: an implicit "near the limit" moves whenever
# the model's window changes, and a lever whose firing point silently shifts
# under you is one you cannot attribute savings to.
DEFAULT_TRIGGER_TOKENS = 150_000

FEATURE_CONTEXT_COMPACTION = "anthropic_context_compaction"

# Model families that accept the compaction edit, with the first version that
# does. Anything not listed here is refused rather than attempted: an
# unsupported model returns a 400 for the whole request, so guessing forward is
# strictly worse than not asking.
_MIN_SUPPORTED: dict[str, tuple[int, int]] = {
    "opus": (4, 6),
    "sonnet": (4, 6),
}

_MODEL_RE = re.compile(r"claude-(opus|sonnet|haiku)-(\d+)[-.](\d+)")
_MODEL_MAJOR_ONLY_RE = re.compile(r"claude-(opus|sonnet|haiku)-(\d+)(?![-.\d])")

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

_enabled: bool | None = None


def compaction_enabled() -> bool:
    """Whether the proxy should request server-side compaction."""
    global _enabled

    if _enabled is None:
        from headroom.rollout import feature_enabled

        _enabled = feature_enabled(FEATURE_CONTEXT_COMPACTION)
    return _enabled


def reset_compaction() -> None:
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


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    return value if value >= minimum else default


def model_supports_compaction(model: Any) -> bool:
    """Whether ``model`` accepts the ``compact_20260112`` edit.

    Deliberately a whitelist walked forward by version rather than a blacklist:
    an unrecognised model name returns ``False``, so a new or aliased model is
    simply not compacted instead of 400-ing every request that mentions it.
    """
    if not isinstance(model, str) or not model:
        return False
    name = model.lower()

    match = _MODEL_RE.search(name)
    if match is not None:
        family, major, minor = match.group(1), int(match.group(2)), int(match.group(3))
        floor = _MIN_SUPPORTED.get(family)
        return floor is not None and (major, minor) >= floor

    # ``claude-opus-5`` style: a bare major with no minor is a later release
    # than any listed floor in the same family.
    match = _MODEL_MAJOR_ONLY_RE.search(name)
    if match is not None:
        family, major = match.group(1), int(match.group(2))
        floor = _MIN_SUPPORTED.get(family)
        return floor is not None and major > floor[0]
    return False


@dataclass(frozen=True)
class CompactionSettings:
    """Resolved tuning for the compaction edit."""

    enabled: bool = False
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> CompactionSettings:
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            trigger_tokens=_env_int(
                ENV_TRIGGER_TOKENS, DEFAULT_TRIGGER_TOKENS, minimum=1
            ),
        )

    def build_edit(self) -> dict[str, Any]:
        return {
            "type": COMPACT_EDIT_TYPE,
            "trigger": {"type": "input_tokens", "value": self.trigger_tokens},
        }


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of one :func:`apply_compaction` call."""

    applied: bool
    reason: str
    edit: dict[str, Any] | None = None


def apply_compaction(
    body: dict[str, Any],
    settings: CompactionSettings,
    *,
    model: Any = None,
) -> CompactionResult:
    """Merge the compaction edit into ``body`` in place.

    ``model`` defaults to ``body["model"]`` so callers that already resolved a
    routed model can pass it explicitly and the two cannot disagree.
    """
    if not settings.enabled:
        return CompactionResult(applied=False, reason="disabled")

    target = body.get("model") if model is None else model
    if not model_supports_compaction(target):
        return CompactionResult(applied=False, reason="model_unsupported")

    edit = settings.build_edit()
    reason = merge_edit(body, edit, _COMPACT_PREFIX)
    if reason != "applied":
        return CompactionResult(applied=False, reason=reason)
    return CompactionResult(applied=True, reason="applied", edit=edit)


def compaction_applied(usage: Any) -> bool:
    """Whether the provider reported actually compacting on this response.

    Anthropic echoes what it did under ``usage.context_management.applied_edits``;
    an entry whose type starts with ``compact`` means the block was produced.
    Absent or unparseable usage reads as ``False`` — this feeds a telemetry tag,
    never a control decision.
    """
    if not isinstance(usage, dict):
        return False
    management = usage.get("context_management")
    if not isinstance(management, dict):
        return False
    applied = management.get("applied_edits")
    if not isinstance(applied, list):
        return False
    return any(
        isinstance(entry, dict) and str(entry.get("type", "")).startswith(_COMPACT_PREFIX)
        for entry in applied
    )
