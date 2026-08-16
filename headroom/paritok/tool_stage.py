"""Proxy stage that applies semantic tool selection to a request payload.

Mirrors the shape of
:func:`headroom.proxy.tool_schema_compaction.compact_tools` —
``(payload) -> (payload, modified, before_bytes, after_bytes)`` — so the proxy
handlers can slot it in beside the existing compaction layers with the same
bookkeeping.

Ordering matters: this runs *before* Headroom's compaction layers. Selection
removes whole schemas, compaction shrinks the ones that remain; doing it in that
order means compaction never spends work on schemas that are about to be
dropped.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from headroom.paritok.config import (
    FEATURE_TOOL_FILTER,
    ParitokConfig,
    resolve_paritok_config,
)
from headroom.paritok.tool_filter import (
    SessionFrozenSelector,
    apply_selection_adaptive,
    looks_like_missing_tool_help,
    recover_tools_from_help,
    tool_name,
)

logger = logging.getLogger(__name__)

# One selector per process: it owns the per-session frozen selections and the
# embedding cache, both of which must survive across requests to keep the
# tools array byte-stable.
_selector: SessionFrozenSelector | None = None


def _get_selector(config: ParitokConfig) -> SessionFrozenSelector:
    global _selector

    if _selector is None:
        _selector = SessionFrozenSelector(
            alpha=config.tool_alpha,
            k_min=config.tool_k_min,
            k_max=config.tool_topk,
        )
    return _selector


# Resolved once per process, not per request. The rollout snapshot is
# process-level provenance: re-reading the environment per request would let a
# single conversation change behaviour mid-session, and the tools array would
# flip shape underneath the provider's prefix cache.
_filter_enabled: bool | None = None


def tool_filter_enabled() -> bool:
    """Whether the proxy handlers should run semantic tool selection."""
    global _filter_enabled

    if _filter_enabled is None:
        from headroom.rollout import feature_enabled

        _filter_enabled = feature_enabled(FEATURE_TOOL_FILTER)
    return _filter_enabled


def reset_selector() -> None:
    """Drop cached selections and the resolved flag.

    Used by tests and by ``POST /admin/runtime-env`` hot-sync, which is the one
    sanctioned point where the process is allowed to change its mind.
    """
    global _filter_enabled, _selector

    _selector = None
    _filter_enabled = None


def _json_byte_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")))


def last_assistant_text(messages: Any) -> str:
    """Plain text of the most recent assistant turn, or ``""``.

    This is where a missing-tool complaint shows up. Handling it on the request
    side rather than by hooking the response means one code path covers both
    streaming and non-streaming, since either way the reply is echoed back as
    conversation history on the following turn.
    """
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return " ".join(parts)
        return ""
    return ""


def _maybe_recover(
    selector: SessionFrozenSelector,
    session_id: str,
    assistant_text: str,
    tools: list[dict[str, Any]],
    config: ParitokConfig,
) -> list[str]:
    """Re-pin tools the agent just said it was missing.

    Gated on the cheap regex first: recovery embeds the entire dropped pool,
    which is far too expensive to pay on every turn. Also gated on the session
    already having a frozen selection — without one nothing was withheld, so a
    complaint is about some other limitation and pinning would be noise.
    """
    if not assistant_text or not looks_like_missing_tool_help(assistant_text):
        return []

    kept = set(selector.frozen_for(session_id))
    if not kept:
        return []

    candidates = [tool for tool in tools if tool_name(tool) not in kept]
    if not candidates:
        return []

    try:
        recovered = recover_tools_from_help(assistant_text, candidates, k=config.tool_recover_k)
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        logger.debug("paritok: tool recovery failed (%s)", exc)
        return []

    if recovered:
        selector.add_to_frozen(session_id, recovered)
        logger.info(
            "paritok: restored %s for session %s after a missing-tool reply",
            ", ".join(recovered),
            session_id,
        )
    return recovered


def select_tools(
    payload: dict[str, Any],
    *,
    session_id: str,
    query: str,
    wire: str = "anthropic",
    assistant_text: str = "",
    config: ParitokConfig | None = None,
) -> tuple[dict[str, Any], bool, int, int]:
    """Reduce ``payload["tools"]`` to the schemas relevant to this session.

    ``assistant_text`` is the previous turn's reply; when it reads as "I don't
    have that tool", the named capability is restored before selecting, so the
    filter self-heals on the turn right after a miss.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``. The
    payload is returned untouched when selection would not help — too few tools
    to bother, the embedding model is unavailable, or the result is not smaller.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    resolved = config or resolve_paritok_config()
    if len(tools) <= resolved.tool_k_min:
        # Nothing to gain, and dropping from an already-small set risks removing
        # the one tool the turn needs.
        return payload, False, 0, 0

    try:
        selector = _get_selector(resolved)
        _maybe_recover(selector, session_id, assistant_text, tools, resolved)
        keep_ordered = selector.select(session_id, query, tools)
        selected = apply_selection_adaptive(tools, keep_ordered, wire=wire)
    except Exception as exc:  # noqa: BLE001 - never fail a request over tool selection
        logger.warning("paritok: tool selection failed, sending all tools (%s)", exc)
        return payload, False, 0, 0

    if not selected:
        # A selection that keeps nothing would strip the agent of every tool.
        logger.warning("paritok: tool selection produced an empty set; keeping all tools")
        return payload, False, 0, 0

    before = _json_byte_len(tools)
    after = _json_byte_len(selected)
    if after >= before:
        return payload, False, before, after

    updated = dict(payload)
    updated["tools"] = selected
    return updated, True, before, after


def dropped_tools(
    original: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tools present in ``original`` but no longer sent in full.

    This is the candidate pool for recovery when the agent reports a missing
    capability.
    """
    kept = {tool_name(tool) for tool in current}
    return [tool for tool in original if tool_name(tool) not in kept]


def recover_for_session(
    session_id: str,
    help_text: str,
    candidates: list[dict[str, Any]],
    *,
    config: ParitokConfig | None = None,
) -> list[str]:
    """Pin the tools the agent said it was missing into this session.

    Returns the recovered names so the caller can retry the turn with them
    restored. Once pinned they stay in the frozen set, so the same miss cannot
    repeat later in the session.
    """
    if not candidates:
        return []
    resolved = config or resolve_paritok_config()
    try:
        recovered = recover_tools_from_help(help_text, candidates, k=resolved.tool_recover_k)
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        logger.debug("paritok: tool recovery failed (%s)", exc)
        return []
    if recovered:
        _get_selector(resolved).add_to_frozen(session_id, recovered)
    return recovered
