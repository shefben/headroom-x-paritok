"""Wire-shape handling shared by every tool-result transform.

Two transforms now rewrite tool-result bodies —
:mod:`headroom.transforms.tool_result_pruning` (rule-based) and
:mod:`headroom.transforms.observation_masking` (age-based) — and a third,
:mod:`headroom.transforms.toon_encoding`, re-serialises them. All three need
exactly the same unglamorous things: find every tool result across both wire
shapes, read its text whatever the content nesting, write a replacement back
without changing that nesting, and park the original in the CCR store so
``headroom_retrieve`` can hand it back.

Keeping one copy of that matters more than it looks. The two wire shapes
(Anthropic ``tool_result`` blocks inside a user message, OpenAI ``role="tool"``
messages) and the two content shapes (bare string, list of typed blocks) give
four combinations, and a transform that handles three of them corrupts the
fourth silently — the request still forwards, the block just stops saying what
it said. One implementation, tested once, is the only version of this worth
having.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CCR_MARKER",
    "ERROR_RE",
    "ToolResultCandidate",
    "block_text",
    "collect_candidates",
    "deep_copy_message",
    "message_text",
    "store_original",
    "tool_use_index",
    "write_candidate",
]

# A CCR marker means an earlier stage already owns this block's bytes.
CCR_MARKER = re.compile(r"<<ccr:[a-f0-9]{12,24}\b")

# Signals that a result is an error even when the block lacks ``is_error``.
# No trailing ``\b``: most of these alternatives end in ``)`` or ``:``, and a
# word boundary after a non-word character requires a word character next —
# which is exactly what does NOT follow "Traceback (most recent call last):".
ERROR_RE = re.compile(
    r"\b(?:traceback \(most recent call last\)|segmentation fault|panic:|fatal:"
    r"|error:|exception:|assertionerror|permission denied|no such file)",
    re.IGNORECASE,
)


@dataclass
class ToolResultCandidate:
    """One tool result located in the message array."""

    message_index: int
    block_index: int | None
    """Index within a content list, or ``None`` for an OpenAI ``role="tool"``
    message whose whole ``content`` is the result."""
    tool_use_id: str
    tool_name: str
    text: str
    is_error: bool
    content_was_list: bool


def block_text(content: Any) -> str:
    """Flatten a tool_result ``content`` value to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def message_text(message: Any) -> str:
    """All text carried by a message, whatever its content shape."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "thinking"):
            value = block.get(key)
            if isinstance(value, str):
                parts.append(value)
        if block.get("type") == "tool_use":
            with contextlib.suppress(TypeError, ValueError):
                parts.append(json.dumps(block.get("input"), sort_keys=True, default=str))
        if block.get("type") == "tool_result":
            parts.append(block_text(block.get("content")))
    return "\n".join(parts)


def tool_use_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Map ``tool_use_id`` to ``(tool_name, canonical_input_json)``.

    Covers both wire shapes: Anthropic ``tool_use`` content blocks and OpenAI
    assistant ``tool_calls``.
    """
    index: dict[str, tuple[str, str]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                block_id = block.get("id")
                if not isinstance(block_id, str):
                    continue
                try:
                    payload = json.dumps(block.get("input"), sort_keys=True, default=str)
                except (TypeError, ValueError):
                    payload = ""
                index[block_id] = (str(block.get("name") or ""), payload)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            if not isinstance(call_id, str):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            index[call_id] = (
                str(function.get("name") or ""),
                str(function.get("arguments") or ""),
            )
    return index


def collect_candidates(
    messages: list[dict[str, Any]],
    start: int,
    stop: int,
    index: dict[str, tuple[str, str]],
) -> list[ToolResultCandidate]:
    """Find every tool result in ``messages[start:stop]``."""
    candidates: list[ToolResultCandidate] = []
    for message_index in range(start, stop):
        message = messages[message_index]
        if not isinstance(message, dict):
            continue

        if message.get("role") == "tool":
            content = message.get("content")
            text = block_text(content)
            if not text:
                continue
            call_id = str(message.get("tool_call_id") or "")
            candidates.append(
                ToolResultCandidate(
                    message_index=message_index,
                    block_index=None,
                    tool_use_id=call_id,
                    tool_name=index.get(call_id, ("", ""))[0]
                    or str(message.get("name") or ""),
                    text=text,
                    is_error=bool(ERROR_RE.search(text)),
                    content_was_list=isinstance(content, list),
                )
            )
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            text = block_text(inner)
            if not text:
                continue
            use_id = str(block.get("tool_use_id") or "")
            candidates.append(
                ToolResultCandidate(
                    message_index=message_index,
                    block_index=block_index,
                    tool_use_id=use_id,
                    tool_name=index.get(use_id, ("", ""))[0],
                    text=text,
                    is_error=bool(block.get("is_error")) or bool(ERROR_RE.search(text)),
                    content_was_list=isinstance(inner, list),
                )
            )
    return candidates


def write_candidate(
    messages: list[dict[str, Any]],
    candidate: ToolResultCandidate,
    text: str,
) -> None:
    """Replace a candidate's body in place, preserving its wire shape."""
    message = messages[candidate.message_index]
    replacement: Any = [{"type": "text", "text": text}] if candidate.content_was_list else text
    if candidate.block_index is None:
        message["content"] = replacement
        return
    block = message["content"][candidate.block_index]
    block["content"] = replacement


def deep_copy_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a message deeply enough that rewriting a block cannot alias input."""
    copied = dict(message)
    content = copied.get("content")
    if isinstance(content, list):
        copied["content"] = [
            dict(block) if isinstance(block, dict) else block for block in content
        ]
    return copied


def store_original(  # noqa: ANN201
    original: str,
    replacement: str,
    original_tokens: int,
    tool_name: str,
    tool_call_id: str,
    strategy: str,
):
    """Write a replaced body to the CCR store; return its hash or ``None``.

    Mirrors :func:`headroom.paritok.refstore.store_paritok_in_ccr` so every
    stage shares one CCR policy. ``None`` means the caller must leave the
    original in place — a marker with nothing behind it is worse than no
    transform at all.
    """
    try:
        from headroom.cache.compression_store import get_compression_store

        store = get_compression_store()
        return store.store(
            original,
            replacement,
            original_tokens=original_tokens,
            compressed_tokens=len(replacement.split()),
            original_item_count=original_tokens,
            compressed_item_count=len(replacement.split()),
            tool_name=tool_name or None,
            tool_call_id=tool_call_id or None,
            compression_strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001 - a store failure degrades, never fails
        logger.debug("%s: CCR store failed (%s)", strategy, exc)
        return None
