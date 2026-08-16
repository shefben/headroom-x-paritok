"""Injectors that place the edit-format block into each wire format.

Mirrors :mod:`headroom.proxy.output_steering` one-for-one — same tail-append
placement, same idempotence-by-sentinel, same "return True only when the body
actually changed" contract. Kept in its own module rather than folded into the
verbosity injectors because the two blocks must be independently replaceable:
a deployment can run strict edit-format steering with verbosity steering off,
and vice versa.

Placement is always the *tail* of the system prompt. Any ``cache_control``
breakpoint the client set sits on an earlier block, so appending after it
leaves the cached prefix byte-identical and only the small, byte-stable
steering block is reprocessed.
"""

from __future__ import annotations

from typing import Any

from headroom.proxy.output_edit_format_policy import (
    EDIT_FORMAT_SENTINEL,
    EDIT_FORMAT_SUFFIX,
    edit_format_text,
)


def replace_or_append_edit_format_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing edit-format block in text, or append one at the tail."""
    start = existing.find(EDIT_FORMAT_SENTINEL)
    if start >= 0:
        end = existing.find(EDIT_FORMAT_SUFFIX, start)
        end = len(existing) if end < 0 else end + len(EDIT_FORMAT_SUFFIX)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip("\n")
        parts = [part for part in (prefix, block, suffix) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing


def apply_edit_format_steering(body: dict[str, Any], mode: str) -> bool:
    """Append the edit-format block to an Anthropic ``system`` prompt."""
    text = edit_format_text(mode)
    if text is None:
        return False

    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": text}]
        return True
    if isinstance(system, str):
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": text},
        ]
        return True
    if isinstance(system, list):
        for block in system:
            # A malformed client block (``{"type": "text", "text": null}``)
            # must not raise here and 500 the request — same guard the
            # verbosity injector carries.
            block_text = block.get("text") if isinstance(block, dict) else None
            if isinstance(block_text, str) and block_text.startswith(EDIT_FORMAT_SENTINEL):
                if block_text == text:
                    return False
                block["text"] = text
                return True
        system.append({"type": "text", "text": text})
        return True
    return False


def apply_openai_chat_edit_format_steering(body: dict[str, Any], mode: str) -> bool:
    """Append the edit-format block to an OpenAI chat/completions body.

    Chat carries the system prompt as a ``role: "system"`` (or ``"developer"``)
    message, so it needs its own injector; the Anthropic ``system`` and
    Responses ``instructions`` variants never reach it.
    """
    text = edit_format_text(mode)
    if text is None:
        return False

    messages = body.get("messages")
    if not isinstance(messages, list):
        return False

    target: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            target = message
    if target is None:
        messages.insert(0, {"role": "system", "content": text})
        return True

    content = target.get("content")
    if content is None:
        target["content"] = text
        return True
    if isinstance(content, str):
        updated, changed = replace_or_append_edit_format_block(content, text)
        if changed:
            target["content"] = updated
        return changed
    if isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and part["text"].startswith(EDIT_FORMAT_SENTINEL)
            ):
                if part["text"] == text:
                    return False
                part["text"] = text
                return True
        content.append({"type": "text", "text": text})
        return True
    return False


def apply_openai_responses_edit_format_steering(body: dict[str, Any], mode: str) -> bool:
    """Append the edit-format block to OpenAI Responses ``instructions``."""
    text = edit_format_text(mode)
    if text is None:
        return False

    instructions = body.get("instructions")
    if instructions is None:
        body["instructions"] = text
        return True
    if not isinstance(instructions, str):
        return False
    updated, changed = replace_or_append_edit_format_block(instructions, text)
    if changed:
        body["instructions"] = updated
    return changed
