"""Paritok content compression as a Headroom pipeline transform.

Sits between CacheAligner and ContentRouter in
:meth:`~headroom.transforms.pipeline.TransformPipeline._build_default_transforms`.
That position is deliberate:

* **After CacheAligner** — the cached prefix has already been assessed, and this
  transform never rewrites inside ``frozen_message_count`` or the
  ``protect_recent`` tail. Rewriting cached bytes would invalidate the
  provider's prefix cache and cost more than the tokens saved.
* **Before ContentRouter** — whatever Paritok does not compress still flows
  through Headroom's full compressor suite (SmartCrusher, CodeCompressor,
  Kompress, and the rest). Paritok augments that pipeline; it never replaces it.

Compressed blocks carry a ``<<ccr:HASH>>`` marker, so ContentRouter recognises
them as already compressed and leaves them alone, while ``headroom_retrieve``
can still recover the original.

**Chaining.** With ``paritok_chain_model`` enabled, Paritok's output is fed
through Headroom's Kompress model before being stored, passing the *true*
original via Kompress's ``ccr_original`` parameter. Chaining inside this stage —
rather than letting ContentRouter make a second pass — is what keeps retrieval
lossless: a naive second pass would store the already-compressed text as the new
"original" and lose the only handle on the real bytes (#2694).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from headroom.config import TransformResult
from headroom.paritok.chunking import count_tokens
from headroom.paritok.config import ParitokConfig, resolve_paritok_config
from headroom.paritok.engine import ParitokEngine, get_engine
from headroom.paritok.refstore import attach_marker, store_paritok_in_ccr
from headroom.paritok.tagger import assign_level, classify_kind_from_content
from headroom.tokenizer import Tokenizer
from headroom.transforms.base import Transform
from headroom.utils import deep_copy_messages

logger = logging.getLogger(__name__)

# Same shape ccr/tool_injection.py scans for, so "has a marker" means the same
# thing here as it does to the code that advertises headroom_retrieve.
_CCR_MARKER = re.compile(r"<<ccr:[a-f0-9]{12,24}\b")

# Roles whose text is a candidate for compression. Tool output and their
# Anthropic tool_result equivalents are where a coding agent's tokens actually
# go; user and system turns are only touched when the caller opts in.
_TOOL_ROLES = frozenset({"tool", "function"})


@dataclass
class _Slot:
    """One compressible text location inside the message list."""

    message_index: int
    block_index: int | None
    """``None`` when the message's ``content`` is a bare string."""
    text: str
    role: str
    tool_name: str | None = None


def _iter_slots(
    messages: list[dict[str, Any]],
    start: int,
    stop: int,
    *,
    compress_user_messages: bool,
    compress_system_messages: bool,
) -> list[_Slot]:
    """Collect compressible text slots in ``messages[start:stop]``.

    Handles both wire shapes: OpenAI-style bare-string content and
    Anthropic-style content-block lists (``text`` / ``tool_result``).
    """
    slots: list[_Slot] = []
    for index in range(start, stop):
        message = messages[index]
        role = message.get("role", "")

        if role == "system" and not compress_system_messages:
            continue
        if role == "user" and not compress_user_messages:
            # A user turn can still carry tool_result blocks (Anthropic puts
            # them there); those are tool output, not the human's words, so
            # they stay eligible. Only bare user text is skipped.
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block_index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = _tool_result_text(block)
                    if text:
                        slots.append(_Slot(index, block_index, text, role, block.get("name")))
            continue
        if role == "assistant":
            # Assistant reasoning is handled by Headroom's thinking_compactor;
            # duplicating it here would double-compress the same tokens.
            continue

        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                slots.append(_Slot(index, None, content, role, message.get("name")))
            continue

        if not isinstance(content, list):
            continue

        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    slots.append(_Slot(index, block_index, text, role, message.get("name")))
            elif block_type == "tool_result":
                text = _tool_result_text(block)
                if text:
                    slots.append(_Slot(index, block_index, text, role, block.get("name")))

    return slots


def _tool_result_text(block: dict[str, Any]) -> str:
    """Extract plain text from an Anthropic ``tool_result`` block.

    Returns ``""`` for shapes this transform cannot safely rewrite (image
    blocks, structured payloads), so they are left untouched.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if len(parts) == 1:
            return parts[0]
    return ""


def _write_slot(messages: list[dict[str, Any]], slot: _Slot, text: str) -> None:
    """Write ``text`` back into the location ``slot`` came from."""
    message = messages[slot.message_index]
    if slot.block_index is None:
        message["content"] = text
        return

    block = message["content"][slot.block_index]
    if block.get("type") == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            block["content"] = text
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = text
                    break
    else:
        block["text"] = text


class ParitokCompressor(Transform):
    """Compress live-zone content with the Paritok-4B model."""

    name = "paritok"

    def __init__(
        self,
        config: ParitokConfig | None = None,
        engine: ParitokEngine | None = None,
        *,
        chain_model: bool = False,
    ) -> None:
        self.config = config or resolve_paritok_config()
        self._engine = engine
        self.chain_model = chain_model
        """Resolved once by the pipeline from its rollout snapshot.

        Deep components must not re-read the environment: a request has to
        observe the same decisions its pipeline recorded as provenance."""

    @property
    def engine(self) -> ParitokEngine:
        if self._engine is None:
            self._engine = get_engine(self.config)
        return self._engine

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Run only when the backend actually answers.

        Whether the feature is enabled at all was decided when the pipeline was
        built — this transform is only in the chain if it was. The availability
        probe is cached per process, so a proxy whose Paritok backend is down
        pays one failed connection rather than one per request.
        """
        if not messages:
            return False
        return self.engine.is_available()

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        frozen_message_count = int(kwargs.get("frozen_message_count", 0) or 0)
        protect_recent = int(kwargs.get("protect_recent", 4) or 0)
        query = kwargs.get("context", "") or ""
        min_tokens = int(
            kwargs.get("min_tokens_to_compress", self.config.min_tokens_to_compress) or 0
        )

        start = max(0, frozen_message_count)
        stop = len(result_messages) - max(0, protect_recent)
        if stop <= start:
            # Everything is either cached prefix or active conversation.
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        slots = _iter_slots(
            result_messages,
            start,
            stop,
            compress_user_messages=bool(kwargs.get("compress_user_messages", False)),
            compress_system_messages=bool(kwargs.get("compress_system_messages", True)),
        )

        eligible = [
            slot
            for slot in slots
            if count_tokens(slot.text) >= min_tokens and "<<ccr:" not in slot.text
        ]
        if not eligible:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        levels = self._assign_levels(eligible)
        requests = [
            (slot.text, query, kind, level)
            for slot, (kind, level, _reason) in zip(eligible, levels)
        ]

        try:
            outcomes = self.engine.compress_many(requests)
        except Exception as exc:  # noqa: BLE001 - compression never fails a request
            logger.warning("paritok: compression pass failed, passing through (%s)", exc)
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
                warnings=[f"paritok: {exc}"],
            )

        for slot, (kind, level, reason), outcome in zip(eligible, levels, outcomes):
            if not outcome.ok:
                if outcome.reason:
                    warnings.append(f"paritok: {slot.role}/{kind}: {outcome.reason}")
                continue

            compressed = outcome.text
            if outcome.dropped:
                # An empty body means "this segment can go". Keep a marker so the
                # content is still retrievable rather than silently deleted.
                compressed = ""

            replacement = self._finalize(
                original=slot.text,
                compressed=compressed,
                slot=slot,
                query=query,
            )
            if replacement is None:
                continue

            text, marker = replacement
            if tokenizer.count_text(text) >= tokenizer.count_text(slot.text):
                # Never make a slot bigger. The pipeline has a global inflation
                # guard, but reverting per slot keeps the wins from good slots.
                continue

            _write_slot(result_messages, slot, text)
            transforms_applied.append(f"paritok:{kind}:{level}:{reason}")
            if marker:
                markers_inserted.append(marker)

        tokens_after = tokenizer.count_messages(result_messages)
        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )

    def _assign_levels(self, slots: list[_Slot]) -> list[tuple[str, str, str]]:
        """Classify each slot's kind and compression level.

        Uses the same rule-based tagger the model was trained against, so the
        ``kind``/``level`` pair it receives matches the training distribution.
        """
        segments = [
            {"kind": classify_kind_from_content(slot.text), "content": slot.text} for slot in slots
        ]
        from headroom.paritok.tagger import detect_stale_files

        stale = detect_stale_files(segments)
        total = len(segments)
        levels: list[tuple[str, str, str]] = []
        for index, segment in enumerate(segments):
            level, reason = assign_level(segment, index, total, stale)
            levels.append((str(segment["kind"]), level, reason))
        return levels

    def _finalize(
        self,
        *,
        original: str,
        compressed: str,
        slot: _Slot,
        query: str,
    ) -> tuple[str, str] | None:
        """Store the original and return ``(replacement_text, marker)``.

        Returns ``None`` when the original could not be stored — without a
        retrievable entry the marker would be unredeemable, so the slot is left
        uncompressed instead.
        """
        original_tokens = count_tokens(original)

        if self.chain_model:
            chained = self._chain_through_kompress(original, compressed)
            if chained is not None:
                # Kompress stored the true original and emitted its own marker.
                return chained, ""
            # Chaining declined. Fall through and store the original ourselves,
            # so a failed second pass costs savings, never recoverability.

        cache_key = store_paritok_in_ccr(
            original,
            compressed,
            original_tokens,
            tool_name=slot.tool_name,
            query_context=query or None,
        )
        if cache_key is None:
            return None
        return attach_marker(compressed, cache_key), cache_key

    def _chain_through_kompress(self, original: str, compressed: str) -> str | None:
        """Run Paritok output through Kompress, storing ``original`` for retrieval.

        ``ccr_original`` exists for exactly this case: ``compressed`` is an
        intermediate, and retrieval must return the real source text.

        Returns ``None`` unless Kompress actually emitted a marker. "The text
        changed" is not proof that it stored anything: under its wall-clock
        deadline (#1171) Kompress gives up partway, keeps the remainder verbatim
        and returns altered text with no CCR entry behind it. Accepting that
        would ship lossily-shortened content whose original is unrecoverable —
        the one outcome this whole design exists to prevent.
        """
        try:
            from headroom.transforms.kompress_compressor import KompressCompressor

            kompress = KompressCompressor()
            result = kompress.compress(compressed, ccr_original=original)
        except Exception as exc:  # noqa: BLE001 - fall back to Paritok-only output
            logger.debug("paritok: chain through Kompress failed (%s)", exc)
            return None
        if not result.compressed or result.compressed == compressed:
            return None
        if not _CCR_MARKER.search(result.compressed):
            logger.debug(
                "paritok: Kompress returned no CCR marker (deadline or passthrough); "
                "storing the original from the Paritok stage instead"
            )
            return None
        return result.compressed
