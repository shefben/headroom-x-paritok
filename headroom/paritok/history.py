"""Paritok history summarization for stale conversation turns.

Paritok upstream keeps long sessions in-window by collapsing old turns into a
single summary message. Headroom cannot do that: PR-B1 retired the
IntelligentContextManager / RollingWindow stage and made live-zone-only
compression the sole strategy, so the pipeline no longer mutates the message
*list* — dropping or merging messages would break the provider's cache
accounting and every index-based tracker downstream.

This transform therefore summarizes **in place**: each stale turn keeps its
position, role and structure, and only its text is replaced by a summary plus a
``<<ccr:HASH>>`` marker. The message count never changes. That achieves the same
token reduction while preserving the invariant.

It complements :class:`~headroom.paritok.transform.ParitokCompressor` rather
than overlapping it:

* the content compressor targets tool output at tagger-assigned levels and
  leaves assistant prose to Headroom's thinking_compactor;
* this stage targets *stale* turns — including assistant prose — at the most
  aggressive level, because a turn far enough back is context, not content.

The two are independently toggleable and safe to run together: whichever runs
first leaves a CCR marker, and the other skips the block on sight.
"""

from __future__ import annotations

import logging
from typing import Any

from headroom.config import TransformResult
from headroom.paritok.chunking import count_tokens
from headroom.paritok.config import ParitokConfig, resolve_paritok_config
from headroom.paritok.engine import ParitokEngine, get_engine
from headroom.paritok.refstore import attach_marker, store_paritok_in_ccr
from headroom.tokenizer import Tokenizer
from headroom.transforms.base import Transform
from headroom.utils import deep_copy_messages

logger = logging.getLogger(__name__)

# Stale turns are summarized at the most aggressive SEG level: the model was
# trained to treat L3 as "ancient context, keep only what a later turn might
# reference".
STALE_LEVEL = "L3"

# Roles whose prose is worth summarizing. Tool output is the content
# compressor's job; system prompts are never touched here because they are
# almost always inside the cached prefix.
_SUMMARIZABLE_ROLES = frozenset({"assistant", "user"})


class ParitokHistorySummarizer(Transform):
    """Summarize stale turns in place with the Paritok-4B model."""

    name = "paritok_history"

    def __init__(
        self,
        config: ParitokConfig | None = None,
        engine: ParitokEngine | None = None,
    ) -> None:
        self.config = config or resolve_paritok_config()
        self._engine = engine

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
        # Stale means older than both Headroom's active window and any extra
        # buffer the operator asked for.
        stop = (
            len(result_messages) - max(0, protect_recent) - max(0, self.config.history_keep_recent)
        )
        if stop <= start:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        targets: list[tuple[int, str]] = []
        for index in range(start, stop):
            message = result_messages[index]
            if message.get("role") not in _SUMMARIZABLE_ROLES:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                # Block-list content in a stale turn is usually tool output,
                # which the content compressor owns.
                continue
            if "<<ccr:" in content:
                continue
            if count_tokens(content) < min_tokens:
                continue
            targets.append((index, content))

        if not targets:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        try:
            outcomes = self.engine.compress_many(
                [(text, query, "user_turn_history", STALE_LEVEL) for _index, text in targets]
            )
        except Exception as exc:  # noqa: BLE001 - summarization never fails a request
            logger.warning("paritok: history summarization failed, passing through (%s)", exc)
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
                warnings=[f"paritok_history: {exc}"],
            )

        for (index, original), outcome in zip(targets, outcomes):
            if not outcome.ok:
                if outcome.reason:
                    warnings.append(f"paritok_history: {outcome.reason}")
                continue

            cache_key = store_paritok_in_ccr(
                original,
                outcome.text,
                count_tokens(original),
                query_context=query or None,
            )
            if cache_key is None:
                # Without a retrievable entry the summary would be lossy with no
                # way back, so leave the turn intact.
                continue

            replacement = attach_marker(outcome.text, cache_key)
            if tokenizer.count_text(replacement) >= tokenizer.count_text(original):
                continue

            result_messages[index]["content"] = replacement
            transforms_applied.append(
                f"paritok_history:{result_messages[index].get('role')}:{STALE_LEVEL}"
            )
            markers_inserted.append(cache_key)

        tokens_after = tokenizer.count_messages(result_messages)
        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )
