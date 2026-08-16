"""Age-based masking of old tool observations.

The finding this implements is a negative one, and it is the useful kind.
JetBrains Research ran the comparison inside SWE-agent on SWE-bench Verified
across five model configurations ("The Complexity Trap", arXiv:2508.21433):
replacing older *observations* with a short placeholder — keeping every action
and every reasoning block intact — **halved cost while matching, and sometimes
slightly exceeding, the solve rate of LLM-based summarization**. The expensive
summarizer bought nothing over the trivial heuristic.

That is the whole design here. This transform runs no model, computes no
embedding, reads no content, and makes exactly one decision per tool result:
is it old enough. :mod:`headroom.transforms.tool_result_pruning` already covers
the clever half of the problem — which results the transcript proves were never
used — and deliberately abstains whenever it is unsure. Masking covers the
other half: results that may well have been used, at a point in the
conversation far enough back that the model is no longer working from them.

Two deviations from the paper, both because Headroom has machinery the paper's
setup did not:

* **Masked bodies stay recoverable.** The paper's placeholder is destructive.
  Here each masked body goes to the ``CompressionStore`` and the placeholder
  carries a ``<<ccr:HASH>>`` marker, so ``headroom_retrieve`` hands the full
  text back if the model turns out to want it. Strictly better, at the cost of
  one store write.
* **Actions and reasoning are never touched**, which the paper also requires,
  but here it falls out of reusing
  :func:`headroom.transforms.tool_result_common.collect_candidates` — it only
  ever finds ``tool_result`` blocks and ``role="tool"`` messages.

Prefix-cache behaviour
----------------------
The masking rule is **monotone**: a result is masked once at least
``keep_recent`` messages follow it, and once masked it stays masked for the
rest of the session. So a given message's bytes change at most once, on the
turn it crosses the threshold, and are stable before and after. That is the
same one-time transition :mod:`~headroom.transforms.tool_result_pruning`
accepts when a candidate's lookahead window fills, and it is the best available
— an age-based rule that never re-cut the prefix would be a rule that never
fires. What it is *not* is the naive alternative, where a verdict keeps
changing as new evidence arrives and the prefix is re-cut every turn.

``frozen_message_count`` from the CacheAligner is honoured, so anything the
aligner has pinned is out of scope regardless of age.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .tool_result_common import (
    CCR_MARKER,
    collect_candidates,
    deep_copy_message,
    store_original,
    tool_use_index,
    write_candidate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MASKING_STRATEGY",
    "ObservationMasker",
    "ObservationMaskingConfig",
]

MASKING_STRATEGY = "observation_masking"

ENV_ENABLED = "HEADROOM_OBSERVATION_MASKING"
ENV_KEEP_RECENT = "HEADROOM_OBSERVATION_MASKING_KEEP_RECENT"
ENV_MIN_TOKENS = "HEADROOM_OBSERVATION_MASKING_MIN_TOKENS"
ENV_MASK_ERRORS = "HEADROOM_OBSERVATION_MASKING_MASK_ERRORS"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


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


@dataclass(frozen=True)
class ObservationMaskingConfig:
    """Tuning for :class:`ObservationMasker`."""

    enabled: bool = False

    keep_recent: int = 12
    """Messages at the tail whose observations are never masked. The paper
    tunes an equivalent ``M`` and finds the result insensitive across a wide
    band; 12 is roughly three tool round-trips of a coding agent, which is the
    window a model demonstrably still works from."""

    min_tokens: int = 200
    """Below this the placeholder plus its marker costs more than the body."""

    mask_errors: bool = False
    """Errors steer the next several turns even when never quoted, so they are
    kept by default — the same posture ``tool_result_pruning`` takes."""

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> ObservationMaskingConfig:
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            keep_recent=_env_int(ENV_KEEP_RECENT, 12, minimum=1),
            min_tokens=_env_int(ENV_MIN_TOKENS, 200, minimum=1),
            mask_errors=_env_bool(ENV_MASK_ERRORS, False),
        )


def _placeholder(tool_name: str, tokens: int) -> str:
    return f"[headroom: {tool_name or 'tool'} output masked, {tokens:,} tokens — retrievable]"


class ObservationMasker(Transform):
    """Replace tool observations older than a fixed window with a placeholder."""

    name = "observation_masking"

    def __init__(self, config: ObservationMaskingConfig | None = None) -> None:
        self.config = config or ObservationMaskingConfig.from_env()

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        return bool(messages) and len(messages) > self.config.keep_recent

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        result_messages = [dict(message) for message in messages]
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []

        frozen_message_count = max(0, int(kwargs.get("frozen_message_count", 0) or 0))
        protect_recent = max(0, int(kwargs.get("protect_recent", 4) or 0))

        # The masking horizon. ``protect_recent`` is the pipeline's own floor on
        # how much tail must stay untouched; taking the stricter of the two
        # means a caller can tighten but never loosen this transform's window.
        stop = len(result_messages) - max(self.config.keep_recent, protect_recent)
        if stop <= frozen_message_count:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        candidates = collect_candidates(
            result_messages, frozen_message_count, stop, tool_use_index(result_messages)
        )
        touched: set[int] = set()

        for candidate in candidates:
            if CCR_MARKER.search(candidate.text):
                # An earlier stage (pruning, Paritok, Kompress) already owns
                # these bytes; masking a marker would orphan its original.
                continue
            if candidate.is_error and not self.config.mask_errors:
                continue

            original_tokens = tokenizer.count_text(candidate.text)
            if original_tokens < self.config.min_tokens:
                continue

            placeholder = _placeholder(candidate.tool_name, original_tokens)
            cache_key = store_original(
                candidate.text,
                placeholder,
                original_tokens,
                candidate.tool_name,
                candidate.tool_use_id,
                MASKING_STRATEGY,
            )
            if cache_key is None:
                # A store failure means the body would become unrecoverable.
                # Leaving it in place loses the saving, which is the cheap half.
                continue
            replacement = f"{placeholder}\n<<ccr:{cache_key}>>"
            if tokenizer.count_text(replacement) >= original_tokens:
                continue

            if candidate.message_index not in touched:
                result_messages[candidate.message_index] = deep_copy_message(
                    result_messages[candidate.message_index]
                )
                touched.add(candidate.message_index)

            write_candidate(result_messages, candidate, replacement)
            transforms_applied.append(
                f"observation_masking:{candidate.tool_name or 'tool'}"
            )
            markers_inserted.append(cache_key)

        tokens_after = tokenizer.count_messages(result_messages)
        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
        )
