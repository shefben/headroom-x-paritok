"""Relevance-based pruning of tool results the model never used.

Measured on SWE-bench trajectories, tool results are roughly two thirds of an
agentic transcript's tokens, and 40-60% of those bytes can be removed with no
measurable loss in task performance. Headroom already reclaims two slices of
that: :mod:`headroom.transforms.read_lifecycle` retires stale and superseded
``Read`` output, and :mod:`headroom.transforms.cross_turn_dedup` folds spans
that appeared verbatim earlier. This transform covers the rest — ``Bash``,
``Grep``, MCP calls, anything that returned bulk text — using the one signal
those two cannot see: **whether the model ever referred to the result again**.

Three rules, each deterministic and each cheap (no model, no embedding):

``empty``
    The result carries no information: blank, whitespace, or a bare
    success acknowledgement. Nothing to recover, nothing to lose.

``superseded``
    The same tool ran again later with byte-identical input. The later output
    is the current truth; the earlier one is a snapshot of a state that has
    since been re-measured.

``unreferenced``
    None of the result's distinctive identifiers — file paths, symbols, error
    codes — reappear anywhere in the messages that follow it. The model read
    this output and moved on without using anything in it.

Goal conditioning
-----------------
The ``unreferenced`` rule measures what the model *already did*, which says
nothing about what it is still working on. Goal conditioning adds the missing
half: each result is also weighed against the user instruction in force when
the tool ran (see :mod:`headroom.transforms.goal_hint`). A result that clearly
bears on that instruction is kept even if nothing quoted it back, and a result
sharing nothing with it needs more evidence of use to survive. The hint is
always resolved *backwards* from the candidate, never from the newest message,
so it obeys the same fixed-prefix constraint as the lookahead window below.

Pruned bodies are not deleted. Each is written to Headroom's
:class:`~headroom.cache.compression_store.CompressionStore` and replaced by a
one-line summary plus a native ``<<ccr:HASH>>`` marker, so ``headroom_retrieve``
can hand the full text back if the model turns out to want it after all. That
also makes ContentRouter skip the block (``_is_already_compressed``), which is
correct — there is nothing left worth compressing.

Prefix-cache safety (the load-bearing constraint)
-------------------------------------------------
``superseded`` and ``unreferenced`` both look *forward*, and a naive forward
scan is exactly what breaks a provider prefix cache: appending turn N+1 could
change the verdict on a message from turn 3, mutating bytes the provider has
already cached and charging a full miss.

The fix is a **bounded lookahead**. A candidate at message index ``i`` is
judged only against ``messages[i+1 : i+1+lookahead]``, and only once that whole
window exists. The verdict therefore depends on a fixed prefix of the
transcript and can never change as the conversation grows: whatever this
transform emitted for message ``i`` on turn N, it emits byte-for-byte again on
turn N+1. The cost is that a supersede or a reference landing past the window
is missed, which loses savings but never correctness.

Failure posture: every rule prefers a false negative. Errors are kept by
default, anything already carrying a CCR marker is left alone, and a store
failure means the original stays in place rather than a marker nothing can
redeem.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .goal_hint import GoalHint, goal_hint_at, goal_overlap, goal_words
from .tool_result_common import (
    CCR_MARKER as _CCR_MARKER,
)
from .tool_result_common import (
    ToolResultCandidate as _Candidate,
)
from .tool_result_common import (
    collect_candidates as _collect_candidates,
)
from .tool_result_common import (
    deep_copy_message as _deep_copy_message,
)
from .tool_result_common import (
    message_text as _message_text,
)
from .tool_result_common import (
    store_original as _store_common,
)
from .tool_result_common import (
    tool_use_index as _tool_use_index,
)
from .tool_result_common import (
    write_candidate as _write_candidate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PRUNING_STRATEGY",
    "ToolResultPruner",
    "ToolResultPruningConfig",
    "extract_identifiers",
    "reference_ratio",
]

# Strategy tag recorded on every CompressionStore entry this module writes, so
# `headroom savings` can attribute the win without a separate ledger.
PRUNING_STRATEGY = "tool_result_pruning"

ENV_ENABLED = "HEADROOM_TOOL_RESULT_PRUNING"
ENV_MIN_TOKENS = "HEADROOM_TOOL_RESULT_PRUNING_MIN_TOKENS"
ENV_LOOKAHEAD = "HEADROOM_TOOL_RESULT_PRUNING_LOOKAHEAD"
ENV_MAX_REFERENCE_RATIO = "HEADROOM_TOOL_RESULT_PRUNING_MAX_REFERENCE_RATIO"
ENV_MIN_IDENTIFIERS = "HEADROOM_TOOL_RESULT_PRUNING_MIN_IDENTIFIERS"
ENV_PRUNE_ERRORS = "HEADROOM_TOOL_RESULT_PRUNING_PRUNE_ERRORS"
ENV_GOAL_CONDITIONING = "HEADROOM_TOOL_RESULT_PRUNING_GOAL"
ENV_GOAL_PROTECT_OVERLAP = "HEADROOM_TOOL_RESULT_PRUNING_GOAL_PROTECT_OVERLAP"
ENV_GOAL_UNRELATED_RATIO = "HEADROOM_TOOL_RESULT_PRUNING_GOAL_UNRELATED_RATIO"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Identifier vocabulary. Deliberately narrow: the decision is only as good as
# the distinctiveness of the tokens it compares, so short words and bare small
# integers are excluded rather than diluting the ratio.
_IDENTIFIER_RE = re.compile(
    r"""
    (?:[A-Za-z0-9_\-.]+ [/\\] [A-Za-z0-9_\-./\\]+)   # a path with a separator
    | (?:[A-Za-z_][A-Za-z0-9_]{4,})                  # an identifier, >= 5 chars
    | (?:0x[0-9a-fA-F]{4,})                          # a hex address / code
    | (?:\d{4,})                                     # a 4+ digit number
    """,
    re.VERBOSE,
)

# Tokens so common in code and tool chatter that their reappearance says
# nothing about whether the model used *this* result. Kept small on purpose —
# every entry here is a judgement call, and over-filtering pushes the ratio
# toward pruning, which is the unsafe direction.
_COMMON_TOKENS = frozenset(
    {
        "assert",
        "class",
        "const",
        "continue",
        "default",
        "define",
        "delete",
        "elif",
        "else",
        "error",
        "except",
        "export",
        "false",
        "final",
        "float",
        "function",
        "https",
        "impor",
        "import",
        "index",
        "instanceof",
        "lambda",
        "match",
        "module",
        "namespace",
        "none",
        "null",
        "print",
        "public",
        "raise",
        "result",
        "return",
        "self",
        "static",
        "string",
        "struct",
        "super",
        "switch",
        "throw",
        "token",
        "true",
        "value",
        "while",
        "yield",
    }
)

# Bodies matching this whole-string pattern carry no recoverable information.
_EMPTY_RESULT_RE = re.compile(
    r"^\s*(?:"
    r"|\(?no (?:output|content|results?|matches?|changes?)\.?\)?"
    r"|ok\.?|done\.?|success(?:fully)?\.?|null|none|\[\]|\{\}"
    r"|command (?:completed|ran) successfully\.?"
    r"|exit code:? 0"
    r")\s*$",
    re.IGNORECASE,
)

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


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    return value if minimum <= value <= maximum else default


@dataclass(frozen=True)
class ToolResultPruningConfig:
    """Tuning for :class:`ToolResultPruner`."""

    enabled: bool = False

    min_tokens: int = 250
    """Skip results below this size. Matches ``CompressConfig.min_tokens_to_compress``
    so every Headroom stage agrees on what counts as worth touching."""

    lookahead_messages: int = 8
    """Messages examined after a candidate. A candidate without this many
    messages behind it is left alone — see the prefix-cache note in the module
    docstring; this bound is what makes the verdict stable across turns."""

    max_reference_ratio: float = 0.02
    """Prune when fewer than this fraction of the result's distinct identifiers
    reappear in the lookahead window. A ratio rather than a count so a
    2,000-line result is not kept alive by one incidental token."""

    min_identifiers: int = 8
    """Below this many distinct identifiers the ratio is not meaningful, so the
    ``unreferenced`` rule abstains."""

    prune_errors: bool = False
    """Errors steer the model's next several turns even when it never quotes
    them, so they are kept by default."""

    prune_superseded: bool = True
    prune_empty: bool = True

    goal_conditioning: bool = True
    """Judge each result against the goal that was in force when it ran (see
    :mod:`headroom.transforms.goal_hint`). On by default because its primary
    effect is protective: a result that clearly bears on what the user asked
    for is kept even when the model never quoted it back."""

    goal_protect_overlap: float = 0.34
    """Keep a result outright once this fraction of the goal's distinctive
    words appear in it. Denominated in the goal's terms, not the result's."""

    goal_unrelated_reference_ratio: float = 0.05
    """Reference threshold applied to results sharing *nothing* with the goal.
    Higher than ``max_reference_ratio``, so an off-goal result needs more
    evidence of being used to survive. This is the only direction in which
    goal conditioning prunes more rather than less, which is why the gap is
    small and why it is a separate knob."""

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> ToolResultPruningConfig:
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            min_tokens=_env_int(ENV_MIN_TOKENS, 250, minimum=1),
            lookahead_messages=_env_int(ENV_LOOKAHEAD, 8, minimum=1),
            max_reference_ratio=_env_float(
                ENV_MAX_REFERENCE_RATIO, 0.02, minimum=0.0, maximum=1.0
            ),
            min_identifiers=_env_int(ENV_MIN_IDENTIFIERS, 8, minimum=1),
            prune_errors=_env_bool(ENV_PRUNE_ERRORS, False),
            goal_conditioning=_env_bool(ENV_GOAL_CONDITIONING, True),
            goal_protect_overlap=_env_float(
                ENV_GOAL_PROTECT_OVERLAP, 0.34, minimum=0.0, maximum=1.0
            ),
            goal_unrelated_reference_ratio=_env_float(
                ENV_GOAL_UNRELATED_RATIO, 0.05, minimum=0.0, maximum=1.0
            ),
        )


def extract_identifiers(text: str, *, limit: int = 512) -> set[str]:
    """Distinctive lowercase tokens from ``text``.

    ``limit`` caps the scan so one enormous result cannot dominate a request's
    latency; the cap is applied to *distinct* tokens in first-seen order, which
    keeps the result deterministic.
    """
    found: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group(0).lower()
        if token in _COMMON_TOKENS:
            continue
        found.add(token)
        if len(found) >= limit:
            break
    return found


def _goal_extract(text: str) -> set[str]:
    """Identifier vocabulary for a *goal*, which is prose rather than output.

    A user turn like "make the retry logic idempotent" contains no path and no
    hex code, so the output-tuned extractor alone would return almost nothing
    and every goal would read as empty. Unioning in the prose content words
    keeps the two sides comparable: both end up as lowercase tokens of five
    characters or more.
    """
    return extract_identifiers(text) | goal_words(text)


def reference_ratio(identifiers: set[str], haystack: str) -> float:
    """Fraction of ``identifiers`` that reappear in ``haystack``.

    Both sides go through the same tokenizer, so a path in the result and the
    same path in a later message compare equal regardless of the punctuation
    around them. Returns 0.0 for an empty identifier set — callers gate on
    ``min_identifiers`` before trusting the value.
    """
    if not identifiers:
        return 0.0
    later = {match.group(0).lower() for match in _IDENTIFIER_RE.finditer(haystack)}
    if not later:
        return 0.0
    return len(identifiers & later) / len(identifiers)


def _store_original(  # noqa: ANN201
    original: str,
    replacement: str,
    original_tokens: int,
    tool_name: str,
    tool_call_id: str,
):
    """Write the pruned body to the CCR store under this stage's strategy tag."""
    return _store_common(
        original,
        replacement,
        original_tokens,
        tool_name,
        tool_call_id,
        PRUNING_STRATEGY,
    )


def _summary_line(candidate: _Candidate, rule: str, tokens: int) -> str:
    tool = candidate.tool_name or "tool"
    return f"[headroom: {tool} result pruned ({rule}), {tokens:,} tokens — retrievable]"


class ToolResultPruner(Transform):
    """Drop tool results the transcript proves the model never used."""

    name = "tool_result_pruning"

    def __init__(self, config: ToolResultPruningConfig | None = None) -> None:
        self.config = config or ToolResultPruningConfig.from_env()
        self._goal_cache: dict[int, GoalHint] = {}

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        # A transcript shorter than one lookahead window has no candidate whose
        # verdict would be stable, so there is nothing to do.
        return bool(messages) and len(messages) > self.config.lookahead_messages

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
        # Goal hints are memoised per call, never across calls: the same index
        # names a different message in a different request.
        self._goal_cache = {}

        frozen_message_count = max(0, int(kwargs.get("frozen_message_count", 0) or 0))
        protect_recent = max(0, int(kwargs.get("protect_recent", 4) or 0))

        # Upper bound: the candidate must have a FULL lookahead window behind
        # it, otherwise its verdict would flip as the conversation grows.
        stop = min(
            len(result_messages) - protect_recent,
            len(result_messages) - self.config.lookahead_messages,
        )
        if stop <= frozen_message_count:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        tool_index = _tool_use_index(result_messages)
        candidates = _collect_candidates(
            result_messages, frozen_message_count, stop, tool_index
        )
        if not candidates:
            return TransformResult(
                messages=result_messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=[],
            )

        # Deep-copy only the messages we may actually rewrite. Copying the whole
        # array would defeat the point of a transform that exists to save work.
        touched: set[int] = set()

        for candidate in candidates:
            rule = self._classify(candidate, result_messages, tool_index, tokenizer)
            if rule is None:
                continue

            original_tokens = tokenizer.count_text(candidate.text)
            summary = _summary_line(candidate, rule, original_tokens)

            if rule == "empty":
                # Nothing to recover, so no store entry and no marker. Spending
                # a CCR entry to make "nothing" retrievable is pure overhead.
                cache_key = None
                replacement = summary
            else:
                cache_key = _store_original(
                    candidate.text,
                    summary,
                    original_tokens,
                    candidate.tool_name,
                    candidate.tool_use_id,
                )
                if cache_key is None:
                    continue
                replacement = f"{summary}\n<<ccr:{cache_key}>>"
            if tokenizer.count_text(replacement) >= original_tokens:
                # Never make a block bigger. Small results reach this only when
                # min_tokens is tuned down; reverting per block keeps the wins.
                continue

            if candidate.message_index not in touched:
                result_messages[candidate.message_index] = _deep_copy_message(
                    result_messages[candidate.message_index]
                )
                touched.add(candidate.message_index)

            _write_candidate(result_messages, candidate, replacement)
            transforms_applied.append(
                f"tool_result_pruning:{rule}:{candidate.tool_name or 'tool'}"
            )
            if cache_key is not None:
                markers_inserted.append(cache_key)

        tokens_after = tokenizer.count_messages(result_messages)
        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
        )

    def _classify(
        self,
        candidate: _Candidate,
        messages: list[dict[str, Any]],
        tool_index: dict[str, tuple[str, str]],
        tokenizer: Tokenizer,
    ) -> str | None:
        """Return the rule that prunes this candidate, or ``None`` to keep it."""
        text = candidate.text
        if _CCR_MARKER.search(text):
            return None  # an earlier stage already owns these bytes

        if tokenizer.count_text(text) < self.config.min_tokens:
            # Below this size no rule can pay for itself: the summary line plus
            # a marker is already longer than the body it would replace.
            return None

        if self.config.prune_empty and _EMPTY_RESULT_RE.match(text):
            return "empty"

        if candidate.is_error and not self.config.prune_errors:
            return None

        if self.config.prune_superseded and self._is_superseded(
            candidate, messages, tool_index
        ):
            return "superseded"

        identifiers = extract_identifiers(text)
        if len(identifiers) < self.config.min_identifiers:
            return None

        # Goal conditioning. The hint is resolved BACKWARDS from the candidate,
        # never from the newest message, so a verdict stays a function of a
        # fixed prefix — the same constraint the lookahead window exists for.
        threshold = self.config.max_reference_ratio
        if self.config.goal_conditioning:
            hint = self._goal_hint(messages, candidate.message_index)
            if hint:
                overlap = goal_overlap(hint, identifiers)
                if overlap >= self.config.goal_protect_overlap:
                    # On-task by the user's own words. Keep it even if the model
                    # never quoted it: the reference signal measures what the
                    # model already did, not what it is still working on.
                    return None
                if overlap <= 0.0:
                    threshold = self.config.goal_unrelated_reference_ratio

        window = messages[
            candidate.message_index + 1 : candidate.message_index
            + 1
            + self.config.lookahead_messages
        ]
        haystack = "\n".join(_message_text(message) for message in window)
        if reference_ratio(identifiers, haystack) < threshold:
            return "unreferenced"
        return None

    def _goal_hint(self, messages: list[dict[str, Any]], index: int) -> GoalHint:
        """Goal in force at ``index``, memoised for this ``apply`` call.

        Several candidates usually share one user turn, and the backward scan
        is linear, so without the cache a transcript with many tool results
        would rescan the same prefix once per result.
        """
        cached = self._goal_cache.get(index)
        if cached is None:
            cached = goal_hint_at(messages, index, _goal_extract)
            self._goal_cache[index] = cached
        return cached

    def _is_superseded(
        self,
        candidate: _Candidate,
        messages: list[dict[str, Any]],
        tool_index: dict[str, tuple[str, str]],
    ) -> bool:
        """True when the same call is repeated inside the lookahead window.

        Compares ``(tool_name, canonical_input)``. An unnamed tool or an
        unparseable input never matches, so an unknown wire shape abstains
        rather than guessing.
        """
        signature = tool_index.get(candidate.tool_use_id)
        if signature is None or not signature[0]:
            return False
        window = messages[
            candidate.message_index + 1 : candidate.message_index
            + 1
            + self.config.lookahead_messages
        ]
        for message in window:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                try:
                    payload = json.dumps(block.get("input"), sort_keys=True, default=str)
                except (TypeError, ValueError):
                    continue
                if (str(block.get("name") or ""), payload) == signature:
                    return True
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                if (
                    str(function.get("name") or ""),
                    str(function.get("arguments") or ""),
                ) == signature:
                    return True
        return False
