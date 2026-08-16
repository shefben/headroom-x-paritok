"""Goal hints — what the agent was actually trying to do at a point in time.

Every relevance decision Headroom makes today is task-agnostic: a tool result
is judged by whether the model referred back to it, never by whether it bears
on the thing the user asked for. SWE-Pruner (arXiv:2601.16746) is the clearest
demonstration that this leaves value on the table. It has the agent state an
explicit objective ("focus on error handling") and prunes context against that
objective rather than against a static importance metric — and reports 23-54%
token reduction on SWE-bench Verified with the solve rate going *up*, not down.
Query-aware selection beating task-agnostic selection is the consistent finding
across this whole literature.

A proxy cannot ask the agent to state its objective, but it does not need to:
the objective is already in the transcript, as the last thing the user said
before the tool ran.

The prefix-cache trap, and the fix
----------------------------------
The obvious implementation — take the *latest* user message and score every
tool result against it — is the same mistake a naive forward-looking prune
rule makes, only worse. The latest user message changes on every turn, so
every historical verdict would be re-derived against new evidence and the
cached prefix would churn from its first tool result onward.

:func:`goal_hint_at` therefore looks *backwards*: the goal in force for a
result at index ``i`` is the most recent user text at index ``<= i``. That is a
function of a fixed prefix of the transcript, so it is identical on every turn
— and it is also the more defensible reading of the signal. A result produced
while the user was asking about error handling should be judged against
"error handling", not against whatever the user asked three turns later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GoalHint", "goal_hint_at", "goal_overlap"]

# Content-block types that carry the user's own words. A user message in an
# agent loop is usually a bag of tool_result blocks; only the text blocks are
# the user talking.
_USER_TEXT_TYPES = frozenset({"text"})

# Cap on how much of a user message counts as the goal. A pasted stack trace or
# a dumped file in the user's turn would otherwise flood the identifier set and
# make every later result look relevant.
_MAX_GOAL_CHARS = 2_000


@dataclass(frozen=True)
class GoalHint:
    """The user's stated objective in force at some point in the transcript."""

    text: str = ""
    identifiers: frozenset[str] = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return bool(self.identifiers)


def _user_text(message: Any) -> str:
    """The user's own words in a message, excluding tool results."""
    if not isinstance(message, dict) or message.get("role") != "user":
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
        if block.get("type") in _USER_TEXT_TYPES:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def goal_hint_at(
    messages: list[dict[str, Any]],
    index: int,
    extract: Any,
) -> GoalHint:
    """The goal in force at ``messages[index]``.

    Scans *backwards* from ``index`` for the nearest user message carrying
    actual text. ``extract`` is the identifier extractor to use — passed in
    rather than imported so this module stays independent of whichever
    transform is asking, and so both share one identifier vocabulary.

    Returns an empty hint when no user text precedes the index, which callers
    must treat as "no opinion" rather than as "nothing is relevant".
    """
    for position in range(min(index, len(messages) - 1), -1, -1):
        text = _user_text(messages[position])
        stripped = text.strip()
        if not stripped:
            continue
        clipped = stripped[:_MAX_GOAL_CHARS]
        return GoalHint(text=clipped, identifiers=frozenset(extract(clipped)))
    return GoalHint()


def goal_overlap(hint: GoalHint, identifiers: set[str]) -> float:
    """Fraction of the goal's identifiers that appear in ``identifiers``.

    Denominated in the *goal's* terms, not the result's: a 5,000-line result
    that happens to contain one of the three words the user used is far more
    likely to be on-task than the reverse ratio would suggest. Returns 0.0 for
    an empty hint, and callers gate on the hint being truthy first.
    """
    if not hint.identifiers:
        return 0.0
    return len(hint.identifiers & identifiers) / len(hint.identifiers)


# Words that appear in nearly every instruction and so carry no signal about
# which result is on-task. Deliberately tiny — see the note in
# ``tool_result_pruning._COMMON_TOKENS``: over-filtering here pushes results
# toward "unrelated to the goal", which is the unsafe direction.
_GOAL_STOPWORDS = frozenset(
    {
        "about",
        "actually",
        "please",
        "should",
        "there",
        "these",
        "thing",
        "think",
        "those",
        "using",
        "where",
        "which",
        "would",
    }
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")


def goal_words(text: str) -> set[str]:
    """Fallback extractor for prose goals with no code-like identifiers.

    The identifier vocabulary the pruner uses is tuned for tool *output* —
    paths, symbols, hex codes. A user turn is prose, and a goal like "make the
    retry logic idempotent" contains no path at all. This picks the content
    words out of that so a prose-only goal still produces a usable hint.
    """
    return {
        match.group(0).lower()
        for match in _WORD_RE.finditer(text)
        if match.group(0).lower() not in _GOAL_STOPWORDS
    }
