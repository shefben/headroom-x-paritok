"""Rule-based SEG tagger — the first half of Paritok's "tagger + model" pipeline.

The Paritok-4B checkpoint was trained on segments that a *rule-based* tagger had
already labelled with a ``kind`` (file_read, log_output, file_operation, ...) and
a ``level`` (L0-L3, the target compression ratio). Inference must reproduce that
labelling so the train and inference distributions match — the model never infers
kind or level itself, it is told them in the prompt.

Ported from ``paritok/strategies/tagger.py`` (itself a port of the training
repo's ``04_label.py``). The classification rules are kept verbatim: changing
them silently degrades compression quality, because the model has only ever seen
content under the labels these rules produce.
"""

from __future__ import annotations

import re

# Kinds that are always kept lightly (L0) — never aggressively compressed.
PROTECTED_KINDS: frozenset[str] = frozenset(
    {"system_prompt", "user_turn_current", "user_turn_history"}
)
# Kinds treated as "tool results" for recency-based level assignment.
TOOL_RESULT_KINDS: frozenset[str] = frozenset({"tool_result", "log_output", "file_read"})

_PATH_RE = re.compile(r'"path":\s*"([^"]+)"')


def reclassify_tool_result(kind: str, content: str) -> str:
    """Refine a generic ``tool_result`` by its content patterns."""
    if kind != "tool_result":
        return kind
    head = content[:300]
    if "Here's the result of running `cat -n`" in head:
        return "file_read"
    if head.startswith("Here's the result of running") and "cat" in head:
        return "file_read"
    if re.match(r"^The file .+ has been (created|edited|saved)", head):
        return "file_edit_confirm"
    if "Here's the files and directories" in head:
        return "directory_listing"
    code_indicators = sum(
        1 for kw in ("import ", "def ", "class ", "from ", "function ", "package ") if kw in head
    )
    if code_indicators >= 2 and content.count("\n") > 10:
        return "file_read"
    return "tool_result"


def classify_kind_from_content(content: str) -> str:
    """Best-effort kind for a lone content string (no conversation context).

    Applies the ``reclassify_tool_result`` refinements first so the file_read /
    ``cat -n`` rules stay reachable. Defaults to ``file_read`` — the most common
    input a coding agent produces.
    """
    head = content[:300]
    if "[tool_calls]:" in head or '"str_replace_editor"' in head:
        return "file_operation"
    if "Traceback" in content or "FAILED" in content or "Error:" in head[:200]:
        return "log_output"
    kind = reclassify_tool_result("tool_result", content)
    if kind != "tool_result":
        return kind
    if "@@" in head[:200] or head.lstrip().startswith(("/", ".", "#!")):
        return "file_read"
    # Code-like content (def/class/import present) → file_read. Without a
    # conversation to place it, a Read result is the default; only fall back to
    # log_output for non-code multi-line blobs.
    if any(kw in head for kw in ("import ", "def ", "class ", "from ", "function ")):
        return "file_read"
    if content[:200].count("\n") > 5:
        return "log_output"
    return "file_read"


def detect_stale_files(segments: list[dict]) -> set[int]:
    """Indices of file accesses superseded by a later access to the same path."""
    fop_positions: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        if seg.get("kind") != "file_operation":
            continue
        for m in _PATH_RE.finditer(seg.get("content", "")):
            path = m.group(1)
            if not path.startswith("/"):
                continue
            if "." not in path.rsplit("/", 1)[-1]:
                continue
            fop_positions.setdefault(path, []).append(i)

    stale: set[int] = set()
    for positions in fop_positions.values():
        if len(positions) <= 1:
            continue
        for pos in positions[:-1]:
            stale.add(pos)
            if pos + 1 < len(segments) and segments[pos + 1].get("kind") == "tool_result":
                stale.add(pos + 1)
    return stale


def assign_level(
    seg: dict,
    seg_idx: int,
    total_segs: int,
    stale_indices: set[int],
) -> tuple[str, str]:
    """Assign L0/L1/L2/L3 from position, absolute distance and kind.

    Returns ``(level, reason)``. The reason string is recorded in transform
    metadata so a surprising compression level can be traced back to the rule
    that produced it.
    """
    kind = seg.get("kind")
    is_current = seg.get("is_current_turn", False)
    relative_pos = seg_idx / max(1, total_segs - 1)
    turns_from_end = total_segs - 1 - seg_idx

    is_very_recent = (relative_pos > 0.85) or (turns_from_end <= 1)
    is_recent = (relative_pos > 0.7) or (turns_from_end <= 4)
    is_ancient = (relative_pos < 0.25) and (turns_from_end > 8)

    # L0 — keep almost everything.
    if kind in PROTECTED_KINDS:
        return "L0", f"protected={kind}"
    if is_current:
        return "L0", "is_current_turn"
    if is_very_recent and kind in TOOL_RESULT_KINDS:
        return "L0", "very_recent_tool_result"

    # L3 — most aggressive.
    if seg_idx in stale_indices:
        return "L3", "stale_file_read"
    if is_ancient and kind not in ("file_operation", "bash_command"):
        return "L3", f"ancient(pos={relative_pos:.2f})"

    # L1 — recent file/result.
    if is_recent:
        if kind == "file_read":
            return "L1", "recent_file_read"
        if kind in TOOL_RESULT_KINDS:
            return "L1", f"recent_{kind}"

    # L1 — actions.
    if kind in ("file_operation", "bash_command", "tool_call"):
        return "L1", f"action_{kind}"

    # L2 — mid-conversation.
    if kind == "file_read":
        return "L2", "mid_file_read"
    if kind == "assistant_thinking":
        return "L2", "thinking"

    if kind in TOOL_RESULT_KINDS:
        if relative_pos < 0.4:
            return "L3", f"old_{kind}"
        return "L2", f"mid_{kind}"

    return "L2", f"default_{kind}"
