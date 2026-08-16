"""System prompts for the Paritok-4B SEG compression model.

These MUST match the prompts used in training verbatim — the checkpoint learned
this exact text distribution, and drifting from it degrades compression quality.
The two files under ``system_prompts/`` are byte-for-byte copies of the Paritok
training prompts; treat them as data, not documentation, and do not reformat
them.

Runtime protocol (see :mod:`headroom.paritok.engine`)::

    SYSTEM: file_read.txt  (kind == "file_read")  OR  other.txt  (all other kinds)
    USER:
        USER INTENT:
        {intent}

        Compress the following segment under the rules in your system prompt.
        Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):

        [SEG id={seg_id} kind={kind} level={level}]
        {content}
        [/SEG]

The model replies with a single ``[SEG ...]<body>[/SEG]`` block; an empty body
means "drop this segment". Levels L0-L3 set the target compression ratio
(L0 ≤ 0.50, L1 ≤ 0.35, L2 ≤ 0.25, L3 ≤ 0.20).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent / "system_prompts"

# Kinds the file_read system prompt was trained on. Everything else (log_output,
# file_operation, assistant_thinking, bash_command, tool_result,
# directory_listing, meta_action, ...) uses the "other" prompt, whose decision
# flow is broader.
_FILE_READ_KINDS = frozenset({"file_read"})


@cache
def _load(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def system_prompt_for_kind(kind: str | None) -> str:
    """Return the verbatim training system prompt for a SEG kind."""
    if kind in _FILE_READ_KINDS:
        return _load("file_read.txt")
    return _load("other.txt")
