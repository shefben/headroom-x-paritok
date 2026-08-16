"""Structural chunking for long Paritok-4B compression inputs.

Sending inputs longer than roughly 2-3k tokens in one shot drives the model out
of its trained distribution and produces structural hallucinations (repeated
pseudo-doc text). Splitting at class/def boundaries first, compressing per
chunk, then deduplicating recreates the chunk granularity the model saw during
training.

Ported from ``paritok/strategies/chunking.py``. Token counts use tiktoken
``cl100k_base`` — not Headroom's per-model tokenizer — because the chunk-size
thresholds below were calibrated against cl100k during training. Sizing chunks
with a different tokenizer would shift the boundaries away from the trained
distribution.
"""

from __future__ import annotations

import functools
import re

# The model runs in an 8192-token context and each call also carries a ~2.2-2.9k
# token system prompt, so a chunk + the system prompt + room to generate must all
# fit. At 3000 the prompt still fits comfortably (3000 + ~2.9k ≈ 5.9k < 8192);
# generation is separately capped against the remaining context. The
# training/benchmark value was 2000.
CHUNK_SIZE = 3000
MAX_SINGLE_BLOCK = 3000

# Match a top-level `class`/`def`, tolerating a leading Read-tool line-number
# prefix (cat -n style: "   43\t"). The optional group matches zero-width on
# clean code, so benchmark reproduction is unchanged; it only adds boundary
# detection for the line-numbered input the proxy actually receives — without
# it, numbered files find 0 boundaries and never chunk.
_TOP_LEVEL_DEF = re.compile(r"^(?:\s*\d+\t)?(class |def )\w+", re.MULTILINE)
_DEF_NAME = re.compile(r"^(class\s+\w+|def\s+\w+)")
_HEADER_OR_DEF = re.compile(r"^(class\s|def\s|# Lines \d)")

_TOKEN_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=1)
def _encoding() -> object | None:
    """Return the cl100k encoder, or None when tiktoken is unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding(_TOKEN_ENCODING)
    except Exception:  # noqa: BLE001 — chunking must degrade, not fail
        return None


def count_tokens(text: str) -> int:
    """cl100k token count, with a character-ratio fallback.

    The fallback only affects chunk *sizing*; it never changes correctness, and
    tiktoken is a core Headroom dependency, so it is a defensive path.
    """
    encoder = _encoding()
    if encoder is None:
        return max(1, len(text) // 4)
    return len(encoder.encode(text, disallowed_special=()))


def _find_structural_boundaries(text: str) -> list[int]:
    return [text[: m.start()].count("\n") for m in _TOP_LEVEL_DEF.finditer(text)]


def _token_split_block(lines: list[str], chunk_size: int) -> list[list[str]]:
    """Hard-split a run of lines so no piece exceeds *chunk_size* tokens.

    Counts the ``"\\n"`` that rejoins the lines. Paritok upstream sums only the
    per-line counts, which undershoots by roughly one token per line — on a
    272-line log chunk that is a ~9% overflow, enough to push prompt +
    generation past ``num_ctx`` and make the backend reject the request. Since a
    rejected call falls back to uncompressed content, the undercount silently
    disabled compression on exactly the boundary-less prose and log output this
    path exists to handle.
    """
    pieces: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        # +1 for the newline that will separate this line from the previous one.
        line_tokens = count_tokens(line) + (1 if current else 0)
        if current_tokens + line_tokens > chunk_size and current:
            pieces.append(current)
            current = [line]
            current_tokens = count_tokens(line)
        else:
            current.append(line)
            current_tokens += line_tokens
    if current:
        pieces.append(current)
    return pieces


def split_into_chunks_structural(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    max_single_block: int = MAX_SINGLE_BLOCK,
) -> list[tuple[str, int, int, int]]:
    """Split code at class/def boundaries.

    Returns ``(chunk_text, start_line, end_line, raw_tokens)`` tuples with
    1-based inclusive line numbers.
    """
    lines = text.split("\n")
    boundaries = _find_structural_boundaries(text)
    chunks: list[tuple[str, int, int, int]] = []

    if not boundaries:
        # No class/def boundaries at all — markdown, prose, directory listings or
        # logs. Do NOT return the whole thing as one chunk: a large boundary-less
        # input would be sent as a single oversized SEG that overflows the model
        # context. Hard-split by tokens so every chunk stays within chunk_size.
        start = 0
        for piece in _token_split_block(lines, chunk_size):
            piece_text = "\n".join(piece)
            chunks.append((piece_text, start + 1, start + len(piece), count_tokens(piece_text)))
            start += len(piece)
        return chunks

    blocks: list[tuple[int, int]] = []
    if boundaries[0] > 0:
        blocks.append((0, boundaries[0]))
    for i, boundary in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        blocks.append((boundary, end))

    current_lines: list[str] = []
    current_start = 0
    current_tokens = 0

    def flush() -> None:
        if not current_lines:
            return
        chunk_text = "\n".join(current_lines)
        chunks.append(
            (
                chunk_text,
                current_start + 1,
                current_start + len(current_lines),
                count_tokens(chunk_text),
            )
        )

    for block_start, block_end in blocks:
        block_lines = lines[block_start:block_end]
        block_tokens = count_tokens("\n".join(block_lines))

        # A single definition larger than the budget: flush what we have, then
        # hard-split the oversized block on its own.
        if block_tokens > max_single_block:
            flush()
            current_lines = []
            current_tokens = 0
            offset = block_start
            for piece in _token_split_block(block_lines, chunk_size):
                piece_text = "\n".join(piece)
                chunks.append(
                    (piece_text, offset + 1, offset + len(piece), count_tokens(piece_text))
                )
                offset += len(piece)
            current_start = block_end
            continue

        # Accumulate whole definitions until the budget is reached.
        if current_tokens + block_tokens <= chunk_size:
            if not current_lines:
                current_start = block_start
            current_lines.extend(block_lines)
            current_tokens += block_tokens
            continue

        flush()
        current_lines = list(block_lines)
        current_tokens = block_tokens
        current_start = block_start

    flush()
    return chunks


def deduplicate_definitions(text: str) -> str:
    """Drop repeated ``class Foo`` / ``def bar`` blocks produced by chunking.

    Overlapping chunks can each emit the same definition; keeping only the first
    occurrence avoids handing the agent a file that appears to define the same
    symbol twice.
    """
    seen_defs: set[str] = set()
    output_lines: list[str] = []
    skip_until_next_def = False

    for line in text.split("\n"):
        match = _DEF_NAME.match(line)
        if match:
            def_name = match.group(1)
            if def_name in seen_defs:
                skip_until_next_def = True
                continue
            seen_defs.add(def_name)
            skip_until_next_def = False
        elif skip_until_next_def:
            if _HEADER_OR_DEF.match(line):
                skip_until_next_def = False
                nested = _DEF_NAME.match(line)
                if nested:
                    def_name = nested.group(1)
                    if def_name in seen_defs:
                        skip_until_next_def = True
                        continue
                    seen_defs.add(def_name)
            else:
                continue

        output_lines.append(line)

    return "\n".join(output_lines)
