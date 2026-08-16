"""Re-serialise uniform JSON tool results as TOON.

Every other compressor in Headroom removes information or defers it to the CCR
store. This one removes nothing: it re-encodes a tool result that happens to be
a uniform JSON array into Token-Oriented Object Notation, which states each
field name once in a header instead of once per row. The data is identical
afterwards; only its punctuation is gone.

Why it is worth a transform of its own: JSON spends its bytes on structure that
repeats. A 100-row array of five-field objects writes those five keys 100 times
along with 100 sets of braces and 500 pairs of quotes. TOON writes the keys
once. The published benchmark (244 retrieval questions, four models) puts the
saving at 42.6% against formatted JSON at **72.2% accuracy versus JSON's
71.4%** — i.e. the compact form was, if anything, marginally easier for the
models to read. On uniform records specifically the gap is much wider: 60.7% on
100-row tabular data.

The caveats are as well measured as the win, and each one is a guard here:

* **Nesting inverts the result.** Deeply nested configuration costs *more* in
  TOON than in compact JSON. Only flat uniform rows are encoded; anything else
  is left exactly as it arrived.
* **Output-side TOON is a mistake.** A February 2026 benchmark of *generation*
  found plain JSON beat both TOON and constrained decoding. This transform is
  input-side only and never touches what the model writes — nothing here asks
  the model to produce TOON, and nothing instructs it to.
* **Small arrays do not amortise.** The header costs a few tokens, so a
  three-row array can come out larger. The final guard is the same one the rest
  of Headroom uses: emit only if it actually shrank.

Cache safety is trivial here and worth stating because nothing else in this
directory gets it for free: the encoding is a pure function of the block's own
bytes. It looks at no other message, so its output cannot change as the
conversation grows, and it needs no lookahead bound, no frozen-prefix logic and
no protected tail.
"""

from __future__ import annotations

import json
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
    tool_use_index,
    write_candidate,
)

logger = logging.getLogger(__name__)

__all__ = ["ToonEncoder", "ToonEncodingConfig", "encode_toon", "try_encode_json_text"]

ENV_ENABLED = "HEADROOM_TOON_ENCODING"
ENV_MIN_ROWS = "HEADROOM_TOON_ENCODING_MIN_ROWS"
ENV_MIN_FIELDS = "HEADROOM_TOON_ENCODING_MIN_FIELDS"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Characters that make a bare TOON value ambiguous. A value containing any of
# them is emitted with JSON string quoting, which TOON reads back identically.
_NEEDS_QUOTING = (",", '"', "\n", "\r", ":", "[", "]", "{", "}")

_INDENT = "  "


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
class ToonEncodingConfig:
    """Tuning for :class:`ToonEncoder`."""

    enabled: bool = False

    min_rows: int = 4
    """Below this the header does not amortise. Four is the point at which the
    saved key repetitions exceed the header on every shape measured."""

    min_fields: int = 2
    """A single-field array is a list of scalars; TOON's tabular form has no
    advantage over JSON there."""

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> ToonEncodingConfig:
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            min_rows=_env_int(ENV_MIN_ROWS, 4, minimum=2),
            min_fields=_env_int(ENV_MIN_FIELDS, 2, minimum=1),
        )


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_scalar(value: Any) -> str:
    """One TOON cell. Quoting only where a bare value would be ambiguous."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if text != text.strip() or any(char in text for char in _NEEDS_QUOTING):
        return json.dumps(text, ensure_ascii=False)
    return text


def _uniform_rows(value: Any, config: ToonEncodingConfig) -> list[str] | None:
    """Field order for a uniform array of flat objects, or ``None``.

    Field order is taken from the first row rather than sorted: the producer's
    order usually carries meaning (id first, timestamps last), and preserving
    it keeps the encoding a pure re-serialisation rather than a reordering.
    """
    if not isinstance(value, list) or len(value) < config.min_rows:
        return None
    first = value[0]
    if not isinstance(first, dict) or len(first) < config.min_fields:
        return None
    fields = [key for key in first if isinstance(key, str)]
    if len(fields) != len(first):
        return None
    expected = set(fields)
    for row in value:
        if not isinstance(row, dict) or set(row.keys()) != expected:
            return None
        if not all(_is_scalar(cell) for cell in row.values()):
            return None
    return fields


def _render_table(key: str, rows: list[dict[str, Any]], fields: list[str], depth: int) -> list[str]:
    pad = _INDENT * depth
    header = f"{pad}{key}[{len(rows)}]{{{','.join(fields)}}}:"
    lines = [header]
    for row in rows:
        cells = ",".join(_render_scalar(row[field]) for field in fields)
        lines.append(f"{pad}{_INDENT}{cells}")
    return lines


def encode_toon(data: Any, config: ToonEncodingConfig) -> str | None:
    """Encode a decoded JSON value as TOON, or ``None`` if it does not qualify.

    Two shapes qualify, and nothing else does:

    * a bare uniform array of flat objects;
    * an object whose values are scalars or uniform arrays of flat objects.

    Anything nested deeper is rejected outright rather than partially encoded.
    A partial encoding would be the worst outcome available: larger than the
    JSON it replaced *and* in a format the model has to switch between
    mid-block.
    """
    fields = _uniform_rows(data, config)
    if fields is not None:
        return "\n".join(_render_table("", data, fields, 0)).lstrip()

    if not isinstance(data, dict) or not data:
        return None

    lines: list[str] = []
    saw_table = False
    for key, value in data.items():
        if not isinstance(key, str):
            return None
        if _is_scalar(value):
            lines.append(f"{key}: {_render_scalar(value)}")
            continue
        nested_fields = _uniform_rows(value, config)
        if nested_fields is None:
            return None
        lines.extend(_render_table(key, value, nested_fields, 0))
        saw_table = True

    if not saw_table:
        # An object of pure scalars is already about as small as JSON; the
        # rewrite would churn bytes for nothing.
        return None
    return "\n".join(lines)


def try_encode_json_text(text: str, config: ToonEncodingConfig) -> str | None:
    """Encode a tool result body if it is JSON that qualifies.

    Returns ``None`` for anything that is not JSON, does not qualify, or does
    not come out smaller — the caller then leaves the body untouched.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return None

    encoded = encode_toon(data, config)
    if encoded is None or len(encoded) >= len(stripped):
        return None
    return encoded


class ToonEncoder(Transform):
    """Re-serialise uniform JSON tool results into TOON."""

    name = "toon_encoding"

    def __init__(self, config: ToonEncodingConfig | None = None) -> None:
        self.config = config or ToonEncodingConfig.from_env()

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        return bool(messages)

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        result_messages = [dict(message) for message in messages]
        transforms_applied: list[str] = []

        frozen_message_count = max(0, int(kwargs.get("frozen_message_count", 0) or 0))

        # No protected tail and no lookahead: the encoding depends only on the
        # block itself, so the newest tool result is as safe to encode as the
        # oldest. Only the frozen prefix is off limits, and only because those
        # bytes are pinned by the CacheAligner.
        candidates = collect_candidates(
            result_messages,
            frozen_message_count,
            len(result_messages),
            tool_use_index(result_messages),
        )
        touched: set[int] = set()

        for candidate in candidates:
            if CCR_MARKER.search(candidate.text):
                continue
            encoded = try_encode_json_text(candidate.text, self.config)
            if encoded is None:
                continue
            if tokenizer.count_text(encoded) >= tokenizer.count_text(candidate.text):
                # Bytes are a proxy; the tokenizer is the thing that bills.
                continue

            if candidate.message_index not in touched:
                result_messages[candidate.message_index] = deep_copy_message(
                    result_messages[candidate.message_index]
                )
                touched.add(candidate.message_index)

            write_candidate(result_messages, candidate, encoded)
            transforms_applied.append(f"toon_encoding:{candidate.tool_name or 'tool'}")

        tokens_after = tokenizer.count_messages(result_messages)
        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
        )
