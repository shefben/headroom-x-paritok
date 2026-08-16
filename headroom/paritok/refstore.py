"""Bridge Paritok-compressed content into Headroom's CCR store.

Paritok upstream tags compressed content with its own ``[REF:id]`` markers
backed by its own shadow storage. Headroom already has that machinery —
:class:`~headroom.cache.compression_store.CompressionStore` plus the
``<<ccr:HASH>>`` marker vocabulary — wired into ``headroom_retrieve``, CCR
inline resolution and the context tracker's proactive expansion.

So Paritok reuses it rather than adding a second format. Originals go into the
same store under ``compression_strategy="paritok_4b"``, and compressed content
carries a native ``<<ccr:HASH>>`` marker. Consequences worth knowing:

* ``headroom_retrieve`` recovers Paritok-compressed content with no new tool.
* ``headroom savings`` and ``headroom_stats`` break Paritok out as its own
  strategy while counting it in the same totals.
* ContentRouter's ``_is_already_compressed`` check sees the marker and declines
  to re-compress the block, which is the correct default (see #2694: a second
  pass would hash the *compressed* text as the new original and destroy the only
  handle on the real bytes).

This module deliberately mirrors
:func:`headroom.transforms.kompress_compressor.store_kompress_in_ccr` so both
compressors share one CCR policy.
"""

from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)

PARITOK_STRATEGY = "paritok_4b"


def store_paritok_in_ccr(
    original: str,
    compressed: str,
    original_tokens: int,
    *,
    tool_name: str | None = None,
    query_context: str | None = None,
) -> str | None:
    """Store an original→compressed mapping and return its retrieval hash.

    Returns ``None`` on any failure. A store failure must not fail the request:
    the caller then keeps the original content uncompressed rather than emitting
    a marker nothing can redeem.
    """
    try:
        from headroom.cache.compression_store import get_compression_store
        from headroom.transforms.kompress_compressor import _kompress_content_signature

        signature = _kompress_content_signature(original)
        compressed_tokens = len(compressed.split())
        store = get_compression_store()
        cache_key = store.store(
            original,
            compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_tokens,
            compressed_item_count=compressed_tokens,
            tool_name=tool_name,
            query_context=query_context or None,
            tool_signature_hash=signature.structure_hash,
            compression_strategy=PARITOK_STRATEGY,
        )
        with contextlib.suppress(Exception):
            from headroom.telemetry import get_toin

            get_toin().record_compression(
                tool_signature=signature,
                original_count=original_tokens,
                compressed_count=compressed_tokens,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                strategy=PARITOK_STRATEGY,
            )
        return cache_key
    except Exception as exc:  # noqa: BLE001 - a store failure degrades, never fails
        logger.debug("paritok: CCR store failed (%s)", exc)
        return None


def marker_for(cache_key: str) -> str:
    """Render the retrieval marker for a stored entry.

    Uses the bare ``<<ccr:HASH>>`` form, which every Headroom consumer already
    parses (``marker_resolution._MARKER_RE`` accepts trailing attributes but does
    not require them).
    """
    return f"<<ccr:{cache_key}>>"


def attach_marker(compressed: str, cache_key: str) -> str:
    """Append the retrieval marker to compressed content."""
    return f"{compressed}\n{marker_for(cache_key)}"
