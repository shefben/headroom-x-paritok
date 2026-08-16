"""Paritok augmentation layer for Headroom.

Paritok (https://github.com/Paritok-official/paritok-4b-v1) is a compression
gateway for coding agents built around three levers Headroom did not have as
first-class features:

  1. **Semantic tool-schema selection** — embed the current query and every
     exposed tool schema, keep the top-k relevant ones in full and stub/drop the
     rest. Complements Headroom's :mod:`headroom.proxy.tool_schema_compaction`,
     which shrinks *every* schema lexically but never decides which ones matter.
  2. **Paritok-4B content compression** — a Qwen3-4B SFT checkpoint trained on
     coding-agent trajectories, driven over an OpenAI-compatible endpoint.
  3. **History summarization** — collapse stale turns to keep long sessions
     inside the context window.

Every lever is off by default and gated through Headroom's canonical rollout
registry (:mod:`headroom.rollout`), so each can be enabled independently::

    PARITOK_TOOL_FILTER=1 PARITOK_CONTENT_COMPRESS=1 headroom proxy

or equivalently ``HEADROOM_FEATURES=paritok_tool_filter,paritok_content_compress``.

**Paritok augments Headroom; it never replaces it.** The compression transform
runs after CacheAligner (so the cached prefix stays byte-stable) and before
ContentRouter, which then applies SmartCrusher / CodeCompressor / Kompress to
whatever remains. Originals are written to Headroom's own
:class:`~headroom.cache.compression_store.CompressionStore` and referenced with
native ``<<ccr:...>>`` markers, so ``headroom_retrieve``, CCR inline resolution
and proactive expansion all work on Paritok-compressed content unchanged.
"""

from __future__ import annotations

from headroom.paritok.config import (
    ParitokBackend,
    ParitokConfig,
    resolve_paritok_config,
)

__all__ = [
    "ParitokBackend",
    "ParitokConfig",
    "resolve_paritok_config",
]
