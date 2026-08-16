"""Configuration for the Paritok augmentation layer.

Feature *toggles* are not defined here — they live in Headroom's canonical
rollout registry (:data:`headroom.rollout.FEATURES`) so they inherit channel
gating, ``HEADROOM_FEATURES`` / ``HEADROOM_DISABLE_FEATURES`` handling and the
provenance digest. This module holds the typed *tuning* values (endpoint,
model, selection width, compression level) plus the typed on/off requests that
:class:`~headroom.config.HeadroomConfig` forwards into ``resolve_rollout``.

That mirrors how ``intercept_tool_results`` requests the
``tool_result_interceptors`` feature — a typed field is a *request*, and the
rollout snapshot is the single source of truth for whether it is live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

# Rollout feature names owned by this package. Kept here (rather than inlined at
# call sites) so the transform, the proxy hook and the doctor check can never
# drift from the registry entries in headroom.rollout.
FEATURE_TOOL_FILTER = "paritok_tool_filter"
FEATURE_CONTENT_COMPRESS = "paritok_content_compress"
FEATURE_HISTORY_SUMMARIZE = "paritok_history_summarize"
FEATURE_CHAIN_MODEL = "paritok_chain_model"

# There is deliberately no ``paritok_ref_expand`` feature. Paritok upstream
# needs one because it injects its own ``read_original`` virtual tool; here the
# compressed originals live in Headroom's own ``CompressionStore`` behind native
# ``<<ccr:HASH>>`` markers, which ``headroom.ccr.tool_injection`` already scans
# for and answers with ``headroom_retrieve``. Expansion is therefore live and
# unconditional, and a toggle for it would only be able to lie.
PARITOK_FEATURES: tuple[str, ...] = (
    FEATURE_TOOL_FILTER,
    FEATURE_CONTENT_COMPRESS,
    FEATURE_HISTORY_SUMMARIZE,
    FEATURE_CHAIN_MODEL,
)


class ParitokBackend(str, Enum):
    """Where the Paritok-4B model runs.

    All three backends speak the same OpenAI-compatible ``/chat/completions``
    wire format; they differ only in default endpoint and auth. ``OLLAMA`` is
    the zero-build default because it needs no compiler toolchain.
    """

    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"

    @classmethod
    def parse(cls, value: str | None) -> ParitokBackend:
        if not value:
            return cls.OLLAMA
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            return cls.OLLAMA


# Default endpoint per backend. vLLM's OpenAI server and Ollama's compatibility
# shim both mount the OpenAI routes under /v1.
_DEFAULT_BASE_URLS: dict[ParitokBackend, str] = {
    ParitokBackend.OLLAMA: "http://localhost:11434/v1",
    ParitokBackend.VLLM: "http://localhost:8000/v1",
    ParitokBackend.OPENAI: "http://localhost:8000/v1",
}

# Ollama registry name published by the Paritok project.
DEFAULT_MODEL = "paritok-4b-v1"

# SEG compression levels understood by the model, with their target ratio
# ceilings. L1 is the level Paritok benchmarked, so it is the default.
VALID_LEVELS: frozenset[str] = frozenset({"L0", "L1", "L2", "L3"})
DEFAULT_LEVEL = "L1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


@dataclass
class ParitokConfig:
    """Tuning values for the Paritok augmentation layer.

    The ``*_requested`` booleans are typed feature *requests* forwarded to
    ``resolve_rollout``; read the resolved rollout snapshot (not these fields)
    to decide whether a lever is actually live.
    """

    # --- typed feature requests (equivalent to the PARITOK_* env aliases) ---
    tool_filter_requested: bool = False
    content_compress_requested: bool = False
    history_summarize_requested: bool = False
    chain_model_requested: bool = False
    """Feed Paritok-compressed text back through Headroom's Kompress model.

    Off by default: Paritok output is already near its floor, and a second
    lossy pass costs latency for little gain. When on, the compressed body is
    left unprotected so ContentRouter compresses it again."""

    # --- model runtime ---
    backend: ParitokBackend = ParitokBackend.OLLAMA
    base_url: str = ""
    """Endpoint override. Empty selects the backend's default."""
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = 120.0
    temperature: float = 0.0
    num_ctx: int = 8192
    """Model context window. Ollama rejects a request up front when
    prompt_tokens + num_predict exceeds this, so generation is capped
    against it rather than discovering the 400 at call time."""
    level: str = DEFAULT_LEVEL

    # --- tool selection ---
    tool_topk: int = 8
    """Upper bound on tools kept at full schema (k_max for dynamic selection)."""
    tool_k_min: int = 5
    tool_alpha: float = 0.9
    """Keep tools scoring >= alpha * best_score, clamped to [k_min, tool_topk]."""
    tool_recover_k: int = 2
    """Tools restored per "I don't have that tool" reply.

    Deliberately far below ``tool_k_min``: recovery re-pins permanently, so a
    generous k would undo the filter a few misses into a session. Two covers
    the common case of the agent naming one capability and the embedder's
    second guess being a near-synonym."""

    # --- content compression scope ---
    min_tokens_to_compress: int = 250
    """Skip content below this size. Matches CompressConfig.min_tokens_to_compress
    so Paritok and Headroom agree on what counts as worth compressing."""
    max_concurrency: int = 4
    """Parallel model calls per request. The model is the slow stage, and
    segments are independent."""

    # --- history summarization ---
    history_keep_recent: int = 0
    """Turns to leave untouched beyond Headroom's own ``protect_recent``.
    0 means defer entirely to ``protect_recent``."""

    _resolved_base_url: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.level not in VALID_LEVELS:
            raise ValueError(f"level must be one of {sorted(VALID_LEVELS)}, got {self.level!r}")

    @property
    def endpoint(self) -> str:
        """Effective base URL: explicit override, else the backend default."""
        return self.base_url or _DEFAULT_BASE_URLS[self.backend]

    def any_requested(self) -> bool:
        """True when at least one lever is requested via a typed field."""
        return any(
            (
                self.tool_filter_requested,
                self.content_compress_requested,
                self.history_summarize_requested,
                self.chain_model_requested,
            )
        )

    def requested_features(self) -> tuple[str, ...]:
        """Rollout feature names implied by the typed requests."""
        pairs = (
            (self.tool_filter_requested, FEATURE_TOOL_FILTER),
            (self.content_compress_requested, FEATURE_CONTENT_COMPRESS),
            (self.history_summarize_requested, FEATURE_HISTORY_SUMMARIZE),
            (self.chain_model_requested, FEATURE_CHAIN_MODEL),
        )
        return tuple(name for requested, name in pairs if requested)


def resolve_paritok_config() -> ParitokConfig:
    """Build a :class:`ParitokConfig` from ``PARITOK_*`` environment variables.

    Only tuning values are read here. The on/off state of each lever comes from
    the rollout snapshot, which reads the same ``PARITOK_*`` names as legacy
    aliases — so ``PARITOK_TOOL_FILTER=1`` both enables the feature and is
    visible in ``headroom doctor`` provenance.
    """
    backend = ParitokBackend.parse(os.environ.get("PARITOK_BACKEND"))
    level = _env_str("PARITOK_LEVEL", DEFAULT_LEVEL).upper()
    if level not in VALID_LEVELS:
        level = DEFAULT_LEVEL

    return ParitokConfig(
        backend=backend,
        base_url=_env_str("PARITOK_ENDPOINT", ""),
        model=_env_str("PARITOK_MODEL", DEFAULT_MODEL),
        api_key=_env_str("PARITOK_API_KEY", ""),
        timeout=_env_float("PARITOK_TIMEOUT", 120.0),
        temperature=_env_float("PARITOK_TEMPERATURE", 0.0),
        num_ctx=_env_int("PARITOK_NUM_CTX", 8192),
        level=level,
        tool_topk=_env_int("PARITOK_TOOL_TOPK", 8),
        tool_k_min=_env_int("PARITOK_TOOL_K_MIN", 5),
        tool_alpha=_env_float("PARITOK_TOOL_ALPHA", 0.9),
        tool_recover_k=_env_int("PARITOK_TOOL_RECOVER_K", 2),
        min_tokens_to_compress=_env_int("PARITOK_MIN_TOKENS", 250),
        max_concurrency=_env_int("PARITOK_MAX_CONCURRENCY", 4),
        history_keep_recent=_env_int("PARITOK_HISTORY_KEEP_RECENT", 0),
    )
