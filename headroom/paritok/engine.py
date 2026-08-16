"""Inference engine for the Paritok-4B compression model.

The engine runs the model *alongside* Headroom's own ONNX Kompress model rather
than replacing it: Kompress stays the default text compressor inside
ContentRouter, and Paritok-4B is an additional pass that runs earlier in the
pipeline. Both can be active in the same process.

**Transport.** Paritok-4B is served over an OpenAI-compatible
``/chat/completions`` endpoint. Ollama, vLLM and any other OpenAI-compatible
server all speak this, so a single HTTP transport covers every supported
backend; they differ only in default port and auth. That keeps the engine free
of a native inference dependency — no compiler toolchain, no GGUF loader — while
still running fully local.

**Failure policy.** Compression is an optimization, never a correctness
requirement. Every failure path (connection refused, timeout, malformed reply,
model offline) returns the *original* content and records the reason. A cold or
missing model must degrade the token savings, not break the agent's turn.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from headroom.paritok.chunking import (
    CHUNK_SIZE,
    count_tokens,
    deduplicate_definitions,
    split_into_chunks_structural,
)
from headroom.paritok.config import ParitokConfig
from headroom.paritok.prompts import system_prompt_for_kind
from headroom.paritok.tagger import classify_kind_from_content

logger = logging.getLogger(__name__)

# Ollama rejects a request up front when prompt_tokens + num_predict exceeds the
# model's context window, returning 400 "exceeds the available context size".
# When capping generation against the remaining context we leave this margin,
# because the model's tokenizer counts differently than our cl100k estimate.
_CTX_SAFETY_MARGIN = 512
# The Qwen tokenizer counts more tokens than the cl100k estimate used to size the
# prompt; inflate the estimate before reserving the rest of the window so the
# real request still fits under num_ctx.
_TOKENIZER_SLACK = 1.15
# Never request fewer than this many output tokens. A chunk always fits, so this
# floor only guards against an unusually large system prompt.
_MIN_NUM_PREDICT = 256

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Unwrap the model's [SEG ...]<body>[/SEG] reply. DOTALL so bodies span lines.
_SEG_RE = re.compile(r"\[SEG\b[^\]]*\]\s*(.*?)\s*\[/SEG\]", re.DOTALL)
# Stray SEG tags, used to scrub a half-wrapped reply (opening tag but the closing
# one truncated by the generation cap, or vice versa) so the marker never leaks.
_SEG_OPEN_RE = re.compile(r"\[SEG\b[^\]]*\]")
_SEG_CLOSE_RE = re.compile(r"\[/SEG\]")


class ParitokUnavailable(RuntimeError):
    """The configured Paritok backend could not be reached."""


@dataclass
class SegmentResult:
    """Outcome of compressing one segment.

    ``ok`` False means *nothing was compressed* and ``text`` is the untouched
    original — callers must treat that as a pass-through, not as a compression
    that happened to save zero tokens.
    """

    text: str
    ok: bool
    reason: str = ""
    dropped: bool = False
    """The model returned an empty body, meaning "this segment can be dropped"."""


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (Qwen3 may emit one before the body)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return _THINK_BLOCK_RE.sub("", text).strip()


def _unwrap_seg(raw: str) -> str:
    """Extract the compressed body from a ``[SEG ...]<body>[/SEG]`` reply.

    Returns the inner body — empty when the model dropped the segment. Falls
    back to the whole stripped text if no SEG wrapper is present, scrubbing any
    stray tag so the marker never leaks into agent-visible content.
    """
    text = _strip_thinking(raw)
    match = _SEG_RE.search(text)
    if match:
        return match.group(1).strip()
    return _SEG_CLOSE_RE.sub("", _SEG_OPEN_RE.sub("", text)).strip()


class ParitokEngine:
    """Compresses text segments with the Paritok-4B model.

    Thread-safe and cheap to hold: the HTTP client is created on first use, and
    availability is probed once and cached so a missing backend costs one failed
    connection per process rather than one per request.
    """

    def __init__(self, config: ParitokConfig) -> None:
        self.config = config
        self._client: object | None = None
        self._client_lock = threading.Lock()
        self._available: bool | None = None
        self._available_lock = threading.Lock()

    # ---- transport -------------------------------------------------------

    def _get_client(self) -> object:
        """Lazily build a pooled httpx client.

        Connection reuse matters here: a single request can issue one model call
        per segment, and a fresh TCP+TLS handshake for each would dominate the
        latency of an otherwise-local call.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - httpx is a core dep
                raise ParitokUnavailable("httpx is required for Paritok compression") from exc

            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.endpoint,
                headers=headers,
                timeout=self.config.timeout,
            )
            return self._client

    def close(self) -> None:
        """Release the pooled HTTP client, if one was created."""
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            getattr(client, "close", lambda: None)()

    # ---- availability ----------------------------------------------------

    def is_available(self, *, refresh: bool = False) -> bool:
        """Whether the configured backend answers and serves the model.

        The result is cached; pass ``refresh=True`` after starting a backend
        mid-session (``headroom doctor`` does this).
        """
        if self._available is not None and not refresh:
            return self._available
        with self._available_lock:
            if self._available is not None and not refresh:
                return self._available
            self._available = self._probe()
            return self._available

    def _probe(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False
        # /v1/models is the OpenAI-compatible discovery route; Ollama and vLLM
        # both implement it, so one probe covers every backend.
        try:
            response = httpx.get(f"{self.config.endpoint.rstrip('/')}/models", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - any failure means "not available"
            logger.debug("paritok: backend probe failed at %s: %s", self.config.endpoint, exc)
            return False
        if response.status_code != 200:
            return False
        try:
            served = {entry.get("id", "") for entry in response.json().get("data", [])}
        except Exception:  # noqa: BLE001 - a 200 with an odd body still means "up"
            return True
        if not served:
            return True
        # Ollama appends ":latest" to tags, so match on prefix.
        return any(name.startswith(self.config.model) for name in served)

    # ---- compression -----------------------------------------------------

    def compress_segment(
        self,
        content: str,
        *,
        query: str | None = None,
        kind: str | None = None,
        level: str | None = None,
    ) -> SegmentResult:
        """Compress one segment, chunking it first when it is too long.

        Inputs above ``CHUNK_SIZE`` are split at class/def boundaries and each
        chunk compressed as its own SEG, then merged and de-duplicated. One-shot
        calls on long inputs drive the model out of distribution and produce
        structural hallucinations.
        """
        if not content.strip():
            return SegmentResult(text=content, ok=False, reason="empty")

        resolved_kind = kind or classify_kind_from_content(content)
        resolved_level = level or self.config.level
        system = system_prompt_for_kind(resolved_kind)

        if count_tokens(content) <= CHUNK_SIZE:
            return self._compress_one(
                system, query, content, resolved_kind, resolved_level, seg_id="s1"
            )

        chunks = split_into_chunks_structural(content)
        parts: list[str] = []
        for index, (chunk_text, start_line, end_line, _tokens) in enumerate(chunks, start=1):
            result = self._compress_one(
                system, query, chunk_text, resolved_kind, resolved_level, seg_id=f"s{index}"
            )
            if not result.ok:
                # One failed chunk would silently delete a slice of the file, so
                # abandon the whole segment and hand back the original.
                return SegmentResult(text=content, ok=False, reason=result.reason)
            if result.text:
                parts.append(f"# Lines {start_line}-{end_line}:\n{result.text}")

        if not parts:
            return SegmentResult(text="", ok=True, dropped=True)
        return SegmentResult(text=deduplicate_definitions("\n\n".join(parts)), ok=True)

    def compress_many(
        self,
        segments: list[tuple[str, str | None, str | None, str | None]],
    ) -> list[SegmentResult]:
        """Compress ``(content, query, kind, level)`` tuples concurrently.

        Segments are independent and the model call dominates latency, so a
        small thread pool turns N sequential round-trips into roughly one.
        """
        if not segments:
            return []
        if len(segments) == 1 or self.config.max_concurrency <= 1:
            return [
                self.compress_segment(content, query=query, kind=kind, level=level)
                for content, query, kind, level in segments
            ]

        workers = min(self.config.max_concurrency, len(segments))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paritok") as pool:
            futures = [
                pool.submit(self.compress_segment, content, query=query, kind=kind, level=level)
                for content, query, kind, level in segments
            ]
            return [future.result() for future in futures]

    def _compress_one(
        self,
        system: str,
        query: str | None,
        content: str,
        kind: str,
        level: str,
        *,
        seg_id: str,
    ) -> SegmentResult:
        """Single chat completion in the training SEG message layout."""
        intent = query.strip() if query else ""
        user_message = (
            f"USER INTENT:\n{intent}\n\n"
            "Compress the following segment under the rules in your system prompt. "
            "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
            f"[SEG id={seg_id} kind={kind} level={level}]\n{content}\n[/SEG]\n"
        )

        max_tokens = self._generation_budget(system, user_message, content)

        try:
            client = self._get_client()
        except ParitokUnavailable as exc:
            return SegmentResult(text=content, ok=False, reason=str(exc))

        try:
            response = client.post(  # type: ignore[attr-defined]
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - never break the turn over compression
            reason = f"{type(exc).__name__}: {exc}"
            logger.debug("paritok: segment compression failed (%s)", reason)
            # A failed call usually means the backend went away; re-probe next
            # request rather than hammering a dead endpoint for every segment.
            self._available = None
            return SegmentResult(text=content, ok=False, reason=reason)

        body = _unwrap_seg(raw)
        if not body:
            return SegmentResult(text="", ok=True, dropped=True)
        return SegmentResult(text=body, ok=True)

    def _generation_budget(self, system: str, user_message: str, content: str) -> int:
        """Cap ``max_tokens`` so prompt + generation fits the model's context.

        The compressed body is always smaller than the input, so this cap is
        never the binding constraint on output quality — generation stops
        naturally well before it. It exists purely to keep the request legal.
        """
        budget = min(count_tokens(content) + 256, CHUNK_SIZE)
        estimated_prompt = int(
            (count_tokens(system) + count_tokens(user_message)) * _TOKENIZER_SLACK
        )
        remaining = self.config.num_ctx - estimated_prompt - _CTX_SAFETY_MARGIN
        if remaining < budget:
            budget = max(remaining, _MIN_NUM_PREDICT)
        return budget


_engine: ParitokEngine | None = None
_engine_lock = threading.Lock()


def get_engine(config: ParitokConfig | None = None) -> ParitokEngine:
    """Return the process-wide engine, creating it on first use.

    Mirrors Headroom's lazy ``_get_pipeline()`` singleton: a proxy with Paritok
    disabled never constructs an engine and never opens a connection.
    """
    global _engine

    if _engine is not None and config is None:
        return _engine

    with _engine_lock:
        if _engine is not None and config is None:
            return _engine
        if config is not None and _engine is not None and _engine.config == config:
            return _engine
        if config is not None and _engine is not None:
            _engine.close()
        from headroom.paritok.config import resolve_paritok_config

        _engine = ParitokEngine(config or resolve_paritok_config())
        return _engine


def reset_engine() -> None:
    """Drop the singleton engine. Used by tests and by runtime-env hot-sync."""
    global _engine

    with _engine_lock:
        if _engine is not None:
            _engine.close()
        _engine = None
