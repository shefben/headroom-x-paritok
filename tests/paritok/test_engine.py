"""Tests for the Paritok-4B inference engine.

The model itself is never contacted: every backend is an OpenAI-compatible HTTP
endpoint, so respx can stand in for Ollama or vLLM exactly. What matters here is
the wire contract (SEG prompt in, SEG body out), the context budget arithmetic,
and that every failure mode degrades to the original text.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from headroom.paritok.config import ParitokBackend, ParitokConfig
from headroom.paritok.engine import (
    ParitokEngine,
    _strip_thinking,
    _unwrap_seg,
    get_engine,
    reset_engine,
)

BASE = "http://localhost:11434/v1"


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
def config() -> ParitokConfig:
    return ParitokConfig(backend=ParitokBackend.OLLAMA, timeout=5.0)


def chat_response(body: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})


class TestSegUnwrapping:
    def test_extracts_body(self) -> None:
        assert _unwrap_seg("[SEG id=s1 kind=file_read level=L1]\nbody\n[/SEG]") == "body"

    def test_empty_body_means_drop(self) -> None:
        assert _unwrap_seg("[SEG id=s1]\n\n[/SEG]") == ""

    def test_strips_qwen_thinking_block(self) -> None:
        raw = "<think>reasoning here</think>[SEG id=s1]\nkept\n[/SEG]"
        assert _unwrap_seg(raw) == "kept"

    def test_truncated_closing_tag_does_not_leak_marker(self) -> None:
        """A generation cut off mid-reply must not put [SEG] into agent context."""
        assert "[SEG" not in _unwrap_seg("[SEG id=s1 kind=file_read]\npartial body")

    def test_reply_without_wrapper_passes_through(self) -> None:
        assert _unwrap_seg("plain text") == "plain text"

    def test_multiline_body_preserved(self) -> None:
        assert _unwrap_seg("[SEG id=s1]\nline1\nline2\n[/SEG]") == "line1\nline2"

    def test_strip_thinking_without_block(self) -> None:
        assert _strip_thinking("no block") == "no block"


class TestCompression:
    @respx.mock
    def test_sends_training_prompt_layout(self, config: ParitokConfig) -> None:
        route = respx.post(f"{BASE}/chat/completions").mock(
            return_value=chat_response("[SEG id=s1]\nsmall\n[/SEG]")
        )
        engine = ParitokEngine(config)

        result = engine.compress_segment("def a():\n    return 1\n", query="fix the bug")

        assert result.ok
        assert result.text == "small"
        sent = route.calls[0].request
        body = sent.read().decode()
        assert "USER INTENT:" in body
        assert "fix the bug" in body
        assert "[SEG id=s1 kind=" in body

    @respx.mock
    def test_dropped_segment_is_flagged(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(
            return_value=chat_response("[SEG id=s1]\n\n[/SEG]")
        )
        engine = ParitokEngine(config)

        result = engine.compress_segment("some stale content here")

        assert result.ok
        assert result.dropped
        assert result.text == ""

    @respx.mock
    def test_empty_input_is_not_sent(self, config: ParitokConfig) -> None:
        route = respx.post(f"{BASE}/chat/completions")
        engine = ParitokEngine(config)

        result = engine.compress_segment("   ")

        assert not result.ok
        assert not route.called

    @respx.mock
    def test_long_input_is_chunked_into_several_calls(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(
            return_value=chat_response("[SEG id=s1]\nc\n[/SEG]")
        )
        engine = ParitokEngine(config)
        body = "\n".join(f"    x{i} = {i}" for i in range(1200))
        content = f"def alpha():\n{body}\n\ndef beta():\n{body}\n"

        result = engine.compress_segment(content)

        assert result.ok
        assert respx.calls.call_count > 1
        assert "# Lines" in result.text

    @respx.mock
    def test_one_failed_chunk_abandons_the_whole_segment(self, config: ParitokConfig) -> None:
        """A partial result would silently delete a slice of the file."""
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500))
        engine = ParitokEngine(config)
        body = "\n".join(f"    x{i} = {i}" for i in range(1200))
        content = f"def alpha():\n{body}\n\ndef beta():\n{body}\n"

        result = engine.compress_segment(content)

        assert not result.ok
        assert result.text == content

    @respx.mock
    def test_compress_many_returns_one_result_per_segment(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(
            return_value=chat_response("[SEG id=s1]\nc\n[/SEG]")
        )
        engine = ParitokEngine(config)

        results = engine.compress_many(
            [("aaa bbb ccc", None, None, None), ("ddd eee fff", None, None, None)]
        )

        assert len(results) == 2
        assert all(r.ok for r in results)


class TestFailOpen:
    @respx.mock
    def test_connection_error_returns_original(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
        engine = ParitokEngine(config)

        result = engine.compress_segment("original content")

        assert not result.ok
        assert result.text == "original content"
        assert "ConnectError" in result.reason

    @respx.mock
    def test_timeout_returns_original(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
        engine = ParitokEngine(config)

        result = engine.compress_segment("original content")

        assert not result.ok
        assert result.text == "original content"

    @respx.mock
    def test_malformed_reply_returns_original(self, config: ParitokConfig) -> None:
        respx.post(f"{BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json={"unexpected": True})
        )
        engine = ParitokEngine(config)

        result = engine.compress_segment("original content")

        assert not result.ok
        assert result.text == "original content"

    @respx.mock
    def test_failure_invalidates_the_availability_cache(self, config: ParitokConfig) -> None:
        """A backend that went away must be re-probed, not assumed up forever."""
        respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
        engine = ParitokEngine(config)
        engine._available = True

        engine.compress_segment("original content")

        assert engine._available is None


class TestAvailability:
    @respx.mock
    def test_available_when_model_is_served(self, config: ParitokConfig) -> None:
        respx.get(f"{BASE}/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "paritok-4b-v1:latest"}]})
        )
        assert ParitokEngine(config).is_available() is True

    @respx.mock
    def test_unavailable_when_model_missing(self, config: ParitokConfig) -> None:
        respx.get(f"{BASE}/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
        )
        assert ParitokEngine(config).is_available() is False

    @respx.mock
    def test_unavailable_when_backend_refuses(self, config: ParitokConfig) -> None:
        respx.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
        assert ParitokEngine(config).is_available() is False

    @respx.mock
    def test_probe_result_is_cached(self, config: ParitokConfig) -> None:
        route = respx.get(f"{BASE}/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "paritok-4b-v1"}]})
        )
        engine = ParitokEngine(config)

        engine.is_available()
        engine.is_available()

        assert route.call_count == 1


class TestGenerationBudget:
    def test_budget_stays_inside_the_context_window(self, config: ParitokConfig) -> None:
        engine = ParitokEngine(config)
        system = "word " * 3000
        user = "word " * 1000

        budget = engine._generation_budget(system, user, "content")

        assert budget >= 256
        assert budget <= config.num_ctx

    def test_small_input_gets_a_small_budget(self, config: ParitokConfig) -> None:
        engine = ParitokEngine(config)
        assert engine._generation_budget("sys", "usr", "tiny") < 1000


class TestSingleton:
    def test_get_engine_reuses_one_instance(self) -> None:
        assert get_engine() is get_engine()

    def test_reset_engine_clears_it(self) -> None:
        first = get_engine()
        reset_engine()
        assert get_engine() is not first


class TestBackendEndpoints:
    def test_ollama_default(self) -> None:
        assert ParitokConfig(backend=ParitokBackend.OLLAMA).endpoint == BASE

    def test_vllm_default(self) -> None:
        assert ParitokConfig(backend=ParitokBackend.VLLM).endpoint == "http://localhost:8000/v1"

    def test_explicit_override_wins(self) -> None:
        config = ParitokConfig(backend=ParitokBackend.OLLAMA, base_url="http://gpu:9000/v1")
        assert config.endpoint == "http://gpu:9000/v1"

    def test_unknown_backend_falls_back_to_ollama(self) -> None:
        assert ParitokBackend.parse("nonsense") is ParitokBackend.OLLAMA

    @respx.mock
    def test_api_key_is_sent_as_bearer(self) -> None:
        config = ParitokConfig(backend=ParitokBackend.VLLM, api_key="secret")
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=chat_response("[SEG id=s1]\nc\n[/SEG]")
        )
        ParitokEngine(config).compress_segment("aaa bbb ccc")

        assert route.calls[0].request.headers["authorization"] == "Bearer secret"
