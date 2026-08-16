"""End-to-end integration: Paritok inside Headroom's public compress() API.

These exercise the whole chain — rollout flag, pipeline construction, transform,
CCR store, retrieval, savings attribution — rather than any single unit. The
model is mocked at the HTTP boundary, so the code path is the real one.
"""

from __future__ import annotations

import copy
import sys

import httpx
import pytest
import respx

from headroom import compress
from headroom.paritok.engine import reset_engine

BASE = "http://localhost:11434/v1"

BIG_TOOL_OUTPUT = "def handler(request):\n    return process(request)\n" * 80


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset both singletons between tests.

    ``compress()`` caches its TransformPipeline in a module global, and the
    pipeline resolves rollout features once at construction — deliberately, so a
    request always matches its pipeline's recorded provenance. That means a
    ``PARITOK_*`` change cannot affect an already-built pipeline, so tests that
    toggle features must drop it. In production the pipeline is built at proxy
    startup, which is exactly the intended behaviour.
    """
    # Reached via sys.modules, not ``import headroom.compress``: headroom's
    # __init__ rebinds the name ``headroom.compress`` to the *function*, so the
    # import form yields a function object, and assigning ``_pipeline`` on it
    # succeeds silently while leaving the real module global untouched.
    compress_module = sys.modules["headroom.compress"]

    compress_module._pipeline = None
    reset_engine()
    yield
    compress_module._pipeline = None
    reset_engine()


def _mock_backend(reply: str = "def handler(request): ...") -> None:
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "paritok-4b-v1"}]})
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"[SEG id=s1]\n{reply}\n[/SEG]"}}]},
        ),
    )


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "fix the request handler"},
        {"role": "tool", "content": BIG_TOOL_OUTPUT},
        {"role": "tool", "content": BIG_TOOL_OUTPUT},
        {"role": "user", "content": "go"},
    ]


class TestDisabledByDefault:
    def test_compress_is_unchanged_without_the_flag(self) -> None:
        """The whole point of a default-off feature: stock behaviour is intact."""
        messages = _messages()
        result = compress(copy.deepcopy(messages), protect_recent=0)

        assert not any(t.startswith("paritok") for t in result.transforms_applied)

    def test_no_backend_call_is_made_when_disabled(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/chat/completions")
            compress(_messages(), protect_recent=0)
            assert not route.called


class TestEnabled:
    @respx.mock
    def test_paritok_runs_and_reports_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        monkeypatch.setenv("PARITOK_MIN_TOKENS", "10")
        _mock_backend()

        result = compress(_messages(), protect_recent=1)

        assert any(t.startswith("paritok:") for t in result.transforms_applied)
        assert result.tokens_saved > 0

    @respx.mock
    def test_compressed_output_stays_retrievable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Savings are only acceptable if the original can still be recovered."""
        from headroom.ccr.marker_resolution import resolve_markers_in_response

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        monkeypatch.setenv("PARITOK_MIN_TOKENS", "10")
        _mock_backend()

        result = compress(_messages(), protect_recent=1)
        restored = resolve_markers_in_response(result.messages)

        # Compare against the message content itself, not str(dict): a dict repr
        # escapes newlines, so a multi-line needle never matches.
        contents = [m.get("content") for m in restored if isinstance(m.get("content"), str)]
        assert any("def handler(request):\n    return process" in c for c in contents)

    @respx.mock
    def test_headroom_transforms_still_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paritok augments the pipeline; ContentRouter must still be in it."""
        from headroom.config import HeadroomConfig
        from headroom.transforms.pipeline import TransformPipeline

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        names = [t.name for t in TransformPipeline(HeadroomConfig()).transforms]

        assert "content_router" in names

    @respx.mock
    def test_savings_are_attributed_to_paritok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.cache.compression_store import get_compression_store

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        monkeypatch.setenv("PARITOK_MIN_TOKENS", "10")
        _mock_backend()

        import re

        result = compress(_messages(), protect_recent=1)

        # Pull the hash straight out of the emitted marker and check what the
        # store recorded against it. Counting entries would not work: the store
        # is content-addressed, so identical content from an earlier test
        # reuses the same entry instead of adding one.
        markers = re.findall(r"<<ccr:([a-f0-9]+)>>", str(result.messages))
        assert markers

        entry = get_compression_store().retrieve(markers[0])
        assert entry is not None
        assert entry.compression_strategy == "paritok_4b"

    @respx.mock
    def test_unreachable_backend_falls_back_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A proxy configured for Paritok with no model must still serve requests."""
        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        respx.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
        messages = _messages()

        result = compress(copy.deepcopy(messages), protect_recent=1)

        assert not any(t.startswith("paritok:") for t in result.transforms_applied)
        assert len(result.messages) == len(messages)
