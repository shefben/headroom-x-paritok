"""Tests for the Paritok content-compression transform.

The behaviours pinned here are the ones that make Paritok safe to leave enabled:
it must never rewrite cached-prefix bytes, never delete content it cannot make
retrievable, and never fail a request when the model is unreachable.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from headroom.config import HeadroomConfig
from headroom.paritok.config import ParitokConfig
from headroom.paritok.engine import SegmentResult
from headroom.paritok.transform import ParitokCompressor
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import get_tokenizer

# Long enough to clear min_tokens_to_compress.
BIG = "def alpha():\n    return compute_something_expensive()\n" * 60
BIGGER = "class Beta:\n    def gamma(self):\n        return 2\n" * 60


class FakeEngine:
    """Stands in for the model: deterministic, offline, and inspectable."""

    def __init__(
        self,
        *,
        available: bool = True,
        reply: str = "COMPRESSED",
        ok: bool = True,
        dropped: bool = False,
    ) -> None:
        self._available = available
        self._reply = reply
        self._ok = ok
        self._dropped = dropped
        self.seen: list[tuple[str, str | None, str | None, str | None]] = []

    def is_available(self, *, refresh: bool = False) -> bool:
        return self._available

    def compress_segment(
        self,
        content: str,
        *,
        query: str | None = None,
        kind: str | None = None,
        level: str | None = None,
    ) -> SegmentResult:
        self.seen.append((content, query, kind, level))
        if not self._ok:
            return SegmentResult(text=content, ok=False, reason="backend down")
        if self._dropped:
            return SegmentResult(text="", ok=True, dropped=True)
        return SegmentResult(text=self._reply, ok=True)

    def compress_many(
        self,
        segments: list[tuple[str, str | None, str | None, str | None]],
    ) -> list[SegmentResult]:
        return [
            self.compress_segment(content, query=query, kind=kind, level=level)
            for content, query, kind, level in segments
        ]


@pytest.fixture
def tokenizer() -> Tokenizer:
    model = "claude-sonnet-4-5-20250929"
    return Tokenizer(get_tokenizer(model), model)


def make_transform(engine: FakeEngine, **overrides: Any) -> ParitokCompressor:
    config = ParitokConfig(min_tokens_to_compress=10, **overrides)
    return ParitokCompressor(config=config, engine=engine)


def tool_message(text: str) -> dict[str, Any]:
    return {"role": "tool", "content": text}


class TestCacheSafety:
    def test_frozen_prefix_is_never_rewritten(self, tokenizer: Tokenizer) -> None:
        """Rewriting cached bytes would invalidate the provider's prefix cache."""
        messages = [
            {"role": "system", "content": BIG},
            tool_message(BIGGER),
            tool_message(BIG),
            {"role": "user", "content": "go"},
        ]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, frozen_message_count=2, protect_recent=0)

        assert result.messages[0] == original[0]
        assert result.messages[1] == original[1]

    def test_protect_recent_tail_is_untouched(self, tokenizer: Tokenizer) -> None:
        messages = [tool_message(BIG), tool_message(BIGGER), tool_message(BIG)]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, frozen_message_count=0, protect_recent=2)

        assert result.messages[1] == original[1]
        assert result.messages[2] == original[2]

    def test_no_window_left_is_a_passthrough(self, tokenizer: Tokenizer) -> None:
        messages = [tool_message(BIG), tool_message(BIGGER)]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, frozen_message_count=1, protect_recent=1)

        assert result.messages == original
        assert result.transforms_applied == []

    def test_input_messages_are_not_mutated(self, tokenizer: Tokenizer) -> None:
        """Callers keep their list; the transform works on a deep copy."""
        messages = [tool_message(BIG), {"role": "user", "content": "go"}]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        transform.apply(messages, tokenizer, protect_recent=0)

        assert messages == original


class TestCompression:
    def test_compresses_tool_output_and_emits_marker(self, tokenizer: Tokenizer) -> None:
        messages = [tool_message(BIG)]
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.transforms_applied
        assert result.markers_inserted
        assert "<<ccr:" in result.messages[0]["content"]
        assert result.tokens_after < result.tokens_before

    def test_compressed_content_is_retrievable(self, tokenizer: Tokenizer) -> None:
        """A marker the store cannot redeem would lose the user's content."""
        from headroom.ccr.marker_resolution import resolve_markers_in_text

        messages = [tool_message(BIG)]
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)
        restored = resolve_markers_in_text(result.messages[0]["content"])

        assert BIG.strip()[:40] in restored

    def test_anthropic_tool_result_blocks_are_compressed(self, tokenizer: Tokenizer) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": BIG},
                ],
            }
        ]
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert "<<ccr:" in result.messages[0]["content"][0]["content"]

    def test_bare_user_text_is_skipped_by_default(self, tokenizer: Tokenizer) -> None:
        messages = [{"role": "user", "content": BIG}, {"role": "user", "content": "go"}]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original

    def test_assistant_messages_are_left_to_thinking_compactor(self, tokenizer: Tokenizer) -> None:
        messages = [{"role": "assistant", "content": BIG}]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original

    def test_small_content_is_below_the_floor(self, tokenizer: Tokenizer) -> None:
        messages = [tool_message("tiny")]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original
        assert result.transforms_applied == []

    def test_already_compressed_content_is_skipped(self, tokenizer: Tokenizer) -> None:
        """Re-compressing a CCR marker would orphan the original (#2694)."""
        engine = FakeEngine()
        messages = [tool_message(BIG + "\n<<ccr:abcdef0123456789abcd>>")]
        transform = make_transform(engine)

        transform.apply(messages, tokenizer, protect_recent=0)

        assert engine.seen == []

    def test_inflating_output_is_rejected_per_slot(self, tokenizer: Tokenizer) -> None:
        engine = FakeEngine(reply=BIG + BIG)
        messages = [tool_message(BIG)]
        original = copy.deepcopy(messages)
        transform = make_transform(engine)

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original
        assert result.transforms_applied == []


class TestFailureHandling:
    def test_unavailable_backend_disables_the_stage(self, tokenizer: Tokenizer) -> None:
        transform = make_transform(FakeEngine(available=False))
        assert transform.should_apply([tool_message(BIG)], tokenizer) is False

    def test_backend_failure_passes_content_through(self, tokenizer: Tokenizer) -> None:
        messages = [tool_message(BIG)]
        original = copy.deepcopy(messages)
        transform = make_transform(FakeEngine(ok=False))

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original
        assert result.transforms_applied == []
        assert result.warnings

    def test_engine_exception_is_contained(self, tokenizer: Tokenizer) -> None:
        class ExplodingEngine(FakeEngine):
            def compress_many(self, segments: list[Any]) -> list[SegmentResult]:
                raise RuntimeError("model exploded")

        messages = [tool_message(BIG)]
        original = copy.deepcopy(messages)
        transform = make_transform(ExplodingEngine())

        result = transform.apply(messages, tokenizer, protect_recent=0)

        assert result.messages == original
        assert result.warnings


class TestChaining:
    """Chaining may cost savings. It may never cost recoverability."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, kompress_output: str) -> list[dict[str, Any]]:
        class FakeResult:
            compressed = kompress_output

        class FakeKompress:
            def compress(self, content: str, ccr_original: str | None = None) -> FakeResult:
                return FakeResult()

        # Patch only the class on the real module. Swapping the whole module out
        # also hides `_kompress_content_signature`, which the CCR store path
        # imports — the fallback would then fail for the wrong reason.
        import headroom.transforms.kompress_compressor as kompress_module

        monkeypatch.setattr(kompress_module, "KompressCompressor", FakeKompress)

        model = "claude-sonnet-4-5-20250929"
        tokenizer = Tokenizer(get_tokenizer(model), model)
        compressor = ParitokCompressor(
            config=ParitokConfig(min_tokens_to_compress=10),
            engine=FakeEngine(reply="SHORT"),
            chain_model=True,
        )
        messages = [{"role": "tool", "content": BIG}, {"role": "user", "content": "go"}]
        return compressor.apply(messages, tokenizer, protect_recent=1).messages

    def test_a_marked_kompress_result_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        marked = "tiny <<ccr:abcdef0123456789abcd>>"
        result = self._run(monkeypatch, marked)

        assert result[0]["content"] == marked

    def test_an_unmarked_kompress_result_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Kompress can bail on its wall-clock deadline (#1171) and return
        altered text with nothing stored behind it. Trusting "the text changed"
        shipped lossy content whose original could not be retrieved."""
        result = self._run(monkeypatch, "kompress gave up partway, no marker here")

        assert "<<ccr:" in result[0]["content"]

    def test_the_rejected_fallback_is_still_retrievable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.ccr.marker_resolution import resolve_markers_in_response

        result = self._run(monkeypatch, "no marker at all")
        restored = resolve_markers_in_response(result)

        assert BIG in restored[0]["content"]


class TestPipelineWiring:
    def test_absent_by_default(self) -> None:
        from headroom.transforms.pipeline import TransformPipeline

        names = [t.name for t in TransformPipeline().transforms]
        assert "paritok" not in names

    def test_present_and_ordered_before_router_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.transforms.pipeline import TransformPipeline

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        names = [t.name for t in TransformPipeline(HeadroomConfig()).transforms]

        assert "paritok" in names
        assert names.index("paritok") < names.index("content_router")

    def test_chain_flag_is_resolved_at_build_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.transforms.pipeline import TransformPipeline

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        monkeypatch.setenv("PARITOK_CHAIN_MODEL", "1")
        transforms = TransformPipeline(HeadroomConfig()).transforms
        paritok = next(t for t in transforms if t.name == "paritok")

        assert paritok.chain_model is True
