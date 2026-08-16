"""Tests for Paritok history summarization.

The invariant that matters most here is structural: Headroom's pipeline no
longer mutates the message *list*, so summarization must shrink turns in place
and leave the count, order and roles untouched.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from headroom.config import HeadroomConfig
from headroom.paritok.config import ParitokConfig
from headroom.paritok.history import ParitokHistorySummarizer
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import get_tokenizer

from .test_transform import FakeEngine

STALE = "Earlier I looked at the parser and found several issues worth noting. " * 30


@pytest.fixture
def tokenizer() -> Tokenizer:
    model = "claude-sonnet-4-5-20250929"
    return Tokenizer(get_tokenizer(model), model)


def make_summarizer(engine: FakeEngine, **overrides: Any) -> ParitokHistorySummarizer:
    config = ParitokConfig(min_tokens_to_compress=10, **overrides)
    return ParitokHistorySummarizer(config=config, engine=engine)


class TestStructuralInvariant:
    def test_message_count_is_unchanged(self, tokenizer: Tokenizer) -> None:
        """PR-B1 retired list mutation; summarization must not reintroduce it."""
        messages = [
            {"role": "user", "content": STALE},
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "now do the thing"},
        ]
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert len(result.messages) == len(messages)

    def test_roles_and_order_are_preserved(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "user", "content": STALE},
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]

    def test_input_is_not_mutated(self, tokenizer: Tokenizer) -> None:
        messages = [{"role": "user", "content": STALE}, {"role": "user", "content": "go"}]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        summarizer.apply(messages, tokenizer, protect_recent=1)

        assert messages == original


class TestScope:
    def test_stale_turns_are_summarized(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert "<<ccr:" in result.messages[0]["content"]
        assert result.transforms_applied

    def test_recent_turns_are_protected(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "assistant", "content": STALE},
            {"role": "assistant", "content": STALE},
        ]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, protect_recent=2)

        assert result.messages == original

    def test_frozen_prefix_is_protected(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "user", "content": STALE},
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, frozen_message_count=1, protect_recent=1)

        assert result.messages[0] == original[0]

    def test_extra_keep_recent_widens_the_protected_tail(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "assistant", "content": STALE},
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(reply="summary"), history_keep_recent=2)

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert result.messages == original

    def test_tool_output_is_left_to_the_content_compressor(self, tokenizer: Tokenizer) -> None:
        messages = [{"role": "tool", "content": STALE}, {"role": "user", "content": "go"}]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(reply="summary"))

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert result.messages == original

    def test_already_marked_content_is_skipped(self, tokenizer: Tokenizer) -> None:
        engine = FakeEngine(reply="summary")
        messages = [
            {"role": "assistant", "content": STALE + "\n<<ccr:abcdef0123456789abcd>>"},
            {"role": "user", "content": "go"},
        ]
        summarizer = make_summarizer(engine)

        summarizer.apply(messages, tokenizer, protect_recent=1)

        assert engine.seen == []

    def test_uses_the_most_aggressive_level(self, tokenizer: Tokenizer) -> None:
        engine = FakeEngine(reply="summary")
        messages = [
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        make_summarizer(engine).apply(messages, tokenizer, protect_recent=1)

        assert engine.seen[0][3] == "L3"


class TestFailureHandling:
    def test_backend_failure_leaves_history_intact(self, tokenizer: Tokenizer) -> None:
        messages = [
            {"role": "assistant", "content": STALE},
            {"role": "user", "content": "go"},
        ]
        original = copy.deepcopy(messages)
        summarizer = make_summarizer(FakeEngine(ok=False))

        result = summarizer.apply(messages, tokenizer, protect_recent=1)

        assert result.messages == original
        assert result.warnings

    def test_unavailable_backend_disables_the_stage(self, tokenizer: Tokenizer) -> None:
        summarizer = make_summarizer(FakeEngine(available=False))
        messages = [{"role": "assistant", "content": STALE}]
        assert summarizer.should_apply(messages, tokenizer) is False


class TestPipelineWiring:
    def test_absent_by_default(self) -> None:
        from headroom.transforms.pipeline import TransformPipeline

        names = [t.name for t in TransformPipeline().transforms]
        assert "paritok_history" not in names

    def test_enabled_independently_of_content_compression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each lever is its own flag; enabling one must not enable the other."""
        from headroom.transforms.pipeline import TransformPipeline

        monkeypatch.setenv("PARITOK_HISTORY_SUMMARIZE", "1")
        names = [t.name for t in TransformPipeline(HeadroomConfig()).transforms]

        assert "paritok_history" in names
        assert "paritok" not in names

    def test_both_levers_can_run_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from headroom.transforms.pipeline import TransformPipeline

        monkeypatch.setenv("PARITOK_CONTENT_COMPRESS", "1")
        monkeypatch.setenv("PARITOK_HISTORY_SUMMARIZE", "1")
        names = [t.name for t in TransformPipeline(HeadroomConfig()).transforms]

        assert names.index("paritok") < names.index("paritok_history")
        assert names.index("paritok_history") < names.index("content_router")
