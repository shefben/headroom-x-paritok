"""Tests for relevance-based tool-result pruning."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.tokenizer import Tokenizer
from headroom.transforms.tool_result_pruning import (
    ToolResultPruner,
    ToolResultPruningConfig,
    extract_identifiers,
    reference_ratio,
)


class _CharTokenizer:
    """Deterministic ~4-chars-per-token counter; no model download in tests."""

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count_text(str(block))
        return total


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_CharTokenizer(), "test-model")


@pytest.fixture(autouse=True)
def _no_real_store(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture CCR writes instead of touching the real store."""
    written: list[tuple[str, str]] = []

    def _fake_store(original, replacement, original_tokens, tool_name, tool_call_id):  # noqa: ANN001, ANN202
        written.append((tool_name, original))
        return "a1b2c3d4e5f6a1b2c3d4"

    monkeypatch.setattr(
        "headroom.transforms.tool_result_pruning._store_original", _fake_store
    )
    return written


def _bulk(seed: str, lines: int = 60) -> str:
    """A large, distinctive tool output nothing else in the transcript mentions."""
    return "\n".join(f"{seed}/module_{index}.py:{index}: symbol_{seed}_{index}" for index in range(lines))


def _transcript(
    result_text: str,
    *,
    follow_up: str = "unrelated follow up text about something entirely different",
    tool_name: str = "Grep",
    tool_input: dict[str, Any] | None = None,
    trailing: int = 12,
) -> list[dict[str, Any]]:
    """A transcript with one tool call and `trailing` messages after its result."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "find the thing"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": tool_name,
                    "input": tool_input if tool_input is not None else {"pattern": "thing"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": result_text}
            ],
        },
    ]
    for index in range(trailing):
        messages.append({"role": "assistant", "content": f"{follow_up} #{index}"})
    return messages


def _result_block(messages: list[dict[str, Any]]) -> Any:
    return messages[2]["content"][0]["content"]


def _enabled(**overrides: Any) -> ToolResultPruningConfig:
    return ToolResultPruningConfig(enabled=True, **overrides)


class TestIdentifiers:
    def test_extracts_paths_symbols_and_codes(self) -> None:
        found = extract_identifiers("src/app/main.py:41 raised OSError 0xdeadbeef code 40412")
        assert "src/app/main.py" in found
        assert "oserror" in found
        assert "0xdeadbeef" in found
        assert "40412" in found

    def test_common_tokens_are_excluded(self) -> None:
        assert "return" not in extract_identifiers("return value import class")

    def test_short_tokens_are_ignored(self) -> None:
        assert extract_identifiers("a bb ccc dddd") == set()

    def test_limit_is_respected(self) -> None:
        text = " ".join(f"symbol_{index}" for index in range(100))
        assert len(extract_identifiers(text, limit=10)) == 10


class TestReferenceRatio:
    def test_full_overlap(self) -> None:
        assert reference_ratio({"alpha_one", "beta_two"}, "alpha_one and beta_two") == 1.0

    def test_no_overlap(self) -> None:
        assert reference_ratio({"alpha_one"}, "nothing relevant here") == 0.0

    def test_partial_overlap(self) -> None:
        ratio = reference_ratio({"alpha_one", "beta_two", "gamma_three", "delta_four"}, "beta_two")
        assert ratio == pytest.approx(0.25)

    def test_empty_identifiers_is_zero(self) -> None:
        assert reference_ratio(set(), "anything") == 0.0


class TestUnreferencedRule:
    def test_prunes_a_result_nothing_refers_to(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(_bulk("alpha"))
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)

        assert result.transforms_applied == ["tool_result_pruning:unreferenced:Grep"]
        assert result.tokens_after < result.tokens_before
        replaced = _result_block(result.messages)
        assert "<<ccr:a1b2c3d4e5f6a1b2c3d4>>" in replaced
        assert "result pruned (unreferenced)" in replaced

    def test_keeps_a_result_the_model_used(self, tokenizer: Tokenizer) -> None:
        body = _bulk("beta")
        # Quote a slab of the result back — the model clearly consumed it.
        messages = _transcript(body, follow_up=body)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []
        assert _result_block(result.messages) == body

    def test_input_messages_are_not_mutated(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(_bulk("gamma"))
        original = _result_block(messages)
        ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert _result_block(messages) == original

    def test_below_min_tokens_is_kept(self, tokenizer: Tokenizer) -> None:
        messages = _transcript("short_symbol_one short_symbol_two short_symbol_three")
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []

    def test_too_few_identifiers_abstains(self, tokenizer: Tokenizer) -> None:
        # Long enough to pass min_tokens, but almost no distinctive tokens.
        messages = _transcript("aaaa " * 400)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []


class TestErrorsAndEmpty:
    def test_errors_are_kept_by_default(self, tokenizer: Tokenizer) -> None:
        body = "Traceback (most recent call last):\n" + _bulk("delta")
        messages = _transcript(body)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []

    def test_errors_prunable_when_opted_in(self, tokenizer: Tokenizer) -> None:
        body = "Traceback (most recent call last):\n" + _bulk("epsilon")
        messages = _transcript(body)
        result = ToolResultPruner(_enabled(prune_errors=True)).apply(messages, tokenizer)
        assert result.transforms_applied == ["tool_result_pruning:unreferenced:Grep"]

    def test_is_error_flag_is_honoured(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(_bulk("zeta"))
        messages[2]["content"][0]["is_error"] = True
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []

    @pytest.mark.parametrize("body", ["", "   ", "(no output)", "Done.", "exit code: 0"])
    def test_small_empty_results_are_left_alone(
        self, tokenizer: Tokenizer, body: str
    ) -> None:
        # A short acknowledgement is already smaller than any replacement we
        # could write, so pruning it would inflate the request.
        messages = _transcript(body)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []
        assert _result_block(result.messages) == body

    def test_large_whitespace_padded_result_is_pruned_without_a_marker(
        self, tokenizer: Tokenizer, _no_real_store: list[tuple[str, str]]
    ) -> None:
        messages = _transcript(" " * 4000 + "(no output)" + "\n" * 4000)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == ["tool_result_pruning:empty:Grep"]
        # Nothing recoverable, so nothing was written to the CCR store.
        assert _no_real_store == []
        assert "<<ccr:" not in _result_block(result.messages)
        assert result.tokens_after < result.tokens_before


class TestSupersededRule:
    def test_identical_repeat_call_supersedes_the_earlier_result(
        self, tokenizer: Tokenizer
    ) -> None:
        body = _bulk("eta")
        # Reference the body afterwards so the `unreferenced` rule cannot fire;
        # only `superseded` can explain a prune here.
        messages = _transcript(body, follow_up=body)
        messages.insert(
            3,
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "Grep", "input": {"pattern": "thing"}}
                ],
            },
        )
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == ["tool_result_pruning:superseded:Grep"]

    def test_different_input_does_not_supersede(self, tokenizer: Tokenizer) -> None:
        body = _bulk("theta")
        messages = _transcript(body, follow_up=body)
        messages.insert(
            3,
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "Grep", "input": {"pattern": "other"}}
                ],
            },
        )
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []

    def test_supersede_can_be_disabled(self, tokenizer: Tokenizer) -> None:
        body = _bulk("iota")
        messages = _transcript(body, follow_up=body)
        messages.insert(
            3,
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "Grep", "input": {"pattern": "thing"}}
                ],
            },
        )
        result = ToolResultPruner(_enabled(prune_superseded=False)).apply(messages, tokenizer)
        assert result.transforms_applied == []


class TestCacheSafety:
    def test_verdict_is_stable_as_the_conversation_grows(self, tokenizer: Tokenizer) -> None:
        """The whole prefix-cache argument in one test.

        Whatever the transform emits for a message on turn N it must emit
        byte-for-byte on turn N+1, otherwise appending a turn rewrites cached
        bytes and costs a full cache miss.
        """
        pruner = ToolResultPruner(_enabled())
        base = _transcript(_bulk("kappa"))
        first = pruner.apply(base, tokenizer)

        grown = [dict(message) for message in base]
        # A later turn that DOES mention the result's identifiers. Without the
        # bounded lookahead this would flip the verdict back to "keep".
        grown.append({"role": "assistant", "content": _bulk("kappa")})
        second = pruner.apply(grown, tokenizer)

        assert _result_block(first.messages) == _result_block(second.messages)

    def test_candidates_without_a_full_lookahead_window_are_skipped(
        self, tokenizer: Tokenizer
    ) -> None:
        # Only 3 trailing messages, lookahead is 8: the verdict would not yet
        # be final, so the transform must abstain.
        messages = _transcript(_bulk("lambda"), trailing=3)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []

    def test_frozen_prefix_is_never_touched(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(_bulk("mu"))
        result = ToolResultPruner(_enabled()).apply(
            messages, tokenizer, frozen_message_count=5
        )
        assert result.transforms_applied == []

    def test_protect_recent_tail_is_never_touched(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(_bulk("nu"))
        result = ToolResultPruner(_enabled()).apply(
            messages, tokenizer, protect_recent=len(messages)
        )
        assert result.transforms_applied == []

    def test_already_compressed_blocks_are_left_alone(self, tokenizer: Tokenizer) -> None:
        body = _bulk("xi") + "\n<<ccr:0123456789abcdef0123>>"
        messages = _transcript(body)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []


class TestWireShapes:
    def test_openai_tool_role_message(self, tokenizer: Tokenizer) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": _bulk("omicron")},
        ]
        messages.extend({"role": "assistant", "content": f"moving on {i}"} for i in range(12))

        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == ["tool_result_pruning:unreferenced:shell"]
        assert "<<ccr:" in result.messages[2]["content"]

    def test_block_list_content_shape_is_preserved(self, tokenizer: Tokenizer) -> None:
        messages = _transcript("placeholder")
        messages[2]["content"][0]["content"] = [{"type": "text", "text": _bulk("pi")}]
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        replaced = _result_block(result.messages)
        assert isinstance(replaced, list)
        assert "<<ccr:" in replaced[0]["text"]

    def test_store_failure_leaves_content_in_place(
        self, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "headroom.transforms.tool_result_pruning._store_original",
            lambda *args, **kwargs: None,
        )
        body = _bulk("rho")
        messages = _transcript(body)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied == []
        assert _result_block(result.messages) == body


class TestConfigFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "HEADROOM_TOOL_RESULT_PRUNING",
            "HEADROOM_TOOL_RESULT_PRUNING_MIN_TOKENS",
            "HEADROOM_TOOL_RESULT_PRUNING_LOOKAHEAD",
            "HEADROOM_TOOL_RESULT_PRUNING_MAX_REFERENCE_RATIO",
            "HEADROOM_TOOL_RESULT_PRUNING_MIN_IDENTIFIERS",
            "HEADROOM_TOOL_RESULT_PRUNING_PRUNE_ERRORS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ToolResultPruningConfig.from_env()
        assert config.enabled is False
        assert config.min_tokens == 250
        assert config.lookahead_messages == 8
        assert config.max_reference_ratio == pytest.approx(0.02)
        assert config.prune_errors is False

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING", "1")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_MIN_TOKENS", "40")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_LOOKAHEAD", "3")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_MAX_REFERENCE_RATIO", "0.5")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_PRUNE_ERRORS", "yes")
        config = ToolResultPruningConfig.from_env()
        assert config.enabled is True
        assert config.min_tokens == 40
        assert config.lookahead_messages == 3
        assert config.max_reference_ratio == pytest.approx(0.5)
        assert config.prune_errors is True

    def test_out_of_range_ratio_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_MAX_REFERENCE_RATIO", "9")
        assert ToolResultPruningConfig.from_env().max_reference_ratio == pytest.approx(0.02)


class TestGoalConditioning:
    """The result is judged against the goal in force when the tool ran.

    Every assertion here also asserts a cache property: the hint is resolved
    backwards from the candidate, never from the newest message, so a verdict
    stays a function of a fixed prefix of the transcript.
    """

    GOAL = "Fix the retry_backoff handler in src/net/retry_helpers.py so it stops doubling."

    @staticmethod
    def _transcript_with_goal(
        result_text: str,
        *,
        goal: str,
        follow_up: str = "moving on to something else entirely now",
        trailing: int = 12,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": goal}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Read",
                        "input": {"path": "x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": result_text}
                ],
            },
        ]
        for index in range(trailing):
            messages.append({"role": "assistant", "content": f"{follow_up} #{index}"})
        return messages

    @staticmethod
    def _on_goal_body() -> str:
        # Carries the goal's distinctive words, and nothing later refers to it.
        return "\n".join(
            [
                "src/net/retry_helpers.py:12: def retry_backoff(attempt):",
                "src/net/retry_helpers.py:13:     doubling = attempt * 2",
            ]
            + [f"src/net/retry_helpers.py:{line}: filler_{line}" for line in range(20, 80)]
        )

    def test_on_goal_results_survive_without_being_referenced(
        self, tokenizer: Tokenizer
    ) -> None:
        messages = self._transcript_with_goal(self._on_goal_body(), goal=self.GOAL)
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert not result.transforms_applied
        assert "retry_backoff" in _result_block(result.messages)

    def test_disabling_goal_conditioning_restores_the_base_verdict(
        self, tokenizer: Tokenizer
    ) -> None:
        messages = self._transcript_with_goal(self._on_goal_body(), goal=self.GOAL)
        result = ToolResultPruner(_enabled(goal_conditioning=False)).apply(
            messages, tokenizer
        )
        assert result.transforms_applied

    def test_off_goal_results_face_a_stricter_threshold(
        self, tokenizer: Tokenizer
    ) -> None:
        # 30 lines yield 60 distinct identifiers (a path and a symbol each); 2
        # of them are referenced later, a ratio of ~0.033 — above the base 0.02
        # (kept) but below the off-goal 0.05 (pruned).
        body = "\n".join(
            f"vendor/pkg_{index}/mod.py: widget_symbol_{index} padding padding padding padding"
            for index in range(30)
        )
        follow_up = "checked widget_symbol_3 and widget_symbol_7 already"
        messages = self._transcript_with_goal(body, goal=self.GOAL, follow_up=follow_up)

        strict = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert strict.transforms_applied

        lenient = ToolResultPruner(
            _enabled(goal_unrelated_reference_ratio=0.02)
        ).apply(messages, tokenizer)
        assert not lenient.transforms_applied

    def test_a_later_user_turn_cannot_change_an_earlier_verdict(
        self, tokenizer: Tokenizer
    ) -> None:
        """The load-bearing cache property of this lever.

        If the hint came from the newest user message, appending a turn that
        happens to mention the old result would flip its verdict and re-cut the
        cached prefix at that point.
        """
        body = self._on_goal_body()
        pruner = ToolResultPruner(_enabled())

        base = self._transcript_with_goal(body, goal=self.GOAL)
        before = _result_block(pruner.apply(base, tokenizer).messages)

        grown = self._transcript_with_goal(body, goal=self.GOAL)
        grown.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "now switch to the widget_symbol vendor packages instead",
                    }
                ],
            }
        )
        grown.extend(
            {"role": "assistant", "content": f"later work {index}"} for index in range(12)
        )
        after = _result_block(pruner.apply(grown, tokenizer).messages)
        assert after == before

    def test_a_goal_with_no_distinctive_words_has_no_opinion(
        self, tokenizer: Tokenizer
    ) -> None:
        messages = self._transcript_with_goal(_bulk("vendor"), goal="do it now")
        with_goal = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        without = ToolResultPruner(_enabled(goal_conditioning=False)).apply(
            messages, tokenizer
        )
        assert bool(with_goal.transforms_applied) == bool(without.transforms_applied)

    def test_tool_results_are_not_mistaken_for_the_goal(
        self, tokenizer: Tokenizer
    ) -> None:
        """A user message full of tool_result blocks is not the user talking."""
        messages = self._transcript_with_goal(self._on_goal_body(), goal=self.GOAL)
        messages.insert(
            2,
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_0",
                        "content": "vendor/pkg_1/mod.py: widget_symbol_1",
                    }
                ],
            },
        )
        result = ToolResultPruner(_enabled()).apply(messages, tokenizer)
        assert not result.transforms_applied

    def test_env_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_GOAL", "0")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_GOAL_PROTECT_OVERLAP", "0.9")
        monkeypatch.setenv("HEADROOM_TOOL_RESULT_PRUNING_GOAL_UNRELATED_RATIO", "0.11")
        config = ToolResultPruningConfig.from_env()
        assert config.goal_conditioning is False
        assert config.goal_protect_overlap == pytest.approx(0.9)
        assert config.goal_unrelated_reference_ratio == pytest.approx(0.11)

    def test_goal_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "HEADROOM_TOOL_RESULT_PRUNING_GOAL",
            "HEADROOM_TOOL_RESULT_PRUNING_GOAL_PROTECT_OVERLAP",
            "HEADROOM_TOOL_RESULT_PRUNING_GOAL_UNRELATED_RATIO",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ToolResultPruningConfig.from_env()
        assert config.goal_conditioning is True
        assert config.goal_protect_overlap == pytest.approx(0.34)
        assert config.goal_unrelated_reference_ratio == pytest.approx(0.05)
