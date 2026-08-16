"""Tests for age-based observation masking."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.tokenizer import Tokenizer
from headroom.transforms.observation_masking import (
    ObservationMasker,
    ObservationMaskingConfig,
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
def _no_real_store(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    written: list[str] = []

    def _fake_store(original, replacement, original_tokens, tool_name, call_id, strategy):  # noqa: ANN001, ANN202
        written.append(strategy)
        return "a1b2c3d4e5f6a1b2c3d4"

    monkeypatch.setattr(
        "headroom.transforms.observation_masking.store_original", _fake_store
    )
    return written


def _bulk(seed: int, lines: int = 60) -> str:
    return "\n".join(f"pkg/mod_{seed}/file_{index}.py: symbol_{seed}_{index}" for index in range(lines))


def _transcript(rounds: int = 12, *, tool: str = "Bash") -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "run the suite"}]}
    ]
    for index in range(rounds):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"tu_{index}",
                        "name": tool,
                        "input": {"command": f"step {index}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tu_{index}",
                        "content": _bulk(index),
                    }
                ],
            }
        )
    return messages


def _result(messages: list[dict[str, Any]], index: int) -> str:
    return messages[index]["content"][0]["content"]


def _enabled(**overrides: Any) -> ObservationMaskingConfig:
    return ObservationMaskingConfig(enabled=True, **overrides)


class TestWindow:
    def test_old_observations_are_masked(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied
        assert "output masked" in _result(result.messages, 2)
        assert "<<ccr:" in _result(result.messages, 2)

    def test_recent_observations_are_untouched(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        result = ObservationMasker(_enabled(keep_recent=6)).apply(messages, tokenizer)
        assert "output masked" not in _result(result.messages, len(messages) - 1)

    def test_short_transcript_is_a_noop(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(rounds=2)
        masker = ObservationMasker(_enabled())
        assert not masker.should_apply(messages, tokenizer)
        assert not masker.apply(messages, tokenizer).transforms_applied

    def test_frozen_prefix_is_respected(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        result = ObservationMasker(_enabled()).apply(
            messages, tokenizer, frozen_message_count=10
        )
        assert "output masked" not in _result(result.messages, 2)
        assert any("observation_masking" in label for label in result.transforms_applied)

    def test_protect_recent_can_tighten_but_not_loosen(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(rounds=14)
        wide = ObservationMasker(_enabled(keep_recent=4)).apply(
            messages, tokenizer, protect_recent=20
        )
        narrow = ObservationMasker(_enabled(keep_recent=4)).apply(
            messages, tokenizer, protect_recent=0
        )
        assert len(wide.transforms_applied) < len(narrow.transforms_applied)


class TestSelection:
    def test_small_results_are_left_alone(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        for index in range(2, len(messages), 2):
            messages[index]["content"][0]["content"] = "ok"
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert not result.transforms_applied

    def test_errors_are_kept_by_default(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        messages[2]["content"][0]["content"] = (
            "Traceback (most recent call last):\n" + _bulk(99)
        )
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert "output masked" not in _result(result.messages, 2)

    def test_errors_are_masked_when_opted_in(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        messages[2]["content"][0]["content"] = (
            "Traceback (most recent call last):\n" + _bulk(99)
        )
        result = ObservationMasker(_enabled(mask_errors=True)).apply(messages, tokenizer)
        assert "output masked" in _result(result.messages, 2)

    def test_ccr_markers_are_never_masked(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        messages[2]["content"][0]["content"] = (
            _bulk(1) + "\n<<ccr:aabbccddeeff112233>>"
        )
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert "<<ccr:aabbccddeeff112233>>" in _result(result.messages, 2)
        assert "output masked" not in _result(result.messages, 2)

    def test_store_failure_leaves_the_original(
        self, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "headroom.transforms.observation_masking.store_original",
            lambda *args: None,
        )
        messages = _transcript()
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert not result.transforms_applied
        assert "pkg/mod_0" in _result(result.messages, 2)


class TestWireShapes:
    def test_openai_tool_messages(self, tokenizer: Tokenizer) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for index in range(12):
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "function": {"name": "shell", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": _bulk(index),
                }
            )
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert result.transforms_applied
        assert "output masked" in result.messages[2]["content"]
        assert "shell" in result.transforms_applied[0]

    def test_list_content_shape_is_preserved(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        messages[2]["content"][0]["content"] = [{"type": "text", "text": _bulk(0)}]
        result = ObservationMasker(_enabled()).apply(messages, tokenizer)
        block = result.messages[2]["content"][0]["content"]
        assert isinstance(block, list)
        assert "output masked" in block[0]["text"]


class TestMonotonicity:
    def test_a_masked_message_stays_masked_as_the_transcript_grows(
        self, tokenizer: Tokenizer
    ) -> None:
        """The property that bounds cache damage to one break per message.

        A message's bytes may change once, on the turn it crosses the horizon.
        They must never change back, and they must never change again — that
        would re-cut the provider prefix on every subsequent turn.
        """
        masker = ObservationMasker(_enabled(keep_recent=6))
        base = _transcript(rounds=10)
        first = masker.apply(base, tokenizer).messages
        masked_at_2 = _result(first, 2)
        assert "output masked" in masked_at_2

        grown = _transcript(rounds=14)
        second = masker.apply(grown, tokenizer).messages
        assert _result(second, 2) == masked_at_2


class TestInputSafety:
    def test_input_messages_are_not_mutated(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        original = _result(messages, 2)
        ObservationMasker(_enabled()).apply(messages, tokenizer)
        assert _result(messages, 2) == original

    def test_disabled_config_still_returns_messages(self, tokenizer: Tokenizer) -> None:
        messages = _transcript()
        result = ObservationMasker(ObservationMaskingConfig()).apply(messages, tokenizer)
        # The transform itself does not read `enabled` — the pipeline gates on
        # the rollout feature — so this only asserts it stays well-behaved.
        assert len(result.messages) == len(messages)


class TestConfigFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "HEADROOM_OBSERVATION_MASKING",
            "HEADROOM_OBSERVATION_MASKING_KEEP_RECENT",
            "HEADROOM_OBSERVATION_MASKING_MIN_TOKENS",
            "HEADROOM_OBSERVATION_MASKING_MASK_ERRORS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ObservationMaskingConfig.from_env()
        assert not config.enabled
        assert config.keep_recent == 12
        assert config.min_tokens == 200
        assert not config.mask_errors

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_OBSERVATION_MASKING", "1")
        monkeypatch.setenv("HEADROOM_OBSERVATION_MASKING_KEEP_RECENT", "3")
        monkeypatch.setenv("HEADROOM_OBSERVATION_MASKING_MASK_ERRORS", "yes")
        config = ObservationMaskingConfig.from_env()
        assert config.enabled
        assert config.keep_recent == 3
        assert config.mask_errors

    def test_invalid_values_fall_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_OBSERVATION_MASKING_KEEP_RECENT", "not-a-number")
        assert ObservationMaskingConfig.from_env().keep_recent == 12
