"""Tests for tail reinjection of the user's current task."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.task_reminder import (
    TASK_REMINDER_SENTINEL,
    TaskReminderSettings,
    apply_responses_task_reminder,
    apply_task_reminder,
    extract_current_task,
    reset_task_reminder,
)

TASK = "Refactor the retry helper so every backoff path is idempotent."


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_task_reminder()


def _on(**overrides: Any) -> TaskReminderSettings:
    return TaskReminderSettings(enabled=True, **overrides)


def _messages(distance: int = 20, task: str = TASK) -> list[dict[str, Any]]:
    """A transcript whose last real user instruction is `distance` back."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": task}]}
    ]
    for index in range(distance - 1):
        messages.append({"role": "assistant", "content": f"step {index}"})
        if len(messages) >= distance:
            break
    while len(messages) < distance:
        messages.append({"role": "assistant", "content": "filler"})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "output"}
            ],
        }
    )
    return messages


def _tail_text(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


class TestExtraction:
    def test_finds_the_latest_real_user_turn(self) -> None:
        messages = _messages()
        extracted = extract_current_task(messages, _on())
        assert extracted is not None
        task, distance = extracted
        assert task == TASK
        assert distance == len(messages) - 1

    def test_tool_results_are_not_the_user_talking(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x" * 200}
                ],
            }
        ]
        assert extract_current_task(messages, _on()) is None

    def test_harness_injected_turns_are_skipped(self) -> None:
        messages = _messages()
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>context blah blah blah</system-reminder>",
                    }
                ],
            }
        )
        extracted = extract_current_task(messages, _on())
        assert extracted is not None
        assert extracted[0] == TASK

    def test_short_acknowledgements_are_skipped(self) -> None:
        messages = _messages()
        messages.append({"role": "user", "content": "ok go on"})
        extracted = extract_current_task(messages, _on())
        assert extracted is not None
        assert extracted[0] == TASK

    def test_long_tasks_are_clipped_at_a_line_boundary(self) -> None:
        long_task = "\n".join(f"line {index} of the instruction" for index in range(40))
        messages = [{"role": "user", "content": long_task}]
        extracted = extract_current_task(messages, _on(max_chars=120))
        assert extracted is not None
        assert extracted[0].endswith("[…]")
        assert len(extracted[0]) <= 130
        assert "\n" in extracted[0]

    def test_junk_inputs(self) -> None:
        assert extract_current_task(None, _on()) is None
        assert extract_current_task([], _on()) is None
        assert extract_current_task([None, 3], _on()) is None


class TestGates:
    def test_disabled_is_inert(self) -> None:
        body = {"messages": _messages()}
        assert apply_task_reminder(body, TaskReminderSettings(), input_tokens=99_999) is None

    def test_short_transcripts_are_skipped(self) -> None:
        body = {"messages": _messages()}
        assert apply_task_reminder(body, _on(), input_tokens=100) is None
        assert TASK_REMINDER_SENTINEL not in _tail_text(body)

    def test_a_task_still_at_the_tail_is_not_restated(self) -> None:
        body = {"messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}]}
        assert apply_task_reminder(body, _on(), input_tokens=99_999) is None

    def test_distance_threshold(self) -> None:
        body = {"messages": _messages(distance=6)}
        assert apply_task_reminder(body, _on(min_distance=12), input_tokens=99_999) is None
        body = {"messages": _messages(distance=20)}
        assert apply_task_reminder(body, _on(min_distance=12), input_tokens=99_999) == TASK

    def test_assistant_tail_is_left_alone(self) -> None:
        messages = _messages()
        messages.append({"role": "assistant", "content": "partial answer"})
        body = {"messages": messages}
        assert apply_task_reminder(body, _on(), input_tokens=99_999) is None

    def test_empty_message_array(self) -> None:
        assert apply_task_reminder({"messages": []}, _on(), input_tokens=99_999) is None
        assert apply_task_reminder({}, _on(), input_tokens=99_999) is None


class TestInjection:
    def test_appends_to_a_list_content_tail(self) -> None:
        body = {"messages": _messages()}
        assert apply_task_reminder(body, _on(), input_tokens=99_999) == TASK
        blocks = body["messages"][-1]["content"]
        assert blocks[0]["type"] == "tool_result"  # original content untouched
        assert blocks[-1]["text"].startswith(TASK_REMINDER_SENTINEL)
        assert TASK in blocks[-1]["text"]

    def test_appends_to_a_string_content_tail(self) -> None:
        messages = _messages()
        messages[-1] = {"role": "user", "content": "tool output text"}
        body = {"messages": messages}
        assert apply_task_reminder(body, _on(), input_tokens=99_999) == TASK
        assert body["messages"][-1]["content"].startswith("tool output text")
        assert TASK_REMINDER_SENTINEL in body["messages"][-1]["content"]

    def test_the_original_instruction_is_not_moved(self) -> None:
        """Duplicate, do not relocate — the placement study's actual finding."""
        body = {"messages": _messages()}
        apply_task_reminder(body, _on(), input_tokens=99_999)
        assert body["messages"][0]["content"][0]["text"] == TASK

    def test_is_idempotent(self) -> None:
        body = {"messages": _messages()}
        assert apply_task_reminder(body, _on(), input_tokens=99_999) == TASK
        assert apply_task_reminder(body, _on(), input_tokens=99_999) is None
        blocks = body["messages"][-1]["content"]
        assert sum(TASK_REMINDER_SENTINEL in str(block) for block in blocks) == 1

    def test_nothing_before_the_tail_message_changes(self) -> None:
        messages = _messages()
        body = {"messages": messages}
        snapshot = [str(message) for message in messages[:-1]]
        apply_task_reminder(body, _on(), input_tokens=99_999)
        assert [str(message) for message in body["messages"][:-1]] == snapshot


class TestResponsesFormat:
    def _input(self, distance: int = 20) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": TASK}]}
        ]
        for index in range(distance):
            items.append(
                {"type": "function_call_output", "call_id": f"c{index}", "output": "x"}
            )
        return items

    def test_appends_a_user_item(self) -> None:
        body = {"input": self._input()}
        assert apply_responses_task_reminder(body, _on(), input_tokens=99_999) == TASK
        last = body["input"][-1]
        assert last["role"] == "user"
        assert last["content"][0]["type"] == "input_text"
        assert TASK_REMINDER_SENTINEL in last["content"][0]["text"]

    def test_distance_counts_tool_output_items(self) -> None:
        body = {"input": self._input(distance=4)}
        assert apply_responses_task_reminder(body, _on(min_distance=12), input_tokens=99_999) is None

    def test_is_idempotent(self) -> None:
        body = {"input": self._input()}
        assert apply_responses_task_reminder(body, _on(), input_tokens=99_999) == TASK
        before = len(body["input"])
        assert apply_responses_task_reminder(body, _on(), input_tokens=99_999) is None
        assert len(body["input"]) == before

    def test_junk_input(self) -> None:
        assert apply_responses_task_reminder({}, _on(), input_tokens=99_999) is None
        assert apply_responses_task_reminder({"input": "text"}, _on(), input_tokens=99_999) is None
        assert apply_responses_task_reminder({"input": []}, _on(), input_tokens=99_999) is None


class TestSettingsFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "HEADROOM_TASK_REMINDER",
            "HEADROOM_TASK_REMINDER_TRIGGER_TOKENS",
            "HEADROOM_TASK_REMINDER_MIN_DISTANCE",
            "HEADROOM_TASK_REMINDER_MAX_CHARS",
        ):
            monkeypatch.delenv(name, raising=False)
        settings = TaskReminderSettings.from_env()
        assert not settings.enabled
        assert settings.trigger_tokens == 30_000
        assert settings.min_distance == 12
        assert settings.max_chars == 600

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_TASK_REMINDER", "1")
        monkeypatch.setenv("HEADROOM_TASK_REMINDER_MIN_DISTANCE", "4")
        settings = TaskReminderSettings.from_env()
        assert settings.enabled
        assert settings.min_distance == 4
