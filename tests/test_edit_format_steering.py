"""Tests for edit-format steering (the output-token lever)."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.edit_format_steering import (
    apply_edit_format_steering,
    apply_openai_chat_edit_format_steering,
    apply_openai_responses_edit_format_steering,
)
from headroom.proxy.output_edit_format_policy import (
    EDIT_FORMAT_SENTINEL,
    edit_format_text,
    request_has_edit_tools,
    resolve_edit_format_mode,
)
from headroom.proxy.output_shaper import (
    OutputShaperSettings,
    shape_openai_chat_request,
    shape_openai_responses_request,
    shape_request,
)


class TestModeResolution:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, "off"),
            ("", "off"),
            ("off", "off"),
            ("0", "off"),
            ("minimal", "minimal"),
            ("STRICT", "strict"),
            ("1", "minimal"),
            ("yes", "minimal"),
            ("nonsense", "off"),
        ],
    )
    def test_resolution(self, raw: str | None, expected: str) -> None:
        assert resolve_edit_format_mode(raw) == expected

    def test_off_has_no_text(self) -> None:
        assert edit_format_text("off") is None

    def test_strict_is_a_superset_of_minimal(self) -> None:
        minimal = edit_format_text("minimal") or ""
        strict = edit_format_text("strict") or ""
        assert len(strict) > len(minimal)
        assert "smallest targeted edit" in minimal
        assert "do not restate what" in strict


class TestEditToolDetection:
    def test_anthropic_named_tools(self) -> None:
        assert request_has_edit_tools([{"name": "Edit"}])
        assert request_has_edit_tools([{"name": "write"}])
        assert not request_has_edit_tools([{"name": "Read"}, {"name": "Grep"}])

    def test_anthropic_typed_text_editor(self) -> None:
        assert request_has_edit_tools([{"type": "text_editor_20250124", "name": "str_replace"}])

    def test_openai_chat_nested_function(self) -> None:
        assert request_has_edit_tools(
            [{"type": "function", "function": {"name": "apply_patch"}}]
        )

    def test_junk_shapes(self) -> None:
        assert not request_has_edit_tools(None)
        assert not request_has_edit_tools("Edit")
        assert not request_has_edit_tools([None, 3, {"name": None}])


class TestAnthropicInjection:
    def test_appends_to_string_system(self) -> None:
        body: dict[str, Any] = {"system": "You are helpful."}
        assert apply_edit_format_steering(body, "minimal")
        assert body["system"][0]["text"] == "You are helpful."
        assert body["system"][1]["text"].startswith(EDIT_FORMAT_SENTINEL)

    def test_appends_after_cache_control_block(self) -> None:
        cached = {"type": "text", "text": "big prefix", "cache_control": {"type": "ephemeral"}}
        body: dict[str, Any] = {"system": [cached]}
        assert apply_edit_format_steering(body, "strict")
        # The cached block must be untouched and still first, or the provider
        # prefix cache is invalidated by our own steering.
        assert body["system"][0] is cached
        assert body["system"][1]["text"].startswith(EDIT_FORMAT_SENTINEL)

    def test_is_idempotent(self) -> None:
        body: dict[str, Any] = {"system": "base"}
        assert apply_edit_format_steering(body, "minimal")
        assert not apply_edit_format_steering(body, "minimal")
        assert len(body["system"]) == 2

    def test_mode_change_replaces_in_place(self) -> None:
        body: dict[str, Any] = {"system": "base"}
        apply_edit_format_steering(body, "minimal")
        assert apply_edit_format_steering(body, "strict")
        assert len(body["system"]) == 2
        assert "do not restate what" in body["system"][1]["text"]

    def test_missing_system_creates_one(self) -> None:
        body: dict[str, Any] = {}
        assert apply_edit_format_steering(body, "minimal")
        assert body["system"][0]["text"].startswith(EDIT_FORMAT_SENTINEL)

    def test_malformed_block_does_not_raise(self) -> None:
        body: dict[str, Any] = {"system": [{"type": "text", "text": None}]}
        assert apply_edit_format_steering(body, "minimal")

    def test_off_is_a_noop(self) -> None:
        body: dict[str, Any] = {"system": "base"}
        assert not apply_edit_format_steering(body, "off")
        assert body["system"] == "base"


class TestOpenAIChatInjection:
    def test_appends_to_last_system_message(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "first"},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "second"},
            ]
        }
        assert apply_openai_chat_edit_format_steering(body, "minimal")
        assert EDIT_FORMAT_SENTINEL in body["messages"][2]["content"]
        assert body["messages"][0]["content"] == "first"

    def test_inserts_a_system_message_when_absent(self) -> None:
        body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
        assert apply_openai_chat_edit_format_steering(body, "minimal")
        assert body["messages"][0]["role"] == "system"

    def test_content_part_list(self) -> None:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": [{"type": "text", "text": "base"}]}]
        }
        assert apply_openai_chat_edit_format_steering(body, "minimal")
        assert body["messages"][0]["content"][1]["text"].startswith(EDIT_FORMAT_SENTINEL)
        assert not apply_openai_chat_edit_format_steering(body, "minimal")

    def test_is_idempotent_on_string_content(self) -> None:
        body: dict[str, Any] = {"messages": [{"role": "system", "content": "base"}]}
        assert apply_openai_chat_edit_format_steering(body, "minimal")
        first = body["messages"][0]["content"]
        assert not apply_openai_chat_edit_format_steering(body, "minimal")
        assert body["messages"][0]["content"] == first


class TestOpenAIResponsesInjection:
    def test_appends_to_instructions(self) -> None:
        body: dict[str, Any] = {"instructions": "base"}
        assert apply_openai_responses_edit_format_steering(body, "minimal")
        assert body["instructions"].startswith("base")
        assert EDIT_FORMAT_SENTINEL in body["instructions"]

    def test_creates_instructions_when_absent(self) -> None:
        body: dict[str, Any] = {}
        assert apply_openai_responses_edit_format_steering(body, "strict")
        assert body["instructions"].startswith(EDIT_FORMAT_SENTINEL)

    def test_is_idempotent(self) -> None:
        body: dict[str, Any] = {"instructions": "base"}
        apply_openai_responses_edit_format_steering(body, "minimal")
        assert not apply_openai_responses_edit_format_steering(body, "minimal")


def _settings(mode: str) -> OutputShaperSettings:
    # verbosity off / effort router off so the assertions isolate this lever.
    return OutputShaperSettings(
        enabled=True,
        verbosity_level=0,
        effort_router_enabled=False,
        edit_format_mode=mode,
    )


class TestShaperIntegration:
    def test_anthropic_applies_when_edit_tools_present(self) -> None:
        body: dict[str, Any] = {"system": "base", "tools": [{"name": "Edit"}], "messages": []}
        result = shape_request(body, _settings("minimal"))
        assert result.changed
        assert result.labels == ["output_shaper:edit_format:minimal"]

    def test_anthropic_skips_without_edit_tools(self) -> None:
        body: dict[str, Any] = {"system": "base", "tools": [{"name": "Read"}], "messages": []}
        result = shape_request(body, _settings("strict"))
        assert not result.changed
        assert body["system"] == "base"

    def test_anthropic_skips_when_off(self) -> None:
        body: dict[str, Any] = {"system": "base", "tools": [{"name": "Edit"}], "messages": []}
        assert not shape_request(body, _settings("off")).changed

    def test_openai_chat_path(self) -> None:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": "base"}],
            "tools": [{"type": "function", "function": {"name": "write"}}],
        }
        result = shape_openai_chat_request(body, _settings("minimal"))
        assert result.changed
        assert EDIT_FORMAT_SENTINEL in body["messages"][0]["content"]

    def test_openai_responses_path(self) -> None:
        body: dict[str, Any] = {
            "instructions": "base",
            "tools": [{"name": "apply_patch"}],
            "input": [],
        }
        result = shape_openai_responses_request(body, _settings("strict"))
        assert result.changed
        assert EDIT_FORMAT_SENTINEL in body["instructions"]

    def test_disabled_shaper_short_circuits(self) -> None:
        body: dict[str, Any] = {"system": "base", "tools": [{"name": "Edit"}], "messages": []}
        settings = OutputShaperSettings(enabled=False, edit_format_mode="strict")
        assert not shape_request(body, settings).changed
        assert body["system"] == "base"

    def test_repeated_shaping_stays_byte_stable(self) -> None:
        body: dict[str, Any] = {"system": "base", "tools": [{"name": "Edit"}], "messages": []}
        settings = _settings("minimal")
        shape_request(body, settings)
        snapshot = str(body["system"])
        assert not shape_request(body, settings).changed
        assert str(body["system"]) == snapshot


class TestSettingsFromEnv:
    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_EDIT_FORMAT", raising=False)
        assert OutputShaperSettings.from_env(enabled=True).edit_format_mode == "off"

    def test_env_selects_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_EDIT_FORMAT", "strict")
        assert OutputShaperSettings.from_env(enabled=True).edit_format_mode == "strict"
