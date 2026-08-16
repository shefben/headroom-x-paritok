"""Tests for Anthropic server-side context editing (``clear_tool_uses``)."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.context_editing import (
    CLEAR_TOOL_USES_EDIT_TYPE,
    DEFAULT_CLEAR_AT_LEAST_TOKENS,
    DEFAULT_EXCLUDE_TOOLS,
    DEFAULT_KEEP_TOOL_USES,
    DEFAULT_TRIGGER_TOKENS,
    ContextEditingSettings,
    apply_context_editing,
    cleared_input_tokens,
    cleared_tool_uses,
    count_tool_uses,
)

_ENV_NAMES = (
    "HEADROOM_CONTEXT_EDITING",
    "HEADROOM_CONTEXT_EDITING_TRIGGER_TOKENS",
    "HEADROOM_CONTEXT_EDITING_KEEP_TOOL_USES",
    "HEADROOM_CONTEXT_EDITING_CLEAR_AT_LEAST_TOKENS",
    "HEADROOM_CONTEXT_EDITING_CLEAR_TOOL_INPUTS",
    "HEADROOM_CONTEXT_EDITING_EXCLUDE_TOOLS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _body(tool_uses: int) -> dict[str, Any]:
    """A minimal Anthropic body carrying ``tool_uses`` tool_use blocks."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
    for index in range(tool_uses):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"tu_{index}", "name": "Read", "input": {}}
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"tu_{index}", "content": "x" * 64}
                ],
            }
        )
    return {"model": "claude-opus-4-5", "messages": messages}


def _enabled(**overrides: Any) -> ContextEditingSettings:
    return ContextEditingSettings(enabled=True, **overrides)


class TestCountToolUses:
    def test_counts_tool_use_blocks_only(self) -> None:
        assert count_tool_uses(_body(3)["messages"]) == 3

    def test_string_content_is_not_a_crash(self) -> None:
        messages = [{"role": "user", "content": "plain string"}]
        assert count_tool_uses(messages) == 0

    def test_junk_entries_are_skipped(self) -> None:
        messages = [None, 7, {"role": "user"}, {"content": None}]
        assert count_tool_uses(messages) == 0

    def test_non_list_is_zero(self) -> None:
        assert count_tool_uses(None) == 0
        assert count_tool_uses("messages") == 0


class TestApply:
    def test_disabled_leaves_body_untouched(self) -> None:
        body = _body(10)
        result = apply_context_editing(body, ContextEditingSettings(enabled=False))
        assert not result.applied
        assert result.reason == "disabled"
        assert "context_management" not in body

    def test_below_keep_threshold_is_a_noop(self) -> None:
        # keep=3 retains everything when only 3 tool uses exist, so an edit
        # could never fire; adding the field would be pure risk.
        body = _body(DEFAULT_KEEP_TOOL_USES)
        result = apply_context_editing(body, _enabled())
        assert not result.applied
        assert result.reason == "below_keep_threshold"
        assert "context_management" not in body

    def test_applies_expected_edit_shape(self) -> None:
        body = _body(10)
        result = apply_context_editing(body, _enabled())
        assert result.applied
        assert result.tool_use_count == 10

        edits = body["context_management"]["edits"]
        assert len(edits) == 1
        edit = edits[0]
        assert edit["type"] == CLEAR_TOOL_USES_EDIT_TYPE
        assert edit["trigger"] == {"type": "input_tokens", "value": DEFAULT_TRIGGER_TOKENS}
        assert edit["keep"] == {"type": "tool_uses", "value": DEFAULT_KEEP_TOOL_USES}
        assert edit["clear_at_least"] == {
            "type": "input_tokens",
            "value": DEFAULT_CLEAR_AT_LEAST_TOKENS,
        }
        assert edit["exclude_tools"] == list(DEFAULT_EXCLUDE_TOOLS)
        # Omitted because Anthropic's own default is False; keeping the body
        # minimal keeps it closest to what the client would have sent.
        assert "clear_tool_inputs" not in edit

    def test_client_configured_edit_wins(self) -> None:
        body = _body(10)
        client_edit = {"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 9}}
        body["context_management"] = {"edits": [client_edit]}
        result = apply_context_editing(body, _enabled())
        assert not result.applied
        assert result.reason == "client_configured"
        assert body["context_management"]["edits"] == [client_edit]

    def test_client_unrelated_edit_is_preserved_and_ours_appended(self) -> None:
        body = _body(10)
        other = {"type": "some_future_edit_20991231"}
        body["context_management"] = {"edits": [other], "vendor_flag": True}
        result = apply_context_editing(body, _enabled())
        assert result.applied
        assert body["context_management"]["vendor_flag"] is True
        edits = body["context_management"]["edits"]
        assert edits[0] is other
        assert edits[1]["type"] == CLEAR_TOOL_USES_EDIT_TYPE

    def test_client_context_management_without_edits_key(self) -> None:
        body = _body(10)
        body["context_management"] = {}
        result = apply_context_editing(body, _enabled())
        assert result.applied
        assert len(body["context_management"]["edits"]) == 1

    def test_malformed_client_value_is_forwarded_unchanged(self) -> None:
        body = _body(10)
        body["context_management"] = "not-a-dict"
        result = apply_context_editing(body, _enabled())
        assert not result.applied
        assert result.reason == "client_malformed"
        assert body["context_management"] == "not-a-dict"

    def test_clear_at_least_zero_omits_the_key(self) -> None:
        body = _body(10)
        apply_context_editing(body, _enabled(clear_at_least_tokens=0))
        assert "clear_at_least" not in body["context_management"]["edits"][0]

    def test_clear_tool_inputs_opt_in(self) -> None:
        body = _body(10)
        apply_context_editing(body, _enabled(clear_tool_inputs=True))
        assert body["context_management"]["edits"][0]["clear_tool_inputs"] is True

    def test_empty_exclude_tools_omits_the_key(self) -> None:
        body = _body(10)
        apply_context_editing(body, _enabled(exclude_tools=()))
        assert "exclude_tools" not in body["context_management"]["edits"][0]


class TestSettingsFromEnv:
    def test_defaults_when_unset(self) -> None:
        settings = ContextEditingSettings.from_env()
        assert settings.enabled is False
        assert settings.trigger_tokens == DEFAULT_TRIGGER_TOKENS
        assert settings.keep_tool_uses == DEFAULT_KEEP_TOOL_USES
        assert settings.clear_at_least_tokens == DEFAULT_CLEAR_AT_LEAST_TOKENS
        assert settings.clear_tool_inputs is False
        assert settings.exclude_tools == DEFAULT_EXCLUDE_TOOLS

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING", "1")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_TRIGGER_TOKENS", "40000")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_KEEP_TOOL_USES", "5")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_CLEAR_AT_LEAST_TOKENS", "0")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_CLEAR_TOOL_INPUTS", "yes")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_EXCLUDE_TOOLS", "web_search, Read")

        settings = ContextEditingSettings.from_env()
        assert settings.enabled is True
        assert settings.trigger_tokens == 40000
        assert settings.keep_tool_uses == 5
        assert settings.clear_at_least_tokens == 0
        assert settings.clear_tool_inputs is True
        assert settings.exclude_tools == ("web_search", "Read")

    def test_explicit_enabled_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING", "0")
        assert ContextEditingSettings.from_env(enabled=True).enabled is True

    def test_invalid_numbers_fall_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_TRIGGER_TOKENS", "nonsense")
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_KEEP_TOOL_USES", "-4")
        settings = ContextEditingSettings.from_env()
        assert settings.trigger_tokens == DEFAULT_TRIGGER_TOKENS
        assert settings.keep_tool_uses == DEFAULT_KEEP_TOOL_USES

    def test_empty_exclude_tools_env_clears_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Set-but-empty means "exclude nothing", which is distinct from unset.
        monkeypatch.setenv("HEADROOM_CONTEXT_EDITING_EXCLUDE_TOOLS", "")
        assert ContextEditingSettings.from_env().exclude_tools == ()


class TestUsageParsing:
    def test_sums_applied_edits(self) -> None:
        usage = {
            "input_tokens": 100,
            "context_management": {
                "applied_edits": [
                    {"type": CLEAR_TOOL_USES_EDIT_TYPE, "cleared_input_tokens": 12000,
                     "cleared_tool_uses": 8},
                    {"type": CLEAR_TOOL_USES_EDIT_TYPE, "cleared_input_tokens": 3000,
                     "cleared_tool_uses": 2},
                ]
            },
        }
        assert cleared_input_tokens(usage) == 15000
        assert cleared_tool_uses(usage) == 10

    def test_absent_context_management_is_zero(self) -> None:
        assert cleared_input_tokens({"input_tokens": 5}) == 0
        assert cleared_tool_uses({"input_tokens": 5}) == 0

    def test_junk_shapes_are_zero(self) -> None:
        assert cleared_input_tokens(None) == 0
        assert cleared_input_tokens({"context_management": []}) == 0
        assert cleared_input_tokens({"context_management": {"applied_edits": "x"}}) == 0
        assert cleared_input_tokens({"context_management": {"applied_edits": [1, None]}}) == 0

    def test_bool_is_not_counted_as_int(self) -> None:
        # bool is a subclass of int; a provider sending `true` must not
        # silently register as one cleared token.
        usage = {"context_management": {"applied_edits": [{"cleared_input_tokens": True}]}}
        assert cleared_input_tokens(usage) == 0
