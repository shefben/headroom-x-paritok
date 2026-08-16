"""Tests for Anthropic server-side compaction."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.context_compaction import (
    COMPACT_EDIT_TYPE,
    CompactionSettings,
    apply_compaction,
    compaction_applied,
    model_supports_compaction,
    reset_compaction,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_compaction()


def _body(model: str = "claude-opus-4-6", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": []}
    body.update(extra)
    return body


def _on() -> CompactionSettings:
    return CompactionSettings(enabled=True)


class TestModelGating:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-opus-5",
            "claude-sonnet-5",
            "anthropic/claude-opus-4-6-20260112",
        ],
    )
    def test_supported(self, model: str) -> None:
        assert model_supports_compaction(model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5",
            "claude-sonnet-4-0",
            "claude-3-5-sonnet-20241022",
            "claude-haiku-4-5",
            "gpt-5.2",
            "",
            None,
            123,
        ],
    )
    def test_unsupported(self, model: Any) -> None:
        assert not model_supports_compaction(model)

    def test_unknown_names_are_refused_not_guessed(self) -> None:
        # A whitelist walked forward: a new alias is simply not compacted,
        # rather than 400-ing every request that mentions it.
        assert not model_supports_compaction("claude-next-turbo")


class TestApply:
    def test_disabled_is_inert(self) -> None:
        body = _body()
        result = apply_compaction(body, CompactionSettings())
        assert not result.applied
        assert result.reason == "disabled"
        assert "context_management" not in body

    def test_unsupported_model_is_inert(self) -> None:
        body = _body("claude-opus-4-5")
        result = apply_compaction(body, _on())
        assert result.reason == "model_unsupported"
        assert "context_management" not in body

    def test_applies_the_edit(self) -> None:
        body = _body()
        result = apply_compaction(body, _on())
        assert result.applied
        edits = body["context_management"]["edits"]
        assert edits[0]["type"] == COMPACT_EDIT_TYPE
        assert edits[0]["trigger"] == {"type": "input_tokens", "value": 150_000}

    def test_explicit_model_argument_wins(self) -> None:
        body = _body("claude-opus-4-5")
        assert apply_compaction(body, _on(), model="claude-opus-4-6").applied

    def test_trigger_is_configurable(self) -> None:
        body = _body()
        apply_compaction(body, CompactionSettings(enabled=True, trigger_tokens=42_000))
        assert body["context_management"]["edits"][0]["trigger"]["value"] == 42_000


class TestClientPrecedence:
    def test_client_compaction_config_wins(self) -> None:
        body = _body(
            context_management={"edits": [{"type": "compact_20260112", "trigger": {}}]}
        )
        result = apply_compaction(body, _on())
        assert not result.applied
        assert result.reason == "client_configured"
        assert len(body["context_management"]["edits"]) == 1

    def test_any_compact_version_counts_as_configured(self) -> None:
        body = _body(context_management={"edits": [{"type": "compact_20990101"}]})
        assert apply_compaction(body, _on()).reason == "client_configured"

    def test_malformed_client_value_is_left_alone(self) -> None:
        body = _body(context_management="nope")
        result = apply_compaction(body, _on())
        assert result.reason == "client_malformed"
        assert body["context_management"] == "nope"

    def test_unrelated_client_edits_are_preserved(self) -> None:
        clear = {"type": "clear_tool_uses_20250919"}
        body = _body(context_management={"edits": [clear]})
        assert apply_compaction(body, _on()).applied
        edits = body["context_management"]["edits"]
        assert edits[0] is clear
        assert edits[1]["type"] == COMPACT_EDIT_TYPE


class TestCoexistenceWithContextEditing:
    def test_both_levers_merge_into_one_array(self) -> None:
        from headroom.proxy.context_editing import (
            ContextEditingSettings,
            apply_context_editing,
        )

        body = _body()
        body["messages"] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{index}", "name": "Bash", "input": {}}
                ],
            }
            for index in range(8)
        ]
        assert apply_context_editing(body, ContextEditingSettings(enabled=True)).applied
        assert apply_compaction(body, _on()).applied
        types = [edit["type"] for edit in body["context_management"]["edits"]]
        assert types == ["clear_tool_uses_20250919", COMPACT_EDIT_TYPE]


class TestReporting:
    def test_reads_applied_edits(self) -> None:
        usage = {
            "context_management": {
                "applied_edits": [{"type": "compact_20260112", "cleared_input_tokens": 9}]
            }
        }
        assert compaction_applied(usage)

    def test_other_edit_types_do_not_count(self) -> None:
        usage = {
            "context_management": {
                "applied_edits": [{"type": "clear_tool_uses_20250919"}]
            }
        }
        assert not compaction_applied(usage)

    @pytest.mark.parametrize(
        "usage",
        [None, {}, {"context_management": None}, {"context_management": {"applied_edits": 3}}],
    )
    def test_junk_usage_reads_as_false(self, usage: Any) -> None:
        assert not compaction_applied(usage)


class TestSettingsFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_CONTEXT_COMPACTION", raising=False)
        monkeypatch.delenv("HEADROOM_CONTEXT_COMPACTION_TRIGGER_TOKENS", raising=False)
        settings = CompactionSettings.from_env()
        assert not settings.enabled
        assert settings.trigger_tokens == 150_000

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_COMPACTION", "1")
        monkeypatch.setenv("HEADROOM_CONTEXT_COMPACTION_TRIGGER_TOKENS", "80000")
        settings = CompactionSettings.from_env()
        assert settings.enabled
        assert settings.trigger_tokens == 80_000

    def test_invalid_trigger_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_COMPACTION_TRIGGER_TOKENS", "nope")
        assert CompactionSettings.from_env().trigger_tokens == 150_000

    def test_enabled_override_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CONTEXT_COMPACTION", "0")
        assert CompactionSettings.from_env(enabled=True).enabled
