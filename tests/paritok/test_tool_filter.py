"""Tests for semantic tool-schema selection.

Selection quality tests use the real bge-small model (CPU, ~130MB) and are
skipped when it is unavailable, so the suite still runs in a minimal install.
The structural tests — stubbing, dropping, cache stability, fail-open — use no
model at all and always run.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.paritok.config import ParitokConfig
from headroom.paritok.tool_filter import (
    _mcp_signal_score,
    _name_words,
    apply_selection_adaptive,
    embeddings_available,
    looks_like_missing_tool_help,
    tool_description,
    tool_name,
)
from headroom.paritok.tool_stage import (
    dropped_tools,
    recover_for_session,
    reset_selector,
    select_tools,
)

needs_embeddings = pytest.mark.skipif(
    not embeddings_available(), reason="fastembed/bge-small unavailable"
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_selector()
    yield
    reset_selector()


def anthropic_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {"arg": {"type": "string", "description": "an argument"}},
        },
    }


CODING_TOOLS = [
    anthropic_tool("Read", "Read a file from the filesystem"),
    anthropic_tool("Grep", "Search file contents with a regular expression"),
    anthropic_tool("Glob", "Find files matching a glob pattern"),
    anthropic_tool("Bash", "Run a shell command"),
    anthropic_tool("Edit", "Edit a file in place"),
    anthropic_tool("Write", "Write a file to disk"),
]

MCP_TOOLS = [
    anthropic_tool("mcp__gmail__send", "Send an email message via Gmail"),
    anthropic_tool("mcp__calendar__create_event", "Create a calendar event"),
    anthropic_tool("mcp__drive__upload", "Upload a file to Google Drive"),
    anthropic_tool("mcp__slack__post", "Post a message to a Slack channel"),
    anthropic_tool("mcp__jira__create_issue", "Create a Jira issue"),
    anthropic_tool("mcp__stripe__refund", "Issue a Stripe refund"),
]

ALL_TOOLS = CODING_TOOLS + MCP_TOOLS


class TestNameSplitting:
    def test_snake_case(self) -> None:
        assert _name_words("create_event") == "create event"

    def test_camel_case(self) -> None:
        assert _name_words("createEvent") == "create Event"

    def test_mcp_prefix_is_stripped(self) -> None:
        assert _name_words("mcp__gmail__send_email") == "gmail send email"

    def test_dotted(self) -> None:
        assert _name_words("drive.files.list") == "drive files list"


class TestToolAccessors:
    def test_anthropic_shape(self) -> None:
        tool = anthropic_tool("Read", "Read a file")
        assert tool_name(tool) == "Read"
        assert tool_description(tool) == "Read a file"

    def test_openai_shape(self) -> None:
        tool = {"type": "function", "function": {"name": "Read", "description": "Read a file"}}
        assert tool_name(tool) == "Read"
        assert tool_description(tool) == "Read a file"

    def test_missing_fields(self) -> None:
        assert tool_name({}) == ""
        assert tool_description({}) == ""


class TestApplySelection:
    def test_selected_tools_keep_full_schema(self) -> None:
        result = apply_selection_adaptive(ALL_TOOLS, ["Read", "Grep"])
        kept = {tool_name(t): t for t in result}
        assert kept["Read"]["input_schema"]["properties"]

    def test_unselected_standard_tools_are_dropped(self) -> None:
        result = apply_selection_adaptive(CODING_TOOLS, ["Read", "Grep"])
        assert {tool_name(t) for t in result} == {"Read", "Grep"}

    def test_mcp_tools_are_stubbed_when_signal_fires(self) -> None:
        """A top-ranked MCP tool implies an MCP task, so keep the rest discoverable."""
        result = apply_selection_adaptive(ALL_TOOLS, ["mcp__gmail__send", "Read"])
        stubs = [t for t in result if "[deferred]" in tool_description(t)]
        assert stubs
        assert all(t["input_schema"]["properties"] == {} for t in stubs)

    def test_pure_coding_selection_drops_mcp_entirely(self) -> None:
        """A coding turn should pay nothing for MCP servers it will never call."""
        result = apply_selection_adaptive(ALL_TOOLS, ["Read", "Grep", "Edit", "Bash"])
        assert not [t for t in result if tool_name(t).startswith("mcp__")]

    def test_openai_stubs_use_the_openai_shape(self) -> None:
        openai_tools = [
            {"type": "function", "function": {"name": "mcp__gmail__send", "description": "d"}},
            {"type": "function", "function": {"name": "Read", "description": "read"}},
        ]
        result = apply_selection_adaptive(openai_tools, ["mcp__gmail__send"], wire="openai")
        stub = [t for t in result if t.get("name") == "Read"]
        # Read is a standard tool and unselected, so it is dropped, not stubbed.
        assert not stub

    def test_responses_stubs_are_flat(self) -> None:
        tools = [*MCP_TOOLS, anthropic_tool("mcp__zoom__join", "Join a Zoom meeting")]
        result = apply_selection_adaptive(tools, ["mcp__gmail__send", "Read"], wire="openai")
        stub = next(t for t in result if "[deferred]" in tool_description(t))

        assert stub["type"] == "function"
        assert "function" not in stub
        assert stub["parameters"] == {"type": "object", "properties": {}}

    def test_chat_completions_stubs_are_nested(self) -> None:
        """A flat stub on the chat wire is rejected by the provider outright."""
        tools = [*MCP_TOOLS, anthropic_tool("mcp__zoom__join", "Join a Zoom meeting")]
        result = apply_selection_adaptive(tools, ["mcp__gmail__send", "Read"], wire="openai_chat")
        stub = next(t for t in result if "[deferred]" in tool_description(t))

        assert stub["type"] == "function"
        assert set(stub["function"]) == {"name", "description", "parameters"}
        assert stub["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_selection_is_smaller_than_the_input(self) -> None:
        result = apply_selection_adaptive(ALL_TOOLS, ["Read", "Grep"])
        assert len(result) < len(ALL_TOOLS)

    def test_core_exec_tools_survive_an_unrelated_selection(self) -> None:
        """Dropping the agent's only way to act would break it outright."""
        tools = [*MCP_TOOLS, anthropic_tool("shell", "Run a shell command")]
        result = apply_selection_adaptive(tools, ["mcp__gmail__send"])
        kept = {tool_name(t) for t in result}
        assert "shell" in kept

    def test_core_exec_tool_keeps_its_full_schema(self) -> None:
        tools = [*MCP_TOOLS, anthropic_tool("apply_patch", "Apply a patch to a file")]
        result = apply_selection_adaptive(tools, ["mcp__gmail__send"])
        patch = next(t for t in result if tool_name(t) == "apply_patch")
        assert patch["input_schema"]["properties"]
        assert "[deferred]" not in tool_description(patch)


class TestMcpSignal:
    def test_top_ranked_mcp_fires(self) -> None:
        assert _mcp_signal_score(["mcp__gmail__send", "Read"]) >= 1.0

    def test_low_ranked_mcp_does_not_fire(self) -> None:
        ranked = ["Read", "Grep", "Edit", "Bash", "Write", "Glob", "mcp__jira__create_issue"]
        assert _mcp_signal_score(ranked) < 1.0

    def test_no_mcp_scores_zero(self) -> None:
        assert _mcp_signal_score(["Read", "Grep"]) == 0.0


class TestMissingToolDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't have a calendar integration available",
            "There's no Gmail tool available in this session",
            "That tool isn't available",
            "I can't complete this request",
        ],
    )
    def test_detects_missing_capability(self, text: str) -> None:
        assert looks_like_missing_tool_help(text)

    @pytest.mark.parametrize(
        "text",
        ["I read the file and fixed the bug.", "Here is the refactored function.", ""],
    )
    def test_ignores_normal_replies(self, text: str) -> None:
        assert not looks_like_missing_tool_help(text)


class TestSelectToolsStage:
    def test_small_tool_sets_are_left_alone(self) -> None:
        payload = {"tools": CODING_TOOLS[:3]}
        result, modified, _b, _a = select_tools(payload, session_id="s", query="fix the bug")
        assert not modified
        assert result["tools"] == CODING_TOOLS[:3]

    def test_missing_tools_key_is_a_noop(self) -> None:
        payload: dict[str, Any] = {"messages": []}
        result, modified, _b, _a = select_tools(payload, session_id="s", query="hi")
        assert not modified
        assert result is payload

    def test_failure_keeps_all_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken embedder must send every tool, not none of them."""
        import headroom.paritok.tool_stage as stage

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("embedder exploded")

        monkeypatch.setattr(stage, "_get_selector", boom)
        payload = {"tools": ALL_TOOLS}

        result, modified, _b, _a = select_tools(payload, session_id="s", query="fix the bug")

        assert not modified
        assert result["tools"] == ALL_TOOLS

    def test_original_payload_is_not_mutated(self) -> None:
        payload = {"tools": list(ALL_TOOLS), "model": "x"}
        select_tools(payload, session_id="s", query="fix the bug in the parser")
        assert len(payload["tools"]) == len(ALL_TOOLS)

    @needs_embeddings
    def test_coding_query_keeps_coding_tools(self) -> None:
        payload = {"tools": ALL_TOOLS}
        result, modified, before, after = select_tools(
            payload, session_id="s1", query="fix the bug in the parser function"
        )
        assert modified
        assert after < before
        kept = {tool_name(t) for t in result["tools"]}
        assert "Read" in kept
        assert "Grep" in kept

    @needs_embeddings
    def test_selection_is_frozen_across_turns(self) -> None:
        """Byte-stable tools[] across turns is what keeps the prefix cache warm."""
        first, _m, _b, _a = select_tools(
            {"tools": ALL_TOOLS}, session_id="s2", query="fix the bug in the parser"
        )
        second, _m2, _b2, _a2 = select_tools(
            {"tools": ALL_TOOLS}, session_id="s2", query="now send an email to the team"
        )
        assert first["tools"] == second["tools"]

    @needs_embeddings
    def test_separate_sessions_select_independently(self) -> None:
        coding, _m, _b, _a = select_tools(
            {"tools": ALL_TOOLS}, session_id="code", query="fix the bug in the parser"
        )
        mail, _m2, _b2, _a2 = select_tools(
            {"tools": ALL_TOOLS}, session_id="mail", query="send an email and schedule a meeting"
        )
        assert {tool_name(t) for t in coding["tools"]} != {tool_name(t) for t in mail["tools"]}

    @needs_embeddings
    def test_recovery_pins_tools_into_the_session(self) -> None:
        selected, _m, _b, _a = select_tools(
            {"tools": ALL_TOOLS}, session_id="rec", query="fix the bug in the parser"
        )
        candidates = dropped_tools(ALL_TOOLS, selected["tools"])
        assert candidates

        recovered = recover_for_session("rec", "I don't have a Gmail tool available", candidates)
        assert recovered

        after, _m2, _b2, _a2 = select_tools(
            {"tools": ALL_TOOLS}, session_id="rec", query="fix the bug in the parser"
        )
        assert set(recovered) & {tool_name(t) for t in after["tools"]}


class TestDroppedTools:
    def test_reports_the_difference(self) -> None:
        current = CODING_TOOLS[:2]
        missing = dropped_tools(ALL_TOOLS, current)
        assert {tool_name(t) for t in missing} == {tool_name(t) for t in ALL_TOOLS[2:]}

    def test_nothing_dropped(self) -> None:
        assert dropped_tools(ALL_TOOLS, ALL_TOOLS) == []


class TestConfigDefaults:
    def test_selection_width_defaults(self) -> None:
        config = ParitokConfig()
        assert config.tool_k_min == 5
        assert config.tool_topk == 8
        assert config.tool_alpha == 0.9
