"""Tests for self-healing recovery of tools the filter dropped.

Recovery runs on the *request* side, not the response side. When the filter
withholds a schema the agent needs, the agent says so in plain text — and that
text arrives back at the proxy as the last assistant message of the very next
request. Detecting it there means one code path covers both streaming and
non-streaming responses, and the tool is restored exactly when it is needed.

Selection quality depends on the real bge-small model, so those tests skip when
it is unavailable. The structural ones — no-op without help text, fail-open,
stickiness bookkeeping — always run.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.paritok.config import ParitokConfig
from headroom.paritok.tool_filter import embeddings_available
from headroom.paritok.tool_stage import (
    last_assistant_text,
    reset_selector,
    select_tools,
    tool_filter_enabled,
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
        "input_schema": {"type": "object", "properties": {}},
    }


TOOLS = [
    anthropic_tool("Read", "Read a file from the filesystem"),
    anthropic_tool("Grep", "Search file contents with a regular expression"),
    anthropic_tool("Glob", "Find files matching a glob pattern"),
    anthropic_tool("Edit", "Edit a file in place"),
    anthropic_tool("SendSlackMessage", "Post a message to a Slack channel"),
    anthropic_tool("CreateJiraIssue", "Open a ticket in Jira"),
    anthropic_tool("QueryDatabase", "Run a SQL query against the production database"),
]

CODING_QUERY = "fix the crash in parser.py"
SLACK_HELP = "I don't have a tool available to send a Slack message, so I stopped here."

CONFIG = ParitokConfig(tool_k_min=3, tool_topk=4, tool_recover_k=1)


def kept_names(payload: dict[str, Any]) -> list[str]:
    return [tool["name"] for tool in payload["tools"]]


def run_turn(session: str, query: str, assistant_text: str = "") -> list[str]:
    payload, _, _, _ = select_tools(
        {"tools": TOOLS},
        session_id=session,
        query=query,
        assistant_text=assistant_text,
        config=CONFIG,
    )
    return kept_names(payload)


class TestSharedEnabledFlag:
    """Both proxy handlers resolve the lever through one function.

    It caches on purpose: the rollout snapshot is process-level provenance, and
    a mid-session flip would change the tools array underneath the provider's
    prefix cache.
    """

    def test_off_by_default(self) -> None:
        assert tool_filter_enabled() is False

    def test_reads_the_legacy_env_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARITOK_TOOL_FILTER", "1")
        reset_selector()

        assert tool_filter_enabled() is True

    def test_the_answer_is_cached_until_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARITOK_TOOL_FILTER", "1")
        reset_selector()
        assert tool_filter_enabled() is True

        monkeypatch.delenv("PARITOK_TOOL_FILTER")
        assert tool_filter_enabled() is True  # still cached

        reset_selector()
        assert tool_filter_enabled() is False


class TestLastAssistantText:
    def test_reads_the_trailing_assistant_turn(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "I don't have that tool."},
            {"role": "user", "content": "try again"},
        ]
        assert last_assistant_text(messages) == "I don't have that tool."

    def test_joins_anthropic_content_blocks(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "No Slack tool"},
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "is available."},
                ],
            },
            {"role": "user", "content": "go"},
        ]
        assert last_assistant_text(messages) == "No Slack tool is available."

    def test_returns_empty_when_there_is_no_assistant_turn(self) -> None:
        assert last_assistant_text([{"role": "user", "content": "hi"}]) == ""

    def test_tolerates_junk(self) -> None:
        assert last_assistant_text(None) == ""
        assert last_assistant_text(["not a dict"]) == ""


class TestNoRecoveryWithoutASignal:
    @needs_embeddings
    def test_a_coding_query_drops_the_slack_tool(self) -> None:
        assert "SendSlackMessage" not in run_turn("s1", CODING_QUERY)

    @needs_embeddings
    def test_ordinary_assistant_text_changes_nothing(self) -> None:
        run_turn("s1", CODING_QUERY)
        second = run_turn("s1", CODING_QUERY, assistant_text="Fixed the off-by-one in the parser.")

        assert "SendSlackMessage" not in second

    def test_empty_help_text_never_calls_the_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The detector must gate the embedder, not the other way round.

        Recovery embeds the whole dropped pool, which is far too expensive to
        pay on every turn.
        """
        import headroom.paritok.tool_stage as stage

        def explode(*args: Any, **kwargs: Any) -> list[str]:
            raise AssertionError("recovery ran without a missing-tool signal")

        monkeypatch.setattr(stage, "recover_tools_from_help", explode)
        run_turn("s1", CODING_QUERY, assistant_text="All done.")


class TestRecovery:
    @needs_embeddings
    def test_the_named_tool_comes_back(self) -> None:
        assert "SendSlackMessage" not in run_turn("s1", CODING_QUERY)

        recovered = run_turn("s1", CODING_QUERY, assistant_text=SLACK_HELP)

        assert "SendSlackMessage" in recovered

    @needs_embeddings
    def test_recovery_is_sticky(self) -> None:
        """Pinning is the point: the same miss must not repeat every turn."""
        run_turn("s1", CODING_QUERY)
        run_turn("s1", CODING_QUERY, assistant_text=SLACK_HELP)

        assert "SendSlackMessage" in run_turn("s1", CODING_QUERY)

    @needs_embeddings
    def test_recovery_does_not_leak_across_sessions(self) -> None:
        run_turn("s1", CODING_QUERY, assistant_text=SLACK_HELP)

        assert "SendSlackMessage" not in run_turn("s2", CODING_QUERY)

    @needs_embeddings
    def test_the_kept_set_stays_small(self) -> None:
        """Recovery restores what was asked for, not the whole dropped pool."""
        before = run_turn("s1", CODING_QUERY)
        after = run_turn("s1", CODING_QUERY, assistant_text=SLACK_HELP)

        assert len(after) <= len(before) + CONFIG.tool_recover_k


class TestFailOpen:
    def test_a_recovery_crash_still_returns_a_usable_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.paritok.tool_stage as stage

        def explode(*args: Any, **kwargs: Any) -> list[str]:
            raise RuntimeError("embedder exploded")

        monkeypatch.setattr(stage, "recover_tools_from_help", explode)

        payload, _, _, _ = select_tools(
            {"tools": TOOLS},
            session_id="s1",
            query=CODING_QUERY,
            assistant_text=SLACK_HELP,
            config=CONFIG,
        )

        assert payload["tools"]

    def test_recovery_is_skipped_when_selection_is_skipped(self) -> None:
        """Below k_min nothing was dropped, so there is nothing to recover."""
        small = TOOLS[:2]
        payload, modified, _, _ = select_tools(
            {"tools": small},
            session_id="s1",
            query=CODING_QUERY,
            assistant_text=SLACK_HELP,
            config=CONFIG,
        )

        assert modified is False
        assert payload["tools"] == small
