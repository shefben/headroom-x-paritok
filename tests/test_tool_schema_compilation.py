"""Tests for deterministic tool-schema compilation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.proxy.tool_schema_compilation import (
    compile_tools,
    reset_tool_schema_compilation,
    resolve_compile_mode,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_tool_schema_compilation()


def _tool(name: str = "search", **overrides: Any) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": name,
        "description": "Search the index for matching records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query"},
                "limit": {
                    "type": "integer",
                    "default": 25,
                    "description": "Maximum number of records to return",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "exact"],
                    "description": "Search strategy to use",
                },
            },
            "required": ["query"],
        },
    }
    tool.update(overrides)
    return tool


def _schema(tool: dict[str, Any]) -> dict[str, Any]:
    return tool["input_schema"]


class TestModeResolution:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, "safe"),
            ("", "safe"),
            ("1", "safe"),
            ("yes", "safe"),
            ("0", "off"),
            ("off", "off"),
            ("safe", "safe"),
            ("FULL", "full"),
            ("nonsense", "off"),
        ],
    )
    def test_resolution(self, raw: str | None, expected: str) -> None:
        assert resolve_compile_mode(raw) == expected


class TestOffAndGuards:
    def test_off_is_a_noop(self) -> None:
        tools = [_tool()]
        result = compile_tools(tools, "off")
        assert not result.modified
        assert result.tools is tools

    def test_empty_and_junk_inputs(self) -> None:
        assert not compile_tools([], "full").modified
        assert not compile_tools(None, "full").modified
        assert not compile_tools("tools", "full").modified

    def test_non_dict_entries_survive_untouched(self) -> None:
        tools = [None, 3, _tool()]
        result = compile_tools(tools, "full")
        assert result.tools[0] is None
        assert result.tools[1] == 3

    def test_provider_native_tool_without_schema_is_skipped(self) -> None:
        tools = [{"type": "text_editor_20250124", "name": "str_replace_editor"}]
        result = compile_tools(tools, "full")
        assert not result.modified

    def test_compilation_that_would_grow_is_discarded(self) -> None:
        # A schema with no descriptions and one short parameter has nothing to
        # absorb, so the signature line would be pure overhead.
        tools = [
            {
                "name": "p",
                "description": "d",
                "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            }
        ]
        result = compile_tools(tools, "full")
        assert not result.modified
        assert result.tools is tools


class TestSafeMode:
    def test_keeps_the_schema_semantically_intact(self) -> None:
        result = compile_tools([_tool()], "safe")
        assert result.modified
        schema = _schema(result.tools[0])
        assert schema["properties"]["mode"]["enum"] == ["fast", "exact"]
        assert schema["properties"]["limit"]["default"] == 25
        assert schema["required"] == ["query"]

    def test_drops_only_redundant_descriptions(self) -> None:
        result = compile_tools([_tool()], "safe")
        properties = _schema(result.tools[0])["properties"]
        # "The query" restates the parameter name; the other two do not.
        assert "description" not in properties["query"]
        assert properties["limit"]["description"]
        assert properties["mode"]["description"]

    def test_emits_no_signature(self) -> None:
        # With the JSON still present a signature would state every type twice.
        result = compile_tools([_tool()], "safe")
        assert "params:" not in result.tools[0]["description"]

    def test_actually_shrinks(self) -> None:
        result = compile_tools([_tool()], "safe")
        assert result.after_bytes < result.before_bytes


class TestFullMode:
    def test_signature_carries_types_defaults_and_enums(self) -> None:
        result = compile_tools([_tool()], "full")
        description = result.tools[0]["description"]
        assert "params: query:str limit:int=25 mode:{fast|exact}?" in description

    def test_schema_is_reduced_to_names_and_required(self) -> None:
        result = compile_tools([_tool()], "full")
        schema = _schema(result.tools[0])
        assert schema["type"] == "object"
        assert schema["properties"] == {"query": {}, "limit": {}, "mode": {}}
        assert schema["required"] == ["query"]

    def test_non_redundant_descriptions_survive_as_doc_lines(self) -> None:
        description = compile_tools([_tool()], "full").tools[0]["description"]
        assert "- limit: Maximum number of records to return" in description
        assert "- mode: Search strategy to use" in description
        assert "- query:" not in description  # redundant, dropped

    def test_original_description_is_preserved_first(self) -> None:
        description = compile_tools([_tool()], "full").tools[0]["description"]
        assert description.startswith("Search the index for matching records.")

    def test_beats_safe_mode_on_a_realistic_catalog(self) -> None:
        tools = [_tool(f"tool_{index}") for index in range(10)]
        safe = compile_tools(tools, "safe")
        full = compile_tools(tools, "full")
        assert full.after_bytes < safe.after_bytes < safe.before_bytes


class TestFullModeRefusals:
    def test_strict_tools_fall_back_to_safe(self) -> None:
        result = compile_tools([_tool(strict=True)], "full")
        # Enum and default must survive: the caller asked the provider to
        # enforce this shape.
        schema = _schema(result.tools[0])
        assert schema["properties"]["mode"]["enum"] == ["fast", "exact"]
        assert "params:" not in result.tools[0]["description"]

    def test_closed_schemas_fall_back_to_safe(self) -> None:
        tool = _tool()
        tool["input_schema"]["additionalProperties"] = False
        result = compile_tools([tool], "full")
        assert "params:" not in result.tools[0]["description"]

    def test_refs_disqualify_full_mode(self) -> None:
        tool = _tool()
        tool["input_schema"]["$defs"] = {"Thing": {"type": "object"}}
        result = compile_tools([tool], "full")
        assert "params:" not in result.tools[0]["description"]

    def test_unknown_property_keyword_disqualifies_full_mode(self) -> None:
        tool = _tool()
        tool["input_schema"]["properties"]["query"]["contentEncoding"] = "base64"
        result = compile_tools([tool], "full")
        assert "params:" not in result.tools[0]["description"]

    def test_deep_nesting_is_left_alone(self) -> None:
        tool = {
            "name": "deep",
            "description": "A deeply nested tool used for configuration purposes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "object",
                        "properties": {
                            "b": {
                                "type": "object",
                                "properties": {"c": {"type": "string"}},
                            }
                        },
                    }
                },
            },
        }
        result = compile_tools([tool], "full")
        assert not result.modified


class TestTypeRendering:
    @staticmethod
    def _probe(name: str, schema: dict[str, Any]) -> dict[str, Any]:
        """A tool big enough that compilation always beats the size guard.

        The padding matters: a two-parameter schema has so little JSON to
        absorb that the ``would grow`` guard discards the compilation and the
        assertion would be testing the guard rather than the renderer.
        """
        properties: dict[str, Any] = {name: schema}
        for index in range(6):
            properties[f"pad_{index}"] = {
                "type": "string",
                "description": "Padding parameter carrying a description of its own.",
            }
        return {
            "name": "t",
            "description": "A tool with one interesting parameter to render.",
            "input_schema": {"type": "object", "properties": properties},
        }

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            ({"type": "string"}, "query:str?"),
            ({"type": "integer"}, "query:int?"),
            ({"type": "boolean"}, "query:bool?"),
            ({"type": "array", "items": {"type": "string"}}, "query:str[]?"),
            ({"type": ["string", "null"]}, "query:str??"),
            ({"enum": ["a", "b"]}, "query:{a|b}?"),
            ({"const": 7}, "query:{7}?"),
            ({}, "query:any?"),
        ],
    )
    def test_rendering(self, schema: dict[str, Any], expected: str) -> None:
        result = compile_tools([self._probe("query", schema)], "full")
        assert result.modified
        assert expected in result.tools[0]["description"]

    def test_values_with_separators_stay_quoted(self) -> None:
        result = compile_tools([self._probe("sep", {"enum": ["a,b", "c"]})], "full")
        assert '{"a,b"|c}' in result.tools[0]["description"]


class TestWireShapes:
    def test_openai_responses_flat_parameters(self) -> None:
        tool = {
            "name": "search",
            "description": "Search the index for matching records.",
            "parameters": _tool()["input_schema"],
        }
        result = compile_tools([tool], "full")
        assert result.modified
        assert "params:" in result.tools[0]["description"]
        assert result.tools[0]["parameters"]["properties"] == {
            "query": {},
            "limit": {},
            "mode": {},
        }

    def test_openai_chat_nested_function(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the index for matching records.",
                "parameters": _tool()["input_schema"],
            },
        }
        result = compile_tools([tool], "full")
        assert result.modified
        assert "params:" in result.tools[0]["function"]["description"]

    def test_chat_strict_flag_is_honoured(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "strict": True,
                "description": "Search the index for matching records.",
                "parameters": _tool()["input_schema"],
            },
        }
        result = compile_tools([tool], "full")
        assert "params:" not in result.tools[0]["function"]["description"]


class TestDeterminism:
    def test_same_input_compiles_to_identical_bytes(self) -> None:
        tools = [_tool(f"tool_{index}") for index in range(5)]
        first = json.dumps(compile_tools(tools, "full").tools, sort_keys=True)
        reset_tool_schema_compilation()
        second = json.dumps(compile_tools(tools, "full").tools, sort_keys=True)
        assert first == second

    def test_input_is_never_mutated(self) -> None:
        tools = [_tool()]
        snapshot = json.dumps(tools, sort_keys=True)
        compile_tools(tools, "full")
        assert json.dumps(tools, sort_keys=True) == snapshot

    def test_cache_returns_an_equal_result(self) -> None:
        tools = [_tool(f"tool_{index}") for index in range(5)]
        first = compile_tools(tools, "full")
        second = compile_tools(tools, "full")
        assert second.modified
        assert second.after_bytes == first.after_bytes
        assert json.dumps(second.tools, sort_keys=True) == json.dumps(
            first.tools, sort_keys=True
        )
