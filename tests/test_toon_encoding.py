"""Tests for TOON re-encoding of uniform JSON tool results."""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.tokenizer import Tokenizer
from headroom.transforms.toon_encoding import (
    ToonEncoder,
    ToonEncodingConfig,
    encode_toon,
    try_encode_json_text,
)


class _CharTokenizer:
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


CONFIG = ToonEncodingConfig(enabled=True)


def _rows(count: int = 8) -> list[dict[str, Any]]:
    return [
        {"id": index, "name": f"user{index}", "role": "admin", "active": True}
        for index in range(count)
    ]


def _transcript(payload: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "list the users"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "api", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": payload}
            ],
        },
    ]


class TestEncoding:
    def test_uniform_array(self) -> None:
        encoded = encode_toon(_rows(4), CONFIG)
        assert encoded is not None
        lines = encoded.splitlines()
        assert lines[0] == "[4]{id,name,role,active}:"
        assert lines[1] == "  0,user0,admin,true"

    def test_object_with_a_nested_table(self) -> None:
        encoded = encode_toon({"total": 8, "items": _rows(8)}, CONFIG)
        assert encoded is not None
        assert encoded.startswith("total: 8\n")
        assert "items[8]{id,name,role,active}:" in encoded

    def test_scalars_render_without_quotes(self) -> None:
        encoded = encode_toon(
            [{"a": 1, "b": 1.5}, {"a": None, "b": True}, {"a": 3, "b": False}, {"a": 4, "b": 0}],
            CONFIG,
        )
        assert encoded is not None
        assert "null,true" in encoded
        assert "3,false" in encoded

    def test_ambiguous_values_are_quoted(self) -> None:
        rows = [
            {"a": "x,y", "b": " padded "},
            {"a": "plain", "b": "ok"},
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]
        encoded = encode_toon(rows, CONFIG)
        assert encoded is not None
        assert '"x,y"," padded "' in encoded
        assert "plain,ok" in encoded

    def test_field_order_follows_the_producer(self) -> None:
        rows = [{"z": index, "a": index} for index in range(4)]
        encoded = encode_toon(rows, CONFIG)
        assert encoded is not None
        assert encoded.splitlines()[0] == "[4]{z,a}:"


class TestRefusals:
    def test_too_few_rows(self) -> None:
        assert encode_toon(_rows(3), CONFIG) is None

    def test_single_field_arrays(self) -> None:
        assert encode_toon([{"a": index} for index in range(8)], CONFIG) is None

    def test_ragged_rows(self) -> None:
        rows: list[dict[str, Any]] = _rows(6)
        del rows[3]["role"]
        assert encode_toon(rows, CONFIG) is None

    def test_nested_values(self) -> None:
        rows = [{"a": index, "b": {"c": index}} for index in range(6)]
        assert encode_toon(rows, CONFIG) is None

    def test_array_of_scalars(self) -> None:
        assert encode_toon([1, 2, 3, 4, 5], CONFIG) is None

    def test_object_of_pure_scalars(self) -> None:
        assert encode_toon({"a": 1, "b": 2}, CONFIG) is None

    def test_object_with_a_non_uniform_value(self) -> None:
        assert encode_toon({"items": [{"a": 1}, {"b": 2}]}, CONFIG) is None

    def test_empty_inputs(self) -> None:
        assert encode_toon([], CONFIG) is None
        assert encode_toon({}, CONFIG) is None
        assert encode_toon(None, CONFIG) is None


class TestTextEntry:
    def test_non_json_is_ignored(self) -> None:
        assert try_encode_json_text("just some log output", CONFIG) is None

    def test_malformed_json_is_ignored(self) -> None:
        assert try_encode_json_text('[{"a": 1', CONFIG) is None

    def test_it_never_returns_something_larger(self) -> None:
        """The size guard, asserted as a property rather than on one input.

        Contriving a qualifying shape where TOON loses to compact JSON is hard
        — for uniform tables it essentially always wins — so the guard is
        stated as the invariant it actually is.
        """
        shapes: list[Any] = [
            _rows(4),
            _rows(50),
            [{"a": "x,y", "b": "p,q"} for _ in range(4)],
            [{"k": "", "v": " "} for _ in range(6)],
            {"total": 4, "items": _rows(4)},
        ]
        for shape in shapes:
            for dump in (
                json.dumps(shape, separators=(",", ":")),
                json.dumps(shape, indent=2),
            ):
                encoded = try_encode_json_text(dump, CONFIG)
                if encoded is not None:
                    assert len(encoded) < len(dump.strip())

    def test_the_guard_discards_an_oversized_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "headroom.transforms.toon_encoding.encode_toon",
            lambda data, config: "x" * 10_000,
        )
        assert try_encode_json_text(json.dumps(_rows(8)), CONFIG) is None

    def test_pretty_json_is_a_large_win(self) -> None:
        raw = json.dumps(_rows(20), indent=2)
        encoded = try_encode_json_text(raw, CONFIG)
        assert encoded is not None
        assert len(encoded) < len(raw) / 2


class TestLossless:
    def test_every_cell_survives(self) -> None:
        rows = _rows(6)
        encoded = encode_toon(rows, CONFIG)
        assert encoded is not None
        body = encoded.splitlines()[1:]
        assert len(body) == len(rows)
        for row, line in zip(rows, body):
            assert str(row["id"]) in line
            assert row["name"] in line


class TestTransform:
    def test_encodes_a_tool_result(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(json.dumps(_rows(20), indent=2))
        result = ToonEncoder(CONFIG).apply(messages, tokenizer)
        assert result.transforms_applied == ["toon_encoding:api"]
        assert result.messages[2]["content"][0]["content"].startswith("[20]{")
        assert result.tokens_after < result.tokens_before

    def test_leaves_prose_alone(self, tokenizer: Tokenizer) -> None:
        messages = _transcript("no json here, just a wall of log lines\n" * 40)
        result = ToonEncoder(CONFIG).apply(messages, tokenizer)
        assert not result.transforms_applied

    def test_skips_ccr_markers(self, tokenizer: Tokenizer) -> None:
        messages = _transcript("<<ccr:aabbccddeeff112233>>")
        result = ToonEncoder(CONFIG).apply(messages, tokenizer)
        assert not result.transforms_applied

    def test_honours_the_frozen_prefix(self, tokenizer: Tokenizer) -> None:
        messages = _transcript(json.dumps(_rows(20), indent=2))
        result = ToonEncoder(CONFIG).apply(messages, tokenizer, frozen_message_count=3)
        assert not result.transforms_applied

    def test_encodes_the_newest_result_too(self, tokenizer: Tokenizer) -> None:
        """No protected tail: the verdict depends on the block alone.

        Every other tool-result stage has to leave the tail untouched because
        its decision reads other messages. This one does not, so refusing to
        encode the newest result would forgo the saving for no reason.
        """
        messages = _transcript(json.dumps(_rows(20), indent=2))
        result = ToonEncoder(CONFIG).apply(messages, tokenizer, protect_recent=8)
        assert result.transforms_applied

    def test_input_is_not_mutated(self, tokenizer: Tokenizer) -> None:
        raw = json.dumps(_rows(20), indent=2)
        messages = _transcript(raw)
        ToonEncoder(CONFIG).apply(messages, tokenizer)
        assert messages[2]["content"][0]["content"] == raw

    def test_openai_tool_message_shape(self, tokenizer: Tokenizer) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "list"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "api", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(_rows(20), indent=2),
            },
        ]
        result = ToonEncoder(CONFIG).apply(messages, tokenizer)
        assert result.transforms_applied == ["toon_encoding:api"]
        assert result.messages[2]["content"].startswith("[20]{")


class TestDeterminism:
    def test_same_block_encodes_identically_regardless_of_position(
        self, tokenizer: Tokenizer
    ) -> None:
        raw = json.dumps(_rows(20), indent=2)
        short = _transcript(raw)
        long = _transcript(raw) + [
            {"role": "assistant", "content": f"later turn {index}"} for index in range(20)
        ]
        first = ToonEncoder(CONFIG).apply(short, tokenizer).messages[2]
        second = ToonEncoder(CONFIG).apply(long, tokenizer).messages[2]
        assert first["content"][0]["content"] == second["content"][0]["content"]


class TestConfigFromEnv:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "HEADROOM_TOON_ENCODING",
            "HEADROOM_TOON_ENCODING_MIN_ROWS",
            "HEADROOM_TOON_ENCODING_MIN_FIELDS",
        ):
            monkeypatch.delenv(name, raising=False)
        config = ToonEncodingConfig.from_env()
        assert not config.enabled
        assert config.min_rows == 4
        assert config.min_fields == 2

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_TOON_ENCODING", "1")
        monkeypatch.setenv("HEADROOM_TOON_ENCODING_MIN_ROWS", "10")
        config = ToonEncodingConfig.from_env()
        assert config.enabled
        assert config.min_rows == 10
