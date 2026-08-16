"""Tests for the ported Paritok segmentation helpers.

These rules were ported verbatim from Paritok because the model was trained
against their output. The tests pin the behaviour that matters for that
contract: every chunk stays inside the model's budget, and kind/level land on
the values the training distribution used.
"""

from __future__ import annotations

from headroom.paritok.chunking import (
    CHUNK_SIZE,
    count_tokens,
    deduplicate_definitions,
    split_into_chunks_structural,
)
from headroom.paritok.tagger import (
    assign_level,
    classify_kind_from_content,
    detect_stale_files,
    reclassify_tool_result,
)


class TestChunking:
    def test_short_code_is_one_chunk(self) -> None:
        text = "def alpha():\n    return 1\n"
        chunks = split_into_chunks_structural(text)
        assert len(chunks) == 1
        assert chunks[0][0] == text

    def test_splits_at_definition_boundaries(self) -> None:
        # Each function is large enough that they cannot share one chunk.
        body = "\n".join(f"    x{i} = {i}" for i in range(400))
        text = f"def alpha():\n{body}\n\ndef beta():\n{body}\n"
        chunks = split_into_chunks_structural(text)
        assert len(chunks) > 1

    def test_every_chunk_fits_the_model_budget(self) -> None:
        """The whole point of chunking: no SEG may overflow the context window."""
        body = "\n".join(f"    value_{i} = compute({i})" for i in range(2000))
        text = f"def huge():\n{body}\n"
        for chunk_text, _start, _end, _tokens in split_into_chunks_structural(text):
            assert count_tokens(chunk_text) <= CHUNK_SIZE

    def test_boundaryless_text_is_still_split(self) -> None:
        """Prose has no class/def boundaries; it must not become one giant SEG."""
        text = "\n".join(f"line {i} of a long log file with some content" for i in range(4000))
        chunks = split_into_chunks_structural(text)
        assert len(chunks) > 1
        for chunk_text, _start, _end, _tokens in chunks:
            assert count_tokens(chunk_text) <= CHUNK_SIZE

    def test_line_numbered_code_finds_boundaries(self) -> None:
        """Read-tool output is cat -n style; boundaries must survive the prefix."""
        text = "   1\tdef alpha():\n   2\t    return 1\n   3\tdef beta():\n   4\t    return 2\n"
        chunks = split_into_chunks_structural(text, chunk_size=5, max_single_block=5)
        assert len(chunks) > 1

    def test_chunks_cover_all_lines_in_order(self) -> None:
        body = "\n".join(f"    x{i} = {i}" for i in range(300))
        text = f"def alpha():\n{body}\n\ndef beta():\n{body}\n"
        chunks = split_into_chunks_structural(text)
        ends = [end for _t, _s, end, _tok in chunks]
        starts = [start for _t, start, _e, _tok in chunks]
        assert starts == sorted(starts)
        assert ends[-1] >= len(text.split("\n")) - 1

    def test_deduplicate_drops_repeated_definitions(self) -> None:
        text = "def alpha():\n    return 1\ndef alpha():\n    return 1\ndef beta():\n    return 2\n"
        result = deduplicate_definitions(text)
        assert result.count("def alpha():") == 1
        assert "def beta():" in result

    def test_deduplicate_keeps_distinct_definitions(self) -> None:
        text = "def alpha():\n    return 1\ndef beta():\n    return 2\n"
        assert deduplicate_definitions(text) == text


class TestTagger:
    def test_cat_n_output_is_file_read(self) -> None:
        content = "Here's the result of running `cat -n` on a file\n   1\timport os\n"
        assert classify_kind_from_content(content) == "file_read"

    def test_traceback_is_log_output(self) -> None:
        assert classify_kind_from_content("Traceback (most recent call last):\n  File x\n") == (
            "log_output"
        )

    def test_edit_confirmation_is_reclassified(self) -> None:
        content = "The file /tmp/a.py has been edited successfully"
        assert reclassify_tool_result("tool_result", content) == "file_edit_confirm"

    def test_directory_listing_detected(self) -> None:
        content = "Here's the files and directories up to 2 levels deep\n- a\n- b\n"
        assert classify_kind_from_content(content) == "directory_listing"

    def test_non_tool_result_kind_passes_through(self) -> None:
        assert reclassify_tool_result("file_read", "anything") == "file_read"

    def test_current_turn_is_protected_at_l0(self) -> None:
        level, reason = assign_level({"kind": "file_read", "is_current_turn": True}, 5, 10, set())
        assert level == "L0"
        assert reason == "is_current_turn"

    def test_protected_kinds_stay_l0(self) -> None:
        level, _reason = assign_level({"kind": "system_prompt"}, 0, 10, set())
        assert level == "L0"

    def test_stale_file_read_gets_most_aggressive_level(self) -> None:
        level, reason = assign_level({"kind": "file_read"}, 3, 20, {3})
        assert level == "L3"
        assert reason == "stale_file_read"

    def test_recent_tool_result_is_barely_compressed(self) -> None:
        level, _reason = assign_level({"kind": "file_read"}, 19, 20, set())
        assert level == "L0"

    def test_detect_stale_files_marks_superseded_reads(self) -> None:
        segments = [
            {"kind": "file_operation", "content": '{"path": "/src/a.py"}'},
            {"kind": "tool_result", "content": "old contents"},
            {"kind": "file_operation", "content": '{"path": "/src/a.py"}'},
        ]
        stale = detect_stale_files(segments)
        # The first access and its result are superseded by the later one.
        assert 0 in stale
        assert 1 in stale
        assert 2 not in stale

    def test_detect_stale_files_ignores_single_access(self) -> None:
        segments = [{"kind": "file_operation", "content": '{"path": "/src/a.py"}'}]
        assert detect_stale_files(segments) == set()
