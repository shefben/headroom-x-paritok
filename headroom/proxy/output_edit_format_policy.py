"""Pure policy for edit-format steering — the output-token lever.

Every compressor in Headroom shrinks what goes *into* the model. Output tokens
bill several times higher than input on every major provider, and on a coding
agent the single largest avoidable output is a model re-emitting code it was
never asked to change: a whole-file ``Write`` where a three-line edit would do,
a fenced copy of a function pasted back "for context", a closing summary that
restates the diff it just produced.

Aider measured this directly. Swapping GPT-4 Turbo's edit format from
whole-file to unified diffs made it "3x less lazy" — same model, same task,
different instruction about *how* to express a change. The format the model is
told to answer in moves output volume more than any decoding parameter does.

Two modes, cumulative:

``minimal``
    Prefer targeted edits over rewrites; never re-emit unchanged code.

``strict``
    Everything in ``minimal``, plus: no post-edit recap, and code shown in
    prose is limited to the changed lines with minimal surrounding context.

The text must stay byte-stable across releases. It lands in the system prompt,
so any edit to these strings invalidates the provider prefix cache for every
conversation that carries the block — treat a wording change as a cache-busting
release, exactly as :mod:`headroom.proxy.output_verbosity_policy` does.

Gating: the block is only worth its own tokens when the request actually
exposes file-editing tools. :func:`request_has_edit_tools` is that check —
steering an agent with no way to edit a file costs bytes and changes nothing.
"""

from __future__ import annotations

from typing import Any, Literal, cast

# Own sentinel, separate from the verbosity block's, so the two levers can be
# applied, replaced and detected independently.
EDIT_FORMAT_SENTINEL = "<headroom_edit_format>"
EDIT_FORMAT_SUFFIX = "</headroom_edit_format>"

EDIT_FORMAT_MODE_ENV = "HEADROOM_EDIT_FORMAT"
EditFormatMode = Literal["off", "minimal", "strict"]
EDIT_FORMAT_MODE_DEFAULT: EditFormatMode = "off"

EDIT_FORMAT_MODES: dict[str, str] = {
    "minimal": (
        "When changing a file, use the smallest targeted edit that does the "
        "job: edit the specific lines rather than rewriting the file. Reserve "
        "whole-file writes for files you are creating. Never reproduce code "
        "you are not changing — no re-emitting a function to show one changed "
        "line, and no pasting a file back after reading or editing it."
    ),
    "strict": (
        "When changing a file, use the smallest targeted edit that does the "
        "job: edit the specific lines rather than rewriting the file. Reserve "
        "whole-file writes for files you are creating. Never reproduce code "
        "you are not changing — no re-emitting a function to show one changed "
        "line, and no pasting a file back after reading or editing it. When "
        "you show code in prose, show only the changed lines plus at most "
        "three lines of context. After an edit succeeds, do not restate what "
        "the edit did; the diff is already in the conversation."
    ),
}

# Tool names that mean "this agent can modify files". Lowercased comparison:
# Claude Code sends ``Edit``/``Write`` where opencode sends ``edit``/``write``,
# and the Anthropic text-editor tool arrives as a ``type`` rather than a name.
_EDIT_TOOL_NAMES = frozenset(
    {
        "edit",
        "multiedit",
        "write",
        "create_file",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "apply_patch",
        "applypatch",
        "notebookedit",
    }
)
_EDIT_TOOL_TYPE_PREFIXES = ("text_editor_", "str_replace_")


def resolve_edit_format_mode(raw: str | None) -> EditFormatMode:
    """Resolve the steering mode from an environment value.

    Unknown values resolve to ``off`` rather than raising: this lever is an
    optimization, and a typo in an env var must never fail a request.
    """
    normalized = (raw or "").strip().lower()
    if not normalized:
        return EDIT_FORMAT_MODE_DEFAULT
    if normalized in ("1", "true", "yes", "on", "enabled"):
        return "minimal"
    if normalized in ("0", "false", "no", "disabled"):
        return "off"
    if normalized in ("off", "minimal", "strict"):
        return cast(EditFormatMode, normalized)
    return EDIT_FORMAT_MODE_DEFAULT


def edit_format_text(mode: str) -> str | None:
    """The full steering block for a mode, or ``None`` when it is ``off``."""
    body = EDIT_FORMAT_MODES.get(mode)
    if body is None:
        return None
    return f"{EDIT_FORMAT_SENTINEL}\n{body}\n{EDIT_FORMAT_SUFFIX}"


def request_has_edit_tools(tools: Any) -> bool:
    """Whether the request exposes any tool that can modify a file.

    Handles all three wire shapes: Anthropic (``name`` / typed ``type``),
    OpenAI Responses (flat ``name``), and OpenAI chat-completions (nested
    under ``function``).
    """
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = str(tool.get("type", ""))
        if tool_type.startswith(_EDIT_TOOL_TYPE_PREFIXES):
            return True

        name = tool.get("name")
        if not isinstance(name, str):
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name.lower() in _EDIT_TOOL_NAMES:
            return True
    return False
