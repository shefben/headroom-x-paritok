"""Tail reinjection of the user's current task.

Instruction adherence decays with distance. The LongIns benchmark isolates the
effect by running the same questions with instructions stated once at the top
versus repeated before each question, and the drop across context length is
steep — GPT-4o falls from roughly 76 to 51 between 256 and 16k tokens. R&R
(arXiv:2403.05004) attacks it directly by re-injecting an instruction reminder
at intervals through a long context.

The important detail, and the one that is easy to get backwards: **duplicate,
do not relocate.** A systematic study of prompt placement found that moving
instructions to the end was the *worst* of four configurations tested, while
head-and-tail together was among the best. So the user's original message stays
exactly where it is and a restatement is added at the tail. Nothing is moved.

Placement and the prefix cache
------------------------------
Every other steering block in Headroom goes in the system prompt, which works
because those blocks are byte-stable — the same text on every turn. This one is
not: it quotes the user's current task, which changes. Putting varying text in
the system prompt would invalidate the cached prefix of every conversation on
every turn, which would cost far more than the adherence is worth.

So this block goes at the **tail of the message array** instead, after every
``cache_control`` breakpoint. Bytes appended there are the bytes the provider
was going to reprocess anyway, so the reminder is free in cache terms.

The block never accumulates. The proxy rewrites the outbound body only; the
client builds its next request from its own history, which never contained the
injected block. There is nothing to strip.

Gating, all of which must hold:

* the transcript is actually long enough for decay to be real;
* the user's last actual instruction is far enough back to have decayed —
  if it is the most recent message, restating it at the tail is pure cost;
* the last message is a ``user`` turn, so appending is wire-legal.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "TASK_REMINDER_SENTINEL",
    "TaskReminderSettings",
    "apply_responses_task_reminder",
    "apply_task_reminder",
    "extract_current_task",
    "reset_task_reminder",
    "task_reminder_enabled",
]

TASK_REMINDER_SENTINEL = "<headroom_task_reminder>"
TASK_REMINDER_SUFFIX = "</headroom_task_reminder>"

# Fixed framing around the variable part. Kept short: this block is paid for on
# every turn it fires, and the model needs the task text, not the ceremony.
_PREAMBLE = "The user's current request, restated because it is now far above:"

FEATURE_TASK_REMINDER = "task_reminder"

ENV_ENABLED = "HEADROOM_TASK_REMINDER"
ENV_TRIGGER_TOKENS = "HEADROOM_TASK_REMINDER_TRIGGER_TOKENS"
ENV_MIN_DISTANCE = "HEADROOM_TASK_REMINDER_MIN_DISTANCE"
ENV_MAX_CHARS = "HEADROOM_TASK_REMINDER_MAX_CHARS"

DEFAULT_TRIGGER_TOKENS = 30_000
DEFAULT_MIN_DISTANCE = 12
DEFAULT_MAX_CHARS = 600

# Below this the "task" is an acknowledgement ("ok", "go on"), and restating it
# tells the model nothing it can act on.
_MIN_TASK_CHARS = 24

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

# Harness-injected user turns are not the user talking. Reinjecting one of these
# as "the user's current request" would actively mislead the model.
_SYNTHETIC_TASK_RE = re.compile(
    r"^\s*(?:<(?:system-reminder|command-name|local-command|budget|env)\b"
    r"|\[Request interrupted"
    r"|Caveat: The messages below)",
    re.IGNORECASE,
)

_enabled: bool | None = None


def task_reminder_enabled() -> bool:
    """Whether the proxy should reinject the current task."""
    global _enabled

    if _enabled is None:
        from headroom.rollout import feature_enabled

        _enabled = feature_enabled(FEATURE_TASK_REMINDER)
    return _enabled


def reset_task_reminder() -> None:
    """Drop the cached rollout resolution (tests, ``POST /admin/runtime-env``)."""
    global _enabled

    _enabled = None


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    return value if value >= minimum else default


@dataclass(frozen=True)
class TaskReminderSettings:
    """Resolved tuning for task reinjection."""

    enabled: bool = False
    trigger_tokens: int = DEFAULT_TRIGGER_TOKENS
    min_distance: int = DEFAULT_MIN_DISTANCE
    max_chars: int = DEFAULT_MAX_CHARS

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> TaskReminderSettings:
        return cls(
            enabled=_env_bool(ENV_ENABLED, False) if enabled is None else enabled,
            trigger_tokens=_env_int(
                ENV_TRIGGER_TOKENS, DEFAULT_TRIGGER_TOKENS, minimum=1
            ),
            min_distance=_env_int(ENV_MIN_DISTANCE, DEFAULT_MIN_DISTANCE, minimum=1),
            max_chars=_env_int(ENV_MAX_CHARS, DEFAULT_MAX_CHARS, minimum=32),
        )


def _user_text(message: Any) -> str:
    """The user's own words in an Anthropic message, excluding tool results."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in ("text", "input_text"):
            # ``input_text`` is the Responses spelling of the same thing; both
            # wire formats reach this helper.
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _clip(text: str, max_chars: int) -> str:
    """Trim a task to ``max_chars``, preferring a line boundary.

    A hard mid-word cut reads as corruption; cutting at the last newline keeps
    the restatement looking like something the user wrote.
    """
    collapsed = text.strip()
    if len(collapsed) <= max_chars:
        return collapsed
    head = collapsed[:max_chars]
    boundary = head.rfind("\n")
    if boundary >= max_chars // 2:
        head = head[:boundary]
    return head.rstrip() + " […]"


def extract_current_task(
    messages: Any,
    settings: TaskReminderSettings,
) -> tuple[str, int] | None:
    """The user's latest real instruction and how far back it is.

    Returns ``(task_text, distance_in_messages)`` or ``None`` when there is
    nothing worth restating: no user text at all, only synthetic harness turns,
    or a task too short to carry an instruction.
    """
    if not isinstance(messages, list) or not messages:
        return None
    for index in range(len(messages) - 1, -1, -1):
        text = _user_text(messages[index]).strip()
        if not text:
            continue
        if TASK_REMINDER_SENTINEL in text:
            # Our own block from an earlier stage in this same request.
            return None
        if _SYNTHETIC_TASK_RE.match(text):
            continue
        if len(text) < _MIN_TASK_CHARS:
            continue
        return _clip(text, settings.max_chars), len(messages) - 1 - index
    return None


def reminder_text(task: str) -> str:
    """The full block, sentinel included."""
    return f"{TASK_REMINDER_SENTINEL}\n{_PREAMBLE}\n{task}\n{TASK_REMINDER_SUFFIX}"


def _already_present(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return TASK_REMINDER_SENTINEL in content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                if TASK_REMINDER_SENTINEL in block["text"]:
                    return True
    return False


def apply_task_reminder(
    body: dict[str, Any],
    settings: TaskReminderSettings,
    *,
    input_tokens: int,
) -> str | None:
    """Append the reminder to the tail of an Anthropic message array.

    Returns the task text on success, ``None`` when any gate failed. Mutates
    ``body["messages"]`` in place, and only ever by appending to the final
    message — never by moving, rewriting or removing anything.
    """
    if not settings.enabled or input_tokens < settings.trigger_tokens:
        return None

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        # An assistant tail is a prefill, and an assistant turn that ends in a
        # tool_use block must keep ending in one. Nothing to do here.
        return None
    if _already_present(last):
        return None

    extracted = extract_current_task(messages, settings)
    if extracted is None:
        return None
    task, distance = extracted
    if distance < settings.min_distance:
        # The instruction is still near the tail; the model has not lost it and
        # restating it would be bytes spent to say what was just said.
        return None

    block = reminder_text(task)
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = f"{content.rstrip()}\n\n{block}" if content.strip() else block
        return task
    if isinstance(content, list):
        content.append({"type": "text", "text": block})
        return task
    return None


def apply_responses_task_reminder(
    body: dict[str, Any],
    settings: TaskReminderSettings,
    *,
    input_tokens: int,
) -> str | None:
    """The OpenAI Responses counterpart, appending an ``input`` item.

    Responses has no "last message must be a user turn" rule, so a fresh user
    item is appended rather than an existing one extended. That is also the
    only shape that cannot disturb a ``function_call``/``function_call_output``
    pairing already in the list.
    """
    if not settings.enabled or input_tokens < settings.trigger_tokens:
        return None

    items = body.get("input")
    if not isinstance(items, list) or not items:
        return None

    # Non-message items are blanked rather than dropped so that positions —
    # and therefore the measured distance — stay aligned with the real list. A
    # Responses transcript interleaves reasoning and tool-output items, and
    # those are exactly the volume that pushes the instruction out of reach.
    normalized: list[dict[str, Any]] = [
        item
        if isinstance(item, dict) and item.get("type") in (None, "message")
        else {}
        for item in items
    ]

    extracted = extract_current_task(normalized, settings)
    if extracted is None:
        return None
    task, distance = extracted
    if distance < settings.min_distance:
        return None

    block = reminder_text(task)
    items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": block}],
        }
    )
    return task
