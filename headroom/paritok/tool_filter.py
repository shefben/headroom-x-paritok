"""Semantic top-k tool-schema selection.

Headroom already compacts tool schemas lexically
(:mod:`headroom.proxy.tool_schema_compaction`): it drops JSON Schema annotation
keys, normalises whitespace and optionally truncates descriptions — across
*every* schema. What it never does is decide which tools are worth sending at
all. With 100+ MCP tools attached, most of the tool budget is spent on schemas
the model will not call this turn.

This module adds that missing layer. It embeds the user's query and each tool's
name+description, keeps the most relevant schemas in full, and drops or stubs
the rest. It runs *before* compaction, so the two compose: select what matters,
then shrink what survived.

**Cache safety.** The selection is frozen per session after the first turn
(:class:`SessionFrozenSelector`). A selection that churned turn to turn would
rewrite the ``tools`` array on every request and invalidate the provider's
prefix cache — costing more than the schemas saved. Freezing also makes the
filter immune to MCP async-load jitter, where the tool pool grows over the first
few turns as servers finish connecting.

**Recovery.** Dropped tools are not lost. When the agent says in plain text that
it lacks a capability, :func:`looks_like_missing_tool_help` detects it and
:func:`recover_tools_from_help` recalls the matching tools so they can be pinned
into the session and the turn retried.

Ported from ``paritok/tool_topk.py``.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_K = 8
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Claude Code's coding head. When the query looks code-related these are pinned,
# because a coding turn nearly always needs them and a purely semantic ranking
# can rank a niche MCP tool above Read/Grep on an ambiguous phrasing.
WHITELIST = ("Read", "Grep", "Glob", "Bash", "Edit", "Write")

# An agent's core execution tools — its only means to actually *do* anything.
# These are pinned unconditionally, not just on code-ish queries: agents like
# Codex expose only a handful of tools, and dropping `shell` or `apply_patch`
# leaves the agent unable to act at all, whatever the query looked like.
CORE_EXEC_TOOLS = frozenset(
    {
        "shell",
        "shell_command",
        "local_shell",
        "container.exec",
        "exec",
        "bash",
        "apply_patch",
        "run_terminal_cmd",
        "str_replace_editor",
        "editor",
    }
)

_CODE_HINT = re.compile(
    r"(\bbugs?\b|\bcode\b|\bfiles?\b|\bfunctions?\b|\brefactor|\btests?\b|\bbuild\b|"
    r"\bimports?\b|\bgrep\b|\brepo\b|\bmodules?\b|\bgit\b|\brun\b|\bfix|\bedit|\brename|"
    r"\bcodebase\b|\bdeprecated\b|\bcompile|\bclass\b|\bmethods?\b|\bvariables?\b|"
    r"\bstack trace\b|\bimplementation\b|\bparser\b|\bcrash|\bserver\b|"
    r"\w+\(\)|[a-z]+_[a-z]+|\.(py|js|ts|go|md|tsx|java|cpp|rs|json|yaml))",
    re.I,
)

# Generic/noisy tools that hijack "send/message/share" queries during recovery.
_GENERIC = frozenset(
    {
        "SendMessage",
        "PushNotification",
        "Monitor",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "Skill",
        "ReportFindings",
    }
)

# On hitting a dropped tool mid-execution the agent plainly says it lacks it.
_HELP_HINT = re.compile(
    r"((do(n'?| no)t|does(n'?| no)t) have\b.{0,40}\b(tool|connector|integration|access|ability)"
    r"|no\b.{0,30}\b(tool|connector|integration)\b.{0,20}\bavailable"
    r"|not available in this (session|environment)"
    r"|is(n'?t| not) available"
    r"|can'?t (complete|do that|help with that|look (this|that) up))",
    re.I,
)


class EmbeddingsUnavailable(RuntimeError):
    """The fastembed / numpy embedding backend is not installed."""


def _name_words(name: str) -> str:
    """Split a tool name into words: snake_case, camelCase, dotted, mcp__srv__act."""
    text = name.replace("mcp__", "").replace("__", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[._]+", " ", text)
    return text.strip()


def _expand(query: str) -> str:
    """Add coding context to a code-ish query so code tools rank above chat tools."""
    if _CODE_HINT.search(query or ""):
        return (query or "") + (
            " (look at code, read and search source files, run commands, fix bugs)"
        )
    return query or ""


@functools.lru_cache(maxsize=1)
def _model() -> Any:
    """Load the shared bge-small embedder.

    Uses ``fastembed``, not sentence-transformers as Paritok upstream does.
    Headroom moved its own embedding path off sentence-transformers in Stage
    3c.1 to drop the torch dependency and to match the Rust SmartCrusher
    byte-for-byte; both call the same ONNX file for the same model, so quality
    is unchanged while the install stays light.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingsUnavailable(
            "Paritok tool filtering needs an embedding backend. "
            'Install it with: pip install "headroom-ai[paritok]"'
        ) from exc

    from headroom.relevance.embedding import _pinned_revision

    revision = _pinned_revision(EMBEDDING_MODEL)
    if revision:
        return TextEmbedding(model_name=EMBEDDING_MODEL, revision=revision)
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def _encode(texts: list[str]) -> Any:
    """Embed ``texts`` into an L2-normalised matrix.

    Normalising here means a dot product is cosine similarity, which is what
    the ranking code assumes.
    """
    import numpy as np

    vectors = np.array(list(_model().embed(texts)))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embeddings_available() -> bool:
    """Whether the embedding model can be loaded (used by ``headroom doctor``)."""
    try:
        _model()
    except Exception:  # noqa: BLE001 - a download or import failure both mean "no"
        return False
    return True


def tool_name(tool: dict) -> str:
    """Tool name across Anthropic (``name``) and OpenAI (``function.name``) shapes."""
    return tool.get("name") or (tool.get("function") or {}).get("name") or ""


def tool_description(tool: dict) -> str:
    return tool.get("description") or (tool.get("function") or {}).get("description") or ""


class TopKToolSelector:
    """Ranks tools against a query, caching tool vectors by the tool set."""

    def __init__(self, k: int = DEFAULT_K) -> None:
        self.k = k
        self._cache: dict[int, tuple[list[str], Any]] = {}

    def _encode_tools(self, tools: list[dict]) -> tuple[list[str], Any]:
        names = [tool_name(t) for t in tools]
        key = hash(tuple(names))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        texts = [f"{_name_words(n)}. {tool_description(t)}" for n, t in zip(names, tools)]
        matrix = _encode(texts)
        self._cache[key] = (names, matrix)
        return names, matrix

    def select_dynamic(
        self,
        user_message: str,
        tools: list[dict],
        alpha: float = 0.9,
        k_min: int = 5,
        k_max: int = 20,
    ) -> list[str]:
        """Keep tools scoring >= ``alpha * best``, clamped to ``[k_min, k_max]``.

        Dynamic width beats a fixed k: a focused query keeps ~5 tools while a
        broad one keeps more, instead of always paying for a fixed budget.
        Returns names in rank order — callers depend on that ordering.
        """
        import numpy as np

        if len(tools) <= k_min:
            return [tool_name(t) for t in tools]

        names, matrix = self._encode_tools(tools)
        present = set(names)
        is_code = bool(_CODE_HINT.search(user_message or ""))
        query_vector = _encode([_expand(user_message)])[0]
        similarities = matrix @ query_vector
        order = np.argsort(-similarities)
        best = float(similarities[order[0]])

        # Core exec tools are pinned first and unconditionally; the coding head
        # is pinned only when the query looks code-related.
        picked = sorted(present & CORE_EXEC_TOOLS)
        if is_code:
            picked.extend(name for name in WHITELIST if name in present and name not in picked)
        for index in order:
            score = float(similarities[index])
            if score < alpha * best and len(picked) >= k_min:
                break
            if names[index] not in picked:
                picked.append(names[index])
            if len(picked) >= k_max:
                break
        # Top up to k_min if the alpha cut was aggressive.
        for index in order:
            if len(picked) >= k_min:
                break
            if names[index] not in picked:
                picked.append(names[index])
        return picked[:k_max]


class SessionFrozenSelector:
    """Selects tools once per session, then freezes the choice.

    Freezing is what makes this safe to run on every request: a stable
    ``tools`` array keeps the provider's prefix cache warm, and the selection
    stops reacting to MCP servers that finish loading mid-session. Recovered
    tools can be pinned later via :meth:`add_to_frozen`.
    """

    def __init__(self, alpha: float = 0.9, k_min: int = 5, k_max: int = DEFAULT_K) -> None:
        self.alpha = alpha
        self.k_min = k_min
        self.k_max = k_max
        self._selector = TopKToolSelector()
        self._frozen: dict[str, list[str]] = {}

    def select(self, session_id: str, user_message: str, tools: list[dict]) -> list[str]:
        present = {tool_name(t) for t in tools}
        frozen = self._frozen.get(session_id)
        if frozen is not None:
            kept = [name for name in frozen if name in present]
            if kept:
                return kept
        chosen = self._selector.select_dynamic(
            user_message, tools, self.alpha, self.k_min, self.k_max
        )
        if len(tools) > self.k_min:
            self._frozen[session_id] = chosen
        return chosen

    def frozen_for(self, session_id: str) -> list[str]:
        """The session's current selection, or empty if it has none yet.

        Empty is meaningful to callers: no frozen selection means no tool has
        been withheld from this session, so there is nothing to recover.
        """
        return list(self._frozen.get(session_id, ()))

    def add_to_frozen(self, session_id: str, names: Iterable[str]) -> None:
        """Pin recovered tools into the session so a miss never repeats."""
        current = self._frozen.setdefault(session_id, [])
        for name in names:
            if name not in current:
                current.append(name)

    def forget(self, session_id: str) -> None:
        self._frozen.pop(session_id, None)


def _is_mcp(name: str) -> bool:
    return name.startswith("mcp__")


def _make_stub(tool: dict, wire: str) -> dict:
    """A name-and-hint-only placeholder that costs a fraction of the full schema.

    The description states plainly that the schema is withheld and tells the
    model to say so. That wording is load-bearing: recovery works by detecting
    the model's own "I don't have that tool" reply
    (:func:`looks_like_missing_tool_help`) and re-injecting the real schema.
    Promising a schema-loading tool call instead would be a dead end — no such
    tool is injected, and the model would burn a turn trying to call it.

    ``wire`` selects the schema shape, and the three are genuinely different:
    ``anthropic`` nests under ``input_schema``, OpenAI's Responses API is flat
    with ``parameters``, and OpenAI chat-completions wraps the whole thing in a
    ``function`` object. Emitting the wrong one makes the provider reject the
    request outright, so a stub must match the wire it is going out on.
    """
    name = tool_name(tool)
    description = (
        "[deferred] "
        + tool_description(tool)[:48]
        + " — full schema withheld; say you need this tool and it will be restored."
    )
    empty_params = {"type": "object", "properties": {}}

    if wire == "openai_chat":
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": empty_params},
        }
    if wire == "openai":
        return {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": empty_params,
        }
    return {"name": name, "description": description, "input_schema": empty_params}


def _mcp_signal_score(keep_ordered: list[str]) -> float:
    """Rank-weighted MCP signal.

    A top-ranked MCP tool strongly implies an MCP task; a low-ranked one is
    usually a false "search/find" recall from a coding query. Weighting by rank
    rather than counting matches cuts false positives sharply (measured on 100
    cases: threshold 1.0 gives 4 false negatives / 11 false positives, versus
    0 / 33 for a plain "any MCP tool present" rule).
    """
    score = 0.0
    for rank, name in enumerate(keep_ordered):
        if _is_mcp(name):
            score += 3.0 if rank < 3 else (1.0 if rank < 6 else 0.3)
    return score


def apply_selection_adaptive(
    tools: list[dict],
    keep_ordered: Iterable[str],
    wire: str = "anthropic",
    mcp_signal_threshold: float = 1.0,
) -> list[dict]:
    """Keep selected tools in full; drop or stub the rest.

    Unselected *standard* tools are dropped outright — the model knows them from
    training and will still call them. Unselected *MCP* tools are stubbed only
    when the selection suggests an MCP-flavoured task, so a pure coding turn
    pays nothing for MCP servers it will never touch.

    ``keep_ordered`` must be in rank order; the MCP signal depends on it.
    """
    keep_list = list(keep_ordered)
    keep_names = set(keep_list)
    stub_mcp = _mcp_signal_score(keep_list) >= mcp_signal_threshold

    output: list[dict] = []
    for tool in tools:
        name = tool_name(tool)
        # Belt and braces: a frozen selection made before an exec tool loaded
        # would otherwise drop it, leaving the agent unable to act.
        if name in keep_names or name in CORE_EXEC_TOOLS:
            output.append(tool)
        elif _is_mcp(name) and stub_mcp:
            output.append(_make_stub(tool, wire))
    return output


def looks_like_missing_tool_help(agent_text: str) -> bool:
    """True when the agent's reply signals it lacked a tool to finish the task."""
    return bool(_HELP_HINT.search(agent_text or ""))


def recover_tools_from_help(
    help_text: str,
    candidate_tools: Iterable[dict],
    k: int = DEFAULT_K,
    exclude_generic: bool = True,
) -> list[str]:
    """Rank dropped tools against the agent's own "I don't have X" wording.

    ``candidate_tools`` is the pool *not* currently in the request. Returns the
    names to inject before retrying the turn, best match first.
    """
    import numpy as np

    candidates = list(candidate_tools)
    if not candidates or not (help_text or "").strip():
        return []

    names = [tool_name(t) for t in candidates]
    texts = [f"{_name_words(n)}. {tool_description(t)}" for n, t in zip(names, candidates)]
    document_vectors = _encode(texts)
    query_vector = _encode([help_text])[0]
    order = np.argsort(-(document_vectors @ query_vector))

    recovered: list[str] = []
    for index in order:
        if exclude_generic and names[index] in _GENERIC:
            continue
        recovered.append(names[index])
        if len(recovered) >= k:
            break
    return recovered
