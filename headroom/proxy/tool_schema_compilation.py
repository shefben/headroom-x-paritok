"""Deterministic tool-schema compilation (TSCG-style).

Headroom already has two independent levers on the tools array, and this is a
third that is orthogonal to both:

* :mod:`headroom.proxy.tool_schema_compaction` is **lexical** — it drops JSON
  Schema annotation keys and normalises description whitespace. It never
  changes how a surviving field is expressed.
* :mod:`headroom.paritok.tool_topk` is **selection** — it decides *which*
  schemas survive. It never changes a schema it keeps.
* This module is **notation**. The schema keeps its meaning and its identity;
  only the representation changes, from JSON-Schema-as-JSON to a dense
  signature line the model reads far more cheaply.

Why notation is its own axis: the TSCG paper (arXiv:2605.04107) decomposed its
own results to separate the effect of *formatting* from the effect of *raw
compression* and watched R² fall from 0.88 to 0.03 — i.e. almost none of the
accuracy gain was explained by the prompt simply being shorter. Reported
savings are 52-57% with a proven ≥51% floor on well-formed schemas, and a
compact model (Phi-4 14B) went from 0% to 84.4% tool-call accuracy at 20 tools.
Those are the author's own benchmark numbers; treat them as directional. What
is not directional is the shape of the win: a JSON Schema spends most of its
bytes on punctuation and key names that repeat once per property, and a
signature line spends none.

Two modes, and the difference between them is entirely about what is allowed
to leave the JSON:

``safe``
    The JSON schema is left semantically intact and only *redundant* property
    descriptions are dropped — the ones that restate a self-explanatory
    parameter name. No signature is emitted, because with the JSON still
    present a signature would state every type twice and come out larger.
    Nothing a provider validates against changes.

``full``
    The JSON is reduced to the minimum a provider needs to accept the tool —
    ``type``, ``properties`` with bare types, and ``required`` — and everything
    else (descriptions, enums, defaults, formats, nested structure) is
    expressed once, densely, in the compiled text. This is where the published
    savings come from, and it is opt-in for a reason: enum and default values
    stop being machine-readable to the provider, so any tool that asked for
    strict validation is skipped rather than compiled.

Both modes end in the same guard the rest of Headroom uses: if compilation did
not actually shrink the payload, the original is returned untouched. A schema
that is already terse cannot be made worse by turning this on.

Cache safety: compilation is a pure function of the tools array and the mode,
so the compiled bytes are identical on every turn of a session. The tools array
sits in the cached prefix, so that stability is the whole requirement — turning
the lever on costs one cache miss, and nothing after it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

__all__ = [
    "COMPILE_MODE_ENV",
    "CompilationResult",
    "CompileMode",
    "compile_tools",
    "resolve_compile_mode",
    "tool_schema_compilation_enabled",
]

# Rollout feature that gates the lever (see headroom.rollout.FEATURES).
FEATURE_TOOL_SCHEMA_COMPILATION = "tool_schema_compilation"

COMPILE_MODE_ENV = "HEADROOM_TOOL_SCHEMA_COMPILE"

CompileMode = Literal["off", "safe", "full"]
COMPILE_MODE_DEFAULT: CompileMode = "safe"
"""Default *when the feature is enabled*. The feature itself is off by default,
so a stock proxy compiles nothing."""

# Marker opening the compiled section inside a tool description. Deliberately
# lowercase ASCII with no punctuation a tokenizer would split: this string is
# repeated once per tool, so its own cost is multiplied by the catalog size.
_SIGNATURE_PREFIX = "params: "
_PARAM_DOC_PREFIX = "- "

# JSON Schema type names to their compiled short form. A short form is only
# worth it when it survives as one token in practice; these all do.
_TYPE_NAMES: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "num",
    "boolean": "bool",
    "object": "obj",
    "array": "arr",
    "null": "null",
}

# Keys the ``full`` mode is allowed to erase from a property schema because the
# compiled signature carries them. Anything NOT in this set is a key we do not
# understand well enough to drop, and its presence disqualifies ``full`` mode
# for that tool (see :func:`_is_compilable`).
_ABSORBED_PROPERTY_KEYS = frozenset(
    {
        "default",
        "description",
        "enum",
        "const",
        "format",
        "items",
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "pattern",
        "properties",
        "required",
        "type",
    }
)

# Schema constructs whose meaning cannot be rendered as a flat signature. A
# tool using any of them is left alone in ``full`` mode rather than compiled
# into something that quietly means less than the original.
_UNCOMPILABLE_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "allOf",
        "anyOf",
        "definitions",
        "not",
        "oneOf",
        "patternProperties",
    }
)

# Maximum nesting depth flattened into dotted paths. Beyond this the structure
# stops being readable as a signature, so the tool falls back to ``safe``.
_MAX_FLATTEN_DEPTH = 2

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}

_cache_lock = threading.Lock()
_cache: dict[str, tuple[list[Any], int, int, int]] = {}
_CACHE_MAX_ENTRIES = 8

_enabled: bool | None = None


def tool_schema_compilation_enabled() -> bool:
    """Whether the rollout feature is live (resolved once per process)."""
    global _enabled

    if _enabled is None:
        from headroom.rollout import feature_enabled

        _enabled = feature_enabled(FEATURE_TOOL_SCHEMA_COMPILATION)
    return _enabled


def reset_tool_schema_compilation() -> None:
    """Drop the cached rollout resolution and the compilation cache."""
    global _enabled

    _enabled = None
    with _cache_lock:
        _cache.clear()


def resolve_compile_mode(raw: str | None) -> CompileMode:
    """Resolve the compilation mode from an environment value.

    An unrecognised value resolves to ``off``, never an exception: this lever
    is an optimization and a typo in an env var must not fail a request. An
    empty value means "the feature decides", which is why it resolves to the
    default mode rather than to ``off``.
    """
    normalized = (raw or "").strip().lower()
    if not normalized:
        return COMPILE_MODE_DEFAULT
    if normalized in _TRUE_VALUES:
        return COMPILE_MODE_DEFAULT
    if normalized in _FALSE_VALUES:
        return "off"
    if normalized in ("off", "safe", "full"):
        return cast(CompileMode, normalized)
    logger.warning("Invalid %s=%r; compilation disabled", COMPILE_MODE_ENV, raw)
    return "off"


@dataclass(frozen=True)
class CompilationResult:
    """Outcome of one :func:`compile_tools` call."""

    tools: list[Any]
    modified: bool
    before_bytes: int
    after_bytes: int
    compiled_count: int = 0
    skipped_count: int = 0


# ---------------------------------------------------------------------------
# Wire-shape access
# ---------------------------------------------------------------------------


def _schema_container(tool: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Locate the schema dict and its key across the three wire shapes.

    Anthropic nests under ``input_schema``, OpenAI Responses is flat with
    ``parameters``, OpenAI chat/completions wraps both in a ``function``
    object. Returning the *container* rather than the schema lets the caller
    replace it in place without re-deriving the shape.
    """
    for key in ("input_schema", "parameters"):
        if isinstance(tool.get(key), dict):
            return tool, key
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        return function, "parameters"
    return None


def _description_container(tool: dict[str, Any]) -> dict[str, Any]:
    """The dict that owns this tool's ``description`` field."""
    function = tool.get("function")
    if isinstance(function, dict) and "parameters" in function:
        return function
    return tool


def _tool_name(tool: dict[str, Any]) -> str:
    name = tool.get("name")
    if isinstance(name, str):
        return name
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def _is_strict(tool: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Whether the caller asked the provider to enforce this schema.

    ``full`` mode moves enums and defaults out of the JSON, so a tool whose
    contract is "the provider guarantees the shape" must never be compiled
    that way. Both the OpenAI ``strict`` flag and a schema that closes itself
    with ``additionalProperties: false`` count as that request.
    """
    if tool.get("strict") is True:
        return True
    function = tool.get("function")
    if isinstance(function, dict) and function.get("strict") is True:
        return True
    return schema.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# Type rendering
# ---------------------------------------------------------------------------


def _render_type(schema: Any, depth: int = 0) -> str:
    """Compact type notation for one property schema.

    ``str`` / ``int[]`` / ``{fast|slow}`` / ``str?``. Unknown or absent types
    render as ``any`` rather than being omitted — a parameter with no stated
    type is a real thing in the wild, and silently dropping it from the
    signature would make the signature a liar about the parameter list.
    """
    if not isinstance(schema, dict):
        return "any"

    const = schema.get("const")
    if const is not None:
        return "{" + _render_literal(const) + "}"

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        rendered = "|".join(_render_literal(value) for value in enum)
        return "{" + rendered + "}"

    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        # ``["string", "null"]`` is the idiomatic nullable spelling; render it
        # as a suffix rather than a union so the common case stays one token.
        names = [name for name in raw_type if isinstance(name, str)]
        non_null = [name for name in names if name != "null"]
        if len(non_null) == 1 and len(names) != len(non_null):
            return _render_type({**schema, "type": non_null[0]}, depth) + "?"
        if not non_null:
            return "null"
        return "|".join(_TYPE_NAMES.get(name, name) for name in non_null)

    if raw_type == "array":
        items = schema.get("items")
        if isinstance(items, dict) and depth < _MAX_FLATTEN_DEPTH:
            return _render_type(items, depth + 1) + "[]"
        return "arr"

    if isinstance(raw_type, str):
        return _TYPE_NAMES.get(raw_type, raw_type)
    return "any"


def _render_literal(value: Any) -> str:
    """Render an enum/default literal without JSON quoting noise."""
    if isinstance(value, str):
        # A value containing the separators would make the signature
        # ambiguous, so those keep their quotes.
        if any(char in value for char in "|{}, "):
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Param:
    """One flattened parameter, ready to render."""

    path: str
    type_text: str
    required: bool
    default: Any
    has_default: bool
    description: str


def _flatten(
    schema: dict[str, Any],
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[_Param] | None:
    """Flatten a schema's properties into dotted-path parameters.

    Returns ``None`` when the schema nests deeper than
    :data:`_MAX_FLATTEN_DEPTH`; the caller treats that as "not compilable",
    because a signature that silently omits a nested field is worse than no
    signature.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []

    required_names = schema.get("required")
    required = set(required_names) if isinstance(required_names, list) else set()

    params: list[_Param] = []
    for name, child in properties.items():
        if not isinstance(name, str):
            continue
        path = f"{prefix}{name}"
        if not isinstance(child, dict):
            params.append(
                _Param(path, "any", name in required, None, False, "")
            )
            continue

        nested = child.get("properties")
        if isinstance(nested, dict) and nested:
            if depth + 1 >= _MAX_FLATTEN_DEPTH:
                return None
            inner = _flatten(child, prefix=f"{path}.", depth=depth + 1)
            if inner is None:
                return None
            params.extend(inner)
            continue

        description = child.get("description")
        params.append(
            _Param(
                path=path,
                type_text=_render_type(child),
                required=name in required,
                default=child.get("default"),
                has_default="default" in child,
                description=" ".join(description.split())
                if isinstance(description, str)
                else "",
            )
        )
    return params


def _render_signature(params: list[_Param]) -> str:
    """One line naming every parameter with its type and default."""
    parts: list[str] = []
    for param in params:
        text = f"{param.path}:{param.type_text}"
        if param.has_default:
            text += f"={_render_literal(param.default)}"
        elif not param.required:
            # Optional-with-no-default is the single most consequential fact a
            # bare JSON schema hides behind an absence, so it gets a mark.
            text += "?"
        parts.append(text)
    return _SIGNATURE_PREFIX + " ".join(parts)


def _is_redundant_description(name: str, description: str) -> bool:
    """Whether a property description tells the model nothing new.

    Two cases, both common in generated schemas: the description restates the
    parameter name ("The query" for ``query``), or the name is one of the
    self-explanatory ones Headroom already recognises for Layer 3 of the
    lexical compactor and the description adds no clause to it.
    """
    if not description:
        return True
    from headroom.proxy.tool_schema_compaction import _is_semantic_param_name

    leaf = name.rsplit(".", 1)[-1]
    normalized = description.strip().rstrip(".").lower()
    spelled = leaf.replace("_", " ").replace("-", " ").lower()
    for article in ("the ", "a ", "an ", ""):
        if normalized == f"{article}{spelled}":
            return True
    return _is_semantic_param_name(leaf) and len(normalized) <= len(spelled) + 12


def _render_param_docs(params: list[_Param]) -> list[str]:
    """One line per parameter that still has something to say."""
    lines: list[str] = []
    for param in params:
        if _is_redundant_description(param.path, param.description):
            continue
        lines.append(f"{_PARAM_DOC_PREFIX}{param.path}: {param.description}")
    return lines


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _is_compilable(schema: dict[str, Any]) -> bool:
    """Whether ``full`` mode may rewrite this schema's JSON.

    Conservative by construction: any key this module does not know how to
    absorb into the signature disqualifies the tool. A new JSON Schema keyword
    appearing in the wild therefore degrades to ``safe`` behaviour instead of
    being silently discarded.
    """
    if schema.get("type") not in (None, "object"):
        return False
    if _UNCOMPILABLE_KEYS & set(schema.keys()):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return False
    for child in properties.values():
        if not isinstance(child, dict):
            continue
        if _UNCOMPILABLE_KEYS & set(child.keys()):
            return False
        if set(child.keys()) - _ABSORBED_PROPERTY_KEYS:
            return False
    return True


def _minimal_schema(schema: dict[str, Any], params: list[_Param]) -> dict[str, Any]:
    """The smallest schema a provider still accepts for these parameters.

    Only the top-level property *names* survive, each mapped to an empty
    schema. Types, enums, defaults and formats are all in the signature by this
    point, and repeating ``{"type": "string"}`` per property is most of what a
    JSON schema costs — dropping it is where the published savings come from.
    The names themselves stay so a provider validating "is this a known
    parameter" still has something to validate against, and so the tools array
    remains readable to anything downstream that inspects it.

    ``required`` is kept in the JSON rather than being left to the signature's
    ``?`` marks: it is a handful of tokens and it is the one constraint
    providers act on when they validate a tool call at all.
    """
    properties = schema.get("properties")
    minimal_properties: dict[str, Any] = {}
    if isinstance(properties, dict):
        for name in properties:
            if isinstance(name, str):
                minimal_properties[name] = {}

    compiled: dict[str, Any] = {"type": "object", "properties": minimal_properties}
    required = [param.path for param in params if param.required and "." not in param.path]
    if required:
        compiled["required"] = required
    return compiled


def _strip_redundant_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """``safe`` mode's only schema edit: drop descriptions that say nothing."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema

    stripped_properties: dict[str, Any] = {}
    changed = False
    for name, child in properties.items():
        if not isinstance(child, dict) or not isinstance(name, str):
            stripped_properties[name] = child
            continue
        description = child.get("description")
        if isinstance(description, str) and _is_redundant_description(name, description):
            stripped_properties[name] = {
                key: value for key, value in child.items() if key != "description"
            }
            changed = True
            continue
        stripped_properties[name] = child

    if not changed:
        return schema
    return {**schema, "properties": stripped_properties}


def _compile_one(tool: Any, mode: CompileMode) -> tuple[Any, bool]:
    """Compile one tool entry; returns ``(tool, compiled)``.

    Never raises on a malformed entry — an unrecognised shape is returned
    untouched, because a tools array the proxy cannot parse is still a tools
    array the provider can.
    """
    if not isinstance(tool, dict):
        return tool, False

    located = _schema_container(tool)
    if located is None:
        return tool, False
    container, schema_key = located
    schema = container[schema_key]

    params = _flatten(schema)
    if params is None or not params:
        return tool, False

    use_full = mode == "full" and not _is_strict(tool, schema) and _is_compilable(schema)

    # The signature only earns its bytes when the JSON it restates is gone. In
    # ``safe`` mode the schema stays, so adding a signature would state every
    # type twice — measurably worse than doing nothing, which is why ``safe``
    # is description-stripping only.
    signature = _render_signature(params) if use_full else ""
    doc_lines = _render_param_docs(params) if use_full else []

    description_owner = _description_container(tool)
    original_description = description_owner.get("description")
    base = (
        " ".join(original_description.split())
        if isinstance(original_description, str)
        else ""
    )
    pieces = [piece for piece in (base, signature, *doc_lines) if piece]
    compiled_description = "\n".join(pieces)

    updated = copy.deepcopy(tool)
    updated_owner = _description_container(updated)
    updated_owner["description"] = compiled_description

    updated_located = _schema_container(updated)
    if updated_located is None:  # pragma: no cover - deepcopy preserves shape
        return tool, False
    updated_container, updated_key = updated_located

    if use_full:
        updated_container[updated_key] = _minimal_schema(schema, params)
    else:
        updated_container[updated_key] = _strip_redundant_descriptions(schema)

    return updated, True


def _json_byte_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")))


def _cache_key(tools: list[Any], mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(tools, sort_keys=True, default=str, separators=(",", ":")).encode()
    )
    digest.update(mode.encode())
    return digest.hexdigest()[:16]


def compile_tools(tools: Any, mode: CompileMode | str) -> CompilationResult:
    """Compile a tools array into signature notation.

    Returns the original array untouched — ``modified=False`` — whenever the
    mode is ``off``, the array is not a non-empty list, nothing was compilable,
    or the compiled form is not actually smaller. That last check is the one
    that makes this lever safe to enable blindly: a catalog of already-terse
    schemas cannot be made worse by it.
    """
    resolved = mode if mode in ("off", "safe", "full") else "off"
    if resolved == "off" or not isinstance(tools, list) or not tools:
        return CompilationResult(tools=tools, modified=False, before_bytes=0, after_bytes=0)

    key = _cache_key(tools, resolved)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        cached_tools, before, after, cached_count = cached
        if cached_count == 0 or after >= before:
            return CompilationResult(
                tools=tools,
                modified=False,
                before_bytes=before,
                after_bytes=after,
                compiled_count=cached_count,
                skipped_count=len(tools) - cached_count,
            )
        return CompilationResult(
            tools=cached_tools,
            modified=True,
            before_bytes=before,
            after_bytes=after,
            compiled_count=cached_count,
            skipped_count=len(tools) - cached_count,
        )

    compiled_tools: list[Any] = []
    compiled_count = 0
    skipped_count = 0
    for tool in tools:
        updated, compiled = _compile_one(tool, cast(CompileMode, resolved))
        compiled_tools.append(updated)
        if compiled:
            compiled_count += 1
        else:
            skipped_count += 1

    before = _json_byte_len(tools)
    after = _json_byte_len(compiled_tools)

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (compiled_tools, before, after, compiled_count)

    if compiled_count == 0 or after >= before:
        return CompilationResult(
            tools=tools,
            modified=False,
            before_bytes=before,
            after_bytes=after,
            compiled_count=compiled_count,
            skipped_count=skipped_count,
        )

    return CompilationResult(
        tools=compiled_tools,
        modified=True,
        before_bytes=before,
        after_bytes=after,
        compiled_count=compiled_count,
        skipped_count=skipped_count,
    )


def compile_tools_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, int, int]:
    """Payload-level wrapper matching ``compact_tools``'s call convention.

    Resolves the mode from the rollout feature plus
    ``HEADROOM_TOOL_SCHEMA_COMPILE`` so handlers do not each re-derive it.
    """
    if not tool_schema_compilation_enabled():
        return payload, False, 0, 0

    from headroom.proxy import runtime_env

    mode = resolve_compile_mode(runtime_env.getenv(COMPILE_MODE_ENV, ""))
    result = compile_tools(payload.get("tools"), mode)
    if not result.modified:
        return payload, False, result.before_bytes, result.after_bytes

    updated = dict(payload)
    updated["tools"] = result.tools
    return updated, True, result.before_bytes, result.after_bytes
