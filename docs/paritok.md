# Paritok augmentation

Headroom bundles the three compression levers from
[Paritok](https://github.com/Paritok-official/paritok-4b-v1) so one proxy does
both jobs. Every lever is **off by default** and independently toggleable, and
each one *augments* Headroom's pipeline rather than replacing any part of it.

## What each lever adds

| Lever | What it does | Why Headroom didn't already do it |
|---|---|---|
| `paritok_tool_filter` | Picks the tool schemas worth sending; drops or stubs the rest | Headroom's `tool_schema_compaction` shrinks *every* schema lexically but never decides which ones matter |
| `paritok_content_compress` | Compresses tool output with the Paritok-4B model | Adds a model trained on coding-agent trajectories ahead of Kompress |
| `paritok_history_summarize` | Summarizes stale turns in place | Headroom retired its history stage (PR-B1); this shrinks turns without touching the message list |
| `paritok_chain_model` | Feeds Paritok output through Kompress too | Off by default — a second lossy pass costs latency for little gain |

## Enabling

Each lever is a standard Headroom rollout feature, so either form works:

```bash
# Legacy-style env aliases
PARITOK_TOOL_FILTER=1 PARITOK_CONTENT_COMPRESS=1 headroom proxy

# Or the canonical feature list
HEADROOM_FEATURES=paritok_tool_filter,paritok_content_compress headroom proxy
```

Any combination is valid. A disabled lever is not merely inert — its transform
is never constructed, so the request path is byte-identical to stock Headroom.

Verify what is actually live:

```bash
headroom doctor          # reports a `paritok` check whenever any lever is on
headroom rollout status  # shows every feature and how it was resolved
```

## Install

```bash
pip install "headroom-ai[paritok]"
```

That pulls only `fastembed` + `numpy`. The tool filter reuses the same
fastembed/bge-small backend as Headroom's `[relevance]` extra rather than
sentence-transformers, so it adds no torch dependency.

The Paritok-4B model is **not** a Python dependency — it is served over an
OpenAI-compatible endpoint:

```bash
ollama pull paritok/paritok-4b-v1
ollama cp paritok/paritok-4b-v1 paritok-4b-v1
```

Only `paritok_content_compress` and `paritok_history_summarize` need the model.
`paritok_tool_filter` is CPU-only and needs no server at all.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `PARITOK_BACKEND` | `ollama` | `ollama`, `vllm`, or `openai` |
| `PARITOK_ENDPOINT` | per backend | Override the base URL |
| `PARITOK_MODEL` | `paritok-4b-v1` | Model name to request |
| `PARITOK_API_KEY` | — | Sent as a bearer token |
| `PARITOK_LEVEL` | `L1` | SEG compression level `L0`–`L3` |
| `PARITOK_NUM_CTX` | `8192` | Model context window |
| `PARITOK_TOOL_TOPK` | `8` | Max tools kept at full schema |
| `PARITOK_TOOL_K_MIN` | `5` | Min tools kept; below this the filter is skipped |
| `PARITOK_TOOL_ALPHA` | `0.9` | Keep tools scoring ≥ `alpha × best` |
| `PARITOK_TOOL_RECOVER_K` | `2` | Tools restored per missing-tool reply |
| `PARITOK_MIN_TOKENS` | `250` | Skip content smaller than this |
| `PARITOK_MAX_CONCURRENCY` | `4` | Parallel model calls per request |
| `PARITOK_HISTORY_KEEP_RECENT` | `0` | Extra turns to protect beyond `protect_recent` |

## How it fits the pipeline

```
request
  ├─ tool selection (Layer 0)      ← paritok_tool_filter
  ├─ tool schema compaction        ← Headroom, unchanged
  ├─ tool description compaction   ← Headroom, unchanged
  └─ transform pipeline
       ├─ CacheAligner             ← Headroom, unchanged
       ├─ Paritok content compress ← paritok_content_compress
       ├─ Paritok history summary  ← paritok_history_summarize
       └─ ContentRouter            ← Headroom, unchanged
            └─ SmartCrusher / CodeCompressor / Kompress / …
```

### Which endpoints run the tool filter

| Endpoint | Tool filter | Why |
|---|---|---|
| `/v1/messages` (Anthropic) | yes | Claude Code's path |
| `/v1/chat/completions` (OpenAI) | yes | opencode, Cline, Aider, LiteLLM |
| `/v1/responses` (Codex) | no | no session id reaches the compression stage |
| Gemini | no | not wired |

The Responses gap is deliberate rather than pending. Selection is frozen per
session so the `tools` array stays byte-stable; without a stable session key
the selection would be recomputed every turn and flip the array underneath the
provider's prefix cache — worse than not filtering at all. Threading a session
id into `_compress_openai_responses_payload` is the fix, and it has to go
through the kwarg-shedding contract that method keeps for subclass overrides.

The other two levers (content compression, history summarization) live in the
transform pipeline, so they apply on every endpoint regardless.

Two ordering decisions are load-bearing:

* **Selection before compaction.** Selection removes whole schemas; compaction
  shrinks what survives. The reverse order would spend work on schemas about to
  be discarded.
* **Paritok after CacheAligner, before ContentRouter.** The cached prefix is
  assessed first, and everything Paritok leaves behind still gets Headroom's
  full compressor suite.

## Cache safety

Paritok is designed to *raise* cache-hit rates, not spend them:

* No stage rewrites anything inside `frozen_message_count` or the
  `protect_recent` tail. Rewriting cached bytes would invalidate the provider's
  prefix cache and cost more than the tokens saved.
* Tool selection is **frozen per session** after the first turn, so the `tools`
  array stays byte-stable turn to turn. This also makes the filter immune to MCP
  async-load jitter, where the tool pool grows over the first few turns.
* Core execution tools (`shell`, `apply_patch`, `bash`, …) are never dropped,
  whatever the query looks like — an agent without them cannot act at all.

## Retrieval

Paritok does not introduce a second marker format. Originals go into Headroom's
own `CompressionStore` tagged `compression_strategy="paritok_4b"`, and
compressed content carries a native `<<ccr:HASH>>` marker. So:

* `headroom_retrieve` recovers Paritok-compressed content with no new tool.
* CCR inline resolution and proactive expansion work unchanged.
* Paritok savings land in the same `headroom savings` totals as everything else.
  They are not broken out separately — `headroom savings` does not group by
  strategy — but each store entry carries `compression_strategy="paritok_4b"`,
  so the attribution is there for anything that wants it.
* ContentRouter sees the marker and declines to re-compress the block — which is
  correct, because a second pass would hash the *compressed* text as the new
  "original" and destroy the only handle on the real bytes (#2694).

When `paritok_chain_model` is on, chaining happens *inside* the Paritok stage
and passes the true original via Kompress's `ccr_original` parameter, so exactly
one store entry maps the real source to the final text.

## Failure behaviour

Compression is an optimization, never a correctness requirement. Every failure
path — backend down, timeout, malformed reply, store failure, embedder missing —
returns the original content and records a warning. A missing model degrades
your token savings; it never breaks a turn or fails a request.

## Recovering dropped tools

The filter self-heals. When it withholds a schema and the agent replies that it
lacks the capability, that reply comes back as conversation history on the very
next request — so recovery is detected on the **request** side, not by hooking
the response:

1. `last_assistant_text()` pulls the previous turn out of `messages`.
2. `looks_like_missing_tool_help()` — a cheap regex — decides whether it is a
   missing-tool complaint. This gate matters: recovery embeds the whole dropped
   pool, which is far too expensive to pay every turn.
3. The dropped pool is re-ranked against the agent's own wording and the best
   `PARITOK_TOOL_RECOVER_K` matches are pinned into the session's frozen set,
   so the same miss cannot repeat.

Doing it on the request side means one code path covers streaming and
non-streaming alike. `tool_recover_k` defaults to **2**, well under `k_min`:
pinning is permanent, so a generous value would unwind the filter a few misses
into a long session.

Stubs say the schema is *withheld* rather than advertising a schema-loading tool
call: no such tool is injected, and promising one would cost the model a wasted
turn.

## Why there is no ref-expand flag

Paritok upstream gates expansion behind a flag because it injects its own
virtual `read_original` tool. Headroom already does that job: `ccr/tool_injection.py`
scans outgoing content for `<<ccr:HASH>>` (`re.compile(r"<<ccr:([a-f0-9]{12,24})\b")`),
verifies the hash against the store, and injects `headroom_retrieve`; the
response handler intercepts the call. Paritok's markers are 24 hex characters
and land in that same store, so expansion is live and unconditional. A toggle
would have nothing to switch off.
