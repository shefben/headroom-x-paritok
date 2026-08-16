# Additional token levers

Levers added alongside the Paritok augmentation, each an independent rollout
feature and each **off by default**. A disabled lever is inert — its transform
is never constructed and its body field is never written, so the request path
is byte-identical to stock Headroom.

| Lever | Feature | Saves | Needs |
|---|---|---|---|
| Server-side context editing | `anthropic_context_editing` | input tokens, provider-side | first-party Anthropic |
| Server-side compaction | `anthropic_context_compaction` | input tokens, provider-side | first-party Anthropic, Opus/Sonnet 4.6+ |
| Tool-schema compilation | `tool_schema_compilation` | input tokens, tools array | nothing |
| TOON re-encoding | `toon_encoding` | input tokens, losslessly | nothing |
| Tool-result pruning | `tool_result_pruning` | input tokens, before compression | nothing |
| Observation masking | `observation_masking` | input tokens, before compression | nothing |
| Edit-format steering | `HEADROOM_EDIT_FORMAT` | **output** tokens | output shaper on |
| Task reinjection | `task_reminder` | nothing — buys adherence | nothing |

Enabling:

```bash
# Legacy-style env aliases
HEADROOM_CONTEXT_EDITING=1 HEADROOM_TOOL_RESULT_PRUNING=1 headroom proxy

# Or the canonical feature list
HEADROOM_FEATURES=anthropic_context_editing,tool_result_pruning,\
observation_masking,toon_encoding,tool_schema_compilation headroom proxy

# Edit-format steering is a mode on the output shaper, not a rollout feature.
# The shaper itself is a BETA-channel feature, so it needs the channel too —
# HEADROOM_OUTPUT_SHAPER=1 alone does nothing on the default stable channel.
HEADROOM_ROLLOUT_CHANNEL=beta HEADROOM_OUTPUT_SHAPER=1 \
  HEADROOM_EDIT_FORMAT=minimal headroom proxy
```

`headroom rollout status` shows every feature and how it resolved.

## Where each lever sits

Three of these act on the tools array, four on the message array, one on the
system prompt. They are independent and stack:

```
tools[]     paritok_tool_filter -> tool_schema_compaction (L1/L2)
                                -> tool_schema_compilation (L4)
messages[]  interceptors -> CacheAligner -> toon_encoding (1.3)
                        -> tool_result_pruning (1.4) -> observation_masking (1.45)
                        -> paritok -> ContentRouter
                        -> task_reminder (tail append, after everything)
body        context_management.edits[]: clear_tool_uses + compact
system      output shaper: verbosity + edit format
```

---

## Already present: Anthropic Tool Search

Progressive tool disclosure is **already shipped** in this tree and is **on by
default** — `HEADROOM_TOOL_SEARCH=0` opts out. `inject_tool_search_deferral`
(`proxy/helpers.py`) marks non-core tool schemas `defer_loading: true` and
injects a `tool_search_tool_regex_20251119` tool, so Anthropic excludes the
deferred schemas from the context window until the model searches for one.
`inject_tool_search_deferral_openai` does the same for `/v1/responses` on
gpt-5.4+. Nothing in this document duplicates it.

Related machinery already in place: third-party upstreams get client-originated
`tool_search_tool_*` entries stripped (custom gateways reject the first-party
shape), and tool-search history repair (#2805) drops `tool_reference` blocks a
side-request cannot validate.

---

## Server-side context editing

Asks Anthropic to clear stale `tool_use`/`tool_result` pairs on its own side,
replacing each with a short placeholder, once the request crosses a trigger.

Why this is not just "history trimming we could do ourselves":

* **The cleared bytes are never billed.** The proxy spends no model call, no
  embedding and no CPU to remove them.
* **It runs after the provider's prompt-cache lookup.** Anthropic evaluates
  cache hits against the request as sent, then applies the edits. Stripping the
  same history client-side changes the cached prefix and costs a full cache
  miss; letting the provider do it does not.

Anthropic reports 84% token reduction on a 100-turn eval, and +29% task
performance from context editing alone (+39% paired with the memory tool, which
this proxy already injects — see `proxy/memory_tool_adapter.py`). Those are
self-reported numbers on Anthropic's own eval; treat them as directional.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_CONTEXT_EDITING` | off | Enable the lever |
| `HEADROOM_CONTEXT_EDITING_TRIGGER_TOKENS` | `100000` | Input tokens before an edit fires |
| `HEADROOM_CONTEXT_EDITING_KEEP_TOOL_USES` | `3` | Most recent tool uses always retained |
| `HEADROOM_CONTEXT_EDITING_CLEAR_AT_LEAST_TOKENS` | `5000` | Minimum worth clearing |
| `HEADROOM_CONTEXT_EDITING_CLEAR_TOOL_INPUTS` | `false` | Also clear the tool *inputs* |
| `HEADROOM_CONTEXT_EDITING_EXCLUDE_TOOLS` | `headroom_retrieve` | Never cleared |

Two defaults are Headroom's own rather than Anthropic's:

* **`clear_at_least` defaults to 5,000, not 0.** Once an edit fires, the suffix
  after the clear point differs from what was cached, so subsequent turns
  re-cache from there. A clear that frees a handful of tokens is strictly worse
  than no clear at all — it pays a re-cache to save nothing. `clear_at_least` is
  the provider-side guard against exactly that.
* **`headroom_retrieve` is excluded.** Its result *is* the recovered original
  the model just spent a turn asking for; clearing it re-creates the gap.

### Behaviour

* **First-party Anthropic only.** `context_management` is an unknown body field
  to custom Anthropic-compatible gateways and they 400 on it, so the lever is
  gated on the same first-party test the tool-search deferral uses.
* **Client requests win.** A caller that already sent a `context_management`
  block with a `clear_tool_uses` edit is left alone — a harness that configured
  its own retention policy knows more about its transcript than we do. An
  unrelated edit in that block is preserved and ours is appended.
* **Inert when it could not fire.** With no more tool uses than `keep` would
  retain, no body field is added at all.
* **Beta header.** `context-management-2025-06-27` is merged, never assigned, so
  the session-sticky baseline and the memory tool's copy of the same token both
  survive.

### Reporting

Anthropic echoes what it actually did under
`usage.context_management.applied_edits[]`. Non-streaming responses record it as
tags `context_editing_cleared_tokens` / `context_editing_cleared_tool_uses`.

Deliberately **not** folded into `tokens_saved`: the proxy never sent those bytes
upstream this turn, so adding them to the client-side savings total would
double-count against `original_tokens`.

Known gap: streaming responses carry usage in the SSE `message_delta` handled by
a shared streaming helper, which does not receive the per-request flag — so the
tag is recorded on non-streaming turns only. The saving itself is unaffected;
only its telemetry is.

---

## Server-side compaction

The sibling lever, and the sharper instrument. Where `clear_tool_uses` deletes
stale tool results, `compact_20260112` summarises the older part of the
conversation into a compaction block and continues from there — the transcript
keeps its meaning instead of acquiring holes.

Beyond the token saving there is a correctness argument. The most common way a
home-grown compactor breaks a session is by summarising away a `tool_use` whose
`tool_result` survives; the next request 400s on an orphaned `tool_use_id`.
Anthropic's implementation maintains that pairing and the user/assistant
alternation by construction.

### The cost context editing does not have

Compaction puts a block into the **assistant response**, and the conversation
only continues from the compacted state if the client echoes that block back.
Headroom does not own the client's transcript.

* A client that passes unknown assistant content blocks through unchanged
  (Claude Code does) gets the full benefit.
* A client that drops them re-sends the raw history. Wasted provider work, not
  a broken session — the next turn simply compacts again.
* A client that validates assistant block types strictly could reject the
  response outright.

That third case is why this lever carries a model-family gate on top of its
feature flag, and why its docs say what they say. Enable it when you know the
client.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_CONTEXT_COMPACTION` | off | Enable the lever |
| `HEADROOM_CONTEXT_COMPACTION_TRIGGER_TOKENS` | `150000` | Input tokens before compaction fires |

Anthropic triggers near the context limit by default; Headroom sets an explicit
value instead. An implicit "near the limit" moves whenever the model's window
changes, and a lever whose firing point silently shifts is one you cannot
attribute savings to.

Model gate: Opus and Sonnet at 4.6 or later. The check is a whitelist walked
forward by version, so an unrecognised or aliased model is simply not compacted
rather than 400-ing every request that mentions it. Haiku is not supported.

Both provider levers merge into the same `context_management.edits` array
through one shared helper, so a request may legitimately carry both and neither
clobbers a client-supplied edit.

---

## Tool-schema compilation

The third independent axis on the tools array. The other two are already here
and neither does this:

| Stage | Axis | What it changes |
|---|---|---|
| `paritok_tool_filter` | selection | *which* schemas survive |
| `tool_schema_compaction` | lexical | drops annotation keys, truncates descriptions |
| `tool_schema_compilation` | **notation** | how a surviving schema is expressed |

The TSCG paper (arXiv:2605.04107) reports 52–57% savings with a proven ≥51%
floor, and a compact model going from 0% to 84.4% tool-call accuracy at 20
tools. The load-bearing result is its own ablation separating *formatting* from
*raw compression*: R² falls from 0.88 to 0.03, i.e. almost none of the accuracy
gain is explained by the prompt merely being shorter.

Measured here on a 12-tool synthetic catalog: **`safe` 24%, `full` 35%**. Real
catalogs with long prose descriptions will show less, because descriptions are
the bytes neither mode can absorb.

### Two modes

`safe` — the JSON schema stays semantically intact; only redundant property
descriptions are dropped (the ones that restate a self-explanatory parameter
name). No signature is emitted, because with the JSON still present a signature
would state every type twice and come out *larger*. Nothing a provider
validates against changes.

`full` — the JSON is reduced to property names plus `required`, and everything
else moves into one dense signature line:

```
Search the index for matching records.
params: query:str limit:int=25 mode:{fast|exact}?
- limit: Maximum number of records to return
- mode: Search strategy to use
```

`!`-free by design: `?` marks optional, `=x` marks a default, `{a|b}` an enum,
`T[]` an array. `required` stays in the JSON because it is the one constraint
providers act on when they validate at all.

### What `full` refuses to touch

`full` moves enums and defaults out of machine-readable JSON, so it declines
any tool where that matters, falling back to `safe` behaviour:

* `strict: true` (either wire position), or `additionalProperties: false` — the
  caller asked the provider to enforce the shape;
* `$ref` / `$defs` / `allOf` / `anyOf` / `oneOf` / `patternProperties`;
* any property keyword this module does not know how to absorb — so a new JSON
  Schema keyword appearing in the wild degrades rather than being discarded;
* nesting deeper than two levels.

Both modes end in the same guard: if compilation did not shrink the payload,
the original is returned untouched. A catalog of already-terse schemas cannot
be made worse by turning this on.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_TOOL_SCHEMA_COMPILATION` | off | Enable the lever |
| `HEADROOM_TOOL_SCHEMA_COMPILE` | `safe` | `off` / `safe` / `full` |

Applies on all three wire shapes (Anthropic `input_schema`, Responses flat
`parameters`, chat nested `function.parameters`). Results are cached by tools
digest, and compilation is a pure function of the array plus the mode — so the
compiled bytes are identical every turn. Enabling costs one cache miss and
nothing after it.

**Not built:** cross-tool factoring of repeated sub-schemas (TSCG's operator 8,
TRON's named definitions). It needs a shared text region for the definitions,
and the only natural one is the system prompt — which couples two independently
cached regions to each other. Recorded rather than attempted.

---

## TOON re-encoding

The only lever here that removes nothing. A tool result that is a uniform JSON
array is re-serialised into Token-Oriented Object Notation, which states each
field name once in a header instead of once per row:

```
[8]{id,name,role,active}:
  0,user0,admin,true
  1,user1,user,true
```

The published benchmark (244 retrieval questions, four models) puts the saving
at 42.6% against formatted JSON at **72.2% accuracy versus JSON's 71.4%** — the
compact form was, if anything, marginally easier to read. On uniform records
the gap is wider: 60.7% on 100-row tabular data. Measured here: a 20-row
four-field array drops from 662 to 189 bytes.

Every caveat in the literature is a guard in the code:

* **Nesting inverts the result** — deeply nested config costs *more* in TOON
  than compact JSON, so only flat uniform rows are encoded and anything else is
  passed through untouched.
* **Output-side TOON is a mistake** — a Feb 2026 generation benchmark found
  plain JSON beat both TOON and constrained decoding. This lever is input-side
  only; nothing asks the model to *produce* TOON and no format primer is
  injected.
* **Small arrays do not amortise** — the header costs a few tokens, hence
  `min_rows`, and the final emit-only-if-smaller guard on top.

Values containing a comma, quote, colon, bracket or leading/trailing whitespace
are emitted with JSON string quoting, which reads back identically.

Cache safety is trivial and worth stating because nothing else in this
directory gets it free: the encoding is a pure function of the block's own
bytes. It reads no other message, so it needs no lookahead bound, no frozen
prefix and no protected tail — the newest tool result is as safe to encode as
the oldest.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_TOON_ENCODING` | off | Enable the transform |
| `HEADROOM_TOON_ENCODING_MIN_ROWS` | `4` | Below this the header does not amortise |
| `HEADROOM_TOON_ENCODING_MIN_FIELDS` | `2` | A one-field array is a list of scalars |

Pipeline slot **1.3** — deliberately first of the three tool-result stages.
It is lossless, so shrinking a result here can take it below the size threshold
at which pruning or masking would have replaced it outright. Reduce first, then
decide what to drop.

---

## Tool-result pruning

Measured on SWE-bench trajectories, tool results are roughly two thirds of an
agentic transcript's tokens and 40–60% of those bytes are removable with no
measurable loss in task performance. Headroom already reclaimed two slices:
`read_lifecycle` retires stale and superseded `Read` output, `cross_turn_dedup`
folds spans that appeared verbatim earlier. This transform covers the rest —
`Bash`, `Grep`, MCP calls, anything that returned bulk text — using the signal
those two cannot see: **whether the model ever referred to the result again**.

Three rules, all deterministic, none needing a model or an embedding:

| Rule | Fires when |
|---|---|
| `empty` | The body carries no information (blank, `(no output)`, `exit code: 0`) |
| `superseded` | The same tool ran again with byte-identical input |
| `unreferenced` | Almost none of the result's distinctive identifiers reappear later |

`unreferenced` is the relevance lever. Identifiers are file paths, symbols ≥5
characters, hex codes and 4+ digit numbers, minus a small stoplist of tokens so
common that their reappearance says nothing. The verdict is a **ratio** — the
fraction of distinct identifiers that reappear — so a 2,000-line result is not
kept alive by one incidental token.

### Goal conditioning

`unreferenced` measures what the model *already did*, which says nothing about
what it is still working on. Goal conditioning supplies the missing half.

SWE-Pruner (arXiv:2601.16746) is the clearest demonstration that task-agnostic
relevance leaves value on the table: it has the agent state an objective
("focus on error handling"), prunes against that, and reports 23–54% token
reduction on SWE-bench Verified **with the solve rate going up**. Query-aware
beating task-agnostic is the consistent finding across this literature.

A proxy cannot ask the agent to state its objective, and does not need to — the
objective is already in the transcript as the last thing the user said before
the tool ran. So each candidate is also weighed against that:

* overlap ≥ `goal_protect_overlap` → **kept outright**, referenced or not;
* overlap of exactly zero → judged at the stricter
  `goal_unrelated_reference_ratio` instead of the base threshold.

Overlap is denominated in the *goal's* terms, not the result's: a 5,000-line
result containing one of the three words the user used is far more likely to be
on-task than the reverse ratio would suggest. Goals are prose and tool output is
not, so the goal's vocabulary unions the output-tuned identifier extractor with
plain content words — otherwise "make the retry logic idempotent" would extract
nothing and every goal would read as empty.

**The prefix-cache trap.** The obvious implementation — score everything against
the *latest* user message — is the same mistake a naive forward-looking rule
makes, only worse: the latest message changes every turn, so every historical
verdict is re-derived against new evidence and the prefix churns from the first
tool result onward. The hint is therefore resolved **backwards**: the goal in
force at index `i` is the most recent user text at index ≤ `i`. Fixed prefix,
identical every turn, and the more defensible reading anyway.

A user message full of `tool_result` blocks is not the user talking; only text
blocks count.

Pruned bodies are not deleted. Each goes into the `CompressionStore` under
`compression_strategy="tool_result_pruning"` and is replaced by a one-line
summary plus a native `<<ccr:HASH>>` marker, so `headroom_retrieve` hands the
full text back on demand and ContentRouter correctly skips the block.

The `empty` rule is the exception: nothing is recoverable, so it writes no store
entry and no marker.

### Cache safety — the load-bearing constraint

`superseded` and `unreferenced` both look *forward*, and a naive forward scan is
exactly what breaks a provider prefix cache: appending turn N+1 could change the
verdict on a message from turn 3, mutating bytes the provider already cached.

The fix is a **bounded lookahead**. A candidate at message index `i` is judged
only against `messages[i+1 : i+1+lookahead]`, and only once that whole window
exists. The verdict therefore depends on a fixed prefix of the transcript and
can never change as the conversation grows — whatever the transform emitted for
message `i` on turn N it emits byte-for-byte on turn N+1.

The cost is that a supersede or a reference landing past the window is missed,
which loses savings but never correctness. `TestCacheSafety` in
`tests/test_tool_result_pruning.py` asserts this directly.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_TOOL_RESULT_PRUNING` | off | Enable the transform |
| `HEADROOM_TOOL_RESULT_PRUNING_MIN_TOKENS` | `250` | Skip smaller results |
| `HEADROOM_TOOL_RESULT_PRUNING_LOOKAHEAD` | `8` | Messages examined after a candidate |
| `HEADROOM_TOOL_RESULT_PRUNING_MAX_REFERENCE_RATIO` | `0.02` | Prune below this reference fraction |
| `HEADROOM_TOOL_RESULT_PRUNING_MIN_IDENTIFIERS` | `8` | Below this, `unreferenced` abstains |
| `HEADROOM_TOOL_RESULT_PRUNING_PRUNE_ERRORS` | `false` | Errors are kept by default |
| `HEADROOM_TOOL_RESULT_PRUNING_GOAL` | `true` | Goal conditioning |
| `HEADROOM_TOOL_RESULT_PRUNING_GOAL_PROTECT_OVERLAP` | `0.34` | Keep outright above this goal overlap |
| `HEADROOM_TOOL_RESULT_PRUNING_GOAL_UNRELATED_RATIO` | `0.05` | Threshold for results sharing nothing with the goal |

Goal conditioning defaults **on** because its primary effect is protective. The
one direction in which it prunes more is the zero-overlap case, which is why
that gap is small and lives behind its own knob.

Every rule prefers a false negative. Errors steer the model's next several turns
even when it never quotes them, so they are kept unless you opt in. Anything
already carrying a CCR marker is left to whichever stage owns it. A store
failure leaves the original in place rather than emitting a marker nothing can
redeem.

### Pipeline position

Slot 1.4 — after CacheAligner and TOON, before Paritok and ContentRouter. A
result this stage removes is one nothing downstream spends a model call, an
embedding or a CCR entry on.

---

## Observation masking

The finding behind this one is negative, and that is what makes it useful.
JetBrains Research ran the comparison inside SWE-agent on SWE-bench Verified
across five model configurations ("The Complexity Trap", arXiv:2508.21433):
replacing older *observations* with a short placeholder — keeping every action
and every reasoning block intact — **halved cost while matching, and sometimes
slightly exceeding, the solve rate of LLM-based summarization**. The expensive
summarizer bought nothing over the trivial heuristic. Their hybrid of both
added only a further 7–11%.

So this transform runs no model, computes no embedding, reads no content, and
makes one decision per tool result: is it old enough. It is the complement of
pruning, not a competitor — pruning removes what the transcript *proves* went
unused and abstains whenever unsure; masking removes what is simply old.

Two deviations from the paper, both because Headroom has machinery its setup
did not:

* **Masked bodies stay recoverable.** The paper's placeholder is destructive;
  here the body goes to the `CompressionStore` and the placeholder carries a
  `<<ccr:HASH>>` marker. Strictly better, at the cost of one store write.
* **Actions and reasoning are structurally out of scope** — the shared
  candidate finder only ever returns `tool_result` blocks and `role="tool"`
  messages, so there is no rule to get wrong.

### Cache behaviour, stated precisely

The rule is **monotone**: a result is masked once `keep_recent` messages follow
it, and stays masked. A given message's bytes therefore change at most once —
on the turn it crosses the horizon — and are stable before and after.

That is the same one-time transition tool-result pruning accepts when a
candidate's lookahead window fills, and it is the best available: an age rule
that never re-cut the prefix would be a rule that never fires. What it is *not*
is the naive alternative, where a verdict keeps changing as new evidence
arrives and the prefix is re-cut every turn. `TestMonotonicity` asserts it.

`frozen_message_count` is honoured, and `protect_recent` can only tighten the
window, never widen it.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_OBSERVATION_MASKING` | off | Enable the transform |
| `HEADROOM_OBSERVATION_MASKING_KEEP_RECENT` | `12` | Tail messages never masked |
| `HEADROOM_OBSERVATION_MASKING_MIN_TOKENS` | `200` | Below this the placeholder costs more |
| `HEADROOM_OBSERVATION_MASKING_MASK_ERRORS` | `false` | Errors kept by default |

Slot 1.45 — after pruning, so pruning gets first refusal on every result.
Anything pruning already replaced carries a CCR marker, which masking skips, so
the two never fight over the same block.

Known limitation, from the authors' own follow-up: masking grows unbounded
while summarization is bounded. On very long horizons masking alone is not
sufficient, and the Paritok history summary still earns its place.

---

## Edit-format steering

The first **output-token** lever in the proxy. Output bills several times higher
than input on every major provider, and on a coding agent the largest avoidable
output is a model re-emitting code it was never asked to change: a whole-file
write where a three-line edit would do, a function pasted back "for context", a
closing summary restating the diff it just produced.

Aider measured the same effect in reverse — moving GPT-4 Turbo from whole-file
to unified diffs made it "3x less lazy". Same model, same task; what changed was
the instruction about *how* to express a change.

Two cumulative modes:

| Mode | Adds |
|---|---|
| `minimal` | Prefer targeted edits over rewrites; never re-emit unchanged code |
| `strict` | …plus no post-edit recap, and code in prose limited to changed lines + 3 context lines |

Applies on all three wire formats: Anthropic `system`, OpenAI chat
`role: system`/`developer`, OpenAI Responses `instructions`.

### Safety properties

* **Tail placement.** The block appends *after* any `cache_control` breakpoint
  the client set, so the cached prefix stays byte-identical and only the small,
  byte-stable steering block is reprocessed.
* **Gated on capability.** The block is only injected when the request actually
  exposes a file-editing tool (`Edit`, `Write`, `apply_patch`, a typed
  `text_editor_*`, …). Telling an agent that cannot edit a file to prefer small
  edits spends bytes on nothing.
* **Idempotent.** Its own sentinel (`<headroom_edit_format>`), separate from the
  verbosity block's, so the two levers are independently replaceable. Re-shaping
  the same body is a no-op.
* **Byte-stable text.** The strings live in `output_edit_format_policy.py` and a
  wording change is a cache-busting release, exactly as for the verbosity
  levels.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_OUTPUT_SHAPER` | off | Required — this is a shaper lever |
| `HEADROOM_EDIT_FORMAT` | `off` | `off` / `minimal` / `strict` |

`1`/`yes`/`true` resolve to `minimal`. An unrecognised value resolves to `off`
rather than raising: a typo in an env var must never fail a request.

---

## Task reinjection

The one lever here that saves nothing. It spends a few hundred tokens to buy
instruction adherence on long runs.

Adherence decays with distance. LongIns isolates it by running the same
questions with instructions stated once at the top versus repeated before each
question; GPT-4o falls from roughly 76 to 51 between 256 and 16k tokens. R&R
(arXiv:2403.05004) attacks it by re-injecting an instruction reminder at
intervals.

The detail that is easy to get backwards: **duplicate, do not relocate.** A
systematic placement study found that moving instructions to the end was the
*worst* of four configurations tested, while head-and-tail together was among
the best. The user's original message therefore stays exactly where it is and a
restatement is added at the tail. Nothing is moved.

### Why the tail of `messages`, not the system prompt

Every other steering block in Headroom lives in the system prompt, which works
because those blocks are byte-stable. This one is not — it quotes the current
task, which changes. Varying text in the system prompt would invalidate the
cached prefix of every conversation on every turn, costing far more than the
adherence is worth.

Appending after the final message puts it past every `cache_control` breakpoint,
so the bytes are ones the provider was going to reprocess anyway. The reminder
is free in cache terms.

It also never accumulates: the proxy rewrites the outbound body only, and the
client builds its next request from its own history, which never contained the
block. There is nothing to strip.

### Gates

All must hold, or nothing is injected:

* input tokens ≥ `trigger_tokens`;
* the last real user instruction is ≥ `min_distance` messages back — if it is
  the most recent message, restating it is pure cost;
* the last message is a `user` turn (an assistant tail is a prefill, and an
  assistant turn ending in `tool_use` must keep ending in one);
* the instruction is not a harness-injected turn (`<system-reminder>`,
  interrupt notices, caveat banners) and is longer than an acknowledgement.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_TASK_REMINDER` | off | Enable the lever |
| `HEADROOM_TASK_REMINDER_TRIGGER_TOKENS` | `30000` | Below this, decay is not yet real |
| `HEADROOM_TASK_REMINDER_MIN_DISTANCE` | `12` | Messages since the instruction |
| `HEADROOM_TASK_REMINDER_MAX_CHARS` | `600` | Clip, preferring a line boundary |

Anthropic `messages` and OpenAI Responses `input` both supported;
chat/completions is not, since its tail is a `role: "tool"` message that cannot
carry the block and inserting a synthetic user turn there changes the shape of
the request.

Documented ceiling: a controlled study across five models found the perfect-
response rate hits zero by 80 concurrent instructions **regardless of format or
placement**. Reinjection does not rescue an overloaded instruction set.

---

## Not built: perplexity-based prompt compression

Recorded because Paritok's content compressor sits adjacent to this family and
the temptation to reach for it will recur.

The LLMLingua line (perplexity-scored token dropping) is the wrong tool for
agent transcripts, and the evidence is consistent across independent papers:

* **Format destruction.** LLMLingua-2 past 30% compression failed *all* tasks
  in a Web Shopping agent benchmark.
* **Structured content.** On tool schemas it scored 80.0% accuracy at 50.8%
  savings against TSCG's 93.3% at 74.8% — and both variants need GPU inference
  and are non-deterministic, which is disqualifying for prefix caching on its
  own.
* **Downstream blowup.** At matched compression in a code-agent setting it
  dropped a model to 35% accuracy while *total* tokens went 116k → 243k.
  Compressing the prompt cost more in generated output than it saved.

The 2025–26 consensus is to split by content type — deterministic compilation
for schemas, query-aware selection for retrieved context, abstractive rewriting
for prose. That is roughly what this proxy already does, and worth not
regressing.

---

## Not built: Code Mode

Deliberately out of scope, recorded so the reasoning is not re-derived.

Code Mode (Anthropic Nov 2025, Cloudflare Feb 2026) replaces a tool pool with
`search()` + `execute()` and has the largest published numbers of any lever —
150,000 → 2,000 tokens; 1.17M → ~1,000 over 2,500+ endpoints. It wins twice:
tool schemas load only when generated code imports them, and intermediate data
is filtered inside the sandbox instead of round-tripping through context.

It does not port to a proxy as-is. Headroom does not execute tools — the client
does. Replacing Claude Code's tools array with `execute(code)` leaves the client
with a tool it cannot run and no way to reach `Read`/`Bash` on the user's
machine. The headline numbers come from MCP tool schemas, and this tree has no
MCP *client* (`mcp_registry/` installs Headroom's server into agent configs; it
does not call out).

Two scopes would be implementable:

* **CCR exec** — `headroom_exec(code, refs)` running sandboxed code in-process
  over `CompressionStore` originals, so retrieving a 40K log becomes "filter the
  log, return 20 lines" instead of dumping all 40K back. Uses the proven
  `ccr/tool_injection.py` + `ccr/response_handler.py` seam.
* **Full MCP Code Mode** — an MCP client in the proxy, connected servers exposed
  as sandbox bindings, their schemas collapsed to two tools. The real 98% lever
  and by far the larger build.

Both need a restricted execution sandbox for model-generated code, which is the
bulk of the risk and the work in either.
