# Gemma 4 MTP + `--n-cpu-moe` + vision (`--mmproj`): fixes

Authoritative writeup of the four changes that make Gemma 4 MTP speculative
decoding work on a VRAM-constrained, CPU-offloaded MoE target with a vision
projector loaded. Supersedes the investigation log in
`mtp-mmproj-textonly-fix.md`.

## Target configuration

- Model: `gemma-4-26B-A4B-it` (MoE, `gemma4` arch), Q4_K_M.
- Draft head: `gemma-4-26B-A4B-it-assistant` (`gemma4_assistant`), Q8_0, via `--mtp-head`.
- GPU: single RTX 5080, 16 GB. 26B does **not** fit fully → `--n-cpu-moe 28`
  (experts on CPU) is mandatory.
- Vision: `--mmproj` (gemma4v projector).
- Speculative: `--spec-type mtp`, `--draft-block-size 3`, TurboQuant KV
  (`-ctk/-ctv turbo4/turbo2`, draft `-ctkd/-ctvd turbo3`), `-fa on`.

Symptom before fixes: MTP never drafted with `--mmproj` loaded; removing
`--mmproj` produced a silent access-violation crash on the first draft.

---

## Fix 1 — MTP must draft on text turns even when `--mmproj` is loaded

**Symptom:** with `--mmproj`, every turn logged
`skipping speculative prime for multimodal prompt` and `#gen drafts = 0`, even
for pure-text prompts.

**Root cause:** `server_tokens::has_mtmd` is set from *mmproj presence*, not from
whether a prompt actually contains image chunks
(`tools/server/server-context.cpp` slot init ~L915 and prompt construction
~L3237: `has_mtmd = mctx != nullptr`). Two speculative gates keyed off `has_mtmd`,
so any prompt on an mmproj server was treated as multimodal and drafting was
disabled.

**Fix:**
- `tools/server/server-common.h`: add `bool has_media() const { return !map_idx_to_media.empty(); }`
  — true only when the prompt actually contains image/media chunks, distinct from
  `has_mtmd` (multimodal-capable container).
- `tools/server/server-context.cpp` draft gate (~L2247): `has_mtmd` → `has_media()`.
- `tools/server/server-context.cpp` prime gate (~L3051): `has_mtmd` → `has_media()`.

`has_mtmd` semantics are unchanged, so the independent cache-reuse / context-shift
asserts that depend on it are unaffected.

---

## Fix 2 — the real crash: decode epilogue run on the self-contained MTP graph

**Symptom:** silent access violation (no assert, no error) on the first MTP draft.
Localized via instrumentation to `ggml`-level, then to graph build.

**Root cause:** `llama_model::build_graph` (`src/llama-model.cpp` ~L9576) runs a
generic decode epilogue after the arch builder:

```
llm->build_pooling(...);
llm->build_sampling();
llm->build_dense_out(...);
llm->res->set_outputs();
```

The Gemma 4 MTP graph (`llm_build_gemma4_mtp`, `src/models/gemma4-assistant.cpp`)
is **self-contained**: it publishes its own outputs (`t_logits` / `t_embd` /
`t_argmax`) and `ggml_build_forward_expand`s them in its constructor. The generic
epilogue does not apply, and `build_pooling` faults on the single-token MTP graph
— the MTP ubatch sets `output[0] = 0` while `n_outputs` is forced to 1, so generic
output handling dereferences a non-existent output.

The reserve build in `ensure_sched_mtp` survived only because its ubatch sets
`output[0] = 1`. Not actually `--n-cpu-moe`-specific — it crashes whenever the MTP
graph is built for a real decode; without `--n-cpu-moe` the 26B OOMs at load
first, so this code path was never reached upstream.

**Fix:** `src/llama-model.cpp` — skip the entire decode epilogue for
`params.gtype == LLM_GRAPH_TYPE_MTP`:

```cpp
if (params.gtype != LLM_GRAPH_TYPE_MTP) {
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);
    llm->build_sampling();
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);
    llm->res->set_outputs();
}
return llm->res->get_gf();
```

This was the load-bearing fix. After it, `graph_compute_mtp` returns status 0 and
MTP drafts fire.

---

## Fix 3 — prime path asserts under `--mmproj` once text turns are allowed

**Symptom:** after Fix 1 + Fix 2, a text turn with `--mmproj` hit
`server-common.cpp:387: GGML_ASSERT(!has_mtmd) failed`.

**Root cause:** Fix 1's prime gate sends text turns into
`common_speculative_begin(spec, slot.prompt.tokens.get_text_tokens())`.
`server_tokens::get_text_tokens()` asserts `!has_mtmd`, but with `--mmproj` loaded
a text-only prompt still has `has_mtmd == true`. The draft path already avoided
this (passes an empty token list when `mctx && mtmd_safe_spec`); the prime path
did not. MTP's `begin()` ignores the prompt entirely (`GGML_UNUSED(prompt)`).

**Fix:** `tools/server/server-context.cpp` prime path (~L3060): pass an empty token
list when the spec impls are mtmd-safe, mirroring the draft path:

```cpp
static const llama_tokens k_empty_prime;
const bool mtmd_safe_prime = mctx && common_speculative_all_impls_mtmd_safe(slot.spec);
common_speculative_begin(slot.spec,
        mtmd_safe_prime ? k_empty_prime : slot.prompt.tokens.get_text_tokens());
```

---

## Fix 4 (minor) — `sched_mtp` op-offload

**Change:** `src/llama-context.cpp` `ensure_sched_mtp` creates `sched_mtp` with
`op_offload = false` (instead of inheriting `cparams.op_offload`). The MTP graph is
GPU-resident (assistant on GPU, target KV on GPU); forcing op-offload off keeps it
there and avoids the scheduler trying to push MTP ops onto the CPU backend under
`--n-cpu-moe`. The CPU backend stays in the backend list as the required fallback
(`ggml_backend_sched_new` asserts the last backend is CPU). Harmless; not strictly
load-bearing (the crash was Fix 2), but defensible and known-good.

---

## Image-turn drafting is safe (by design)

After the fixes, MTP also drafts on image turns (the per-turn draft gate sees no
live media at generation time). This is correct and beneficial, not a bug:

- **Speculative decoding is output-correct by construction** — the target verifies
  every drafted token and rejects any it disagrees with. Drafting on an image turn
  can never corrupt output; worst case is wasted compute at low acceptance.
- MTP per-step draft only needs the **last accepted token's** hidden state (always a
  text position) and cross-attends into the target KV (image K/V live there fine).
  It never needs per-token hidden states over image positions.
- The only image-incompatible step is the **prime**, which is still skipped on
  media prompts (and for MTP `begin()` is a near no-op anyway).

Observed: ~60–75% token acceptance on image turns, image content parsed correctly.

---

## Verification

Run with `--mmproj`, `--spec-type mtp`, valid `--mtp-head`, `--n-cpu-moe 28`.

1. Text turn: no `skipping speculative prime` line; `statistics mtp` shows
   `#gen drafts > 0`, `#acc tokens > 0`; generation correct.
2. Image turn: `skipping speculative prime for multimodal prompt` logged for the
   prime; drafting still occurs during generation; image described correctly.
3. Without `--mmproj`: unchanged; MTP drafts as before.

Confirmed result: `statistics mtp: #calls(b,g,a) = 1 634 459, #gen drafts = 634,
#acc drafts = 459, #gen tokens = 1268, #acc tokens = 804`.

---

## Instrumentation used (removed after the fix)

The crash was a silent access violation (no assert/message), so it was localized
with env-gated, immediately-flushed `fprintf(stderr, ...)` checkpoints. Enabled via
`LLAMA_MTP_TRACE=1`; each printed line was `fflush`-ed so the **last line before the
crash** named the failing call. Tags and locations:

| Tag | File | Covered |
|---|---|---|
| `[DEC_TRACE]`   | src/llama-context.cpp | target decode: `process_ubatch`, embeddings extract |
| `[SPEC_TRACE]`  | common/speculative.cpp | draft submit: `h_prev` read, `decode_mtp_async` |
| `[MTP_TRACE]`   | src/llama-context.cpp | `decode_mtp_async`, `ensure_sched_mtp`, `process_ubatch_mtp`, `decode_mtp_run` |
| `[GRAPH_TRACE]` | src/llama-graph.cpp | `build_attn_mtp` `get_k`/`get_v` |
| `[G4MTP_TRACE]` | src/models/gemma4-assistant.cpp | assistant graph build stages |
| `[BG_TRACE]`    | src/llama-model.cpp | `build_graph` decode epilogue (pinpointed `build_pooling`) |

Method: bisect down the call stack one layer per rebuild — server draft call →
`decode_mtp_async` → `ensure_sched_mtp` (sched create / reserve / build) →
`process_ubatch_mtp` (build / alloc / compute) → `build_graph` epilogue. Two false
leads ruled out along the way: dropping the CPU backend from `sched_mtp` (illegal —
CPU-last assert) and `op_offload=false` (no effect on the crash). `GGML_SCHED_DEBUG`
was unhelpful because its `GGML_LOG_DEBUG` output is filtered by the server log
level and buffered out of order.

All `LLAMA_MTP_TRACE` instrumentation has been removed post-fix. The four fixes
above remain.

## Fix 5 — vision breaks on templates that ignore the `media_marker` part type

Symptom (e.g. Gemma 4 12B coder finetune): text turns work, but the first image turn
fails with

```
render_message_to_json: Neither string content nor typed content is supported by the template.
tokenize: error: number of bitmaps (1) does not match number of markers (0)
```

Cause. On an OAI `/v1/chat/completions` request with an image, the server rewrites the
image content part into a part of type **`media_marker`** whose text is the mtmd marker
`<__media__>` (`tools/server/server-common.cpp`, ~L1028). `render_message_to_json`
(`common/chat.cpp`) then chooses how to present content to the Jinja template from the
template's detected `jinja::caps`:

- `supports_string_content` only → parts concatenated to a string (marker preserved inline).
- `supports_typed_content` only / both → content handed over as an array of typed parts,
  including `{type:"media_marker", text:"<__media__>"}`.

The 12B's template advertises **neither** (caps detection fails — that's the warning). The
original code's fallback for the neither-case was the *typed-parts* path. The template's
content loop only handles `text`/`image`/`audio`/`video` types and has no `media_marker`
branch, so the marker part is silently dropped → 0 markers in the prompt → `mtmd` aborts
against the 1 attached bitmap. The 26B works only because its template is detected as
string-capable, so the marker survives as inline text.

Fix (`common/chat.cpp`, `render_message_to_json`): when the template supports neither form,
fall back to **string** content (concatenated, marker inline) rather than typed parts. This
is the same string path text-only turns already use for this template, so there is no
regression, and it is general — any model whose template doesn't know the internal
`media_marker` type now works.

## Files changed (fixes only, after instrumentation removal)

- `tools/server/server-common.h` — `has_media()`
- `tools/server/server-context.cpp` — draft gate, prime gate
- `src/llama-model.cpp` — skip decode epilogue for MTP graphs
- `src/llama-context.cpp` — `sched_mtp` op_offload = false
- `common/chat.cpp` — `render_message_to_json` neither-caps fallback to string content (Fix 5)
