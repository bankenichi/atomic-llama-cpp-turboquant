# Fix: MTP/NextN speculative decoding on text turns when `--mmproj` is loaded

## Problem

With an mmproj loaded (`--mmproj`), `--spec-type mtp` (and `nextn`) never produced
drafts, even for pure-text turns. Symptom in the server log:

```
slot update_slots: ... skipping speculative prime for multimodal prompt
statistics mtp: #calls(b,g,a) = 0 0 0, #gen drafts = 0, #acc drafts = 0
```

`docs/speculative.md` claims text-only turns on a multimodal slot draft as usual
and only image turns fall back. The implementation did not honor that.

## Root cause

`server_tokens::has_mtmd` is set from **mmproj presence**, not from whether a
prompt actually contains image chunks:

- `tools/server/server-context.cpp:915` — `slot.prompt.tokens.has_mtmd = mctx != nullptr;`
- `tools/server/server-context.cpp:3237` — prompt tokens constructed with `has_mtmd = mctx != nullptr`

Two speculative gates keyed off `has_mtmd`, so any prompt on an mmproj server was
treated as multimodal and drafting was disabled:

- `tools/server/server-context.cpp:2247` — `skip_draft_mtmd = mctx && has_mtmd` → `n_draft_max = 0`
- `tools/server/server-context.cpp:3049` — prime skipped when `has_mtmd`

The real constraint (per the surrounding comments) is narrower: image **positions**
lack per-token target hidden states, which breaks MTP/NextN prime. Pure-text
prompts have hidden states for every position and are safe to draft. The draft
call at L2253 already passes an empty prompt and reads hidden states from the
target for the mmproj+MTP case, so MTP was designed to coexist with mmproj — the
`has_mtmd` skip was the only blocker.

## Fix

Distinguish "multimodal-capable container" (`has_mtmd`) from "prompt actually
contains media" (new `has_media()`), and gate speculation on the latter.

1. `tools/server/server-common.h` — add accessor:
   ```cpp
   bool has_media() const { return !map_idx_to_media.empty(); }
   ```
   `map_idx_to_media` is the server_tokens map of start-index → image chunk; it is
   non-empty iff the prompt's token list contains real media.

2. `tools/server/server-context.cpp:2247` — `has_mtmd` → `has_media()`.
3. `tools/server/server-context.cpp:3049` — `has_mtmd` → `has_media()`.

`has_mtmd` semantics are unchanged, so the independent cache-reuse and
context-shift asserts (`GGML_ASSERT(!has_mtmd)` at L2194/L2436) and the
`!has_mtmd` cache-reuse gate are unaffected — those features remain correctly
disabled whenever an mmproj is loaded.

## Scope / known limitation

`has_media()` checks the **whole current prompt**. A multi-turn request whose
history includes an image keeps that chunk in `slot.prompt.tokens`, so drafting
stays disabled for the rest of that request/session — conservative and safe
(avoids the hidden-state desync). Fully per-turn re-enable (drafting on a text
turn that follows an image once image positions leave the active window) would
require batch-level media detection and is left as follow-up. The dominant case —
text-only usage on a server that has an mmproj loaded "just in case" — is fully
fixed.

## Files changed

- `tools/server/server-common.h` (+`has_media()`)
- `tools/server/server-context.cpp` (2 gate sites)

## Build

```powershell
cmake --build build --config Release -j 4 --target llama-server
```

(Incremental: only `server-context.cpp` recompiles + relink.)

## Test criteria

Run the server **with** `--mmproj`, `--spec-type mtp`, and a valid `--mtp-head`.

1. **Text-only turn** — send a text prompt with no image.
   - PASS: log does **not** print `skipping speculative prime for multimodal prompt`;
     `statistics mtp` shows `#gen drafts > 0` and `#acc tokens > 0`; eval tok/s
     rises materially above the non-speculative baseline.
   - FAIL: drafts remain 0.
2. **Image turn** — send a prompt containing an image.
   - PASS: log prints `skipping speculative prime for multimodal prompt`; the
     image is still described correctly; `#gen drafts` does not increase for that
     turn (graceful fallback preserved).
3. **No-mmproj regression** — run without `--mmproj`.
   - PASS: behavior unchanged; MTP drafts as before.

---

# SOLVED: MTP crash under `--n-cpu-moe`

**Root cause:** `llama_model::build_graph` (src/llama-model.cpp ~L9576) runs a generic
decode epilogue after the arch builder — `build_pooling`, `build_sampling`,
`build_dense_out`, `res->set_outputs()`. The Gemma 4 MTP graph
(`llm_build_gemma4_mtp`) is self-contained: it publishes its own `t_logits` /
`t_embd` / `t_argmax` and expands them in its ctor. `build_pooling` faults on the
single-token MTP graph (the MTP ubatch sets `output[0]=0` while n_outputs is forced
to 1, so the generic output handling derefs a non-existent output). The reserve build
in `ensure_sched_mtp` survived only because its ubatch sets `output[0]=1`.

Not actually `--n-cpu-moe`-specific — it crashes whenever the MTP graph is built for a
real decode; without `--n-cpu-moe` the 26B just OOMs at load first, so it was never
reached.

**Fix:** skip the entire decode epilogue for `params.gtype == LLM_GRAPH_TYPE_MTP` in
`llama_model::build_graph`. Verified: MTP draft fires, ~75% batch acceptance.

**Remaining cleanup:** the `LLAMA_MTP_TRACE`-gated instrumentation (DEC/SPEC/MTP/GRAPH/
G4MTP/BG traces across llama-context.cpp, common/speculative.cpp, llama-graph.cpp,
models/gemma4-assistant.cpp, llama-model.cpp) can be stripped now or left gated. The
`op_offload=false` for sched_mtp (llama-context.cpp ensure_sched_mtp) is harmless and
can stay or revert to `cparams.op_offload`.

---

## (historical) OPEN BUG: MTP crash under `--n-cpu-moe` — investigation log

When the Gemma 4 target is split with `--n-cpu-moe` (required to fit 26B in 16 GB),
`--spec-type mtp` crashes with a silent access violation on the first draft.

## Localized via LLAMA_MTP_TRACE instrumentation (gated, left in place)

Full call path traced; crash is in:

`llama_context::process_ubatch_mtp` (src/llama-context.cpp ~L1403)
→ `ggml_backend_sched_alloc_graph(sched_mtp, gf)`

Everything before it succeeds: `ensure_sched_mtp` builds + reserves the graph,
the gemma4_assistant graph builds fully (`build_one_step done`), `sched_reserve`
passes. The real-decode `alloc_graph` is what faults.

## Ruled out
- Excluding the CPU backend from `sched_mtp`: illegal — `ggml_backend_sched_new`
  asserts the last backend is CPU (ggml-backend.cpp:1729).
- `op_offload=false` on `sched_mtp`: no effect, still crashes in `alloc_graph`.

## Leading hypothesis (for next session)
`sched_mtp` is a *separate* scheduler. The MTP graph's K/V leaves are **views into
the target's KV buffers** (`mctx_cur->get_k/get_v`, llama-graph.cpp build_attn_mtp
~L2523), which are owned/allocated by the *main* scheduler. `alloc_graph` on
`sched_mtp` doesn't recognize these externally-owned tensors and faults during
backend assignment. Without `--n-cpu-moe` there is 1 device backend and the
assignment path differs, so it never triggers (and that config OOMs at load on
16 GB anyway, so it was never exercised).

Next step: inspect how `sched_mtp` (created in `ensure_sched_mtp`) handles the
target-owned KV view leaves during `alloc_graph` — they likely need to be treated
as pre-allocated (fixed buffer, no realloc). Compare against how the main decode
sched handles the same get_k/get_v views (it shares one scheduler with the buffers,
which is why it works there). A correct fix probably makes the MTP cross-attention
read the target KV through tensors registered with `sched_mtp`, or pins those
leaves' buffers before `alloc_graph`.

## Instrumentation map (all `LLAMA_MTP_TRACE`-gated, safe to keep or strip)
- `[DEC_TRACE]`  src/llama-context.cpp — target decode + embeddings extract
- `[SPEC_TRACE]` common/speculative.cpp — draft submit (h_prev read, async submit)
- `[MTP_TRACE]`  src/llama-context.cpp — decode_mtp_async / ensure_sched_mtp / process_ubatch_mtp / decode_mtp_run
- `[GRAPH_TRACE]` src/llama-graph.cpp — build_attn_mtp get_k/get_v
- `[G4MTP_TRACE]` src/models/gemma4-assistant.cpp — assistant graph build stages
