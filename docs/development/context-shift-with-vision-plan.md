# Plan: context-shift (and cache-reuse) with `--mmproj` loaded

Status: **design / not yet implemented.** This document is the implementation
contract; build the feature from this, then update it with results.

## Goal

Re-enable `llama-server` **context shift** when an `mmproj` (vision projector) is
loaded, for models that use standard RoPE (Gemma 4). Today the server hard-disables
`ctx_shift` (and `cache_reuse`) whenever `mctx != nullptr`, so a long conversation on
a vision-capable server eventually errors/truncates instead of sliding the window.

Out of scope: M-RoPE models (Qwen2-VL-style, `n_pos_per_embd > 1`) — their image
tokens carry 2-D positions that a uniform shift would corrupt; keep them disabled.
Cache-reuse is a possible follow-up but not required for v1.

## Why it's currently disabled

- `tools/server/server-context.cpp` ~L844: at load, if `mmproj` is present and the
  spec/feature isn't multimodal-safe, `ctx_shift` is disabled with a warning.
- `tools/server/server-context.cpp` ~L2162-2166: the shift loop has
  `if (mctx) { GGML_ABORT("not supported by multimodal"); }`.
- The shift rebuilds the token array with `get_text_tokens()` (asserts `!has_mtmd`)
  and raw index math that ignores `map_idx_to_media`.

Root reason: an image occupies multiple KV cells with `LLAMA_TOKEN_NULL` placeholders;
a naive middle-eviction can split an image and desync the media map and RoPE positions.

## Feasibility (already confirmed)

- `llama_kv_cache::get_can_shift()` returns **true** for Gemma 4 (only STEP35 and
  `n_pos_per_embd > 1` return false). So TurboQuant KV + gemma4 supports the
  `seq_rm` + `seq_add` position shift the feature needs.
- `server_tokens::keep_first()` (`tools/server/server-common.cpp` ~L405) is **already
  chunk-aware** — it refuses to split an image and maintains `map_idx_to_media`. That
  is the pattern to extend to middle-eviction.
- `server_tokens::has_media()` already exists (returns `!map_idx_to_media.empty()`).

## Design

### 1. New `server_tokens` method: chunk-aware middle erase

`tools/server/server-common.h` / `.cpp`:

```cpp
// Remove `count` tokens starting at `pos`, shifting later tokens down. Refuses to
// split a media chunk (asserts the erase window aligns to chunk boundaries) and
// decrements map_idx_to_media keys for chunks after the erased range. Returns the
// actual number erased (may differ from `count` if snapped — see helper below).
size_t erase_range(size_t pos, size_t count);
```

Mirror `keep_first`'s media handling: walk `map_idx_to_media`, drop chunks fully
inside `[pos, pos+count)`, and for surviving chunks with key `>= pos+count`, rewrite
the key to `key - count`. Assert no chunk straddles either boundary (the caller must
snap first — see step 2).

### 2. Boundary snapping helper

A small free function (or method) that, given `[n_keep, n_keep + n_discard)`, adjusts
the window so it never starts or ends inside an image chunk:
- if `n_keep` lands mid-chunk, move it back to the chunk start;
- if `n_keep + n_discard` lands mid-chunk, extend `n_discard` to the chunk end
  (discard the whole image) — never partial.

Gemma 4 is 1 token : 1 position (not M-RoPE), so KV cells, token indices, and
positions map 1:1, which keeps the snap arithmetic simple.

### 3. Wire into the shift loop (`server-context.cpp` ~L2150)

- Replace `if (mctx) GGML_ABORT(...)` with: allow when
  `model n_pos_per_embd == 1` (standard RoPE); keep the abort for M-RoPE.
- Snap `n_keep` / `n_discard` (step 2).
- Keep the existing `llama_memory_seq_rm(n_keep, n_keep+n_discard)` +
  `llama_memory_seq_add(...)` — they already do the KV-level shift and work for gemma4.
- Replace the `get_text_tokens()` + raw-array rebuild (the `GGML_ASSERT(!has_mtmd)`
  branch) with `slot.prompt.tokens.erase_range(n_keep, n_discard)` (step 1).

### 4. Re-enable the load-time gate (`server-context.cpp` ~L844)

Condition the disable on `n_pos_per_embd > 1` rather than on `mctx != nullptr`. For
standard-RoPE multimodal models, leave `ctx_shift` enabled.

## Files touched

- `tools/server/server-common.h` — declare `erase_range` (+ snap helper if a method).
- `tools/server/server-common.cpp` — implement `erase_range` + snap helper.
- `tools/server/server-context.cpp` — load-time gate (~L844), shift loop (~L2150-2205).

## Implementation checklist (strict order)

1. [x] Add `erase_range` + `snap_past_media` to `server_tokens`
       (`tools/server/server-common.{h,cpp}`).
2. [x] In the shift loop (`server-context.cpp` ~L2167), remove the `GGML_ABORT`, snap
       `n_keep`/`n_discard` to image edges when media present, route the token update
       through `erase_range` for mmproj prompts (original fast path kept for non-mmproj).
       `seq_rm`/`seq_add` unchanged.
3. [x] Flip the load-time gate (~L842) to disable only for `LLAMA_ROPE_TYPE_MROPE`
       (standard-RoPE multimodal stays enabled).
4. [ ] Build `llama-server` (Windows CUDA) and smoke-test (see criteria).
5. [ ] If turbo-K shift produces incoherent output after a shift, fall back to a
       guard that only enables shift when KV types are non-turbo (document it).
6. [ ] Update this doc + `AGENTS.md` with the result.

> Items 1-3 implemented and built. The gate uses
> `llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE`; added a guard that fails the
> turn if the discard window is entirely one image (`n_discard <= 0` after snap).

## Results (tested)

- **f16 / q8 KV: WORKS.** In-place shift on a standard-RoPE multimodal model (Gemma 4 +
  `--mmproj`) is coherent across the shift. Log shows `slot context shift, n_keep=…
  n_discard=…`; `erase_range` + `seq_rm`/`seq_add` are correct. Feature complete for
  non-turbo KV.
- **TurboQuant KV (`-ctk turbo4 -ctv turbo2`): garbles at the shift point.** Root cause:
  `seq_add` re-applies RoPE to the K cache (K-shift), and there is no K-shift kernel for
  TurboQuant's WHT-rotated quantized K. (`get_can_shift()` returns true on arch alone, so
  the shift is attempted and corrupts.)

## Decision for turbo KV: reprefill-on-overflow (chosen)

Rather than disable shift on turbo (defeats the daily driver) or patch the WHT-domain
K-shift kernel (deep, multi-backend, high correctness risk — RoPE doesn't commute with
WHT), make the overflow path **re-encode** the retained tokens when the KV can't
K-shift. This adds the missing functionality, is KV-type-agnostic, and is the mechanism
llama.cpp used before in-place K-shift existed. Cost: an occasional ~2-4 s reprefill at
the overflow point (rare, especially at large `--ctx-size`).

### Reprefill design (next implementation)

In the shift loop, branch on `llama_memory_can_shift(llama_get_memory(ctx))`:
- **can shift (f16/q8):** keep the current in-place path (done).
- **cannot shift (turbo):** instead of `seq_add`:
  1. snap `n_keep` to an image boundary (`snap_past_media`);
  2. `erase_range(n_keep, n_discard)` on `slot.prompt.tokens` (chunk-aware, already built);
  3. `llama_memory_seq_rm(mem, slot.id, n_keep, -1)` — drop KV from `n_keep` onward
     (KV for `[0, n_keep)` is kept as-is, no shift needed);
  4. re-enter prompt processing so the retained suffix is re-encoded at contiguous
     positions `n_keep…`. **Hook TBD:** set slot state back to prompt-processing and the
     processed-length to `n_keep` (study `SLOT_STATE_PROCESSING_PROMPT` / how the slot
     syncs `slot.prompt.tokens` vs KV via common-prefix; the suffix beyond the kept KV
     must be re-decoded).

### Turbo K-shift kernel — deferred (future optimization)

The only thing reprefill gives up is the occasional reprefill pause. If that ever
matters, implement RoPE-delta application on TurboQuant K (dequant → inverse-WHT →
rotate by Δpos → WHT → requant, per shifted cell), and flip `get_can_shift()` to allow
turbo. High effort, not scheduled.

### Checklist addendum

7. [x] **Model shiftability correctly at the KV layer** (the root fix): `get_can_shift()`
       (`src/llama-kv-cache.cpp`) now returns false for turbo K types (in addition to
       STEP35 / M-RoPE). Removed the server's "multimodal disables ctx_shift" gate
       (`server-context.cpp` ~L842) — shiftability is now decided solely by
       `llama_memory_can_shift()` (existing check ~L864). Net: f16/q8 multimodal
       (Gemma 4 + `--mmproj`) shifts in-place (previously fully disabled); turbo / M-RoPE
       fall back to graceful truncation (no corruption).
8. [x] Reprefill branch for non-K-shiftable KV (turbo / M-RoPE). The shift loop now
       branches on `llama_memory_can_shift()`: **in-place** when true (f16/q8),
       **reprefill** otherwise. Reprefill keeps the head KV `[0, n_keep)`, drops the rest
       (`seq_rm n_keep..-1`), trims tokens with `erase_range`, and re-encodes the retained
       recent window at contiguous positions via a normal `llama_decode` loop (works for
       any KV type — turbo K recomputed correctly). The resume point is clean because the
       shift loop runs before the generating batch-add (`pos_next()` reflects the new end).
       `ctx_shift` is no longer disabled at load for non-K-shiftable KV (only `cache_reuse`).
       Guard: if the retained window holds an image (`LLAMA_TOKEN_NULL`), bail gracefully
       (re-encoding an image needs the mtmd path — future work).
9. [x] Build + tested:
       - f16/q8 multimodal shift coherent (`mode = shift`).
       - **turbo** overflow now CONTINUES via reprefill (`mode = reprefill`) — generated
         well past `n_ctx` (2461 tokens after a 4096 ctx), coherent. The previously-
         garbling case is fixed.
       - Required a third fix: `common_init_from_params` (`common/common.cpp:1327`)
         auto-disables `ctx_shift` for non-K-shiftable KV; the server now captures the
         user's `--context-shift` request and re-asserts it after init (it has the
         reprefill fallback that llama-cli lacks).

## FINAL STATUS: complete

Context shift works with vision and with TurboQuant KV. f16/q8 use in-place K-shift;
turbo (and any non-K-shiftable KV) use reprefill. Only remaining limitation: a retained
window containing an image falls back to graceful truncation (re-encoding an image needs
the mtmd path) — future work, low priority.

## Test criteria

Run with `--mmproj`, a small `--ctx-size` (e.g. 4096) to force shifts quickly, gemma4.

- **T1 text-only overflow:** feed a conversation past `n_ctx`. PASS = it slides
  (logs `slot context shift, n_keep=… n_discard=…`) and keeps generating coherently;
  no `GGML_ASSERT`, no abort.
- **T2 image then long text:** send an image, then enough text to trigger a shift.
  PASS = shift fires, snaps around the image (whole image kept or whole image
  discarded — never split), output stays coherent, image still referenced correctly
  while it remains in window.
- **T3 image inside the discard window:** arrange the oldest image to fall in
  `[n_keep, n_keep+n_discard)`. PASS = the whole image is evicted, `map_idx_to_media`
  has no dangling entry, no crash.
- **T4 M-RoPE still disabled:** on a Qwen2-VL-style model, confirm `ctx_shift` stays
  disabled (gate keys on `n_pos_per_embd`).
- **T5 turbo KV coherence:** with `-ctk turbo4 -ctv turbo2`, confirm output after a
  shift is coherent (validates the K-shift on quantized cells).

## Risks

- **Turbo-K shift correctness** (T5) — the WHT-rotated K under `seq_add` is the least-
  trodden path; if it desyncs, gate shift to non-turbo KV (checklist 5).
- **Snap arithmetic off-by-one** at chunk boundaries — covered by T2/T3.
- Effort estimate: ~2-3 focused days, medium complexity, low-to-medium risk (the KV
  shift capability is already present for gemma4).
