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

1. [ ] Add `erase_range` + the boundary-snap helper to `server_tokens`; unit-reason
       through: pure text, single image before window, image straddling start,
       image straddling end, image fully inside window.
2. [ ] In the shift loop, replace the `GGML_ABORT` with the std-RoPE allow + snap +
       `erase_range`. Keep `seq_rm`/`seq_add` as-is.
3. [ ] Flip the load-time gate to key on `n_pos_per_embd > 1` instead of `mctx`.
4. [ ] Build `llama-server` (Windows CUDA) and smoke-test (see criteria).
5. [ ] If turbo-K shift produces incoherent output after a shift, fall back to a
       guard that only enables shift when KV types are non-turbo (document it).
6. [ ] Update this doc + `AGENTS.md` with the result.

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
