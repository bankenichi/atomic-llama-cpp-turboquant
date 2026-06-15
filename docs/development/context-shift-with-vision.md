# Context shift with vision (and all KV types)

Upstream/parent hard-disabled `ctx_shift` whenever an `mmproj` was loaded, and the
in-place K-shift only works on K-shiftable KV (f16/q8) — TurboQuant KV corrupts under a
position shift. This fork makes context management work with vision **and** any KV type.

## Behavior

- **f16 / q8 KV** → in-place K-shift (fast). Works with multimodal: the discard window is
  snapped to whole-image boundaries so an image is never split.
- **TurboQuant / M-RoPE KV** (can't K-shift) → **reprefill** fallback: keep the head KV,
  drop the rest, and re-encode the retained recent window at contiguous positions via a
  normal forward pass (turbo K is recomputed correctly). Costs an occasional re-encode
  pause at the overflow point.
- A retained window that still contains an **image** can't be reprefilled (would need the
  mtmd vision path) → the turn fails gracefully with a clear error instead of corrupting.
- A single prompt larger than `n_ctx` is rejected (unchanged, correct).

## Implementation (key pieces)

- `tools/server/server-common.{h,cpp}`
  - `snap_past_media(pos)` — push a boundary that lands inside an image forward to the
    image's end (never split an image).
  - `erase_range(pos, count)` — chunk-aware middle-eviction; shifts later tokens down and
    keeps `map_idx_to_media` in sync.
- `tools/server/server-context.cpp`
  - The multimodal `ctx_shift` disable at load was **removed** — shiftability is decided by
    the KV layer, not by "is multimodal".
  - The shift loop snaps the window to image boundaries, then branches on
    `llama_memory_can_shift()`: in-place K-shift if true, reprefill if false.
  - After `common_init_from_params`, the server **re-asserts** the user's `--context-shift`
    request (that call auto-disables it for non-K-shiftable KV; the server has the reprefill
    fallback that `llama-cli` lacks).
- `src/llama-kv-cache.cpp`
  - `get_can_shift()` returns false for turbo K types (`GGML_TYPE_TURBO2_0/3_0/4_0`), in
    addition to STEP35 / M-RoPE. This is the true "can do in-place K-shift" signal.

## Log signals

`slot context shift, n_keep=… n_discard=… mode = shift|reprefill`. `mode = reprefill`
confirms the turbo path; the snap shows up as `n_keep` being pushed past an image.

## Future work (low priority)

Reprefill of a retained **image** would require re-running it through the mtmd vision
pipeline at the new position. Until then, an image in the recent window on non-K-shiftable
KV falls back to a graceful error. (Alternatively, a TurboQuant K-shift kernel — dequant →
inverse-WHT → RoPE Δ → WHT → requant — would enable true in-place shift for turbo, but
that's a deep, multi-backend effort and reprefill covers the need.)
