# Context shift with vision (and all KV types)

Upstream/parent hard-disabled `ctx_shift` whenever an `mmproj` was loaded, and the
in-place K-shift only works on K-shiftable KV (f16/q8) — TurboQuant KV corrupts under a
position shift. This fork makes context management work with vision **and** any KV type.

## Behavior

- **f16 / q8 KV** → in-place K-shift (fast). Works with multimodal: the discard window is
  snapped to whole-image boundaries so an image is never split.
- **TurboQuant / M-RoPE KV** (can't K-shift) → transactional **reprefill** fallback:
  remove the selected middle window and replay the complete retained text/media sequence
  at contiguous positions. Turbo K and M-RoPE coordinates are recomputed rather than
  shifted in an incompatible representation. The original target/draft state is restored
  if any replay step fails.
- Retained **images** are replayed through the normal mtmd vision path. Media boundaries
  are snapped before eviction and the retained chunk metadata stays synchronized with the
  serialized token stream.
- DFlash/DSpark always use reprefill during target context shift. Their shiftable Q8 draft
  KV lives in a private sliding position domain, so a target-absolute erase range would be
  wrong after the first rollover even though the underlying cache supports RoPE shift.
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
  - Reprefill checkpoints target bytes, draft bytes, and speculative private-window state
    before clearing either context. Text and mtmd chunks are then replayed through their
    regular decode paths, including DSpark's target-to-local mapping.
- `src/llama-kv-cache.cpp`
  - `get_can_shift()` returns false for turbo K types (`GGML_TYPE_TURBO2_0/3_0/4_0`), in
    addition to STEP35 / M-RoPE. This is the true "can do in-place K-shift" signal.

## Log signals

`slot context shift, n_keep=… n_discard=… mode = shift|reprefill`. `mode = reprefill`
confirms the turbo path; the snap shows up as `n_keep` being pushed past an image.

## Provenance and remaining optimization

The media-boundary snapping follows the upstream chunk-aware context-shift work referenced
at the implementation site (`ggml-org/llama.cpp` PR #24786). Transactional retained-media
replay and private DSpark-window coordination are local Atomic adaptations. A native
TurboQuant K-shift kernel—dequantize, inverse WHT, apply the RoPE delta, WHT, and
requantize—could reduce the occasional reprefill pause, but is an optimization rather than
a correctness requirement now that text and media replay are supported.
