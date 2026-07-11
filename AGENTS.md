# AGENTS.md — orientation for agentic work on this repo

This is a **personal fork** of `AtomicBot-ai/atomic-llama-cpp-turboquant` (itself a fork
of `ggml-org/llama.cpp`), maintained by a single owner for private use. No upstream PRs,
no external contributors. Optimize for the owner's setup, not upstream generality.

## Owner's stack (what to optimize for)

- **OS / GPU:** Windows, single NVIDIA RTX 5080 (Blackwell, `sm_120`), 16 GB VRAM.
- **Primary model:** Gemma 4 26B-A4B (MoE) Q4_K_M + vision `mmproj` + Gemma 4 assistant MTP head.
- Also runs Qwen 3.6 MoE with NextN speculative decoding.
- Builds locally (Visual Studio) and via CI (below).

## Local modifications — do NOT regress these

Authoritative writeup: [`docs/development/mtp-cpumoe-vision-fixes.md`](docs/development/mtp-cpumoe-vision-fixes.md).

1. `tools/server/server-common.h` — `has_media()` accessor (real media vs `has_mtmd`).
2. `tools/server/server-context.cpp` — speculative draft + prime gates keyed on `has_media()`,
   not mmproj presence; prime passes empty tokens for mtmd-safe specs.
3. `src/llama-model.cpp` — skip the decode epilogue (`build_pooling`/`build_sampling`/
   `build_dense_out`/`set_outputs`) for `LLM_GRAPH_TYPE_MTP` graphs. **The load-bearing MTP
   crash fix.**
4. `src/llama-context.cpp` — `sched_mtp` created with `op_offload = false`.
5. Build fixes in-tree: `ggml/CMakeLists.txt` (`ASM_MASM` under MSVC) and
   `vendor/cpp-httplib/httplib.cpp` (`const_cast` for OpenSSL 3.x `X509_get_*_name`).
6. **Context shift with vision + all KV types** (`server-common.{h,cpp}`,
   `server-context.cpp`, `src/llama-kv-cache.cpp`): in-place K-shift for f16/q8 with
   chunk-aware `erase_range` + image-boundary `snap_past_media`; reprefill fallback for
   turbo / M-RoPE that re-encodes the retained window. `get_can_shift()` reports turbo K as
   non-shiftable; the server re-asserts `ctx_shift` over the common-layer auto-disable.
   See [`docs/development/context-shift-with-vision.md`](docs/development/context-shift-with-vision.md).
7. **Vision with templates that ignore the internal `media_marker` part type**
   (`common/chat.cpp`, `render_message_to_json`): when a chat template advertises support for
   *neither* string nor typed content (caps detection failed — the "Neither string content nor
   typed content" warning), fall back to **string** content (marker preserved inline) instead of
   typed parts. Otherwise such a template drops the server's `media_marker` part, leaving 0 image
   markers for an attached image and `mtmd` aborts with `number of bitmaps (1) does not match
   number of markers (0)`. Hit with the Gemma 4 12B coder finetune; fix is general. Full
   12B-vision writeup (encoder-free `gemma4uv`, the PR#24118 conversion bug, the mmproj swap):
   [`docs/development/gemma4-12b-vision.md`](docs/development/gemma4-12b-vision.md).
8. **Expert-weight prefetch overlap** (`ggml/src/ggml-backend.cpp`): ported from
   `thecodacus/llama.cpp` (`fable5/prefetch-experts`). During prefill, upload offloaded
   `MUL_MAT_ID` expert tensors through a second same-device backend instance (own stream) into
   rotating event-ordered staging slots, so upload N+1 overlaps compute N instead of blocking on
   the routing-id readback. Opt-in: env `GGML_SCHED_PREFETCH_EXPERTS=1` (value = slot count,
   default 3 = one MoE layer) **and** `op_offload`/`--n-cpu-moe` active. Default behavior
   unchanged. Speeds the `n-cpu-moe` prefill both stacks bottleneck on.
9. **Pin mmap-backed CPU weights** (`src/llama-mmap.{cpp,h}`, `src/llama-model-loader.cpp`):
   ported from the same branch. After load, `cudaHostRegister` the mmap pages of the retained
   (CPU-resident) weights via the backend's `ggml_backend_register_host_buffer` proc — previously
   dead code, now wired up — so H2D expert streaming uses DMA. Opt-in: env
   `GGML_CUDA_REGISTER_HOST=1`. **Fork addition beyond upstream:** a `_WIN32` branch in
   `register_host` (GetSystemInfo page size) so it actually pins on this Windows build; the
   original was POSIX-only. Both ports documented in
   [`docs/development/thecodacus-prefetch-mmap-port.md`](docs/development/thecodacus-prefetch-mmap-port.md).
10. **NextN (Qwen) speculative decoding: synchronous multi-step by default**
    (`common/speculative.cpp`, `common_speculative_state_nextn`). The class already contained a
    correct synchronous multi-step draft (`for k < n_max` AR loop — equivalent to the reference
    MTP impl), but defaulted to an async "depth-2" worker that hardcaps drafting at **1 token/round**,
    leaving Qwen decode ~30% slower than a multi-step build. Flipped `pipeline_enabled` to default
    **false** so the sync path (drafts up to `--draft-max` tokens/round) is the default; the async
    worker is now opt-in via `LLAMA_NEXTN_ASYNC=1` (a NextN-only env var, deliberately distinct
    from Gemma MTP's `LLAMA_PIPELINE_DEPTH2`). **NextN-only — the Gemma MTP path
    (`common_speculative_state_mtp`, `--spec-type mtp`) is a separate class and is untouched.**
    Launcher `-Qwen` now passes `--draft-max 2`.

Net effect: Gemma 4 MTP speculative decoding runs with `--n-cpu-moe` **and** `--mmproj`
(vision) on a single server, drafting on text turns and falling back cleanly on image turns;
context shift works with vision on every KV type; and prefill throughput on CPU-offloaded MoE
can be raised via the two opt-in env gates above.

## Building (Windows + CUDA)

- **Toolchain gotcha:** CUDA's `nvcc` can crash against bleeding-edge MSVC (VS 2026). Use the
  **VS 2022 (v143) toolset** for CUDA builds. CI uses `windows-2022` + CUDA 13.1, target `sm_120`.
- Local: `cmake -B build -DGGML_CUDA=ON` then `cmake --build build --config Release --target llama-server`.
- CI: `.github/workflows/windows-cuda.yml` builds the **full tool suite** (CUDA, `sm_120`,
  vision; static BoringSSL so no external OpenSSL DLLs), bundles the CUDA runtime DLLs
  (`cudart`/`cublas`/`cublasLt`) so the zip is self-contained (only the NVIDIA driver + VC++
  Redist needed), and publishes to the rolling **`latest`** GitHub Release on each push to
  `main` / manual dispatch. It fails the build if the CUDA DLLs aren't bundled. It is the
  **only** workflow — other platforms build manually.

## Running & performance (hard-won; details in the fixes doc)

- **Launcher:** `run-gemma.ps1` (gitignored, machine-specific paths). Default = tuned daily driver.
- **Tuned config:** no MTP, `--n-cpu-moe 14`, `--ctx-size 0`, vision on, TurboQuant KV
  (`-ctk turbo4 -ctv turbo2`), `-fa on`. ≈53 tk/s short prompts, ≈40 at 60k context.
- **VRAM ceiling: keep dedicated ≤ ~15 GB.** Past that, per-decode activation scratch spills to
  shared memory and throughput craters (moe 12 → <5 tk/s).
- **`--n-cpu-moe` is the dominant gen-speed lever** (~420 MB per layer; first-N layers). Lower it
  until you hit the 15 GB wall.
- **MTP is ~break-even on this CPU-offloaded MoE** — memory-bandwidth bound on CPU-resident experts
  (verified: 0 vs 654 drafts → identical tk/s). Reclaiming the draft head's VRAM for a real expert
  layer beats running MTP here. MTP only pays off when most experts are GPU-resident.
- **Disable thinking** (e.g. SillyTavern): `--reasoning off`. Connect ST via **Chat Completion** to
  `/v1` so the server applies the model's (nonstandard `<|turn>`) template.

## Docs map

- `docs/development/performance-features.md` — **runtime perf toggles + usage** (prefetch,
  mmap host-pin, NextN sync multi-step) with a quick-reference table and the launcher flag map.
- `docs/development/mtp-cpumoe-vision-fixes.md` — fixes + full tuning notes (**authoritative**).
- `docs/development/thecodacus-prefetch-mmap-port.md` — prefetch/mmap-pin port rationale + patches.
- `docs/development/gemma4-12b-vision.md` — Gemma 4 12B encoder-free vision.
- `MTP.md`, `NEXTN.md`, `docs/speculative.md` — fork speculative-decoding internals.
- `README.md` — trimmed for this fork; headline features + upstream usage reference.
- `NOTICE.md` — lineage / attribution.

## Conventions

- Keep `LICENSE` (MIT) and `NOTICE.md` intact.
- Document non-trivial changes under `docs/development/`.
- No CI beyond the single Windows-CUDA build.
