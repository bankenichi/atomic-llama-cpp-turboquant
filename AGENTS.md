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

Net effect: Gemma 4 MTP speculative decoding runs with `--n-cpu-moe` **and** `--mmproj`
(vision) on a single server, drafting on text turns and falling back cleanly on image turns;
context shift works with vision on every KV type.

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

- `docs/development/mtp-cpumoe-vision-fixes.md` — fixes + full tuning notes (**authoritative**).
- `MTP.md`, `NEXTN.md`, `docs/speculative.md` — fork speculative-decoding internals.
- `README.md` — trimmed for this fork; headline features + upstream usage reference.
- `NOTICE.md` — lineage / attribution.

## Conventions

- Keep `LICENSE` (MIT) and `NOTICE.md` intact.
- Document non-trivial changes under `docs/development/`.
- No CI beyond the single Windows-CUDA build.
