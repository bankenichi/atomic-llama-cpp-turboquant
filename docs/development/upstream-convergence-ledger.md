# Upstream convergence ledger

This ledger tracks the semantic disposition of owner-relevant behavior during
the 2026-08 upstream and Prism convergence. `Pending` means that a similarly
named upstream feature has not yet been proven equivalent.

| Capability | Current fork source | Upstream observation at `9a286ac` | Decision/status | Required proof |
|---|---|---|---|---|
| Gemma 4 assistant architecture | `src/models/gemma4-assistant.cpp`, MTP code and `MTP.md` | Upstream has a dedicated `LLM_ARCH_GEMMA4_ASSISTANT` model and synchronous `draft-mtp` context | Adopt upstream architecture; exact fixture validation pending | Exact assistant loads; text drafts; acceptance/rollback; no output-epilogue crash |
| MTP with CPU-offloaded MoE | `src/llama-model.cpp`, `src/llama-context.cpp` | Upstream moved MTP into a separate draft context | Augment: force `cparams.op_offload = false` for MTP draft contexts | `--n-cpu-moe` with MTP completes prefill and generation |
| MTP with loaded mmproj | server `has_media()` gating | Upstream speculative/media plumbing changed; the fork's real-media accessor and server gates survived the merge | Retain fork gating; focused fixture validation pending | Text turn drafts with mmproj loaded; image turn falls back safely |
| Qwen NextN | `common/speculative.cpp`, Qwen model overrides | Upstream has synchronous multi-step MTP/NextN staging and integrated Qwen 3.5 graph implementations | Adopt upstream synchronous implementation; audit the legacy opt-in async mode separately | Multi-step drafts up to configured maximum; optional async remains isolated |
| Paired NextN context shift | `llama_context_nextn_seq_rm`, server shift paths | Not yet proven upstream-equivalent | Reimplement centrally if upstream lacks it | Target/draft positions equal after remove, shift, cache relocation, reprefill |
| Vision-aware context shift | server token/media helpers and KV cache | Not found under the local helper names upstream | Reimplement on modern server APIs | Image boundaries never split; shiftable and reprefill paths are transactional |
| Template `media_marker` fallback | `common/chat.cpp` | Upstream recognizes `media_marker`; exact neither-capability fallback not yet proven | Pending focused diff/test | One image produces one marker with the failing Gemma 4 coder template |
| TurboQuant KV | `ggml` CPU/CUDA/Metal/Vulkan/HIP and llama graph/cache | No upstream TurboQuant types | Reimplemented on the current graph/cache APIs; CUDA exact-model and vision-shift acceptance pass | CUDA SET_ROWS 3/3; WHT 6/6; exact combined server pass |
| TQ3/TQ4 weights | `ggml` CPU/CUDA/Metal/Vulkan and quantizer | No upstream TQ3/TQ4 types | Private type registration and CPU fallback traits restored with non-colliding runtime IDs | Existing private types enumerate, quantize, and build; optimized paths remain backend-specific |
| Expert prefetch overlap | `ggml/src/ggml-backend.cpp` | Environment gate not present upstream | Reimplement on current scheduler | Default unchanged; opt-in overlaps transfers and preserves output |
| mmap host registration | `src/llama-mmap.*`, model loader | Upstream exposes the CUDA registration proc but does not wire retained mmap ranges into it | Augment: restored loader wiring and Windows page-aligned registration | Windows mmap pages register/unregister only when opted in |
| MSVC ASM_MASM fix | `ggml/CMakeLists.txt` | Current upstream build configures successfully with VS/v143 and CUDA | Existing fork guard retained; clean configure/build passed | Clean Windows CUDA configure/build |
| OpenSSL const fix | `vendor/cpp-httplib/httplib.cpp` | The converged build uses static BoringSSL and the modern vendor tree | Existing compatibility change retained; full TLS tool build passed | Full tool build with configured TLS backend |
| Windows CUDA release | `.github/workflows/windows-cuda.yml` | Upstream has many workflows | Keep private single workflow and forward-port commands | Full tools, `sm_120`, vision, BoringSSL, CUDA DLL verification |
| Expert mmap/prefetch docs | development docs | No direct upstream equivalent | Preserve and update | Commands and gates match final code |
| Gemma 4 12B encoder-free vision | model/converter/mtmd changes | Upstream vision evolved substantially | Pending exact artifact test | Existing 12B target and corrected mmproj load and infer |
| Context shift with all KV types | server plus `llama-kv-cache.cpp` | Upstream context code changed | Reimplement required semantics | f16/q8 in-place; Turbo/non-shiftable reprefill; media-safe behavior |

## External compatibility additions

| Capability | Source | Status | Required proof |
|---|---|---|---|
| Upstream group-64 `Q2_0` type 42 | upstream | Merged unchanged; CUDA backend operation matrix passes | Standard upstream Q2 model loads and backend tests pass |
| Modern Prism group-128 type 142 | Prism modern branch | Integrated across CPU/CUDA quantization and matrix paths | CUDA backend tests and exact group-128 target inference pass |
| Legacy Bonsai group-128 type 42 | exact HF files / Prism legacy branch | Physical-layout preflight normalizes coherent legacy files before allocation | Exact target loads without rewrite; ambiguous/mixed encodings reject |
| Modern upstream DSpark | upstream | Merged; sidecar loader opens the computed draft path | Current DFlash/DSpark tests and loader behavior retained |
| Legacy Bonsai `dspark` | Prism legacy branch | Compatibility adapter integrated through the current DSpark runtime | Exact Q4_1 sidecar loads and produces accepted drafts |
| Bonsai vision | exact HF target and mmproj | Integrated without weakening real-media speculative gates | Exact text/image/text and forced Turbo context-shift session pass |

Rows are updated in the same commits that make or verify the associated decision.

## Upstream parity checkpoint

The pinned upstream merge was configured in a fresh `build-convergence` tree with
VS/v143, CUDA, and `sm_120a`. The complete tool suite, including `llama-server`
and all `mtmd` adapters, built successfully. CTest passed 46/46 tests; notably,
the full CUDA backend-operation matrix and refreshed quant-type selection tests
both pass, improving on the pre-convergence baseline. The exact Bonsai base,
DSpark sidecar, and mmproj subsequently passed focused target-only, speculative,
vision, and transactional context-shift proofs. See
[`bonsai-dspark-turboquant.md`](bonsai-dspark-turboquant.md) for artifacts,
hashes, commands, and the final acceptance matrix.
