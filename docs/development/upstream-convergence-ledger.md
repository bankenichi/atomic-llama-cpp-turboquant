# Upstream convergence ledger

This ledger tracks the semantic disposition of owner-relevant behavior during
the 2026-08 upstream and Prism convergence. `Pending` means that a similarly
named upstream feature has not yet been proven equivalent.

| Capability | Current fork source | Upstream observation at `9a286ac` | Decision/status | Required proof |
|---|---|---|---|---|
| Gemma 4 assistant architecture | `src/models/gemma4-assistant.cpp`, MTP code and `MTP.md` | Upstream has `LLM_ARCH_GEMMA4_ASSISTANT` and `draft-mtp` | Pending detailed equivalence audit | Exact assistant loads; text drafts; acceptance/rollback; no output-epilogue crash |
| MTP with CPU-offloaded MoE | `src/llama-model.cpp`, `src/llama-context.cpp` | Upstream implementation exists, local scheduler/output fixes not yet compared | Pending; retain behavior if absent | `--n-cpu-moe` with MTP completes prefill and generation |
| MTP with loaded mmproj | server `has_media()` gating | Upstream speculative/media gating has changed | Pending; real-media semantics are mandatory | Text turn drafts with mmproj loaded; image turn falls back safely |
| Qwen NextN | `common/speculative.cpp`, Qwen model overrides | Upstream has modern NextN/MTP staging | Adopt upstream architecture, audit local sync multi-step behavior | Multi-step drafts up to configured maximum; optional async remains isolated |
| Paired NextN context shift | `llama_context_nextn_seq_rm`, server shift paths | Not yet proven upstream-equivalent | Reimplement centrally if upstream lacks it | Target/draft positions equal after remove, shift, cache relocation, reprefill |
| Vision-aware context shift | server token/media helpers and KV cache | Not found under the local helper names upstream | Reimplement on modern server APIs | Image boundaries never split; shiftable and reprefill paths are transactional |
| Template `media_marker` fallback | `common/chat.cpp` | Upstream recognizes `media_marker`; exact neither-capability fallback not yet proven | Pending focused diff/test | One image produces one marker with the failing Gemma 4 coder template |
| TurboQuant KV | `ggml` CPU/CUDA/Metal/Vulkan/HIP and llama graph/cache | No upstream TurboQuant types | Reimplement/forward-port completely | Existing quant tests, FA paths, KV serialization, context tests |
| TQ3/TQ4 weights | `ggml` CPU/CUDA/Metal/Vulkan and quantizer | No upstream TQ3/TQ4 types | Reimplement with non-colliding IDs and legacy GGUF compatibility | Existing files load; CPU/CUDA numerical and MoE tests pass |
| Expert prefetch overlap | `ggml/src/ggml-backend.cpp` | Environment gate not present upstream | Reimplement on current scheduler | Default unchanged; opt-in overlaps transfers and preserves output |
| mmap host registration | `src/llama-mmap.*`, model loader | Upstream exposes CUDA host registration proc; loader wiring and Windows handling not yet equivalent | Augment if needed | Windows mmap pages register/unregister only when opted in |
| MSVC ASM_MASM fix | `ggml/CMakeLists.txt` | Pending after upstream configure | Retain only if current VS/v143 configure requires it | Clean Windows CUDA configure/build |
| OpenSSL const fix | `vendor/cpp-httplib/httplib.cpp` | Vendor tree changed upstream | Retain only if current dependency still requires it | Full tool build with configured TLS backend |
| Windows CUDA release | `.github/workflows/windows-cuda.yml` | Upstream has many workflows | Keep private single workflow and forward-port commands | Full tools, `sm_120`, vision, BoringSSL, CUDA DLL verification |
| Expert mmap/prefetch docs | development docs | No direct upstream equivalent | Preserve and update | Commands and gates match final code |
| Gemma 4 12B encoder-free vision | model/converter/mtmd changes | Upstream vision evolved substantially | Pending exact artifact test | Existing 12B target and corrected mmproj load and infer |
| Context shift with all KV types | server plus `llama-kv-cache.cpp` | Upstream context code changed | Reimplement required semantics | f16/q8 in-place; Turbo/non-shiftable reprefill; media-safe behavior |

## External compatibility additions

| Capability | Source | Status | Required proof |
|---|---|---|---|
| Upstream group-64 `Q2_0` type 42 | upstream | Pending merge | Standard upstream Q2 model loads and backend tests pass |
| Modern Prism group-128 type 142 | Prism modern branch | Pending integration | CPU/CUDA quant, `get_rows`, MMVQ, MMQ pass |
| Legacy Bonsai group-128 type 42 | exact HF files / Prism legacy branch | Pending physical-layout preflight | Exact target and drafter load without rewrite; ambiguous files reject |
| Modern upstream DSpark | upstream | Pending merge | Upstream DSpark tests and sample model continue to work |
| Legacy Bonsai `dspark` | Prism legacy branch | Pending port | Exact Q4_1 sidecar drafts accepted tokens with greedy parity |
| Bonsai vision | exact HF target and mmproj | Pending model integration | Text/image/text session and context-shift gates pass |

Rows are updated in the same commits that make or verify the associated decision.
