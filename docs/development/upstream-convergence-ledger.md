# Upstream convergence ledger

This ledger tracks the semantic disposition of owner-relevant behavior during
the 2026-08 upstream and Prism convergence. `Pending` means that a similarly
named upstream feature has not yet been proven equivalent.

| Capability | Current fork source | Upstream observation at `9a286ac` | Decision/status | Required proof |
|---|---|---|---|---|
| Gemma 4 assistant architecture | `src/models/gemma4-assistant.cpp`, MTP code and `MTP.md` | Upstream has a dedicated `LLM_ARCH_GEMMA4_ASSISTANT` model and synchronous `draft-mtp` context | Adopted upstream architecture while retaining the legacy raw-prefix and tensor-name adapter; exact fixture validated | Exact assistant loads; text and image-conditioned turns draft; acceptance/rollback; no output-epilogue crash |
| MTP with CPU-offloaded MoE | `src/llama-model.cpp`, `src/llama-context.cpp` | Upstream moved MTP into a separate draft context | Augment: force `cparams.op_offload = false` for MTP draft contexts | `--n-cpu-moe` with MTP completes prefill and generation |
| Media-aware MTP/NextN | `common/speculative.cpp`, server policy coordinator | Upstream provides the target-feature MTP driver; Atomic previously distinguished loaded mmproj from real media at `ead60fb4f` | Local adaptation: process architecture-defined target hidden rows and preserve/reuse the post-media boundary | Gemma assistant and Qwen embedded NextN accept drafts during image answers and retained-image follow-ups |
| Media-aware EAGLE3 | upstream EAGLE3 plus Atomic compact-position adaptation | Upstream EAGLE3 skips embedding batches | Local adaptation: retain the final target-derived image feature at a compact private boundary and roll back provisional KV in that domain | Gemma 4 EAGLE3 accepts multi-token drafts during image answers and cached full-history follow-ups |
| Media-aware draft-simple and n-grams | `common/speculative.cpp` | Upstream standalone draft cannot decode a foreign mmproj embedding; n-grams have no model KV | Local adaptation: compact text projection for draft-simple; explicit stateless policy for all n-grams | draft-simple and every maintained n-gram mode complete image/full-history requests; useful fixtures record accepted drafts |
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
| Bonsai vision + DFlash/DSpark | exact HF target, mmproj, and Prism sidecar | Integrated with M-RoPE-aware target-feature pooling instead of a request-wide media veto | Exact image answer and retained-image follow-up accept DFlash/DSpark drafts; target-only vision remains intact |

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

## 2026-11 convergence (upstream `f4f6f2e7d`, Prism `d12ca13bb`)

Details, merge points and validation: [`upstream-convergence-2026-11.md`](upstream-convergence-2026-11.md).

| Capability | Upstream observation at `f4f6f2e7d` | Decision/status | Proof |
|---|---|---|---|
| Private GGUF type IDs | Upstream claims raw 43 (`F8_E4M3`) | Adopt upstream 43; normalize only legacy 45/46; reject 44 | `test-gguf` raw 43/45/46/44 cases |
| Prism `Q2_0_G128` | Prism renamed it `PQ2_0` (same ID 142, same layout) | Adopt Prism's name; legacy raw 42 detection unchanged | `test-backend-ops` PQ2_0 cases; Bonsai acceptance |
| DFlash family | Split into per-head model files and speculative subclasses (#32016) | Model side adopts upstream's split with fork/Prism features re-applied; speculative side keeps the fork driver for every head | `test-dspark-legacy`, `test-dspark-logsnr-meta`, Bonsai acceptance |
| DFlash/MTP device features | Target hidden rows read from device buffers (#31506, #31927) | Not adopted for the fork drivers (performance only; host path kept upstream) | — |
| Server prompt cache | Moved to `server-slot-sched` with shared immutable checkpoints (#32154) | Fork coherence rules (main/draft/spec payload, `draft_synchronized`) ported | `test-server-spec-state` |
| KV rotation | Per-layer device rotation matrices (#31073) | Adopted; Prism's ARC-LAB `attn_rot_dev` copies removed | `test-llama-context` rot column |
| CUDA MMA FlashAttention | Head sizes 40/72, K/V rotation, `kv_rows` | Combined with Prism in-place quantized K/V and the run-time `V_is_K_view` selector; quantized path skipped with `kv_rows` | `test-backend-ops -b CUDA0` |
| Few-row MMQ | New for batch 2-24 and `MUL_MAT_ID` | Adopted; disabled under `GGML_CUDA_BATCH_INVARIANT` | `test-backend-ops -b CUDA0` |
| Prism fused MoE weighted reduction | — (Prism only) | Kept; aliased inputs fall back to the unfused path | `test-moe-weighted-reduce` 48/48 |
| Recurrent state zeroing | `build_rs` fills instead of scaling by 0 (#31913) | Same change applied to Prism's rows-path cache view | `test-llama-context` qwen35 rollback/replay |
| DSpark auto-detection | Upstream reports `draft-dflash` for every DFlash head | Fork still reports `draft-dspark` for Markov heads (requirement 8) | `test-dspark-legacy` |
