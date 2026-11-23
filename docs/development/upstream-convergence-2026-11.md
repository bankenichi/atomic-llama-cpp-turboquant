# Upstream + Prism convergence, 2026-11

Status: **merged and building; validation in progress** (see [Validation](#validation)).
Nothing is pushed. `main` and `integration/upstream-prism-convergence` were not
moved.

This round brings the fork from upstream `d775b8967` / Prism `0810d4bfb` (the
2026-08 convergence, see [`upstream-prism-convergence.md`](upstream-prism-convergence.md))
to upstream `f4f6f2e7d` (b12643+3) and Prism `d12ca13bb`. All upstream and
Prism history arrives through real merge commits with authorship intact.

## Branches and tips

| Item | Value |
|---|---|
| Starting point | `19265bcd0` (`integration/upstream-prism-convergence`) |
| Backup | `backup/pre-upstream-convergence-20261122` = `19265bcd0` (local only) |
| Work branch | `integration/upstream-convergence-2026-11` |
| Upstream tip merged | `f4f6f2e7db76e98c95cc9b7fc608ea669c806044` (b12643+3), fetched 2026-11-22 |
| Prism tip merged | `prism/prism` `d12ca13bb9f2136c8e03bfceccf20b06d115fc52` |
| `prism/master` | `5ea87ddad`, a stale upstream mirror fully contained in upstream; Prism's real line is `prism/prism` |

## Merge points

| Stage | Commit | Merges | Conflicted files | Notes |
|---|---|---|---|---|
| 1 | `8098b4779` | upstream `46847e615` (b11392) | 88 | batch_ext / `llama_process` API migration; fused DFlash encoder (#27310) |
| 2 | `fafb88636` | upstream `d854d98ba` (b11889, Prism's own upstream base) | 45 | plus `c55ebc739` fix(cuda): group-128 MMQ loader on the new precision/gate loader API |
| 3 | `e56b44cf7`, `9b99a704d` | `e56b44cf7` renames Q2_0_G128 to Prism's `PQ2_0` (ID 142 unchanged); `9b99a704d` merges `prism/prism` `d12ca13bb` | 57 | merged into the work branch as `36546fd04` (2 conflicts, the same loader port on both sides) |
| fix | `01a18bea0` | — | — | fix(cuda): reject aliased inputs in upstream's fused MoE weighted reduction |
| 4 | `1503ae990`, `8f2f728d5` | `1503ae990` merges upstream `f4f6f2e7d` onto stage 3; `8f2f728d5` brings it onto the work branch | 125 (+0) | the largest stage: DFlash split, server slot scheduler, MMA FA head sizes, few-row MMQ, F8_E4M3 |
| post | see `git log 8f2f728d5..` | — | — | build fixes for merge interactions, docs, `-ModelRoot` |

Conflicted-file counts are from `git merge-tree` on each merge's parents; a file
counts once however many hunks it had. Total: 317 conflicted files over the four
upstream/Prism merges.

## Conflicts and decisions

### Type IDs (AGENTS.md requirement 7)

- Upstream `GGML_TYPE_F8_E4M3 = 43` and `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42` are
  taken as-is. The GGUF reader no longer remaps raw 43/44: raw 43 is F8_E4M3 and
  raw 44 is rejected. Only the fork's historical weight IDs 45/46 still
  normalize (to TQ3_1S/TQ4_1S, 243/244). The old 43/44 were Turbo3/Turbo4, runtime
  KV formats never written to model files. `test-gguf` checks raw 43/45/46 and the
  rejection of 44. AGENTS.md requirement 7 was updated to say this.
- Prism IDs: `PQ2_0` 142 (the fork's former Q2_0_G128, same layout), `PTQ1_0` 143,
  `PQ2_0M` 145 (KV only). Fork IDs 240-244 unchanged. `GGML_TYPE_COUNT` = 245.
- `llama_ftype`: Prism PQ2_0 141, PQ2_0_LEGACY 142 (the fork's old value, still
  loads), PTQ1_0 143; fork TQ 240/241. gguf-py follows `llama.h` (Prism's python
  128/129 disagree with its own C side).
- Raw 42 detection of the published Bonsai files is unchanged (span-based
  normalization to 142).

### DFlash / DSpark (requirements 2 and 8)

Upstream split DFlash into per-head files and graphs (#32016: `dflash.cpp`,
`dspark.cpp`, `dflash2.cpp`, `lilicorr.cpp`, `variant`/`backbone` enums). The
upstream structure is taken and the fork/Prism additions are re-applied on top:

- legacy Prism `dspark` architecture: `dspark.dspark.*` keys, `dspark.`-prefixed
  tensor names, vocabulary from `token_embd`, legacy Markov names
  (`load_tensors_dspark(n_vocab_draft, n_vocab_model, legacy_dspark)`);
- DFly per-layer context fusion (encoder and injection) and the TreeFlash
  predecessor-correction head (`graph_dflash`);
- DSpark GIDD log-SNR conditioning in `graph_base::build_layers` (all DFlash heads;
  asserts on the DSV4 backbone as before);
- confidence-head / Markov-head consistency checks; drafter-own token embeddings;
- `build_embd_rows` for every draft embedding lookup (Hadamard-latent tables);
- DFlash2 selector copy avoidance for single-sequence runs.

Prism's dedicated DSpark model class moved to `src/models/dspark-prism.cpp`
(upstream now owns `dspark.cpp`); it stays opt-in via `LLAMA_DSPARK_RUNTIME=prism`.

In `common/speculative.cpp` the fork's DFlash driver is kept whole (private
4096-position draft window with rollover, image-feature pooling, prompt lookup,
beam). Upstream's per-head driver subclasses are not adopted. The factory accepts
upstream's new `draft-dflash2` / `draft-lilicorr` types and validates them with
upstream's `llama_model_dflash_variant()`; one fork driver serves every head.

### MTP / NextN

The fork/Prism MTP driver is kept whole: deferred catch-up, chained drafts,
real-position image rows (#30257, adopted in stage 1). Upstream's
device-resident target hidden rows for MTP (#31927) and DFlash (#31506) are
**not** adopted for these drivers. They only save copies, and upstream keeps the
host path we use.

### Server

Upstream moved the prompt cache into `server-slot-sched.{h,cpp}` and made
checkpoints shared immutable pointers (#32154). The fork's prompt-cache rules
(atomic main/draft/spec payload, `draft_synchronized`, no consumption on a failed
restore) are ported into the new file. The fork's checkpoint buffer reuse now
happens only when no prompt-cache entry shares the checkpoint.

### CUDA

- Flash attention: upstream's padded head sizes 40/72, K/V block rotation and
  `kv_rows` combined with Prism's in-place quantized K/V MMA variants and the
  fork's run-time `V_is_K_view` selector (requirement 5). Two merge
  interactions fixed: the rotation uses the K/V-type-aware stage count, and the
  in-place quantized path is skipped when `kv_rows` is present (its tile loads do
  not gather rows). Sparse MMA kernels follow upstream (also on HIP).
- MMQ: upstream few-row MMQ (#31533, #32065) is disabled under
  `GGML_CUDA_BATCH_INVARIANT`, so 1-8 columns keep one arithmetic. Prism's PQ2_0
  J=128 path was ported from `expert_bounds`/`ncols_max` to upstream's tile specs.
  Upstream's NVFP4 scale fusion is merged with Prism's activation gate/RMS
  quantizers (asserted mutually exclusive).
- q8_1 quantizer: Prism's layouts kept; upstream's correction-sum fix (#31247)
  applies to the default layout.
- RMS norm: upstream's fused NORM+MUL(+ADD) and non-F32 inputs keep the fork's
  fixed-rounding float4 path (`GGML_CUDA_RMS_NORM_VECTOR`).
- `GGML_CUDA_CC_DGX_SPARK` renamed to upstream's `GGML_CUDA_CC_GB10` (same value).
- Fork/Prism cache types (`TURBO*`, `PQ2_0M`) accept only F32 `SET_ROWS` sources;
  upstream's new BF16 source path covers its own types.
- MoE: the Prism fused weighted-reduction kernel failed every `x_in_dst` layout
  of `test-moe-weighted-reduce` on CUDA. This is a Prism bug, not a merge
  artifact: the code path is identical on `prism/prism`. Real graphs are not
  affected, because `x` is a `MUL_MAT_ID` output and the allocator keeps
  dependencies apart. Fixed in `01a18bea0` with an explicit overlap check that
  falls back to the unfused path; the test stays as the guard (48/48 after the fix).

### Other

- KV cache: upstream keeps rotation matrices on the device per layer
  (`init_rotation`, #31073), which replaces Prism's ARC-LAB `attn_rot_dev` copies.
- Recurrent memory: upstream's per-cell `rs_max` rollback bound plus Prism replay.
- CPU: upstream's row-parallel concat replaces the fork's equivalent change.
- Draft fit keeps the target's `n_rs_seq` (Prism #173) in upstream's relocated
  `common_fit_apply`.
- `test-backend-ops` gated-delta-net cases: the fork/Prism constructor parameters
  stay first; upstream's new cases pass explicit defaults for them.
- CI: only `.github/workflows/windows-cuda.yml`; new upstream workflows dropped.
- Vulkan, SYCL, Metal, WebGPU conflicts were resolved but **not built** here.
  Vulkan FWHT keeps Prism's signed/f16 variant; WebGPU takes upstream's mat-vec row
  limit (4), so Prism's 8-column routing (#301) was not carried over.

## Capabilities not preserved or changed

- Upstream's device-resident hidden rows for MTP and DFlash (#31927, #31506) are
  not used by the fork's drivers. This is a performance difference only; the
  host path is unchanged.
- Upstream's few-row MMQ is off in batch-invariant mode by design.
- WebGPU: Prism's 8-column mat-vec routing (#301) is not carried over (untested
  backend).
- Prism's ARC-LAB `LLAMA_SYNC_TRIM` bit 2 (device copies of the attention
  rotation) is replaced by upstream's equivalent and no longer does anything.

## Validation

<!-- filled in as results arrive -->

| Check | Result |
|---|---|
| Stage 1 CUDA build / CTest / backend-ops | 502/502 built; CTest 47/48 (quant-type-selection, remote drift); `test-backend-ops -b CUDA0` 17192/17192 |
| Stage 2 | built; CTest 49/50 (same); backend-ops 19227/19227 |
| Stage 3 | 634/634 built; CTest 54/56 (quant-type-selection drift; moe-weighted-reduce, then fixed: 48/48); backend-ops 21116/21116; Granite PPL 7.6369 |
| Stage 4 CUDA build | 611/611 targets after the build fixes (`12a2cc8c7`, `9bd028f0f`), `build-converge` |
| Stage 4 CTest (CUDA) | 62/66; failures: `test-llama-context` (see below), the two `python3` tests and the `test-quant-type-selection` drift |
| Stage 4 `test-backend-ops -b CUDA0` | 23030/23031; the one failure (`MUL_MAT pq2_0 m=1 n=1 k=1024`, NMSE 0.0020 vs 0.0005 for a single output element) passes 5/5 when run alone, so it is input-order sensitivity, not a kernel change |
| CPU build + CTest | `build-cpu-converge` complete; CTest 63/66: the two `python3` tests (Windows Store stub; both pass with a real interpreter: 25/25 and 1/1) and the `test-quant-type-selection` drift |
| Granite-1B F16 perplexity | 7.6363 ± 0.0459 vs reference 7.6360 (+0.004 %), CUDA, `-c 2048 -ngl 99` |
| Granite HF → GGUF conversion | converted from the `E:\moeq\hf` snapshot with `C:\moeq-env` python; all 242 tensors bit-identical to the reference GGUF (metadata differs only in names derived from the snapshot directory, the newer chat-template revision and two newer tokenizer keys); output deleted afterwards |
| Acceptance: Bonsai + DSpark + mmproj | pending |
| Acceptance: Gemma 4 26B-A4B + assistant + mmproj | pending |
| Acceptance: Qwen 3.6 35B-A3B NextN + mmproj | pending |

`test-quant-type-selection` reads `ggml-org/Qwen3.6-27B-GGUF` from `resolve/main`.
That repository was re-uploaded with merged `attn_qkv_gate` tensors, so the test
fails on remote drift, not on a code change.

Upstream moved `test-backend-ops` out of CTest (#28740); it is run directly.

## Build recipe

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
set HF_UI_VERSION=b12643
cmake -S . -B build-converge -G "Ninja Multi-Config" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DLLAMA_OPENSSL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON ^
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-converge --config Release -j 8 -- -k 0
ctest --test-dir build-converge -C Release --output-on-failure
build-converge\bin\Release\test-backend-ops.exe test -b CUDA0
```

The CPU tree is the same without `-DGGML_CUDA=ON` and with the
`CMAKE_CUDA_ARCHITECTURES` option dropped (`build-cpu-converge`). `HF_UI_VERSION`
must name a real upstream release tag; the fork's own commit count produces a
`bNNNNN` that does not exist on Hugging Face. Delete `<build>\tools\ui\dist`
when it changes.

On the owner's machine every GPU job runs under the machine-wide lock with a
declared VRAM reservation, sized close to its measured peak (see
`E:\moeq\locks\peaks.log`):

```text
C:\moeq-env\Scripts\python.exe -m moeq.gpulock run --label "<what>" --vram-mib <N> -- <command>
```

The exact-model acceptance runs omit `--vram-mib`, so they reserve the whole budget and run alone.

## Notable upstream changes since 2026-08-22 for this owner

Roughly 2,070 upstream commits landed between `d775b8967` (2026-08-22) and
`f4f6f2e7d`. These are the ones that matter for this fork's models and
hardware (RTX 5080 / sm_120, Windows, Bonsai + DSpark, Gemma 4 + assistant,
Qwen 3.6 MoE + NextN, TurboQuant KV):

**Speculative decoding**
- DFlash family refactor into per-head model files and drivers (#32016);
  `draft-dflash2` and `draft-lilicorr` are separate spec types; DFlash2
  probabilistic drafting (#30663) and Ripple DFlash2 support (#31904).
- Fused DFlash encoder into the KV injection (#27310); device copies of target
  features for DFlash (#31506) and MTP (#31927).
- Probabilistic draft sampling for draft-simple and MTP (#27694), with a warning
  when `--spec-draft-p-min` is combined with it (#31294); drafts stop at EOG
  (#29638); server logprobs with speculative decoding (#27196).
- MTP: save/restore of the hidden state with the prompt cache (#30141), draft
  state in slot files (#30574), CUDA graphs for the MTP draft (#28549), mtmd
  input for MTP at real positions (#30257, adopted), `--spec-draft-n-ubatch`
  (#30459), speculative checkpoint restore for drafts longer than `n_rs_seq`
  (#31350), MoE cache not applied to the draft model (#31497).
- New DSpark backbones: Gemma 4 (#29226), DeepSeek-V4.1 (#31086), Mistral Large 4.

**CUDA (all of it applies to sm_120)**
- OCP FP8 (`F8_E4M3`) weights (#28413) and an FP8 KV cache (#28392).
- Few-row MMQ for batch 2-24 (#31533) and for `MUL_MAT_ID` (#32065): faster
  speculative verification batches. Fork note: disabled when
  `GGML_CUDA_BATCH_INVARIANT` is set.
- Fused MoE gate/up + GLU in MMQ (#31188), fused FFN gate/up + GLU with
  interleaved rows (#29948), shared experts fused into MMVQ (#29184), grouped
  top-k MoE fusion (#29181, #31288), duplicate expert IDs (#32505).
- Fused NORM + MUL (+ ADD) (#32454); BF16 for RMS_NORM, ROPE, SET_ROWS, unary,
  GLU, binary and scale ops (#32135, #32387, #29675); MMVF for thin BF16
  matrices (#31959, #29633).
- MMA flash attention for head sizes 40/72 (#30285), whole-tile scheduling
  (#29435), `flash_attn_ext_rows` (per-sequence K/V rows, unified KV for
  multi-sequence) (#29510).
- Chunked CUDA graphs (#30804), graph re-instantiation on cluster-dim changes
  (#31284), graph compat check skipped on reuse (#32004).
- SKU-aware FP16/BF16 accumulation precision gating (#30672); faster MMVQ for
  low-bit types (#30517); 64-bit dst offsets in MMQ (#32374).
- `ggml_cuda_shfl*` wrappers replace raw `__shfl_*_sync` (#31617); use them in
  new kernels.

**Models and memory**
- A GPU cache for MoE experts kept in host memory (#29887, SLRU #30637,
  multi-GPU #30112, CPU share `--moe-cache-cpu-share`). This is relevant for
  Gemma/Qwen with `--n-cpu-moe`.
- KV rotation matrices kept on the GPU (#31073) and restored exactly (#28498).
- Gemma 4: graph optimizations (#31356), multi-sequence prompt processing
  (#32342), tool calls during thought (#32232), `test-llama-archs` coverage
  (#31401).
- `lora : support prism.hadamard folded weights` (#30910), which is upstream
  awareness of Prism's Hadamard-latent weights.
- GGUF reader hardening against invalid enum values (#32641).

**Server**
- Prompt-cache / slot scheduler refactor with a prompt running hash (#32154).
  Slots now prefer empty slots, and only restorable states are cached (#30567,
  #30498).
- Security: SSRF protection for CORS proxy and media URLs (#32115), explicit
  opt-in for tools on non-loopback addresses (#32113), DNS-rebinding protection
  (#32098). Private media URLs now need `--media-allow-private-urls`.
- Realtime API (`/v1/realtime`), TTS (`/tts`, `/v1/audio/speech`), and
  reasoning-effort levels from the chat template (#31271, #32332).
- Default port is now 9931 (#30159). The fork's launchers pass `--port`
  explicitly.

## Notes for adding a new weight type

These notes are about the owner's planned low-rank binary-factor expert format:

    W ≈ diag(s_post) · sign(U) · sign(V)ᵀ · diag(s_pre),   signs packed 32 per int32

where `W` is `n_out × n_in`, `U` is `n_out × r` and `V` is `n_in × r`.

### Option A: no new ggml type (try this first)

`y = W x = s_post ⊙ (sign(U) · (sign(V)ᵀ · (s_pre ⊙ x)))` is two binary
mat-vecs around two element-wise scales. Every piece already exists:

- `sign(V)ᵀ` and `sign(U)` can be stored as upstream `Q1_0` tensors (128 signs
  plus an fp16 scale per block, 1.125 bpw, `QK1_0 = 128`). `Q1_0` already has CPU,
  CUDA dequant, MMVQ, MMQ (including MoE `MUL_MAT_ID`) and `get_rows` support.
- `s_pre` varies along K, so it cannot live in a block scale. Apply it to the
  activations: `ggml_mul(x, s_pre)`, broadcast over tokens. For MoE, use one row
  per expert gathered with the expert ids, the same pattern as the per-expert
  scales in `build_moe_ffn`.
- `s_post` is per output row. It can be folded into `sign(U)`'s block scales when
  `r` is a multiple of 128 (every block of a row gets `d = s_post[row]`, fp16
  precision), or applied as a second `ggml_mul` after the second mat-vec.
- Graph: in `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`), replace
  each expert matmul with two `ggml_mul_mat_id` calls (rank-`r` intermediate)
  plus the scales. The rank dimension becomes the intermediate width.

Constraints: `n_in` and `r` must be multiples of 128 (Q1_0 block). The
intermediate `r`-vector per token/expert is small, so the cost is two
bandwidth-bound passes over 1-bit data.

### Option B: a dedicated type with 32 signs per int32

Use this when the exact packing matters (1.0 bpw, no per-block scale, 32-wide
blocks). Pick the next fork-private ID, **245**, and raise `GGML_TYPE_COUNT` to
246. Keep it out of upstream's ascending range (currently up to 43) and Prism's
140s. Then touch, in order:

1. **Registration**
   - `ggml/include/ggml.h`: `enum ggml_type`, plus a comment in the fork block.
   - `ggml/src/ggml-common.h`: `QK_*` and the block struct (`uint32_t qs[1]` for
     32 signs), with a size `static_assert`.
   - `ggml/src/ggml.c`: `type_traits[]` (`type_name`, `blck_size`, `type_size`,
     `is_quantized`, `to_float`, `from_float_ref`).
   - `ggml/src/ggml-quants.{c,h}`: reference quantize/dequantize and
     `ggml_validate_row_data` (a sign-only row has nothing to validate except its
     size).
   - `gguf-py/gguf/constants.py`: `GGMLQuantizationType` and `GGML_QUANT_SIZES`;
     optionally `gguf-py/gguf/quants.py` for numpy (de)quant.
   - `include/llama.h`: `llama_ftype`, if whole-model ftypes make sense.
     `src/llama-model-loader.cpp`: ftype name and the type-to-ftype guess.
     `src/llama-quant.cpp` and `tools/quantize/quantize.cpp` `QUANT_OPTIONS`.
   - `ggml/src/gguf.cpp` needs nothing beyond `type_traits` as long as the ID is
     not one of the normalized legacy IDs (45/46).
2. **CPU** (`ggml/src/ggml-cpu/`)
   - `ggml-cpu.c` `type_traits_cpu[]`: `from_float`, `vec_dot`, `vec_dot_type`
     (`GGML_TYPE_Q8_0` is the simplest partner), `nrows`.
   - `quants.{c,h}`: generic `ggml_vec_dot_<type>_q8_0`.
   - `arch/x86/quants.c`: AVX2 popcount/xor version if CPU speed matters.
   - `ops.cpp`: add the type to the quantized `case` lists (add, add1, acc,
     out_prod, set, get_rows, ...); grep for `GGML_TYPE_PTQ1_0` and mirror it.
3. **CUDA dequant** (`ggml/src/ggml-cuda/`)
   - `common.cuh`: `ggml_cuda_type_traits<type>` (`qk`, `qr`, `qi`).
   - `convert.cu`: `ggml_get_to_fp16_cuda`, `ggml_get_to_fp32_cuda`, and the
     `_nc` / BF16 variants. The cuBLAS fallback and `get_rows` (`getrows.cu`)
     go through these.
4. **CUDA mat-vec (MMVQ)**
   - `vecdotq.cuh`: `vec_dot_<type>_q8_1` plus `VDR_*_MMVQ`. With 32 signs per
     int32 and a q8_1 activation, one int32 of signs pairs with eight int32 of
     activation bytes. Expand signs to ±1 bytes and use `dp4a`, or use the
     `sum(y) - 2*sum(y where bit=0)` identity with the q8_1 block sum.
   - `mmvq.cu`: `get_vec_dot_q_cuda`, `get_vdr_mmvq`,
     `mul_mat_vec_q_switch_type`, the per-architecture rows/nwarps tables
     (`calc_nwarps` / `calc_rows_per_block`), and `get_mmvq_mmid_max_batch` for
     the MoE batch limit. `mmvq.cuh` `ggml_cuda_should_use_mmvq`.
5. **CUDA mat-mat (MMQ)**
   - `mmq.cuh`: `load_tiles_<type>` (unpack signs into the Q8_0 SRAM layout is
     the least work), `ggml_cuda_mmq_get_load_tiles`,
     `ggml_cuda_mmq_get_vec_dot`, and `ggml_cuda_mmq_get_config` together with
     `mmq-config-*.cuh` `CASE(...)` entries per architecture.
   - `mmq.cu`: `ggml_cuda_mul_mat_q_switch_type` and `ggml_cuda_should_use_mmq`.
   - `template-instances/generate_cu_files.py`: add the type so that an
     `mmq-instance-<type>.cu` is generated and checked in.
   - Few-row MMQ (`mmq-few.cu`) needs a decoder (`mmq_few_unit_size`); leave it
     at 0 initially.
6. **MoE dispatch**
   - `ggml-cuda.cu` `ggml_cuda_mul_mat_id`: MMVQ with ids up to
     `get_mmvq_mmid_max_batch`, then few-row MMQ, then MMQ with ids (tile specs
     from `ggml_cuda_launch_mm_ids_helper` in `mmid.cu`), and finally the
     per-expert fallback (`get_rows` + `mul_mat`). A type that MMVQ and MMQ
     support needs no special case here.
   - `ggml-cuda.cu` `ggml_cuda_mul_mat` is the dense order (mmvq, few-row MMQ,
     MMQ, cuBLAS), and `ggml_backend_cuda_device_supports_op` (`MUL_MAT`,
     `MUL_MAT_ID`, `GET_ROWS`) must list the type.
   - Fused gate/up + GLU paths (`ggml_cuda_should_fuse_mul_mat`, MMQ fusion) are
     per type; keep them off for the new type until it is verified.
7. **Tests**
   - `tests/test-backend-ops.cpp`: add the type to `all_types`/`base_types` and
     add `test_mul_mat` / `test_mul_mat_id` shapes at the real expert sizes and
     batches 1-8.
   - `tests/test-quantize-fns.cpp`, and `test-gguf` if the ID is near a legacy
     one.
   - Run `test-backend-ops test -b CUDA0 -o MUL_MAT` and `-o MUL_MAT_ID` under the
     GPU lock.

The low-rank structure itself still lives in the graph as in option A. A
dedicated type only replaces `Q1_0` for the two sign factors. A single fused
CUDA op for the whole `diag·sign·sign·diag` product would be a later
optimization on top of either option.
