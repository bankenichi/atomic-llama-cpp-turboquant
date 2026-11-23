# Upstream + Prism convergence, 2026-11

Status: **in progress** (stages 1-2 on the integration branch, stage 3
committed on a side branch, stage 4 merge mid-resolution). The full writeup
(merge points, conflicts, test results, notable upstream changes, notes for a
new weight type) is still to be written; this file currently holds the RESUME
notes.

## RESUME

### Branches and tips

| Item | Value |
|---|---|
| Starting point | `19265bcd0` (`integration/upstream-prism-convergence`) |
| Backup | `backup/pre-upstream-convergence-20261122` = `19265bcd0` (local only) |
| Work branch | `integration/upstream-convergence-2026-11` (this checkout) |
| upstream tip merged at the end | `f4f6f2e7db76e98c95cc9b7fc608ea669c806044` (b12643+3) |
| Prism tip | `prism/prism` `d12ca13bb9f2136c8e03bfceccf20b06d115fc52` |
| `prism/master` | `5ea87ddad`, a stale upstream mirror fully contained in upstream; Prism's real line is `prism/prism` |

Merge points:

1. `8098b4779` Merge upstream b11392 (`46847e615`). Validated (see below).
2. `fafb88636` Merge upstream b11889 (`d854d98ba`, Prism's own upstream base),
   plus `c55ebc739` fix(cuda) for the group-128 MMQ loader API.
   **Build running unattended** (see "Running jobs").
3. Branch `wip/convergence-m3` (worktree `C:\Users\kenic\Documents\acl-wt`):
   `e56b44cf7` refactor (Q2_0_G128 -> Prism's PQ2_0 names, id 142 unchanged),
   `9b99a704d` Merge `prism/prism`. Committed, **not built yet**.
4. Branch `wip/convergence-m4` in the same worktree: `git merge` of
   `f4f6f2e7d` **in progress, uncommitted**. Resolved so far: type
   registration (ggml.h/ggml.c/gguf.cpp/gguf-py/llama.h), llama-arch,
   llama-graph, llama-context, llama-hparams, llama-model(-loader),
   delta-net-base, qwen35, models.h, test-gguf. Still conflicted (file: hunks):
   common.cpp 2, common.h 2, speculative.cpp 17, docs/speculative.md 3,
   ggml/cmake/common.cmake 1, ggml-rpc.h 1, ggml-backend.cpp 2, ggml-cpu.c 3,
   ops.cpp 1, cpu/quants.h 1, cuda: common.cuh 2, convert.cu 4,
   fattn-common.cuh 2, fattn-mma-f16.cuh 3, fattn-vec.cuh 2, fattn.cu 2,
   fwht.cu 1, gated-delta-net-mma.cu 1, ggml-cuda.cu 5, mmq-config-gcn.cuh 1,
   mmq.cu 6, mmq.cuh 2, mmvf.cuh 1, mmvq.cu 6, norm.cu 5, quantize.cu 3,
   quantize.cuh 3, scale.cu 1; metal device.m 1, fusion.cpp 1, misc.metal 1;
   sycl fattn-mkl 1, gated_delta_net 1, ggml-sycl.cpp 6; vulkan types.h 1,
   ggml-vulkan.cpp 6, fwht.comp 2, shaders-gen 1; webgpu 1; llama-context.h 1,
   llama-ext.h 3, llama-grammar.cpp 2, llama-kv-cache.cpp 7, llama-kv-cache.h 4,
   llama-memory-recurrent.cpp 2, models/dflash.cpp 9, models/dspark.cpp 1,
   tests/CMakeLists.txt 1, test-backend-ops.cpp 15, test-llama-archs.cpp 2,
   tools/server: CMakeLists 1, server-context.cpp 6, server-task.cpp/.h 1/1,
   .github/actions/windows-setup-cuda/action.yml 1.
   Standing rules already applied in it: fork-deleted paths stay deleted, new
   upstream CI dropped, AGENTS.md/README.md/docs/build.md kept as ours.

When stage 2 passes: `git merge --ff-only wip/convergence-m3` here, build and
test stage 3, then finish stage 4 in the worktree, commit, fast-forward, build,
test. Remove the worktree afterwards (`git worktree remove`).

### Decisions taken (to carry into the ledger)

- Type ids: ggml `F8_E4M3` = 43 (upstream). Fork private 240-244 unchanged.
  Prism `PQ2_0` 142, `PTQ1_0` 143, `PQ2_0M` 145. GGUF reader now normalizes
  only raw 45/46 -> TQ3_1S/TQ4_1S; raw 43/44 are no longer remapped (43 is
  F8_E4M3; Turbo2/3/4 were KV-only and never written to model GGUFs).
  **AGENTS.md requirement 7 text must be updated** ("Old private IDs 43-46"
  -> 45-46). `test-gguf` rewritten accordingly (worktree, stage 4).
- llama_ftype: Prism PQ2_0 141, PQ2_0_LEGACY 142 (the fork's old value),
  PTQ1_0 143; fork TQ 240/241; upstream F8_E4M3 42. gguf-py follows llama.h
  (Prism's python 128/129 disagree with its C side).
- `dspark` architecture: fork DFlash-based runtime stays default (4096
  rollover window + image pooling, requirements 2/8); Prism's dedicated
  runtime opt-in via `LLAMA_DSPARK_RUNTIME=prism` (model class and
  speculative impl).
- DFlash/DSpark image pooling now pools raw target features per position bin
  (upstream fused the DFlash encoder into the injection decode, #27310);
  needs Bonsai image re-validation.
- draft-simple keeps the private text domain; MTP adopts upstream #30257
  (image rows into MTP draft at real positions); EAGLE3 keeps compact media
  boundary.
- Expert prefetch re-wired before upstream's new `ggml_backend_sched_copy_input`
  (selective expert copy moved to a user callback, #29943).
- CUDA FA: upstream `GGML_CUDA_FA_QUANTS`; f16-q8_0 always compiled; Turbo
  pairs via `FATTN_VEC_CASE_ALWAYS`; run-time V_is_K_view selector carries
  upstream sparse kernels and Prism's `type_KV`.
- Prism GDN: raw gates / state rows / replay kept; GDN fused-op probes dropped
  because upstream removed the non-fused GDN path (#32078). Raw-gate path is
  limited to separate (non-merged) beta/alpha projections.
- Silent duplicate code from the fork's August Prism import vs `prism/prism`
  was removed in stage 3 (scan: dupscan over both-sides-added lines).
- Upstream moved `test-backend-ops` out of CTest (#28740); it must be run
  directly (`test-backend-ops test -b CUDA0`). Consider re-registering.
- `test-quant-type-selection` fails because it reads
  `ggml-org/Qwen3.6-27B-GGUF` from `resolve/main` (remote re-upload), not a
  code regression; re-check on the final tree.

### Validation so far

- Stage 1 (`8098b4779`): full CUDA tool suite built (VS 2022 v143 toolset via
  `vcvarsall x64 -vcvars_ver=14.44`, CUDA 12.8, `sm_120a`, Ninja Multi-Config,
  `build-converge`). CTest 47/48 (only quant-type-selection, see above).
  `test-backend-ops test -b CUDA0`: 17192/17192 passed, including the 576x512
  F16-K/Q8_0-V separate-V and Q8/Q8 K-subview cases, Turbo SET_ROWS, TURBO_WHT.
- Stage 2: build in progress. Stages 3-4: not built. CPU build, acceptance runs
  and the Granite check: not started.

### Build recipe used

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
set HF_UI_VERSION=b11889   & rem upstream tag of the stage; fork commit count gives a nonexistent bNNNNN
cmake -S . -B build-converge -G "Ninja Multi-Config" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DLLAMA_OPENSSL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON ^
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-converge --config Release -j 8 -- -k 0
ctest --test-dir build-converge -C Release --output-on-failure
build-converge\bin\Release\test-backend-ops.exe test -b CUDA0
```

Delete `build-converge\tools\ui\dist` when `HF_UI_VERSION` changes. Final
stage: use the upstream tag of the merged tip (b12643) or `latest`.

### Running jobs

- Stage 2 CUDA build, started unattended:
  log `C:\Users\kenic\AppData\Local\Temp\claude\ec5fee28b324d4b7\47cec40e-c522-4cb0-ac80-6091b0e5cb46\scratchpad\build-m2b.log`
  (ends with `exit=<code>`; check for `FAILED`/`error`). After it succeeds run
  CTest and the CUDA backend-op matrix as above (GPU needs >= 4 GiB free).

### Still to do after stage 4

Ledger section, convergence doc merge points, NOTICE (Prism contributors,
F8/Prism types), this file's full writeup (notable upstream changes, notes
for a new weight type), `scripts/start-hidden-atomic-server.ps1 -ModelRoot`
(default `C:\Program Files\llamacpp`; models now in `D:\LLM Models`), CPU
build + CTest, acceptance runs (Bonsai+DSpark+mmproj, Gemma 4 + assistant +
mmproj, Qwen 3.6 NextN + mmproj) only with < 1.5 GiB VRAM in use, Granite
perplexity (PPL 7.6360 +/- 0.1 %) and conversion checks.
