# AGENTS.md — repository orientation

This is a personal fork of `AtomicBot-ai/atomic-llama-cpp-turboquant`, itself a
fork of `ggml-org/llama.cpp`. It has one owner, no external contributors, and no
intention of submitting upstream PRs. Optimize for the owner's machine and models
while keeping the repository mergeable with future upstream releases.

## Owner target

- Windows, one NVIDIA RTX 5080 (Blackwell, compute capability 12.0), 16 GB VRAM.
- Primary convergence target: Prism ML Ternary Bonsai 27B base + DSpark + mmproj.
- Required regressions: Gemma 4 26B-A4B + historical assistant + vision, and Qwen
  3.6 MoE with embedded NextN.
- Vision is a key capability. Never treat mmproj/text coexistence or image turns as
  optional test coverage.
- Long-horizon subservers and media-generation workloads are out of scope until
  base-model compatibility is complete and stable.

## Non-regression requirements

1. **Real-media gating** (`server_tokens::has_media`) keeps speculation enabled for
   text requests merely because an mmproj is loaded.
2. **Media-aware speculative coordination** (`common/speculative.cpp` and
   `server-spec-state.h`) selects a policy per implementation and sequence.
   DFlash/DSpark consume pooled target image features, EAGLE/MTP and draft-simple
   use explicit private position domains, and n-grams remain stateless. A retained
   image in full-history requests must never become a permanent speculation veto.
3. **Modern MTP contexts** use `LLAMA_CONTEXT_TYPE_MTP`, which selects the
   decoder-MTP graph through the normal context path. Draft-context construction
   in `common/speculative.cpp` keeps `op_offload = false`; preserve that setting
   for Gemma assistants on CPU-offloaded MoE targets.
4. **Legacy Gemma assistant GGUFs** retain support for raw
   `general.architecture = gemma4_assistant`, their legacy metadata, and `mtp.*`
   tensor names. The loader's raw architecture-prefix override is intentional.
5. **Vision context shift** supports in-place f16/q8 K shift with chunk-aware erase
   and image-boundary snapping. Turbo and M-RoPE use transactional reprefill.
6. **Unknown-template vision fallback** in `common/chat.cpp` renders string content
   when template capability detection supports neither string nor typed parts, so
   the internal media marker cannot be dropped.
7. **Ternary Bonsai GGUF compatibility** normalizes decisive legacy raw type 42
   spans to type 142 (Prism `PQ2_0`, formerly named Q2_0-G128). Ambiguous type 42
   retains the standardized group-64 meaning; contradictory physical evidence
   fails. Old private weight IDs 45-46 normalize to TQ3_1S/TQ4_1S (243-244). Raw 43
   is upstream F8_E4M3 and raw 44 is rejected; the old 43-44 Turbo IDs were
   KV-cache-only and never written to model files.
8. **DSpark** accepts modern DFlash and legacy Prism metadata, validates integer
   values strictly, and caps draft context at 4096 positions per parallel slot.
   Its target-absolute/private-local mapping slides in bounded chunks, is saved
   with checkpoints, and forces paired rebuild or transactional reprefill where
   target-absolute cache operations would be unsafe. Long target sequences must
   continue drafting past position 4096.
9. **Adaptive Bonsai flash attention** supports separate F16 K / Q8_0 V at
    576x512 without using the DeepSeek MLA K-subview stride. Keep both CUDA layout
    tests.
10. **Expert upload prefetch** in `ggml-backend.cpp` remains opt-in through
    `GGML_SCHED_PREFETCH_EXPERTS`; it requires op offload/`--n-cpu-moe`.
11. **mmap host registration** remains opt-in through
    `GGML_CUDA_REGISTER_HOST`; keep the Windows page-alignment path.
12. **Windows build fixes**: MSVC uses `ASM_MASM`; the vendored HTTP library keeps
    its OpenSSL const-correctness compatibility.

The authoritative convergence writeup is
[`docs/development/upstream-prism-convergence.md`](docs/development/upstream-prism-convergence.md).

## Current speculative CLI

- Gemma assistant: `--model-draft assistant.gguf --spec-type draft-mtp
  --spec-draft-n-max 2`.
- Qwen embedded NextN: no draft path; `--spec-type draft-mtp
  --spec-draft-n-max 2`.
- Prism DSpark: `--model-draft dspark.gguf --spec-type draft-dspark
  --spec-draft-n-max 4`.

Removed names must not return to maintained scripts or current docs:
`--mtp-head`, `--draft-block-size`, `--draft-max`, `--draft-min`,
`--spec-type mtp`, and `--spec-type nextn`.

## Build and validation

CUDA can fail against bleeding-edge MSVC. Use Visual Studio 2022 v143. CI uses a
Windows 2022 runner, CUDA, Blackwell code generation, and publishes the complete
tool suite with CUDA runtime DLLs through the sole workflow
`.github/workflows/windows-cuda.yml`.

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure
```

Run exact model inference only as a hidden child process with stdout/stderr
redirected and a finite timeout. Direct interactive inference has crashed the
controlling Codex process on backend faults.

Before publishing model, server-cache, speculative, GGUF, quantization, or CUDA
changes, require:

- full Windows CUDA tool-suite build;
- full CTest suite;
- CPU build/tests for shared code changes;
- exact Bonsai base + DSpark + mmproj + TurboQuant text/vision/context-shift run;
- exact historical Gemma target + assistant + mmproj run with accepted drafts;
- exact Qwen embedded NextN run with accepted drafts;
- focused F16/Q8 separate-V and Q8/Q8 K-subview CUDA operator tests.

## Runtime constraints

- Keep dedicated VRAM at roughly 15 GB or below; shared-memory spill destroys
  decode throughput.
- `--n-cpu-moe` is the primary capacity/performance control for MoE models.
- MTP may be break-even on a CPU-memory-bound target; preserve it for capability
  and GPU-resident configurations, but do not claim universal speedup.
- `--reasoning off` is useful for clients that do not want thinking output.

## Documentation map

- `README.md` — current fork overview and executable examples.
- `MTP.md`, `NEXTN.md`, `docs/speculative.md` — speculative operation and CLI.
- `docs/development/upstream-prism-convergence.md` — architecture and acceptance.
- `docs/development/context-shift-with-vision.md` — KV/media shift behavior.
- `docs/development/media-aware-speculative-decoding.md` — image-answer drafting,
  private position domains, failure isolation, validation, and provenance.
- `docs/development/mtp-cpumoe-vision-fixes.md` — historical Gemma root causes.
- `docs/development/gemma4-12b-vision.md` — template/encoder-free vision history.
- `docs/development/performance-features.md` — runtime performance switches.
- `docs/development/ternary-performance-tuning.md` — validated 32K/64K/128K
  Bonsai profiles, quality gates, rejected settings, and confirmation medians.
- `NOTICE.md` — lineage and attribution.

## Conventions

- Keep `LICENSE` and `NOTICE.md` intact.
- Preserve imported commit authorship and name the source repository/branch or PR
  when porting code. Distinguish direct ports from local adaptations; provenance is
  a correctness requirement for this fork, not optional release polish.
- Document nontrivial private changes under `docs/development/`.
- Comment non-obvious invariants at the implementation site, especially file-format
  collision rules, cross-context cache state, model compatibility aliases, and
  backend layout assumptions. Explain *why* the branch exists and what regression
  a tempting simplification would reintroduce; avoid comments that merely restate
  the next line.
- Maintain only the single Windows-CUDA CI workflow.
- Preserve unrelated owner changes in a dirty worktree.
- Create and publish a backup branch/tag before pushing a major convergence to
  the main branch. Never force-push.
