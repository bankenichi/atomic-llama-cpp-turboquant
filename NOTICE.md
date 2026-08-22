# Attribution & Provenance

This repository is a personal fork maintained for private experimentation. It is
**not affiliated with, endorsed by, or intended to be merged upstream into** any
of its parent projects.

## Lineage

- Primary upstream: **ggml-org/llama.cpp**
  ([repository](https://github.com/ggml-org/llama.cpp)). The model runtime,
  server, GGML backends, GGUF implementation, multimodal stack, generic
  speculative framework, build system, and most tests originate with the ggml
  authors and llama.cpp contributors.
- Immediate parent fork: **AtomicBot-ai/atomic-llama-cpp-turboquant**
  ([repository](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant)).
  This lineage supplied the original Gemma assistant MTP, Qwen NextN,
  TurboQuant KV/weight formats, launchers, benchmark history, and related fixes
  that this personal fork continues to maintain.
- Ternary/DSpark integration source: **PrismML-Eng/llama.cpp**
  ([repository](https://github.com/PrismML-Eng/llama.cpp)) and the
  [`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
  release. The Q2_0-G128 backend work and Prism DSpark extensions were imported
  with their Git authorship intact. Contributors visible in that imported history
  include Pasha Khosravi and Brian / `bri-prism`.
- Expert-prefetch and mmap host-registration source:
  **thecodacus/llama.cpp**, branch
  [`fable5/prefetch-experts`](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts).
  This fork adds the Windows page-registration implementation and integrates the
  feature with the converged scheduler, but the original mechanism and reported
  benchmark work are credited to that source.

Git history is the authoritative per-line/per-commit attribution record. Imported
commits and merge ancestry must not be squashed merely to make the history look
local.

## Feature provenance

| Area | Origin retained in this fork | Local adaptation |
|---|---|---|
| Core inference, GGUF, server, mtmd, generic speculative decoding | ggml-org/llama.cpp and its contributors | Upstream convergence plus private compatibility hooks |
| Gemma 4 assistant MTP and Qwen NextN lineage | AtomicBot-ai fork history; contributions include Biogenic Ooze / Ooze and other credited commit authors | Migrated to current `draft-mtp`, preserved historical assistant GGUFs, added vision/cache coherence |
| TurboQuant KV and TQ3/TQ4 weight formats | AtomicBot-ai fork history and its contributors, including Tom Turney / `TheTom` where recorded | Private GGUF IDs moved away from upstream assignments; Bonsai mixed-KV CUDA dispatch fixed |
| Q2_0-G128 ternary backend | PrismML-Eng fork; imported commits retain Pasha Khosravi and Brian / `bri-prism` authorship | Legacy raw-type-42 detection and collision-free runtime ID 142 |
| DSpark | Upstream llama.cpp DSpark work (including [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) and [PR #26275](https://github.com/ggml-org/llama.cpp/pull/26275)) plus PrismML-Eng's block-diffusion/GIDD extensions | Legacy Prism metadata namespace, strict parsing, context/vocabulary bounds, and M-RoPE-aware target-feature pooling for image prefill |
| Media-aware speculative decoding | Upstream EAGLE3 ([PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)), DFlash/DSpark, and MTP drivers; Atomic mmproj/speculation coexistence direction by Biogenic Ooze at commit `ead60fb4f` | Per-implementation media policies, compact EAGLE/draft-simple position domains, retained-image MTP bridging, and isolated failure finalization |
| Expert upload overlap and mmap host pinning | thecodacus branch linked above | Windows `GetSystemInfo` page handling and this fork's env-gated integration |
| Gemma 4 vision, context-shift, prompt-cache, and coexistence fixes | This personal fork, built on upstream mtmd and AtomicBot model work | Owner-specific integration and regression coverage |
| Launcher configuration UI | [`bankenichi/llama-config-ui`](https://github.com/bankenichi/llama-config-ui), retained as a separately versioned submodule with its original Git history | Typed Atomic launcher adapter, feature validation, managed lifecycle, profiles, diagnostics, and tests |

Named contributor summary (non-exhaustive; Git history remains authoritative):

- Upstream Gemma 4 MTP work in the merged history includes Aman Gupta
  ([llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)),
  Max Krasnyansky's E2B/E4B assistant work
  ([PR #24282](https://github.com/ggml-org/llama.cpp/pull/24282)), and Johannes
  Gäßler's CUDA FlashAttention correction
  ([PR #25148](https://github.com/ggml-org/llama.cpp/pull/25148)).
- Atomic fork MTP/NextN work records Biogenic Ooze / Ooze as the primary author of
  the assistant integration, centroid routing, shared-model NextN path, benchmarks,
  and documentation; Dr Sujit Vasanth contributed assistant tensor-prefix and
  Gemma/TurboQuant cache fixes. Other commit authors remain credited in history.
- TurboQuant history records major work by Tom Turney / `TheTom` and Gabe Ortiz,
  with backend/fix contributions from Simon Gardling, Terry Simons, Misha
  Skvortsov, Andreas Livet, Tuklus-Labs, and others. This list is descriptive, not
  a replacement for the commit log.
- Upstream DSpark work records 王金旭 (`wjinxu`) for the speculative implementation
  and speculators-format support, Aman Gupta for conversion/separate-sidecar work,
  and later model-specific contributions including Tarek Dakhran. Prism's imported
  extensions retain Brian / `bri-prism` authorship.
- Prism Q2_0-G128 history records Pasha Khosravi for the core, CPU, CUDA, Metal,
  quantization, and row-dispatch series, with Brian / `bri-prism` contributing
  AVX-512 and follow-up integration/fixes.

Where an external model, conversion, importance matrix, or projector is used, its
own model card and license remain controlling for that artifact. In particular,
Qwen, Gemma, Unsloth, AtomicChat, and Prism model files are not relicensed by this
source repository.

## License

All upstream code remains under the **MIT License** as set out in `LICENSE`
(Copyright (c) 2023-2026 The ggml authors, and the respective contributors to the
projects listed above). Local modifications in this repository are likewise
released under the same MIT License. The original copyright notice and license
text are retained unmodified in `LICENSE`.

## Local modifications

Changes made in this fork are documented under `docs/development/` — see in
particular `docs/development/mtp-cpumoe-vision-fixes.md` for the Gemma 4 MTP +
`--n-cpu-moe` + vision (`--mmproj`) fixes.

Important provenance-oriented records include:

- [`docs/development/upstream-prism-convergence.md`](docs/development/upstream-prism-convergence.md)
  — what was merged, what was adapted, and the validation matrix;
- [`docs/development/bonsai-dspark-turboquant.md`](docs/development/bonsai-dspark-turboquant.md)
  — ternary/DSpark/TurboQuant implementation details;
- [`docs/development/thecodacus-prefetch-mmap-port.md`](docs/development/thecodacus-prefetch-mmap-port.md)
  — source comparison and port rationale;
- [`docs/development/launcher-config-ui.md`](docs/development/launcher-config-ui.md)
  — canonical launcher contract, UI integration, profile migration, and tests;
- [`docs/development/media-aware-speculative-decoding.md`](docs/development/media-aware-speculative-decoding.md)
  — media policy design, private position domains, exact-model results, and
  algorithm/source provenance;
- [`docs/development/pre-convergence-baseline-2026-08-21.md`](docs/development/pre-convergence-baseline-2026-08-21.md)
  and [`docs/development/upstream-convergence-ledger.md`](docs/development/upstream-convergence-ledger.md)
  — pre-merge baseline and capability decisions.

## Attribution policy for future changes

1. Keep `LICENSE`, this notice, imported commit authorship, and source links intact.
2. When porting a feature, name the source repository/branch or upstream PR in the
   commit message and the relevant development document.
3. Distinguish a direct port, an adaptation, and a new local implementation. Do not
   relabel inherited behavior as original work by this fork.
4. Preserve model/conversion provenance separately from source-code licensing.
5. If implementation is rewritten, retain credit for the design or investigation
   that materially informed it.
