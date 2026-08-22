# Atomic llama.cpp

Personal Windows/CUDA fork of
[`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant),
itself derived from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
This repository is maintained for one private deployment and is not intended to
submit changes upstream. See [`NOTICE.md`](NOTICE.md) for lineage and attribution.

The current primary target is Prism ML's
[`Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf):
ternary Q2_0-G128 weights, DSpark block-diffusion drafting, and vision in one
`llama-server`. Gemma 4 vision with an external assistant head and Qwen 3.6
embedded NextN remain first-class regression targets.

## Fork capabilities

- Current `llama.cpp` server, model, backend, and multimodal architecture.
- Prism Bonsai/Q2_0-G128 model loading and the published legacy raw-type-42 GGUFs.
- `draft-dspark` speculative decoding, with strict metadata validation and a
  4096-position draft-context limit per parallel slot.
- Unified `draft-mtp` decoding for Gemma assistant sidecars and embedded Qwen
  NextN heads.
- Vision-aware speculative lifecycle: text turns draft while an mmproj is loaded;
  real media turns cannot leave a stale draft context; the next text turn rebuilds
  target and draft together.
- Vision context shifting for f16/q8 KV plus transactional reprefill for TurboQuant
  and M-RoPE layouts.
- TurboQuant KV (`turbo2`, `turbo3`, `turbo4`) and TQ3/TQ4 weight formats on CPU
  and CUDA, including adaptive boundary-layer promotion.
- CPU-offloaded MoE expert prefetch and Windows mmap host registration.
- Windows CUDA CI for Blackwell (`sm_120`) with a self-contained runtime bundle.

The vision behavior is a compatibility requirement, not an optional extension.
Changes to prompt caching, speculative decoding, chat templates, KV shifting, or
model loading must preserve text and image turns on the same server.

## Build on the owner machine

CUDA builds must use the Visual Studio 2022 v143 toolset. Newer MSVC toolsets can
crash `nvcc` even when host-only compilation succeeds.

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --config Release --target llama-server -j 8
ctest --test-dir build -C Release --output-on-failure
```

The only CI workflow is [`.github/workflows/windows-cuda.yml`](.github/workflows/windows-cuda.yml).
It builds the complete tool suite and publishes the rolling `latest` release.

## Run Ternary Bonsai + DSpark + vision

```powershell
build\bin\Release\llama-server.exe `
  -m C:\models\Ternary-Bonsai-27B-Q2_0.gguf `
  --model-draft C:\models\Ternary-Bonsai-27B-dspark-Q4_1.gguf `
  --spec-type draft-dspark `
  --spec-draft-n-max 4 `
  --mmproj C:\models\Ternary-Bonsai-27B-mmproj-Q8_0.gguf `
  -ngl 99 -ngld 99 `
  -ctk f16 -ctv turbo2 -fa on `
  --ctx-size 32768 --parallel 1 --context-shift `
  --reasoning off
```

`turbo2` uses its adaptive policy by default. Boundary V-cache layers promote to
Q8_0 while K remains f16; the restricted CUDA build contains and tests this mixed
576x512 attention path. Set `TURBO_LAYER_ADAPTIVE=0` only for a controlled A/B test.

Published Prism files that stored Q2_0-G128 under raw GGUF type 42 are detected by
physical tensor spans and normalized to this fork's collision-free runtime type
142. Raw type 42 remains standard group-64 Q2_0 when no span proves the legacy
layout; contradictory layouts are rejected. Historical private types 43-46 are
normalized to the current private IDs 241-244.

## Run Gemma 4 assistant MTP + vision

```powershell
build\bin\Release\llama-server.exe `
  -m C:\models\gemma-4-target.gguf `
  --model-draft C:\models\gemma-4-assistant.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --mmproj C:\models\gemma-4-mmproj.gguf `
  -ngl 999 -ngld 999 --n-cpu-moe 14 `
  -ctk turbo4 -ctv turbo2 -ctkd turbo3 -ctvd turbo3 `
  -fa on --context-shift --reasoning off
```

Both modern `gemma4-mtp` files and historical `gemma4_assistant` GGUFs from this
fork are supported. The legacy spelling is normalized during load while retaining
its old metadata and tensor names.

## Run Qwen embedded NextN

For a combined Qwen GGUF containing its native NextN tensors, do not pass a second
model path. The same model is reopened with the MTP architecture automatically.

```powershell
build\bin\Release\llama-server.exe `
  -m C:\models\Qwen3.6-Native-MTP.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  -ngl 999 -ngld 999 --n-cpu-moe 12 `
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 `
  -fa on --context-shift --reasoning off
```

The checked-in launcher is [`scripts/run-qwen36-27b-nextn-server.sh`](scripts/run-qwen36-27b-nextn-server.sh).

## Performance switches

| Feature | Enable | Requirement |
|---|---|---|
| Expert H2D prefetch | `GGML_SCHED_PREFETCH_EXPERTS=1` | `--n-cpu-moe`; pinned host allocation is best |
| Register mmap pages | `GGML_CUDA_REGISTER_HOST=1` | mmap enabled; Windows and POSIX supported |
| Turbo adaptive KV | enabled by default | Turbo KV type; `TURBO_LAYER_ADAPTIVE=0` disables |

Keep dedicated VRAM below roughly 15 GB on the RTX 5080. Crossing that boundary
can spill per-decode scratch into shared memory and collapse throughput. For MoE
models, `--n-cpu-moe` is the main capacity/performance control.

## Documentation

- [`AGENTS.md`](AGENTS.md) — authoritative repository invariants and owner setup.
- [`docs/development/upstream-prism-convergence.md`](docs/development/upstream-prism-convergence.md)
  — convergence design, compatibility rules, and acceptance matrix.
- [`MTP.md`](MTP.md) — current Gemma/Qwen `draft-mtp` behavior.
- [`NEXTN.md`](NEXTN.md) — Qwen embedded NextN operation.
- [`docs/speculative.md`](docs/speculative.md) — generic speculative CLI.
- [`docs/development/context-shift-with-vision.md`](docs/development/context-shift-with-vision.md)
  — vision/KV shifting and reprefill.
- [`docs/development/performance-features.md`](docs/development/performance-features.md)
  — MoE prefetch, mmap pinning, and TurboQuant runtime controls.

For general `llama.cpp` usage, model conversion, and backend documentation, use
the [upstream documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs).
