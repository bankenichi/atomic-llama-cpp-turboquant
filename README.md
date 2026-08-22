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

This README is the operator's entry point, not a replacement for upstream's
general documentation. It explains what is unusual about this fork, which model
artifacts belong together, how the private features interact, and how to recognize
the common failure modes. The implementation history and source credits are in
[`NOTICE.md`](NOTICE.md); detailed engineering decisions live under
[`docs/development/`](docs/development/).

## How the pieces fit together

The server runs one authoritative target model. Optional components attach to it:

```text
request
  |
  +-- text --------------------> target model ----> verified tokens
  |                                ^     |
  |                                |     +-- hidden-state taps
  |                                |
  |                         speculative drafter
  |                         (DSpark, Gemma MTP,
  |                          or Qwen NextN)
  |
  +-- image/audio/video ------> mmproj + target model
                                      |
                                      +-- draft marked out of sync

next text request on that slot: clear/rebuild target + draft together, then resume drafting
```

Speculative output is never committed on trust. The drafter proposes tokens and
the target verifies them, so a rejected proposal changes performance, not model
semantics. Vision is deliberately target-only for the current drafters. The server
therefore tracks whether each slot's target and draft represent the same prefix;
this state travels with the secondary prompt cache and survives cache misses.

There are three supported draft layouts:

| Target | Draft artifact | CLI type | Notes |
|---|---|---|---|
| Ternary Bonsai 27B | Prism DSpark sidecar | `draft-dspark` | Block-diffusion draft, maximum 4096 positions per slot |
| Gemma 4 | Separate assistant GGUF | `draft-mtp` | Supports modern and historical `gemma4_assistant` files |
| Qwen 3.6 native-MTP | Embedded in target GGUF | `draft-mtp` | Reopens the same file with the draft architecture; no second path |

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

Prerequisites:

- Windows 11 and an NVIDIA driver supporting the installed CUDA toolkit.
- Visual Studio 2022 with Desktop development with C++ and the v143 toolset.
- CMake and a CUDA toolkit capable of compiling Blackwell `sm_120` code.
- Enough disk space for two build trees if CPU and CUDA regressions are run side
  by side. Model files are not stored in this repository.

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --config Release --target llama-server -j 8
ctest --test-dir build -C Release --output-on-failure
```

The only CI workflow is [`.github/workflows/windows-cuda.yml`](.github/workflows/windows-cuda.yml).
It builds the complete tool suite and publishes the rolling `latest` release.

For a host-only diagnostic build, use a separate directory so it cannot reuse
CUDA configuration accidentally:

```powershell
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu --config Release -j 8
ctest --test-dir build-cpu -C Release --output-on-failure
```

## Model artifacts

For the primary Prism deployment, download the matching base, DSpark, and mmproj
files from the
[`prism-ml/Ternary-Bonsai-27B-gguf` repository](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/tree/main).
Do not mix a projector or drafter from a differently converted target merely
because the architecture name matches: vocabulary, hidden-state taps, mask token,
and projector geometry must agree with the target.

The owner deployment stores the matching files under `C:\Program Files\llamacpp`:

```powershell
$BonsaiModel = 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf'
$BonsaiDraft = 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-dspark-Q4_1.gguf'
$BonsaiMmproj = 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-mmproj-Q8_0.gguf'
```

The exact published filenames can change independently of this source tree, so
the commands below use descriptive placeholders. Confirm the model card and file
list before copying paths into a launcher.

## Run Ternary Bonsai + DSpark + vision

The ignored owner wrapper delegates to the tracked canonical launcher and validates
the selected binary, files, draft mode, cache types, and DSpark limits:

```powershell
.\run-gemma.ps1 -Ternary
.\run-gemma.ps1 -Ternary -NoSpec        # target-only, vision retained
.\run-gemma.ps1 -Ternary -NoVision      # DSpark text-only
.\run-gemma.ps1 -Ternary -FullKV        # full-f16 target diagnostic; Q8_0 draft
.\run-gemma.ps1 -Ternary -Preview       # resolve and validate without launching
```

The equivalent native command is:

```powershell
build\bin\Release\llama-server.exe `
  -m 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf' `
  --model-draft 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-dspark-Q4_1.gguf' `
  --spec-type draft-dspark `
  --spec-draft-n-max 4 `
  --mmproj 'C:\Program Files\llamacpp\Ternary-Bonsai-27B-mmproj-Q8_0.gguf' `
  -ngl 999 -ngld 999 `
  -ctk turbo4 -ctv turbo3 -ctkd q8_0 -ctvd q8_0 -fa on `
  --ctx-size 32768 --parallel 1 --context-shift `
  --reasoning off
```

The normal Ternary target cache is `turbo4` K / `turbo3` V. DSpark uses lossless
`q8_0` K/V rather than spending memory on f16 draft KV. `-Ternary -FullKV` raises
only the target cache to f16 for controlled diagnostics; the draft cache remains
q8_0/q8_0. The CUDA suite still retains the separate F16 K / Q8_0 V 576x512
operator test because that layout is part of the fork's adaptive/compatibility
surface even though it is no longer the owner launcher's default.

Published Prism files that stored Q2_0-G128 under raw GGUF type 42 are detected by
physical tensor spans and normalized to this fork's collision-free runtime type
142. Raw type 42 remains standard group-64 Q2_0 when no span proves the legacy
layout; contradictory layouts are rejected. Historical private types 43-46 are
normalized to the current private IDs 241-244.

The type-42 rule is intentionally conservative. Some aligned tensor spans are
valid under both group sizes; in that case the reader keeps upstream's standardized
group-64 meaning rather than guessing. A legacy Prism shard needs at least one
physically decisive tensor. See
[`docs/development/bonsai-dspark-turboquant.md`](docs/development/bonsai-dspark-turboquant.md)
for the format and backend details.

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
  -ngl 999 -ngld 999 --n-cpu-moe 28 `
  -ctk turbo4 -ctv turbo3 -ctkd f16 -ctvd f16 `
  -fa on --context-shift --reasoning off
```

This is the owner's normal Qwen cache configuration: target K/V use
`turbo4`/`turbo3`, while the small embedded draft context remains explicitly f16.
All-f16 target KV is retained as a diagnostic profile, but is not the daily-driver
default. The PowerShell wrapper command is `./run-gemma.ps1 -Qwen`; portable shell
launchers are [`scripts/run-qwen36-27b-nextn-server.sh`](scripts/run-qwen36-27b-nextn-server.sh)
and [`scripts/run-qwen36-35ba3b-nextn-server.sh`](scripts/run-qwen36-35ba3b-nextn-server.sh).

## Canonical launcher

[`scripts/atomic-launcher.ps1`](scripts/atomic-launcher.ps1) is the shared launch
contract for the owner wrapper and configuration UI. It supports `Describe`,
`Resolve`, `Validate`, `Preview`, and `Launch` actions; non-launch actions return
structured JSON. Profiles live in
[`scripts/atomic-launcher/profiles.json`](scripts/atomic-launcher/profiles.json),
while explicit settings from the wrapper or UI take precedence. The launcher probes
the selected executable and fails preflight when a required Atomic option or
speculative type is absent instead of silently dropping it.

The configuration UI is versioned as the
[`tools/llama-config-ui`](tools/llama-config-ui) submodule. Initialize it with
`git submodule update --init --recursive`, then run `python server.py` from that
directory. It provides typed Gemma/Qwen/Ternary topology, binary capability
probing, validation, managed start/stop, readiness, logs, metrics, redacted
diagnostics, and non-destructive import of the older Program Files profiles. See
[`docs/development/launcher-config-ui.md`](docs/development/launcher-config-ui.md).

## Confirm the server before connecting a frontend

Start with a deterministic text request. This isolates model/draft loading from
chat-template and media issues:

```powershell
$body = @{
  model = 'local-model'
  messages = @(@{ role = 'user'; content = 'Reply with only the number 42.' })
  temperature = 0
  max_tokens = 16
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8080/v1/chat/completions' `
  -ContentType 'application/json' `
  -Body $body
```

Then send one real image request and another unrelated text request on the same
server. The post-media text request is the important cache-coherence check: logs
should show a full prompt rebuild before speculative drafting resumes. Frontends
such as SillyTavern should use the OpenAI-compatible Chat Completion endpoint at
`/v1`; that lets the server apply the model's own template and media markers.

## Performance switches

| Feature | Enable | Requirement |
|---|---|---|
| Expert H2D prefetch | `GGML_SCHED_PREFETCH_EXPERTS=1` | `--n-cpu-moe`; pinned host allocation is best |
| Register mmap pages | `GGML_CUDA_REGISTER_HOST=1` | mmap enabled; Windows and POSIX supported |
| Turbo adaptive KV | enabled by default | Turbo KV type; `TURBO_LAYER_ADAPTIVE=0` disables |

Keep dedicated VRAM below roughly 15 GB on the RTX 5080. Crossing that boundary
can spill per-decode scratch into shared memory and collapse throughput. For MoE
models, `--n-cpu-moe` is the main capacity/performance control.

These switches are opt-in because they change resource behavior:

- Expert prefetch creates a second same-device backend/stream and rotating staging
  slots. It is useful only when expert tensors are actually CPU-resident.
- Host registration pins mmap-backed CPU weight pages for DMA. Pinning consumes a
  finite OS resource; disable it first when diagnosing allocation failures.
- Adaptive TurboQuant promotes selected boundary layers. Disable it only for an
  A/B comparison, not as a routine compatibility workaround.

## Troubleshooting

`number of bitmaps does not match number of markers`
: The chat template dropped the internal media marker. Use this fork's current
  server/chat code and the matching mmproj; do not bypass `/v1/chat/completions`
  with a frontend-specific prompt formatter.

Draft loads but accepts zero tokens
: Confirm the target/draft pair, `--spec-type`, and draft precision. For DSpark,
  also check that the sidecar metadata names, block size, and mask token are valid.
  For Qwen, verify that the target actually contains the embedded NextN tensors.

Failure after an image followed by text
: Run with verbose server logs and look for `forcing full prompt rebuild`. Missing
  that transition means target/draft cache coherence regressed; do not work around
  it by globally disabling vision or prompt caching.

Sudden throughput collapse near full VRAM
: Increase `--n-cpu-moe` (offload more expert layers to CPU), reduce context or
  parallel slots, or reclaim draft/model cache memory. The useful limit is the
  point before Windows reports substantial shared-GPU-memory use, not the nominal
  16 GB capacity.

Unknown or conflicting raw tensor type 42
: Modern files should use type 142 for Q2_0-G128. Legacy Prism files are accepted
  only when their physical spans prove the group-128 layout. A mixed or corrupted
  shard is rejected rather than silently misread.

## Source lineage and credit

This fork combines work from several projects and contributor communities. In
particular, the inference framework comes from `ggml-org/llama.cpp`, the original
MTP/TurboQuant fork lineage comes through `AtomicBot-ai`, ternary Bonsai and Prism
DSpark compatibility come from `PrismML-Eng`, and expert-prefetch/mmap-pinning work
was ported from `thecodacus/llama.cpp`. Those are not merely inspirations: portions
of their implementations and history are present here. See [`NOTICE.md`](NOTICE.md)
for links, named contributors where known, imported commit anchors, and the policy
for preserving attribution.

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
- [`docs/development/launcher-config-ui.md`](docs/development/launcher-config-ui.md)
  — canonical launcher, owner wrapper, UI control plane, migration, and validation.

For general `llama.cpp` usage, model conversion, and backend documentation, use
the [upstream documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs).
