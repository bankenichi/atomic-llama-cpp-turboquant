# Canonical launcher and configuration UI

The Atomic fork has one tracked launch contract and two user-facing entry points:

- `scripts/atomic-launcher.ps1` is the versioned implementation;
- ignored `run-gemma.ps1` supplies owner-local convenience switches;
- `tools/llama-config-ui` is a submodule that calls the tracked launcher through
  structured JSON.

This split is intentional. Model-stack rules, cache defaults, environment gates,
and validation live in one place. The wrapper and UI do not carry separate copies
of speculative-decoding semantics that can drift to removed flag names.

## Repository and runtime boundaries

Source stays in:

```text
C:\Users\kenic\Documents\atomic-llama-cpp-turboquant
```

Installed models and optional release binaries stay in:

```text
C:\Program Files\llamacpp
```

The tracked launcher first honors an explicit server path, then
`ATOMIC_LLAMA_SERVER`, then checks the convergence build, normal build, and Program
Files release executable. Model filenames are resolved beneath
`ATOMIC_MODEL_ROOT`, defaulting to Program Files. Explicit target, draft, and mmproj
paths override those filenames.

## Launcher actions

```powershell
scripts\atomic-launcher.ps1 -Action Describe -Stack ternary -Json
scripts\atomic-launcher.ps1 -Action Resolve  -Stack qwen    -Preset standard -Json
scripts\atomic-launcher.ps1 -Action Validate -Stack gemma26 -Preset mtp -Json
scripts\atomic-launcher.ps1 -Action Preview  -Stack ternary -Preset standard -Json
scripts\atomic-launcher.ps1 -Action Launch   -Stack ternary -Preset standard
```

`Describe` returns the profile catalog, executable capability probe, and supported
fork environment settings. `Resolve` returns the effective typed configuration.
`Validate` returns errors and warnings. `Preview` returns the exact executable,
literal argv array, redacted argv, environment, validation, and configuration.
`Launch` repeats validation before executing.

Overrides are supplied in a UTF-8 JSON file:

```json
{
  "port": 9000,
  "vision_enabled": true,
  "spec_enabled": true,
  "spec_draft_n_max": 4,
  "extra_arguments": ["--verbose"]
}
```

```powershell
scripts\atomic-launcher.ps1 `
  -Action Preview `
  -Stack ternary `
  -Preset standard `
  -ConfigPath .\local-overrides.json `
  -Json
```

The launcher always builds an argument array. It does not concatenate a command
string or evaluate user input. Raw arguments that conflict with structured model,
cache, context, speculative, or server fields fail validation.

## Owner wrapper

The ignored root wrapper retains the familiar model selectors and delegates to the
canonical launcher:

```powershell
.\run-gemma.ps1                    # Gemma 26B target + vision
.\run-gemma.ps1 -Mtp               # Gemma assistant MTP + vision
.\run-gemma.ps1 -g12               # Gemma 12B dense + vision
.\run-gemma.ps1 -Qwen              # Qwen embedded NextN
.\run-gemma.ps1 -Ternary           # Bonsai DSpark + vision
.\run-gemma.ps1 -Ternary -NoSpec   # target-only + vision
.\run-gemma.ps1 -Ternary -NoVision # DSpark text-only
.\run-gemma.ps1 -Ternary -FullKV   # full-f16 target diagnostic; Q8_0 draft
```

Add `-Preview` to any command to validate files and binary capabilities without
starting the server. The local wrapper uses `C:\Program Files\llamacpp` filenames
from the tracked catalog, including `mtp-gemma-4-26B-A4B-it.gguf` rather than the
retired `C:\llm-work` path.

## Stack behavior

### Ternary Bonsai

The standard profile uses the exact owner files, `draft-dspark`, four draft tokens,
target turbo4/turbo3 KV, lossless draft q8_0/q8_0 KV, 32K target context, and vision. Validation
requires the sidecar, limits the selected draft length to 16, and reports the
runtime-enforced 4096 draft positions per parallel slot. The launcher does not
duplicate GGUF type-42 recognition or DSpark metadata parsing; the C++ loader
remains authoritative.

### Qwen embedded NextN

The standard profile uses target turbo4/turbo3 KV, lossless draft q8_0/q8_0 KV, 28 CPU MoE
layers, and two synchronous draft tokens. No draft model path is emitted. Historical
`LLAMA_NEXTN_ASYNC`, `LLAMA_PIPELINE_DEPTH2`, `--spec-type nextn`, and
`--spec-type mtp` controls are not part of the converged runtime and must not return.

### Gemma

Gemma 26B standard is target-only. The `mtp` preset adds the external assistant,
draft turbo3/turbo3 KV, and current `draft-mtp` arguments. `op_offload = false` for
the assistant context remains a C++ invariant, not a launcher toggle. Gemma 12B is
dense, vision-capable, and target-only.

## Vision, context, and performance

The UI and launcher permit mmproj plus speculative configuration. They do not infer
that a loaded projector means the current request contains media. The server's
`has_media()` accessor reports real media to the speculative coordinator, which
selects the implementation's embedding-prefill, post-media boundary, compact text,
or stateless policy. Image answers and retained-image follow-ups can therefore
draft without the UI inventing its own gating rules.

Turbo K with context shifting produces a warning because the server must use
transactional reprefill. The configuration remains valid. Image-boundary snapping,
template marker preservation, and target/draft prompt-cache coherence are server
responsibilities and retain their existing regression tests.

Fork environment settings are first-class configuration:

| Setting | Environment | Notes |
|---|---|---|
| Expert prefetch | `GGML_SCHED_PREFETCH_EXPERTS` | MoE plus op offload; `1` selects the runtime's default three slots |
| mmap host registration | `GGML_CUDA_REGISTER_HOST=1` | Requires mmap; compared separately from `--no-mmap` prefetch |
| Adaptive Turbo KV | `TURBO_LAYER_ADAPTIVE=N` | unset uses the fork default; `0` is an explicit A/B override |

Before launch the launcher clears every environment setting it owns, then applies
the selected plan. A previous launch from the same terminal cannot silently leave a
disabled feature enabled.

## UI submodule

Initialize or update it with:

```powershell
git submodule update --init --recursive
git submodule update --remote tools/llama-config-ui
```

Run locally:

```powershell
cd tools\llama-config-ui
python server.py
```

Atomic mode calls `scripts/atomic-launcher.ps1`; the original generic
`llama-args.txt` editor remains collapsed under a compatibility section. Atomic mode
provides typed stack controls, capability refresh, immutable starter profiles,
mutable ignored user profiles, non-destructive legacy import, validation, preview,
managed start/stop, readiness, logs, metrics, and redacted diagnostics.

The UI stores its managed PID together with the OS process-creation marker. Recovery
requires both to match, preventing PID reuse from authorizing termination of an
unrelated process. Graceful stop targets that child process group; forced
termination is limited to the same verified process. Model inference runs with
stdout/stderr redirected so a backend fault cannot crash the controlling Codex
process.

## Profile migration

The UI can read the existing Program Files `profiles.json`. It does not modify the
source. Recognized old flags map to typed settings; unrecognized fields are retained
under `legacy_unknown`. Existing destination user profiles receive a timestamped
backup before an import merge.

Starter profiles contain no committed owner secrets. Mutable Atomic profiles,
active configuration, PID, logs, and diagnostics are ignored inside the submodule.

## Tests

Launcher tests:

```powershell
pwsh -NoProfile -File tests\test-atomic-launcher.ps1
```

UI tests and syntax checks:

```powershell
cd tools\llama-config-ui
python -m unittest discover -s tests -v
python -m py_compile server.py atomic_runtime.py
node --check app.js
node --check atomic.js
```

Browser acceptance covers Ternary defaults, Qwen profile switching, validation
messages, exact cache values, and DSpark rejection above 16 draft slots. Real model
acceptance uses the hidden managed child, readiness endpoint, a finite request
timeout, captured logs, and explicit stop.

## Provenance

`llama-config-ui` remains a separately versioned project by `bankenichi`; its Git
history and `NOTICE.md` preserve the original generic UI authorship. The Atomic
adapter is a local integration with this fork's tracked launcher. It does not claim
authorship of llama.cpp, AtomicBot, Prism, TurboQuant, or other inference work named
in the parent repository's `NOTICE.md`.
