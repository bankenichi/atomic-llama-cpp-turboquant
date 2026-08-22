# Atomic launcher and configuration UI parity design

Date: 2026-08-21

Status: approved for implementation

## Purpose

Provide one authoritative way to configure and launch every owner-relevant runtime
path in this fork, then make `llama-config-ui` a complete control surface for that
launcher. Parity means operational support for the behavior implemented during the
upstream/Prism convergence, not merely adding argument names or example presets to
the UI.

The implementation has two storage boundaries:

- source, scripts, documentation, and the UI submodule live in
  `C:\Users\kenic\Documents\atomic-llama-cpp-turboquant`;
- installed binaries and model artifacts live in `C:\Program Files\llamacpp`.

The launcher and UI must support the existing Gemma 4 MTP and Qwen 3.6 NextN
workflows as well as Ternary Bonsai target-only, vision, and DSpark workflows.
Vision remains a core capability.

## Chosen architecture

Use a canonical-launcher architecture. A tracked, machine-independent PowerShell
launcher owns feature semantics, validation, environment construction, and final
argument construction. The ignored owner-specific `run-gemma.ps1` remains a thin
convenience entry point that supplies local defaults from `C:\Program Files\llamacpp`.

The UI must consume the launcher's structured interface instead of maintaining an
independent translation from controls to `llama-server` arguments. This eliminates
the current sources of drift: duplicated presets, stale speculative type names,
incorrect cache defaults, and UI controls that can describe configurations the
runtime cannot execute.

Two rejected alternatives are:

1. Continue constructing raw commands independently in the UI. This is flexible,
   but repeats feature logic and will drift whenever the fork changes.
2. Add only new profiles and fields to the existing UI. This is smaller, but does
   not provide validation, capability discovery, environment handling, or reliable
   support for the new execution paths.

## Repository layout and ownership

The Atomic repository will contain:

- a tracked launcher core under `scripts/`;
- the ignored machine-local `run-gemma.ps1`, updated to delegate to the core;
- launcher tests and tracked example configuration;
- `tools/llama-config-ui` as a Git submodule pointing to
  `https://github.com/bankenichi/llama-config-ui`;
- Atomic-side integration documentation and provenance.

The UI remains a separately versioned project. Changes are committed and pushed in
its own repository, after creating a backup branch at its current remote head. The
Atomic repository then records the tested submodule commit. Existing dirty files in
`C:\Program Files\llamacpp`, including the local UI `profiles.json`, are treated as
owner data: they may be read for migration but may not be overwritten or discarded.

The Atomic integration branch already has a convergence backup. Before pushing the
new launcher/UI work, create a fresh backup reference for the pre-UI state as well.

## Canonical launcher interface

The tracked launcher exposes both a human CLI and a stable machine interface. The
machine interface returns JSON and includes these operations:

- describe supported stacks, speculative engines, cache types, performance
  features, defaults, and constraints;
- resolve a profile plus explicit overrides into an effective configuration;
- validate paths, combinations, numeric bounds, executable capabilities, and
  owner-platform constraints without launching;
- emit the exact command, environment, warnings, and resolved paths;
- launch in the foreground or as a managed child process for the UI.

Configuration precedence is deterministic:

1. built-in stack defaults;
2. selected profile;
3. machine-local configuration;
4. explicit CLI or UI overrides.

Passwords, API keys, and other secrets are never written into committed defaults or
normal diagnostic output. Paths are passed as literal PowerShell arguments and are
never reconstructed through an unsafe command string.

The local `run-gemma.ps1` preserves its current convenient switches where practical
and delegates all resolution and validation to the tracked launcher. Existing
owner commands should continue to work. Newly supported Ternary commands use the
same `-NoSpec`, `-NoVision`, performance, context, port, and extra-argument concepts
as Gemma and Qwen rather than introducing a separate one-off script.

## Supported model stacks

### Gemma 4

Support target-only and assistant-head MTP operation, with or without vision. The
daily profile preserves the owner's tuned TurboQuant KV and CPU-MoE placement. MTP
remains optional because it can be break-even when too many experts are offloaded.
The launcher validates the assistant model and draft-specific options only when MTP
is enabled.

The default local model paths are updated to the actual files in
`C:\Program Files\llamacpp`, including the Gemma assistant head now stored there.

### Qwen 3.6

Support embedded NextN through `draft-mtp` and the converged synchronous multi-step
implementation. Historical `LLAMA_NEXTN_ASYNC` and `LLAMA_PIPELINE_DEPTH2` controls
do not exist in the converged runtime and must not be reintroduced or advertised.
The normal owner cache configuration is target K `turbo4` and target V `turbo3`;
it is not the all-f16 configuration currently shown in the README. Draft-cache
types remain explicit in the resolved configuration so command previews and
documentation cannot imply an unknown default.

The standard profile uses the owner's normal CPU-MoE placement and a draft maximum
of two tokens. A target-only profile remains available for diagnosis and comparison.

### Ternary Bonsai 27B

Support the exact local artifacts:

- `Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf`;
- `Ternary-Bonsai-27B-dspark-Q4_1.gguf`;
- `Ternary-Bonsai-27B-mmproj-Q8_0.gguf`.

The standard profile runs DSpark plus vision. Additional supported configurations
are target-only with vision, DSpark text-only, target-only text, and a full-f16
diagnostic configuration. These are real launcher modes, not documentation-only
examples.

DSpark validation enforces the runtime contract before launch:

- speculative type is `draft-dspark`;
- the draft model is present and distinct from the target;
- draft context does not exceed the sidecar's 4096-token limit;
- draft length does not exceed the implemented 16-token slot limit;
- required mask-token and architecture metadata are available to the runtime;
- draft GPU-layer and cache settings are internally consistent;
- incompatible embedded-NextN or Gemma-MTP controls are rejected.

The launcher does not duplicate GGUF legacy-type detection or DSpark graph logic.
Those remain runtime responsibilities. It does make the selected execution path and
its constraints explicit and prevents known-invalid launches.

## Feature-parity contract

The launcher and UI must expose and correctly operate the following feature groups.
Each feature is represented by structured capability data, validation rules,
effective-command output, and documentation.

### Speculative decoding

- Gemma assistant MTP;
- Qwen embedded NextN using the synchronous multi-step default;
- Ternary Bonsai DSpark;
- target-only fallbacks for all stacks;
- target and draft context, GPU layers, cache types, maximum/minimum draft length,
  confidence threshold, and other options supported by the built binary;
- cancellation-safe stop/restart through the server process boundary.

The runtime's media-aware behavior is preserved: loading an mmproj does not disable
text speculation, real media turns use the safe target path where required, and
later text turns may resume speculation only after the runtime has restored coherent
draft state. The UI explains and reports this behavior; it does not attempt to
implement speculative gating itself.

### Vision and context management

- mmproj selection and existence validation;
- explicit vision enable/disable state;
- prompt cache controls;
- context shifting and its effective state;
- target/draft cache coherence controls exposed by the runtime;
- clear warnings when a selected cache or model combination requires the reprefill
  path rather than in-place shifting;
- no UI rule that disables speculation merely because vision is configured.

The server remains authoritative for image-boundary snapping, marker preservation,
media-aware shifts, and target/draft cache rebuilds.

### KV cache and TurboQuant

- target K and V cache types;
- draft K and V cache types;
- `turbo2`, `turbo3`, `turbo4`, f16, q8, and other cache types reported by the
  current executable;
- attention and context options that affect supported cache combinations;
- standard, compact, and full-f16 configurations without silently changing the
  user's explicit values.

### Offload and performance controls

- target and draft GPU layers;
- CPU-MoE layer count;
- batch and micro-batch sizes;
- flash attention;
- expert prefetch through `GGML_SCHED_PREFETCH_EXPERTS`, including slot count;
- mmap host registration through `GGML_CUDA_REGISTER_HOST`;
- adaptive TurboQuant settings such as `TURBO_LAYER_ADAPTIVE` where applicable;
- relevant thread, NUMA, mmap, mlock, and device controls supported by the current
  binary.

Environment variables are first-class structured configuration, not free-form text
hidden in an argument field. The command preview shows them separately from CLI
arguments.

### Server, API, and chat behavior

- host, port, slots, parallelism, timeouts, and API controls;
- reasoning controls used by the owner's clients;
- chat-template and template-argument controls;
- metrics, health/readiness, logs, and server lifecycle;
- arbitrary additional arguments as an escape hatch, with a warning when they
  conflict with structured settings.

## Runtime capability discovery

On startup or explicit refresh, the launcher probes the selected `llama-server`
binary's help output and version information. It records which expected options and
enumerated values are actually available. The UI combines this probe with the
fork-specific capability description from the launcher.

This has three outcomes:

1. supported controls are enabled and produce the exact current spelling;
2. unsupported controls are disabled with an explanation;
3. a required option missing from a supposedly current Atomic binary is a failed
   preflight, not a silently omitted argument.

Probe results include the executable path, version/build identity when available,
and timestamp. A refresh is required after changing executable paths.

## UI behavior

The UI remains a lightweight local application but is reorganized around a typed
configuration model and launcher adapter rather than a monolithic flag map.

Primary views are:

- **Stack and models:** Gemma, Qwen, or Ternary; target, draft/assistant, and mmproj;
- **Execution:** speculative engine, draft controls, context, cache, and offload;
- **Performance:** fork-specific environment gates and tuning controls;
- **Server and chat:** networking, API, reasoning, templates, and prompt caching;
- **Review and run:** validation results, resolved command/environment, warnings,
  launch controls, readiness, metrics, and logs.

Changing stacks updates applicable controls but does not erase stored values for
other stacks. Advanced raw arguments remain available for newly introduced upstream
flags, but every current Atomic feature has a typed control and validation rule.

The UI backend invokes the canonical launcher for resolution, validation, preview,
and launch. It must not shell-interpolate user input. Process state contains the
child PID, start time, effective configuration, log locations, health state, and
exit result. Stop first requests graceful termination, then offers forced
termination only when the child does not exit within the documented timeout.

When metrics are enabled and exposed by the server, the UI displays available
speculative counters and basic throughput. Missing metrics do not block a launch;
the UI reports them as unavailable rather than fabricating zero values.

## Profiles and migration

Ship immutable starter profiles separately from mutable user profiles. Starter
profiles demonstrate supported configurations; mutable profiles contain local paths
and owner edits. Profile resolution uses a schema version and stable setting keys,
not HTML element IDs.

On first use, the UI can import the existing
`C:\Program Files\llamacpp\llama-config-ui\profiles.json`. Migration:

1. reads without modifying the source file;
2. creates a timestamped backup of the destination user-profile file if it exists;
3. maps recognized legacy names, including stale `mtp` speculative values, to the
   current typed representation only when the intended stack makes the mapping
   unambiguous;
4. preserves unknown fields in a migration report instead of dropping them;
5. never commits absolute owner paths into the submodule's starter profiles.

The UI also continues to support importing and exporting effective configuration in
a human-readable format suitable for diagnosis.

## Error handling and diagnostics

Validation errors are actionable and identify the setting, rejected value, and
expected constraint. Important examples include missing model files, an unavailable
binary, unsupported flags, incompatible speculative modes, excessive DSpark context
or draft length, invalid cache types, and conflicting raw overrides.

Warnings describe performance or fallback behavior without blocking a valid launch,
including high VRAM-risk configurations, disabled vision, full-f16 KV cost,
non-shiftable TurboQuant context behavior, and speculative fallback on media turns.

Every launch records a redacted diagnostic bundle containing:

- launcher and executable identity;
- resolved non-secret configuration;
- exact argument vector and environment keys;
- validation warnings;
- process exit code and relevant logs.

The UI provides a copy/export action for this bundle. It never records API keys or
other marked secrets.

## Documentation and provenance

Documentation must explain why the profiles and constraints exist, not merely list
commands. Update:

- the main README's Gemma, Qwen, and Ternary launch guidance;
- Qwen examples to reflect target `turbo4`/`turbo3` and the actual draft-cache
  behavior;
- a launcher/UI operating guide;
- development notes describing the canonical-launcher contract and feature matrix;
- `NOTICE.md` or the appropriate lineage document for the UI submodule and any code
  adapted from its existing implementation.

Preserve all existing AtomicBot, llama.cpp, PrismML, thecodacus, and other recorded
attributions. Preserve the UI project's license and original authorship. New or
ported non-obvious code receives comments that explain invariants, source lineage,
or platform constraints rather than narrating straightforward syntax.

## Testing strategy

### Launcher tests

- profile resolution and precedence;
- literal path and argument handling, including spaces in Program Files paths;
- Gemma, Qwen, and Ternary command snapshots;
- target-only, vision, speculative, and diagnostic variants;
- environment construction and redaction;
- capability-probe parsing;
- DSpark bounds and incompatible-mode rejection;
- Qwen cache defaults and current speculative type;
- dry-run/validation behavior with missing and present files.

### UI tests

- typed configuration serialization and schema migration;
- import without source modification;
- legacy `mtp` mapping only in unambiguous contexts;
- launcher request/response handling;
- validation and warning presentation;
- start, readiness, graceful stop, crash, and restart state transitions;
- command/environment preview and secret redaction;
- profile isolation between model stacks.

### Integration gates

- probe the current Windows CUDA `llama-server` build;
- launch and reach readiness for Gemma target-only and configured MTP;
- launch and reach readiness for Qwen NextN with the standard owner cache settings;
- launch and reach readiness for Ternary target-only, target-plus-vision, and
  DSpark-plus-vision;
- execute a text request for every stack;
- execute a Ternary image request and a later text request on the same server;
- confirm DSpark resumes safely after the media turn when the runtime permits it;
- exercise stop/restart and capture a diagnostic bundle;
- rerun the Atomic C++/CUDA regression suite when launcher or documentation work
  reveals any runtime defect requiring a source fix.

Real-model launch tests may use local artifacts and remain outside CI. Deterministic
launcher and UI tests are included in their respective repositories. The Atomic
Windows CUDA workflow remains the only repository CI workflow.

## Implementation sequence

1. Create fresh backup references for the Atomic and UI repositories.
2. Audit the current executable's help output and the UI's complete flag map.
3. Implement and test the tracked canonical launcher.
4. Convert the local `run-gemma.ps1` into a compatible owner-specific delegate and
   verify its Gemma, Qwen, and Ternary previews.
5. Correct tracked README and shell-wrapper Qwen guidance.
6. Add a clean `llama-config-ui` submodule checkout under `tools/` without touching
   the dirty Program Files checkout.
7. Implement the typed model, launcher adapter, feature controls, process lifecycle,
   migration, diagnostics, and tests in the UI repository.
8. Commit and push the UI repository, then pin that commit in Atomic.
9. Run real-model smoke tests and the relevant Atomic regression gates.
10. Finish operating documentation, provenance, comments, and backup-aware push of
    the Atomic branch.

## Out of scope

This work does not implement concurrent ComfyUI, ACE-Step, llama, or Kobold process
orchestration. Those remain long-horizon resource-management goals. The launcher
may preserve headroom-oriented settings and report memory-related warnings, but it
does not become a general workload scheduler.

The UI does not reimplement GGUF parsing, ternary kernels, DSpark graphs, media
embedding, context-shift transactions, or speculative acceptance. It configures,
validates, launches, and observes those runtime features through the Atomic binary.

## Completion rule

The work is complete only when the wrapper, tracked launcher, and UI can configure,
validate, launch, stop, and diagnose every listed Gemma, Qwen, and Ternary execution
path using the current Atomic binary; when the real owner model paths work from the
Documents worktree; when Qwen documentation matches the actual standard cache
configuration; when existing profiles are preserved through migration; and when
the UI submodule, attribution, tests, and integration documentation are committed
and pushed behind fresh backup references.
