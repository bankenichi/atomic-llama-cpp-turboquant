# Atomic launcher and configuration UI parity implementation plan

Date: 2026-08-21

Design: `docs/superpowers/specs/2026-08-21-launcher-config-ui-parity-design.md`

## Goal

Deliver a tested canonical launcher and an updated `llama-config-ui` submodule that
can validate, launch, monitor, and diagnose every current Atomic Gemma, Qwen, and
Ternary execution path without duplicating command semantics.

## Task 1: Protect both repositories

1. Verify the Atomic worktree contains only the committed design/plan work.
2. Create and push a fresh Atomic backup branch at the pre-implementation commit.
3. Verify the Program Files UI checkout's dirty `profiles.json` remains untouched.
4. Create and push a UI backup branch at `origin/master` without switching or
   cleaning the dirty checkout.
5. Record both backup refs in the implementation notes.

## Task 2: Audit the executable and existing controls

1. Capture current `llama-server --help`, build/version output, and speculative
   type values from the convergence build.
2. Inventory launcher-relevant CLI flags and Atomic environment gates.
3. Audit the current UI flag map, process control, profile schema, and launcher
   assumptions.
4. Produce a feature matrix mapping each current Atomic feature to its canonical
   config key, validation rule, CLI/env representation, and UI control.

## Task 3: Build the canonical launcher with tests first

Files:

- `scripts/atomic-launcher.ps1`
- `scripts/atomic-launcher/AtomicLauncher.psm1`
- `scripts/atomic-launcher/profiles.json`
- `tests/test-atomic-launcher.ps1`

1. Add failing tests for profile precedence, stack defaults, literal Program Files
   paths, and deterministic argument ordering.
2. Implement typed configuration resolution for Gemma 26B, Gemma 12B, Qwen 35B,
   and Ternary Bonsai.
3. Add failing tests for Gemma MTP, Qwen embedded NextN, Ternary DSpark, vision,
   target-only operation, target/draft cache types, and environment gates.
4. Implement exact CLI argument and child-environment construction.
5. Add failing tests for invalid stack/mode combinations, missing files, DSpark
   context/draft limits, unsupported binary capabilities, and raw-argument
   conflicts.
6. Implement validation and executable capability probing.
7. Add JSON actions for `Describe`, `Resolve`, `Validate`, `Preview`, and `Launch`.
8. Ensure preview/diagnostic output is redacted and launch uses an argument vector,
   never an interpolated command string.
9. Run the launcher test suite and representative previews against real local paths.

## Task 4: Update the owner wrapper and tracked launch guidance

Files:

- ignored local `run-gemma.ps1`
- `README.md`
- `scripts/run-qwen36-27b-nextn-server.sh`
- `scripts/run-qwen36-35ba3b-nextn-server.sh`
- relevant development documentation

1. Preserve existing wrapper switches and convert it to a thin call into the
   canonical launcher.
2. Add `-Ternary` and consistent target-only, no-vision, full-f16, DSpark, context,
   offload, performance, and extra-argument controls.
3. Correct the Gemma assistant path and all model paths to
   `C:\Program Files\llamacpp`.
4. Make Qwen's normal target KV `turbo4`/`turbo3` and its draft-cache values
   explicit.
5. Correct README and portable shell examples so they distinguish the owner-tuned
   profile from full-f16 diagnostics.
6. Validate all wrapper modes through preview before starting real servers.

## Task 5: Add a clean UI submodule and preserve local data

Files:

- `.gitmodules`
- `tools/llama-config-ui` submodule pointer

1. Add the UI repository beneath `tools/llama-config-ui` from its remote URL.
2. Confirm this does not modify the dirty Program Files checkout.
3. Work only inside the clean submodule checkout.
4. Add default/user profile separation and a migration fixture based on the legacy
   schema without committing the owner's private paths or profile contents.

## Task 6: Implement the UI launcher adapter and typed schema

UI repository files will follow its existing Python/vanilla-JS organization, with
new focused modules where the current monolithic files obscure ownership.

1. Add backend tests for canonical-launcher JSON calls, argument-vector safety,
   timeout/error handling, and capability refresh.
2. Implement the launcher adapter using `subprocess` argument arrays.
3. Add schema tests for all Atomic configuration keys, starter profiles, stack
   isolation, versioning, import, and unknown-field preservation.
4. Implement typed profile storage and non-destructive legacy migration.
5. Add process-state tests for launch, readiness, graceful stop, forced stop,
   abnormal exit, and restart.
6. Implement managed server lifecycle, log capture, health checks, and diagnostic
   bundle export with secret redaction.

## Task 7: Implement complete feature controls

1. Add stack/model controls for Gemma, Qwen, and Ternary target/draft/mmproj
   topology.
2. Add speculative controls for Gemma MTP, synchronous Qwen NextN, Ternary DSpark,
   and target-only fallbacks. Do not expose the removed historical
   `LLAMA_NEXTN_ASYNC` or `LLAMA_PIPELINE_DEPTH2` paths.
3. Add vision, prompt cache, context shifting, target/draft KV, TurboQuant, GPU
   layers, CPU MoE, batching, and flash-attention controls.
4. Add structured Atomic environment controls for expert prefetch, slot count,
   host registration, and adaptive TurboQuant.
5. Add server/API/chat/reasoning/template controls and a conflict-aware advanced
   arguments escape hatch.
6. Add review/run output for validation, warnings, exact argv, environment,
   readiness, logs, and available speculative/throughput metrics.
7. Disable unavailable controls from live capability data and treat missing required
   Atomic features as preflight errors.
8. Add focused comments for media/speculation behavior, DSpark invariants,
   migration, process ownership, and Windows quoting.

## Task 8: UI verification and publication

1. Run Python tests, JavaScript checks, and a browser smoke test.
2. Verify profile import against a copy of the legacy local file and confirm the
   source hash is unchanged.
3. Exercise preview and managed-launch flows against the Atomic convergence binary.
4. Update UI README, license/attribution references, and operating instructions.
5. Commit and push the UI repository on its normal branch, preserving the backup
   ref.
6. Pin the tested UI commit in the Atomic submodule.

## Task 9: Real-model and regression validation

1. Run target-only Ternary text.
2. Run Ternary target plus mmproj image processing.
3. Run Ternary DSpark plus vision, including text/image/text state recovery.
4. Run Qwen standard NextN with target `turbo4`/`turbo3`.
5. Run representative Gemma target-only and MTP previews/smokes.
6. Verify start, readiness, logs, metrics availability, stop, and restart through
   the UI.
7. If any runtime defect is exposed, fix it with focused C++ regression tests and
   rerun the relevant Windows CPU/CUDA suite.

## Task 10: Documentation, review, and delivery

1. Update Atomic README, development guides, docs map, and provenance/NOTICE as
   required.
2. Document starter profiles, full configuration behavior, warnings, model paths,
   UI setup, submodule update commands, and troubleshooting.
3. Run diff checks, launcher/UI tests, relevant C++ tests, and final real-model
   smoke tests.
4. Review all new code for correctness, security, maintainability, comments, and
   preservation of vision/runtime invariants.
5. Commit Atomic changes in coherent units.
6. Push behind the fresh backup ref and confirm remote branch/worktree parity.
7. Report exact commits, backup refs, test results, and any measured limitations.
