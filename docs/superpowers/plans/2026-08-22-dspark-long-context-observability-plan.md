# DSpark long-context continuity implementation plan

Date: 2026-08-22

Design: `docs/superpowers/specs/2026-08-22-dspark-long-context-observability-design.md`

## 1. Add testable private-window arithmetic

- Add a small DFlash/DSpark window state and rollover decision helper to
  `common/speculative.h` / `common/speculative.cpp`.
- Make target-to-local translation, reserve checks, rollover chunk selection,
  and versioned state validation independently testable.
- Add focused cases to `tests/test-dspark-legacy.cpp` for no-op mapping, exact
  boundary, repeated rollover, oversized injection chunking, reset, and invalid
  restored payloads.

## 2. Integrate the window into DFlash/DSpark

- Initialize one mapping per speculative sequence using the effective
  per-sequence draft context.
- Reject a draft memory that cannot shift or cannot reserve a complete trained
  noise block.
- Translate text and pooled-media injections from absolute target positions to
  private draft positions.
- Roll the KV cache before injection and drafting using sequence remove/add.
- Remove provisional noise-block positions immediately after sampling so target
  verification exclusively commits the next state.
- Serialize and restore the mapping with the prompt cache; clear contradictory
  or unversioned state.
- Preserve the published 4096-position cap in
  `common_base_params_to_speculative()`.

## 3. Correct M-RoPE recurrent diagnostics

- Extract the position-continuity predicate into a narrow helper in
  `src/llama-memory-recurrent.h`.
- Keep the existing scalar warning condition.
- Treat multidimensional positions as not eligible for a scalar row-count
  continuity assertion, with a comment describing image temporal/spatial
  coordinates.
- Extend an existing lightweight recurrent test or add a small unit test that
  proves scalar gaps warn while M-RoPE repetition does not.

## 4. Improve effective configuration logs

- Promote DFlash/DSpark context, K/V, device, and flash information to a concise
  INFO line.
- Log window capacity, reserve, and rollover chunk once at initialization.
- Log each actual rollover once with sequence and origin movement.
- Rely on the existing post-validation `llama_context` INFO lines for effective
  target context and flash attention, and the KV-cache allocation lines for
  effective target K/V types; avoid duplicating misleading requested values.

## 5. Build and focused validation

- Run formatting/diff checks and focused DSpark, server-state, launcher, media
  speculation, recurrent, and backend layout tests.
- Build and test a CPU configuration for shared speculative and recurrent code.
- Build the complete Windows CUDA tool suite with VS2022 v143 and Blackwell
  `120a`, then run full CTest and the required focused CUDA operators.

## 6. Exact-model regression

- Launch `llama-server.exe` only as a hidden child with redirected output and a
  finite watchdog.
- Use the owner's exact Ternary Bonsai base, Q8_0 DSpark sidecar, and Q8_0
  mmproj, with Turbo4 K / Turbo3 V and the standard launcher flags.
- Exercise text beyond 4096 draft-local positions, image answer, retained-image
  follow-up, and a forced target context shift.
- Require continuing draft acceptance after rollover and after media reprefill;
  reject any memory-slot failure, permanent speculative disablement, or scalar
  recurrent warning spam from image batches.

## 7. Documentation and publication

- Update the convergence and speculative documentation with the private-window
  behavior, diagnostics, limits, and Atomic provenance.
- Update `AGENTS.md` non-regression requirements if the new invariant is not
  already explicit.
- Rebuild from a clean committed revision so runtime build identity matches the
  source under test.
- Run the repository's required code-review skill, address findings, create and
  publish a backup branch, then commit and push the implementation without
  force-pushing.
