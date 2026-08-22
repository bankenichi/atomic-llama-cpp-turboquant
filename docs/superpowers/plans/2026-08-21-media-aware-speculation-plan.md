# Media-aware speculative decoding implementation plan

Date: 2026-08-21

Design: `docs/superpowers/specs/2026-08-21-media-aware-speculation-design.md`

## Goal

Replace the request-wide media veto with per-implementation, per-sequence media
capabilities; enable safe image-answer and full-history speculation; and validate
every maintained speculative type without weakening target correctness, vision,
prompt-cache, or context-shift behavior.

## Task 1: Protect and baseline the repository

1. Commit this plan on the integration branch.
2. Create and push `backup/pre-media-speculation-20260821` at the exact
   pre-implementation commit without switching branches.
3. Record the current branch, clean-tree state, build directories, and available
   local target/draft/mmproj files.
4. Capture focused CPU test results and the current blanket-gate behavior as a
   regression baseline.

## Task 2: Introduce an implementation-level media contract

Files:

- `common/speculative.h`
- `common/speculative.cpp`
- focused speculative C++ tests

1. Add failing tests for policy naming, default fail-closed behavior, per-sequence
   eligibility, and mixed implementation chains.
2. Add a media policy enum and request facts that distinguish embedding-prefill,
   stateless, post-media-rebase, and unsupported implementations.
3. Track eligibility and synchronization per implementation and sequence.
4. Expose aggregate queries for whether a sequence has any eligible drafter.
5. Add transition diagnostics without per-token log spam.

## Task 3: Isolate batch processing and failures

Files:

- `common/speculative.cpp`
- `common/speculative.h`
- batch-filtering tests

1. Add failing tests proving one implementation failure does not short-circuit a
   later implementation.
2. Add filtered batch construction for per-sequence eligibility while preserving
   tokens, embeddings, positions, outputs, and sequence membership.
3. Make processing transactional per implementation/sequence and clear pending
   ownership on failure.
4. Add state-generation metadata needed to reject incompatible restores.

## Task 4: Replace server-wide media gating

Files:

- `tools/server/server-context.cpp`
- `tools/server/server-spec-state.h`
- `tests/test-server-spec-state.cpp`
- `tools/server/tests/unit/test_speculative.py`
- `tools/server/tests/unit/test_vision_api.py`

1. Add failing state tests for supported media, unsupported media, rebase,
   per-sequence failure, and cache restore.
2. Replace `server_slot::can_speculate()` media logic with the coordinator's
   per-sequence eligibility query.
3. Pass real token and embedding post-decode batches to eligible implementations.
4. Force a paired rebuild only when the selected implementation policy requires
   it, not whenever historical media exists.
5. Preserve real-media gating: merely loading an mmproj must never affect text
   speculation.

## Task 5: Validate and enable DFlash/DSpark embedding prefill

Files:

- `common/speculative.cpp`
- DFlash/DSpark focused tests
- server speculative/vision tests

1. Test token-only, embedding-only, mixed, and chunked target batches.
2. Verify every configured target tap and row enters the encoder in target order.
3. Verify draft KV insertion at the exact target positions.
4. Test zero/partial/full acceptance, rejection, rollback, and cancellation after
   media prefill.
5. Advertise embedding-prefill capability only after all focused tests pass.
6. Enable image-answer drafting and repeated full-history drafting by default.

## Task 6: Validate and enable stateless n-gram media operation

Files:

- `common/speculative.cpp`
- n-gram speculative tests

1. Test every maintained n-gram type against prompts containing media markers and
   expanded embedding rows.
2. Restrict learned history to real text tokens and committed generated tokens.
3. Test media-boundary truncation, cache restore, and chained operation after a
   stateful implementation becomes ineligible.
4. Advertise stateless media capability and verify exact target-token parity.

## Task 7: Implement MTP/NextN media bridging

Files:

- `common/speculative.cpp`
- `tests/test-speculative-mtp.cpp`
- relevant MTP graph/context tests
- server speculative/vision tests

1. Inventory the exact inputs required by shared-target MTP, independent Gemma
   assistant, and embedded Qwen NextN.
2. Add failing synthetic tests for embedding rows, post-media token rows, warm-up,
   and draft-local rebase.
3. Feed target features where the architecture defines them; never invent image
   token IDs.
4. Add a post-media rebase for architectures lacking a required image-row input.
5. Preserve `LLAMA_CONTEXT_TYPE_MTP` and Gemma assistant `op_offload = false`.
6. Test accepted drafts during the image answer and after full-history resend.

## Task 8: Implement EAGLE3 and draft-simple rebasing

Files:

- `common/speculative.cpp`
- EAGLE3 boundary-state tests
- draft-simple speculative tests

1. Add an explicit target-to-draft position mapping for rebased implementations.
2. Rebase EAGLE3 from the first architecture-valid post-media boundary and enforce
   its minimum warm-up state.
3. Build draft-simple's compact text projection without forwarding target mmproj
   embeddings or fake image tokens.
4. Keep rollback and acceptance in each implementation's local position domain.
5. Require deterministic target parity and nonzero useful drafts before automatic
   image-answer activation.

## Task 9: Integrate prompt cache and context shifting

Files:

- `tools/server/server-context.cpp`
- `tools/server/server-common.h`
- `tools/server/server-spec-state.h`
- prompt-cache and context-shift tests

1. Store media policy, state generation, target boundary, draft-local boundary,
   and synchronization with speculative cache state.
2. Reject incompatible cache restores without disturbing the target cache.
3. Test image-boundary snapping and chunk-aware erase for shiftable f16/q8 KV.
4. Test transactional Turbo/M-RoPE reprefill and reconstruct every eligible
   implementation before resuming.
5. Verify an image request in one parallel slot cannot affect text speculation in
   another slot.

## Task 10: Add observability and correct launcher diagnostics

Files:

- speculative/server metrics and logging code
- `scripts/atomic-launcher/AtomicLauncher.psm1`
- launcher tests and relevant documentation

1. Log implementation activation, skip reason, rebase, invalidation, and restore
   once per state transition.
2. Expose media-eligible proposal and acceptance counters in metrics/timings.
3. Correct the deprecated `--mlock --no-mmap` combination exposed by the audit to
   the unambiguous modern `--load-mode mlock` representation.
4. Add launcher regression tests proving the effective load mode and all Ternary
   DSpark/TurboQuant arguments.

## Task 11: Build and exact-model validation

1. Build and test the shared code on CPU.
2. Build the complete Windows CUDA Release tool suite for `sm_120` with VS 2022
   v143 and run the full CTest suite.
3. Run focused F16/Q8 separate-V and Q8/Q8 K-subview CUDA tests.
4. Run exact Bonsai + DSpark + mmproj + Turbo4/Turbo3 hidden-child tests for image
   answer, repeated full history, second image, cache, rollback, and forced shift.
5. Compare exact greedy committed tokens with target-only output and require
   nonzero accepted drafts.
6. Run exact historical Gemma assistant + mmproj and Qwen embedded NextN + mmproj
   regression sequences with accepted post-media drafts.
7. Run compatible EAGLE3, draft-simple, and every n-gram configuration available
   locally; record any model fixture constraint distinctly from code correctness.

## Task 12: Documentation, review, and publication

1. Update README, `docs/speculative.md`, `MTP.md`, `NEXTN.md`, convergence docs,
   launcher/UI help, and the feature ledger with behavior and measured results.
2. Preserve and extend source provenance for any imported implementation.
3. Review all non-obvious comments for the invariant and regression they protect.
4. Run diff checks and the complete relevant test matrix.
5. Commit coherent implementation units and review the final commit range.
6. Push the integration branch only behind the published backup ref; never force
   push.
7. Confirm local/remote commit parity and report exact validation evidence.

## Completion condition

Do not declare completion for a DSpark-only exception. Completion requires an
explicit tested policy for every maintained speculative type, no permanent
full-history media disablement, exact target-token parity, coherent cache/shift
state, and passing exact Bonsai, Gemma, and Qwen vision regressions.
