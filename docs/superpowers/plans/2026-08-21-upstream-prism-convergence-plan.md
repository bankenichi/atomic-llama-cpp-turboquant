# Upstream and Prism convergence execution plan

Design contract:
`docs/superpowers/specs/2026-08-21-upstream-prism-convergence-design.md`

## Safety and branch policy

- Code baseline tag: `pre-upstream-parity-a637defb`
- Backup branch: `backup/pre-upstream-convergence-20260821`
- Working branch: `integration/upstream-prism-convergence`
- Do not rewrite or force-update `main`.
- Do not push until the backup branch exists locally and the working branch has
  at least reached a buildable checkpoint.
- Keep commits organized by convergence layer so failures can be bisected.

## Milestone 0: baseline and inventory

- [x] Record the current compiler, CUDA, CMake generator, and v143 toolset.
- [x] Build the pre-merge Release tree and run available tests.
- [x] Record current model launchers and locally available model fixtures.
- [x] Create `docs/development/upstream-convergence-ledger.md` from the local
      feature inventory.
- [x] Record the exact upstream and Prism commit tips.

Exit condition: the current fork can be restored and its observable behavior is
recorded before conflicts are resolved.

## Milestone 1: full upstream merge

- [x] Merge pinned upstream commit
      `9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e` with a merge commit.
- [x] Resolve structural conflicts in favor of current upstream architecture.
- [x] Preserve the private README, notice, agent instructions, development docs,
      and single Windows CUDA workflow.
- [x] Restore a clean configure with VS/v143 and CUDA `sm_120`.
- [x] Build CPU libraries and core tests before repairing optional backends.
- [x] Build CUDA libraries, `llama-cli`, and `llama-server`.
- [x] Run the upstream test suite available on Windows (46/46 pass).

Exit condition: complete upstream is present and the tree builds on the owner's
Windows/CUDA platform, even if some fork-only features are temporarily disabled
behind explicit diagnostics.

## Milestone 2: local feature equivalence

For every ledger row, choose and document exactly one outcome:

- adopt the upstream implementation;
- augment upstream with missing owner-required behavior;
- reimplement the local feature against modern interfaces.

Required feature groups:

- [ ] Gemma 4 assistant/MTP loading, scheduling, output graph, and async overlap.
- [ ] Qwen NextN multi-step default and optional async implementation.
- [ ] Paired target/draft memory lifecycle and context-shift correctness.
- [ ] Media-aware speculative gating based on real media rather than mmproj.
- [ ] Vision-safe in-place shift and non-shiftable-KV reprefill.
- [ ] Chat-template media-marker fallback.
- [ ] TurboQuant KV types and attention paths.
- [ ] TQ3/TQ4 weight formats and CUDA paths.
- [ ] Expert-prefetch overlap.
- [ ] mmap host registration with Windows page handling.
- [ ] Required MSVC, CUDA, OpenSSL, and release-packaging fixes.

Exit condition: all documented local behavior is either proven upstream-equivalent
or restored, with focused tests for every fork augmentation.

## Milestone 3: quantization namespace convergence

- [ ] Preserve upstream group-64 `Q2_0` type 42 unchanged.
- [ ] Integrate modern Prism `Q2_0_G128` type 142.
- [ ] Move fork-private TurboQuant runtime types to a documented high range.
- [ ] Preserve legacy TQ3/TQ4 GGUF loading through explicit normalization.
- [ ] Implement GGUF physical-layout preflight for historical type-42 group-128
      files.
- [ ] Reject ambiguous and mixed encodings before allocation.
- [ ] Make new group-128 output use type 142.
- [ ] Add synthetic group-64, group-128, legacy, ambiguous, and mixed fixtures.

Exit condition: upstream, modern Prism, legacy Bonsai, and existing fork GGUFs
have unambiguous non-colliding interpretations.

## Milestone 4: Prism backend integration

- [ ] Forward-port the Prism group-128 reference implementation.
- [ ] Forward-port CPU traits, dot products, `get_rows`, and x86 optimization.
- [ ] Forward-port CUDA conversion, dequantization, `get_rows`, MMVQ, and MMQ.
- [ ] Port quantizer selection, row validation, and file-type reporting.
- [ ] Exclude unverified Hopper-only WGMMA dispatch on Blackwell `sm_120`.
- [ ] Run CPU/CUDA numerical comparisons and backend tests.

Exit condition: the exact Bonsai target loads and performs target-only text
generation through validated group-128 CPU and CUDA kernels.

## Milestone 5: exact Bonsai DSpark

- [ ] Preserve modern upstream DFlash/DSpark behavior and tests.
- [ ] Add legacy `dspark` architecture metadata and tensor mappings.
- [ ] Port target-layer capture and feature staging.
- [ ] Port the six-layer non-causal draft graph.
- [ ] Port masked block-diffusion and anchor behavior.
- [ ] Port Markov resampling and quantized Markov heads.
- [ ] Port confidence pruning and log-SNR conditioning.
- [ ] Implement full accept/reject/rollback/cancel/rebase lifecycle.
- [ ] Bound draft context and graph reservation to the sidecar's 4096 context.
- [ ] Add synthetic graph tests and real-model smoke tests.

Exit condition: the exact Q4_1 sidecar loads without rewriting, emits accepted
drafts, and greedy verification commits the same tokens as target-only greedy
decoding.

## Milestone 6: vision acceptance

- [ ] Load the published Bonsai mmproj with the exact ternary target.
- [ ] Validate target-only image inference.
- [ ] Validate text-only speculation while an mmproj is loaded.
- [ ] Validate safe speculative fallback on a real media turn.
- [ ] Validate text -> image -> text in one conversation.
- [ ] Rebase and resume speculation after media only when draft state is coherent.
- [ ] Validate image-aligned in-place context shift.
- [ ] Validate non-shiftable/TurboQuant media reprefill or a pre-mutation explicit
      rejection for unsupported retained-media cases.
- [ ] Re-run Gemma 4 vision and media-marker regression tests.

Exit condition: Bonsai and existing model vision workflows satisfy every vision
requirement in the design contract.

## Milestone 7: final parity and delivery

- [ ] Fetch upstream again and merge changes newer than the pinned snapshot.
- [ ] Re-run the feature-equivalence audit for the final upstream tip.
- [ ] Build the full Release tool suite with CUDA `sm_120`.
- [ ] Run C/C++, server, quantization, speculative, context, and vision tests.
- [ ] Package the self-contained Windows release and verify CUDA DLLs.
- [ ] Update README, lineage, build, performance, Bonsai, DSpark, and vision docs.
- [ ] Run a final code review and resolve all correctness findings.
- [ ] Merge the integration branch into `main` without rewriting history.
- [ ] Push the backup branch/tag before or with the converged `main` update.

Exit condition: all design acceptance gates pass on the final upstream tip and
the remote contains a recoverable pre-convergence branch/tag.
