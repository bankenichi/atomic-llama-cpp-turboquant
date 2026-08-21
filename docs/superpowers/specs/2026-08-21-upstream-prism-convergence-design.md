# Upstream and Prism convergence design

Date: 2026-08-21

Status: approved for implementation

## Purpose

Bring this personal fork to full feature parity with current `ggml-org/llama.cpp`,
preserve or reimplement every owner-relevant customization that upstream does not
fully replace, and then integrate complete Prism ternary and DSpark support. The
primary acceptance target is the exact published Prism model pair:

- `Ternary-Bonsai-27B-Q2_0.gguf`
- `Ternary-Bonsai-27B-dspark-Q4_1.gguf`

Vision is a core capability of the resulting model server. It is not an optional
follow-up. The converged implementation is incomplete until target-only vision,
media-aware context management, and safe speculative behavior around media turns
are all validated.

This remains a private, single-owner Windows/CUDA fork. There is no upstream PR
goal and no requirement to generalize owner-specific behavior beyond what is
needed to keep the code coherent and maintainable.

## Pinned source state

The convergence starts from these exact points:

- Current fork: `a637defb785e567cd4e595f516c195a22947d8a8`
- Common ancestor: `b3d758750a268bf93f084ccfa3060fb9a203192a`
- Upstream snapshot: `9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e`
- Prism legacy branch snapshot: `9ca265a57f85f2117942490f421f64a226dd9847`
- Prism modern `prism-v6-dspark-dflash` snapshot:
  `27462403a0cd7b9ca8943281cbe02566daefe2b3`

At design time, the current fork has 229 commits not in the common ancestor and
upstream has 1,764 commits after it. Prism's modern branch is 20 Prism commits
past an August 6 upstream base and 273 upstream commits behind the pinned
August 21 upstream snapshot.

The pinned upstream commit is the reproducible initial merge target. Immediately
before completion, upstream will be fetched again and any newer commits will be
integrated through the same gates so that "parity" is current at delivery.

## Non-negotiable outcomes

1. The final tree contains the complete current upstream implementation, except
   repository policy and packaging intentionally kept private to this fork.
2. No owner-relevant local feature is silently lost because upstream has a
   similarly named feature.
3. Standard upstream models and formats continue to work, including upstream
   group-64 `Q2_0`.
4. Modern Prism group-128 `Q2_0_G128` works on CPU and CUDA.
5. The exact legacy Bonsai group-128 files load without an external conversion
   step despite using historical on-disk tensor type 42.
6. Modern upstream DSpark remains supported.
7. The exact legacy Bonsai `dspark` sidecar runs with its complete algorithm,
   including target taps, Markov resampling, confidence pruning, and log-SNR
   conditioning.
8. Vision remains usable with the Bonsai target and with the existing Gemma/Qwen
   workflows. Media turns must never corrupt target or draft state.
9. Windows CUDA `sm_120` is the hard validation platform. Existing source paths
   for other backends are preserved or forward-ported where they are part of a
   retained local feature, but the repository keeps one Windows CUDA CI workflow.

## Convergence strategy

### Protected integration

Create a permanent safety tag at the current fork commit and perform all work on
a dedicated integration branch. Do not rewrite or destructively reset `main`.
Only converge `main` after every acceptance gate passes.

Before merging, capture:

- the current Release build result;
- available unit and server test results;
- representative Gemma 4 MTP, Qwen NextN, and vision launch commands;
- model-based smoke results when the owner's local model files are available;
- the final-tree diff and a feature inventory mapped to the authoritative fork
  documentation.

### Full upstream merge

Merge the pinned upstream commit into the integration branch. This is a real
history merge, not a selective upstream cherry-pick. It preserves non-conflicting
local changes automatically and exposes overlapping implementations as conflicts.

Conflict resolution follows these rules:

1. Prefer upstream's current architecture, public APIs, model classes, graph
   construction, server organization, and speculative framework.
2. Port required fork semantics into those current interfaces rather than
   retaining obsolete duplicate subsystems.
3. Treat each documented local behavior as a requirement until an upstream
   equivalent is demonstrated by code inspection and tests.
4. Drop an old local implementation only when it is superseded, not merely
   because it is inconvenient to merge.
5. Keep the private README, `NOTICE.md`, `AGENTS.md`, development notes, and the
   single Windows CUDA release workflow.

### Feature-equivalence ledger

Maintain a convergence ledger in `docs/development/` with one row per local
capability. Each row records:

- old files and commits;
- corresponding upstream implementation, if any;
- decision: adopt, augment, or reimplement;
- tests proving equivalent behavior;
- remaining platform limitations.

The ledger must cover at least:

- TurboQuant KV types `turbo2`, `turbo3`, and `turbo4`;
- TQ3/TQ4 model-weight formats and accelerated CUDA paths;
- Gemma 4 assistant/MTP loading, graph behavior, and async overlap;
- Qwen NextN loading, synchronous multi-step drafting, and optional async mode;
- real-media versus loaded-mmproj speculative gating;
- vision-safe context shifting for shiftable and non-shiftable KV types;
- string fallback for templates that drop typed `media_marker` parts;
- expert-weight prefetch overlap;
- mmap-backed host registration, including Windows page handling;
- CUDA/MSVC, OpenSSL, and self-contained release packaging fixes.

## Speculative state and context shifting

The two identified NextN defects are requirements for the modernized code, not
patches to preserve in the obsolete server implementation:

1. In-place context shifts must remove and reposition target and draft memory in
   lockstep.
2. Reprefill context shifts must truncate and reconstruct draft state from the
   retained target hidden states; stale draft KV may not survive.

The same invariant applies to prompt-cache chunk relocation and every other
operation that changes absolute positions.

The preferred modern interface owns a target/draft pair and exposes centralized
remove, shift, clear, cancel, and rebase operations. Server code must not perform
a target-only positional mutation when a paired draft context exists.

Reprefill performs the following transaction:

1. cancel or drain in-flight speculative work;
2. truncate target and draft state at the same retained boundary;
3. decode retained target tokens in bounded chunks with the hidden outputs the
   active drafter requires;
4. rebuild draft state at exactly the same absolute positions;
5. reset acceptance and pending-draft bookkeeping;
6. resume generation only after target/draft position invariants pass.

Gemma MTP, upstream MTP, NextN, DFlash, modern DSpark, and legacy Bonsai DSpark
may require different hidden-state inputs. The shared lifecycle interface may
delegate graph-specific rebuilding to each speculative implementation.

## Quantization type namespace

### Required encodings

The converged tree must distinguish three encodings:

| Encoding | On-disk type | Block layout | Required behavior |
|---|---:|---|---|
| Upstream `Q2_0` | 42 | group 64, 18 bytes | Native upstream support |
| Modern Prism `Q2_0_G128` | 142 | group 128, 34 bytes | Native Prism support |
| Legacy Prism/Bonsai `Q2_0` | 42 | group 128, 34 bytes | Detect and remap to internal type 142 |

The existing fork assigned private TurboQuant identifiers in the same numeric
region. Those identifiers must move to a documented private high range that does
not collide with upstream or Prism. Existing local TQ-weight GGUF compatibility
must be retained through explicit legacy decoding or normalization.

### Legacy Bonsai detection

Filename and model-name checks are not acceptable. Tensor type 42 is ambiguous
between two physical layouts, so the GGUF loader performs a tensor-directory
preflight before allocating tensors:

1. collect every type-42 tensor's dimensions, offset, the next distinct tensor
   offset, file alignment, and data-section/file boundary;
2. calculate the legal aligned span for both group-64 and group-128 layouts;
3. accumulate evidence across all type-42 tensors;
4. accept group 64 if the file is consistently group 64;
5. accept and internally remap to type 142 if it is consistently group 128;
6. reject mixed or ambiguous layouts with a diagnostic naming the conflicting
   tensors and calculated spans.

This logic must be covered by synthetic GGUF fixtures for both layouts and by the
actual Bonsai headers. New files produced by this fork use unambiguous modern
type 142 for group-128 data.

### Backend completeness

Merge and forward-port Prism's group-128 implementation for:

- reference quantize/dequantize;
- CPU traits, dot products, `get_rows`, and optimized x86 paths;
- CUDA conversion, dequantization, `get_rows`, MMVQ, and MMQ;
- quantizer selection, tensor row validation, and file-type reporting;
- backend and quantization tests.

Use the normal CUDA implementation on Blackwell `sm_120`. Do not route the RTX
5080 through an unverified Hopper-only WGMMA specialization.

## DSpark integration

### Preserve upstream DSpark

The pinned upstream already contains `draft-dspark` built on the current DFlash
framework. Merge and retain it without reducing its supported formats or model
families.

### Add exact legacy Bonsai architecture

The published Bonsai sidecar reports `general.architecture=dspark` and is not
merely a renamed modern DFlash file. It has:

- six draft blocks;
- context length 4096;
- block size 4;
- mask token 248319;
- target layers `[1, 16, 31, 46, 61]`;
- five concatenated 5120-wide target taps;
- rank-256 Markov heads;
- confidence head with Markov features;
- log-SNR conditioning;
- its own token embeddings and output projection.

Port the complete Prism legacy architecture and speculative state unless graph
equivalence with modern DFlash is proven. Required behavior includes:

- independent capture of every configured target layer;
- target-feature projection and normalization;
- the six-layer non-causal draft transformer;
- full block-diffusion masking and anchor behavior;
- sequential Markov resampling;
- confidence-based pruning;
- log-SNR embedding and bounds validation;
- quantized Markov heads;
- acceptance, rejection, rollback, cancellation, and context reuse;
- bounded draft context and graph reservation.

The recommended Q4_1 drafter contains a group-128 `Q2_0` token embedding, so
legacy group-128 loading is a prerequisite even when the rest of the drafter is
Q4_1.

## Vision as a core capability

Vision is validated throughout convergence rather than appended after text
generation works.

### Required behavior

1. A loaded mmproj must not disable speculation on text-only turns.
2. A turn containing real media must use a correctness-preserving path when a
   speculative implementation cannot consume image-position hidden states.
3. Media markers must survive chat templates whose capability detection fails
   or which ignore typed content parts.
4. Context-shift boundaries must never split an image chunk.
5. In-place K shifts must keep media/token bookkeeping aligned.
6. Reprefill must either re-encode retained media correctly or reject the shift
   before mutating state with a clear error. Silent partial-image handling is
   forbidden.
7. After an image turn, later text turns must use coherent target state and may
   resume speculation only after their drafter has been safely rebased.
8. Bonsai target-only vision with its published mmproj is an acceptance gate.
9. Existing Gemma 4 vision and MTP/NextN-safe media fallback behavior must not
   regress.

Initial Bonsai DSpark acceptance does not require drafting image tokens. It does
require correct target-only processing of media turns and safe return to text
drafting afterward. Native image-position DSpark drafting can be added only if
the model exposes all required target taps for those positions and equivalence is
demonstrated.

## Integration sequence

1. Protect and benchmark the current fork.
2. Merge the pinned upstream snapshot completely.
3. Restore buildability using the upstream architecture.
4. Complete the feature-equivalence ledger and restore all non-superseded local
   behavior, including vision and context-shift invariants.
5. Pass the existing model and feature regression suite on the converged
   upstream base.
6. Integrate Prism's modern group-128 implementation and DSpark extensions.
7. Add legacy type-42 group-128 detection/remapping.
8. Port the exact legacy Bonsai DSpark architecture and speculative state.
9. Validate target-only Bonsai text inference.
10. Validate Bonsai target-only vision.
11. Validate Bonsai DSpark text inference, rollback, context shift, image-turn
    fallback, and safe post-image resumption.
12. Fetch and merge any newer upstream commits, rerun every gate, update docs,
    and converge `main`.

## Testing and acceptance

### Build gates

- Configure with VS 2022/v143, CUDA enabled, target `sm_120`.
- Build the full Release tool suite, not only `llama-server`.
- Run the C/C++ test suite available on Windows.
- Run server unit tests applicable to the local environment.
- Produce the self-contained Windows archive and verify CUDA runtime DLLs.

### Quantization gates

- Reference round-trip and error-bound tests for group-64 and group-128 Q2.
- CPU dot-product tests against dequantized reference values.
- CUDA `get_rows`, MMVQ, and MMQ comparisons against CPU reference.
- Synthetic GGUF load tests for upstream type 42, modern type 142, legacy type
  42/group-128, ambiguous layout rejection, and mixed-layout rejection.
- Existing TurboQuant and TQ-weight tests after identifier migration.

### Speculative and context gates

- Target/draft position equality after remove, shift, cache reuse, and reprefill.
- Correct cancellation with in-flight async work.
- Zero-accept, partial-accept, and full-accept rollback.
- Repeated requests on the same server slot.
- Context shifting under shiftable KV and TurboQuant reprefill KV.
- Gemma 4 MTP and Qwen NextN real-model smoke tests when fixtures are available.

### Exact Bonsai gates

- Load the exact target without rewriting it.
- Generate coherent deterministic target-only text on CPU reference and RTX
  5080 CUDA.
- Load the exact Q4_1 drafter without rewriting it.
- Produce nonzero accepted DSpark drafts.
- In greedy mode, verification yields the same committed token stream as
  target-only greedy decoding.
- Draft allocations remain bounded by the draft's 4096-token limit rather than
  the target's maximum context.
- Run target-only image inference with the published mmproj.
- Run text, image, then text again in one conversation without corrupting KV,
  media bookkeeping, or speculative state.
- Exercise context shift after media with both a shiftable KV type and the
  configured non-shiftable/TurboQuant fallback behavior.

### Regression gates

- Existing Gemma 4 26B-A4B text and vision.
- Existing Gemma 4 assistant MTP fallback behavior around media.
- Existing Qwen NextN multi-step drafting.
- Existing TurboQuant KV and TQ-weight paths.
- Expert prefetch and mmap host registration remain opt-in and default behavior
  remains unchanged.

## Documentation deliverables

Update or add:

- the feature-equivalence convergence ledger;
- a current fork architecture and lineage note;
- quantization type-ID and legacy GGUF compatibility documentation;
- Bonsai text, vision, and DSpark invocation examples;
- Windows/CUDA build and validation instructions;
- measured limitations and any unsupported backend combinations.

`LICENSE` and `NOTICE.md` remain intact.

## Out of scope for this convergence

Concurrent ComfyUI, ACE-Step, or secondary llama/kobold resource orchestration is
a long-horizon goal. It must not distort or delay correct model, DSpark, and
vision implementation. The convergence may record memory measurements useful to
that future work, but it will not introduce workload managers or dynamic process
orchestration.

## Completion rule

The work is complete only when upstream parity, retained fork behavior, modern
and legacy ternary formats, exact Bonsai DSpark, and exact Bonsai vision all pass
their acceptance gates. A partial loader, external model rewrite, text-only
success, or unverified speculative output does not satisfy this design.
