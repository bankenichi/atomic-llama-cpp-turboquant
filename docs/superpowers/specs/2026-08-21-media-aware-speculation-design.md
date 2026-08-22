# Media-aware speculative decoding design

Date: 2026-08-21

Status: implemented; full build and regression validation in progress

## Purpose

Make vision and speculative decoding coexist correctly throughout a conversation.
The server currently disables every speculative implementation whenever the
request contains real media. Because OpenAI-compatible clients normally resend
the complete conversation, the original image remains in every later request and
speculation is effectively disabled for the rest of that conversation.

That behavior is a defect. Media safety differs by speculative algorithm and must
be expressed, implemented, and tested at the algorithm boundary. A request-wide
`has_media()` veto is neither accurate nor sufficiently precise.

The primary acceptance target is the owner's exact Ternary Bonsai 27B target,
DSpark sidecar, and mmproj configuration. The work also covers draft-simple,
EAGLE3, MTP/NextN, DFlash/DSpark, every maintained n-gram implementation, and
chains containing more than one speculative implementation.

## Existing behavior and root cause

`server_slot::can_speculate()` in `tools/server/server-context.cpp` currently
requires the request to contain no media. The same predicate controls prompt
mirroring, generation, context shifting, and speculative callbacks. A media
request therefore advances only the target context and marks the slot's draft
state unsynchronized. A later request can recover only if its submitted prompt
contains no media, which full-history multimodal clients generally never do.

The blanket gate also hides important differences among the implementations in
`common/speculative.cpp`:

| Implementation | Current media behavior | Design requirement |
|---|---|---|
| DFlash / DSpark | `process()` already accepts target token or embedding batches and injects target-layer features at their actual positions, but the server prevents media batches from reaching it | Validate the existing embedding path and enable it for image prefill and image-answer drafting |
| MTP / embedded NextN | Explicitly skips embedding batches | Add a model-correct target-feature bridge or a verified post-media rebase; never pretend an image embedding is a token |
| EAGLE3 | Explicitly skips embedding batches | Add a verified feature bridge or post-media rebase appropriate to EAGLE state |
| draft-simple | Sends target batches to an independent draft context; target mmproj embeddings are not generally compatible with that model | Add a text projection with independent draft positions, or use a compatible draft-side media encoder when one is actually loaded |
| n-gram variants | Do not maintain model KV and generate proposals from token history | Prove stateless media safety and keep them eligible during media requests |

`common_speculative_process()` currently combines implementations with a
short-circuiting boolean expression. One failure can prevent later
implementations from seeing a batch. This is incompatible with per-implementation
media handling and will be replaced with isolated processing and failure state.

## Chosen architecture

### Per-implementation media policy

Every `common_speculative_impl` exposes an effective media policy. The policy is
determined from both the algorithm and the loaded model; it is not inferred from
the CLI name alone.

The supported policies are:

- **Embedding prefill:** consumes target-produced embedding batches and all
  algorithm-required target features at their real target positions. This is the
  intended DFlash/DSpark path.
- **Stateless:** keeps no model-side prompt state and can propose tokens without
  mirroring image embeddings. This is the intended n-gram path.
- **Post-media rebase:** cannot ingest the media embedding itself, but can build a
  coherent private state from a defined boundary after media. Its draft-position
  mapping is independent of target absolute positions where necessary.
- **Unsupported:** cannot safely propose for this request. It remains loaded but
  is ineligible until a later request or rebuild establishes a supported state.

The policy query must explain why an implementation is ineligible so the server
can produce a useful diagnostic. Unknown implementations and unproven model
combinations default to unsupported.

### Eligibility is per implementation and per sequence

Media in one server slot must not disable speculation in another parallel slot.
The speculative coordinator therefore tracks eligibility, synchronization, and
the last successful implementation for each implementation/sequence pair.

At request start, the coordinator receives request facts including whether real
media is present, whether the prompt cache restored compatible speculative state,
and which media boundaries are new or reused. It selects eligible implementations
for that sequence. `begin`, `process`, and `draft` operate only on eligible
implementation/sequence pairs.

When a batch contains multiple sequences, the coordinator creates a filtered
view for each implementation. It retains every applicable row, restricts its
sequence membership to eligible sequences, and preserves token, embedding,
position, output, and sequence metadata. A shared batch row must never advance an
ineligible sequence accidentally.

A chain remains useful when only part of it supports media. For example, an
unsupported draft-simple model may be skipped while a verified n-gram
implementation continues to propose. The coordinator tries eligible
implementations in configured order and records which implementation produced
each pending draft.

### Separate target and draft position domains

Embedding-prefill algorithms retain the target's absolute positions. Algorithms
that use a post-media rebase may maintain a compact draft-side prompt which omits
unrepresentable image embeddings. These implementations own an explicit mapping
between target commit positions and draft-local rollback positions.

The mapping is never guessed from token counts. It is updated only when target
tokens are committed, prompt-cache state is restored, or a documented media
boundary is rebased. Target verification always occurs at target positions;
draft rollback always occurs in that implementation's local position domain.

This separation prevents two unsafe shortcuts:

- passing target-sized mmproj embeddings into an unrelated draft model;
- creating holes in draft KV while claiming that target and draft positions are
  synchronized.

### Algorithm-specific bridges

#### DFlash and DSpark

The existing embedding-aware `process()` path is the starting point. It must
capture every configured target-layer feature for both token and image rows,
construct the DFlash encoder input with identical row ordering, and inject draft
KV at the exact positions supplied by the target batch.

The path becomes eligible only after tests prove that mixed token/embedding
batches, chunked image prefill, cache reuse, rejection rollback, and context
shift preserve target/draft coherence. Once those gates pass, image-answer
drafting is enabled by default without a new CLI flag.

#### MTP and embedded NextN

MTP must not synthesize token IDs for image rows. The implementation will use the
target features the loaded MTP architecture actually consumes. If a model needs
an input that does not exist for image positions, the implementation establishes
a post-media boundary and warms the MTP state from subsequent target token rows.

Shared-target-memory MTP and independent assistant contexts may use different
bridges. Legacy Gemma assistant behavior and modern `LLAMA_CONTEXT_TYPE_MTP`
remain separate supported cases. Embedded Qwen NextN must keep using its normal
target context and graph selection. Both cases must resume safely within a
full-history multimodal request; the existence of an earlier image cannot remain
a permanent veto.

#### EAGLE3

EAGLE3 receives only the target features and token-aligned inputs defined by its
architecture. When image rows cannot supply a required token-aligned input, the
coordinator rebases EAGLE3 at the first valid post-media boundary and records the
new draft-local origin. Drafting begins only after the minimum warm-up state for
the loaded EAGLE model is present.

#### Draft-simple

An independent draft model receives neither foreign mmproj embeddings nor fake
image tokens. Without a compatible draft-side media encoder, it builds a compact
text projection around stable media boundaries and owns a draft-local position
domain. Its proposals remain correctness-safe because every token is verified by
the fully multimodal target, but the implementation must meet a nonzero-utility
gate before it is enabled automatically for image answers.

If a compatible draft-side media encoder is provided in the future, that is a
separate embedding-prefill policy and must validate model dimensions, embedding
semantics, and template/media expansion before use.

#### N-gram implementations

N-gram implementations remain independent of model KV. They may use only real
text-token history and committed generated tokens; internal media markers and
expanded image placeholder rows cannot be learned as normal token continuations.
The target verifies every proposal. After focused tests confirm this contract,
n-gram drafting remains active during the image answer and all later turns.

## Request and batch flow

1. The server parses the request and records real media boundaries separately
   from the fact that an mmproj is loaded.
2. Prompt-cache lookup restores target state and compatible per-implementation
   speculative metadata. A cache hit never infers synchronization solely from
   `has_media()`.
3. The speculative coordinator evaluates each implementation's effective media
   policy for the affected sequence.
4. Target prefill runs normally. Post-decode callbacks send token and embedding
   batches to every eligible stateful implementation through filtered views.
5. Each implementation either commits the complete batch transaction or
   invalidates only its affected sequence state.
6. Eligible implementations propose in configured order. The target verifies all
   proposals using the normal speculative acceptance path.
7. Acceptance, rejection, rollback, cache save, cache restore, and context shift
   update the producing implementation in its own position domain.
8. A later image or incompatible cache restore reevaluates policy and rebases only
   the implementations that require it.

`server_slot::can_speculate()` becomes a query for whether the sequence has at
least one currently eligible implementation. A loaded mmproj and historical
media are not sufficient reasons to return false.

## Transactional failure behavior

Media processing is fail-closed and isolated:

- An implementation reports success only after all rows and required features in
  its filtered batch have been committed.
- On an error, its affected sequence is invalidated, pending proposals from that
  implementation are discarded, and its last-implementation pointer is cleared.
- Other implementations continue if their own state remains valid.
- The target completes the request without speculation if no implementation
  remains eligible.
- A partial draft context cannot be marked synchronized or written to the
  secondary prompt cache.
- Cache metadata records the implementation policy, state generation, target
  boundary, draft-local boundary, and synchronization result needed to reject an
  incompatible restore.

Runtime logs state, once per transition rather than once per token, which
implementation is active, rebased, skipped, invalidated, or restored. Metrics
separate media-eligible draft counts and acceptance from ordinary text counts so
the behavior can be confirmed from server logs.

## Context shift and cache invariants

The existing vision rules remain mandatory: shifts cannot split image chunks,
and Turbo/M-RoPE reprefill is transactional. This design adds these speculative
invariants:

1. Embedding-prefill implementations shift or reprefill every retained media and
   text position paired with the target.
2. Post-media-rebase implementations recompute their explicit target-to-local
   boundary mapping after a target shift.
3. Stateless implementations discard token history that the target discarded and
   never retain n-grams crossing a removed media boundary.
4. No implementation resumes until its post-shift position and generation checks
   pass.
5. A secondary-cache entry is usable only by the same speculative implementation
   configuration and a compatible state generation.

## Default behavior and compatibility

No new opt-in flag is required. Proven media-capable implementations work
automatically with their existing `--spec-type` configuration. Users can still
disable speculation through existing configuration when diagnosing a model.

Text-only behavior, CLI names, draft limits, and performance defaults remain
unchanged. In particular:

- Gemma assistant uses `--spec-type draft-mtp`;
- Qwen embedded NextN uses `--spec-type draft-mtp` without a draft path;
- Prism uses `--spec-type draft-dspark`;
- removed legacy option names do not return.

An implementation/model pair that cannot satisfy the media contract falls back
to target-only execution for the affected sequence with a precise reason. It
must still recover automatically when a valid rebase becomes possible; permanent
conversation-wide disablement is forbidden.

## Validation strategy

### Focused automated tests

Extend the C++ speculative and server state tests to cover:

- the complete policy matrix;
- per-implementation and per-sequence eligibility;
- mixed text and media sequences in one batched decode;
- filtered batch row and sequence membership;
- a failing implementation followed by a healthy implementation in one chain;
- state invalidation without target interruption;
- cache save/restore generation checks;
- target and draft-local rollback positions;
- repeated media boundaries and forced context shift;
- zero, partial, and full proposal acceptance.

Extend server tests to cover an OpenAI-compatible conversation that resends its
original image on every request. Metrics and logs must show continued eligible
drafting rather than a request-wide media veto.

Synthetic implementation fixtures exercise token batches, embedding batches,
multi-sequence rows, deterministic failures, and compact rebase mappings without
requiring large models. Existing CUDA layout tests remain part of the full gate.

### Correctness oracle

For every model-based test, deterministic greedy target-only generation is the
semantic oracle. Invalid positions, unverified commits, decode failures, or stale
boundaries are defects. Bit-for-bit token identity is recorded but is not a hard
gate for quantized CPU/GPU kernels: even a text-only one-token Gemma verification
batch changed one near-tied final token because batched and sequential reductions
use different floating-point orders. This is ordinary speculative numerical
behavior, not permission to accept a token that the target did not verify.

State checks compare target positions, draft positions or explicit mappings,
pending-proposal ownership, cache generations, and rollback endpoints after each
operation. A test that merely produces plausible text is insufficient.

### Exact owner-model gates

Run inference only as a hidden child process with redirected output and a finite
timeout, as required by repository policy.

For Ternary Bonsai 27B + DSpark + mmproj + Turbo4 K / Turbo3 V:

- image prefill followed by an answer with nonzero accepted DSpark drafts;
- target-only/speculative greedy token equality;
- all configured DSpark proposal positions exercised;
- a second text turn that resends the full image history and continues drafting;
- a second image in the same conversation;
- prompt-cache reuse and cache miss;
- forced context shift at a deliberately small context;
- rejection, rollback, and recovery;
- Q8_0/Q8_0 draft KV and the configured 4096-position-per-slot cap.

For historical Gemma 4 + assistant + mmproj:

- target-only/speculative greedy token equality;
- accepted drafts before media, during the image answer when the corrected bridge
  is eligible, and after full-history resend;
- CPU-offloaded MoE operation with assistant `op_offload = false`;
- context shift and prompt-cache reuse.

For Qwen embedded NextN + mmproj:

- the same parity and repeated-history sequence;
- proof that the embedded MTP context uses `LLAMA_CONTEXT_TYPE_MTP`;
- accepted multi-step drafts after media rebase;
- no independent draft-model path.

EAGLE3 and draft-simple use locally available compatible target/draft pairs. Each
must prove parity, rebase, rollback, and repeated-history recovery. N-gram modes
run against a small deterministic model and the exact Bonsai target to prove both
the stateless contract and real-server integration.

### Build and regression gates

Before publishing the implementation:

- full Windows CUDA Release tool-suite build for `sm_120`;
- full CTest suite;
- CPU build and tests for the shared speculative coordinator changes;
- server unit tests, including vision API and speculative tests;
- exact Bonsai, Gemma, and Qwen model gates above;
- focused F16/Q8 separate-V and Q8/Q8 K-subview CUDA tests;
- text-only speculative regression tests for every maintained type;
- target-only vision regression to prove that failure isolation never harms the
  primary model path.

The recurring non-consecutive-position warning seen during image processing is
compared with a target-only baseline. This change may not introduce additional
warnings or hide a position mismatch behind logging changes.

## Documentation and provenance

Implementation comments explain why media policies, independent position
domains, and transactional invalidation exist and which corruption or permanent
disablement they prevent. Comments must not merely restate branches.

Update `docs/speculative.md`, `MTP.md`, `NEXTN.md`, the README examples, and
`docs/development/upstream-prism-convergence.md` with the final behavior and test
results. Preserve original authorship when importing a media bridge from
`ggml-org/llama.cpp`, PrismML-Eng, or another source, and record the exact source
repository, branch, and commit. Local adaptations are identified separately.

## Implementation sequence

1. Add the policy contract, per-sequence eligibility, diagnostics, and synthetic
   coordinator tests without enabling new media behavior.
2. Remove short-circuit cross-implementation processing and add transactional
   invalidation.
3. Validate and enable DFlash/DSpark embedding prefill.
4. Validate and enable stateless n-gram media operation.
5. Implement and validate MTP/NextN media bridging for Gemma and Qwen.
6. Implement and validate EAGLE3 post-media rebase.
7. Implement and validate draft-simple compact text projection.
8. Integrate prompt-cache and context-shift state for every policy.
9. Run exact model, CPU, CUDA, server, and regression gates.
10. Update human-facing documentation, provenance, presets if behavior is
    exposed there, and the convergence ledger.

Each algorithm is enabled when its own safety gates pass, but the branch is not
complete until all maintained speculative types have a defined, tested media
path and full-history conversations no longer cause permanent disablement.

## Out of scope

This work does not add subserver orchestration, ComfyUI integration, ACE-Step,
new speculative algorithms, new model formats, or a generalized draft-side
multimodal loader. It also does not change the established TurboQuant defaults.
Those concerns must not dilute the correctness work around model speculation and
vision.

## Completion rule

The implementation is complete only when:

- no blanket request-wide media veto remains;
- every maintained speculative implementation has an explicit, tested media
  policy;
- safe implementations can draft during image answers;
- full-history media conversations do not permanently disable later drafting;
- every committed speculative token is target-verified, with any greedy
  target-only difference attributable to a reproduced floating-point near-tie
  rather than media state or position corruption;
- cache reuse, rollback, batching, and context shift preserve all position and
  synchronization invariants;
- the exact Bonsai, Gemma, and Qwen vision/speculation regressions pass.

A DSpark-only exception, a manual history-stripping workaround, an invented image
token, or a silent target-only fallback for the remainder of the conversation is
not a complete solution.
