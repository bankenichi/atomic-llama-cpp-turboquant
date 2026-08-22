# DSpark long-context continuity and runtime observability

Date: 2026-08-22

## Status and scope

This design closes the defects found while reviewing the exact Ternary Bonsai
27B + DSpark + mmproj server log. It covers:

- DSpark and DFlash drafting after the target sequence crosses the drafter's
  trained 4096-position limit;
- false recurrent-cache continuity warnings produced by image M-RoPE batches;
- normal-verbosity startup diagnostics for effective target and draft runtime
  settings; and
- exact regression coverage for text, retained vision, and target context shift.

It does not expand the project into workload coexistence or subserver
orchestration. Base-model, speculation, and vision correctness remain the
priority.

## Evidence and root causes

The 2026-08-22 exact-model run accepted DSpark proposals before and after image
prefill, but six long generations eventually failed in `llama_decode(ctx_dft)`
with `failed to find a memory slot`. Each failure occurred when the target-space
draft boundary approached 4096. The server correctly isolated the failed drafter
and rebuilt the paired target/draft state on the next request, but drafting was
lost for the rest of the active generation and throughput fell to target-only
decode speed.

`common_base_params_to_speculative()` intentionally caps published DSpark
sidecars at 4096 positions per parallel sequence. That cap reflects the trained
context and must remain. The defect is that the DFlash/DSpark implementation
currently addresses the private draft KV cache with absolute target positions.

The same run emitted repeated `non-consecutive token position` warnings from
`llama_memory_recurrent::find_slot()`. The warning assumes a scalar position that
advances once per decoded row. M-RoPE image rows deliberately reuse the temporal
coordinate while their spatial coordinates vary, and the following text starts
after the image's declared one-dimensional span. The recurrent state itself is
updated in processing order; only the scalar continuity diagnostic is invalid
for a multidimensional position batch.

Finally, command-line capture proved the requested K/V types and flash-attention
mode were passed, but normal server logs did not print enough effective context
state to distinguish requested settings from settings accepted by context
construction.

## Considered approaches

### Stop drafting at the trained boundary

The implementation could detect the impending overflow and disable DSpark
without allowing a decode failure. This would improve the error path but retain
the main behavioral defect: long responses permanently lose drafting. It is not
a complete solution.

### Allocate the draft context to the full target size

Giving the sidecar a 32768-position context avoids the immediate allocation
failure but exceeds the published model's trained 4096-position window, consumes
unnecessary VRAM, and does not establish that positional behavior remains valid.
This also violates the fork's explicit per-slot DSpark cap.

### Use a private sliding draft window (selected)

The target retains its full absolute position domain. DFlash/DSpark maintains a
per-sequence target-space origin and translates incoming target positions to
draft-local positions. Before an injection or a noise block would exceed the
4096-position draft capacity, the implementation removes an old draft prefix,
shifts the retained KV suffix toward zero, and advances the origin by the same
amount.

This keeps the sidecar inside its trained window and preserves continuous
drafting. Translation does not change target correctness because every proposal
is still verified by the full target model. For standard RoPE attention, shifting
all retained K positions by the same delta preserves their pairwise relative
positions. The implementation must require a shift-capable draft memory; current
Q8_0/Q8_0 DSpark KV satisfies that contract. An incompatible future cache type
must fail initialization with a useful message rather than silently corrupting
state.

## Detailed design

### Per-sequence window state

Each DFlash/DSpark sequence owns:

- `origin`: the absolute target position represented by draft-local position 0;
- `next`: the first free draft-local position after committed injected KV;
- a short-lived verification boundary used to remove provisional noise rows; and
- media accumulation already required to pool M-RoPE encoder rows into the
  media chunk's declared one-dimensional span.

The mapping is `local = target - origin`. Empty/reset sequences begin with
`origin = 0` and `next = 0`. A position-zero target rebuild clears both values.
Prompt-cache state serializes a versioned payload containing the origin and next
boundary so restored draft KV and its mapping cannot diverge.

### Rollover policy

The draft capacity is `llama_n_ctx(ctx_dft) / llama_n_seq_max(ctx_dft)`. The
implementation reserves enough trailing space for the largest configured noise
block. Before injecting a contiguous target-derived feature span or decoding a
noise block, it ensures the mapped end plus this reserve fits.

Rollover discards at least one fixed chunk and may discard more when a large
prefill chunk requires it. The retained suffix is shifted with
`llama_memory_seq_rm()` followed by `llama_memory_seq_add()`. Chunked rollover
avoids a K-shift on every generated token. Injection chunks are bounded so a
single call never requires more committed rows than the usable window.

Provisional noise-block KV is removed after proposal sampling. The verified
target feature batch is the only source of committed draft KV for those
positions. This makes ownership explicit and avoids accumulating duplicate or
rejected speculative rows.

### Media and context-shift behavior

Real media continues to use target-derived hidden features. Image encoder rows
are pooled into the declared one-dimensional media span before draft injection;
the resulting absolute media positions pass through the same private-window
mapping as text.

The existing server media resynchronization contract remains authoritative:
real media invalidates stale cross-context state, and the next required rebuild
starts both contexts from position zero. Transactional context-shift reprefill
replays retained text and media. During replay, DSpark begins at origin zero and
rolls its private window as necessary; the target still keeps its own full
retained sequence.

### M-RoPE recurrent diagnostic

The scalar consecutive-position warning remains for `ubatch.n_pos == 1`. For
`ubatch.n_pos > 1`, the diagnostic is skipped with an implementation comment
explaining why: row count is not a valid delta for multidimensional RoPE. No
state update, attention position, or cache eviction behavior is changed.

This is deliberately narrower than globally suppressing the warning. Scalar
text models retain the existing protection against unexpected backtracking or
gaps.

### Effective runtime logging

At normal verbosity the draft initialization log reports:

- speculative implementation and configured draft depth;
- effective per-sequence draft context capacity;
- K and V cache types;
- flash-attention state;
- device/offload selection; and
- sliding-window reserve and rollover chunk.

Target context initialization reports effective context capacity, K/V types,
and flash-attention state after validation and automatic adjustments. Existing
model-loading and build-information logs remain the source of model paths,
offload counts, and revision identity.

Rollover logs one concise informational event containing sequence, old/new
origin, discarded positions, and retained count. It must not log once per token.

## Failure behavior

- Draft memory that cannot shift is rejected when DFlash/DSpark initializes.
- Impossible position regressions or contradictory restored state return a
  coherent speculative failure; the coordinator may fall back for safety, and
  the next request performs the established paired rebuild.
- A prefill span larger than the usable private window is streamed through the
  window rather than allocated as one batch.
- Version-mismatched prompt-cache mapping state is cleared. It is never applied
  to restored KV optimistically.

## Validation

Unit and focused tests will cover window mapping, chunked rollover, state
serialization, provisional-block cleanup, and the scalar-versus-M-RoPE warning
predicate. Existing media speculation, server cache, DSpark metadata, launcher,
and CUDA layout tests remain required.

Exact hidden-process validation must include:

1. Ternary Bonsai + Q8_0 DSpark + mmproj + Turbo4/Turbo3 at the standard 32768
   target context;
2. a text generation that crosses multiple draft-window rollovers and continues
   accepting proposals without `failed to find a memory slot` or drafter
   disablement;
3. an image answer followed by retained-image text turns, including a rollover;
4. a deliberately smaller target context that forces transactional retained-media
   context shift and resumes drafting afterward; and
5. full CPU and Windows CUDA builds/tests plus the focused separate-V and
   K-subview CUDA operator tests required by `AGENTS.md`.

## Provenance

DFlash originates in ggml-org/llama.cpp PR #22105. DSpark support originates in
ggml-org/llama.cpp PR #25173, with the legacy Prism Markov implementation from
PrismML-Eng/llama.cpp commit `ba62b7023`. Atomic's media feature injection and
resynchronization are documented in the 2026-08-21 media-aware speculation
design. The private sliding-window policy and diagnostics described here are
Atomic adaptations prompted by exact Ternary Bonsai long-context testing; they
are not represented as upstream or Prism behavior.
