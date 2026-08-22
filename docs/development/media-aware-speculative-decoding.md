# Media-aware speculative decoding

This document explains why Atomic can keep speculative decoding active during an
image answer and during later OpenAI-compatible requests that resend the original
image. It is the implementation companion to the operator-facing overview in
[`../speculative.md`](../speculative.md).

## Problem

The old server gate treated any request containing real media as target-only.
That looked conservative for a single image turn, but OpenAI-compatible clients
usually resend the whole conversation. The old image therefore remained in every
future request and speculation was disabled for the practical lifetime of the
conversation.

An mmproj embedding is not a token ID, so simply forwarding it into every draft
model would be incorrect. The complete solution is algorithm-specific: each
drafter declares how it represents the target's media-conditioned boundary.

## Runtime contract

`common_speculative_media_policy` defines four behaviors:

| Policy | Implementations | Contract |
|---|---|---|
| embedding prefill | DFlash, DSpark | Encode target-layer features for image rows and inject draft KV at real target context positions |
| post-media rebase | EAGLE3, MTP/NextN, draft-simple | Preserve a model-valid target-derived or text-only boundary in an explicit draft position domain |
| stateless | every n-gram mode | Use committed text tokens without maintaining model KV |
| unsupported | unknown/none | Fail closed for the affected implementation and sequence |

Eligibility is tracked for every implementation/sequence pair. Server batches
must have the same eligibility mask; one slot's image cannot alter another slot's
text drafter. A processing failure disables only the failing implementation. The
target batch has already been decoded, so its verification result is finalized
even when the proposal producer becomes ineligible during the mirror callback.
This ordering prevents stale `spec_i_batch` state from reaching the next cycle.

## Algorithm details

### DFlash and DSpark

The target can expose hundreds of vision-encoder rows at repeated M-RoPE
positions. The draft KV requires consecutive one-dimensional positions. The
server announces both the encoder-row count and the real context-position span;
the speculative driver accumulates every target-derived encoded row, partitions
the rows over that span, mean-pools each partition, and injects one feature row per
context position.

This is not image-token fabrication. The draft receives only target-layer
features produced for the real image, and the target still verifies every proposed
text token. DSpark and DFlash share this prefill path; DSpark adds its Markov head
when generating the proposal block.

### EAGLE3

EAGLE expects `(token[P+1], g[P])`. Image rows provide `g[P]` but not a valid next
token for every patch. Atomic keeps the final target-derived image feature as a
deferred boundary at the next compact draft position. The first real post-image
token completes that pair.

Target verification continues at target absolute positions. EAGLE rollback uses
compact positions. Before verification, provisional seed/proposal KV is removed
from the compact boundary because the server's generic target-position rollback
cannot address it. Verified target features then rebuild the committed compact
rows. Boundary serialization includes the compact-domain flag so prompt-cache
restore cannot reinterpret a local position as an absolute target position.

### MTP and embedded NextN

MTP captures the target hidden rows defined by the loaded architecture. Independent
Qwen NextN contexts retain the final target hidden row and pair it with the next
real token; shared-memory Gemma assistants already attend the target memory.
Neither path turns an image embedding into a token ID. The same bridge works when
the next request reuses a cached prefix containing the original image.

### Standalone draft model

A normal draft model cannot decode the target mmproj's vectors: embedding width
and semantics belong to the target. Atomic clears that draft sequence and rebuilds
only the serializable text history in a compact local domain. Verification batches
are translated into that domain, and provisional KV is removed before the target's
committed rows are mirrored. This path requires a tokenizer-compatible draft model.

### N-grams

N-gram implementations have no model KV and need no image representation. They
learn from real serialized text tokens and committed generation. Expanded image
rows never enter their token history. All maintained variants remain eligible on
image and retained-image requests.

## Cache and context-shift invariants

- A loaded mmproj is capability, not evidence that the current request has media.
- A cache entry is coherent only when every active stateful implementation is
  coherent; a stateless fallback cannot make a failed model drafter cacheable.
- Cache misses do not invent synchronization. Cache hits restore their recorded
  draft bytes, coherence, and optional boundary state.
- DFlash/DSpark use target positions and may follow a paired in-place f16/q8 shift.
- Post-media EAGLE/MTP/draft-simple use private domains. A retained-media context
  shift therefore forces transactional reprefill even if both KV formats are
  otherwise shiftable.
- TurboQuant and M-RoPE already use transactional reprefill. Failure restores the
  original target, draft, prompt, and boundary state together.

## Correctness and numerical reproducibility

The target verifies every committed proposal. Invalid positions, decode errors,
stale boundaries, or a token committed without target verification are defects.

Bit-for-bit equivalence to sequential target-only greedy decoding is a useful
diagnostic, but not a valid universal requirement for quantized inference. On the
owner's CPU-offloaded Gemma target, a one-token speculative batch and a text-only
target run matched for 31 tokens before selecting `process` versus `step` at a
near tie. The same difference occurred with no image in the request, proving it
was caused by batched floating-point reduction order rather than media state.

## Exact-model validation, 2026-08-21

All inference ran as a hidden child process with redirected logs and finite
timeouts through [`../../scripts/tests/test-media-speculation.ps1`](../../scripts/tests/test-media-speculation.ps1).

| Configuration | Image answer | Retained-image follow-up |
|---|---:|---:|
| Ternary Bonsai 27B + DSpark Q4_1, depth 4, q8_0/q8_0 draft KV | 11/48 accepted | 2/4 accepted |
| Same sidecar through DFlash mode, depth 4 | 0/39 accepted | 2/3 accepted |
| Qwen 3.6 35B-A3B embedded NextN, depth 2, q8_0/q8_0 draft KV | 11/22 accepted | 2/2 accepted |
| Gemma 4 26B-A4B + historical assistant, depth 2 | 14/18 accepted | 2/2 accepted |
| Gemma 4 26B-A4B + EAGLE3 Q4_K_M, depth 4 | 12/39 accepted | 8/16 accepted |
| Ternary model used as its own CPU draft-simple fixture, depth 4 | 4/11 accepted | 4/4 accepted |

All five n-gram modes completed the same full-history sequence without media or
position errors; `ngram-cache` produced accepted drafts on the short patterned
fixture. The other short n-gram fixtures produced no proposal, which is expected
when their configured history pattern does not repeat.

### Build and structural validation

The same worktree passed the complete 48-test CTest suite in both CPU-only and
CUDA builds. The CUDA run included the exhaustive backend-operator comparison.
Two Bonsai-sensitive flash-attention layouts were then selected explicitly:

- separate F16 K / Q8_0 V at 576x512 passed on the RTX 5080;
- Q8_0/Q8_0 V-as-K-subview at 576x512 passed for the supported 1024-token,
  one-row and 64-row cases. The deliberately unsupported 113-token alignment
  fixture remained correctly rejected by the CUDA capability check.

The canonical launcher regression suite passed all 53 checks. The config UI
passed its nine Python unit tests plus Python bytecode compilation and JavaScript
syntax checks. These tests supplement the exact-model results above; they do not
replace image-answer and retained-history inference.

## Provenance

- EAGLE3 core and deferred-boundary design: `ggml-org/llama.cpp` PR #18039,
  represented in this history by commits `88a39274e` and later `f87067841`.
- Upstream DSpark: `ggml-org/llama.cpp` PR #25173, commit `84075273c`.
- Prism DSpark/Markov and CUDA foundation: `PrismML-Eng/llama.cpp`, imported
  commit `ba62b7023` by `bri-prism`, with earlier Prism work by Pasha Khosravi.
- Atomic mmproj/speculation coexistence direction: commit `ead60fb4f` by Biogenic
  Ooze.
- Per-policy coordination, M-RoPE feature pooling, compact private position
  domains, retained-image bridges, and failure finalization are local Atomic
  adaptations. They do not replace the authorship of the upstream algorithms.

Model weights and projectors remain governed by their own model cards and
licenses. The EAGLE validation sidecar was the Q4_K_M conversion published at
[`williamliao/gemma-4-26B-A4B-it-speculator.eagle3-F16-GGUF`](https://huggingface.co/williamliao/gemma-4-26B-A4B-it-speculator.eagle3-F16-GGUF),
derived from Red Hat's Gemma 4 EAGLE3 speculator.
