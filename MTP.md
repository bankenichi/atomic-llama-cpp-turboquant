# MTP speculative decoding

This fork uses the current unified `draft-mtp` implementation in
[`common/speculative.cpp`](common/speculative.cpp). It covers two model layouts:

- Gemma 4 target plus an external Gemma assistant GGUF.
- A combined Qwen GGUF with embedded NextN/MTP tensors.

The implementation is a convergence of the current `llama.cpp` speculative API
with the earlier Gemma assistant and Qwen NextN work inherited from
[`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant).
The imported Git history, contributor names, and additional source links are
preserved in [`NOTICE.md`](NOTICE.md). This document describes the resulting code,
not a claim that the local fork originated MTP or NextN.

The historical `--spec-type mtp`, `--mtp-head`, `--draft-block-size`,
`--draft-max`, and `--draft-min` interface is removed. Use the generic speculative
arguments documented below.

## What MTP does

Multi-Token Prediction uses a trained auxiliary head to propose more than one
future token for each target-model verification pass. In simplified form:

1. The target decodes the accepted prefix and publishes the hidden row required
   by the auxiliary head.
2. The MTP/NextN context combines that hidden row with the last accepted token.
3. It proposes a short chain, normally two tokens on this machine.
4. The target evaluates the proposal in a batch.
5. Only the matching prefix is committed. At the first mismatch, normal target
   sampling wins and the unaccepted draft suffix is discarded.

This preserves the target model's sampling semantics. A poor draft head reduces
acceptance and can make generation slower, but it must not silently substitute its
own output for target verification.

The current driver supports three internal modes even though they share one CLI
name:

| Internal mode | Typical model | Memory relationship | Draft stepping |
|---|---|---|---|
| Shared target memory | Gemma 4 assistant | Draft reads target KV through `ctx_other` | All trained assistant layers participate in its graph |
| Single trained head | Qwen 3.6 | Separate draft context over embedded tensors | Autoregressive chain through one NextN head |
| Chained trained heads | Multi-head Step/NextN files | Separate per-head draft KV | Step *i* selects trained head *i*; depth is clamped to available heads |

These distinctions are load-bearing. In particular, the Gemma path must not be
changed to perform the Qwen catch-up decode, and a chained-head model must clear
only the draft region before selecting the next head so it does not read KV written
by a different head.

## CLI

External Gemma assistant:

```powershell
llama-server `
  -m C:\models\gemma-target.gguf `
  --model-draft C:\models\gemma-assistant.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 0
```

Embedded Qwen NextN:

```powershell
llama-server `
  -m C:\models\qwen-native-mtp.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 0
```

For embedded NextN, omit `--model-draft`. The loader detects the combined GGUF
and reopens it with the draft architecture. See [`NEXTN.md`](NEXTN.md).

The checked-in Gemma launchers under [`scripts/`](scripts/) use these names.

### Important arguments

`--model-draft PATH`
: External assistant/sidecar path. Required for Gemma assistant files; omitted for
  a Qwen combined GGUF whose NextN tensors are embedded in the target.

`--spec-type draft-mtp`
: Selects the unified MTP implementation. The old `mtp` and `nextn` names refer to
  removed private drivers and are intentionally rejected.

`--spec-draft-n-max N`
: Maximum proposal depth. Start at 2. A larger value increases draft work and is
  useful only when the model contains enough trained heads or the autoregressive
  head maintains high acceptance.

`--spec-draft-n-min N`
: Minimum useful proposal length. Zero permits an immediate fallback to target-only
  decoding when confidence or implementation conditions stop drafting.

`--spec-draft-p-min P`
: Optional confidence cutoff. Zero disables confidence early-stop. Raising it can
  avoid low-value draft suffixes but can also reduce the amount of work amortized by
  one target verification.

`-ngld`, `-ctkd`, `-ctvd`
: Draft GPU layers and draft KV types. These are separate from the target's `-ngl`,
  `-ctk`, and `-ctv` settings even when both contexts read one combined GGUF.

## Runtime design

The target produces the hidden representation required by the MTP head. The draft
context consumes that representation and the last accepted token, proposes up to
`--spec-draft-n-max` tokens, and the target verifies the proposal. Rejected tokens
never enter the committed target sequence, so speculative decoding does not change
output semantics.

`common_speculative_impl_draft_mtp` handles both shared-memory and independent
draft layouts:

- Gemma assistant models use the target as `ctx_other` and read the target state
  required by their graph.
- Qwen combined models expose one or more NextN layers from the same GGUF.
- Multi-head NextN files clamp the draft chain to the number of trained heads.
- The draft and target memory operations are coordinated through the common
  speculative wrapper so prompt removal, copying, and rebuilds apply to both.

The Gemma assistant runs in a dedicated `LLAMA_CONTEXT_TYPE_MTP` context, which
selects the decoder-MTP graph through the current context API. Draft-context
construction disables op offload; that remains a load-bearing compatibility
requirement for CPU-offloaded MoE targets.

### Data flow across batches

The target exposes pre-normalized NextN embeddings for the rows it just decoded.
For independent draft memory, `process()` shifts those rows by one token: the hidden
state at position *p* is paired with token *p + 1*. The final hidden row has no next
token in the current batch, so it is retained in `pending_h` and paired when the
following batch arrives. Losing that boundary state produces a draft that is one
position out of phase even though both KV caches may appear to have the same length.

During verification, hidden rows are copied into a per-sequence buffer. `accept()`
selects the row corresponding to the number of accepted tokens, ensuring the next
draft starts at the actual committed boundary rather than at the end of the rejected
proposal.

For Gemma's shared-memory path, the assistant reads the target KV at the accepted
boundary and does not maintain an independent copy of the target prefix. For Qwen,
`process()` mirrors accepted target tokens into the draft context. The server's
coherence state therefore describes whether the relevant memory for the selected
mode matches the target—not merely whether a draft context object exists.

## Implementation map

| Concern | Current location |
|---|---|
| Unified MTP driver and hidden-row carryover | `common/speculative.cpp`, `common_speculative_impl_draft_mtp` |
| Speculative type/argument plumbing | `common/arg.cpp`, `common/common.cpp` |
| Gemma assistant loader and graph | `src/models/gemma4-assistant.cpp` |
| Qwen/Step NextN graph selection | model implementations under `src/models/` and `src/llama-context.cpp` |
| Target-to-draft context relationship | `src/llama-context.cpp`, `src/llama-ext.h` |
| Server request, cache, and media lifecycle | `tools/server/server-context.cpp`, `tools/server/server-spec-state.h` |
| Secondary prompt-cache serialization | `tools/server/server-task.{h,cpp}` |
| Focused regression coverage | `tests/test-speculative-mtp.cpp`, `tests/test-server-spec-state.cpp`, `tests/test-arg-parser.cpp` |

## Gemma assistant compatibility

Modern files identify the assistant using the current Gemma MTP architecture.
Older files published by this fork use `general.architecture = gemma4_assistant`,
legacy metadata keys such as `n_embd_backbone`, and `mtp.*` tensor names. The model
loader records the raw architecture prefix, normalizes it to the current internal
architecture, and selects the legacy metadata/tensor map when needed.

Do not remove the raw-prefix override. Without it, an old assistant can be
recognized as Gemma MTP but still fail because its metadata is looked up under the
wrong namespace.

The two formats differ in more than the `general.architecture` string:

| Concept | Historical assistant | Current representation |
|---|---|---|
| Architecture prefix | `gemma4_assistant` | current Gemma MTP architecture |
| Target hidden width | legacy `n_embd_backbone` key | generic input/output embedding metadata |
| Prediction depth | assistant block count | `nextn_predict_layers` |
| Pre/post projections | `mtp.*` tensor names | `nextn.*` tensor names |
| Ordered-embedding head | `mtp.centroids` / token ordering | masked-embedding tensor namespace |

The loader preserves the raw architecture name before internal normalization and
uses it to choose the complete metadata and tensor map. Supporting only the old
architecture enum, while resolving tensors through the new namespace, is an
incomplete compatibility implementation.

## Vision lifecycle

Loading `--mmproj` does not disable MTP. The server distinguishes multimodal
capability from a request that actually contains media:

- Text request with mmproj loaded: target and draft run normally.
- Request containing image/audio/video data: the target handles the projector
  embeddings, while MTP consumes only the target hidden rows defined by its
  architecture. It never invents token IDs for image rows.
- Full-history follow-up containing the same image: the coherent cached prefix and
  MTP boundary are reused, and drafting continues during the follow-up.

This is enforced by the per-implementation media policy in
`common/speculative.cpp` and the slot coherence state in
`server_spec_context_state`. If a bridge cannot process a batch, only that
implementation is invalidated; the target answer remains valid.

Secondary RAM prompt-cache entries save four related pieces as one logical unit:

- target sequence bytes;
- draft sequence bytes, when the draft has independent memory;
- the `draft_synchronized` coherence bit;
- optional implementation state such as an Eagle/NextN deferred boundary row.

A cache miss is a no-op: it must not manufacture synchronization. A cache
hit restores the entry's own coherence bit. Even an empty optional-state payload is
applied, because empty means “clear the previous destination boundary,” not “leave
whatever state this slot happened to have.” Failed restores leave the cache entry
intact so another slot is not handed partially consumed state.

For context shifting with media, see
[`docs/development/context-shift-with-vision.md`](docs/development/context-shift-with-vision.md).

## Capacity and tuning

- Start with `--spec-draft-n-max 2`. Longer chains only help when acceptance and
  draft cost justify them.
- `-ngld` controls draft GPU placement independently from `-ngl`.
- `-ctkd` and `-ctvd` select draft KV types.
- On the RTX 5080, preserve roughly 1 GB of headroom below the 16 GB hardware
  capacity. A draft that pushes decode scratch into shared memory can be slower
  than baseline.
- On heavily CPU-offloaded MoE targets, MTP may be throughput-neutral because
  target verification remains memory-bandwidth bound.

Measure baseline and speculative modes with the same prompt, sampling settings,
context length, KV types, and GPU placement. Useful counters are total drafted
tokens, accepted tokens, acceptance ratio, prompt throughput, and generation
throughput. A high acceptance ratio is not sufficient if the draft evicts a target
expert layer or pushes CUDA scratch into shared GPU memory.

On this owner's 16 GB RTX 5080, an external Gemma assistant is often near break-even
with substantial CPU MoE offload: reclaiming its VRAM for one more target expert
layer can be more valuable. MTP becomes more attractive when most target experts
fit on GPU. Treat that as a hardware-specific observation, not a general judgment
about the technique.

## Failure signatures

- **Assistant metadata key missing:** confirm whether the file is historical
  `gemma4_assistant` or a modern assistant. A mixed conversion may advertise one
  architecture while using the other tensor namespace.
- **Hidden width assertion:** target and assistant are not a compatible pair, or
  the conversion recorded the wrong target backbone width.
- **Zero acceptance with successful load:** check draft precision, prompt template,
  and whether hidden-state rows are aligned at the accepted boundary.
- **Failure only after media:** inspect the implementation policy, boundary state,
  and draft/target positions. Do not disable mmproj globally; a retained image is
  expected to coexist with MTP.
- **Slower than target-only:** reduce draft depth, compare KV formats, and account
  for the VRAM cost of the assistant before changing correctness code.

## Validation contract

Before changing MTP, model loading, server cache reuse, or vision code, validate:

1. Exact historical Gemma target + assistant + mmproj loads.
2. A text request returns correctly and logs nonzero draft acceptance.
3. A real image answer records nonzero accepted drafts on the same server.
4. A full-history follow-up resends the image, reuses a coherent prefix, and
   records accepted drafts without a forced media-only invalidation.
5. Embedded Qwen NextN satisfies the same image-answer and retained-history gate.

The convergence acceptance matrix is recorded in
[`docs/development/upstream-prism-convergence.md`](docs/development/upstream-prism-convergence.md).

## Historical implementation record

Before convergence, this fork used a private async Gemma MTP API, depth-2 worker,
centroid-head presets, and CLI such as `--mtp-head`/`--draft-block-size`. That design
is no longer the executable architecture, but its detailed rationale, measurements,
and contributor history remain available in the repository at
[`MTP.md` before convergence](https://github.com/bankenichi/atomic-llama-cpp-turboquant/blob/e3cfbe4f7/MTP.md).
Consult it when tracing provenance or old benchmark logs; do not copy its removed
flags or APIs into current launchers.
