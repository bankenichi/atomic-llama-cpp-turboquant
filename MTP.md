# MTP speculative decoding

This fork uses the current unified `draft-mtp` implementation in
[`common/speculative.cpp`](common/speculative.cpp). It covers two model layouts:

- Gemma 4 target plus an external Gemma assistant GGUF.
- A combined Qwen GGUF with embedded NextN/MTP tensors.

The historical `--spec-type mtp`, `--mtp-head`, `--draft-block-size`,
`--draft-max`, and `--draft-min` interface is removed. Use the generic speculative
arguments documented below.

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

## Gemma assistant compatibility

Modern files identify the assistant using the current Gemma MTP architecture.
Older files published by this fork use `general.architecture = gemma4_assistant`,
legacy metadata keys such as `n_embd_backbone`, and `mtp.*` tensor names. The model
loader records the raw architecture prefix, normalizes it to the current internal
architecture, and selects the legacy metadata/tensor map when needed.

Do not remove the raw-prefix override. Without it, an old assistant can be
recognized as Gemma MTP but still fail because its metadata is looked up under the
wrong namespace.

## Vision lifecycle

Loading `--mmproj` does not disable MTP. The server distinguishes multimodal
capability from a request that actually contains media:

- Text request with mmproj loaded: target and draft run normally.
- Request containing image/audio/video data: target handles the media; the draft
  context is marked invalid rather than pretending it consumed the media tokens.
- Next text request on that slot: cached prompt reuse is forced to position zero,
  both memories are cleared, and target plus draft are rebuilt together.

This is enforced by `server_spec_context_state` and unit-tested independently.
It prevents a target prompt-cache hit from being paired with an empty or stale
draft cache after a media turn.

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

## Validation contract

Before changing MTP, model loading, server cache reuse, or vision code, validate:

1. Exact historical Gemma target + assistant + mmproj loads.
2. A text request returns correctly and logs nonzero draft acceptance.
3. A real image request succeeds on the same server.
4. A following cached text request rebuilds target and draft safely.
5. Embedded Qwen NextN returns correctly with nonzero accepted drafts.

The convergence acceptance matrix is recorded in
[`docs/development/upstream-prism-convergence.md`](docs/development/upstream-prism-convergence.md).
