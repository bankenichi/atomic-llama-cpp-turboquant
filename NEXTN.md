# Qwen embedded NextN

Qwen 3.x combined MTP GGUFs store their NextN draft layers alongside the target
weights. This fork runs them through the unified `draft-mtp` speculative engine.
The old fork-specific `nextn` speculative type and async-worker environment switch
no longer exist after convergence with current `llama.cpp`.

The Qwen weights and architecture originate with the Qwen team. The combined GGUF
and quantization ecosystem used during development includes Unsloth's MTP-aware
conversions and the AtomicChat UDT artifacts inherited from this fork's AtomicBot
lineage. Those projects retain credit for their weights, conversion inputs, and
packaging work; see [Model and artifact provenance](#model-and-artifact-provenance)
and [`NOTICE.md`](NOTICE.md).

## Mental model

A combined file contains one normal Qwen target plus auxiliary NextN tensors after
the target layers. `llama-server` does not need a second physical model file. It
opens the same GGUF through a draft architecture, creates a separate lightweight
draft context, and uses the target's published pre-normalized hidden rows as draft
inputs.

```text
combined Qwen GGUF
  +-- target tensors ----------> target context ----> verification logits
  |                                  |
  |                                  +-- pre-normalized hidden row h[p]
  |
  +-- embedded NextN tensors --> draft context <----- token[p + 1]
                                      |
                                      +-- proposed token(s)
```

The target remains authoritative. A proposal is useful only for the prefix that
matches target verification. This means NextN can be enabled or disabled without
changing the intended sampled distribution, assuming identical target sampling
settings.

## Run

```powershell
llama-server `
  -m C:\models\Qwen3.6-Native-MTP.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 0 `
  -ngl 999 -ngld 999 `
  -ctk turbo4 -ctv turbo3 -ctkd q8_0 -ctvd q8_0 `
  -fa on --context-shift
```

Do not pass the same file through `--model-draft`. When the target GGUF contains
the recognized NextN tensors, speculative setup reopens the same path with the
draft architecture and avoids requiring a separately exported sidecar.

The maintained launcher is
[`scripts/run-qwen36-27b-nextn-server.sh`](scripts/run-qwen36-27b-nextn-server.sh).

### Why there is no `--model-draft`

The current model resolver inspects the target GGUF for the known NextN tensor
signature. When found, it constructs the draft model from the same source path with
an architecture override. Passing the same path again as `--model-draft` is at best
redundant and can select a less efficient or incompatible path depending on model
metadata.

For a sharded or unusually converted file, pass `--spec-type draft-mtp` explicitly
because auto-detection examines the first shard. If that shard does not contain the
signature used by the resolver, omission of `--spec-type` can look like a successful
target-only launch rather than a load error.

## Detection and loading

Draft auto-detection examines the first GGUF shard for the NextN tensor layout.
For Qwen-style combined files, the last target block's `nextn` projection is the
signature used to select `draft-mtp`. Sharded or unusual files should pass
`--spec-type draft-mtp` explicitly.

The draft model is loaded with its architecture override while sharing the same
source file. Target and draft have separate contexts and KV configuration, but
the common speculative wrapper coordinates their sequence memory. The target's
pre-normalized hidden state becomes the input to the NextN layer, and each proposed
token is verified by the target before commitment.

Files with several trained NextN heads use the chain-head path. Draft depth is
clamped to the number of available heads, so `--spec-draft-n-max` cannot address a
head that does not exist.

### Boundary-state handling

For an independent draft context, hidden row `h[p]` is paired with the next token
`x[p + 1]`. At the end of a decode batch, that next token has not arrived yet, so
the driver stores the last hidden row in `pending_h`. The following batch supplies
the token and completes the pair. During speculative verification, the accepted
count selects which hidden row becomes the next boundary.

This is why restoring KV bytes alone is not always sufficient. The secondary
server prompt cache also carries optional speculative implementation state. An
empty payload explicitly clears an old destination boundary; treating it as “do
nothing” can pair the restored cache with a hidden row from an unrelated request.

### Single-head and chained-head files

- A single-head Qwen model reuses the same trained NextN layer autoregressively for
  each requested draft step.
- A multi-head file assigns a trained head to each future step. Before head *i* is
  selected, only the speculative region is removed from that head's draft memory;
  the accepted prefix remains intact.
- `--spec-draft-n-max` is clamped to the number of trained heads for a chained file.
  Asking for a deeper proposal cannot synthesize an untrained head.

## Tuning

- Use `--spec-draft-n-max 2` as the starting point on CUDA. Measure rather than
  assuming a deeper chain is faster.
- `--spec-draft-n-min 0` allows a round to fall back without forcing a low-value
  proposal.
- Use `-ngld`, `-ctkd`, and `-ctvd` for draft placement and draft KV independently
  from the target.
- A combined GGUF should preserve NextN tensors at adequate precision; excessive
  draft-head quantization reduces acceptance even when target quality is intact.

There is no `LLAMA_NEXTN_ASYNC` or `LLAMA_PIPELINE_DEPTH2` control in the converged
driver. Those variables belonged to the former private implementations and should
not be added to launchers.

The old synchronous-versus-depth-2 discussion is still useful history: it explains
why this fork once defaulted to a synchronous multi-step path. It is not a current
runtime switch. The converged driver owns draft scheduling through the generic
speculative interface, so reintroducing either environment variable would create a
second scheduling policy that the current cache and media lifecycle do not know
about.

### Capacity trade-offs

Even though the GGUF is not duplicated on disk, the draft context still needs graph
buffers, KV, and backend allocations. On a 16 GB GPU, compare NextN against a target-
only run at the same `--n-cpu-moe` value and then compare again after reclaiming the
draft memory for target layers. The second comparison is often the meaningful one.

For MoE targets, accepted drafts can amortize expensive target expert evaluation.
For a dense target, a full NextN transformer block may be a larger fraction of the
target cost, so the same acceptance ratio can yield less throughput improvement.

## Vision

An mmproj may remain loaded while NextN handles both text and image-conditioned
turns. Target-produced hidden rows bridge the multimodal prefill into the embedded
MTP context, so a full-history client can resend the original image without
permanently disabling drafting. See [`MTP.md`](MTP.md) for the shared driver.

The distinction is between *media capability* and *media in this request*:

- merely loading `--mmproj` must not disable NextN on text turns;
- image embeddings remain target-owned and are never represented as fake tokens;
- the MTP context captures the target hidden boundary it actually consumes;
- image answers and retained-image follow-ups may both produce accepted drafts;
- prompt-cache hits restore their recorded boundary and coherence, while misses do
  not invent either.

## Model and artifact provenance

The following artifacts were part of the inherited development and benchmark
history. They are listed to preserve credit and reproducibility, not as a promise
that every external filename will remain available forever.

| Artifact | Source and contribution |
|---|---|
| Qwen model weights | Qwen team; follow the license and model card attached to the chosen model |
| Qwen 3.6 MTP GGUF baselines | [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) and [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) |
| MTP-aware importance matrix/BF16 conversion inputs | Unsloth artifacts distributed with those repositories |
| UDT quantization masks and packaging | [`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF) and [`AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF) |
| TurboQuant basis used by the inherited UDT recipe | AtomicBot fork history and [@TheTom](https://github.com/TheTom) contributions |

The historical UDT recipe kept `blk.*.nextn.*` and `mtp.*` tensors at Q8_0 to
protect draft acceptance and raised attention Q/K precision for TurboQuant KV.
Those choices are documented in the older artifact repositories and retained in
Git history. They should not be silently presented as upstream llama.cpp defaults.

## Troubleshooting

Server starts but no draft statistics appear
: Ensure `--spec-type draft-mtp` is present and the GGUF contains the embedded
  NextN signature. Auto-detection may miss unusual sharding.

Draft context reports a hidden-width mismatch
: The combined file is internally inconsistent or was converted with target and
  NextN tensors from different checkpoints.

Acceptance is unexpectedly low
: Compare the embedded-head tensor precision, prompt template, and KV types against
  a known-good quant. Target quality can look normal while an over-quantized draft
  head loses predictive accuracy.

Text fails only after an image turn
: Inspect the media bridge and cached MTP boundary. A retained image should not
  force target-only operation or require vision and NextN to run in separate
  servers.

Performance regresses despite high acceptance
: Draft compute or memory placement is too expensive. Test `--spec-draft-n-max 1`
  and 2, inspect VRAM/shared-memory use, and compare against using that capacity for
  another GPU-resident target layer.

## Validation

The current exact smoke target is a Qwen 3.6 35B-A3B native-MTP combined GGUF on
the RTX 5080. The acceptance test requires a correct text response and nonzero
accepted drafts. Model inference must run in a child process with redirected logs
and a timeout so a backend fault cannot terminate the controlling Codex session.

The broader regression contract is:

1. Combined GGUF detection selects `draft-mtp` without a second model path.
2. A deterministic text prompt returns the expected target answer and records
   accepted drafts.
3. A real image answer succeeds with the matching mmproj and accepted drafts.
4. A full-history follow-up resends the image and continues accepting drafts from
   a coherent cached prefix.
5. Context shifting and prompt-cache save/load do not retain a stale boundary row.
6. Target-only mode remains available as the correctness/performance baseline.

## Historical implementation record

The pre-convergence fork had a separate `nextn` speculative class, shared-model
special cases, and an experimental depth-2 worker. Its full architecture notes,
UDT recipe, benchmark host details, and contributor credits are retained at
[`NEXTN.md` before convergence](https://github.com/bankenichi/atomic-llama-cpp-turboquant/blob/e3cfbe4f7/NEXTN.md).
Those measurements remain useful historical evidence, but their CLI and scheduling
controls describe the old implementation rather than the current `draft-mtp` path.
