# Qwen embedded NextN

Qwen 3.x combined MTP GGUFs store their NextN draft layers alongside the target
weights. This fork runs them through the unified `draft-mtp` speculative engine.
The old fork-specific `nextn` speculative type and async-worker environment switch
no longer exist after convergence with current `llama.cpp`.

## Run

```powershell
llama-server `
  -m C:\models\Qwen3.6-Native-MTP.gguf `
  --spec-type draft-mtp `
  --spec-draft-n-max 2 `
  --spec-draft-n-min 0 `
  -ngl 999 -ngld 999 `
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 `
  -fa on --context-shift
```

Do not pass the same file through `--model-draft`. When the target GGUF contains
the recognized NextN tensors, speculative setup reopens the same path with the
draft architecture and avoids requiring a separately exported sidecar.

The maintained launcher is
[`scripts/run-qwen36-27b-nextn-server.sh`](scripts/run-qwen36-27b-nextn-server.sh).

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

## Vision

An mmproj may remain loaded while NextN handles text turns. A request with real
media invalidates draft synchronization, and the next text request performs a full
paired rebuild before speculative generation resumes. This is the same lifecycle
used by Gemma MTP and DSpark; see [`MTP.md`](MTP.md).

## Validation

The current exact smoke target is a Qwen 3.6 35B-A3B native-MTP combined GGUF on
the RTX 5080. The acceptance test requires a correct text response and nonzero
accepted drafts. Model inference must run in a child process with redirected logs
and a timeout so a backend fault cannot terminate the controlling Codex session.
