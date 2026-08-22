# Upstream + Prism convergence

## Goal

Bring the personal fork to current `ggml-org/llama.cpp` structure, preserve every
owner-critical vision/MTP/TurboQuant behavior, and integrate Prism's ternary Bonsai
and DSpark support as native capabilities rather than a side branch.

The convergence branch merged upstream through `d775b8967` and Prism work through
`0810d4bfb`, then reapplied private behavior against the modern model, server,
speculative, and GGUF APIs.

## Resulting architecture

### Ternary Bonsai

The Bonsai base uses a group-128 Q2_0 physical format. Runtime type
`GGML_TYPE_Q2_0_G128` is assigned collision-free ID 142. Published Prism GGUFs
predate that assignment and encode the format under raw type 42, which current
upstream assigns to group-64 Q2_0.

The GGUF reader resolves this collision from tensor spans:

- a span possible only under group-128 normalizes every raw-type-42 tensor in the
  shard to type 142;
- a span possible only under group-64 retains standard type 42;
- contradictory evidence rejects the file;
- if all spans are alignment-ambiguous, raw 42 retains its standardized group-64
  meaning because the published legacy metadata contains no reliable discriminator.

Private TurboQuant/TQ types moved to IDs 240-244. Historical on-disk types 43-46
are unambiguous and normalize to 241-244. Historical Turbo2 ID 42 is not claimed:
Turbo2 was a runtime KV format, while its bytes collide with Prism Q2_0-G128.

### DSpark

`common_speculative_impl_draft_dflash` supports both modern `dflash.*` metadata and
the published `dspark.dspark.*` namespace. Block size and mask-token metadata use
strict integer parsing with bounds; malformed values fail instead of being partly
accepted by `atoi`.

DSpark sidecars are trained for 4096 positions. Draft context is capped at
`4096 * n_parallel`, while a smaller explicit context is preserved. The target may
retain a larger context independently. The configured mask token is parsed
strictly and must fall inside the draft vocabulary.

### Unified MTP/NextN

Current `draft-mtp` replaces the private `mtp` and `nextn` speculative classes.
Gemma uses `--model-draft`; Qwen combined files reopen the target GGUF through the
draft architecture without an explicit draft path. Maintained CLI uses
`--spec-draft-n-max/min`.

Historical Gemma assistant files remain loadable. The raw architecture name is
preserved long enough to choose legacy metadata and tensor namespaces after it is
normalized to the current internal architecture.

### Vision and server caches

An mmproj being loaded means the server is media-capable, not that every request
contains media. Draft and prime gates use `has_media()`.

A media request advances only the target context for implementations that cannot
consume the media representation. `server_spec_context_state` records this. The
next text request cannot reuse a target-only cache prefix: it forces `n_past = 0`,
clears checkpoints, removes both sequence memories, and rebuilds them together.
The secondary RAM prompt cache stores the coherence bit and optional speculative
implementation state alongside target/draft bytes. Cache misses are explicit
no-ops, so an unrelated text request cannot erase media invalidation; cache hits
restore the saved entry's state rather than inferring coherence from the new task.

Vision context shift keeps two paths:

- in-place f16/q8 K shift with chunk-aware range erasure and media-boundary snap;
- transactional retained-window reprefill for TurboQuant and M-RoPE.

### Adaptive TurboQuant CUDA

Turbo2's default adaptive policy can promote Bonsai boundary V layers to Q8_0
while K stays f16. This yields a separate F16 K / Q8_0 V cache with native
576x512 attention geometry.

The CUDA MMA code previously inferred `V_is_K_view` solely from `DKQ == 576`, an
assumption valid for DeepSeek MLA but false for Bonsai. That selected K's 576-wide
stride for a separate 512-wide V allocation and produced large numerical error.
Dispatch now examines the actual graph view relationship and selects one of two
compiled 576 specializations. The original MLA K-subview specialization remains.

## Acceptance matrix

All model runs execute in hidden child processes with redirected logs and timeouts.

| Target | Configuration | Result |
|---|---|---|
| Ternary Bonsai 27B | exact Q2_0-G128 base + DSpark Q4_1 + exact mmproj, CUDA, `-ctk f16 -ctv turbo2`, context reprefill | vision identified Apollo 11; following text returned 42; DSpark accepted 13/16 drafts |
| Gemma 4 26B-A4B | exact owner target + historical `gemma4_assistant` Q8_0 + exact mmproj + CPU MoE + Turbo KV | returned 42; multimodal model loaded; 2/2 drafts accepted |
| Qwen 3.6 35B-A3B | exact combined native-MTP GGUF + CPU MoE + embedded `draft-mtp` | returned 42; 2/2 drafts accepted |
| CUDA operator | Bonsai 576x512, separate F16 K / Q8_0 V | numerical comparison passed |
| CUDA operator | 576x512 Q8_0 K/V, V subview of K | existing MLA controls passed |
| Full Windows CUDA build | complete tool suite, `sm_120` | built successfully; 48/48 CTest tests passed |
| Full Windows CPU build | complete tool suite | built successfully; 48/48 CTest tests passed |
| Maintained shell launchers | `bash -n` | all passed |

The focused regression tests are:

- `test-gguf`: group-64, group-128, mixed, ambiguous raw type 42, and legacy
  private type normalization;
- `test-arg-parser`: DSpark per-slot context cap;
- `test-server-spec-state`: media invalidation, paired rebuild, and secondary-cache
  miss/hit coherence transfer;
- `test-backend-ops`: separate adaptive Bonsai KV layout plus existing MLA layout.

## Maintained launch syntax

```text
# Bonsai
--model-draft dspark.gguf --spec-type draft-dspark --spec-draft-n-max 4

# Gemma assistant
--model-draft assistant.gguf --spec-type draft-mtp --spec-draft-n-max 2

# Qwen embedded NextN
--spec-type draft-mtp --spec-draft-n-max 2
```

Do not copy historical commands using `--mtp-head`, `--draft-block-size`,
`--draft-max`, `--draft-min`, `--spec-type mtp`, or `--spec-type nextn` into
current scripts.
