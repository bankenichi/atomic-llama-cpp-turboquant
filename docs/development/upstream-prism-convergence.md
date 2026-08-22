# Upstream + Prism convergence

## Goal

Bring the personal fork to current `ggml-org/llama.cpp` structure, preserve every
owner-critical vision/MTP/TurboQuant behavior, and integrate Prism's ternary Bonsai
and DSpark support as native capabilities rather than a side branch.

The convergence branch merged upstream through `d775b8967` and Prism work through
`0810d4bfb`, then reapplied private behavior against the modern model, server,
speculative, and GGUF APIs.

## Provenance and merge boundaries

This effort combines independent lines of work; “convergence” does not mean local
authorship of everything that now coexists:

- `ggml-org/llama.cpp` through `d775b8967` supplies the modern server, model API,
  mtmd stack, GGUF reader, and generic speculative implementations. Upstream
  contributor authorship remains in the merged history.
- `PrismML-Eng/llama.cpp` through `0810d4bfb` supplies the Prism ternary and DSpark
  work used as the second integration source. The imported commits retain authors
  such as Pasha Khosravi and Brian / `bri-prism` rather than being replayed as local
  patches.
- `AtomicBot-ai/atomic-llama-cpp-turboquant` is the parent of this personal fork and
  supplies the pre-convergence Gemma MTP, Qwen NextN, TurboQuant, launcher, and
  benchmark lineage.
- `thecodacus/llama.cpp` branch `fable5/prefetch-experts` is the source of expert
  prefetch overlap and mmap host pinning; Windows page-registration and subsequent
  integration are local adaptations.
- Commits after those merge points implement the collision policy, historical
  assistant/DSpark compatibility, adaptive Bonsai CUDA correction, and server
  coherence rules required to make the lines work together.

See [`../../NOTICE.md`](../../NOTICE.md) for links and the standing attribution
policy. The merge commits and imported authorship are intentional project records
and should not be flattened during future rebases.

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
retain a larger context independently. Atomic maps that larger absolute target
domain into a per-sequence private draft window: old DSpark KV is removed and
shifted toward local position zero in 512-position chunks while space for a
complete trained block stays reserved. This preserves the sidecar's trained
limit without permanently disabling drafting when a target conversation crosses
position 4096. The configured mask token is parsed strictly and must fall inside
the draft vocabulary.

The target-to-local origin and committed local end are versioned checkpoint
state. A prompt prefix outside the retained draft window forces a paired rebuild,
cache-chunk relocation is disabled for private draft positions, and target
context shift selects transactional reprefill. Provisional diffusion rows are
removed immediately after sampling; only target-verified feature rows become
committed draft KV. These rollover and cache-coordination rules are local Atomic
adaptations on top of upstream DFlash/DSpark and Prism's legacy implementation.

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
contains media. `has_media()` detects real media, while the speculative coordinator
chooses a policy for each implementation and sequence:

- DFlash/DSpark accumulate target-derived encoder features and mean-pool M-RoPE
  image rows into the image's actual one-dimensional context-position span.
- EAGLE3 keeps the final target-derived image feature at a compact private
  boundary. Its provisional draft KV is removed in that same private domain before
  target verification rebuilds committed rows.
- MTP/NextN consumes the target hidden boundary supported by the loaded model;
  image embeddings are not fabricated as tokens.
- draft-simple reconstructs serializable text in a compact draft-only position
  domain and never decodes a foreign mmproj vector.
- n-gram implementations remain stateless and use real text-token history.

This makes a retained image in a full-history client request compatible with
continued drafting. `server_spec_context_state` still records whether cacheable
state is coherent. If one implementation fails, it alone becomes ineligible; the
already-decoded target verification batch is finalized and the target answer
continues. The secondary RAM prompt cache stores the coherence bit and optional
boundary state alongside target/draft bytes. Cache misses are explicit no-ops and
cache hits restore the saved entry rather than inferring state from a new task.

Vision context shift keeps two paths:

- in-place f16/q8 K shift with chunk-aware range erasure and media-boundary snap;
- transactional retained-window reprefill for TurboQuant and M-RoPE.

DFlash/DSpark always select the reprefill path during a target context shift,
even for otherwise shiftable Q8 draft KV, because their local positions no
longer equal target positions after the first rollover. During retained-media
replay, pooled image features and text both pass through the same mapping.

The coherence bit is semantic state, not a cache optimization hint. It answers a
specific question: “do the active stateful drafters represent a safe continuation
of the target's accepted prefix?” Real media does not make the answer false when
the active policy consumed or explicitly rebased that media boundary. It becomes
false only when a required implementation is skipped, fails, or cannot restore its
state. This timing prevents target prefill success from hiding a failed drafter.

The secondary prompt cache distinguishes three outcomes—restore failure, cache
miss, and cache hit—because they have different state transitions. A miss preserves
the current invalidation. A hit restores target bytes, draft bytes, coherence, and
optional implementation boundary state as one entry. A failed multi-part restore
does not consume the entry.

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
| Ternary Bonsai 27B | exact Q2_0-G128 base + DSpark Q4_1 + exact mmproj, CUDA, target `turbo4`/`turbo3`, draft Q8_0/Q8_0 | a 15,626-token text prompt crossed repeated 4096-position draft-window rollovers and accepted 38/90 drafts; an image answer accepted 86/162; its retained-image 8,283-token follow-up crossed another seven rollovers and accepted 30/126 |
| Ternary Bonsai context shift | same exact stack, forced 2048 target context, 1,050-token image prompt + 1,400-token generation | transactional reprefill shifted the target context and generation completed; DSpark remained active and accepted 904/1,979 drafts |
| Gemma 4 26B-A4B | exact owner target + historical `gemma4_assistant` + exact mmproj + CPU MoE + Turbo KV | text returned 42 and vision identified the UI screenshot; both turns accepted 2/2 drafts |
| Qwen 3.6 35B-A3B | exact combined native-MTP GGUF + matching mmproj + CPU MoE + embedded `draft-mtp` | text returned 42 and vision identified the UI screenshot; text accepted 2/2 drafts and vision accepted 2/6 |
| CUDA operator | Bonsai 576x512, separate F16 K / Q8_0 V | numerical comparison passed |
| CUDA operator | 576x512 Q8_0 K/V, V subview of K | numerical comparison passed |
| Full Windows CUDA build | complete tool suite, `sm_120` | built successfully; 48/48 CTest tests passed |
| Full Windows CPU build | complete tool suite | built successfully; 48/48 CTest tests passed |
| Maintained shell launchers | `bash -n` | all passed |

The focused regression tests are:

- `test-gguf`: group-64, group-128, mixed, ambiguous raw type 42, and legacy
  private type normalization;
- `test-arg-parser`: DSpark per-slot context cap;
- `test-server-spec-state`: policy selection plus supported/unsupported media
  coherence transitions;
- `test-backend-ops`: separate adaptive Bonsai KV layout plus existing MLA layout.

The owner-machine helper starts each exact stack as a hidden child, redirects both
streams, enforces a readiness deadline, and reports its PID and log paths as JSON:

```powershell
.\scripts\start-hidden-atomic-server.ps1 -Stack Ternary -ContextSize 32768 -Port 8097
.\scripts\start-hidden-atomic-server.ps1 -Stack Gemma   -ContextSize 8192  -Port 8098
.\scripts\start-hidden-atomic-server.ps1 -Stack Qwen    -ContextSize 8192  -Port 8099
```

The helper intentionally uses the owner's exact filenames under
`C:\Program Files\llamacpp`; it is regression infrastructure, not a portable
model-discovery wrapper. On success the caller owns the returned process and must
stop it after requests finish. On load failure or readiness timeout the helper
terminates its own child so a failed run cannot strand VRAM.

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
