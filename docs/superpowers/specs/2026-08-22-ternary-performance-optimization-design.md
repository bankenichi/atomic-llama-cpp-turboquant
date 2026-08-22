# Ternary Bonsai performance optimization design

**Date:** 2026-08-22  
**Status:** Approved benchmark design  
**Scope:** Ternary Bonsai 27B + DSpark + mmproj on the owner's RTX 5080

## Objective

Find the fastest reliable configuration for each practical context tier—32K,
64K, and 128K—without sacrificing model capability, long-context retrieval, or
vision. The result may be three different profiles because attention cost,
DSpark acceptance, and VRAM pressure change substantially with occupied context.

The existing Turbo4 K / Turbo3 V profile is the quality and performance
reference. Lower-precision KV candidates are eligible only after deterministic
quality validation. The work does not revisit 192K or 256K: completed benchmarks
already place near-full decode at 5.16 tok/s or lower, outside the owner's 20
tok/s usability floor.

## Fixed system and model configuration

Every comparison uses the same binaries and model artifacts:

- Ternary Bonsai 27B Q2_0-G128 target;
- matching Prism DSpark Q4_1 sidecar;
- matching Q8_0 mmproj, loaded for every canonical run;
- all 65 target layers and all seven draft layers offloaded to the RTX 5080;
- DSpark draft KV fixed at lossless Q8_0 K/V;
- Flash Attention enabled;
- one server slot and one parallel sequence;
- context shifting enabled;
- no CPU layer offload;
- temperature 0, top-k 1, and seed 4242 for deterministic comparisons.

The benchmark records driver/build identity, available system RAM, dedicated GPU
memory before load, after server readiness, and after generation. Runs with an
unexplained background-VRAM difference above 256 MiB from their paired baseline
are marked contaminated and repeated rather than silently compared.

Exact model inference must remain a hidden child process with stdout and stderr
redirected and a finite request timeout. A 48-token generation probe stops a
point immediately when decode is below 20 tok/s. Child cleanup is verified after
every point.

## Workloads

### Short interactive workload

The existing fixed 1,798-token technical prompt measures normal interactive
decode with each configured context allocation. It separates steady-state graph
and speculative overhead from near-full attention cost.

### Near-full occupied workload

Each tier evaluates exactly `ctx_size - 512` deterministic prompt tokens, then
forces a 48-token watchdog probe. Passing points generate 192 additional measured
tokens. The three occupied prompts are therefore 32,256, 65,024, and 130,560
tokens.

This workload records:

- prompt-evaluation time and rate;
- probe and measured decode rate;
- DSpark accepted and proposed token counts;
- graph reuse counters;
- GPU and system-memory residency;
- cache, rollover, CUDA, and server errors.

## Staged optimization funnel

The benchmark is deliberately staged. A setting must demonstrate a measurable
benefit before it is combined with other settings or subjected to the expensive
quality suite.

### Stage 1: quality-neutral runtime screen

Each candidate is first tested independently against the Turbo4/Turbo3 baseline.
Independent A/B tests prevent an apparent win from being incorrectly attributed
to a bundle of unrelated switches.

1. **CUDA graph QKV optimization**
   - default behavior;
   - `GGML_CUDA_GRAPH_OPT=1`.

2. **Prompt batch geometry**
   - logical 2048 / physical 512 (current default);
   - logical 2048 / physical 1024;
   - logical 4096 / physical 1024.

   Batch geometry is judged primarily on prefill. It must not create a transient
   VRAM spill or reduce decode enough to erase the prefill gain.

3. **CPU submission profile for the hybrid i9-14900K**
   - current automatic thread placement and normal priority;
   - P-core-focused CPU ranges, strict affinity, high (not realtime) priority,
     and full polling.

   Main and draft settings remain symmetrical unless isolated evidence shows that
   the server and DSpark benefit from different placement. The test must confirm
   both throughput and stable desktop behavior; realtime priority is excluded.

4. **DSpark proposal control**
   - depth 4 with confidence threshold 0.00 (current default);
   - depth 4 with confidence thresholds 0.05, 0.10, and 0.20;
   - depth 3 as an explicit 64K control, because the previous near-full 64K
     workload favored depth 3 when depth-4 acceptance fell to 69%.

   Speculative controls are correctness-neutral only when target verification
   remains authoritative. Output tokens must match the deterministic target
   result, and acceptance counters must remain internally consistent.

Candidates that do not improve their intended metric are removed. The surviving
independent wins are then combined and re-run against a fresh baseline to detect
negative interactions.

### Stage 2: target KV bandwidth sweep

Using the best quality-neutral runtime configuration from Stage 1, test:

1. Turbo4 K / Turbo3 V — reference;
2. Turbo4 K / Turbo2 V;
3. Turbo3 K / Turbo3 V;
4. Turbo3 K / Turbo2 V.

Draft KV remains Q8_0/Q8_0. Adaptive Turbo layer selection remains enabled,
because it is part of the fork's normal quality policy; disabling it would make
the result unrepresentative of the intended deployment.

All four pairs run at short and near-full occupancy for 32K, 64K, and 128K.
Only pairs that improve at least one tier proceed to Stage 3.

### Stage 3: deterministic quality validation

Performance alone cannot promote a lower-precision KV profile. Every surviving
candidate is compared with Turbo4/Turbo3 under identical greedy sampling and
prompt construction.

The quality suite has three independent gates:

1. **Objective text capability**
   - arithmetic and multi-step logic with explicit answer keys;
   - constrained code reasoning and transformation tasks with mechanically
     checkable outputs;
   - structured instruction following with exact schema validation.

2. **Long-context retrieval**
   - unique facts placed near the beginning, middle, and end of each occupied
     context tier;
   - answers requested in a fixed JSON structure;
   - all needles must be recovered exactly at 32K, 64K, and 128K.

3. **Real-media vision**
   - at least two local, reproducible images with explicit answer keys;
   - object/content recognition and spatial or textual detail questions;
   - DSpark remains enabled for the image answer and a following text-only turn;
   - accepted drafts must resume correctly after the media turn.

The harness also requests top-token probabilities for deterministic completions.
Token agreement and probability drift are diagnostic evidence, not a substitute
for objective scoring: semantically equivalent open text may diverge without
representing a capability loss, while an exact retrieval failure is decisive.

## Repetition and promotion rules

Screening points run once. A candidate that appears to win is interleaved with a
fresh baseline and repeated until each has three clean samples. This controls for
temperature, clocks, and changing desktop VRAM without tripling every losing
configuration.

A configuration may be promoted for a context tier only when:

- median decode improves by at least 3%, or median prefill improves by at least
  5% without a material decode regression;
- all three confirmation runs remain above 20 decode tok/s;
- no objective text score declines;
- every long-context needle is recovered;
- every required vision answer remains correct;
- image-answer and following-text DSpark behavior remains correct;
- no cache, rollover, CUDA, or speculative error is introduced;
- dedicated VRAM remains approximately 15 GB or below without shared-memory
  spill.

Results below the improvement thresholds are treated as noise even if their raw
number is marginally higher.

## Deliverables

The completed work produces:

- a machine-readable result file for every point;
- redirected server logs retained with each result;
- a human-readable comparison table covering performance, acceptance, memory,
  and quality;
- an explicit winner—or a documented decision to retain the baseline—for 32K,
  64K, and 128K;
- launcher and configuration-UI presets only for configurations that pass every
  promotion gate;
- documentation explaining why rejected settings were not adopted.

No production preset is changed during screening. Profile changes occur only
after the complete deterministic and vision validation suite passes.

## Expected interpretation

Likely gains fall into different categories. Batch geometry can improve prefill
without helping decode. CUDA graph and CPU submission tuning may provide smaller
steady-state gains. DSpark confidence control may specifically recover the 64K
depth-4 acceptance dip. Lower-bit KV has the largest plausible long-context
bandwidth benefit, but also the only direct capability risk and therefore carries
the strictest validation burden.

The design intentionally permits different winners by context tier. A single
compromise profile is not considered superior to three evidence-backed presets
when the owner's workloads have distinct 32K, 64K, and 128K niches.
