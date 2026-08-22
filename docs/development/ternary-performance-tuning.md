# Ternary Bonsai performance tuning on RTX 5080

This report records the context-tier optimization completed on 2026-08-22 for
the owner's RTX 5080. It is intentionally model- and machine-specific: the
recommended profiles are not generic llama.cpp defaults.

The exact target is Hikari07jp's
[`Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf`](https://huggingface.co/Hikari07jp/Ternary-Bonsai-27B-Abliterated-LowDeg-GGUF/blob/main/Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf),
a direct low-degradation derivative of Prism ML's Ternary Bonsai 27B using the
official Q2_0-G128 ternary pack. The matching Q4_1 DSpark sidecar and Q8_0
multimodal projector come from Prism's release. DSpark and the base-model
artifacts originate with
[`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) and
[`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/tree/main).
TurboQuant cache formats and their adaptive layer policy come from the Atomic
fork lineage. This document reports local integration and tuning; it does not
claim authorship of those upstream components.

## Recommended profiles

| Tier | Target KV | DSpark depth | Draft KV | Near-full measured median | Decision |
|---|---|---:|---|---:|---|
| 32K | Turbo3 K / Turbo3 V | 4 | Q8_0 / Q8_0 | 142.9 tok/s | Promote; 3.9% over the 137.6 tok/s reference median |
| 64K | Turbo3 K / Turbo2 V | 3 | Q8_0 / Q8_0 | 101.9 tok/s | Promote; 7.0% over the 95.3 tok/s reference median |
| 128K | Turbo3 K / Turbo3 V | 4 | Q8_0 / Q8_0 | 80.5 tok/s | Promote; 8.3% over the 74.4 tok/s reference median |

The promoted 32K and 128K profiles change only target KV relative to the
Turbo4/Turbo3 reference. The 64K profile also uses depth 3 because the independent
depth screen found a large acceptance/throughput advantage over depth 4. Draft KV
remains fixed at the owner's lossless Q8_0/Q8_0 policy at every tier; this sweep
did not compare it with f16.

## Controlled setup

Every canonical point used the same exact target, sidecar, projector, CUDA build,
all target and draft layers on GPU, Flash Attention, one server slot, temperature
0, top-k 1, and seed 4242. Exact inference ran as a hidden child with redirected
logs and a finite timeout. Near-full prompts occupied `ctx_size - 512` tokens.
Each confirmation point first ran a 48-token usability watchdog and, if it stayed
above 20 tok/s, an independent 192-token measurement.

The tested artifact identity is deliberately explicit because the target is a
derivative GGUF rather than Prism's canonical base file:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf` | 7,165,121,600 | `527f276ddf047b3494de964985b5529b9fc3ecf739ea64d8fab3a6ef8403e81d` |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | 1,946,393,568 | `c4810091d244eddc61a0cc4966e584b0959f141e3c66c0d371a6652d9f647da9` |
| `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` | 629,246,880 | `eb561d41a7bbeb0fcf04883c8af11078ef6cae0a66862a0b68443cfca495269d` |
| `build-rollover/bin/Release/llama-server.exe` | 10,752 | `9a8c68ca829d06feba3f342b37e12e9631164a4ef09672712ba123428cd78526` |

The binary was built from source commit
`7bdb1947d1c6286be249861d23587034fab9a40e`. Commit
`ce97dca2c2962696887060e50a78908abae86824` added the approved benchmark plan;
it did not change the executable. The working-tree benchmark harness changes are
documented here. Tests used NVIDIA driver 610.88, and the GPU reported 16,303 MiB
total dedicated memory.

The complete approved method and promotion rules are in
[`docs/superpowers/specs/2026-08-22-ternary-performance-optimization-design.md`](../superpowers/specs/2026-08-22-ternary-performance-optimization-design.md).
Local machine-readable results and server logs are retained under
`build-rollover/benchmarks/`; that directory is intentionally ignored because it
contains machine-specific build and runtime output. The committed
[`ternary-performance-results-20260822.json`](ternary-performance-results-20260822.json)
manifest preserves build/artifact identity, the screening matrix, clean
confirmation samples, contamination decisions, and exact quality outcomes.

## Screening results

Quality-neutral runtime switches did not produce a reliable independent win:

- `GGML_CUDA_GRAPH_OPT=1` was neutral on the short workload and slower near full.
- Larger physical prompt batches improved prefill by roughly 1–2% but reduced
  decode by 15–18% and used about 1.1 GiB more VRAM.
- P-core-only CPU placement did not beat clean automatic-placement samples.
- DSpark confidence thresholds from 0.05 through 0.20 did not remove proposals
  in the tested 64K workload, so they added no useful control.
- At 64K, depth 3 raised near-full decode from roughly 70.7 to 82.8 tok/s because
  depth-4 acceptance had fallen to about 69%. This is why 64K is the only tier
  promoted with depth 3.

The target-KV sweep then tested Turbo4/Turbo3, Turbo4/Turbo2,
Turbo3/Turbo3, and Turbo3/Turbo2 at short and near-full occupancy. Important
single-pass near-full observations were:

| Tier/depth | T4/T3 | T4/T2 | T3/T3 | T3/T2 |
|---|---:|---:|---:|---:|
| 32K / 4 | 116.0 | 92.4 | 122.7 | 99.5 |
| 64K / 3 | 82.7 | 82.7 | 81.9 | 89.7 |
| 64K / 4 | 68.0 | 85.6 | 77.3 | 79.3 |
| 128K / 4 | 60.0 | 61.0 | 72.1 | 58.3 |

Those screening values select candidates; they are not the final medians. The
64K depth-4 Turbo4/Turbo2 result was explicitly tested because depth 4 can be
useful when acceptance is high, but it later exhausted 2,048 output tokens during
the near-full retrieval task without completing the required JSON. It is therefore not exposed as a
recommended profile.

## Deterministic quality gates

Each finalist and its reference ran four mechanically scored text tasks, a
three-needle near-full retrieval task, and two real local images. Vision coverage
included an image answer and a retained-image follow-up; DSpark acceptance
counters were present after both, confirming that a retained image did not become
a permanent speculation veto.

| Configuration | Text | Retrieval | Vision | Result |
|---|---:|---:|---:|---|
| 32K d4 T4/T3 reference | 4/4 | pass at 26,972 tokens | 3/3 | pass |
| 32K d4 T3/T3 | 4/4 | pass at 26,972 tokens | 3/3 | pass |
| 64K d3 T4/T3 reference | 4/4 | pass at 57,912 tokens | 3/3 | pass |
| 64K d3 T3/T2 | 4/4 | pass at 57,912 tokens | 3/3 | pass |
| 64K d4 T4/T2 | 4/4 | fail at 57,912 tokens (`finish_reason=length`) | 3/3 | reject |
| 128K d4 T4/T3 reference | 4/4 | pass at 119,809 tokens | 3/3 | pass |
| 128K d4 T3/T3 | 4/4 | pass at 119,809 tokens | 3/3 | pass |

The final quality harness requires exact normalized constrained answers, an exact
three-key retrieval JSON object, fixed image answer keys, and `finish_reason=stop`.
It also associates nonzero DSpark acceptance counters with the image answer and
retained-image follow-up by request name. This matters because an early harness
version scored token-capped thought traces and substring matches as answers; that
measurement bug was corrected and the entire matrix regenerated before promotion.

## Confirmation samples

Each promoted tier has three clean, interleaved reference/candidate pairs using
the same 192-token measured request. Measured medians never mix in the earlier
48-token screening probes. At 128K, the first attempted pair had a 268 MiB
background-VRAM delta and was excluded; a replacement pair was run so three clean
pairs remained. Probe medians independently confirm the direction.

| Tier | Reference measured median | Candidate measured median | Change | Probe change |
|---|---:|---:|---:|---:|
| 32K | 137.6 | 142.9 | +3.9% | +2.6% |
| 64K | 95.3 | 101.9 | +7.0% | +3.7% |
| 128K | 74.4 | 80.5 | +8.3% | +14.7% |

All clean confirmation samples remained above 20 tok/s. The clean 128K pairs also
reduced median ready dedicated VRAM by 312 MiB and raised measured-request draft
acceptance from 93.2% to 96.8%.

## Launching the tiers

The tracked owner-compatibility wrapper selects the optimized profile explicitly:

```powershell
.\run-gemma.ps1 -Ternary                          # standard 32K
.\run-gemma.ps1 -Ternary -TernaryPreset 64k      # 64K, T3/T2, depth 3
.\run-gemma.ps1 -Ternary -TernaryPreset 128k     # 128K, T3/T3, depth 4
```

For compatibility with existing commands, `-Ternary -CtxSize 65536` and
`-Ternary -CtxSize 131072` automatically select the matching tuned preset.
The canonical launcher exposes the same profiles as `-Preset 64k` and
`-Preset 128k`, and the config UI ships corresponding read-only starter profiles.
