# Runtime performance features

The owner deployment is a single RTX 5080 with 16 GB VRAM. Performance work must
preserve correctness for vision, speculative decoding, and CPU-offloaded MoE.

## Quick reference

| Feature | Enable | Requires | Main trade-off |
|---|---|---|---|
| Expert upload prefetch | `GGML_SCHED_PREFETCH_EXPERTS=1` | `--n-cpu-moe`, CUDA async/events | Staging slots consume VRAM |
| mmap host registration | `GGML_CUDA_REGISTER_HOST=1` | mmap enabled | Pins mapped pages in RAM |
| Turbo adaptive KV | default; disable with `TURBO_LAYER_ADAPTIVE=0` | Turbo KV | Boundary layers use higher precision |
| MTP/NextN | `--spec-type draft-mtp --spec-draft-n-max 2` | matching assistant or embedded head | Draft weights/KV consume capacity |
| DSpark | `--spec-type draft-dspark --model-draft ...` | matching Prism sidecar | Draft context is capped at 4096/slot |

## Expert upload prefetch

`GGML_SCHED_PREFETCH_EXPERTS` overlaps H2D transfer of CPU-resident
`MUL_MAT_ID` expert tensors with computation through a second same-device backend
instance and event-ordered rotating staging slots.

```powershell
$env:GGML_SCHED_PREFETCH_EXPERTS = "1"
llama-server ... --n-cpu-moe 14
```

The value is the staging-slot count. `1` selects the default of three slots; a
larger numeric value requests that many slots. The path activates only when op
offload is active. It is most useful with pinned host allocations, including the
allocation produced by `--no-mmap`.

## mmap host registration

`GGML_CUDA_REGISTER_HOST=1` calls the backend host-buffer registration function on
the mmap pages retained for CPU weights. The Windows implementation aligns ranges
with `GetSystemInfo` before `cudaHostRegister`.

```powershell
$env:GGML_CUDA_REGISTER_HOST = "1"
llama-server ... # keep mmap enabled; do not pass --no-mmap
```

Use this when mmap startup/RAM behavior matters but CPU expert transfers still
need DMA. Compare it with `--no-mmap` plus expert prefetch; do not assume combining
both switches is faster.

## TurboQuant KV

Turbo KV types are selected with the normal cache-type arguments:

```text
-ctk turbo4 -ctv turbo2 -ctkd turbo3 -ctvd turbo3 -fa on
```

Adaptive selection is enabled by default. For Turbo2 V, boundary layers may use
Q8_0 to protect quality. Bonsai's native attention geometry is 576 K by 512 V;
with f16 K this becomes a mixed F16/Q8_0 flash-attention pair. CUDA chooses the
separate-V MMA stride specialization for this layout rather than the DeepSeek MLA
K-subview specialization.

Turbo K does not support in-place rotary K shift. The server therefore performs a
transactional reprefill of the retained prompt window during context shift. See
[`context-shift-with-vision.md`](context-shift-with-vision.md).

## Speculative decoding

Current CLI:

```text
--spec-type draft-mtp --spec-draft-n-max 2
--spec-type draft-dspark --model-draft dspark.gguf --spec-draft-n-max 4
```

Removed names such as `--draft-max`, `--draft-block-size`, `--mtp-head`,
`--spec-type mtp`, and `--spec-type nextn` must not appear in maintained launchers.
See [`../../MTP.md`](../../MTP.md), [`../../NEXTN.md`](../../NEXTN.md), and
[`../speculative.md`](../speculative.md).

## Capacity rule

Keep dedicated GPU memory at or below roughly 15 GB. Above that point, transient
decode buffers can spill to shared system memory and reduce throughput sharply.
For MoE models, tune `--n-cpu-moe` before changing speculative depth or KV
precision. A speculative head that displaces a full expert layer may be a net loss.
