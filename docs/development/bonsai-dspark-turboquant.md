# Ternary Bonsai, DSpark, vision, and TurboQuant convergence

This document records the completed 2026-08 convergence needed to run Prism's
Ternary Bonsai 27B release as a first-class model on the owner's Windows/CUDA
system. The work combines the current `llama.cpp` architecture, Prism's ternary
and DSpark implementation, and the private fork's vision and TurboQuant behavior.

## Supported artifacts

The acceptance target is the published
[`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/tree/main)
set, tested without rewriting any file:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-27B-Q2_0.gguf` | 7,165,121,600 | `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757` |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | 1,946,393,568 | `c4810091d244eddc61a0cc4966e584b0959f141e3c66c0d371a6652d9f647da9` |
| `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` | 629,246,880 | `eb561d41a7bbeb0fcf04883c8af11078ef6cae0a66862a0b68443cfca495269d` |

The base file uses the historical Prism physical encoding in which GGUF tensor
type 42 contains group-128 Q2 rows. Type 42 now means upstream group-64 Q2, so
the loader performs a physical-layout preflight and normalizes a coherent legacy
file to the private group-128 runtime type before allocation. Modern group-128
files use the non-colliding type 142. Ambiguous or mixed layouts are rejected
instead of being guessed tensor by tensor.

## DSpark integration

The server and CLI accept the historical `dspark` sidecar through
`--spec-type draft-dspark -md <sidecar>`. A compatibility adapter maps its
metadata and tensors onto the current DFlash/DSpark runtime while retaining the
sidecar's own embeddings, six-layer non-causal draft graph, log-SNR conditioning,
confidence pruning, Markov resampling, and target verification lifecycle.

The sidecar path is opened directly. It is not accidentally replaced by the base
model path, and its 4096-token training context bounds its draft reservation.
Greedy target verification remains authoritative: rejected proposals are rolled
back and target-only greedy output is unchanged.

## Vision remains a primary path

Loading an mmproj does not disable speculation for text. The server distinguishes
an attached image from the mere presence of an mtmd context:

- text turns may draft while the mmproj is loaded;
- media turns bypass drafting and prime the target safely;
- later text turns resume only from coherent draft state;
- templates that cannot represent typed content retain the internal media marker
  through the string-content fallback;
- context shifting snaps past media boundaries;
- Turbo K uses transactional retained-window reprefill rather than an invalid
  in-place RoPE shift.

The exact Bonsai target, DSpark sidecar, and mmproj were tested in one server
process with a text/image/text sequence. The image turn correctly identified the
Apollo 11 moon landing, two forced context shifts completed through reprefill,
the following text turn answered 19 + 23 correctly, and DSpark accepted 13 of 16
drafted tokens in that run.

## TurboQuant KV cache design

TurboQuant cache types remain private runtime formats:

- `turbo2`, `turbo3`, and `turbo4` are available for K and V;
- head widths are padded independently to 128 elements when required (Bonsai's
  576-wide K head becomes 640); padding never changes the logical model width;
- K/V writes quantize complete attention-head groups via `SET_ROWS`;
- a first-class `GGML_OP_TURBO_WHT` rotates Q before attention and inversely
  rotates the result after attention;
- the CPU reference and CUDA implementations cover 32-, 64-, and 128-element
  forward/inverse transforms; the KV graph uses 128;
- CUDA Flash Attention includes homogeneous and layer-adaptive mixed Turbo/Q8/F16
  K/V combinations; the padded 640x512 Bonsai case uses the TILE kernel;
- Metal and Vulkan provide the 128-element WHT operation used by the graph.

TurboQuant requires Flash Attention. `-fa auto` is promoted to enabled; an
explicit `-fa off` fails during context creation rather than constructing an
unsupported graph.

`TURBO_LAYER_ADAPTIVE` retains the existing policies. For a `turbo2` V cache on
models with at least eight layers, policy 7 remains the default: the first two and
last two V-cache layers use Q8 while the middle layers use Turbo2.

### InnerQ safety decision

The inherited `TURBO_INNERQ` implementation calibrated channel scales while the
KV cache was already being populated. Activating a new scale after earlier rows
were stored creates a cache containing two incompatible representations and was
observed to corrupt generation. Dynamic activation is therefore refused with a
warning. The dormant tensor/update plumbing is retained for a future correct
implementation that atomically requantizes every populated K row before changing
the active scale.

## Recommended launch

```powershell
.\llama-server.exe `
  -m Ternary-Bonsai-27B-Q2_0.gguf `
  -md Ternary-Bonsai-27B-dspark-Q4_1.gguf `
  --spec-type draft-dspark --spec-draft-n-max 4 `
  --mmproj Ternary-Bonsai-27B-mmproj-Q8_0.gguf `
  -ngl 99 -ngld 99 `
  -ctk turbo4 -ctv turbo2 -fa on `
  --ctx-size 0 --context-shift --reasoning off
```

Tune GPU layers and context size to keep dedicated VRAM below the owner's
approximately 15 GiB performance cliff. Bonsai itself is dense; do not apply the
Gemma/Qwen `--n-cpu-moe` tuning rules to it.

## Acceptance matrix

| Path | Result |
|---|---|
| Full Windows CUDA `sm_120a` build | Pass |
| CPU-only build of core/backend tests | Pass |
| CUDA `SET_ROWS`, Turbo2/3/4 at exact Bonsai cache row width | 3/3 pass |
| CUDA WHT, forward/inverse at group sizes 32/64/128 | 6/6 pass |
| Exact base, target-only text, Turbo4/Turbo2 KV | Pass |
| Exact base + DSpark, text, Turbo4/Turbo2 KV | Pass |
| Exact base + DSpark + mmproj, image inference | Pass |
| Combined server, forced vision-aware Turbo context shifts | Pass, reprefill used |
| Text after media in the same server | Pass, drafting resumes |
| Generic Windows test suite | 46/46 pass at the upstream checkpoint; rerun for final delivery |

Metal and Vulkan code is restored for source parity but cannot be compiled on the
owner's Windows/CUDA-only toolchain (no Apple Metal compiler or GLSL compiler is
installed). The release workflow intentionally builds CUDA only.
