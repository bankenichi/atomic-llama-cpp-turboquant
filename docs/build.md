# Build (Windows + CUDA)

This is a **personal fork** maintained for **CUDA on Windows** only — Blackwell `sm_120`
(RTX 50-series). Other backends (Metal, Vulkan, HIP, SYCL, …) still exist in the underlying
engine but are not built, tested, or documented here; build them yourself from upstream
llama.cpp if you need them. See [`AGENTS.md`](../AGENTS.md) for the orientation overview and
[`docs/development/mtp-cpumoe-vision-fixes.md`](development/mtp-cpumoe-vision-fixes.md) for the
fork's local modifications.

For a ready-to-run binary instead of building, grab the bundled zip from the rolling
**`latest`** GitHub Release (produced by `.github/workflows/windows-cuda.yml`). It includes the
CUDA runtime DLLs, so it runs with just the NVIDIA driver + VC++ Redist installed.

## Prerequisites

- **Visual Studio 2022** (Community is fine) with the **Desktop development with C++** workload.
  This provides MSVC, CMake, and Ninja.
- **CUDA Toolkit** (13.1 is what CI uses) from the
  [NVIDIA developer site](https://developer.nvidia.com/cuda-downloads).
- An NVIDIA GPU. The fork targets `sm_120` (Blackwell); change `CMAKE_CUDA_ARCHITECTURES` for
  other cards.

> **Toolchain gotcha:** CUDA's `nvcc` can crash (`cudafe++` access violation) against
> bleeding-edge MSVC such as the VS 2026 preview. Build with the **VS 2022 (v143) toolset**.
> From a *Developer PowerShell/Command Prompt for VS 2022* the right toolset is already active;
> if you have a newer VS also installed, force v143 with `-T v143` (Visual Studio generator) or
> by launching the VS 2022 environment explicitly.

## Build

From a **Developer Command Prompt for VS 2022** (x64), in the repo root:

```bat
cmake -S . -B build -G "Ninja Multi-Config" ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DGGML_CUDA_CUB_3DOT2=ON ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build --config Release
```

Notes on the flags:

- `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` — CUDA build for Blackwell `sm_120`.
- `-DGGML_CUDA_CUB_3DOT2=ON` — use the CUB 3.2 path (needed with recent CUDA toolkits).
- `-DLLAMA_BUILD_BORINGSSL=ON` — statically link BoringSSL so the HTTPS-capable `llama-server`
  needs no external OpenSSL DLLs. (Use `-DLLAMA_CURL=OFF` to drop the libcurl dependency.)
- Add `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF` to build just the tools faster.

To build only the server: `cmake --build build --config Release --target llama-server`.
Omit `--target` to build the full tool suite (`llama-server`, `llama-cli`, `llama-quantize`,
`llama-mtmd-cli`, `llama-bench`, `llama-perplexity`, `llama-gguf-split`, …). Binaries land in
`build\bin\Release`.

For faster repeated builds, install [ccache](https://ccache.dev/) (CI uses it).

### Targeting a different GPU

`CMAKE_CUDA_ARCHITECTURES` takes the compute capability without the dot — e.g. `89` for an
RTX 4090, `86` for a 3080 Ti, or a list like `"86;89;120"` for a multi-card / portable build.
Find your card's value at ["CUDA GPUs"](https://developer.nvidia.com/cuda-gpus). For a build
that runs on any CUDA GPU (larger binary, some JIT at first run) use `-DGGML_NATIVE=OFF`.

### Overriding the CUDA version

With multiple toolkits installed, point CMake at a specific `nvcc`:

```bat
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:/path/to/cuda/bin/nvcc.exe"
```

## Useful CUDA build options

| Option                          | Default | Description |
|---------------------------------|---------|-------------|
| `GGML_CUDA_FORCE_MMQ`           | off     | Force custom quantized matmul kernels even without int8 tensor-core support. Lower VRAM, slower at large batch. |
| `GGML_CUDA_FORCE_CUBLAS`        | off     | Force FP16 cuBLAS instead of the custom kernels. Possible numerical overflow; higher memory use. |
| `GGML_CUDA_FA_ALL_QUANTS`       | off     | Compile all KV-cache quant combos for FlashAttention. Finer KV control, much longer compile. |

## Useful runtime environment variables

- `CUDA_VISIBLE_DEVICES="-0"` — hide a device (e.g. run on a non-primary GPU).
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — allow spilling to system RAM instead of OOM-crashing.
  On Windows this is also the NVIDIA Control Panel "System Memory Fallback" setting. Note that
  for this fork's VRAM-tight Gemma stack, spilling *craters* throughput — keep dedicated VRAM
  under ~15 GB instead (see the fixes doc).
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` / `..._16F` — force FP32 / FP16 cuBLAS compute type.

## Notes

- The GPU may still accelerate parts of the computation even with `-ngl 0`; use `--device none`
  to fully disable GPU use.
- If `nvcc` warns `Cannot find valid GPU for '-arch=native'`, set `CMAKE_CUDA_ARCHITECTURES`
  explicitly (as above) or do a non-native build.
