# Pre-convergence baseline: 2026-08-21

Baseline code: `a637defb785e567cd4e595f516c195a22947d8a8`

The design and planning documentation commits do not change the compiled source,
so the build recorded from `fd715c856` is the same code baseline.

## Protected references

- Tag: `pre-upstream-parity-a637defb`
- Backup branch: `backup/pre-upstream-convergence-20260821`
- Integration branch: `integration/upstream-prism-convergence`

## Toolchain

- Windows SDK: 10.0.26100.0
- CMake: 4.3.1
- Generator: Visual Studio 18 2026
- Compiler toolset: v143
- MSBuild: 18.7.1
- CUDA toolkit: 12.8.93
- CUDA architecture selected by CMake: `120a-real`
- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0, 16,302 MiB
- OpenSSL found by the local build: 4.0.0

The installed Visual Studio shell is version 18, but CUDA compilation uses the
v143 toolset as required by the fork's build guidance.

## Build result

Command:

```powershell
cmake --build build --config Release --parallel 8
```

Result: pass. The complete configured target set built, including CUDA, the
libraries, tests, examples, `llama-cli`, and `llama-server`.

One pre-existing MSBuild warning reports a stale `ZERO_CHECK` project name clash
in the reused build directory. It did not fail the build.

## Test result

Command:

```powershell
ctest --test-dir build -C Release --output-on-failure --parallel 8
```

Result: 41 of 43 tests passed (95%). Two baseline failures are recorded so they
are not incorrectly attributed to the upstream merge:

1. `test-backend-ops` exits with Windows code `0xc0000409`. The last diagnostic
   is CUDA `GET_ROWS` dispatch rejecting `tq3_1s` in
   `ggml_cuda_get_rows_switch_src0_type`.
2. `test-quant-type-selection` reports stale Qwen 3.5 expected snapshots for
   NextN-layer tensors. It also skips the Nemotron metadata case after an HF API
   HTTP 401. Nine of eleven model schemas pass, one skips, and the two Qwen 3.5
   schemas fail their expected type selections.

Both failures are convergence ledger items. They are not accepted as final-state
failures merely because they existed before the merge.

## Locally available real-model fixtures

The gitignored `run-gemma.ps1` launcher references and successfully finds these
fixtures:

| Fixture | Approximate size |
|---|---:|
| Gemma 4 26B-A4B target Q4_K_M | 15.64 GiB |
| Gemma 4 26B-A4B assistant Q8_0 | 0.43 GiB |
| Gemma 4 26B-A4B mmproj BF16 | 1.11 GiB |
| Gemma 4 12B target Q4_K_M | 6.87 GiB |
| Gemma 4 12B mmproj F16 | 0.16 GiB |
| Qwen 3.6 35B-A3B target with NextN | 15.75 GiB |
| Qwen mmproj | 0.84 GiB |

The launcher records the owner-tested profiles for Gemma text/vision/MTP and
Qwen text/vision/NextN. These files provide real regression fixtures after the
modern upstream tree is buildable.

## Source targets

- Initial upstream merge target:
  `9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e`
- Prism legacy reference:
  `9ca265a57f85f2117942490f421f64a226dd9847`
- Prism modern reference:
  `27462403a0cd7b9ca8943281cbe02566daefe2b3`
