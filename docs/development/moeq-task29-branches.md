# Branches from moeq Task 29 (2026-11-24)

Written by an AI agent (Claude Opus 5.5) for the owner's moeq project; merged on the owner's decision.

## Merged into `main`

Each as its own merge commit (`--no-ff`), so each can be reverted alone:

| Branch | What | Merge commit |
|---|---|---|
| `feat/cuda-mmq-iq1m` | CUDA MMQ (batched quantized matmul) path for IQ1_M, with `test-backend-ops` cases | `dfa0d86a4` |
| `fix/sched-hash-set-leafs` | `ggml_backend_sched_new` sizes its hash set for nodes and leafs (Granite-3.1-1B with a q4_0/q8_0 KV cache and attention rotation aborted), with a `test-backend-sched` case | `3187057c3` |
| `feat/imatrix-collector` | `llama-imatrix`: the MUL_MAT_ID collector accumulates over `-t` threads, byte-identical output | `d74026b27` |

## Kept unmerged on purpose

`feat/cuda-mmq-iq1m-fd715c8` and `feat/imatrix-collector-fd715c8` carry the same IQ1_M kernel and imatrix
collector (plus a cherry-pick of upstream #31910, the IQ2 quantization speedup) on `fd715c856`, the source of the
frozen llama.cpp copy that moeq scores its baselines with (`E:\moeq\tools\llama-cpp-962495a4c`, built from
`962495a4c`). They predate the 2026-11 upstream convergence: do not merge them into `main`; keep them so that build
can be reproduced.

## Upstream

No pull request to ggml-org/llama.cpp (owner's decision: upstream does not take agent-written changes).
