#pragma once

#include "common.cuh"

// moeq binary-factor scales: [GET_ROWS of an F16 table by expert ids] -> RESHAPE -> MUL fused into one kernel, which
// reads the table row of each (slot, token) directly instead of writing and re-reading the gathered F32 rows.
// Returns true when cgraph->nodes[i .. i+2] are that pattern and the fused kernel ran (the caller skips 2 nodes).
bool ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
