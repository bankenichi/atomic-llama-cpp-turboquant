#pragma once

#include "common.cuh"

// moeq binary-factor scales: [GET_ROWS of an F16 table by expert ids] -> RESHAPE -> MUL computed by one kernel, which
// reads the table row of each (slot, token) directly instead of writing and re-reading the gathered F32 rows (the
// result is bit-identical). The graph puts the gathers for s_post right before their MUL, but those for s_pre early
// in the layer: a qualifying GET_ROWS whose only consumer (through a RESHAPE) is a later MUL is skipped and the MUL
// computes the product.
// Returns 2 when nodes i .. i+2 were computed (GET_ROWS, RESHAPE, MUL adjacent), -1 when node i is handled and nothing
// else is skipped (a deferred GET_ROWS, or the MUL of one), 0 otherwise.
int ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
