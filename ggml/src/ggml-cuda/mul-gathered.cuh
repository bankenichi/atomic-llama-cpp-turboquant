#pragma once

#include "common.cuh"

// moeq binary-factor scales: [GET_ROWS of an F16 table by expert ids] -> RESHAPE -> MUL computed by one kernel, which
// reads the table row of each (slot, token) directly instead of writing and re-reading the gathered F32 rows. The
// kernel does what the two nodes do (one exact F16-to-F32 conversion, one F32 multiply per element), so the result is
// bit-identical. It fires at the GET_ROWS when its RESHAPE's only consumer is the MUL and every node between the
// GET_ROWS and the MUL is a view or no-op (the graph builder, llama-factored.h, places each gather there); anything
// else is computed unfused, never deferred, so no node is ever skipped without its result being written.
// Returns the number of nodes after i that were computed with it (the MUL's distance from i), or 0.
// Counted for every GET_ROWS that feeds a MUL through a RESHAPE: fused or left unfused (wrong types, other nodes in
// between, fusion switched off). Per graph evaluation the counts are logged to stderr whenever they change, and the
// process totals at exit; ggml_backend_cuda_mul_gathered_stats / _reset (proc addresses of the CUDA registry) expose
// them to tests.
int ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool fusion_allowed);

void ggml_cuda_mul_gathered_graph_begin();
void ggml_cuda_mul_gathered_graph_end();

// totals since the last reset: fused, left unfused, graph evaluations that had at least one candidate
void ggml_backend_cuda_mul_gathered_stats(int64_t * fused, int64_t * unfused, int64_t * graphs);
void ggml_backend_cuda_mul_gathered_reset();
