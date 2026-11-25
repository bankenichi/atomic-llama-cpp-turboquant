#include "mul-gathered.cuh"

#include "ggml-impl.h"

// dst[i, k, t] = a[i, k or 0, t] * table[i, ids[k + U*t]]: one block per (slot, token) row
static __global__ void k_mul_gathered_rows(
        const float * __restrict__ a, const half * __restrict__ table, const int32_t * __restrict__ ids,
        float * __restrict__ dst, const int64_t n, const int64_t U,
        const int64_t a_s1, const int64_t a_s2, const int64_t table_s1, const int64_t dst_s1) {
    const int64_t row = blockIdx.x;
    const int64_t k   = row % U;
    const int64_t t   = row / U;
    const int32_t e   = ids[row];

    const float * arow = a + k*a_s1 + t*a_s2;
    const half  * trow = table + e*table_s1;
    float       * drow = dst + row*dst_s1;
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) {
        drow[c] = arow[c] * __half2float(trow[c]);
    }
}

bool ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * gr  = cgraph->nodes[i];
    const ggml_tensor * rs  = cgraph->nodes[i + 1];
    const ggml_tensor * mul = cgraph->nodes[i + 2];
    if (gr->op != GGML_OP_GET_ROWS || rs->op != GGML_OP_RESHAPE || mul->op != GGML_OP_MUL) {
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if ((cgraph->nodes[i + k]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    if (rs->src[0] != gr || !ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
        return false;
    }
    const ggml_tensor * table = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    // the scale table [n, E] (F16 rows) gathered by a flat list of expert ids [U*T]
    if (table->type != GGML_TYPE_F16 || table->nb[0] != sizeof(half) || table->ne[2] != 1 || table->ne[3] != 1 ||
            ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ids->ne[1] != 1 || ids->ne[2] != 1 ||
            gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) || !ggml_is_contiguous(rs)) {
        return false;
    }
    const ggml_tensor * a = mul->src[0] == rs ? mul->src[1] : mul->src[1] == rs ? mul->src[0] : nullptr;
    if (a == nullptr || a == rs) {
        return false;
    }
    const int64_t n = rs->ne[0], U = rs->ne[1], T = rs->ne[2];
    if (rs->ne[3] != 1 || U*T != ids->ne[0] || !ggml_are_same_shape(mul, rs) || mul->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(mul) || a->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || a->ne[0] != n ||
            (a->ne[1] != U && a->ne[1] != 1) || a->ne[2] != T || a->ne[3] != 1 ||
            a->nb[1] % sizeof(float) || a->nb[2] % sizeof(float)) {
        return false;
    }
    const int64_t a_s1 = a->ne[1] == 1 ? 0 : a->nb[1]/sizeof(float);
    const int64_t a_s2 = a->nb[2]/sizeof(float);

    cudaStream_t stream = ctx.stream();
    k_mul_gathered_rows<<<(unsigned) (U*T), 256, 0, stream>>>(
        (const float *) a->data, (const half *) table->data, (const int32_t *) ids->data, (float *) mul->data,
        n, U, a_s1, a_s2, table->nb[1]/sizeof(half), mul->nb[1]/sizeof(float) * 1 /* rows of n */ );
    CUDA_CHECK(cudaGetLastError());
    return true;
}
