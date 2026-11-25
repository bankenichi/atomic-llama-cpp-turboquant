#include "mul-gathered.cuh"

#include "ggml-impl.h"

#include <unordered_set>

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

// the gathers whose GET_ROWS was skipped; their MUL computes the product from the table (cleared as they are used)
static thread_local std::unordered_set<const ggml_tensor *> g_deferred_gathers;

static bool gather_ok(const ggml_tensor * gr) {
    const ggml_tensor * table = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    return gr->op == GGML_OP_GET_ROWS && table && ids &&
        table->type == GGML_TYPE_F16 && table->nb[0] == sizeof(half) && table->ne[2] == 1 && table->ne[3] == 1 &&
        ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ids->ne[1] == 1 && ids->ne[2] == 1 &&
        gr->type == GGML_TYPE_F32 && ggml_is_contiguous(gr);
}

// mul = a * reshape(gr) with the shapes the fused kernel handles; returns the operand a, or nullptr
static const ggml_tensor * mul_operand(const ggml_tensor * mul, const ggml_tensor * rs, const ggml_tensor * gr) {
    if (mul->op != GGML_OP_MUL || rs->op != GGML_OP_RESHAPE || rs->src[0] != gr || !ggml_is_contiguous(rs)) {
        return nullptr;
    }
    const ggml_tensor * a = mul->src[0] == rs ? mul->src[1] : mul->src[1] == rs ? mul->src[0] : nullptr;
    if (a == nullptr || a == rs) {
        return nullptr;
    }
    const int64_t n = rs->ne[0], U = rs->ne[1], T = rs->ne[2];
    if (rs->ne[3] != 1 || U*T != gr->src[1]->ne[0] || !ggml_are_same_shape(mul, rs) || mul->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(mul) || a->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || a->ne[0] != n ||
            (a->ne[1] != U && a->ne[1] != 1) || a->ne[2] != T || a->ne[3] != 1 ||
            a->nb[1] % sizeof(float) || a->nb[2] % sizeof(float)) {
        return nullptr;
    }
    return a;
}

static void launch(ggml_backend_cuda_context & ctx, const ggml_tensor * gr, const ggml_tensor * rs,
                   const ggml_tensor * a, ggml_tensor * mul) {
    const int64_t n = rs->ne[0], U = rs->ne[1], T = rs->ne[2];
    const int64_t a_s1 = a->ne[1] == 1 ? 0 : a->nb[1]/sizeof(float);
    const int64_t a_s2 = a->nb[2]/sizeof(float);
    k_mul_gathered_rows<<<(unsigned) (U*T), 256, 0, ctx.stream()>>>(
        (const float *) a->data, (const half *) gr->src[0]->data, (const int32_t *) gr->src[1]->data, (float *) mul->data,
        n, U, a_s1, a_s2, gr->src[0]->nb[1]/sizeof(half), mul->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

int ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];

    // the MUL of a gather that was skipped: compute the product from the table
    if (node->op == GGML_OP_MUL && !g_deferred_gathers.empty()) {
        for (int s = 0; s < 2; ++s) {
            const ggml_tensor * rs = node->src[s];
            if (rs && rs->op == GGML_OP_RESHAPE && rs->src[0] && g_deferred_gathers.count(rs->src[0])) {
                const ggml_tensor * gr = rs->src[0];
                const ggml_tensor * a  = mul_operand(node, rs, gr);
                GGML_ASSERT(a != nullptr && "a deferred gather's MUL changed shape");
                launch(ctx, gr, rs, a, node);
                g_deferred_gathers.erase(gr);
                return -1;
            }
        }
        return 0;
    }

    if (node->op != GGML_OP_GET_ROWS || !gather_ok(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    // its only consumer is a RESHAPE, whose only consumer is a MUL, within the next 64 nodes
    int rs_i = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (rs_i < 0) {
            if (t->op == GGML_OP_RESHAPE && t->src[0] == node) {
                if (!ggml_node_has_n_uses(cgraph, j, 1)) {
                    return 0;
                }
                rs_i = j;
            }
            continue;
        }
        const ggml_tensor * rs = cgraph->nodes[rs_i];
        if (t->src[0] == rs || t->src[1] == rs) {
            if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || mul_operand(t, rs, node) == nullptr) {
                return 0;
            }
            if (rs_i == i + 1 && j == i + 2) {
                // adjacent: compute here and skip the RESHAPE and the MUL
                launch(ctx, node, rs, mul_operand(t, rs, node), cgraph->nodes[j]);
                return 2;
            }
            // the MUL comes later (its other operand may not exist yet): skip this gather, the MUL computes it
            g_deferred_gathers.insert(node);
            return -1;
        }
    }
    return 0;
}
