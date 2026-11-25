#include "mul-gathered.cuh"

#include "ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <utility>

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

// ---- counters ------------------------------------------------------------------------------------------------------

static std::atomic<int64_t> g_fused{0};
static std::atomic<int64_t> g_unfused{0};
static std::atomic<int64_t> g_graphs{0};
static std::atomic<int64_t> g_total_fused{0};      // process totals (not reset by tests)
static std::atomic<int64_t> g_total_unfused{0};
static std::atomic<int64_t> g_total_graphs{0};
static thread_local int64_t t_graph_fused   = 0;
static thread_local int64_t t_graph_unfused = 0;

static void print_totals() {
    const int64_t f = g_total_fused.load(), u = g_total_unfused.load(), g = g_total_graphs.load();
    if (f + u > 0) {
        std::fprintf(stderr, "ggml_cuda: S1_32 scale gathers over %lld graph evaluations: %lld fused, %lld left unfused\n",
                (long long) g, (long long) f, (long long) u);
    }
}

void ggml_cuda_mul_gathered_graph_begin() {
    t_graph_fused   = 0;
    t_graph_unfused = 0;
}

void ggml_cuda_mul_gathered_graph_end() {
    if (t_graph_fused + t_graph_unfused == 0) {
        return;   // no candidate, or a CUDA graph replay (nodes not evaluated)
    }
    g_graphs++;
    g_total_graphs++;
    // one line per distinct (fused, unfused) pattern, so every job shows what its graphs did without one line per token
    static std::mutex mtx;
    static std::set<std::pair<int64_t, int64_t>> seen;
    static bool registered = false;
    std::lock_guard<std::mutex> lock(mtx);
    if (!registered) {
        std::atexit(print_totals);
        registered = true;
    }
    if (seen.size() < 32 && seen.insert({t_graph_fused, t_graph_unfused}).second) {
        std::fprintf(stderr, "ggml_cuda: S1_32 scale gathers in a graph: %lld fused, %lld left unfused%s\n",
                (long long) t_graph_fused, (long long) t_graph_unfused,
                t_graph_unfused > 0 ? " (fusion off, or a shape it does not cover)" : "");
    }
}

void ggml_backend_cuda_mul_gathered_stats(int64_t * fused, int64_t * unfused, int64_t * graphs) {
    if (fused)   { *fused   = g_fused.load(); }
    if (unfused) { *unfused = g_unfused.load(); }
    if (graphs)  { *graphs  = g_graphs.load(); }
}

void ggml_backend_cuda_mul_gathered_reset() {
    g_fused   = 0;
    g_unfused = 0;
    g_graphs  = 0;
}

static void count(bool fused) {
    if (fused) {
        g_fused++;
        g_total_fused++;
        t_graph_fused++;
    } else {
        g_unfused++;
        g_total_unfused++;
        t_graph_unfused++;
    }
}

// ---- matching ------------------------------------------------------------------------------------------------------

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

static bool is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
        t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
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

int ggml_cuda_try_fuse_mul_gathered(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool fusion_allowed) {
    // GGML_CUDA_MUL_GATHERED=0 turns this fusion off alone (GGML_CUDA_DISABLE_FUSION turns off every fusion)
    static const bool off = [] {
        const char * s = getenv("GGML_CUDA_MUL_GATHERED");
        const bool v = s != nullptr && atoi(s) == 0;
        if (v) {
            std::fprintf(stderr, "ggml_cuda: GGML_CUDA_MUL_GATHERED=0: the S1_32 scale gathers are not fused\n");
        }
        return v;
    }();

    ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_GET_ROWS || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 0;
    }
    // a candidate (counted): a GET_ROWS whose result is reshaped (within the next 64 nodes); it fuses when the RESHAPE's
    // only consumer is a matching MUL
    int rs_i = -1, use_i = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (rs_i < 0) {
            if (t->op == GGML_OP_RESHAPE && t->src[0] == node) {
                rs_i = j;
            }
            continue;
        }
        if (t->src[0] == cgraph->nodes[rs_i] || t->src[1] == cgraph->nodes[rs_i]) {
            use_i = j;
            break;
        }
    }
    if (rs_i < 0) {
        return 0;   // not a scale gather (token embeddings, output rows)
    }
    const int mul_i = use_i >= 0 && cgraph->nodes[use_i]->op == GGML_OP_MUL ? use_i : -1;
    if (mul_i < 0) {
        count(false);
        return 0;
    }
    const ggml_tensor * rs  = cgraph->nodes[rs_i];
    ggml_tensor       * mul = cgraph->nodes[mul_i];

    bool ok = fusion_allowed && !off && gather_ok(node) && ggml_node_has_n_uses(cgraph, i, 1) &&
        // the RESHAPE is a view of the gather: ggml_node_has_n_uses refuses every view, so check its count directly
        ggml_node_get_use_count(cgraph, rs_i) == 1 && (rs->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
        (mul->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && mul_operand(mul, rs, node) != nullptr;
    // computed here, at the gather: every node between must be a view or no-op, so the MUL's other operand already
    // exists and nothing in between needs the gathered rows
    for (int j = i + 1; ok && j < mul_i; ++j) {
        ok = is_view_or_noop(cgraph->nodes[j]);
    }
    count(ok);
    if (!ok) {
        return 0;
    }
    launch(ctx, node, rs, mul_operand(mul, rs, node), mul);
    return mul_i - i;
}
