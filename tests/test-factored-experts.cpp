// moeq binary-factor experts on the CPU: the op sequence llm_build_factored_mm_id builds (the one
// llm_graph_context::build_factored_mm_id puts in the model graph) against
//  (1) a double-precision reference y = s_post[e] * (sign(U_e) (sign(V_e)^T (s_pre[e] * x))) from the same signs and
//      fp16 scales, and
//  (2) an ordinary F32 MUL_MAT_ID over the dequantized dense matrices W_e = diag(s_post) sign(U) sign(V)^T diag(s_pre).
//
// Run without arguments it checks the production path (activations quantized to Q8_0 before each sign product);
// run with "f32" it sets GGML_S1_32_DOT=f32 before the CPU backend initializes and checks the float reference path,
// whose results must equal the references to float rounding.
#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-factored.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct bank_case {
    const char * name;
    int64_t n_in, rank, n_out, n_expert, n_used, n_tokens;
    bool per_slot;     // input [n_in, n_used, n_tokens] (a down bank) instead of [n_in, 1, n_tokens]
    bool view_ids;     // ids as a non-contiguous view, as the router's top-k is
    bool outlier;      // one massive activation
    bool dense;        // also run the dense F32 MUL_MAT_ID (costs E*n_out*r*n_in to build)
};

static double nmse(const float * got, const std::vector<double> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        num += d*d;
        den += ref[i]*ref[i];
    }
    return den > 0.0 ? num/den : num;
}

static int run_case(const bank_case & c, bool f32, std::mt19937 & rng) {
    const int64_t E = c.n_expert, n_in = c.n_in, r = c.rank, n_out = c.n_out, U = c.n_used, T = c.n_tokens;
    const int64_t n_slots_in = c.per_slot ? U : 1;

    ggml_init_params params = { /*.mem_size =*/ (size_t) 512*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(params);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> coin(0, 1);

    // signs: V^T rows [E][r][n_in], U rows [E][n_out][r]
    std::vector<float> vt((size_t) E*r*n_in), uu((size_t) E*n_out*r);
    for (float & s : vt) { s = coin(rng) ? 1.0f : -1.0f; }
    for (float & s : uu) { s = coin(rng) ? 1.0f : -1.0f; }
    // scales: random, either sign, rounded to fp16 as moeq stores them
    std::vector<float> sp((size_t) E*n_in), so((size_t) E*n_out);
    for (float & s : sp) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.05f*nd(rng))); }
    for (float & s : so) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.05f*nd(rng))); }

    ggml_tensor * v   = ggml_new_tensor_3d(ctx, GGML_TYPE_S1_32, n_in, r, E);
    ggml_tensor * u   = ggml_new_tensor_3d(ctx, GGML_TYPE_S1_32, r, n_out, E);
    ggml_tensor * tsp = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_in, 1, E);
    ggml_tensor * tso = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_out, 1, E);
    ggml_quantize_chunk(GGML_TYPE_S1_32, vt.data(), v->data, 0, E*r, n_in, nullptr);
    ggml_quantize_chunk(GGML_TYPE_S1_32, uu.data(), u->data, 0, E*n_out, r, nullptr);
    ggml_fp32_to_fp16_row(sp.data(), (ggml_fp16_t *) tsp->data, E*n_in);
    ggml_fp32_to_fp16_row(so.data(), (ggml_fp16_t *) tso->data, E*n_out);

    // input
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_slots_in, T);
    float * xd = (float *) x->data;
    for (int64_t i = 0; i < n_in*n_slots_in*T; ++i) { xd[i] = nd(rng); }
    if (c.outlier) { xd[3] = 3000.0f; }

    // ids: distinct experts per token, optionally as a view of a wider tensor
    ggml_tensor * ids_store = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.view_ids ? U + 2 : U, T);
    ggml_tensor * ids = c.view_ids ? ggml_view_2d(ctx, ids_store, U, T, ids_store->nb[1], 0) : ids_store;
    std::vector<int> sel((size_t) U*T);
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int> perm(E);
        for (int e = 0; e < E; ++e) { perm[e] = e; }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t k = 0; k < ids_store->ne[0]; ++k) {
            ((int32_t *) ids_store->data)[t*ids_store->ne[0] + k] = perm[k % E];
        }
        for (int64_t k = 0; k < U; ++k) { sel[t*U + k] = perm[k]; }
    }

    // the graph the model runs
    const llm_factored_mm_id_result res = llm_build_factored_mm_id(ctx, v, u, tsp, tso, x, ids);

    // the same experts as dense F32 matrices through an ordinary MUL_MAT_ID
    ggml_tensor * y_dense = nullptr;
    if (c.dense) {
        ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_out, E);
        std::vector<double> acc(n_in);
        for (int64_t e = 0; e < E; ++e) {
            for (int64_t o = 0; o < n_out; ++o) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t j = 0; j < r; ++j) {
                    const double s = uu[(e*n_out + o)*r + j];
                    const float * row = vt.data() + (e*r + j)*n_in;
                    for (int64_t i = 0; i < n_in; ++i) { acc[i] += s*row[i]; }
                }
                for (int64_t i = 0; i < n_in; ++i) {
                    ((float *) wd->data)[(e*n_out + o)*n_in + i] = (float) (so[e*n_out + o] * acc[i] * sp[e*n_in + i]);
                }
            }
        }
        y_dense = ggml_mul_mat_id(ctx, wd, x, ids);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res.y);
    if (y_dense) {
        ggml_build_forward_expand(gf, y_dense);
    }
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // double reference, including the intermediate z
    std::vector<double> ref_y((size_t) n_out*U*T), ref_z((size_t) r*U*T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < U; ++k) {
            const int e = sel[t*U + k];
            const float * xin = xd + (t*n_slots_in + (c.per_slot ? k : 0))*n_in;
            std::vector<double> xs(n_in), z(r);
            for (int64_t i = 0; i < n_in; ++i) { xs[i] = (double) sp[e*n_in + i] * xin[i]; }
            for (int64_t j = 0; j < r; ++j) {
                double acc = 0.0;
                for (int64_t i = 0; i < n_in; ++i) { acc += vt[(e*r + j)*n_in + i] * xs[i]; }
                z[j] = acc;
                ref_z[(t*U + k)*r + j] = acc;
            }
            for (int64_t o = 0; o < n_out; ++o) {
                double acc = 0.0;
                for (int64_t j = 0; j < r; ++j) { acc += uu[(e*n_out + o)*r + j] * z[j]; }
                ref_y[(t*U + k)*n_out + o] = so[e*n_out + o] * acc;
            }
        }
    }

    const double e_y     = nmse((const float *) res.y->data, ref_y);
    const double e_z     = nmse((const float *) res.z->data, ref_z);
    double e_dense = 0.0, e_vs_dense = 0.0;
    if (y_dense) {
        e_dense = nmse((const float *) y_dense->data, ref_y);
        // the dense path computes with F32 weights; compare the factored output with it too
        std::vector<double> dense_d((size_t) n_out*U*T);
        for (size_t i = 0; i < dense_d.size(); ++i) { dense_d[i] = ((const float *) y_dense->data)[i]; }
        e_vs_dense = nmse((const float *) res.y->data, dense_d);
    }

    // float path: float rounding only. Q8_0 path: two activation quantizations per bank.
    const double tol = f32 ? 1e-10 : 5e-4;
    const bool ok = e_y <= tol && e_z <= tol && e_vs_dense <= tol && e_dense <= 1e-10;
    std::printf("%-34s %s  nmse(y, ref) %.3e  nmse(z, ref) %.3e  nmse(y, dense mul_mat_id) %.3e  nmse(dense, ref) %.3e  [%s]\n",
                c.name, f32 ? "f32" : "q8 ", e_y, e_z, e_vs_dense, e_dense, ok ? "ok" : "FAILED");

    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const bool f32 = argc > 1 && std::strcmp(argv[1], "f32") == 0;
    if (f32) {
        // read once by the CPU backend when it first initializes, which has not happened yet
#ifdef _WIN32
        _putenv_s("GGML_S1_32_DOT", "f32");
#else
        setenv("GGML_S1_32_DOT", "f32", 1);
#endif
    }

    const bank_case cases[] = {
        { "small, broadcast input",            128,  96,  160,  8, 3, 5, false, false, false, true  },
        { "small, per-slot input, view ids",   128,  96,  160,  8, 3, 5, true,  true,  false, true  },
        { "Granite-1B in-bank (r 480), 4 exp", 1024, 480, 1024,  4, 4, 6, false, true,  false, true  },
        { "Granite-1B out-bank (r 320), 4 exp", 512, 320, 1024,  4, 4, 6, true,  true,  false, true  },
        { "Granite-1B in-bank, 32 exp",       1024, 480, 1024, 32, 8, 4, false, true,  false, false },
        { "Granite-1B in-bank, massive act.", 1024, 480, 1024, 32, 8, 3, false, false, true,  false },
        { "Granite-1B cap (r 1024 / 512)",    1024, 1024, 1024, 32, 8, 3, false, true, false, false },
    };

    std::mt19937 rng(1234);
    int failed = 0;
    for (const bank_case & c : cases) {
        failed += run_case(c, f32, rng);
    }
    std::printf("test-factored-experts (%s): %s\n", f32 ? "f32" : "q8", failed == 0 ? "all cases passed" : "FAILED");
    return failed == 0 ? 0 : 1;
}
