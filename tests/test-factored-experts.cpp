// moeq binary-factor experts: the op sequence llm_build_factored_mm_id builds (the one
// llm_graph_context::build_factored_mm_id puts in the model graph) against
//  (1) a double-precision reference y = s_post[e] * (sign(U_e) (sign(V_e)^T (s_pre[e] * x))) from the same signs and
//      fp16 scales, and
//  (2) an ordinary F32 MUL_MAT_ID over the dequantized dense matrices W_e = diag(s_post) sign(U) sign(V)^T diag(s_pre).
//
// Run without arguments it checks the CPU's production path (activations quantized to Q8_0 before each sign
// product); run with "f32" it sets GGML_S1_32_DOT=f32 before the CPU backend initializes and checks the float
// reference path, whose results must equal the references to float rounding. Either way it then runs the same cases
// on every GPU device the build has (activations quantized to q8_1 for small batches; the batched path for larger
// ones) and compares those with the double reference and with the CPU's result.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-factored.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct bank_case {
    const char * name;
    int64_t n_in, rank, n_out, n_expert, n_used, n_tokens;
    bool per_slot;     // input [n_in, n_used, n_tokens] (a down bank) instead of [n_in, 1, n_tokens]
    bool view_ids;     // ids as a non-contiguous view, as the router's top-k is
    bool outlier;      // one massive activation
    bool dense;        // also run the dense F32 MUL_MAT_ID (costs E*n_out*r*n_in to build)
    float scale_mul = 1.0f; // multiplies the scales: real models' scaled activations reach 1e3-1e4, beyond F16 sums
};

// one case's random data and its double-precision reference
struct bank_data {
    std::vector<float>   vt, uu;         // signs: V^T rows [E][r][n_in], U rows [E][n_out][r]
    std::vector<float>   sp, so;         // fp16-rounded scales [E][n_in], [E][n_out]
    std::vector<float>   x;              // input [T][n_slots_in][n_in]
    std::vector<int32_t> ids;            // [T][ids_ne0]
    int64_t              ids_ne0 = 0;
    std::vector<int>     sel;            // the experts actually used, [T][U]
    std::vector<double>  ref_y, ref_z;   // [T][U][n_out], [T][U][r]
};

typedef void (*s1_32_set_mode_t)(int);
typedef int  (*s1_32_get_mode_t)(void);

static s1_32_set_mode_t g_set_mode = nullptr;
static s1_32_get_mode_t g_get_mode = nullptr;

static double nmse(const float * got, const std::vector<double> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        num += d*d;
        den += ref[i]*ref[i];
    }
    return den > 0.0 ? num/den : num;
}

static bank_data make_data(const bank_case & c, std::mt19937 & rng) {
    const int64_t E = c.n_expert, n_in = c.n_in, r = c.rank, n_out = c.n_out, U = c.n_used, T = c.n_tokens;
    const int64_t n_slots_in = c.per_slot ? U : 1;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> coin(0, 1);

    bank_data d;
    d.vt.resize((size_t) E*r*n_in);
    d.uu.resize((size_t) E*n_out*r);
    for (float & s : d.vt) { s = coin(rng) ? 1.0f : -1.0f; }
    for (float & s : d.uu) { s = coin(rng) ? 1.0f : -1.0f; }
    // scales: random, either sign, rounded to fp16 as moeq stores them
    d.sp.resize((size_t) E*n_in);
    d.so.resize((size_t) E*n_out);
    for (float & s : d.sp) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.05f*c.scale_mul*nd(rng))); }
    for (float & s : d.so) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.05f*nd(rng))); }

    d.x.resize((size_t) n_in*n_slots_in*T);
    for (float & v : d.x) { v = nd(rng); }
    if (c.outlier) { d.x[3] = 3000.0f; }

    // ids: distinct experts per token, optionally as a view of a wider tensor
    d.ids_ne0 = c.view_ids ? U + 2 : U;
    d.ids.resize((size_t) d.ids_ne0*T);
    d.sel.resize((size_t) U*T);
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int> perm(E);
        for (int e = 0; e < E; ++e) { perm[e] = e; }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t k = 0; k < d.ids_ne0; ++k) { d.ids[t*d.ids_ne0 + k] = perm[k % E]; }
        for (int64_t k = 0; k < U; ++k) { d.sel[t*U + k] = perm[k]; }
    }

    // double reference, including the intermediate z
    d.ref_y.resize((size_t) n_out*U*T);
    d.ref_z.resize((size_t) r*U*T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < U; ++k) {
            const int e = d.sel[t*U + k];
            const float * xin = d.x.data() + (t*n_slots_in + (c.per_slot ? k : 0))*n_in;
            std::vector<double> xs(n_in), z(r);
            for (int64_t i = 0; i < n_in; ++i) { xs[i] = (double) d.sp[e*n_in + i] * xin[i]; }
            for (int64_t j = 0; j < r; ++j) {
                double acc = 0.0;
                const float * row = d.vt.data() + (e*r + j)*n_in;
                for (int64_t i = 0; i < n_in; ++i) { acc += row[i] * xs[i]; }
                z[j] = acc;
                d.ref_z[(t*U + k)*r + j] = acc;
            }
            for (int64_t o = 0; o < n_out; ++o) {
                double acc = 0.0;
                const float * row = d.uu.data() + (e*n_out + o)*r;
                for (int64_t j = 0; j < r; ++j) { acc += row[j] * z[j]; }
                d.ref_y[(t*U + k)*n_out + o] = d.so[e*n_out + o] * acc;
            }
        }
    }
    return d;
}

// the graph's inputs, in a context that may or may not hold their data
struct bank_tensors {
    ggml_tensor * v, * u, * sp, * so, * x, * ids_store, * ids;
};

static bank_tensors new_tensors(ggml_context * ctx, const bank_case & c, const bank_data & d) {
    const int64_t E = c.n_expert, n_in = c.n_in, r = c.rank, n_out = c.n_out, U = c.n_used, T = c.n_tokens;
    bank_tensors t;
    t.v   = ggml_new_tensor_3d(ctx, GGML_TYPE_S1_32, n_in, r, E);
    t.u   = ggml_new_tensor_3d(ctx, GGML_TYPE_S1_32, r, n_out, E);
    t.sp  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_in, 1, E);
    t.so  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_out, 1, E);
    t.x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, c.per_slot ? U : 1, T);
    t.ids_store = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, d.ids_ne0, T);
    t.ids = c.view_ids ? ggml_view_2d(ctx, t.ids_store, U, T, t.ids_store->nb[1], 0) : t.ids_store;
    // the model names its sign tensors *_exps_v / *_exps_u; the CPU's per-step float switch keys on those names
    ggml_set_name(t.v, "test_exps_v");
    ggml_set_name(t.u, "test_exps_u");
    return t;
}

// the inputs' bytes as the tensors store them
struct bank_bytes {
    std::vector<uint8_t>     v, u;
    std::vector<ggml_fp16_t> sp, so;
};

static bank_bytes encode(const bank_case & c, const bank_data & d) {
    const int64_t E = c.n_expert, n_in = c.n_in, r = c.rank, n_out = c.n_out;
    bank_bytes b;
    b.v.resize(ggml_row_size(GGML_TYPE_S1_32, n_in)*r*E);
    b.u.resize(ggml_row_size(GGML_TYPE_S1_32, r)*n_out*E);
    ggml_quantize_chunk(GGML_TYPE_S1_32, d.vt.data(), b.v.data(), 0, E*r, n_in, nullptr);
    ggml_quantize_chunk(GGML_TYPE_S1_32, d.uu.data(), b.u.data(), 0, E*n_out, r, nullptr);
    b.sp.resize((size_t) E*n_in);
    b.so.resize((size_t) E*n_out);
    ggml_fp32_to_fp16_row(d.sp.data(), b.sp.data(), E*n_in);
    ggml_fp32_to_fp16_row(d.so.data(), b.so.data(), E*n_out);
    return b;
}

// the CPU path, computed in a context that holds the data; returns y for the GPU comparison
static int run_case(const bank_case & c, bool f32, std::mt19937 & rng, std::vector<float> * y_out = nullptr,
                    bank_data * data_out = nullptr) {
    const int64_t E = c.n_expert, n_in = c.n_in, r = c.rank, n_out = c.n_out, U = c.n_used, T = c.n_tokens;

    const bank_data d = make_data(c, rng);
    const bank_bytes b = encode(c, d);

    ggml_init_params params = { /*.mem_size =*/ (size_t) 512*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(params);
    const bank_tensors t = new_tensors(ctx, c, d);
    memcpy(t.v->data, b.v.data(), b.v.size());
    memcpy(t.u->data, b.u.data(), b.u.size());
    memcpy(t.sp->data, b.sp.data(), ggml_nbytes(t.sp));
    memcpy(t.so->data, b.so.data(), ggml_nbytes(t.so));
    memcpy(t.x->data, d.x.data(), ggml_nbytes(t.x));
    memcpy(t.ids_store->data, d.ids.data(), ggml_nbytes(t.ids_store));

    // the graph the model runs
    const llm_factored_mm_id_result res = llm_build_factored_mm_id(ctx, t.v, t.u, t.sp, t.so, t.x, t.ids);

    // the same experts as dense F32 matrices through an ordinary MUL_MAT_ID
    ggml_tensor * y_dense = nullptr;
    if (c.dense) {
        ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, n_out, E);
        std::vector<double> acc(n_in);
        for (int64_t e = 0; e < E; ++e) {
            for (int64_t o = 0; o < n_out; ++o) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t j = 0; j < r; ++j) {
                    const double s = d.uu[(e*n_out + o)*r + j];
                    const float * row = d.vt.data() + (e*r + j)*n_in;
                    for (int64_t i = 0; i < n_in; ++i) { acc[i] += s*row[i]; }
                }
                for (int64_t i = 0; i < n_in; ++i) {
                    ((float *) wd->data)[(e*n_out + o)*n_in + i] = (float) (d.so[e*n_out + o] * acc[i] * d.sp[e*n_in + i]);
                }
            }
        }
        y_dense = ggml_mul_mat_id(ctx, wd, t.x, t.ids);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res.y);
    if (y_dense) {
        ggml_build_forward_expand(gf, y_dense);
    }
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const double e_y = nmse((const float *) res.y->data, d.ref_y);
    const double e_z = nmse((const float *) res.z->data, d.ref_z);
    double e_dense = 0.0, e_vs_dense = 0.0;
    if (y_dense) {
        e_dense = nmse((const float *) y_dense->data, d.ref_y);
        // the dense path computes with F32 weights; compare the factored output with it too
        std::vector<double> dense_d((size_t) n_out*U*T);
        for (size_t i = 0; i < dense_d.size(); ++i) { dense_d[i] = ((const float *) y_dense->data)[i]; }
        e_vs_dense = nmse((const float *) res.y->data, dense_d);
    }

    // float path: float rounding only. Q8_0 path: two activation quantizations per bank.
    const double tol = f32 ? 1e-10 : 5e-4;
    const bool ok = e_y <= tol && e_z <= tol && e_vs_dense <= tol && e_dense <= 1e-10;
    std::printf("%-40s CPU %s  nmse(y, ref) %.3e  nmse(z, ref) %.3e  nmse(y, dense mul_mat_id) %.3e  nmse(dense, ref) %.3e  [%s]\n",
                c.name, f32 ? "f32" : "q8 ", e_y, e_z, e_vs_dense, e_dense, ok ? "ok" : "FAILED");

    if (y_out) {
        y_out->assign((const float *) res.y->data, (const float *) res.y->data + n_out*U*T);
    }
    if (data_out) {
        *data_out = d;
    }
    ggml_free(ctx);
    return ok ? 0 : 1;
}

// the same op sequence on a backend device; returns 0 on success or when the device does not support the ops
static int run_case_device(const bank_case & c, ggml_backend_dev_t dev, const bank_data & d, const std::vector<float> & y_cpu) {
    const int64_t r = c.rank, n_out = c.n_out, U = c.n_used, T = c.n_tokens;

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::printf("%-40s %s: cannot initialize\n", c.name, ggml_backend_dev_name(dev));
        return 1;
    }

    ggml_init_params params = { /*.mem_size =*/ ggml_tensor_overhead()*64 + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(params);
    const bank_tensors t = new_tensors(ctx, c, d);
    const llm_factored_mm_id_result res = llm_build_factored_mm_id(ctx, t.v, t.u, t.sp, t.so, t.x, t.ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res.y);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_is_empty(node) && node->op != GGML_OP_NONE && node->op != GGML_OP_RESHAPE && node->op != GGML_OP_VIEW &&
                node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE && !ggml_backend_supports_op(backend, node)) {
            std::printf("%-40s %s: %s not supported, skipped\n", c.name, ggml_backend_dev_name(dev), ggml_op_desc(node));
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 0;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const bank_bytes b = encode(c, d);
    ggml_backend_tensor_set(t.v, b.v.data(), 0, b.v.size());
    ggml_backend_tensor_set(t.u, b.u.data(), 0, b.u.size());
    ggml_backend_tensor_set(t.sp, b.sp.data(), 0, ggml_nbytes(t.sp));
    ggml_backend_tensor_set(t.so, b.so.data(), 0, ggml_nbytes(t.so));
    ggml_backend_tensor_set(t.x, d.x.data(), 0, ggml_nbytes(t.x));
    ggml_backend_tensor_set(t.ids_store, d.ids.data(), 0, ggml_nbytes(t.ids_store));

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    int failed = 0;
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("%-40s %s: graph compute failed (%d)\n", c.name, ggml_backend_dev_name(dev), (int) st);
        failed = 1;
    } else {
        std::vector<float> y((size_t) n_out*U*T), z((size_t) r*U*T);
        ggml_backend_tensor_get(res.y, y.data(), 0, y.size()*sizeof(float));
        ggml_backend_tensor_get(res.z, z.data(), 0, z.size()*sizeof(float));
        const double e_y = nmse(y.data(), d.ref_y);
        const double e_z = nmse(z.data(), d.ref_z);
        std::vector<double> cpu_d(y_cpu.begin(), y_cpu.end());
        const double e_cpu = nmse(y.data(), cpu_d);
        // q8_1 activations: two quantizations per bank, as on the CPU's Q8_0 path
        const double tol = 5e-4;
        const bool ok = e_y <= tol && e_z <= tol && std::isfinite(e_y) && std::isfinite(e_z);
        std::printf("%-40s %-5s nmse(y, ref) %.3e  nmse(z, ref) %.3e  nmse(y, CPU y) %.3e  [%s]\n",
                    c.name, ggml_backend_dev_name(dev), e_y, e_z, e_cpu, ok ? "ok" : "FAILED");
        failed = ok ? 0 : 1;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return failed;
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

    // the float switch is fork-private: reached through the CPU backend's proc address, not a public header
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    g_set_mode = cpu_reg ? (s1_32_set_mode_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_s1_32_dot_mode") : nullptr;
    g_get_mode = cpu_reg ? (s1_32_get_mode_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_get_s1_32_dot_mode") : nullptr;
    if (!g_set_mode || !g_get_mode) {
        std::printf("the CPU backend does not offer ggml_backend_cpu_{set,get}_s1_32_dot_mode\n");
        return 1;
    }

    const bank_case cases[] = {
        { "small, broadcast input",              128,  96,  160,  8, 3,  5, false, false, false, true  },
        { "small, per-slot input, view ids",     128,  96,  160,  8, 3,  5, true,  true,  false, true  },
        { "Granite-1B in-bank (r 480), 4 exp",  1024, 480, 1024,  4, 4,  6, false, true,  false, true  },
        { "Granite-1B out-bank (r 320), 4 exp",  512, 320, 1024,  4, 4,  6, true,  true,  false, true  },
        { "Granite-1B in-bank, 32 exp",         1024, 480, 1024, 32, 8,  4, false, true,  false, false },
        { "Granite-1B in-bank, massive act.",   1024, 480, 1024, 32, 8,  3, false, false, true,  false },
        { "Granite-1B cap (r 1024 / 512)",      1024, 1024, 1024, 32, 8, 3, false, true,  false, false },
        { "Granite-1B out-bank, 32 exp, 1 tok",  512, 320, 1024, 32, 8,  1, true,  true,  false, false },
        { "Granite-1B in-bank, 32 exp, 8 tok",  1024, 480, 1024, 32, 8,  8, false, true,  false, false },
        { "Granite-1B in-bank, 32 exp, 64 tok", 1024, 480, 1024, 32, 8, 64, false, true,  false, false },
        { "Granite-1B out-bank, 32 exp, 64 tok", 512, 320, 1024, 32, 8, 64, true,  true,  false, false },
        { "Qwen3.6 in-bank (r 416), 16 exp",    2048, 416, 1024, 16, 8,  9, false, true,  false, false },
        // large scaled activations through the batched route (64 tokens): s_pre * x reaches about 1e4 and the sums
        // over the rank exceed F16's range, which an F16 GEMM turned into inf/NaN
        { "Granite-1B in-bank, 64 tok, large",  1024, 480, 1024, 32, 8, 64, false, true,  true,  false, 100.0f },
        { "Granite-1B out-bank, 64 tok, large",  512, 320, 1024, 32, 8, 64, true,  true,  true,  false, 100.0f },
    };

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        }
    }

    std::mt19937 rng(1234);
    int failed = 0;
    for (const bank_case & c : cases) {
        std::vector<float> y_cpu;
        bank_data d;
        failed += run_case(c, f32, rng, &y_cpu, &d);
        for (ggml_backend_dev_t dev : gpus) {
            failed += run_case_device(c, dev, d, y_cpu);
        }
    }

    // the other path, switched in-process between graph computations, then back
    const int mode = f32 ? 1 : 0;
    if (g_get_mode() != mode) {
        std::printf("ggml_backend_cpu_get_s1_32_dot_mode() reports %d, GGML_S1_32_DOT selected %d\n", g_get_mode(), mode);
        failed++;
    }
    g_set_mode(f32 ? 0 : 1);
    failed += run_case(cases[2], !f32, rng);
    g_set_mode(mode);
    failed += run_case(cases[2], f32, rng);

    // the per-step modes: float in one product, Q8_0 in the other; z is exact (to float rounding) only when the first
    // product is float
    for (int m = 2; m <= 3; ++m) {
        g_set_mode(m);
        const bank_case & c = cases[2];
        const bank_data d = make_data(c, rng);
        const bank_bytes b = encode(c, d);
        ggml_init_params params = { (size_t) 256*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(params);
        const bank_tensors t = new_tensors(ctx, c, d);
        memcpy(t.v->data, b.v.data(), b.v.size());
        memcpy(t.u->data, b.u.data(), b.u.size());
        memcpy(t.sp->data, b.sp.data(), ggml_nbytes(t.sp));
        memcpy(t.so->data, b.so.data(), ggml_nbytes(t.so));
        memcpy(t.x->data, d.x.data(), ggml_nbytes(t.x));
        memcpy(t.ids_store->data, d.ids.data(), ggml_nbytes(t.ids_store));
        const llm_factored_mm_id_result res = llm_build_factored_mm_id(ctx, t.v, t.u, t.sp, t.so, t.x, t.ids);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, res.y);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        const double e_y = nmse((const float *) res.y->data, d.ref_y);
        const double e_z = nmse((const float *) res.z->data, d.ref_z);
        const bool z_exact = e_z <= 1e-10;
        const bool ok = (m == 2) == z_exact && e_y <= 5e-4 && e_y > 1e-10;
        std::printf("%-40s CPU mode %d (float in the %s product)  nmse(y, ref) %.3e  nmse(z, ref) %.3e  [%s]\n",
                    c.name, m, m == 2 ? "first" : "second", e_y, e_z, ok ? "ok" : "FAILED");
        failed += ok ? 0 : 1;
        ggml_free(ctx);
    }
    g_set_mode(mode);

    std::printf("test-factored-experts (%s): %s\n", f32 ? "f32" : "q8", failed == 0 ? "all cases passed" : "FAILED");
    return failed == 0 ? 0 : 1;
}
