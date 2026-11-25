// moeq binary-factor experts through llama: a small Granite-MoE model is written twice, once with dense F32 experts
// and once packed (the eight factored tensors per layer: S1_32 signs, F16 scales, the layout moeq export-packed
// writes), from the same signs and scales; both files are loaded with llama_model_load_from_file and decoded.
//
// Checks:
//  - the packed file loads, builds its graph and computes logits; on the CPU's float path they equal the dense file's
//    to float rounding (tensor names, loader shapes, gate-before-up order, glue and scale gathers all as written);
//  - on the CPU's Q8_0 path they differ from the dense file's only by activation quantization;
//  - the loader refuses: a sign tensor of another type, a layer with one bank only, a rank that differs between V and U;
//  - on every GPU device the build has, the fully offloaded packed model agrees with the CPU float path to the level
//    of activation quantization.
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t N_VOCAB = 96, N_EMBD = 128, N_HEAD = 4, N_HEAD_KV = 2, N_FF = 64, N_EXPERT = 4, N_USED = 2, N_LAYER = 2;
constexpr int64_t R_GU = 96, R_DN = 64;   // ranks: multiples of 32, as moeq writes them
constexpr int64_t HEAD_DIM = N_EMBD / N_HEAD;

typedef void (*s1_32_set_mode_t)(int);

// one bank's factors: V^T rows [E][r][n_in], U rows [E][n_out][r] (+-1), fp16-rounded scales
struct bank {
    int64_t n_in, n_out, r;
    std::vector<float> vt, uu, sp, so;

    // dense W_e[o][i] = so[o] sum_j uu[o][j] vt[j][i] sp[i]
    std::vector<float> dense() const {
        std::vector<float> w((size_t) N_EXPERT*n_out*n_in);
        std::vector<double> acc(n_in);
        for (int64_t e = 0; e < N_EXPERT; ++e) {
            for (int64_t o = 0; o < n_out; ++o) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t j = 0; j < r; ++j) {
                    const double s = uu[(e*n_out + o)*r + j];
                    const float * row = vt.data() + (e*r + j)*n_in;
                    for (int64_t i = 0; i < n_in; ++i) { acc[i] += s*row[i]; }
                }
                for (int64_t i = 0; i < n_in; ++i) {
                    w[(e*n_out + o)*n_in + i] = (float) (so[e*n_out + o]*acc[i]*sp[e*n_in + i]);
                }
            }
        }
        return w;
    }
};

static bank make_bank(int64_t n_in, int64_t n_out, int64_t r, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> coin(0, 1);
    bank b { n_in, n_out, r, {}, {}, {}, {} };
    b.vt.resize((size_t) N_EXPERT*r*n_in);
    b.uu.resize((size_t) N_EXPERT*n_out*r);
    for (float & s : b.vt) { s = coin(rng) ? 1.0f : -1.0f; }
    for (float & s : b.uu) { s = coin(rng) ? 1.0f : -1.0f; }
    b.sp.resize((size_t) N_EXPERT*n_in);
    b.so.resize((size_t) N_EXPERT*n_out);
    // W entries ~ |sp| |so| sqrt(r): aim at about 0.1
    const float s_out = 0.1f/(0.5f*std::sqrt((float) r));
    for (float & s : b.sp) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.5f*nd(rng))); }
    for (float & s : b.so) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s_out*(0.5f + std::fabs(nd(rng))))); }
    return b;
}

struct layer_weights {
    std::vector<float> attn_norm, wq, wk, wv, wo, ffn_norm, gate_inp;
    bank gu, dn;
};

struct model_weights {
    std::vector<float> tok_embd, output_norm, output;
    std::vector<layer_weights> layers;
};

static std::vector<float> normal(size_t n, float stdev, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, stdev);
    std::vector<float> v(n);
    for (float & x : v) { x = nd(rng); }
    return v;
}

static model_weights make_weights(uint32_t seed) {
    std::mt19937 rng(seed);
    model_weights m;
    m.tok_embd    = normal(N_EMBD*N_VOCAB, 1.0f, rng);
    m.output_norm = std::vector<float>(N_EMBD, 1.0f);
    m.output      = normal(N_EMBD*N_VOCAB, 0.1f, rng);
    for (int il = 0; il < N_LAYER; ++il) {
        layer_weights l;
        l.attn_norm = std::vector<float>(N_EMBD, 1.0f);
        l.wq        = normal(N_EMBD*N_EMBD, 0.1f, rng);
        l.wk        = normal(N_EMBD*HEAD_DIM*N_HEAD_KV, 0.1f, rng);
        l.wv        = normal(N_EMBD*HEAD_DIM*N_HEAD_KV, 0.1f, rng);
        l.wo        = normal(N_EMBD*N_EMBD, 0.1f, rng);
        l.ffn_norm  = std::vector<float>(N_EMBD, 1.0f);
        l.gate_inp  = normal(N_EMBD*N_EXPERT, 0.3f, rng);
        l.gu        = make_bank(N_EMBD, 2*N_FF, R_GU, rng);
        l.dn        = make_bank(N_FF, N_EMBD, R_DN, rng);
        m.layers.push_back(std::move(l));
    }
    return m;
}

enum class variant {
    dense,          // F32 expert tensors (gate, up, down)
    packed,         // the eight factored tensors per layer
    bad_sign_type,  // layer 0's gate_up V stored as F16 instead of S1_32
    one_bank,       // layer 1 without its down bank's factored tensors
    bad_scale_type, // layer 0's gate_up s_pre stored as BF16 (scales must be F16 or F32)
    rank_mismatch,  // layer 0's gate_up U written with a rank other than V's
};

struct gguf_writer {
    gguf_context * g;
    ggml_context * ctx;

    gguf_writer() {
        g = gguf_init_empty();
        ggml_init_params p = { (size_t) 64*1024*1024, nullptr, false };
        ctx = ggml_init(p);
    }
    ~gguf_writer() {
        gguf_free(g);
        ggml_free(ctx);
    }

    void f32(const std::string & name, const std::vector<int64_t> & ne, const std::vector<float> & data) {
        add(name, GGML_TYPE_F32, ne, [&](ggml_tensor * t) { std::memcpy(t->data, data.data(), ggml_nbytes(t)); });
    }
    void f16(const std::string & name, const std::vector<int64_t> & ne, const std::vector<float> & data) {
        add(name, GGML_TYPE_F16, ne, [&](ggml_tensor * t) { ggml_fp32_to_fp16_row(data.data(), (ggml_fp16_t *) t->data, ggml_nelements(t)); });
    }
    void s1_32(const std::string & name, const std::vector<int64_t> & ne, const std::vector<float> & data) {
        add(name, GGML_TYPE_S1_32, ne, [&](ggml_tensor * t) {
            ggml_quantize_chunk(GGML_TYPE_S1_32, data.data(), t->data, 0, ggml_nrows(t), t->ne[0], nullptr);
        });
    }
    void add(const std::string & name, ggml_type type, const std::vector<int64_t> & ne, const std::function<void(ggml_tensor *)> & fill) {
        ggml_tensor * t = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        ggml_set_name(t, name.c_str());
        fill(t);
        gguf_add_tensor(g, t);
    }
};

static std::string blk(int il, const char * name) {
    return "blk." + std::to_string(il) + "." + name;
}

static void write_model(const std::string & path, const model_weights & m, variant var) {
    gguf_writer w;
    const char * arch = "granitemoe";
    auto key = [&](const char * k) { return std::string(arch) + "." + k; };
    gguf_set_val_str(w.g, "general.architecture", arch);
    gguf_set_val_u32(w.g, key("vocab_size").c_str(), (uint32_t) N_VOCAB);
    gguf_set_val_u32(w.g, key("context_length").c_str(), 256);
    gguf_set_val_u32(w.g, key("embedding_length").c_str(), (uint32_t) N_EMBD);
    gguf_set_val_u32(w.g, key("feed_forward_length").c_str(), (uint32_t) N_FF);
    gguf_set_val_u32(w.g, key("block_count").c_str(), (uint32_t) N_LAYER);
    gguf_set_val_u32(w.g, key("attention.head_count").c_str(), (uint32_t) N_HEAD);
    gguf_set_val_u32(w.g, key("attention.head_count_kv").c_str(), (uint32_t) N_HEAD_KV);
    gguf_set_val_f32(w.g, key("rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_f32(w.g, key("attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    gguf_set_val_u32(w.g, key("expert_count").c_str(), (uint32_t) N_EXPERT);
    gguf_set_val_u32(w.g, key("expert_used_count").c_str(), (uint32_t) N_USED);
    gguf_set_val_f32(w.g, key("logit_scale").c_str(), 2.0f);
    gguf_set_val_f32(w.g, key("residual_scale").c_str(), 0.5f);
    gguf_set_val_f32(w.g, key("embedding_scale").c_str(), 2.0f);
    gguf_set_val_f32(w.g, key("attention.scale").c_str(), 1.0f/HEAD_DIM);
    gguf_set_val_str(w.g, "tokenizer.ggml.model", "test");
    {
        std::vector<std::string> toks(N_VOCAB);
        std::vector<const char *> ptrs(N_VOCAB);
        std::vector<float> scores(N_VOCAB, 0.0f);
        for (int i = 0; i < N_VOCAB; ++i) {
            toks[i] = "tok_" + std::to_string(i);
            ptrs[i] = toks[i].c_str();
        }
        gguf_set_arr_str(w.g, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
        gguf_set_arr_data(w.g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    }
    if (var != variant::dense) {
        gguf_set_val_u32(w.g, "general.file_type", (uint32_t) LLAMA_FTYPE_MOSTLY_S1_32_EXPS);
    }

    w.f32("token_embd.weight", { N_EMBD, N_VOCAB }, m.tok_embd);
    w.f32("output_norm.weight", { N_EMBD }, m.output_norm);
    w.f32("output.weight", { N_EMBD, N_VOCAB }, m.output);
    for (int il = 0; il < N_LAYER; ++il) {
        const layer_weights & l = m.layers[il];
        w.f32(blk(il, "attn_norm.weight"), { N_EMBD }, l.attn_norm);
        w.f32(blk(il, "attn_q.weight"), { N_EMBD, N_EMBD }, l.wq);
        w.f32(blk(il, "attn_k.weight"), { N_EMBD, HEAD_DIM*N_HEAD_KV }, l.wk);
        w.f32(blk(il, "attn_v.weight"), { N_EMBD, HEAD_DIM*N_HEAD_KV }, l.wv);
        w.f32(blk(il, "attn_output.weight"), { N_EMBD, N_EMBD }, l.wo);
        w.f32(blk(il, "ffn_norm.weight"), { N_EMBD }, l.ffn_norm);
        w.f32(blk(il, "ffn_gate_inp.weight"), { N_EMBD, N_EXPERT }, l.gate_inp);
        if (var == variant::dense) {
            // gate rows first, then up (moeq's fused bank order)
            const std::vector<float> gu = l.gu.dense();
            std::vector<float> gate((size_t) N_EXPERT*N_FF*N_EMBD), up(gate.size());
            for (int64_t e = 0; e < N_EXPERT; ++e) {
                std::memcpy(gate.data() + e*N_FF*N_EMBD, gu.data() + (e*2*N_FF)*N_EMBD,        N_FF*N_EMBD*sizeof(float));
                std::memcpy(up.data()   + e*N_FF*N_EMBD, gu.data() + (e*2*N_FF + N_FF)*N_EMBD, N_FF*N_EMBD*sizeof(float));
            }
            w.f32(blk(il, "ffn_gate_exps.weight"), { N_EMBD, N_FF, N_EXPERT }, gate);
            w.f32(blk(il, "ffn_up_exps.weight"), { N_EMBD, N_FF, N_EXPERT }, up);
            w.f32(blk(il, "ffn_down_exps.weight"), { N_FF, N_EMBD, N_EXPERT }, l.dn.dense());
            continue;
        }
        const bank & gu = l.gu;
        if (var == variant::bad_sign_type && il == 0) {
            w.f16(blk(il, "ffn_gate_up_exps_v.weight"), { gu.n_in, gu.r, N_EXPERT }, gu.vt);
        } else {
            w.s1_32(blk(il, "ffn_gate_up_exps_v.weight"), { gu.n_in, gu.r, N_EXPERT }, gu.vt);
        }
        if (var == variant::bad_scale_type && il == 0) {
            w.add(blk(il, "ffn_gate_up_exps_v.scale"), GGML_TYPE_BF16, { gu.n_in, 1, N_EXPERT }, [&](ggml_tensor * t) {
                ggml_fp32_to_bf16_row(gu.sp.data(), (ggml_bf16_t *) t->data, ggml_nelements(t));
            });
        } else {
            w.f16(blk(il, "ffn_gate_up_exps_v.scale"), { gu.n_in, 1, N_EXPERT }, gu.sp);
        }
        if (var == variant::rank_mismatch && il == 0) {
            std::vector<float> uu((size_t) N_EXPERT*gu.n_out*(gu.r - 32), 1.0f);
            w.s1_32(blk(il, "ffn_gate_up_exps_u.weight"), { gu.r - 32, gu.n_out, N_EXPERT }, uu);
        } else {
            w.s1_32(blk(il, "ffn_gate_up_exps_u.weight"), { gu.r, gu.n_out, N_EXPERT }, gu.uu);
        }
        w.f16(blk(il, "ffn_gate_up_exps_u.scale"), { gu.n_out, 1, N_EXPERT }, gu.so);
        if (var == variant::one_bank && il == 1) {
            continue;
        }
        const bank & dn = l.dn;
        w.s1_32(blk(il, "ffn_down_exps_v.weight"), { dn.n_in, dn.r, N_EXPERT }, dn.vt);
        w.f16(blk(il, "ffn_down_exps_v.scale"), { dn.n_in, 1, N_EXPERT }, dn.sp);
        w.s1_32(blk(il, "ffn_down_exps_u.weight"), { dn.r, dn.n_out, N_EXPERT }, dn.uu);
        w.f16(blk(il, "ffn_down_exps_u.scale"), { dn.n_out, 1, N_EXPERT }, dn.so);
    }
    if (!gguf_write_to_file(w.g, path.c_str(), false)) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }
}

static bool silent(float, void *) { return true; }

static llama_model * load(const std::string & path, ggml_backend_dev_t gpu) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent;
    static ggml_backend_dev_t none[] = { nullptr };
    ggml_backend_dev_t one[] = { gpu, nullptr };
    if (gpu) {
        mp.devices = one;
        mp.n_gpu_layers = 99;
    } else {
        mp.devices = none;
        mp.n_gpu_layers = 0;
    }
    return llama_model_load_from_file(path.c_str(), mp);
}

static std::vector<float> logits(llama_model * model, const std::vector<llama_token> & tokens, int n_ubatch, bool gpu) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    cp.n_ubatch = n_ubatch;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = gpu;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "cannot create a context\n");
        std::exit(1);
    }
    std::vector<float> out;
    {
        common_batch batch(ctx);
        for (size_t p = 0; p < tokens.size(); ++p) {
            batch.add(tokens[p], (llama_pos) p, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            std::fprintf(stderr, "decode failed\n");
            std::exit(1);
        }
        out.assign(llama_get_logits(ctx), llama_get_logits(ctx) + tokens.size()*N_VOCAB);
    }
    llama_free(ctx);
    return out;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        num += d*d;
        den += (double) ref[i]*ref[i];
    }
    return den > 0 ? num/den : num;
}

} // namespace

int main() {
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
    }, nullptr);

    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    auto set_mode = cpu_reg ? (s1_32_set_mode_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_s1_32_dot_mode") : nullptr;
    if (!set_mode) {
        std::printf("the CPU backend does not offer ggml_backend_cpu_set_s1_32_dot_mode\n");
        return 1;
    }

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "test-packed-experts";
    std::filesystem::create_directories(dir);
    auto path = [&](const char * name) { return (dir / name).string(); };

    const model_weights mw = make_weights(42);
    write_model(path("dense.gguf"), mw, variant::dense);
    write_model(path("packed.gguf"), mw, variant::packed);
    write_model(path("bad_sign_type.gguf"), mw, variant::bad_sign_type);
    write_model(path("bad_scale_type.gguf"), mw, variant::bad_scale_type);
    write_model(path("one_bank.gguf"), mw, variant::one_bank);
    write_model(path("rank_mismatch.gguf"), mw, variant::rank_mismatch);

    std::vector<llama_token> tokens(24);
    std::mt19937 rng(7);
    for (llama_token & t : tokens) { t = (llama_token) (rng() % N_VOCAB); }

    int failed = 0;
    auto check = [&](const char * what, bool ok, double value) {
        std::printf("%-62s %.3e  [%s]\n", what, value, ok ? "ok" : "FAILED");
        failed += ok ? 0 : 1;
    };

    llama_model * dense = load(path("dense.gguf"), nullptr);
    llama_model * packed = load(path("packed.gguf"), nullptr);
    if (!dense || !packed) {
        std::printf("loading failed: dense %p, packed %p\n", (void *) dense, (void *) packed);
        return 1;
    }
    char desc[128];
    llama_model_desc(packed, desc, sizeof(desc));
    std::printf("packed model: %s, %d layers\n", desc, llama_model_n_layer(packed));
    check("the packed file's general.file_type reads as S1_32 experts",
          llama_model_ftype(packed) == LLAMA_FTYPE_MOSTLY_S1_32_EXPS, (double) llama_model_ftype(packed));

    set_mode(0);
    const std::vector<float> ref = logits(dense, tokens, 512, false);
    set_mode(1);
    const std::vector<float> f32 = logits(packed, tokens, 512, false);
    const std::vector<float> f32_ub1 = logits(packed, tokens, 1, false);
    set_mode(0);
    const std::vector<float> q8 = logits(packed, tokens, 512, false);

    const double e_f32 = nmse(ref, f32), e_f32_ub1 = nmse(ref, f32_ub1), e_q8 = nmse(ref, q8);
    check("packed, CPU float path, vs dense F32 experts (nmse of logits)", e_f32 <= 1e-9, e_f32);
    check("packed, CPU float path, one token per ubatch, vs dense", e_f32_ub1 <= 1e-9, e_f32_ub1);
    check("packed, CPU Q8_0 path, vs dense (activation quantization)", e_q8 <= 2e-3 && e_q8 > 1e-12, e_q8);

    // the loader's refusals
    const struct { const char * file; const char * what; } refusals[] = {
        { "bad_sign_type.gguf", "refuses a sign tensor stored as F16" },
        { "bad_scale_type.gguf", "refuses a scale stored as BF16" },
        { "one_bank.gguf",      "refuses a layer with only one factored bank" },
        { "rank_mismatch.gguf", "refuses a U whose rank differs from V's" },
    };
    for (const auto & r : refusals) {
        llama_model * m = load(path(r.file), nullptr);
        check(r.what, m == nullptr, 0.0);
        if (m) {
            llama_model_free(m);
        }
    }

    // every GPU device: the fully offloaded packed model against the CPU float path, one token per ubatch (the
    // mat-vec path) and the whole sequence in one ubatch (the batched path)
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        llama_model * g = load(path("packed.gguf"), dev);
        if (!g) {
            check("packed model loads on the GPU", false, 0.0);
            continue;
        }
        for (int ub : { 1, 8, 24 }) {
            const std::vector<float> lg = logits(g, tokens, ub, true);
            const double e = nmse(f32, lg);
            char what[128];
            std::snprintf(what, sizeof(what), "packed on %s, ubatch %d, vs CPU float path", ggml_backend_dev_name(dev), ub);
            check(what, e <= 2e-3 && std::isfinite(e), e);
        }
        llama_model_free(g);
    }

    llama_model_free(dense);
    llama_model_free(packed);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    llama_backend_free();
    std::printf("test-packed-experts: %s\n", failed == 0 ? "all checks passed" : "FAILED");
    return failed == 0 ? 0 : 1;
}
