// moeq binary-factor experts at the model level: a packed GGUF (S1_32 signs and F16 scales in place of a layer's
// feed-forward matrices) must load and give the logits of the dense model that holds the same matrices.
//
//   test-factored-model MODEL_DIR
//
// MODEL_DIR holds the random models test-llama-archs writes (-o): qwen35-dense.gguf (Qwen3.5's dense feed-forward,
// the 9B's), qwen35moe-moe.gguf and olmoe-moe.gguf (stacked experts through build_moe_ffn). For each, with random signs
// and scales, two files are written: "dense", the layers' feed-forward tensors replaced by the F32 matrices
// diag(s_post) sign(U) sign(V)^T diag(s_pre) (gate rows first), and "packed", the same layers holding the factored
// tensors instead (dense: blk.N.ffn_gate_up_{v,u}, blk.N.ffn_down_{v,u}; MoE: blk.N.ffn_gate_up_exps_{v,u},
// blk.N.ffn_down_exps_{v,u}; .weight S1_32, .scale F16). Checked on the CPU, for the first layer only (a partial-depth
// student) and for every layer:
//   - both files load; the float path (F32 activations) gives the dense file's logits to float rounding,
//     the 8-bit path (Q8_0 activations) to activation-quantization noise;
//   - the loader refuses a sign tensor that is not S1_32, a scale that is neither F16 nor F32, and a layer with only
//     one of its two banks.
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include <cstdlib>

// the CPU's S1_32 activation mode is fork-private, reached through the CPU backend's proc address (0: Q8_0, 1: F32)
static void * s1_32_mode_fn(const char * name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    void * fn = reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
    if (fn == nullptr) {
        std::fprintf(stderr, "the CPU backend does not offer %s\n", name);
        std::exit(1);
    }
    return fn;
}

static void s1_32_set_mode(int mode) {
    ((void (*)(int)) s1_32_mode_fn("ggml_backend_cpu_set_s1_32_dot_mode"))(mode);
}

static int s1_32_get_mode() {
    return ((int (*)(void)) s1_32_mode_fn("ggml_backend_cpu_get_s1_32_dot_mode"))();
}

namespace fs = std::filesystem;

struct bank_data {
    int64_t n_in = 0, n_out = 0, rank = 0, n_expert = 0;
    std::vector<int8_t> vt;    // [E][r][n_in]   sign(V)^T
    std::vector<int8_t> u;     // [E][n_out][r]  sign(U)
    std::vector<float>  s_pre, s_post;   // [E][n_in], [E][n_out], already F16-rounded

    void init(int64_t in, int64_t out, int64_t r, int64_t e, std::mt19937 & rng) {
        n_in = in; n_out = out; rank = r; n_expert = e;
        std::bernoulli_distribution coin(0.5);
        std::uniform_real_distribution<float> mag(0.05f, 0.12f);
        vt.resize(e*r*in);
        u.resize(e*out*r);
        for (auto & x : vt) { x = coin(rng) ? 1 : -1; }
        for (auto & x : u)  { x = coin(rng) ? 1 : -1; }
        auto scale = [&]() { return ggml_fp16_to_fp32(ggml_fp32_to_fp16((coin(rng) ? 1.0f : -1.0f)*mag(rng))); };
        s_pre.resize(e*in);
        s_post.resize(e*out);
        for (auto & x : s_pre)  { x = scale(); }
        for (auto & x : s_post) { x = scale(); }
    }

    // the dense matrix of expert e, row-major [n_out][n_in]
    std::vector<float> dense(int64_t e) const {
        std::vector<float> w(n_out*n_in);
        std::vector<double> row(n_in);
        for (int64_t o = 0; o < n_out; ++o) {
            std::fill(row.begin(), row.end(), 0.0);
            for (int64_t k = 0; k < rank; ++k) {
                const int8_t uk = u[(e*n_out + o)*rank + k];
                const int8_t * v = &vt[(e*rank + k)*n_in];
                for (int64_t i = 0; i < n_in; ++i) {
                    row[i] += uk*v[i];
                }
            }
            for (int64_t i = 0; i < n_in; ++i) {
                w[o*n_in + i] = float(s_post[e*n_out + o]*row[i]*s_pre[e*n_in + i]);
            }
        }
        return w;
    }
};

// signs as S1_32 rows: 32 per uint32, bit j = element j of the block, 1 = +1
static void pack_rows(uint8_t * dst, const int8_t * src, int64_t n_rows, int64_t row) {
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t b = 0; b < row/32; ++b) {
            uint32_t w = 0;
            for (int j = 0; j < 32; ++j) {
                if (src[r*row + b*32 + j] > 0) {
                    w |= 1u << j;
                }
            }
            std::memcpy(dst + (r*(row/32) + b)*4, &w, 4);
        }
    }
}

struct layer_plan {
    bank_data gu, dn;
};

enum class defect { none, sign_f16, scale_q8, one_bank };

struct rewrite {
    gguf_context * src = nullptr;
    ggml_context * src_data = nullptr;
    bool moe = false;
    float embd_scale = 1.0f;
    std::vector<ggml_context *> owned;

    ~rewrite() {
        for (auto * c : owned) { ggml_free(c); }
        if (src) { gguf_free(src); }
        if (src_data) { ggml_free(src_data); }
    }

    ggml_tensor * get(const std::string & name) const { return ggml_get_tensor(src_data, name.c_str()); }

    ggml_context * new_ctx(size_t bytes) {
        ggml_init_params p = { bytes + 64*ggml_tensor_overhead(), nullptr, false };
        ggml_context * c = ggml_init(p);
        owned.push_back(c);
        return c;
    }
};

// the feed-forward tensors of layer il that a factored layer replaces (with their per-tensor scale companions)
static bool replaced_name(const std::string & name, int il, bool moe) {
    const std::string pre = "blk." + std::to_string(il) + ".ffn_";
    if (name.rfind(pre, 0) != 0) {
        return false;
    }
    const std::string rest = name.substr(pre.size());
    static const char * dense_parts[] = { "gate.", "up.", "down." };
    static const char * moe_parts[]   = { "gate_exps.", "up_exps.", "down_exps.", "gate_up_exps." };
    if (moe) {
        for (const char * p : moe_parts) { if (rest.rfind(p, 0) == 0) { return true; } }
    } else {
        for (const char * p : dense_parts) { if (rest.rfind(p, 0) == 0) { return true; } }
    }
    return false;
}

static void write_file(rewrite & rw, const std::set<int> & layers, std::vector<layer_plan> & plans, bool packed,
                       defect bad, const std::string & path) {
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, rw.src);
    const int64_t n_tensors = gguf_get_n_tensors(rw.src);
    std::set<int> done;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const std::string name = gguf_get_tensor_name(rw.src, ti);
        int il = -1;
        for (int l : layers) {
            if (replaced_name(name, l, rw.moe)) { il = l; }
        }
        if (il < 0) {
            ggml_tensor * t = rw.get(name);
            if (rw.embd_scale != 1.0f && name == "token_embd.weight" &&
                (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {
                // large residual activations (the Qwen models reach position-0 norms of 38 and more): scaled embeddings
                ggml_context * c = rw.new_ctx(ggml_nbytes(t));
                ggml_tensor * s = ggml_dup_tensor(c, t);
                ggml_set_name(s, name.c_str());
                for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                    if (t->type == GGML_TYPE_F32) {
                        ((float *) s->data)[i] = rw.embd_scale*((const float *) t->data)[i];
                    } else {
                        ((ggml_fp16_t *) s->data)[i] = ggml_fp32_to_fp16(rw.embd_scale*ggml_fp16_to_fp32(((const ggml_fp16_t *) t->data)[i]));
                    }
                }
                gguf_add_tensor(dst, s);
            } else {
                gguf_add_tensor(dst, t);
            }
            continue;
        }
        if (name.find(".weight") == std::string::npos || done.count(il)) {
            continue;   // a companion scale, or a second matrix of a layer already written
        }
        done.insert(il);
        const layer_plan & lp = plans[il];
        const std::string blk = "blk." + std::to_string(il) + ".";
        if (!packed) {
            // the dense matrices under the file's own names and shapes
            const int64_t n_ff = lp.dn.n_in;
            const int64_t E = lp.gu.n_expert;
            std::vector<std::string> names;
            for (int64_t t2 = 0; t2 < n_tensors; ++t2) {
                const std::string n2 = gguf_get_tensor_name(rw.src, t2);
                if (replaced_name(n2, il, rw.moe) && n2.find(".weight") != std::string::npos) {
                    names.push_back(n2);
                }
            }
            for (const std::string & n2 : names) {
                const ggml_tensor * old = rw.get(n2);
                ggml_context * c = rw.new_ctx(ggml_nelements(old)*sizeof(float));
                ggml_tensor * t = ggml_new_tensor(c, GGML_TYPE_F32, ggml_n_dims(old), old->ne);
                ggml_set_name(t, n2.c_str());
                float * out = (float *) t->data;
                const bool down = n2.find("ffn_down") != std::string::npos;
                const bool gate_up = n2.find("gate_up") != std::string::npos;
                const bool up = !gate_up && n2.find("ffn_up") != std::string::npos;
                for (int64_t e = 0; e < E; ++e) {
                    const std::vector<float> w = down ? lp.dn.dense(e) : lp.gu.dense(e);
                    const int64_t rows = down ? lp.dn.n_out : (gate_up ? 2*n_ff : n_ff);
                    const int64_t cols = down ? lp.dn.n_in : lp.gu.n_in;
                    const int64_t first = up ? n_ff : 0;
                    std::memcpy(out + e*rows*cols, w.data() + first*cols, rows*cols*sizeof(float));
                }
                gguf_add_tensor(dst, t);
            }
            continue;
        }
        const std::string base_gu = rw.moe ? "ffn_gate_up_exps" : "ffn_gate_up";
        const std::string base_dn = rw.moe ? "ffn_down_exps" : "ffn_down";
        for (int bank = 0; bank < 2; ++bank) {
            if (bad == defect::one_bank && bank == 1) {
                continue;
            }
            const bank_data & bd = bank == 0 ? lp.gu : lp.dn;
            const std::string base = blk + (bank == 0 ? base_gu : base_dn);
            const int64_t E = bd.n_expert;
            const ggml_type sign_type = bad == defect::sign_f16 ? GGML_TYPE_F16 : GGML_TYPE_S1_32;
            const ggml_type scale_type = bad == defect::scale_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            ggml_context * c = rw.new_ctx(4*(E*bd.rank*(bd.n_in + bd.n_out) + E*(bd.n_in + bd.n_out)) + 4096);
            auto mk = [&](ggml_type type, int64_t ne0, int64_t ne1, const std::string & name) {
                ggml_tensor * t = rw.moe ? ggml_new_tensor_3d(c, type, ne0, ne1, E) :
                                  ne1 == 1 ? ggml_new_tensor_1d(c, type, ne0) : ggml_new_tensor_2d(c, type, ne0, ne1);
                ggml_set_name(t, name.c_str());
                return t;
            };
            ggml_tensor * v  = mk(sign_type,  bd.n_in,  bd.rank,  base + "_v.weight");
            ggml_tensor * sp = mk(scale_type, bd.n_in,  1,        base + "_v.scale");
            ggml_tensor * u  = mk(sign_type,  bd.rank,  bd.n_out, base + "_u.weight");
            ggml_tensor * so = mk(scale_type, bd.n_out, 1,        base + "_u.scale");
            for (int64_t e = 0; e < E; ++e) {
                if (sign_type == GGML_TYPE_S1_32) {
                    pack_rows((uint8_t *) v->data + e*v->nb[2], &bd.vt[e*bd.rank*bd.n_in], bd.rank, bd.n_in);
                    pack_rows((uint8_t *) u->data + e*u->nb[2], &bd.u[e*bd.n_out*bd.rank], bd.n_out, bd.rank);
                } else {
                    ggml_fp16_t * vd = (ggml_fp16_t *) ((uint8_t *) v->data + e*v->nb[2]);
                    ggml_fp16_t * ud = (ggml_fp16_t *) ((uint8_t *) u->data + e*u->nb[2]);
                    for (int64_t i = 0; i < bd.rank*bd.n_in; ++i)  { vd[i] = ggml_fp32_to_fp16(bd.vt[e*bd.rank*bd.n_in + i]); }
                    for (int64_t i = 0; i < bd.n_out*bd.rank; ++i) { ud[i] = ggml_fp32_to_fp16(bd.u[e*bd.n_out*bd.rank + i]); }
                }
                if (scale_type == GGML_TYPE_F16) {
                    ggml_fp16_t * a = (ggml_fp16_t *) ((uint8_t *) sp->data + e*sp->nb[rw.moe ? 2 : 1]);
                    ggml_fp16_t * b = (ggml_fp16_t *) ((uint8_t *) so->data + e*so->nb[rw.moe ? 2 : 1]);
                    for (int64_t i = 0; i < bd.n_in; ++i)  { a[i] = ggml_fp32_to_fp16(bd.s_pre[e*bd.n_in + i]); }
                    for (int64_t i = 0; i < bd.n_out; ++i) { b[i] = ggml_fp32_to_fp16(bd.s_post[e*bd.n_out + i]); }
                } else {
                    ggml_quantize_chunk(scale_type, &bd.s_pre[e*bd.n_in],  (uint8_t *) sp->data + e*sp->nb[rw.moe ? 2 : 1], 0, 1, bd.n_in,  nullptr);
                    ggml_quantize_chunk(scale_type, &bd.s_post[e*bd.n_out], (uint8_t *) so->data + e*so->nb[rw.moe ? 2 : 1], 0, 1, bd.n_out, nullptr);
                }
            }
            gguf_add_tensor(dst, v);
            gguf_add_tensor(dst, sp);
            gguf_add_tensor(dst, u);
            gguf_add_tensor(dst, so);
        }
    }
    if (!gguf_write_to_file(dst, path.c_str(), false)) {
        throw std::runtime_error("cannot write " + path);
    }
    gguf_free(dst);
}

static std::vector<float> logits_of(const std::string & path, const std::vector<llama_token> & tokens, bool * loaded) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    *loaded = model != nullptr;
    if (!model) {
        return {};
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("cannot create a context for " + path);
    }
    {
        common_batch batch(ctx);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.add(tokens[i], (llama_pos) i, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("decode failed for " + path);
        }
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        out.insert(out.end(), l, l + n_vocab);
    }
    llama_free(ctx);
    llama_model_free(model);
    return out;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0, n = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (double(a[i]) - b[i])*(double(a[i]) - b[i]);
        n += double(a[i])*a[i];
    }
    return d/n;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR (the models of test-llama-archs -o)\n", argv[0]);
        return 1;
    }
    const fs::path dir = argv[1];
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
    }, nullptr);

    struct model_case { const char * file; bool moe; int64_t r_gu, r_dn; };
    const model_case cases[] = {
        { "qwen35-dense.gguf",  false, 64, 96 },
        { "qwen35moe-moe.gguf", true,  64, 96 },
        { "olmoe-moe.gguf",     true,  96, 64 },
        { "qwen4exp-moe.gguf",  true,  64, 96 },   // Qwen3.8-Flash-Next's architecture (its MTP draft head loads through the same lambda)
    };
    const fs::path tmp = fs::temp_directory_path();
    std::mt19937 rng(1234);
    std::vector<llama_token> tokens;
    int failures = 0;

    for (const model_case & mc : cases) {
        const fs::path src_path = dir / mc.file;
        if (!fs::exists(src_path)) {
            std::fprintf(stderr, "%s: missing (run test-llama-archs -o %s)\n", src_path.string().c_str(), dir.string().c_str());
            ++failures;
            continue;
        }
        rewrite rw;
        rw.moe = mc.moe;
        gguf_init_params ip = { false, &rw.src_data };
        rw.src = gguf_init_from_file(src_path.string().c_str(), ip);
        if (!rw.src) {
            std::fprintf(stderr, "cannot read %s\n", src_path.string().c_str());
            ++failures;
            continue;
        }
        const int n_layer = (int) gguf_get_val_u32(rw.src, gguf_find_key(rw.src,
                (std::string(gguf_get_val_str(rw.src, gguf_find_key(rw.src, "general.architecture"))) + ".block_count").c_str()));
        std::vector<layer_plan> plans(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            const ggml_tensor * down = rw.get("blk." + std::to_string(il) + (mc.moe ? ".ffn_down_exps.weight" : ".ffn_down.weight"));
            if (!down) {
                throw std::runtime_error(std::string(mc.file) + ": no down tensor in layer " + std::to_string(il));
            }
            const int64_t n_ff = down->ne[0], n_embd = down->ne[1], E = mc.moe ? down->ne[2] : 1;
            plans[il].gu.init(n_embd, 2*n_ff, mc.r_gu, E, rng);
            plans[il].dn.init(n_ff, n_embd, mc.r_dn, E, rng);
        }
        if (tokens.empty()) {
            std::uniform_int_distribution<int> tok(0, 127);
            for (int i = 0; i < 24; ++i) { tokens.push_back(tok(rng)); }
        }

        std::set<int> first = { 0 }, all;
        for (int il = 0; il < n_layer; ++il) { all.insert(il); }
        // big = 1000: the same matrices with s_pre x 1000 and s_post / 1000, so the packed product's input (s_pre * x)
        // and its intermediate z = sign(V)^T (s_pre * x) are 1000 times larger (sums of +-1 times large values, as the
        // Qwen models' large activations give them) while the dense matrix stays the same up to F16 rounding
        std::vector<layer_plan> plans_big = plans;
        for (auto & lp : plans_big) {
            for (bank_data * bd : { &lp.gu, &lp.dn }) {
                for (auto & s : bd->s_pre)  { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s*1000.0f)); }
                for (auto & s : bd->s_post) { s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s/1000.0f)); }
            }
        }
        for (const bool big : { false, true })
        for (const std::set<int> * layers : { &first, &all }) {
            std::vector<layer_plan> & pl = big ? plans_big : plans;
            const std::string tag = std::string(mc.file) + (layers == &first ? " (layer 0 packed" : " (all layers packed") +
                                    (big ? ", s_pre x1000 and s_post / 1000: z 1000x larger)" : ")");
            const std::string p_dense  = (tmp / "test-factored-model-dense.gguf").string();
            const std::string p_packed = (tmp / "test-factored-model-packed.gguf").string();
            write_file(rw, *layers, pl, false, defect::none, p_dense);
            write_file(rw, *layers, pl, true,  defect::none, p_packed);
            bool ok_d = false, ok_p = false, ok_q = false;
            s1_32_set_mode(1);
            const std::vector<float> ld = logits_of(p_dense, tokens, &ok_d);
            const std::vector<float> lf = logits_of(p_packed, tokens, &ok_p);
            s1_32_set_mode(0);
            const std::vector<float> lq = logits_of(p_packed, tokens, &ok_q);
            if (!ok_d || !ok_p || !ok_q) {
                std::fprintf(stderr, "FAIL %s: a file did not load (dense %d, packed %d)\n", tag.c_str(), ok_d, ok_p);
                ++failures;
                continue;
            }
            const double e_f32 = nmse(ld, lf), e_q8 = nmse(ld, lq);
            const bool pass = e_f32 < 1e-8 && e_q8 < 1e-4 && std::isfinite(e_q8);
            std::printf("%s %s: NMSE of the logits against the dense file: float path %.3e, 8-bit path %.3e\n",
                        pass ? "ok  " : "FAIL", tag.c_str(), e_f32, e_q8);
            failures += !pass;
        }

        rw.embd_scale = 1.0f;

        // the loader's refusals
        const std::pair<defect, const char *> bads[] = {
            { defect::sign_f16, "sign tensors stored as F16" },
            { defect::scale_q8, "scales stored as Q8_0" },
            { defect::one_bank, "one bank only" },
        };
        for (const auto & bad : bads) {
            const std::string p_bad = (tmp / "test-factored-model-bad.gguf").string();
            write_file(rw, first, plans, true, bad.first, p_bad);
            bool loaded = false;
            try {
                logits_of(p_bad, tokens, &loaded);
            } catch (const std::exception &) {
                loaded = true;   // it loaded and then failed: not a refusal
            }
            std::printf("%s %s: a file with %s is %s\n", loaded ? "FAIL" : "ok  ", mc.file, bad.second,
                        loaded ? "loaded" : "refused");
            failures += loaded;
        }
    }
    llama_backend_free();
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
