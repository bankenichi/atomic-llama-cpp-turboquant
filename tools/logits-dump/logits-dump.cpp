// llama-logits-dump: one model's logits at every position of token windows, written raw for checks outside llama.cpp,
// or compared on the fly with a stored reference (moeq Task 33: the fork's Qwen3.5 graphs against transformers,
// packed files against dense ones, one model in memory at a time).
//
//   llama-logits-dump -m MODEL.gguf (-f TEXT | --ids IDS.bin | --ref REF.bin) [-c 64] [--chunks N] [-t 8]
//                     [-dot q8|f32] [-fa on|off|auto] [--no-repack] -o OUT
//
// Windows: -f gives llama-perplexity's (the text tokenized with the vocabulary's special tokens, cut into n_ctx-token
// chunks, the first token of each replaced by BOS when the vocabulary adds BOS); --ids gives little-endian int32 token
// ids, used as they are, n_ctx per window; --ref takes the windows (and n_ctx) of a reference dump. Each window is
// decoded alone from an empty memory, every position output. -dot selects the S1_32 path (q8: Q8_0 activations, f32:
// the float reference path; default as GGML_S1_32_DOT says), -fa flash attention.
//
// Without --ref, OUT is the dump: "LGT1", then u32 n_ctx, u32 n_vocab, u32 n_windows; per window n_ctx int32 token
// ids, then n_ctx x n_vocab float32 logits (position-major).
// With --ref REF.bin (such a dump of the reference model), OUT is JSON: per window and position p the KL divergence
// KL(reference || this model) of the next-token distributions in double precision, whether the top tokens agree, and
// both models' negative log-likelihood of token p+1 (positions 0 .. n_ctx-2), plus the seconds per window.
#include "common.h"
#include "llama.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL.gguf (-f TEXT | --ids IDS.bin | --ref REF.bin) [-c 64] [--chunks N] [-t 8] "
                         "[-dot q8|f32] [-fa on|off|auto] [--no-repack] -o OUT\n", argv0);
}

struct pos_result { double kld; bool same_top; double nll_ref, nll_cand; };

// KL(p_ref || p_cand), whether the top tokens agree, and both negative log-likelihoods of `next`
static pos_result compare_position(const float * a, const float * b, int n_vocab, int next) {
    float max_a = a[0], max_b = b[0];
    int top_a = 0, top_b = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (a[v] > max_a) { max_a = a[v]; top_a = v; }
        if (b[v] > max_b) { max_b = b[v]; top_b = v; }
    }
    double sum_a = 0.0, sum_b = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        sum_a += std::exp((double) a[v] - max_a);
        sum_b += std::exp((double) b[v] - max_b);
    }
    const double lse_a = max_a + std::log(sum_a);
    const double lse_b = max_b + std::log(sum_b);
    double kl = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double la = a[v] - lse_a;
        const double lb = b[v] - lse_b;
        kl += std::exp(la)*(la - lb);
    }
    return { std::max(kl, 0.0), top_a == top_b, next >= 0 ? lse_a - a[next] : NAN, next >= 0 ? lse_b - b[next] : NAN };
}

int main(int argc, char ** argv) {
    std::string path_model, path_text, path_ids, path_ref, path_out;
    int n_ctx = 64, n_threads = 8, n_chunks_max = -1, dot = -1;
    bool repack = true;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          { path_model = next(); }
        else if (a == "-f")          { path_text = next(); }
        else if (a == "--ids")       { path_ids = next(); }
        else if (a == "--ref")       { path_ref = next(); }
        else if (a == "-o")          { path_out = next(); }
        else if (a == "-c")          { n_ctx = std::atoi(next()); }
        else if (a == "-t")          { n_threads = std::atoi(next()); }
        else if (a == "--chunks")    { n_chunks_max = std::atoi(next()); }
        else if (a == "--no-repack") { repack = false; }
        else if (a == "-dot") {
            const std::string v = next();
            dot = v == "f32" ? 1 : v == "q8" ? 0 : -1;
        }
        else if (a == "-fa") {
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        else { usage(argv[0]); return 1; }
    }
    const int n_sources = !path_text.empty() + !path_ids.empty() + !path_ref.empty();
    if (path_model.empty() || path_out.empty() || n_sources != 1 || n_ctx < 1) {
        usage(argv[0]);
        return 1;
    }

    // the reference dump: its header gives n_ctx and the windows
    FILE * ref = nullptr;
    uint32_t ref_header[3] = { 0, 0, 0 };
    if (!path_ref.empty()) {
        ref = std::fopen(path_ref.c_str(), "rb");
        char magic[4];
        if (!ref || std::fread(magic, 1, 4, ref) != 4 || std::memcmp(magic, "LGT1", 4) != 0 ||
            std::fread(ref_header, sizeof(uint32_t), 3, ref) != 3) {
            std::fprintf(stderr, "%s is not a logits dump\n", path_ref.c_str());
            return 1;
        }
        n_ctx = (int) ref_header[0];
    }

    llama_backend_init();
    if (dot >= 0) {
        s1_32_set_mode(dot == 1 ? 1 : 0);
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = 0;
    mp.use_extra_bufts = repack;
    llama_model * model = llama_model_load_from_file(path_model.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "failed to load %s\n", path_model.c_str());
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.n_ubatch        = n_ctx;
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn_type = fa;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create a context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (ref && (int) ref_header[1] != n_vocab) {
        std::fprintf(stderr, "the reference has %u logits per position, this model %d\n", ref_header[1], n_vocab);
        return 1;
    }

    std::vector<llama_token> tokens;
    const bool from_text = !path_text.empty();
    if (from_text) {
        std::ifstream in(path_text, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        tokens = common_tokenize(ctx, ss.str(), true);
    } else if (!path_ids.empty()) {
        std::ifstream in(path_ids, std::ios::binary | std::ios::ate);
        const std::streamsize n = in.tellg();
        in.seekg(0);
        tokens.resize(size_t(n)/sizeof(int32_t));
        in.read((char *) tokens.data(), tokens.size()*sizeof(int32_t));
    }
    int n_chunks = ref ? (int) ref_header[2] : int(tokens.size()/n_ctx);
    if (n_chunks_max >= 0) {
        n_chunks = std::min(n_chunks, n_chunks_max);
    }
    const bool add_bos = from_text && llama_vocab_get_add_bos(vocab);
    std::printf("%s: %d windows of %d, n_vocab %d, S1_32 dot %s, flash attention %s%s\n", path_model.c_str(), n_chunks,
                n_ctx, n_vocab, s1_32_get_mode() == 1 ? "f32" : "q8", llama_flash_attn_type_name(fa),
                ref ? (", against " + path_ref).c_str() : "");

    FILE * out = std::fopen(path_out.c_str(), ref ? "w" : "wb");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path_out.c_str());
        return 1;
    }
    if (!ref) {
        const uint32_t header[3] = { (uint32_t) n_ctx, (uint32_t) n_vocab, (uint32_t) n_chunks };
        std::fwrite("LGT1", 1, 4, out);
        std::fwrite(header, sizeof(uint32_t), 3, out);
    } else {
        auto json_escaped = [](const std::string & s) {       // Windows paths hold backslashes
            std::string r;
            for (char c : s) {
                if (c == '\\' || c == '"') {
                    r += '\\';
                }
                r += c;
            }
            return r;
        };
        std::fprintf(out, "{\n  \"model\": \"%s\",\n  \"reference\": \"%s\",\n  \"n_ctx\": %d,\n  \"s1_32_dot\": \"%s\",\n"
                     "  \"flash_attn\": \"%s\",\n  \"threads\": %d,\n  \"windows\": [\n", json_escaped(path_model).c_str(),
                     json_escaped(path_ref).c_str(),
                     n_ctx, s1_32_get_mode() == 1 ? "f32" : "q8", llama_flash_attn_type_name(fa), n_threads);
    }

    common_batch batch(ctx);
    std::vector<llama_token> win(n_ctx);
    std::vector<float> ref_logits;
    if (ref) {
        ref_logits.resize(size_t(n_ctx)*n_vocab);
    }
    double seconds = 0.0;
    for (int w = 0; w < n_chunks; ++w) {
        if (ref) {
            if (std::fread(win.data(), sizeof(int32_t), n_ctx, ref) != (size_t) n_ctx ||
                std::fread(ref_logits.data(), sizeof(float), ref_logits.size(), ref) != ref_logits.size()) {
                std::fprintf(stderr, "%s ends at window %d\n", path_ref.c_str(), w);
                return 1;
            }
        } else {
            std::copy(tokens.begin() + size_t(w)*n_ctx, tokens.begin() + size_t(w + 1)*n_ctx, win.begin());
            if (add_bos) {
                win[0] = llama_vocab_bos(vocab);
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.clear();
        for (int p = 0; p < n_ctx; ++p) {
            batch.add(win[p], p, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            std::fprintf(stderr, "decode failed at window %d\n", w);
            return 1;
        }
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!ref) {
            std::fwrite(win.data(), sizeof(int32_t), n_ctx, out);
            for (int p = 0; p < n_ctx; ++p) {
                std::fwrite(llama_get_logits_ith(ctx, p), sizeof(float), n_vocab, out);
            }
        } else {
            std::string kl = "[", top = "[", nr = "[", nc = "[";
            char buf[64];
            for (int p = 0; p < n_ctx; ++p) {
                const pos_result r = compare_position(ref_logits.data() + size_t(p)*n_vocab, llama_get_logits_ith(ctx, p),
                                                      n_vocab, p + 1 < n_ctx ? win[p + 1] : -1);
                std::snprintf(buf, sizeof(buf), "%s%.9g", p ? ", " : "", r.kld);
                kl += buf;
                top += (p ? ", " : "") + std::string(r.same_top ? "1" : "0");
                if (p + 1 < n_ctx) {
                    std::snprintf(buf, sizeof(buf), "%s%.9g", p ? ", " : "", r.nll_ref);
                    nr += buf;
                    std::snprintf(buf, sizeof(buf), "%s%.9g", p ? ", " : "", r.nll_cand);
                    nc += buf;
                }
            }
            std::fprintf(out, "%s    {\"kl\": %s], \"same_top\": %s], \"nll_ref\": %s], \"nll\": %s]}", w ? ",\n" : "",
                         kl.c_str(), top.c_str(), nr.c_str(), nc.c_str());
        }
        if ((w + 1) % 10 == 0 || w + 1 == n_chunks) {
            std::printf("[%d/%d] %.2f s/window\n", w + 1, n_chunks, seconds/(w + 1));
            std::fflush(stdout);
        }
    }
    if (ref) {
        std::fprintf(out, "\n  ],\n  \"seconds_per_window\": %.4f\n}\n", seconds/std::max(n_chunks, 1));
        std::fclose(ref);
    }
    std::fclose(out);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
