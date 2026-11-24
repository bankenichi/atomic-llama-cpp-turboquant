// llama-kld-compare: the KL divergence between models at every position of the llama-perplexity windows, computed
// exactly from the models' logits in one process.
//
//   llama-kld-compare -m A.gguf -m B.gguf [-m C.gguf ...] -f TEXT [-c 2048] [-t 8] [--chunks N] [-o OUT.json]
//
// Every pair (i < j) is compared as KL(model i || model j); each model decodes each window once.
//
// Written for checking moeq's packed experts (S1_32) against the dense export of the same student: llama-perplexity
// scores only the second half of each window and stores the reference as 16-bit log-probabilities, while this tool
// also gives position 0 (the sink position) and every other position, at full precision.
//
// Windows are llama-perplexity's: the text tokenized with the vocabulary's special tokens, cut into n_ctx-token
// chunks, the first token of each replaced by BOS when the vocabulary adds BOS. The logits at position p predict
// token p+1; positions 0 .. n_ctx-2 are compared, KL per position in double precision.
// Reported for three position sets: "protocol" (n_ctx/2 .. n_ctx-2, what llama-perplexity scores), "pos0" (one value
// per window) and "all"; each with mean, median, 99th percentile and maximum (percentiles interpolated as in
// llama-perplexity), the share of positions whose top token agrees, and the perplexity of both models on the
// protocol positions. The JSON output adds the per-window protocol means (for bootstrap intervals) and the
// per-window position-0 values.
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
#include <thread>
#include <vector>

struct set_stats {
    std::vector<float> kld;
    size_t same_top = 0;
    double nll_ref = 0.0, nll_cand = 0.0;
    size_t n_nll = 0;
};

static float percentile(const std::vector<float> & sorted, float fraction) {
    if (sorted.empty()) {
        return NAN;
    }
    if (fraction <= 0) {
        return sorted.front();
    }
    if (fraction >= 1) {
        return sorted.back();
    }
    float p = fraction*(sorted.size() - 1);
    const size_t ip = size_t(p);
    p -= ip;
    return (1 - p)*sorted[ip] + p*sorted[std::min(ip + 1, sorted.size() - 1)];
}

struct summary {
    size_t n;
    double mean, median, p99, max, same_top, ppl_ref, ppl_cand;
};

static summary summarize(set_stats s) {
    std::sort(s.kld.begin(), s.kld.end());
    double sum = 0.0;
    for (float v : s.kld) {
        sum += v;
    }
    summary r;
    r.n        = s.kld.size();
    r.mean     = r.n ? sum/r.n : NAN;
    r.median   = percentile(s.kld, 0.5f);
    r.p99      = percentile(s.kld, 0.99f);
    r.max      = r.n ? s.kld.back() : NAN;
    r.same_top = r.n ? double(s.same_top)/r.n : NAN;
    r.ppl_ref  = s.n_nll ? std::exp(s.nll_ref/s.n_nll)  : NAN;
    r.ppl_cand = s.n_nll ? std::exp(s.nll_cand/s.n_nll) : NAN;
    return r;
}

static void print_summary(const char * name, const summary & r) {
    std::printf("%-9s n %8zu  mean KLD %.6f  median %.6f  99%% %.6f  max %.6f  same top %.4f  PPL ref %.4f cand %.4f\n",
                name, r.n, r.mean, r.median, r.p99, r.max, r.same_top, r.ppl_ref, r.ppl_cand);
}

static std::string json_summary(const summary & r) {
    std::ostringstream o;
    o.precision(9);
    o << "{\"n\": " << r.n << ", \"mean\": " << r.mean << ", \"median\": " << r.median << ", \"p99\": " << r.p99
      << ", \"max\": " << r.max << ", \"same_top\": " << r.same_top << ", \"ppl_reference\": " << r.ppl_ref
      << ", \"ppl_candidate\": " << r.ppl_cand << "}";
    return o.str();
}

// KL(p_ref || p_cand), whether the top tokens agree, and both negative log-likelihoods of `next`
struct pos_result { float kld; bool same_top; double nll_ref, nll_cand; };

static pos_result compare_position(const float * a, const float * b, int n_vocab, llama_token next) {
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
    return { (float) std::max(kl, 0.0), top_a == top_b, lse_a - a[next], lse_b - b[next] };
}

static bool decode_window(llama_context * ctx, common_batch & batch, const llama_token * tokens, int n_ctx,
                          std::vector<float> & logits) {
    llama_memory_clear(llama_get_memory(ctx), true);
    batch.clear();
    for (int p = 0; p < n_ctx; ++p) {
        batch.add(tokens[p], p, 0, p < n_ctx - 1);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * out = llama_get_logits(ctx);
    logits.assign(out, out + size_t(n_ctx - 1)*n_vocab);
    return true;
}

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s [-fa on|off|auto] [-dot q8|f32] -m MODEL.gguf [...] -m MODEL.gguf ... -f TEXT [-c 2048] "
                         "[-t 8] [--chunks N] [--pairs 0-1,0-2,...] [-o OUT.json]\n"
                         "  -fa and -dot apply to the models that follow them (defaults: auto, and the S1_32 path that\n"
                         "  GGML_S1_32_DOT selects); every pair (i < j) is compared as KL(model i || model j) unless\n"
                         "  --pairs lists the pairs\n", argv0);
}

struct pair_acc {
    int i = 0, j = 0;
    set_stats protocol, pos0, all;
    std::vector<double> window_protocol_mean;
    std::vector<float>  window_pos0;
    std::vector<double> pos_sum; // per position index, summed over windows
};

int main(int argc, char ** argv) {
    std::vector<std::string> paths;
    std::vector<llama_flash_attn_type> fa_of;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::vector<int> dot_of;  // -1: as the environment says, 0: q8, 1: f32
    int dot = -1;
    std::vector<std::pair<int, int>> pair_list;
    std::string path_text, path_out;
    int n_ctx = 2048, n_threads = 8, n_chunks_max = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--candidate") { paths.push_back(next()); fa_of.push_back(fa); dot_of.push_back(dot); }
        else if (a == "-dot") {
            const std::string v = next();
            dot = v == "f32" ? 1 : v == "q8" ? 0 : -1;
        }
        else if (a == "-fa") {
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        else if (a == "--pairs") {
            std::stringstream ps(next());
            std::string item;
            while (std::getline(ps, item, ',')) {
                int pi = -1, pj = -1;
                if (std::sscanf(item.c_str(), "%d-%d", &pi, &pj) != 2 || pi < 0 || pj <= pi) { usage(argv[0]); return 1; }
                pair_list.emplace_back(pi, pj);
            }
        }
        else if (a == "-f")       { path_text = next(); }
        else if (a == "-o")       { path_out = next(); }
        else if (a == "-c")       { n_ctx = std::atoi(next()); }
        else if (a == "-t")       { n_threads = std::atoi(next()); }
        else if (a == "--chunks") { n_chunks_max = std::atoi(next()); }
        else { usage(argv[0]); return 1; }
    }
    if (paths.size() < 2 || path_text.empty() || n_ctx < 4) {
        usage(argv[0]);
        return 1;
    }
    const char * s1_dot = std::getenv("GGML_S1_32_DOT");
    std::printf("S1_32 dot path: %s\n", s1_dot && std::strcmp(s1_dot, "f32") == 0 ? "f32 (float reference)" : "q8_0 (default)");

    llama_backend_init();
    const bool dot_env = ggml_cpu_get_s1_32_dot_f32();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;

    const size_t n_models = paths.size();
    std::vector<llama_model *> models(n_models, nullptr);
    std::vector<llama_context *> ctxs(n_models, nullptr);
    for (size_t m = 0; m < n_models; ++m) {
        cp.flash_attn_type = fa_of[m];
        models[m] = llama_model_load_from_file(paths[m].c_str(), mp);
        ctxs[m]   = models[m] ? llama_init_from_model(models[m], cp) : nullptr;
        if (!ctxs[m]) {
            std::fprintf(stderr, "failed to load %s\n", paths[m].c_str());
            return 1;
        }
    }
    const llama_vocab * vocab = llama_model_get_vocab(models[0]);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t m = 1; m < n_models; ++m) {
        if (llama_vocab_n_tokens(llama_model_get_vocab(models[m])) != n_vocab) {
            std::fprintf(stderr, "%s has another vocabulary size\n", paths[m].c_str());
            return 1;
        }
    }

    std::ifstream in(path_text, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<llama_token> tokens = common_tokenize(ctxs[0], ss.str(), true);
    int n_chunks = int(tokens.size()/n_ctx);
    if (n_chunks_max >= 0) {
        n_chunks = std::min(n_chunks, n_chunks_max);
    }
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int first = n_ctx/2;
    std::printf("%zu tokens, %d windows of %d, add_bos %d, protocol positions %d..%d, %zu models\n", tokens.size(), n_chunks,
                n_ctx, (int) add_bos, first, n_ctx - 2, n_models);
    for (size_t m = 0; m < n_models; ++m) {
        std::printf("  model %zu: %s (flash attention %s, S1_32 dot %s)\n", m, paths[m].c_str(),
                    llama_flash_attn_type_name(fa_of[m]), (dot_of[m] < 0 ? dot_env : dot_of[m] == 1) ? "f32" : "q8");
    }

    std::vector<common_batch> batches;
    batches.reserve(n_models);
    for (size_t m = 0; m < n_models; ++m) {
        batches.emplace_back(ctxs[m]);
    }
    std::vector<std::vector<float>> logits(n_models);
    std::vector<double> t_model(n_models, 0.0);
    std::vector<pair_acc> pairs;
    if (pair_list.empty()) {
        for (int i = 0; i < (int) n_models; ++i) {
            for (int j = i + 1; j < (int) n_models; ++j) {
                pair_list.emplace_back(i, j);
            }
        }
    }
    for (const auto & ij : pair_list) {
        if (ij.second >= (int) n_models) {
            std::fprintf(stderr, "pair %d-%d names a model that is not loaded\n", ij.first, ij.second);
            return 1;
        }
        pair_acc pa;
        pa.i = ij.first;
        pa.j = ij.second;
        pa.pos_sum.assign(n_ctx - 1, 0.0);
        pairs.push_back(std::move(pa));
    }
    std::vector<pos_result> res(n_ctx - 1);

    for (int w = 0; w < n_chunks; ++w) {
        std::vector<llama_token> win(tokens.begin() + size_t(w)*n_ctx, tokens.begin() + size_t(w + 1)*n_ctx);
        if (add_bos) {
            win[0] = llama_vocab_bos(vocab);
        }
        for (size_t m = 0; m < n_models; ++m) {
            ggml_cpu_set_s1_32_dot_f32(dot_of[m] < 0 ? dot_env : dot_of[m] == 1);
            const auto t0 = std::chrono::steady_clock::now();
            if (!decode_window(ctxs[m], batches[m], win.data(), n_ctx, logits[m])) {
                std::fprintf(stderr, "decode failed: model %zu, window %d\n", m, w);
                return 1;
            }
            t_model[m] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        for (pair_acc & pa : pairs) {
            const float * la = logits[pa.i].data();
            const float * lb = logits[pa.j].data();
            std::vector<std::thread> workers;
            const int n_workers = std::max(1, n_threads);
            for (int k = 0; k < n_workers; ++k) {
                workers.emplace_back([&, k]() {
                    for (int p = k; p < n_ctx - 1; p += n_workers) {
                        res[p] = compare_position(la + size_t(p)*n_vocab, lb + size_t(p)*n_vocab, n_vocab, win[p + 1]);
                    }
                });
            }
            for (auto & t : workers) {
                t.join();
            }
            double wsum = 0.0;
            for (int p = 0; p < n_ctx - 1; ++p) {
                const pos_result & r = res[p];
                pa.pos_sum[p] += r.kld;
                auto add = [&](set_stats & s) {
                    s.kld.push_back(r.kld);
                    s.same_top += r.same_top;
                    s.nll_ref += r.nll_ref;
                    s.nll_cand += r.nll_cand;
                    s.n_nll++;
                };
                add(pa.all);
                if (p == 0) {
                    add(pa.pos0);
                    pa.window_pos0.push_back(r.kld);
                }
                if (p >= first) {
                    add(pa.protocol);
                    wsum += r.kld;
                }
            }
            pa.window_protocol_mean.push_back(wsum/(n_ctx - 1 - first));
        }
        if ((w + 1) % 10 == 0 || w + 1 == n_chunks) {
            std::printf("[%3d/%d]", w + 1, n_chunks);
            for (const pair_acc & pa : pairs) {
                double sum = 0.0;
                for (float v : pa.protocol.kld) {
                    sum += v;
                }
                std::printf("  KL(%d||%d) %.6f", pa.i, pa.j, sum/pa.protocol.kld.size());
            }
            std::printf("  | s/window:");
            for (size_t m = 0; m < n_models; ++m) {
                std::printf(" %.2f", t_model[m]/(w + 1));
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    std::ofstream o;
    if (!path_out.empty()) {
        o.open(path_out);
        o.precision(9);
        o << "{\n  \"text\": \"" << path_text << "\",\n  \"n_ctx\": " << n_ctx << ",\n  \"windows\": " << n_chunks << ",\n";
        o << "  \"s1_32_dot\": \"" << (s1_dot ? s1_dot : "q8_0") << "\",\n  \"threads\": " << n_threads << ",\n  \"models\": [";
        for (size_t m = 0; m < n_models; ++m) {
            o << (m ? ", " : "") << "{\"path\": \"" << paths[m] << "\", \"flash_attn\": \"" << llama_flash_attn_type_name(fa_of[m])
              << "\", \"s1_32_dot\": \"" << ((dot_of[m] < 0 ? dot_env : dot_of[m] == 1) ? "f32" : "q8")
              << "\", \"seconds_per_window\": " << t_model[m]/n_chunks << "}";
        }
        o << "],\n  \"pairs\": [\n";
    }
    for (size_t k = 0; k < pairs.size(); ++k) {
        pair_acc & pa = pairs[k];
        const summary sp = summarize(pa.protocol), s0 = summarize(pa.pos0), sa = summarize(pa.all);
        std::printf("KL(model %d || model %d)\n", pa.i, pa.j);
        print_summary("protocol", sp);
        print_summary("pos0", s0);
        print_summary("all", sa);
        if (o.is_open()) {
            o << "    {\"reference\": " << pa.i << ", \"candidate\": " << pa.j << ", \"protocol\": " << json_summary(sp)
              << ", \"pos0\": " << json_summary(s0) << ", \"all\": " << json_summary(sa) << ",\n     \"window_protocol_mean\": [";
            for (size_t i = 0; i < pa.window_protocol_mean.size(); ++i) {
                o << (i ? ", " : "") << pa.window_protocol_mean[i];
            }
            o << "],\n     \"window_pos0\": [";
            for (size_t i = 0; i < pa.window_pos0.size(); ++i) {
                o << (i ? ", " : "") << pa.window_pos0[i];
            }
            o << "],\n     \"position_mean\": [";
            for (size_t i = 0; i < pa.pos_sum.size(); ++i) {
                o << (i ? ", " : "") << pa.pos_sum[i]/n_chunks;
            }
            o << "]}" << (k + 1 < pairs.size() ? ",\n" : "\n");
        }
    }
    std::printf("seconds per window:");
    for (size_t m = 0; m < n_models; ++m) {
        std::printf(" model %zu %.2f", m, t_model[m]/n_chunks);
    }
    std::printf("\n");
    if (o.is_open()) {
        o << "  ]\n}\n";
    }

    for (size_t m = 0; m < n_models; ++m) {
        llama_free(ctxs[m]);
        llama_model_free(models[m]);
    }
    llama_backend_free();
    return 0;
}
