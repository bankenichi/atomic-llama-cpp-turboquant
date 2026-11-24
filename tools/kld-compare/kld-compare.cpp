// llama-kld-compare: the KL divergence between models at every position of the llama-perplexity windows, computed
// exactly from the models' logits in one process.
//
//   llama-kld-compare [per-model options] -m A.gguf [per-model options] -m B.gguf [...] -f TEXT [-c 2048] [-t 8]
//                     [--chunks N] [--positions P] [--head H] [--pairs 0-1,...] [--capture REGEX]
//                     [--capture-windows N] [-o OUT.json]
//
// Every pair (i < j) is compared as KL(model i || model j); each model decodes each window once.
//
// Written for checking moeq's packed experts (S1_32) against the dense export of the same student and one backend
// against another: llama-perplexity scores only the second half of each window and stores the reference as 16-bit
// log-probabilities, while this tool also gives position 0 (the sink position) and every other position, at full
// precision, and can compare the models' intermediate tensors.
//
// Windows are llama-perplexity's: the text tokenized with the vocabulary's special tokens, cut into n_ctx-token
// chunks, the first token of each replaced by BOS when the vocabulary adds BOS. The logits at position p predict
// token p+1; positions 0 .. n_ctx-2 are compared, KL per position in double precision. With --positions P only the
// first P tokens of each window are decoded and positions 0 .. P-1 compared (P < n_ctx).
// Reported for four position sets: "protocol" (n_ctx/2 .. n_ctx-2, what llama-perplexity scores; empty with a short
// --positions), "pos0" (one value per window), "head" (positions 0 .. H-1, default 64) and "all"; each with mean,
// median, 99th percentile and maximum (percentiles interpolated as in llama-perplexity), the share of positions whose
// top token agrees, and the perplexity of both models on the set's positions. The JSON output adds the per-window
// protocol means (for bootstrap intervals), the per-window position-0 values, and the mean and maximum per position.
//
// Per-model options apply to the models that follow them:
//   -fa on|off|auto       flash attention (default auto)
//   -dot q8|f32|f32_v|f32_u  the CPU's S1_32 activation precision (default: as GGML_S1_32_DOT says): Q8_0, F32 in
//                         every sign product, F32 in each bank's first product only, F32 in its second only
//   -ngl N                layers on the GPU (default 0: the model and every operation stay on the CPU)
//   -ub N                 micro-batch size (default 512)
// The same file with the same -ngl is loaded once and shared by its contexts.
//
// --capture REGEX records, during the first --capture-windows windows (default 4), every F32 graph tensor whose name
// matches (e.g. "ffn_moe_(gate_up|down)(_xs|_z|_uz)?-") in every model, and compares each pair's tensors of the same
// name: normalized squared error sum((b-a)^2)/sum(a^2) and the largest |b-a| relative to max|a|, summarized per
// tensor kind (the name without its layer number) as median and maximum over layers and windows.
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
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
    std::printf("%-9s n %8zu  mean KLD %.6g  median %.6g  99%% %.6g  max %.6g  same top %.4f  PPL ref %.4f cand %.4f\n",
                name, r.n, r.mean, r.median, r.p99, r.max, r.same_top, r.ppl_ref, r.ppl_cand);
}

static std::string json_num(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream o;
    o.precision(9);
    o << v;
    return o.str();
}

static std::string json_summary(const summary & r) {
    std::ostringstream o;
    o << "{\"n\": " << r.n << ", \"mean\": " << json_num(r.mean) << ", \"median\": " << json_num(r.median)
      << ", \"p99\": " << json_num(r.p99) << ", \"max\": " << json_num(r.max) << ", \"same_top\": " << json_num(r.same_top)
      << ", \"ppl_reference\": " << json_num(r.ppl_ref) << ", \"ppl_candidate\": " << json_num(r.ppl_cand) << "}";
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

// decodes the first n_eval tokens; logits for positions 0 .. n_cmp-1
static bool decode_window(llama_context * ctx, common_batch & batch, const llama_token * tokens, int n_eval, int n_cmp,
                          std::vector<float> & logits) {
    llama_memory_clear(llama_get_memory(ctx), true);
    batch.clear();
    for (int p = 0; p < n_eval; ++p) {
        batch.add(tokens[p], p, 0, p < n_cmp);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * out = llama_get_logits(ctx);
    logits.assign(out, out + size_t(n_cmp)*n_vocab);
    return true;
}

// the intermediate tensors one model computed for one window
struct capture_state {
    const std::regex * re = nullptr;
    bool active = false;
    std::map<std::string, std::vector<float>> tensors;
};

static bool capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    capture_state * st = (capture_state *) user_data;
    if (ask) {
        return st->active && t->type == GGML_TYPE_F32 && std::regex_search(t->name, *st->re);
    }
    if (!st->active || t->type != GGML_TYPE_F32 || !std::regex_search(t->name, *st->re)) {
        return true;
    }
    std::vector<float> & dst = st->tensors[t->name];
    dst.resize(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    } else {
        // gather row by row
        size_t k = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    const size_t off = i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
                    if (t->nb[0] == sizeof(float)) {
                        ggml_backend_tensor_get(t, dst.data() + k, off, t->ne[0]*sizeof(float));
                        k += t->ne[0];
                    } else {
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                            ggml_backend_tensor_get(t, dst.data() + k++, off + i0*t->nb[0], sizeof(float));
                        }
                    }
                }
            }
        }
    }
    return true;
}

struct capture_acc {
    std::vector<float> nmse, maxrel;
    std::string worst;
    float worst_nmse = -1.0f;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [-fa on|off|auto] [-dot q8|f32|f32_v|f32_u] [-ngl N] [-ub N] -m MODEL.gguf [...] -m MODEL.gguf ...\n"
        "          -f TEXT [-c 2048] [-t 8] [--chunks N] [--positions P] [--head H] [--pairs 0-1,0-2,...]\n"
        "          [--capture REGEX] [--capture-windows N] [-o OUT.json]\n"
        "  -fa, -dot, -ngl and -ub apply to the models that follow them; every pair (i < j) is compared as\n"
        "  KL(model i || model j) unless --pairs lists the pairs\n", argv0);
}

struct pair_acc {
    int i = 0, j = 0;
    set_stats protocol, pos0, head, all;
    std::vector<double> window_protocol_mean;
    std::vector<float>  window_pos0;
    std::vector<double> pos_sum; // per position index, summed over windows
    std::vector<float>  pos_max; // per position index, the largest over windows
    std::map<std::string, capture_acc> captures;
};

struct model_spec {
    std::string path;
    llama_flash_attn_type fa;
    int dot;     // -1: as the environment says, else the CPU's S1_32 mode
    int ngl;
    int ubatch;
};

static const char * dot_name(int mode) {
    switch (mode) {
        case 0:  return "q8";
        case 1:  return "f32";
        case 2:  return "f32_v";
        case 3:  return "f32_u";
        default: return "?";
    }
}

typedef void (*s1_32_set_mode_t)(int);
typedef int  (*s1_32_get_mode_t)(void);

int main(int argc, char ** argv) {
    std::vector<model_spec> specs;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    int dot = -1, ngl = 0, ubatch = 512;
    std::vector<std::pair<int, int>> pair_list;
    std::string path_text, path_out, capture_re;
    int n_ctx = 2048, n_threads = 8, n_chunks_max = -1, n_positions = -1, n_head = 64, capture_windows = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--candidate") { specs.push_back({ next(), fa, dot, ngl, ubatch }); }
        else if (a == "-dot") {
            const std::string v = next();
            dot = v == "q8" ? 0 : v == "f32" ? 1 : v == "f32_v" ? 2 : v == "f32_u" ? 3 : -2;
            if (dot == -2) { usage(argv[0]); return 1; }
        }
        else if (a == "-fa") {
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
        else if (a == "-ngl")     { ngl = std::atoi(next()); }
        else if (a == "-ub")      { ubatch = std::atoi(next()); }
        else if (a == "--pairs") {
            std::stringstream ps(next());
            std::string item;
            while (std::getline(ps, item, ',')) {
                int pi = -1, pj = -1;
                if (std::sscanf(item.c_str(), "%d-%d", &pi, &pj) != 2 || pi < 0 || pj <= pi) { usage(argv[0]); return 1; }
                pair_list.emplace_back(pi, pj);
            }
        }
        else if (a == "-f")                { path_text = next(); }
        else if (a == "-o")                { path_out = next(); }
        else if (a == "-c")                { n_ctx = std::atoi(next()); }
        else if (a == "-t")                { n_threads = std::atoi(next()); }
        else if (a == "--chunks")          { n_chunks_max = std::atoi(next()); }
        else if (a == "--positions")       { n_positions = std::atoi(next()); }
        else if (a == "--head")            { n_head = std::atoi(next()); }
        else if (a == "--capture")         { capture_re = next(); }
        else if (a == "--capture-windows") { capture_windows = std::atoi(next()); }
        else { usage(argv[0]); return 1; }
    }
    if (specs.size() < 2 || path_text.empty() || n_ctx < 4 || ubatch < 1) {
        usage(argv[0]);
        return 1;
    }
    // tokens decoded per window and positions compared
    const int n_eval = n_positions > 0 && n_positions < n_ctx ? n_positions : n_ctx;
    const int n_cmp  = n_eval < n_ctx ? n_eval : n_ctx - 1;
    n_head = std::max(1, std::min(n_head, n_cmp));

    llama_backend_init();

    // the CPU's S1_32 switch is fork-private: reached through the CPU backend's proc address
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    auto set_mode = cpu_reg ? (s1_32_set_mode_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_s1_32_dot_mode") : nullptr;
    auto get_mode = cpu_reg ? (s1_32_get_mode_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_get_s1_32_dot_mode") : nullptr;
    if (!set_mode || !get_mode) {
        std::fprintf(stderr, "the CPU backend does not offer ggml_backend_cpu_{set,get}_s1_32_dot_mode\n");
        return 1;
    }
    const int env_mode = get_mode();
    std::printf("S1_32 CPU dot path from the environment: %s\n", dot_name(env_mode));

    std::regex re;
    if (!capture_re.empty()) {
        re = std::regex(capture_re);
    }

    const size_t n_models = specs.size();
    std::vector<llama_model *> models(n_models, nullptr);
    std::vector<bool> owns_model(n_models, false);
    std::vector<llama_context *> ctxs(n_models, nullptr);
    std::vector<capture_state> captures(n_models);
    ggml_backend_dev_t no_devices[] = { nullptr };
    for (size_t m = 0; m < n_models; ++m) {
        const model_spec & s = specs[m];
        for (size_t k = 0; k < m; ++k) {
            if (specs[k].path == s.path && specs[k].ngl == s.ngl) {
                models[m] = models[k];
                break;
            }
        }
        if (!models[m]) {
            llama_model_params mp = llama_model_default_params();
            mp.n_gpu_layers = s.ngl;
            if (s.ngl == 0) {
                mp.devices = no_devices; // no GPU device at all: weights and every operation on the CPU
            }
            models[m] = llama_model_load_from_file(s.path.c_str(), mp);
            owns_model[m] = true;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_ctx;
        cp.n_ubatch        = std::min(s.ubatch, n_ctx);
        cp.n_seq_max       = 1;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.flash_attn_type = s.fa;
        if (s.ngl == 0) {
            cp.op_offload = false;
        }
        if (!capture_re.empty()) {
            captures[m].re      = &re;
            cp.cb_eval           = capture_cb;
            cp.cb_eval_user_data = &captures[m];
        }
        ctxs[m] = models[m] ? llama_init_from_model(models[m], cp) : nullptr;
        if (!ctxs[m]) {
            std::fprintf(stderr, "failed to load %s\n", s.path.c_str());
            return 1;
        }
    }
    const llama_vocab * vocab = llama_model_get_vocab(models[0]);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t m = 1; m < n_models; ++m) {
        if (llama_vocab_n_tokens(llama_model_get_vocab(models[m])) != n_vocab) {
            std::fprintf(stderr, "%s has another vocabulary size\n", specs[m].path.c_str());
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
    std::printf("%zu tokens, %d windows of %d, add_bos %d, %d tokens decoded per window, positions 0..%d compared "
                "(protocol %d..%d, head 0..%d), %zu models\n", tokens.size(), n_chunks, n_ctx, (int) add_bos, n_eval,
                n_cmp - 1, first, n_ctx - 2, n_head - 1, n_models);
    for (size_t m = 0; m < n_models; ++m) {
        const model_spec & s = specs[m];
        std::printf("  model %zu: %s (flash attention %s, S1_32 CPU dot %s, %d GPU layers, ubatch %d%s)\n", m, s.path.c_str(),
                    llama_flash_attn_type_name(s.fa), dot_name(s.dot < 0 ? env_mode : s.dot), s.ngl, std::min(s.ubatch, n_ctx),
                    owns_model[m] ? "" : ", shares the model loaded above");
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
        pa.pos_sum.assign(n_cmp, 0.0);
        pa.pos_max.assign(n_cmp, 0.0f);
        pairs.push_back(std::move(pa));
    }
    std::vector<pos_result> res(n_cmp);

    for (int w = 0; w < n_chunks; ++w) {
        std::vector<llama_token> win(tokens.begin() + size_t(w)*n_ctx, tokens.begin() + size_t(w + 1)*n_ctx);
        if (add_bos) {
            win[0] = llama_vocab_bos(vocab);
        }
        const bool capturing = !capture_re.empty() && w < capture_windows;
        for (size_t m = 0; m < n_models; ++m) {
            set_mode(specs[m].dot < 0 ? env_mode : specs[m].dot);
            captures[m].active = capturing;
            captures[m].tensors.clear();
            const auto t0 = std::chrono::steady_clock::now();
            if (!decode_window(ctxs[m], batches[m], win.data(), n_eval, n_cmp, logits[m])) {
                std::fprintf(stderr, "decode failed: model %zu, window %d\n", m, w);
                return 1;
            }
            t_model[m] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            captures[m].active = false;
        }
        for (pair_acc & pa : pairs) {
            const float * la = logits[pa.i].data();
            const float * lb = logits[pa.j].data();
            std::vector<std::thread> workers;
            const int n_workers = std::max(1, n_threads);
            for (int k = 0; k < n_workers; ++k) {
                workers.emplace_back([&, k]() {
                    for (int p = k; p < n_cmp; p += n_workers) {
                        res[p] = compare_position(la + size_t(p)*n_vocab, lb + size_t(p)*n_vocab, n_vocab, win[p + 1]);
                    }
                });
            }
            for (auto & t : workers) {
                t.join();
            }
            double wsum = 0.0;
            int wn = 0;
            for (int p = 0; p < n_cmp; ++p) {
                const pos_result & r = res[p];
                pa.pos_sum[p] += r.kld;
                pa.pos_max[p] = std::max(pa.pos_max[p], r.kld);
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
                if (p < n_head) {
                    add(pa.head);
                }
                if (p >= first) {
                    add(pa.protocol);
                    wsum += r.kld;
                    wn++;
                }
            }
            if (wn > 0) {
                pa.window_protocol_mean.push_back(wsum/wn);
            }
            if (capturing) {
                const auto & ta = captures[pa.i].tensors;
                const auto & tb = captures[pa.j].tensors;
                for (const auto & kv : ta) {
                    auto it = tb.find(kv.first);
                    if (it == tb.end() || it->second.size() != kv.second.size()) {
                        continue;
                    }
                    const std::vector<float> & a = kv.second;
                    const std::vector<float> & b = it->second;
                    double num = 0.0, den = 0.0, maxd = 0.0, maxa = 0.0;
                    for (size_t q = 0; q < a.size(); ++q) {
                        const double d = (double) b[q] - a[q];
                        num += d*d;
                        den += (double) a[q]*a[q];
                        maxd = std::max(maxd, std::fabs(d));
                        maxa = std::max(maxa, std::fabs((double) a[q]));
                    }
                    const float e = (float) (den > 0 ? num/den : num);
                    const float mr = (float) (maxa > 0 ? maxd/maxa : maxd);
                    const size_t dash = kv.first.rfind('-');
                    const std::string kind = dash == std::string::npos ? kv.first : kv.first.substr(0, dash);
                    capture_acc & acc = pa.captures[kind];
                    acc.nmse.push_back(std::isfinite(e) ? e : INFINITY);
                    acc.maxrel.push_back(std::isfinite(mr) ? mr : INFINITY);
                    if (!(e <= acc.worst_nmse)) {
                        acc.worst_nmse = e;
                        acc.worst = kv.first + " window " + std::to_string(w);
                    }
                }
            }
        }
        if ((w + 1) % 10 == 0 || w + 1 == n_chunks) {
            std::printf("[%3d/%d]", w + 1, n_chunks);
            for (const pair_acc & pa : pairs) {
                const set_stats & s = pa.protocol.kld.empty() ? pa.all : pa.protocol;
                double sum = 0.0;
                for (float v : s.kld) {
                    sum += v;
                }
                std::printf("  KL(%d||%d) %.6g", pa.i, pa.j, sum/s.kld.size());
            }
            std::printf("  | s/window:");
            for (size_t m = 0; m < n_models; ++m) {
                std::printf(" %.3f", t_model[m]/(w + 1));
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    std::ofstream o;
    if (!path_out.empty()) {
        o.open(path_out);
        o.precision(9);
        o << "{\n  \"text\": \"" << path_text << "\",\n  \"n_ctx\": " << n_ctx << ",\n  \"windows\": " << n_chunks
          << ",\n  \"tokens_per_window\": " << n_eval << ",\n  \"positions_compared\": " << n_cmp << ",\n  \"head\": " << n_head << ",\n";
        o << "  \"s1_32_dot\": \"" << dot_name(env_mode) << "\",\n  \"threads\": " << n_threads << ",\n  \"models\": [";
        for (size_t m = 0; m < n_models; ++m) {
            const model_spec & s = specs[m];
            o << (m ? ", " : "") << "{\"path\": \"" << s.path << "\", \"flash_attn\": \"" << llama_flash_attn_type_name(s.fa)
              << "\", \"s1_32_dot\": \"" << dot_name(s.dot < 0 ? env_mode : s.dot) << "\", \"n_gpu_layers\": " << s.ngl
              << ", \"ubatch\": " << std::min(s.ubatch, n_ctx) << ", \"seconds_per_window\": " << t_model[m]/n_chunks << "}";
        }
        o << "],\n  \"pairs\": [\n";
    }
    for (size_t k = 0; k < pairs.size(); ++k) {
        pair_acc & pa = pairs[k];
        const summary sp = summarize(pa.protocol), s0 = summarize(pa.pos0), sh = summarize(pa.head), sa = summarize(pa.all);
        std::printf("KL(model %d || model %d)\n", pa.i, pa.j);
        if (sp.n) {
            print_summary("protocol", sp);
        }
        print_summary("pos0", s0);
        print_summary("head", sh);
        print_summary("all", sa);
        std::vector<std::string> cap_json;
        for (auto & kv : pa.captures) {
            capture_acc & acc = kv.second;
            std::vector<float> e = acc.nmse, mr = acc.maxrel;
            std::sort(e.begin(), e.end());
            std::sort(mr.begin(), mr.end());
            std::printf("  tensor %-26s n %4zu  nmse median %.3e max %.3e  max|d|/max|a| median %.3e max %.3e  worst %s\n",
                        kv.first.c_str(), e.size(), percentile(e, 0.5f), e.back(), percentile(mr, 0.5f), mr.back(), acc.worst.c_str());
            std::ostringstream c;
            c << "{\"kind\": \"" << kv.first << "\", \"n\": " << e.size() << ", \"nmse_median\": " << json_num(percentile(e, 0.5f))
              << ", \"nmse_max\": " << json_num(e.back()) << ", \"maxrel_median\": " << json_num(percentile(mr, 0.5f))
              << ", \"maxrel_max\": " << json_num(mr.back()) << ", \"worst\": \"" << acc.worst << "\"}";
            cap_json.push_back(c.str());
        }
        if (o.is_open()) {
            o << "    {\"reference\": " << pa.i << ", \"candidate\": " << pa.j << ", \"protocol\": " << json_summary(sp)
              << ", \"pos0\": " << json_summary(s0) << ", \"head\": " << json_summary(sh) << ", \"all\": " << json_summary(sa)
              << ",\n     \"window_protocol_mean\": [";
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
            o << "],\n     \"position_max\": [";
            for (size_t i = 0; i < pa.pos_max.size(); ++i) {
                o << (i ? ", " : "") << pa.pos_max[i];
            }
            o << "],\n     \"captures\": [";
            for (size_t i = 0; i < cap_json.size(); ++i) {
                o << (i ? ",\n       " : "\n       ") << cap_json[i];
            }
            o << "]}" << (k + 1 < pairs.size() ? ",\n" : "\n");
        }
    }
    std::printf("seconds per window:");
    for (size_t m = 0; m < n_models; ++m) {
        std::printf(" model %zu %.3f", m, t_model[m]/n_chunks);
    }
    std::printf("\n");
    if (o.is_open()) {
        o << "  ]\n}\n";
    }

    for (size_t m = 0; m < n_models; ++m) {
        llama_free(ctxs[m]);
    }
    for (size_t m = 0; m < n_models; ++m) {
        if (owns_model[m]) {
            llama_model_free(models[m]);
        }
    }
    llama_backend_free();
    return 0;
}
