// Qwen3.5 with the merged qkv+gate projection: files written by the converter since upstream #32102 hold
// blk.N.attn_qkv_gate (attn_qkv's rows, then attn_gate's) for the Gated DeltaNet layers. Such a file must load and give
// the logits of the same model with separate attn_qkv and attn_gate, for a batch of several tokens (the gate z is then
// a strided view of the merged projection; the gated norm used to assert on it).
//
//   test-qwen35-merged-qkv-gate MODEL_DIR
//
// MODEL_DIR holds test-llama-archs' random qwen35-dense.gguf (-o). Two files are written from it: "separate" (the
// layers' per-tensor scales of attn_qkv and attn_gate dropped, which the merged path has no place for) and "merged"
// (the same with each layer's attn_qkv and attn_gate concatenated into attn_qkv_gate). Both are decoded on the CPU on
// 24 tokens at once; the logits must agree to float rounding.
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool starts_with(const std::string & s, const std::string & p) { return s.rfind(p, 0) == 0; }
static bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// blk.N.attn_qkv.* or blk.N.attn_gate.* -> N, else -1
static int qkv_gate_layer(const std::string & name, std::string & part) {
    if (!starts_with(name, "blk.")) {
        return -1;
    }
    const size_t dot = name.find('.', 4);
    const std::string rest = name.substr(dot + 1);
    for (const char * p : { "attn_qkv.", "attn_gate." }) {
        if (starts_with(rest, p)) {
            part = rest;
            return std::stoi(name.substr(4, dot - 4));
        }
    }
    return -1;
}

static void write_file(gguf_context * src, ggml_context * data, bool merged, const std::string & path,
                       std::vector<ggml_context *> & owned, int * n_merged) {
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    *n_merged = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(src); ++i) {
        const std::string name = gguf_get_tensor_name(src, i);
        std::string part;
        const int il = qkv_gate_layer(name, part);
        if (il < 0) {
            gguf_add_tensor(dst, ggml_get_tensor(data, name.c_str()));
            continue;
        }
        if (!ends_with(part, ".weight")) {
            continue;                                      // per-tensor scales: dropped in both files
        }
        if (!merged) {
            gguf_add_tensor(dst, ggml_get_tensor(data, name.c_str()));
            continue;
        }
        if (part != "attn_qkv.weight") {
            continue;                                      // the gate goes into the merged tensor
        }
        const std::string blk = "blk." + std::to_string(il) + ".";
        ggml_tensor * qkv  = ggml_get_tensor(data, (blk + "attn_qkv.weight").c_str());
        ggml_tensor * gate = ggml_get_tensor(data, (blk + "attn_gate.weight").c_str());
        if (!gate || qkv->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || qkv->ne[0] != gate->ne[0]) {
            throw std::runtime_error("layer " + std::to_string(il) + ": cannot merge attn_qkv and attn_gate");
        }
        ggml_init_params p = { ggml_nbytes(qkv) + ggml_nbytes(gate) + ggml_tensor_overhead(), nullptr, false };
        ggml_context * c = ggml_init(p);
        owned.push_back(c);
        ggml_tensor * m = ggml_new_tensor_2d(c, GGML_TYPE_F32, qkv->ne[0], qkv->ne[1] + gate->ne[1]);
        ggml_set_name(m, (blk + "attn_qkv_gate.weight").c_str());
        std::memcpy(m->data, qkv->data, ggml_nbytes(qkv));
        std::memcpy((char *) m->data + ggml_nbytes(qkv), gate->data, ggml_nbytes(gate));
        gguf_add_tensor(dst, m);
        ++*n_merged;
    }
    if (!gguf_write_to_file(dst, path.c_str(), false)) {
        throw std::runtime_error("cannot write " + path);
    }
    gguf_free(dst);
}

static std::vector<float> logits_of(const std::string & path, const std::vector<llama_token> & tokens) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        throw std::runtime_error("cannot load " + path);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        throw std::runtime_error("cannot create a context for " + path);
    }
    std::vector<float> out;
    {
        common_batch batch(ctx);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.add(tokens[i], (llama_pos) i, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("decode failed for " + path);
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (size_t i = 0; i < tokens.size(); ++i) {
            const float * l = llama_get_logits_ith(ctx, (int32_t) i);
            out.insert(out.end(), l, l + n_vocab);
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR (the models of test-llama-archs -o)\n", argv[0]);
        return 1;
    }
    const fs::path src_path = fs::path(argv[1]) / "qwen35-dense.gguf";
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
    }, nullptr);

    ggml_context * data = nullptr;
    gguf_init_params ip = { false, &data };
    gguf_context * src = gguf_init_from_file(src_path.string().c_str(), ip);
    if (!src) {
        std::fprintf(stderr, "cannot read %s (run test-llama-archs -a \"^qwen35$\" -o DIR)\n", src_path.string().c_str());
        return 1;
    }
    std::vector<ggml_context *> owned;
    const fs::path tmp = fs::temp_directory_path();
    const std::string p_sep = (tmp / "test-qwen35-separate.gguf").string();
    const std::string p_mrg = (tmp / "test-qwen35-merged.gguf").string();
    int n_sep = 0, n_mrg = 0;
    write_file(src, data, false, p_sep, owned, &n_sep);
    write_file(src, data, true,  p_mrg, owned, &n_mrg);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> tok(0, 127);
    std::vector<llama_token> tokens;
    for (int i = 0; i < 24; ++i) {
        tokens.push_back(tok(rng));
    }
    const std::vector<float> a = logits_of(p_sep, tokens);
    const std::vector<float> b = logits_of(p_mrg, tokens);
    double d = 0.0, n = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += (double(a[i]) - b[i])*(double(a[i]) - b[i]);
        n += double(a[i])*a[i];
    }
    const double nmse = d/n;
    const bool pass = n_mrg > 0 && std::isfinite(nmse) && nmse < 1e-10;
    std::printf("%s: %d Gated DeltaNet layers merged into attn_qkv_gate; 24-token batch, NMSE of the logits merged vs "
                "separate %.3e\n", pass ? "ok" : "FAIL", n_mrg, nmse);

    for (auto * c : owned) { ggml_free(c); }
    gguf_free(src);
    ggml_free(data);
    llama_backend_free();
    return pass ? 0 : 1;
}
