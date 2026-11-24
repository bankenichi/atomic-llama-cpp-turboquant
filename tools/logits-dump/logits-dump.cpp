// llama-logits-dump: one model's logits at every position of token windows, written raw for checks outside llama.cpp
// (moeq Task 33: the fork's Qwen3.5 graphs against transformers, packed files against dense ones).
//
//   llama-logits-dump -m MODEL.gguf (-f TEXT | --ids IDS.bin) [-c 64] [--chunks N] [-t 8] [-dot q8|f32] [--no-repack]
//                     -o OUT.bin
//
// -f: llama-perplexity's windows (the text tokenized with the vocabulary's special tokens, cut into n_ctx-token chunks,
// the first token of each replaced by BOS when the vocabulary adds BOS). --ids: little-endian int32 token ids, used as
// they are, n_ctx per window. Each window is decoded alone from an empty memory, every position output.
// -dot selects the S1_32 path (q8: Q8_0 activations, f32: the float reference path); default as GGML_S1_32_DOT says.
//
// OUT.bin: "LGT1", then u32 n_ctx, u32 n_vocab, u32 n_windows; per window n_ctx int32 token ids, then n_ctx x n_vocab
// float32 logits (position-major).
#include "common.h"
#include "llama.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL.gguf (-f TEXT | --ids IDS.bin) [-c 64] [--chunks N] [-t 8] [-dot q8|f32] "
                         "[--no-repack] -o OUT.bin\n", argv0);
}

int main(int argc, char ** argv) {
    std::string path_model, path_text, path_ids, path_out;
    int n_ctx = 64, n_threads = 8, n_chunks_max = -1, dot = -1;
    bool repack = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-m")          { path_model = next(); }
        else if (a == "-f")          { path_text = next(); }
        else if (a == "--ids")       { path_ids = next(); }
        else if (a == "-o")          { path_out = next(); }
        else if (a == "-c")          { n_ctx = std::atoi(next()); }
        else if (a == "-t")          { n_threads = std::atoi(next()); }
        else if (a == "--chunks")    { n_chunks_max = std::atoi(next()); }
        else if (a == "--no-repack") { repack = false; }
        else if (a == "-dot") {
            const std::string v = next();
            dot = v == "f32" ? 1 : v == "q8" ? 0 : -1;
        }
        else { usage(argv[0]); return 1; }
    }
    if (path_model.empty() || path_out.empty() || path_text.empty() == path_ids.empty() || n_ctx < 1) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();
    if (dot >= 0) {
        ggml_cpu_set_s1_32_dot_f32(dot == 1);
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
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create a context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens;
    const bool from_text = !path_text.empty();
    if (from_text) {
        std::ifstream in(path_text, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        tokens = common_tokenize(ctx, ss.str(), true);
    } else {
        std::ifstream in(path_ids, std::ios::binary | std::ios::ate);
        const std::streamsize n = in.tellg();
        in.seekg(0);
        tokens.resize(size_t(n)/sizeof(int32_t));
        in.read((char *) tokens.data(), tokens.size()*sizeof(int32_t));
    }
    int n_chunks = int(tokens.size()/n_ctx);
    if (n_chunks_max >= 0) {
        n_chunks = std::min(n_chunks, n_chunks_max);
    }
    const bool add_bos = from_text && llama_vocab_get_add_bos(vocab);
    std::printf("%s: %zu tokens, %d windows of %d, n_vocab %d, S1_32 dot %s\n", path_model.c_str(), tokens.size(), n_chunks,
                n_ctx, n_vocab, ggml_cpu_get_s1_32_dot_f32() ? "f32" : "q8");

    FILE * out = std::fopen(path_out.c_str(), "wb");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path_out.c_str());
        return 1;
    }
    const uint32_t header[3] = { (uint32_t) n_ctx, (uint32_t) n_vocab, (uint32_t) n_chunks };
    std::fwrite("LGT1", 1, 4, out);
    std::fwrite(header, sizeof(uint32_t), 3, out);

    common_batch batch(ctx);
    for (int w = 0; w < n_chunks; ++w) {
        std::vector<llama_token> win(tokens.begin() + size_t(w)*n_ctx, tokens.begin() + size_t(w + 1)*n_ctx);
        if (add_bos) {
            win[0] = llama_vocab_bos(vocab);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.clear();
        for (int p = 0; p < n_ctx; ++p) {
            batch.add(win[p], p, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            std::fprintf(stderr, "decode failed at window %d\n", w);
            return 1;
        }
        std::fwrite(win.data(), sizeof(int32_t), n_ctx, out);
        for (int p = 0; p < n_ctx; ++p) {
            std::fwrite(llama_get_logits_ith(ctx, p), sizeof(float), n_vocab, out);
        }
    }
    std::fclose(out);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
