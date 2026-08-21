#include "ggml.h"
#include "gguf.h"
#include "speculative.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

static std::filesystem::path write_probe(const char * arch, const char * markov_name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
            ("llama-dspark-legacy-" + std::to_string(stamp) + ".gguf");

    ggml_init_params params = {
        /* .mem_size   = */ 4*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * tensor_ctx = ggml_init(params);
    if (!tensor_ctx) {
        throw std::runtime_error("failed to create tensor context");
    }

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "general.architecture", arch);

    ggml_tensor * markov = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 1);
    ggml_set_name(markov, markov_name);
    gguf_add_tensor(gguf_ctx, markov);
    gguf_write_to_file(gguf_ctx, path.string().c_str(), false);

    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
    return path;
}

static void expect_dspark(const char * arch, const char * markov_name) {
    const auto path = write_probe(arch, markov_name);
    try {
        const auto types = common_speculative_types_from_gguf(path.string());
        if (types.size() != 1 || types[0] != COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) {
            throw std::runtime_error(std::string("failed to detect DSpark schema for architecture '") + arch + "'");
        }
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
    std::filesystem::remove(path);
}

int main() {
    expect_dspark("dflash", "markov_w1.weight");
    expect_dspark("dspark", "dspark.markov_head_a.weight");
    std::puts("DSpark GGUF schema detection passed");
    return 0;
}
