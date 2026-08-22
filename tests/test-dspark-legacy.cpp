#include "ggml.h"
#include "gguf.h"
#include "speculative.h"
#include "../src/llama-memory-recurrent.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

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

static void test_draft_window() {
    constexpr llama_pos capacity = 4096;
    constexpr llama_pos reserve = 4;
    constexpr llama_pos chunk = 512;

    common_speculative_draft_window_state state;
    common_speculative_draft_window_plan plan;
    require(common_speculative_draft_window_plan_make(
                state, 0, 100, capacity, reserve, chunk, plan),
            "initial draft window plan failed");
    require(!plan.reset && plan.discard == 0 && plan.local_begin == 0 && plan.local_end == 100,
            "initial draft window mapping is wrong");

    state.next = capacity - reserve;
    require(common_speculative_draft_window_plan_make(
                state, state.next, 1, capacity, reserve, chunk, plan),
            "boundary rollover plan failed");
    require(plan.discard == chunk && plan.origin == chunk &&
                plan.local_begin == capacity - reserve - chunk,
            "boundary rollover did not discard one chunk");

    state.origin = plan.origin;
    state.next = plan.local_end;
    const llama_pos absolute_next = state.origin + state.next;
    require(common_speculative_draft_window_plan_make(
                state, absolute_next, chunk, capacity, reserve, chunk, plan),
            "repeated rollover plan failed");
    require(plan.discard == chunk && plan.origin == 2*chunk,
            "repeated rollover did not advance the target origin");

    state = { 1024, 3000 };
    require(common_speculative_draft_window_plan_make(
                state, 2000, 10, capacity, reserve, chunk, plan),
            "in-window prefix rollback failed");
    require(plan.truncate_from == 976 && plan.local_begin == 976 && plan.local_end == 986,
            "in-window prefix rollback mapped the wrong local boundary");
    require(!common_speculative_draft_window_plan_make(
                state, 1000, 1, capacity, reserve, chunk, plan),
            "prefix older than retained window was accepted");
    require(!common_speculative_draft_window_plan_make(
                state, state.origin + state.next + 1, 1, capacity, reserve, chunk, plan),
            "gap after committed draft state was accepted");

    require(common_speculative_draft_window_plan_make(
                state, 0, 1, capacity, reserve, chunk, plan) && plan.reset,
            "position-zero rebuild did not reset the private window");

    state = { 1536, 2048 };
    const auto data = common_speculative_draft_window_state_serialize(state);
    common_speculative_draft_window_state restored;
    require(common_speculative_draft_window_state_restore(data, capacity, reserve, restored),
            "valid private-window state did not restore");
    require(restored.origin == state.origin && restored.next == state.next,
            "restored private-window state differs from saved state");
    require(common_speculative_draft_window_matches_positions(restored, 0, restored.next - 1),
            "matching restored KV bounds were rejected");
    require(!common_speculative_draft_window_matches_positions(restored, -1, -1),
            "missing KV was accepted for a non-empty restored mapping");
    require(!common_speculative_draft_window_matches_positions(restored, 0, restored.next - 2),
            "truncated KV was accepted for a complete restored mapping");
    require(common_speculative_draft_window_matches_positions({}, -1, -1),
            "empty mapping did not accept an empty KV sequence");

    auto bad = data;
    bad[0] ^= 0xff;
    require(!common_speculative_draft_window_state_restore(bad, capacity, reserve, restored),
            "invalid private-window state magic was accepted");
    require(restored.origin == 0 && restored.next == 0,
            "invalid private-window state did not clear the destination");

    constexpr llama_pos tiny_capacity = 9;
    constexpr llama_pos tiny_reserve = 4;
    state = { 0, tiny_capacity - tiny_reserve };
    require(common_speculative_draft_window_plan_make(
                state, state.next, tiny_capacity - tiny_reserve,
                tiny_capacity, tiny_reserve, 2, plan),
            "full verification-sized span could not reserve one rollover");
    require(plan.discard == tiny_capacity - tiny_reserve && plan.local_begin == 0,
            "full verification-sized span retained rows that would be discarded mid-verification");
}

static void test_recurrent_position_diagnostic() {
    require(llama_recurrent_has_nonconsecutive_scalar_positions(1, 10, 14, 2),
            "scalar position gap was not detected");
    require(!llama_recurrent_has_nonconsecutive_scalar_positions(1, 10, 12, 2),
            "consecutive scalar positions were rejected");
    require(!llama_recurrent_has_nonconsecutive_scalar_positions(4, 10, 10, 512),
            "M-RoPE temporal repetition triggered the scalar diagnostic");
    require(!llama_recurrent_has_nonconsecutive_scalar_positions(4, 10, 42, 512),
            "M-RoPE image-boundary jump triggered the scalar diagnostic");
}

int main() {
    expect_dspark("dflash", "markov_w1.weight");
    expect_dspark("dspark", "dspark.markov_head_a.weight");
    test_draft_window();
    test_recurrent_position_diagnostic();
    std::puts("DSpark schema, private-window, and M-RoPE diagnostics passed");
    return 0;
}
