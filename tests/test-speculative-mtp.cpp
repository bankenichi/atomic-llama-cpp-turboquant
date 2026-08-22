#include "speculative.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Optional current-API smoke test for external Gemma assistant and combined
// Qwen NextN GGUF metadata. Exact end-to-end server acceptance lives in the
// convergence procedure because those model files are intentionally not in CI.

static bool has_mtp(const char * path) {
    const auto types = common_speculative_types_from_gguf(path);
    return std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end();
}

int main() {
    const char * assistant = std::getenv("LLAMA_MTP_TEST_HEAD");
    const char * combined  = std::getenv("LLAMA_MTP_TEST_COMBINED");

    if ((!assistant || std::strlen(assistant) == 0) && (!combined || std::strlen(combined) == 0)) {
        std::cout << "skip: set LLAMA_MTP_TEST_HEAD and/or LLAMA_MTP_TEST_COMBINED\n";
        return 0;
    }

    if (assistant && std::strlen(assistant) > 0 && !has_mtp(assistant)) {
        std::cerr << "assistant GGUF was not inferred as draft-mtp\n";
        return 1;
    }
    if (combined && std::strlen(combined) > 0 && !has_mtp(combined)) {
        std::cerr << "combined GGUF was not inferred as draft-mtp\n";
        return 1;
    }

    return 0;
}
