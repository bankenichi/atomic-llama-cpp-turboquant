#include "server-spec-state.h"
#include "server-task.h"

#include <cassert>

int main() {
    server_spec_context_state state;

    // A fresh slot has no proven common target/draft prefix.
    assert(state.begin_request(false));
    state.mark_synchronized();
    assert(!state.begin_request(false));

    // A media request invalidates the draft while allowing target reuse.
    assert(!state.begin_request(true));
    assert(state.begin_request(false));

    // After a forced full text rebuild, cache reuse is safe again.
    state.mark_synchronized();
    assert(!state.begin_request(false));

    state.invalidate();
    assert(state.begin_request(false));

    // A secondary-cache miss is a no-op: it cannot erase media invalidation.
    server_prompt_cache empty_cache(1, 0);
    server_prompt current;
    current.tokens = server_tokens(llama_tokens { 1, 2 }, false);
    server_tokens unrelated(llama_tokens { 9, 10 }, false);
    const auto miss = empty_cache.load(current, unrelated, nullptr, nullptr, 0);
    assert(miss.success);
    assert(!miss.restored);
    state.restore_from_cache(miss.restored, miss.draft_synchronized);
    assert(state.begin_request(false));

    // Cache hits carry the saved entry's coherence, independent of the new
    // request. Empty context byte arrays let this metadata path be tested
    // without loading a model.
    server_prompt cached;
    cached.tokens = server_tokens(llama_tokens { 9, 10 }, false);
    server_prompt_cache incoherent_cache(1, 0);
    assert(incoherent_cache.alloc(cached, 0, 0, false, {}) != nullptr);

    server_prompt destination;
    server_tokens matching(llama_tokens { 9, 10, 11 }, false);
    const auto hit = incoherent_cache.load(destination, matching, nullptr, nullptr, 0);
    assert(hit.success);
    assert(hit.restored);
    assert(!hit.draft_synchronized);
    state.mark_synchronized();
    state.restore_from_cache(hit.restored, hit.draft_synchronized);
    assert(state.begin_request(false));
    return 0;
}
