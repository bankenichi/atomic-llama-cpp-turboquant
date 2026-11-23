#include "server-slot-sched.h"
#include "server-spec-state.h"
#include "server-task.h"
#include "speculative.h"

#include <cassert>
#include <cstdlib>

// CMake runs this executable in Release builds, where the CRT assert macro is
// compiled out. Keep these state-machine checks active in every configuration.
#undef assert
#define assert(condition) do { if (!(condition)) { std::abort(); } } while (false)

int main() {
    // Media behavior is an algorithm contract, not a server-wide guess based on
    // whether an mmproj is loaded.
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_EMBEDDING_PREFILL);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_EMBEDDING_PREFILL);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NGRAM_MOD) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS);
    assert(common_speculative_media_policy_for_type(COMMON_SPECULATIVE_TYPE_NONE) ==
            COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED);

    server_spec_context_state state;

    // A fresh slot has no proven common target/draft prefix.
    assert(state.begin_request(false, true));
    state.mark_synchronized();
    assert(!state.begin_request(false, true));

    // A media-capable implementation advances with the target. Historical media
    // in a full-history client request must not invalidate an already coherent
    // speculative prefix.
    assert(!state.begin_request(true, true));
    assert(state.is_synchronized());

    // An implementation without a media bridge still fails closed. A later
    // text-only request must rebuild the pair before speculation resumes.
    assert(!state.begin_request(true, false));
    assert(!state.is_synchronized());
    assert(state.begin_request(false, true));

    // After a forced full text rebuild, cache reuse is safe again.
    state.mark_synchronized();
    assert(!state.begin_request(false, true));

    state.invalidate();
    assert(state.begin_request(false, true));

    // A secondary-cache miss is a no-op: it cannot erase media invalidation.
    server_prompt_cache empty_cache(1, 0);
    server_prompt current;
    current.tokens = server_tokens(llama_tokens { 1, 2 }, false);
    server_tokens unrelated(llama_tokens { 9, 10 }, false);
    const auto miss = empty_cache.load(current, unrelated, {}, nullptr, nullptr, 0);
    assert(miss.success);
    assert(!miss.restored);
    state.restore_from_cache(miss.restored, miss.draft_synchronized);
    assert(state.begin_request(false, true));

    // Cache hits carry the saved entry's coherence, independent of the new
    // request. Empty context byte arrays let this metadata path be tested
    // without loading a model.
    server_prompt cached;
    cached.tokens = server_tokens(llama_tokens { 9, 10 }, false);
    server_prompt_cache incoherent_cache(1, 0);
    assert(incoherent_cache.alloc(cached, 0, 0, false, {}) != nullptr);

    server_prompt destination;
    server_tokens matching(llama_tokens { 9, 10, 11 }, false);
    const auto hit = incoherent_cache.load(destination, matching, {}, nullptr, nullptr, 0);
    assert(hit.success);
    assert(hit.restored);
    assert(!hit.draft_synchronized);
    state.mark_synchronized();
    state.restore_from_cache(hit.restored, hit.draft_synchronized);
    assert(state.begin_request(false, true));
    return 0;
}
