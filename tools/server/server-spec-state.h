#pragma once

// Tracks whether a slot's active speculative implementations represent a prefix
// that can safely continue from the target. Media invalidates this state only
// when no configured implementation has a proven media policy; capable
// implementations advance or rebase alongside the target.
struct server_spec_context_state {
    bool synchronized = false;

    bool begin_request(bool has_media, bool media_supported) {
        if (has_media && !media_supported) {
            synchronized = false;
            return false;
        }
        return !synchronized;
    }

    void mark_synchronized() {
        synchronized = true;
    }

    void invalidate() {
        synchronized = false;
    }

    bool is_synchronized() const {
        return synchronized;
    }

    // A secondary-cache miss is a no-op and must preserve the current state.
    // A hit restores the coherence recorded with that cache entry; it must
    // never be inferred from the media status of the incoming request.
    void restore_from_cache(bool restored, bool cached_synchronized) {
        if (restored) {
            synchronized = cached_synchronized;
        }
    }
};
