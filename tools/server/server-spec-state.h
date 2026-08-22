#pragma once

// Tracks whether a slot's draft context represents the same prefix as its
// target context. Multimodal requests intentionally bypass the drafter, so a
// later text request must not reuse the target prefix until both contexts have
// been cleared and rebuilt together.
struct server_spec_context_state {
    bool synchronized = false;

    bool begin_request(bool has_media) {
        if (has_media) {
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
