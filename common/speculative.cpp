#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cerrno>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <cinttypes>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual void begin_request(llama_seq_id /*seq_id*/, bool /*has_media*/) {}

    virtual void begin_media(
            llama_seq_id /*seq_id*/,
            llama_pos /*pos_start*/,
            int32_t /*n_pos*/,
            int32_t /*n_tokens*/) {}

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    virtual bool has_private_positions(llama_seq_id /*seq_id*/) const { return false; }
    virtual bool can_reuse_prefix(llama_seq_id /*seq_id*/, llama_pos /*target_pos*/) const { return true; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    std::vector<uint8_t> media_request;
    std::vector<uint8_t> media_rebased;
    std::vector<llama_pos> local_pos_next;
    std::vector<llama_pos> verify_pos_first;
    std::vector<int32_t> verify_rows;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        media_request.assign(n_seq, 0);
        media_rebased.assign(n_seq, 0);
        local_pos_next.assign(n_seq, 0);
        verify_pos_first.assign(n_seq, -1);
        verify_rows.assign(n_seq, 0);

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin_request(llama_seq_id seq_id, bool has_media) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        media_request[seq_id] = has_media;
        media_rebased[seq_id] = false;
        verify_pos_first[seq_id] = -1;
        verify_rows[seq_id] = 0;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || !media_request[seq_id]) {
            return;
        }

        auto * ctx_dft = params.ctx_dft;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1)) {
            SPC_ERR("draft-simple failed to clear seq %d for media text projection\n", (int) seq_id);
            return;
        }

        // Reconstruct only the serializable text history in a compact private
        // position domain. The target retains the full image-conditioned state
        // and verifies every proposal; the draft never receives foreign mmproj
        // vectors or a fabricated image token.
        size_t offset = 0;
        while (offset < prompt.size()) {
            common_batch_clear(batch);
            const size_t n_chunk = std::min<size_t>(llama_n_batch(ctx_dft), prompt.size() - offset);
            for (size_t i = 0; i < n_chunk; ++i) {
                common_batch_add(batch, prompt[offset + i], (llama_pos) (offset + i), { seq_id }, false);
            }
            if (llama_decode(ctx_dft, batch) != 0) {
                SPC_ERR("draft-simple failed to rebuild media text projection for seq %d\n", (int) seq_id);
                return;
            }
            offset += n_chunk;
        }

        local_pos_next[seq_id] = (llama_pos) prompt.size();
        media_rebased[seq_id] = true;
    }

    bool process(const llama_batch & batch) override {
        // The independent draft model cannot consume target-mmproj embeddings:
        // their width and semantics belong to the target. Prompt embedding rows
        // are therefore ignored until begin() reconstructs the serializable text
        // in a compact private domain. Later verification rows are translated
        // into that same domain; target absolute positions never address draft KV.
        if (batch.embd != nullptr) {
            return true;
        }
        if (batch.token == nullptr) {
            SPC_ERR("%s", "draft-simple received a batch without tokens or embeddings\n");
            return false;
        }

        auto * ctx_dft = params.ctx_dft;

        bool needs_projection = false;
        for (int32_t row = 0; row < batch.n_tokens; ++row) {
            const llama_seq_id seq_id = batch.seq_id ? batch.seq_id[row][0] : 0;
            if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq && media_request[seq_id]) {
                if (!media_rebased[seq_id]) {
                    // Prompt prefill will be rebuilt atomically by begin() once
                    // the complete text projection is available.
                    return true;
                }
                needs_projection = true;
            }
        }

        if (needs_projection) {
            common_batch_clear(this->batch);
            std::vector<llama_pos> row_pos(n_seq, -1);
            for (int32_t row = 0; row < batch.n_tokens; ++row) {
                const llama_seq_id seq_id = batch.seq_id ? batch.seq_id[row][0] : 0;
                if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || !media_rebased[seq_id]) {
                    continue;
                }
                if (row_pos[seq_id] < 0) {
                    row_pos[seq_id] = local_pos_next[seq_id];
                    verify_pos_first[seq_id] = local_pos_next[seq_id];
                    verify_rows[seq_id] = 0;
                }
                common_batch_add(this->batch, batch.token[row], row_pos[seq_id]++, { seq_id }, false);
                verify_rows[seq_id]++;
            }

            if (this->batch.n_tokens > 0 && llama_decode(ctx_dft, this->batch) != 0) {
                SPC_ERR("%s", "draft-simple failed to decode a rebased verification batch\n");
                return false;
            }
            return true;
        }

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            const llama_pos pos = media_rebased[seq_id] ? local_pos_next[seq_id] : dp.n_past;
            common_batch_add(batch, dp.id_last, pos, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(params.ctx_tgt, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                const llama_pos pos = media_rebased[seq_id]
                    ? local_pos_next[seq_id] + i + 1
                    : dp.n_past + i + 1;
                common_batch_add(batch, id, pos, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }

        // Verification owns the committed draft-side state. Remove the seed and
        // provisional tokens so the translated target batch can be decoded once
        // at the compact local boundary.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (media_rebased[seq_id]) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, local_pos_next[seq_id], -1);
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || !media_rebased[seq_id] || verify_rows[seq_id] <= 0) {
            return;
        }

        const int32_t n_keep = std::min<int32_t>(verify_rows[seq_id], (int32_t) n_accepted + 1);
        local_pos_next[seq_id] = verify_pos_first[seq_id] + n_keep;
        llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, local_pos_next[seq_id], -1);
        verify_rows[seq_id] = 0;
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;
    std::vector<uint8_t>            compact_positions;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);
        compact_positions.assign(n_seq, 0);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        if (compact_positions[seq_id]) {
            // The serialized prompt omits expanded image rows, so its token
            // count is not comparable with EAGLE's compact private positions.
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd != nullptr;
        if (has_tokens == has_embeddings) {
            SPC_ERR("%s", "EAGLE3 target batch must contain exactly one of tokens or embeddings\n");
            return false;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        if (has_embeddings) {
            // Atomic media adaptation of ggml-org/llama.cpp's EAGLE3 deferred-
            // boundary implementation. Image rows have target features but no
            // token IDs, so they cannot
            // form EAGLE's (token[P+1], g[P]) decoder pairs. Compress the media
            // boundary into its final target-derived g row. The first real token
            // after the image completes that deferred pair, allowing drafting to
            // resume during the same answer without inventing image tokens.
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                const int32_t beg = i_batch_beg[seq_id];
                const int32_t end = i_batch_end[seq_id];
                if (beg < 0 || end < 0) {
                    continue;
                }

                const int32_t n_rows = end - beg + 1;
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(
                        llama_get_memory(ctx_dft), seq_id);

                // The image consumes many target rows and may use repeated
                // M-RoPE positions. EAGLE cannot reproduce those positions
                // because its decoder requires token[P+1] for every g[P]. Keep
                // the final target-derived image feature at the next compact
                // draft position instead; target verification remains in the
                // target's absolute position domain.
                compact_positions[seq_id] = true;
                verify_pos_first[seq_id] = dft_pos_max + 1;
                pending_pos_last[seq_id] = dft_pos_max + 1;
                verify_g_rows[seq_id]    = n_rows;
                verify_g[seq_id].resize((size_t) n_rows * n_embd_dec);
                std::memcpy(verify_g[seq_id].data(),
                            g_embd + (size_t) beg * n_embd_dec,
                            row_bytes * n_rows);
                std::memcpy(pending_g_last[seq_id].data(),
                            g_embd + (size_t) end * n_embd_dec,
                            row_bytes);
            }

            return true;
        }

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            if (batch_in.pos[beg] == 0) {
                // A position-zero target rebuild starts a new position domain.
                // Do not carry a compact boundary from an unrelated slot use.
                compact_positions[seq_id] = false;
                pending_pos_last[seq_id] = -1;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            const bool compact = compact_positions[seq_id];
            if (pending_pos >= 0 && (compact || pending_pos + 1 == batch_in.pos[beg])) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                const llama_pos dft_pos = compact
                    ? pending_pos + 1 + (k - beg)
                    : batch_in.pos[k];
                common_batch_add(batch, batch_in.token[k + 1], dft_pos, { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = compact ? pending_pos + 1 : batch_in.pos[beg];
            pending_pos_last[seq_id] = compact ? pending_pos + n_rows : batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(params.ctx_tgt, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size() ||
                        (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }

        // The server's generic draft rollback uses target absolute positions.
        // Those positions are intentionally larger than EAGLE's compact media
        // domain, so it cannot remove provisional compact rows for us. Target
        // verification will immediately rebuild the committed rows from target
        // features; discard the seed and proposals now to avoid overlapping the
        // verification batch with stale speculative KV.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (compact_positions[seq_id] && pending_pos_last[seq_id] >= 0) {
                llama_memory_seq_rm(
                        llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + sizeof(uint8_t) + g.size() * sizeof(float));
        std::memcpy(data.data(), &pos, sizeof(llama_pos));
        data[sizeof(llama_pos)] = compact_positions[seq_id];
        std::memcpy(data.data() + sizeof(llama_pos) + sizeof(uint8_t), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const size_t legacy_size = sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float);
        const size_t current_size = legacy_size + sizeof(uint8_t);
        if (data.size() != legacy_size && data.size() != current_size) {
            pending_pos_last[seq_id] = -1;
            compact_positions[seq_id] = false;
            std::fill(pending_g_last[seq_id].begin(), pending_g_last[seq_id].end(), 0.0f);
            return;
        }

        std::memcpy(&pending_pos_last[seq_id], data.data(), sizeof(llama_pos));
        const size_t feature_offset = data.size() == current_size
            ? sizeof(llama_pos) + sizeof(uint8_t)
            : sizeof(llama_pos);
        compact_positions[seq_id] = data.size() == current_size ? data[sizeof(llama_pos)] : false;
        std::memcpy(pending_g_last[seq_id].data(), data.data() + feature_offset,
                (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    struct media_prefill_state {
        bool active = false;
        llama_pos pos_start = -1;
        int32_t n_pos = 0;
        int32_t n_tokens = 0;
        int32_t n_processed = 0;
        std::vector<float> encoded_rows;
    };
    std::vector<media_prefill_state> media_prefill;

    // The target can be 32K or larger while published DSpark sidecars are
    // trained for 4096 positions. Keep an explicitly mapped private KV domain
    // rather than addressing the draft with target-absolute positions.
    std::vector<common_speculative_draft_window_state> windows;
    std::vector<llama_pos> verify_local_begin;
    std::vector<int32_t> verify_rows;
    std::vector<uint8_t> awaiting_verification;

    llama_pos window_capacity = 0;
    llama_pos window_reserve = 0;
    llama_pos rollover_chunk = 0;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;
        media_prefill.resize(n_seq);
        windows.resize(n_seq);
        verify_local_begin.assign(n_seq, -1);
        verify_rows.assign(n_seq, 0);
        awaiting_verification.assign(n_seq, 0);

        // Modern DFlash and the original Prism DSpark exporter use different
        // metadata namespaces for the same block-diffusion runtime contract.
        auto get_int_metadata = [&](const char * key, int64_t min_value, int64_t max_value, int64_t & value) {
            char buf[64] = {};
            if (llama_model_meta_val_str(model_dft, key, buf, sizeof(buf)) < 0) {
                return false;
            }

            value = common_speculative_parse_metadata_int(key, buf, min_value, max_value);
            return true;
        };

        block_size = 16;
        {
            int64_t parsed_block_size = block_size;
            if (get_int_metadata("dflash.block_size", 1, 4096, parsed_block_size) ||
                get_int_metadata("dspark.dspark.block_size", 1, 4096, parsed_block_size)) {
                block_size = (int32_t) parsed_block_size;
            }
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
        }
        const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
        const int32_t n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        if (n_vocab_dft <= 0) {
            throw std::runtime_error("DSpark draft model has an empty vocabulary");
        }

        mask_token_id = llama_vocab_mask(vocab_dft);
        if (mask_token_id == LLAMA_TOKEN_NULL) {
            int64_t parsed_mask_token = 0;
            if (get_int_metadata("dspark.dspark.mask_token_id", 0, n_vocab_dft - 1, parsed_mask_token)) {
                mask_token_id = (llama_token) parsed_mask_token;
            }
        }
        if (mask_token_id == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("DSpark draft model has no mask token id");
        }
        if (mask_token_id < 0 || mask_token_id >= n_vocab_dft) {
            throw std::runtime_error(string_format(
                    "DSpark mask token id %d is outside the draft vocabulary [0, %d)",
                    mask_token_id, n_vocab_dft));
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        window_capacity = (llama_pos) llama_n_ctx_seq(ctx_dft);
        window_reserve = this->params.n_max + (is_dspark && sample_from_anchor ? 0 : 1);
        if (window_reserve <= 0 || window_reserve >= window_capacity) {
            throw std::runtime_error(string_format(
                    "%s draft block reserve %d does not fit the %d-position draft context",
                    common_speculative_type_to_str(type).c_str(), (int) window_reserve, (int) window_capacity));
        }
        const llama_pos max_verification_rows = (llama_pos) this->params.n_max + 1;
        if (max_verification_rows > window_capacity - window_reserve) {
            // Verification reinjects the sampled anchor plus every proposal.
            // The entire span must coexist until accept() trims its rejected
            // suffix; rolling within it would discard rows from its beginning.
            throw std::runtime_error(string_format(
                    "%s verification span %d does not fit the usable %d-position draft window",
                    common_speculative_type_to_str(type).c_str(), (int) max_verification_rows,
                    (int) (window_capacity - window_reserve)));
        }
        if (!llama_memory_can_shift(llama_get_memory(ctx_dft))) {
            // Rollover requires translating every retained key by the same
            // delta. Continuing with an unshiftable cache would either fail at
            // 4096 again or silently attend with inconsistent RoPE positions.
            throw std::runtime_error(string_format(
                    "%s requires a shift-capable draft KV cache for long-context rollover",
                    common_speculative_type_to_str(type).c_str()));
        }
        rollover_chunk = std::min<llama_pos>(512, window_capacity - window_reserve);

        LOG_INF("%s: - draft_ctx_per_seq=%d, cache_k_requested=%s, cache_v_requested=%s, "
                "flash_attn=%s, gpu_layers_requested=%d, devices=[%s], op_offload=disabled, "
                "window_reserve=%d, rollover_chunk=%d\n",
                __func__, (int) window_capacity,
                ggml_type_name(this->params.cache_type_k), ggml_type_name(this->params.cache_type_v),
                llama_context_is_flash_attn(ctx_dft) ? "enabled" : "disabled",
                this->params.n_gpu_layers,
                common_speculative_get_devices_str(this->params.devices).c_str(),
                (int) window_reserve, (int) rollover_chunk);

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    bool apply_window_plan(
            llama_seq_id seq_id,
            const common_speculative_draft_window_plan & plan) {
        auto * mem_dft = llama_get_memory(params.ctx_dft);
        auto & state = windows[seq_id];

        if (plan.reset) {
            if (!llama_memory_seq_rm(mem_dft, seq_id, -1, -1)) {
                SPC_ERR("%s failed to reset private draft sequence %d\n",
                        common_speculative_type_to_str(type).c_str(), (int) seq_id);
                return false;
            }
            state = {};
        }

        if (plan.truncate_from >= 0) {
            if (!llama_memory_seq_rm(mem_dft, seq_id, plan.truncate_from, -1)) {
                SPC_ERR("%s failed to truncate private draft sequence %d at local position %d\n",
                        common_speculative_type_to_str(type).c_str(), (int) seq_id,
                        (int) plan.truncate_from);
                return false;
            }
            state.next = plan.truncate_from;
        }

        if (plan.discard > 0) {
            const llama_pos old_origin = state.origin;
            const llama_pos retained = state.next - plan.discard;
            if (!llama_memory_seq_rm(mem_dft, seq_id, 0, plan.discard)) {
                SPC_ERR("%s failed to discard private draft prefix [0, %d) for sequence %d\n",
                        common_speculative_type_to_str(type).c_str(), (int) plan.discard, (int) seq_id);
                return false;
            }
            llama_memory_seq_add(mem_dft, seq_id, plan.discard, -1, -plan.discard);
            state.origin += plan.discard;
            state.next   -= plan.discard;
            LOG_INF("%s: rolled draft window for seq %d: origin %d -> %d, discarded=%d, retained=%d\n",
                    __func__, (int) seq_id, (int) old_origin, (int) state.origin,
                    (int) plan.discard, (int) retained);
        }

        if (state.origin != plan.origin || state.next != plan.local_begin) {
            SPC_ERR("%s private draft mapping diverged for sequence %d (origin=%d/%d, next=%d/%d)\n",
                    common_speculative_type_to_str(type).c_str(), (int) seq_id,
                    (int) state.origin, (int) plan.origin, (int) state.next, (int) plan.local_begin);
            return false;
        }
        return true;
    }

    bool prepare_window(
            llama_seq_id seq_id,
            llama_pos target_begin,
            llama_pos n_positions,
            common_speculative_draft_window_plan & plan) {
        if (!common_speculative_draft_window_plan_make(
                    windows[seq_id], target_begin, n_positions,
                    window_capacity, window_reserve, rollover_chunk, plan)) {
            const auto & state = windows[seq_id];
            SPC_ERR("%s cannot map target span [%d, %d) into seq %d private window "
                    "(origin=%d, next=%d, capacity=%d, reserve=%d)\n",
                    common_speculative_type_to_str(type).c_str(), (int) target_begin,
                    (int) (target_begin + n_positions), (int) seq_id,
                    (int) state.origin, (int) state.next,
                    (int) window_capacity, (int) window_reserve);
            return false;
        }
        return apply_window_plan(seq_id, plan);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        const llama_pos expected = windows[seq_id].next - 1;
        if (pos_max != expected && !(pos_max < 0 && expected < 0)) {
            LOG_WRN("%s: private draft KV ends at %d but mapping expects %d for seq %d; "
                    "the next injection will rebuild or reject the inconsistent prefix\n",
                    __func__, (int) pos_max, (int) expected, (int) seq_id);
        }
        GGML_UNUSED(prompt);
    }

    void begin_media(
            llama_seq_id seq_id,
            llama_pos pos_start,
            int32_t n_pos,
            int32_t n_tokens) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        auto & state = media_prefill[seq_id];
        state = {};
        if (pos_start < 0 || n_pos <= 0 || n_tokens <= 0) {
            LOG_ERR("%s: invalid media span for seq %d: start=%d, n_pos=%d, n_tokens=%d\n",
                    __func__, (int) seq_id, (int) pos_start, n_pos, n_tokens);
            return;
        }

        state.active      = true;
        state.pos_start   = pos_start;
        state.n_pos       = n_pos;
        state.n_tokens    = n_tokens;
        state.n_processed = 0;
        state.encoded_rows.reserve((size_t) n_tokens * n_embd_dec);
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Atomic media adaptation of upstream DFlash/DSpark (ggml-org PR
        // #25173) and Prism's earlier Markov implementation (commit ba62b7023).
        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // A draft ubatch can be larger than the usable private window when a
        // deliberately small context is selected. Bound each injection to the
        // retained domain so the planner can roll before every chunk instead of
        // rejecting an otherwise valid long prompt in one oversized request.
        const int32_t n_ubatch = std::min<int32_t>(
                (int32_t) llama_n_ubatch(ctx_dft), (int32_t) (window_capacity - window_reserve));

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            if (awaiting_verification[seq_id]) {
                const llama_pos usable = window_capacity - window_reserve;
                if (has_embeddings || n_rows > usable) {
                    LOG_ERR("%s: verification feature span for seq %d has %d rows (usable=%d, embeddings=%s)\n",
                            __func__, (int) seq_id, n_rows, (int) usable,
                            has_embeddings ? "yes" : "no");
                    return false;
                }

                const llama_pos target_begin = batch_in.pos[i_batch_beg[seq_id]];
                for (int32_t i = 1; i < n_rows; ++i) {
                    if (batch_in.pos[i_batch_beg[seq_id] + i] != target_begin + i) {
                        LOG_ERR("%s: non-consecutive verification positions for seq %d at row %d\n",
                                __func__, (int) seq_id, i);
                        return false;
                    }
                }

                // Reserve the full verification span before chunked encoding.
                // prepare_window() applies only the required prefix rollover;
                // each successful chunk below advances the committed boundary.
                common_speculative_draft_window_plan span_plan;
                if (!prepare_window(seq_id, target_begin, n_rows, span_plan)) {
                    return false;
                }
                verify_local_begin[seq_id] = span_plan.local_begin;
                verify_rows[seq_id] = n_rows;
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                auto & media = media_prefill[seq_id];
                if (has_embeddings && media.active) {
                    // M-RoPE image encoders can emit many patch rows at the same
                    // one-dimensional text position. Draft KV requires unique,
                    // consecutive positions, so accumulate the target-derived
                    // features across callback chunks and pool them into the
                    // media chunk's declared one-dimensional position span.
                    media.encoded_rows.insert(
                            media.encoded_rows.end(), inp_g, inp_g + (size_t) n_chunk * n_embd_dec);
                    media.n_processed += n_chunk;

                    if (media.n_processed > media.n_tokens) {
                        LOG_ERR("%s: media encoder produced %d rows, expected %d\n",
                                __func__, media.n_processed, media.n_tokens);
                        return false;
                    }
                    if (media.n_processed < media.n_tokens) {
                        continue;
                    }

                    std::vector<float> pooled((size_t) media.n_pos * n_embd_dec, 0.0f);
                    for (int32_t p = 0; p < media.n_pos; ++p) {
                        int32_t row_beg = (int32_t) ((int64_t) p * media.n_tokens / media.n_pos);
                        int32_t row_end = (int32_t) ((int64_t) (p + 1) * media.n_tokens / media.n_pos);
                        row_beg = std::min(row_beg, media.n_tokens - 1);
                        row_end = std::clamp(row_end, row_beg + 1, media.n_tokens);

                        float * dst = pooled.data() + (size_t) p * n_embd_dec;
                        for (int32_t row = row_beg; row < row_end; ++row) {
                            const float * src = media.encoded_rows.data() + (size_t) row * n_embd_dec;
                            for (int32_t d = 0; d < n_embd_dec; ++d) {
                                dst[d] += src[d];
                            }
                        }
                        const float scale = 1.0f / (row_end - row_beg);
                        for (int32_t d = 0; d < n_embd_dec; ++d) {
                            dst[d] *= scale;
                        }
                    }

                    for (int32_t pos_off = 0; pos_off < media.n_pos; pos_off += n_ubatch) {
                        const int32_t n_inject = std::min(n_ubatch, media.n_pos - pos_off);
                        common_speculative_draft_window_plan window_plan;
                        const llama_pos target_begin = media.pos_start + pos_off;
                        if (!prepare_window(seq_id, target_begin, n_inject, window_plan)) {
                            return false;
                        }
                        batch_inject.n_tokens = n_inject;
                        std::memcpy(
                                batch_inject.embd,
                                pooled.data() + (size_t) pos_off * n_embd_dec,
                                (size_t) n_inject * n_embd_dec * sizeof(float));
                        for (int32_t i = 0; i < n_inject; ++i) {
                            batch_inject.pos[i]       = window_plan.local_begin + i;
                            batch_inject.n_seq_id[i]  = 1;
                            batch_inject.seq_id[i][0] = seq_id;
                            batch_inject.logits[i]    = false;
                        }

                        const int32_t rc_inject = llama_decode(ctx_dft, batch_inject);
                        if (rc_inject != 0) {
                            LOG_ERR("%s: media KV injection failed rc=%d (seq=%d, pos=%d, n=%d)\n",
                                    __func__, rc_inject, (int) seq_id,
                                    (int) target_begin, n_inject);
                            return false;
                        }
                        windows[seq_id].next = window_plan.local_end;
                    }

                    LOG_DBG("%s: pooled %d media rows into %d draft positions [%d, %d) for seq %d\n",
                            __func__, media.n_tokens, media.n_pos, (int) media.pos_start,
                            (int) (media.pos_start + media.n_pos), (int) seq_id);
                    media = {};
                    continue;
                }

                // Translate the target's absolute span into the sidecar's
                // bounded private position domain before committing decoder KV.
                const llama_pos target_begin = batch_in.pos[i_batch_beg[seq_id] + offset];
                for (int32_t i = 1; i < n_chunk; ++i) {
                    if (batch_in.pos[i_batch_beg[seq_id] + offset + i] != target_begin + i) {
                        LOG_ERR("%s: non-consecutive target feature positions for seq %d at row %d\n",
                                __func__, (int) seq_id, i);
                        return false;
                    }
                }

                common_speculative_draft_window_plan window_plan;
                if (!prepare_window(seq_id, target_begin, n_chunk, window_plan)) {
                    return false;
                }

                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    batch_inject.pos[i]       = window_plan.local_begin + i;
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
                windows[seq_id].next = window_plan.local_end;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
        std::vector<llama_pos> block_local_begin(n_seq, -1);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            common_speculative_draft_window_plan window_plan;
            if (!prepare_window(seq_id, dp.n_past, 0, window_plan) ||
                    window_plan.local_begin + n_block_tokens > window_capacity) {
                dp.drafting = false;
                continue;
            }

            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            block_local_begin[seq_id] = window_plan.local_begin;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id,
                        window_plan.local_begin + i, { seq_id }, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (block_local_begin[seq_id] >= 0) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, block_local_begin[seq_id], -1);
                }
            }
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(params.ctx_tgt, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(params.ctx_tgt, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }

            awaiting_verification[seq_id] = !result.empty();
            verify_local_begin[seq_id] = -1;
            verify_rows[seq_id] = 0;
        }

        // Noise/mask rows are provisional. The target verification callback
        // reinjects only the sampled token plus verified proposals, and accept()
        // then trims rejected rows. Removing the block here prevents private KV
        // from accumulating duplicate speculative positions across rollovers.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (block_local_begin[seq_id] >= 0 &&
                    !llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, block_local_begin[seq_id], -1)) {
                SPC_ERR("%s failed to remove provisional block for sequence %d at local position %d\n",
                        common_speculative_type_to_str(type).c_str(), (int) seq_id,
                        (int) block_local_begin[seq_id]);
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || !awaiting_verification[seq_id]) {
            return;
        }

        if (verify_local_begin[seq_id] < 0 || verify_rows[seq_id] <= 0) {
            SPC_ERR("%s did not receive a verification feature batch for sequence %d\n",
                    common_speculative_type_to_str(type).c_str(), (int) seq_id);
            awaiting_verification[seq_id] = false;
            return;
        }

        // The verification batch starts with the target's sampled anchor, then
        // the proposals. Commit the anchor and the accepted prefix only.
        const llama_pos n_keep = std::min<llama_pos>(verify_rows[seq_id], (llama_pos) n_accepted + 1);
        const llama_pos next = verify_local_begin[seq_id] + n_keep;
        if (!llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, next, -1)) {
            SPC_ERR("%s failed to roll back rejected verification rows for sequence %d at local position %d\n",
                    common_speculative_type_to_str(type).c_str(), (int) seq_id, (int) next);
        } else {
            windows[seq_id].next = next;
        }

        awaiting_verification[seq_id] = false;
        verify_local_begin[seq_id] = -1;
        verify_rows[seq_id] = 0;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        data = common_speculative_draft_window_state_serialize(windows[seq_id]);
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        common_speculative_draft_window_state restored;
        auto * mem_dft = llama_get_memory(params.ctx_dft);
        bool restored_ok = common_speculative_draft_window_state_restore(
                data, window_capacity, window_reserve, restored);
        if (restored_ok) {
            // Prompt and speculative checkpoints can contain provisional or
            // rejected rows past the saved logical boundary. Never allow those
            // rows to survive state assignment to another slot.
            restored_ok = llama_memory_seq_rm(mem_dft, seq_id, restored.next, -1);
            restored_ok = restored_ok && common_speculative_draft_window_matches_positions(
                    restored,
                    llama_memory_seq_pos_min(mem_dft, seq_id),
                    llama_memory_seq_pos_max(mem_dft, seq_id));
        }
        if (!restored_ok) {
            SPC_WRN("%s discarded contradictory private draft checkpoint for sequence %d\n",
                    common_speculative_type_to_str(type).c_str(), (int) seq_id);
            windows[seq_id] = {};
            llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
        } else {
            windows[seq_id] = restored;
        }
        awaiting_verification[seq_id] = false;
        verify_local_begin[seq_id] = -1;
        verify_rows[seq_id] = 0;
        media_prefill[seq_id] = {};
    }

    bool has_private_positions(llama_seq_id /*seq_id*/) const override {
        return true;
    }

    bool can_reuse_prefix(llama_seq_id seq_id, llama_pos target_pos) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || target_pos < 0) {
            return false;
        }
        const auto & state = windows[seq_id];
        return target_pos == 0 ||
            (target_pos >= state.origin && (int64_t) target_pos <= (int64_t) state.origin + state.next);
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        const bool has_tokens      = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd != nullptr;
        if (has_tokens == has_embeddings) {
            SPC_ERR("%s", "MTP target batch must contain exactly one of tokens or embeddings\n");
            return false;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared && has_tokens) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        // Atomic continuation of Biogenic Ooze's mmproj/speculation coexistence
        // work (commit ead60fb4f) on the converged upstream MTP driver.
        // Independent MTP contexts cannot decode image embeddings because their
        // graph requires a real next-token ID paired with h_nextn. Retaining the
        // final image h row is the rebase: the first following text token pairs
        // with this target-derived boundary, while shared-memory Gemma assistants
        // already attend the target memory directly. In both cases no fake media
        // token enters the draft graph.

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(params.ctx_tgt, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size() ||
                        (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    // Eligibility belongs to an implementation/sequence pair. A failure or an
    // unsupported media policy must not disable a healthy fallback or another
    // parallel server slot.
    std::vector<std::vector<uint8_t>> active;

};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch == "gemma4-assistant" || arch == "gemma4_assistant") {
        SPC_INF("auto-detected speculative type 'draft-mtp' from Gemma assistant architecture '%s'\n", arch.c_str());
        return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    }

    if (arch != "dflash" && arch != "dspark") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const bool has_markov = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0 ||
                            gguf_find_tensor(gguf_ctx.get(), "dspark.markov_head_a.weight") >= 0;
    const auto type = has_markov
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();
    const bool has_dspark = std::find(
            params.speculative.types.begin(),
            params.speculative.types.end(),
            COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params.speculative.types.end();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    if (has_dspark) {
        // Published DSpark sidecars are trained for 4096 positions. The target
        // may use a much larger context, but allocating or addressing the draft
        // beyond its trained window is both wasteful and unsafe. n_ctx is shared
        // across sequences, so preserve the full 4096 positions per parallel slot.
        constexpr int32_t dspark_n_ctx_per_seq = 4096;
        const int64_t cap64 = (int64_t) dspark_n_ctx_per_seq * std::max(1, params.n_parallel);
        const int32_t cap = (int32_t) std::min<int64_t>(cap64, INT32_MAX);
        if (result.n_ctx == 0 || result.n_ctx > cap) {
            result.n_ctx = cap;
        }
    }

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

        // The MTP assistant graph is intended to remain device-resident even when the
        // target uses op offload for CPU-resident MoE experts. Inheriting the target's
        // setting lets the scheduler place assistant ops on the CPU backend and defeats
        // both the shared-KV path and speculative throughput.
        cparams.op_offload = false;
    }

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

int64_t common_speculative_parse_metadata_int(
        const char * key, const char * value, int64_t min_value, int64_t max_value) {
    // GGUF metadata reaches this compatibility path as text. `atoi()` would
    // silently accept prefixes such as "16junk", map malformed strings to zero,
    // and overflow without a usable diagnostic, all of which can corrupt DSpark
    // batch geometry or select an out-of-vocabulary mask token.
    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    while (end && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (errno == ERANGE || end == value || (end && *end != '\0') || parsed < min_value || parsed > max_value) {
        throw std::runtime_error(string_format(
                "DSpark metadata '%s' must be an integer in [%lld, %lld], got '%s'",
                key, (long long) min_value, (long long) max_value, value));
    }
    return parsed;
}

common_speculative_media_policy common_speculative_media_policy_for_type(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
            return COMMON_SPECULATIVE_MEDIA_POLICY_EMBEDDING_PREFILL;
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            return COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE;
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
            return COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS;
        case COMMON_SPECULATIVE_TYPE_NONE:
        case COMMON_SPECULATIVE_TYPE_COUNT:
            return COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED;
    }

    return COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED;
}

const char * common_speculative_media_policy_name(common_speculative_media_policy policy) {
    switch (policy) {
        case COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED:         return "unsupported";
        case COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS:           return "stateless";
        case COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE:   return "post-media-rebase";
        case COMMON_SPECULATIVE_MEDIA_POLICY_EMBEDDING_PREFILL:   return "embedding-prefill";
    }

    return "unknown";
}

bool common_speculative_restore_boundary_state(
        const std::vector<uint8_t> & data,
        int32_t n_embd,
        llama_pos & pos,
        std::vector<float> & values) {
    // Clear first even when validation fails. Cache assignment can move a slot
    // from an entry with a deferred boundary to one without it; retaining the
    // destination's previous values would splice two unrelated requests.
    pos = -1;
    values.assign(std::max(0, n_embd), 0.0f);

    if (n_embd < 0 || data.size() != sizeof(llama_pos) + (size_t) n_embd * sizeof(float)) {
        return false;
    }

    std::memcpy(&pos, data.data(), sizeof(llama_pos));
    std::memcpy(values.data(), data.data() + sizeof(llama_pos), (size_t) n_embd * sizeof(float));
    return true;
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

bool common_speculative_draft_window_plan_make(
        const common_speculative_draft_window_state & state,
        llama_pos target_begin,
        llama_pos n_positions,
        llama_pos capacity,
        llama_pos reserve,
        llama_pos rollover_chunk,
        common_speculative_draft_window_plan & plan) {
    plan = {};

    if (target_begin < 0 || n_positions < 0 || capacity <= 0 || reserve < 0 ||
            reserve >= capacity || rollover_chunk <= 0) {
        return false;
    }

    common_speculative_draft_window_state cur = state;
    if (cur.origin < 0 || cur.next < 0 || cur.next > capacity - reserve) {
        return false;
    }

    // A position-zero decode is the unambiguous full-rebuild boundary. Do not
    // carry a private mapping from the slot's previous request across it.
    if (target_begin == 0 && (cur.origin != 0 || cur.next != 0)) {
        plan.reset = true;
        cur = {};
    }

    const int64_t absolute_next = (int64_t) cur.origin + cur.next;
    if (absolute_next > std::numeric_limits<llama_pos>::max() ||
            target_begin < cur.origin || target_begin > absolute_next) {
        return false;
    }

    if (target_begin < absolute_next) {
        plan.truncate_from = target_begin - cur.origin;
        cur.next = plan.truncate_from;
    }

    const llama_pos usable = capacity - reserve;
    if (n_positions > usable) {
        return false;
    }

    const int64_t end_before = (int64_t) cur.next + n_positions;
    if (end_before > usable) {
        const llama_pos overflow = (llama_pos) (end_before - usable);
        llama_pos discard = std::max(overflow, rollover_chunk);
        const int64_t rounded = ((int64_t) discard + rollover_chunk - 1) / rollover_chunk * rollover_chunk;
        discard = (llama_pos) std::min<int64_t>(cur.next, rounded);
        if (discard < overflow) {
            return false;
        }
        plan.discard = discard;
        cur.origin += discard;
        cur.next   -= discard;
    }

    plan.origin      = cur.origin;
    plan.local_begin = cur.next;
    plan.local_end   = cur.next + n_positions;
    return plan.local_end <= usable;
}

std::vector<uint8_t> common_speculative_draft_window_state_serialize(
        const common_speculative_draft_window_state & state) {
    static constexpr uint8_t magic[] = { 'A', 'D', 'W', 1 };
    std::vector<uint8_t> data(sizeof(magic) + 2*sizeof(llama_pos));
    std::memcpy(data.data(), magic, sizeof(magic));
    std::memcpy(data.data() + sizeof(magic), &state.origin, sizeof(llama_pos));
    std::memcpy(data.data() + sizeof(magic) + sizeof(llama_pos), &state.next, sizeof(llama_pos));
    return data;
}

bool common_speculative_draft_window_state_restore(
        const std::vector<uint8_t> & data,
        llama_pos capacity,
        llama_pos reserve,
        common_speculative_draft_window_state & state) {
    static constexpr uint8_t magic[] = { 'A', 'D', 'W', 1 };
    state = {};
    if (capacity <= 0 || reserve < 0 || reserve >= capacity ||
            data.size() != sizeof(magic) + 2*sizeof(llama_pos) ||
            std::memcmp(data.data(), magic, sizeof(magic)) != 0) {
        return false;
    }

    common_speculative_draft_window_state restored;
    std::memcpy(&restored.origin, data.data() + sizeof(magic), sizeof(llama_pos));
    std::memcpy(&restored.next, data.data() + sizeof(magic) + sizeof(llama_pos), sizeof(llama_pos));
    if (restored.origin < 0 || restored.next < 0 || restored.next > capacity - reserve ||
            (int64_t) restored.origin + restored.next > std::numeric_limits<llama_pos>::max()) {
        return false;
    }

    state = restored;
    return true;
}

bool common_speculative_draft_window_matches_positions(
        const common_speculative_draft_window_state & state,
        llama_pos pos_min,
        llama_pos pos_max) {
    if (state.next == 0) {
        return pos_min < 0 && pos_max < 0;
    }
    return pos_min == 0 && pos_max == state.next - 1;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .active    = */ {},
    };

    result->active.assign(result->impls.size(), std::vector<uint8_t>(n_seq, 1));

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

bool common_speculative_supports_media(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (common_speculative_media_policy_for_type(impl->type) !=
                COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED) {
            return true;
        }
    }

    return false;
}

bool common_speculative_begin_request(
        common_speculative * spec, llama_seq_id seq_id, bool has_media) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return false;
    }

    bool any = false;
    for (size_t i = 0; i < spec->impls.size(); ++i) {
        const auto policy = common_speculative_media_policy_for_type(spec->impls[i]->type);
        const bool eligible = !has_media || policy != COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED;
        spec->active[i][seq_id] = eligible;
        spec->impls[i]->begin_request(seq_id, has_media);
        any = any || eligible;
    }
    spec->impl_last[seq_id] = nullptr;
    return any;
}

void common_speculative_begin_media(
        common_speculative * spec,
        llama_seq_id seq_id,
        llama_pos pos_start,
        int32_t n_pos,
        int32_t n_tokens) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return;
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id]) {
            spec->impls[i]->begin_media(seq_id, pos_start, n_pos, n_tokens);
        }
    }
}

uint64_t common_speculative_eligibility_mask(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return 0;
    }

    GGML_ASSERT(spec->impls.size() <= 64);
    uint64_t result = 0;
    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id]) {
            result |= UINT64_C(1) << i;
        }
    }
    return result;
}

bool common_speculative_can_speculate(
        const common_speculative * spec, llama_seq_id seq_id) {
    return common_speculative_eligibility_mask(spec, seq_id) != 0;
}

bool common_speculative_state_is_coherent(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return false;
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        const auto policy = common_speculative_media_policy_for_type(spec->impls[i]->type);
        if (policy != COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS && !spec->active[i][seq_id]) {
            return false;
        }
    }
    return true;
}

bool common_speculative_requires_media_reprefill(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return false;
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id] &&
                common_speculative_media_policy_for_type(spec->impls[i]->type) ==
                    COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE) {
            return true;
        }
    }
    return false;
}

bool common_speculative_has_private_positions(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return false;
    }
    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id] && spec->impls[i]->has_private_positions(seq_id)) {
            return true;
        }
    }
    return false;
}

bool common_speculative_can_reuse_prefix(
        const common_speculative * spec, llama_seq_id seq_id, llama_pos target_pos) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return true;
    }
    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id] && !spec->impls[i]->can_reuse_prefix(seq_id, target_pos)) {
            return false;
        }
    }
    return true;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl = spec->impls[i];
        if (!spec->active[i][seq_id]) {
            continue;
        }
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    if (spec == nullptr) {
        return true;
    }

    std::vector<llama_seq_id> seq_ids;
    for (int32_t row = 0; row < batch.n_tokens; ++row) {
        const int32_t n_seq_id = batch.n_seq_id ? batch.n_seq_id[row] : 1;
        for (int32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = batch.seq_id ? batch.seq_id[row][j] : 0;
            if (seq_id >= 0 && seq_id < (llama_seq_id) spec->dparams.size() &&
                    std::find(seq_ids.begin(), seq_ids.end(), seq_id) == seq_ids.end()) {
                seq_ids.push_back(seq_id);
            }
        }
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl = spec->impls[i];

        bool active_for_batch = false;
        for (const llama_seq_id seq_id : seq_ids) {
            active_for_batch = active_for_batch || spec->active[i][seq_id];
        }
        if (!active_for_batch) {
            continue;
        }

        // Server batches are grouped by eligibility mask. Keeping this check in
        // the coordinator catches callers that would otherwise advance an
        // inactive sequence through a shared implementation context.
        for (const llama_seq_id seq_id : seq_ids) {
            if (!spec->active[i][seq_id]) {
                SPC_ERR("mixed eligibility in speculative batch for impl %s; disabling it for the batch\n",
                        common_speculative_type_to_str(impl->type).c_str());
                for (const llama_seq_id affected : seq_ids) {
                    spec->active[i][affected] = false;
                    if (spec->impl_last[affected] == impl.get()) {
                        spec->impl_last[affected] = nullptr;
                    }
                }
                active_for_batch = false;
                break;
            }
        }
        if (!active_for_batch) {
            continue;
        }

        // Always evaluate later implementations. The former short-circuiting
        // boolean skipped healthy fallbacks after the first failure.
        if (!impl->process(batch)) {
            SPC_ERR("disabling speculative impl %s after batch processing failure\n",
                    common_speculative_type_to_str(impl->type).c_str());
            for (const llama_seq_id seq_id : seq_ids) {
                spec->active[i][seq_id] = false;
                if (spec->impl_last[seq_id] == impl.get()) {
                    spec->impl_last[seq_id] = nullptr;
                }
            }
        }
    }

    // Implementation failures are isolated above. The already-decoded target
    // batch remains valid and must be allowed to complete target-only.
    return true;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (size_t impl_i = 0; impl_i < spec->impls.size(); ++impl_i) {
        auto & impl = spec->impls[impl_i];
        std::vector<llama_seq_id> masked;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            if (dparams[seq_id].drafting && !spec->active[impl_i][seq_id]) {
                dparams[seq_id].drafting = false;
                masked.push_back(seq_id);
            }
        }

        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        for (const llama_seq_id seq_id : masked) {
            dparams[seq_id].drafting = true;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        // The target may accept proposals from a drafter that failed while
        // mirroring the verification batch. Those target logits and tokens are
        // still valid, but the failed implementation was deliberately detached
        // and must not receive an accept/rollback callback.
    } else {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl_other = spec->impls[i];
        if (impl_other.get() != impl && spec->active[i][seq_id]) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        if (spec->active[i][seq_id] && spec->impls[i]->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
