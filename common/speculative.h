#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// Real media has different implications for each speculative algorithm. Keep
// this contract in the speculative layer so the server does not have to infer
// safety from an implementation name or from the mere presence of an mmproj.
enum common_speculative_media_policy {
    COMMON_SPECULATIVE_MEDIA_POLICY_UNSUPPORTED,
    COMMON_SPECULATIVE_MEDIA_POLICY_STATELESS,
    COMMON_SPECULATIVE_MEDIA_POLICY_POST_MEDIA_REBASE,
    COMMON_SPECULATIVE_MEDIA_POLICY_EMBEDDING_PREFILL,
};

common_speculative_media_policy common_speculative_media_policy_for_type(common_speculative_type type);

const char * common_speculative_media_policy_name(common_speculative_media_policy policy);

// True when at least one configured implementation has a defined media path.
// The coordinator may still skip an unsupported implementation in a chain.
bool common_speculative_supports_media(const common_speculative * spec);

// Select the implementations eligible for a sequence's new request. Returns
// whether at least one implementation can serve the request.
bool common_speculative_begin_request(
        common_speculative * spec, llama_seq_id seq_id, bool has_media);

// Announces one real-media chunk before its target embedding batches are
// decoded. `n_tokens` is the number of encoder rows; `n_pos` is the number of
// one-dimensional context positions consumed by the chunk. They differ for
// M-RoPE vision models.
void common_speculative_begin_media(
        common_speculative * spec,
        llama_seq_id seq_id,
        llama_pos pos_start,
        int32_t n_pos,
        int32_t n_tokens);

bool common_speculative_can_speculate(const common_speculative * spec, llama_seq_id seq_id);

// Used by server batching so sequences with different implementation state are
// never mirrored through the same unfiltered draft batch.
uint64_t common_speculative_eligibility_mask(const common_speculative * spec, llama_seq_id seq_id);

// False after a stateful implementation was skipped or failed, even if a
// stateless fallback remains eligible.
bool common_speculative_state_is_coherent(const common_speculative * spec, llama_seq_id seq_id);

// Post-media implementations own a draft position domain that cannot be shifted
// with target absolute ranges. The server uses this query to select transactional
// reprefill for retained-media context shifts.
bool common_speculative_requires_media_reprefill(
        const common_speculative * spec, llama_seq_id seq_id);

// True when an active implementation translates target positions into a
// private draft domain. Such a context must not receive target-absolute cache
// shifts and can only reuse prefixes still represented by its retained window.
bool common_speculative_has_private_positions(
        const common_speculative * spec, llama_seq_id seq_id);

bool common_speculative_can_reuse_prefix(
        const common_speculative * spec, llama_seq_id seq_id, llama_pos target_pos);

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// DFlash/DSpark keep a private position domain because published sidecars have a
// much shorter trained context than their targets. These POD helpers are public
// only so rollover and cache-state behavior can be tested without loading a
// model; the speculative implementation remains the sole owner of draft KV.
struct common_speculative_draft_window_state {
    llama_pos origin = 0; // absolute target position represented by local position zero
    llama_pos next   = 0; // next uncommitted position in the local draft window
};

struct common_speculative_draft_window_plan {
    bool reset = false;
    llama_pos truncate_from = -1;
    llama_pos discard = 0;
    llama_pos origin = 0;
    llama_pos local_begin = 0;
    llama_pos local_end = 0;
};

// Plan an injection of [target_begin, target_begin + n_positions). The caller
// applies reset/truncate/discard to KV in that order and commits local_end only
// after decode succeeds. Returns false for a gap, a prefix older than the
// retained window, or an impossible capacity/reserve combination.
bool common_speculative_draft_window_plan_make(
        const common_speculative_draft_window_state & state,
        llama_pos target_begin,
        llama_pos n_positions,
        llama_pos capacity,
        llama_pos reserve,
        llama_pos rollover_chunk,
        common_speculative_draft_window_plan & plan);

std::vector<uint8_t> common_speculative_draft_window_state_serialize(
        const common_speculative_draft_window_state & state);

bool common_speculative_draft_window_state_restore(
        const std::vector<uint8_t> & data,
        llama_pos capacity,
        llama_pos reserve,
        common_speculative_draft_window_state & state);

// Check the dense local-position invariant expected after restoring draft KV.
// Serialized mapping bytes are not sufficient evidence that the corresponding
// physical cache was restored successfully.
bool common_speculative_draft_window_matches_positions(
        const common_speculative_draft_window_state & state,
        llama_pos pos_min,
        llama_pos pos_max);

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// Strict integer parser shared by speculative metadata loaders and tests.
// Throws for empty, partial, overflowing, or out-of-range values.
int64_t common_speculative_parse_metadata_int(
        const char * key, const char * value, int64_t min_value, int64_t max_value);

// Restores the optional Eagle3 deferred-boundary payload. Empty/invalid data
// explicitly clears the destination so cache restoration cannot retain state
// from the previously assigned slot.
bool common_speculative_restore_boundary_state(
        const std::vector<uint8_t> & data,
        int32_t n_embd,
        llama_pos & pos,
        std::vector<float> & values);
// return true if the target and draft models have compatible vocabs
bool common_speculative_are_compatible(const llama_model * model_tgt, const llama_model * model_dft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple, draft-mtp and draft-dflash (DFlash2 only) sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
