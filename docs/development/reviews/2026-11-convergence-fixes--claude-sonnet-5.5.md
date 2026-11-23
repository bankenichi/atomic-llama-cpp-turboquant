Reviewer model: Claude Sonnet 5.5 (AI review) | Date: 2026-11-23 | Branch `integration/upstream-convergence-2026-11`, head `1cebbe734` | Commits reviewed: `01a18bea0`, `b09de0b68`, `1d2b2c9bb`, `12a2cc8c7`, `9bd028f0f`, `c55ebc739` (plus the doc / AGENTS.md requirement 7 claims). This is an AI-authored review; it is committed unedited.

# Review of the post-merge bug-fix commits (2026-11 convergence)

Method: static reading and `git` only. Nothing was built, no tests were run, no GPU was touched. Every "fix looks right" below means "the logic is sound by reading", not "verified by execution". Line numbers are for `1cebbe734`.

## Summary

| Commit | Verdict |
|---|---|
| `01a18bea0` CUDA fused MoE reduce aliasing | Fix is right for the case it targets; two small gaps (F5) |
| `b09de0b68` recurrent topology | Fix is right and complete for the shortcuts it names; performance cost in multi-slot contexts is under-documented (F4) |
| `1d2b2c9bb` DSpark | Batch-width half is right. The 4096 cap half restores the old behaviour but the cap is keyed on the wrong thing (F1) and does not bound the per-slot window under unified KV (F2) |
| `9bd028f0f` | `fill` instead of `scale 0` is right; auto-detection change is right (F6, F7 nits) |
| `12a2cc8c7` | Mostly mechanical and plausible; one silent feature loss (F3) |
| `c55ebc739` | Right; superseded in shape by the stage-3 rename, current tree is consistent |
| Doc / AGENTS.md req. 7 | Matches the code and `test-gguf` |

## Findings

### F1: DSpark 4096 cap is keyed on the configured spec type, not on the draft model's head. Important, confidence medium
- `common/speculative.cpp:4810-4817` (new, `1d2b2c9bb`) and `common/speculative.cpp:4673-4745` (`common_base_params_to_speculative`, the `has_dspark` block).
- Both caps test `params.speculative.types` for `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`. The driver factory explicitly allows `--spec-type draft-dflash` to run any DFlash-family draft, including a Markov-head (DSpark) sidecar. See `common/speculative.cpp:5170-5183`: "draft-dflash runs every DFlash family draft"; the driver itself only logs a warning at `:2150-2157` and runs DSpark mode.
- Failure scenario: `llama-server -md <dspark sidecar> --spec-type draft-dflash -c 32768 --parallel 1`.
  - Type list is `{DRAFT_DFLASH}`, so neither cap applies.
  - Draft `n_ctx` becomes the target's 32768, `window_capacity = llama_n_ctx_seq(ctx_dft)` (`:2289`) is 32768, and the private window is never rolled over at 4096.
  - That is exactly the failure `1d2b2c9bb` set out to fix (AGENTS.md requirement 8), reached through a supported option. Consequences: draft positions beyond the trained 4096 (acceptance degrades) and a larger draft KV.
- Why this is the cause and not a symptom: the commit re-applies the cap at the place upstream overrides it, but keeps the old selector (the type string). The auto-detection fix in `9bd028f0f` makes the default path report `draft-dspark`, which hides the gap for auto-detected and `--spec-type draft-dspark` launches.
- Suggested fix: decide after the draft model is loaded, using `llama_model_has_dspark_markov_head(model_dft)` (or the driver's `is_dspark`), and apply the cap to `cparams.n_ctx` there. The model is loaded before `llama_init_from_model` in `common_speculative_init_result`, so the information is available.
- Missing test: nothing exercises the cap at all (no test calls `common_speculative_init_result` / `common_base_params_to_speculative` for DSpark). `test-dspark-legacy` only covers auto-detection and metadata.

### F2: With unified KV the "per slot" cap is not per slot. Minor, pre-existing, confidence medium
- `common/speculative.cpp:4815` (`cap = 4096 * n_parallel`), `src/llama-context.cpp:425-426` (`kv_unified` gives `n_ctx_seq = n_ctx`), `tools/server/server.cpp:130-133` (auto `n_parallel` resolves to 4 with `kv_unified = true`).
- Default `llama-server` launch plus DSpark: the draft context gets `n_ctx = 16384` and `n_ctx_seq = 16384`, so `window_capacity` is 16384 per sequence. Requirement 8 says 4096 per parallel slot.
- Same code is in `backup/pre-upstream-convergence-20261122` (`common/speculative.cpp:3009-3010`), so this is not merge-induced. It matters because the doc line "acceptance run now shows `draft_ctx_per_seq=4096`" (convergence doc, line 133) holds only for single-slot non-unified runs, which is the owner's `--parallel 1` setup.
- Suggested fix: clamp `window_capacity` in the driver to 4096 for DSpark, or make the draft context non-unified. Worth stating the limitation in the doc.

### F3: `mtp_chain_support()` now returns 0 whenever `fused_kv_rows` is on, which includes the default multi-slot server. Minor (performance), confidence medium-high
- `src/llama-context.cpp:1809-1812` (`12a2cc8c7`); `fused_kv_rows` is auto-enabled for `kv_unified` + flash attention (`:388-392`, `:757-759`).
- Default `llama-server` (auto parallel, unified KV, FA) therefore silently loses the MTP chain and falls back to per-step drafting. `--parallel 1` (non-unified) is unaffected, so the owner's setup is fine.
- The convergence doc (line 84) says the "MTP driver is kept whole: deferred catch-up, chained drafts", which is not true for this configuration. No log line says the chain is off.
- The `GGML_ASSERT(inp_attn->self_attn_mask.rows == nullptr)` in `qwen35.cpp` `build_chain` is a good tripwire if this guard is ever bypassed.
- Suggested fix: log once when the chain is declined for this reason, and add a line to the doc's known-limitations list.

### F4: `b09de0b68` removes the direct-gather and GDN state-rows shortcuts for every `n_seq_max > 1` context; the cost is not quantified. Minor, confidence high
- `src/llama-graph.cpp:4281`, `src/models/qwen35.cpp:535`.
- Correctness: sound.
  - `llama_memory_recurrent` is created with `size = max(1, n_seq_max)` (`src/llama-model.cpp:3242`, `:3450-3459`, `:3530`).
  - With `n_seq_max == 1` the reserve graph (`is_full`, `n_rs = size = 1`, `n_seqs = 1`) and the decode graph take the same branch. Prefill versus single-token decode are both reserved, so `n_seq_tokens == 1` in `gdn_state_rows_path` is not a new topology hazard.
  - With `n_seq_max > 1` neither takes the shortcut, so topology is batch-independent, which is what `GGML_SCHED_DEBUG_REALLOC` checks.
- A grep for other batch-dependent shortcuts in recurrent code (`n_seqs == 1`, `n_rs == 1`, `get_n_rs() == 1`) found only these two sites (plus an assert in `delta-net-base.cpp:131` that is guarded by `state_rows == nullptr`). No missed call site found.
- Not covered: default `llama-server` is 4 slots, so Prism's CUDA gather fusions and the Metal rows path are off in the default server configuration. The doc says they "still apply to `--parallel 1`" (true) but does not say they are off by default. No before/after number is given.
- Test gap: `test-llama-context` `rf` (`tests/llama-context/cases-kv.cpp:384`, `n_seq_max = 4`) guards the multi-slot side. Nothing guards that single-slot contexts still take the shortcut (a silent perf loss if someone breaks the condition).

### F5: `01a18bea0` overlap check: two small gaps. Minor/nit, confidence medium
The fix itself is right: it rejects `dst` overlapping `experts`/`expert_scale`/`weights` at both upstream dispatch sites (`ggml-cuda.cu:6138-6139`, `:6164-6165`); the only other callers of `ggml_cuda_op_moe_weighted_reduction` are those two plus the verify branch, and `ggml_cuda_op_moe_weighted_reduce` (Prism's exact path) already checks `x` and `w` at `:5435-5439`. Falling back to the unfused nodes is correct on aliased layouts and costs nothing in real graphs, because `graph_optimize` adds alloc deps for the same pattern (`:7668-7680`), so aliasing never reaches the kernel there.
- (a) The gather-fused path (`GGML_CUDA_MOE_EP_FUSION=1`, `:6135-6139`) passes `nullptr` for `expert_scale` and also reads `table` and `ids` in the kernel; neither is in the disjointness check. A hand-placed `ids` inside `dst` would still race. `table` is a weight and `ids` an allocator-protected intermediate (the alloc-dep code at `:7673-7676` covers `ids`), so this is hypothetical today.
- (b) `src->buffer != nullptr` is used as "skip if unallocated", which treats an unallocated tensor as disjoint. Fine for allocated graphs; nit only.
- Test: `tests/test-moe-weighted-reduce.cpp` only builds the plain `MUL(experts, weights)` form (no `expert_scale`, no EP gather). The `x_in_dst`/`w_in_dst` layouts do cover the reported failure. Adding an `expert_scale` layout would cover the second dispatch branch.
- The claim "the same code path exists on prism/prism" is plausible (the test file exists on `prism/prism`) but I did not diff the dispatcher against that ref.

### F6: Zeroing fix in `build_rs_cache_view` is only reachable on CPU/Metal, so it was necessarily validated on CPU. Nit, confidence high
- `src/llama-graph.cpp:4389`; `qwen35.cpp:163-177` sets `gdn_state_rows_dev_ok = false` for any non-Metal GPU, so CUDA never builds the rows path. The fix is correct (`ggml_fill_inplace(…, 0.0f)` matches `build_rs` at `:4266`, and a zero-sized view is a no-op as in `build_rs`), but the Metal behaviour (which is the path this code exists for) was not exercised on this machine. Say so in the validation table.
- No other `scale(…, 0)`-as-zeroing remains in the recurrent state code. `src/models/metis16.cpp:231` uses `ggml_scale(confidence, 0.0f)` as a zero tensor for a metadata concat; it would propagate a NaN `confidence`, but it is not state and is not part of these commits.

### F7: Auto-detection can report `draft-dspark` for a file whose loader will not load a Markov head. Nit, confidence low-medium
- `common/speculative.cpp:4525-4526` checks both `markov_w1.weight` and `dspark.markov_head_a.weight` regardless of `general.architecture`. The loader picks the tensor name from the architecture (`src/models/dflash.cpp:44`: `legacy_dspark ? "dspark.markov_head_a.weight" : "markov_w1.weight"`). A `dflash`-arch file carrying the legacy tensor name (or a `dspark`-arch file with the modern name) would be reported as DSpark while the loader sees no Markov head. The driver already warns and runs the right path (`:2150-2157`), so the effect is a misleading type string only. `test-dspark-legacy` covers the two consistent pairings only.

### F8: Smaller notes on `12a2cc8c7` and `c55ebc739`
- `ggml-cuda.cu:8465-8469` (supports_op): TURBO2/3/4 and PQ2_0M `SET_ROWS` now accept F32 sources only. If the graph ever produces a BF16 K for these caches (upstream's BF16 activation work), the op is unsupported on the cache's backend instead of crashing in the template, which is the safe choice. I did not trace whether `cpy_k` casts to F32 first; worth a graph-level check, since a refused op on a pre-assigned CUDA KV buffer is a scheduler corner. Confidence low.
- `mmq-pq2-j128.cuh:826-829`: `ncols_max == ncols_dst` and `expert_bounds == nullptr` become `tile_spec == nullptr`; `ids_dst == nullptr` is also required, and dense `ntj = ceil(ne1/J_best)` with `ncols_dst = ne1` (`mmq.cu:291`, `:343-348`), so `ntx = ceil(ncols_dst/128)` is consistent for the J=128 branch. The J128 path does not test `args.fusion` (x_scale/gate); PQ2_0 is not in `ggml_cuda_mmq_fusion_supported` and the generic path asserts `!args.fusion.gate`, so it is unreachable today. A one-line `args.fusion.gate == nullptr && args.fusion.x_scale == nullptr` guard would make it robust. Nit.
- `llama-kv-cache.cpp:1453-1458`: `n_pad_cur = 256` now matches `get_n_kv` (`:1472`). Right.
- `qwen35.cpp:536`: dropping `fused_gdn_ar` makes the state-rows path depend only on `n_seq_tokens == 1`; consistent with upstream making fused GDN unconditional.
- `c55ebc739`: right. The current tree has the gate/precision-aware signature on `ggml_cuda_mmq_load_tiles_pq2_0` (`mmq-load-tiles.cuh:212`) and `mmq.cu:17-18`; no `Q2_0_G128` references remain in the CUDA directory. The stale "Q2_0 group 128" comment at `mmq-load-tiles.cuh:211` is cosmetic.

## Fixes that look right (explicit)
- `01a18bea0`: correct condition, correct fallback, both dispatch sites covered.
- `b09de0b68`: reserve and decode now take identical branches for `n_seq_max == 1` (memory has exactly one cell) and for `n_seq_max > 1`; no missed shortcut site found in `src/`.
- `9bd028f0f` zeroing: correct and consistent with upstream `build_rs`.
- `1d2b2c9bb` batch width: `llama_batch_ext_select_n_embd_inp` returning `n_embd_inp_enc()` for `LLM_ARCH_DSPARK` is right. `dflash.cpp:63` sets `n_embd_inp_enc_impl` for the shared model class, and Prism's opt-in class leaves it unset, so `n_embd_inp_enc()` falls back to `n_embd_inp()` (`llama-hparams.cpp:132-133`). `llama-context.cpp:245` already special-cased `LLM_ARCH_DSPARK`, so this was the last site.
- Cap arithmetic: `min(llama_n_ctx(ctx_tgt), 4096 * max(1, n_parallel))` matches the earlier `result.n_ctx == 0 || > cap` behaviour for any non-zero target context. No off-by-one; 4096 multiples are 256-aligned. The cap itself is right; the issues are in what selects it (F1, F2).

## Design doc and AGENTS.md requirement 7 versus code
All claims checked against the code hold:
- `GGML_TYPE_F8_E4M3 = 43` (`ggml.h:262`), `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42` (`llama.h:161`).
- `gguf_normalize_legacy_private_type` (`ggml/src/gguf.cpp:459-475`) maps only 45 and 46 to TQ3_1S/TQ4_1S; 43 and 44 pass through.
- Raw 44 is rejected because it is below `GGML_TYPE_COUNT` (245) but has no type traits, so `blck_size == 0` triggers the "not a multiple of block size" error (`gguf.cpp:900-906`). `test-gguf` (`tests/test-gguf.cpp:1607-1625`) checks 43/45/46 and the rejection of 44.
- Nit: rejection of 44 is incidental (relies on empty traits) rather than an explicit "unassigned type" check, the error text is misleading, and it prints `ggml_type_name(44)`, which is a NULL `%s` argument. A small explicit check would match the "raw 44 is rejected" wording in AGENTS.md.
- Convergence doc line 133 (`draft_ctx_per_seq=4096`) holds for the single-slot acceptance run only (see F2). Line 84 on chained MTP drafts needs the F3 caveat.

## Missing tests (collected)
1. DSpark 4096 cap: no test, and F1 shows the selector can be wrong.
2. Single-slot recurrent contexts still take the shortcut (perf guard).
3. `test-moe-weighted-reduce`: no `expert_scale` or EP-gather layouts.
4. Auto-detection with arch/tensor-name mismatch (F7).
5. Metal rows-path zeroing (F6) cannot be covered on this machine.
