# Runtime performance features & usage

Fork-added performance toggles and speculative-decoding behavior, with exact usage. All are
runtime-controlled (env vars / CLI flags); defaults preserve upstream behavior except where noted.
The launcher `run-gemma.ps1` (gitignored) wraps these — the flag map is at the bottom.

Related docs: [`mtp-cpumoe-vision-fixes.md`](mtp-cpumoe-vision-fixes.md) (MTP/vision fixes),
[`thecodacus-prefetch-mmap-port.md`](thecodacus-prefetch-mmap-port.md) (prefetch/mmap port rationale),
[`gemma4-12b-vision.md`](gemma4-12b-vision.md) (Gemma 4 12B vision).

## Quick reference

| Feature | Enable | Requires | Notes |
|---|---|---|---|
| Expert-upload prefetch | env `GGML_SCHED_PREFETCH_EXPERTS=1` (value = slot count, default 3) | `--n-cpu-moe` (op_offload); keep `--no-mmap` | Overlaps expert H2D with compute during prefill. Biggest prefill win. |
| mmap host-pin | env `GGML_CUDA_REGISTER_HOST=1` | mmap on (i.e. **omit** `--no-mmap`) | Pins mmap CPU weights. Alternative to `--no-mmap`; **negates prefetch** — don't combine. |
| NextN sync multi-step | default on | `--spec-type nextn --model-draft <gguf>`, `--draft-max N` | Drafts N tokens/round. Async single-step is opt-in via `LLAMA_PIPELINE_DEPTH2=1`. |

---

## 1. Expert-upload prefetch (`GGML_SCHED_PREFETCH_EXPERTS`)

**What:** during large-batch prefill, uploads offloaded `MUL_MAT_ID` expert tensors through a
second same-device backend instance (own stream) into rotating event-ordered staging slots, so the
upload of tensor N+1 overlaps compute of tensor N instead of blocking on the routing-id readback.
Ported from `thecodacus/llama.cpp`. See AGENTS.md mod #8.

**Enable:** `set GGML_SCHED_PREFETCH_EXPERTS=1` (cmd) / `$env:GGML_SCHED_PREFETCH_EXPERTS="1"` (PS).
Value is the staging-slot count: `1` → default 3 (one MoE layer of look-ahead); higher = more
overlap at the cost of one max expert-tensor of VRAM per slot.

**Requires:** `op_offload` active (i.e. `--n-cpu-moe N`) and a backend with async + events (CUDA).
Only activates on the offloaded-expert path; no effect otherwise.

**Interaction:** the async overlap rides the `cudaMallocHost` pinned buffer that `--no-mmap`
produces. Keep `--no-mmap` on. Do **not** combine with mmap host-pin (below) — under mmap the
prefetch gain disappears.

## 2. mmap host-pin (`GGML_CUDA_REGISTER_HOST`)

**What:** after model load, `cudaHostRegister`s the mmap pages backing CPU-resident weights (via
the backend's `ggml_backend_register_host_buffer` proc — previously dead code) so H2D expert
streaming uses DMA. Ported from `thecodacus/llama.cpp`; a fork-added `_WIN32` branch makes it work
on Windows (the upstream code was POSIX-only). See AGENTS.md mod #9.

**Enable:** `$env:GGML_CUDA_REGISTER_HOST="1"` **and** run with mmap on — i.e. **omit** `--no-mmap`
(register_host only runs inside `if (use_mmap)`). Look for `pinned NNNN.NN MiB of mapped model
memory` in the load log to confirm it engaged.

**When to use:** rarely. `--no-mmap` already puts CPU experts in a pinned `CUDA_Host` buffer, so
pinning gives no extra prefill speed — its only value is keeping H2D fast while using mmap (lower
RAM / faster cold start). It **negates the prefetch win**, so use it *instead of* prefetch, not
alongside (`run-gemma.ps1` warns if both are set).

## 3. NextN (Qwen) speculative decoding — synchronous multi-step

**What:** `common_speculative_state_nextn` drafts up to `--draft-max` tokens per round via a
synchronous autoregressive loop (the reference MTP behavior). The class also contains an async
"depth-2" worker that overlaps one draft with the next verify but **caps drafting at 1 token/round**;
it is no longer the default. See AGENTS.md mod #10.

**Enable:** `--spec-type nextn --model-draft <same-gguf>` (the Qwen MTP/NextN GGUF re-loaded with
`override_arch`), plus `--draft-max N` for the draft depth (**2 is the sweet spot**; deeper drafts
lose acceptance). Default on; nothing extra needed beyond the flags.

**Async worker (opt-in):** `set LLAMA_PIPELINE_DEPTH2=1` re-enables the single-step async overlap
path. Not recommended on CUDA (single-step underperforms multi-step); kept for future work on a
proper multi-step async pipeline.

**Scope:** NextN-only. Gemma MTP (`--spec-type mtp`, `common_speculative_state_mtp`) is a separate
class and is unaffected by any of the above.

**Direct-run caveat:** without `--draft-max`, the NextN default depth is 16 (over-drafts). Always
pass `--draft-max 2` when running `llama-server` directly; the launcher sets it for you.

---

## Launcher flag map (`run-gemma.ps1`)

| Wrapper flag | Underlying |
|---|---|
| (default, MoE) | `GGML_SCHED_PREFETCH_EXPERTS=1` — prefetch on for 26B/Qwen |
| `-NoPrefetch` | unset `GGML_SCHED_PREFETCH_EXPERTS` |
| `-PinHost` | `GGML_CUDA_REGISTER_HOST=1` **and** drops `--no-mmap`; warns if prefetch also on |
| `-Qwen` | `--spec-type nextn --model-draft <qwen gguf> --draft-max <DraftBlockSize> --context-shift`, `--n-cpu-moe 28`, turbo4/turbo3 KV |
| `-Qwen -NoSpec` | Qwen without speculative decoding |
| `-DraftBlockSize N` | Qwen: `--draft-max N` (draft-token count). Gemma MTP: `--draft-block-size N` (drafts N-1) |
| `-g12` | dense Gemma 4 12B + unsloth mmproj, `q8_0` KV, no MoE/MTP/prefetch |
| `-Mtp` | Gemma 26B `--spec-type mtp --mtp-head <assistant>` (unaffected by the NextN change) |
