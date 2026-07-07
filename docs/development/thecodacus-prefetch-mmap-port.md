# thecodacus/llama.cpp — feature extraction + port plan

> **Status: both features IMPLEMENTED in this fork** (AGENTS.md mods #8 expert-prefetch, #9
> mmap-pin incl. the Windows branch). This doc is retained as the rationale/design record. The
> `0001..0003` patch files are the original upstream form; the in-tree edits are hand-adapted to
> this older base and env-gated off by default. Build + A/B test steps are in the checklist below.

Analysis of `github.com/thecodacus/llama.cpp` for two features to bring into this fork, plus a
TurboQuant/MTP consistency check against the Qwen fork.

- **This (atomic) fork:** Gemma stack, active repo.
- **Qwen fork:** `llama.cpp-turboquant-mtp` (builds `C:\Program Files\llamacpp`, runs Qwen).
  From `PippBauda/llama.cpp-turboquant-mtp` = upstream + TurboQuant+ tqp-v0.1.1 + MTP PR #22673.

The two target features live on branch **`origin/fable5/prefetch-experts`** (3 commits). A larger
25-commit `moe-cache` branch implements a full LRU expert cache (`--moe-expert-cache-size`); it is
the heavier experimental route and is **not** recommended over the clean 3-commit version.

Patch files (generated from the branch, applied by a downstream agent):
`0001-llama-pin-mmap-backed-CPU-weights-for-faster-H2D-upl.patch`,
`0002-ggml-overlap-offloaded-expert-weight-uploads-with-co.patch`,
`0003-ggml-size-prefetch-slots-per-layer-and-fix-fallback-.patch`.

---

## Feature 1 — pin mmap-backed CPU weights (the "dead code" wire-up)

**Commit:** `20f5994b` · **Files:** `src/llama-mmap.cpp` (+36), `src/llama-mmap.h` (+8),
`src/llama-model-loader.cpp` (+16).

**What it does.** The CUDA backend already ships `ggml_backend_cuda_register_host_buffer()`
(→ `cudaHostRegister`, gated by env `GGML_CUDA_REGISTER_HOST`) and exposes it as the proc address
`ggml_backend_register_host_buffer` — but **nothing ever calls it**. That is the dead code. The
patch adds `llama_mmap::register_host(...)` and calls it from `llama_model_loader::load_all_data`
after load, page-locking the mmap pages backing CPU-resident weights so H2D copies during
`n-cpu-moe` expert streaming use DMA instead of a staged pageable copy.

**Perf (author, RTX 3060, Qwen3.6-35B-A3B pp2048):** 1144 → 1385 t/s. Opt-in, off by default.

**Confirmed present as dead code in this fork:** `ggml_backend_cuda_register_host_buffer`
(ggml-cuda.cu ~L4587) gated on `getenv("GGML_CUDA_REGISTER_HOST")`, exposed as a proc address
(~L5367), and never called — no `register_host` in `src/llama-mmap.cpp`. Anchor points for the
patch all exist (`llama_mmap::~llama_mmap() = default;` ~L613, `unmap_fragment`, the
`load_all_data` unmap block ~L1676).

**⚠ Windows caveat (blocking for this build).** The patch's `register_host` body is wrapped in
`#ifdef _POSIX_MAPPED_FILES` and uses `sysconf(_SC_PAGESIZE)`. On Windows it hits `#else` and
returns 0 — **a no-op**. `llama-mmap.cpp` has a separate `_WIN32` mmap impl (~L525). To get the
benefit on the Windows/CUDA build, add a Windows branch to `register_host`: page size from
`GetSystemInfo().dwPageSize`, register `pimpl->addr + first` over the mapped view.
`cudaHostRegister`/`cudaHostUnregister` work on Windows; only the page-alignment plumbing is
POSIX-gated. Until that branch is added, Feature 1 is a Linux-only win.

---

## Feature 2 — overlap offloaded expert-weight uploads with compute (expert prefetch)

**Commits:** `1163cb34` (core, +102) and `5f83fbbe` (per-layer slot sizing + use-after-free fix,
+90/−19). **File:** `ggml/src/ggml-backend.cpp` only.

**What it does.** In `ggml_backend_sched`, at large batch (prefill) virtually every expert is used,
so waiting on the per-layer routing-id readback before uploading experts buys nothing but forces a
full device sync per expert tensor (3× per MoE layer). The patch stands up a **second backend
instance on the same device** (own stream) with **event-ordered staging slots**, and for
`GGML_OP_MUL_MAT_ID` weight inputs on a host buffer uploads the full expert tensor async into a
rotating slot — so the upload of tensor N+1 overlaps compute of tensor N. `5f83fbbe` raises the
slot count to 3 (one full MoE layer of look-ahead), sizes slots from the graph max, and fixes a
use-after-free where a failed slot regrow left staging tensors dangling into freed device memory
(graph reuse then carried the bad pointers into later evals).

**Perf (author, RTX 3060, Qwen3.6-35B-A3B pp2048, -ncmoe 26, ub 2048):** 1143 (mainline) → 1663 →
1880 t/s. Output verified token-identical; decode path untouched. Opt-in.

**Gate:** `GGML_SCHED_PREFETCH_EXPERTS=1` **and** `op_offload` true (i.e. `--n-cpu-moe` in use).
Requires a backend with async + events (CUDA qualifies).

**Portability:** platform-agnostic — pure ggml backend/event/stream API, works on Windows/CUDA.
Anchor points confirmed here: `struct ggml_backend_sched` L774, `bool op_offload;` L819,
`ggml_backend_sched_compute_splits` L1534, input-copy loop L1547. This is the **high-value,
low-risk** one — it directly targets the `n-cpu-moe` prefill path this stack bottlenecks on.

---

## Port checklist (execution order)

1. **Feature 2 first (biggest win, platform-clean).**
   - Apply `0002` then `0003` to `ggml/src/ggml-backend.cpp`. Base is newer upstream than this
     fork, so `git apply` will likely reject on offsets — apply by hand against the anchors above
     (struct fields after `bool op_offload;`; `ggml_backend_sched_prefetch_init` before
     `ggml_backend_sched_compute_splits`; the prefetch block inside the input-copy `else`; the
     event-record after graph compute; the getenv in `ggml_backend_sched_new`).
   - Build `llama-server`. Smoke test text-only.
   - A/B with `-ncmoe` set: `GGML_SCHED_PREFETCH_EXPERTS=0` vs `=1`, compare prompt-eval t/s.
     **Verify greedy (temp 0) decode output is byte-identical** on vs off — the author's bar.
2. **Feature 1 (optional; needs Windows work to matter).**
   - Apply `0001` for the POSIX path + loader wiring (inert but harmless on Windows).
   - Add the `_WIN32` branch to `register_host` (GetSystemInfo page size + register the mapped
     view) so it actually pins on this build.
   - A/B `GGML_CUDA_REGISTER_HOST=1` vs unset; expect a smaller pp bump than Feature 2. Watch the
     load log for `pinned N MiB of mapped model memory`.
3. Keep both behind their env gates (already the case) so default behavior is unchanged. Add to
   `AGENTS.md` "Local modifications" if adopted.

### Testing criteria (concrete)
- Feature 2: with `-ncmoe` and a ≥2048-token prompt, `GGML_SCHED_PREFETCH_EXPERTS=1` gives higher
  prompt-eval t/s than `=0`, and greedy decode output is byte-identical between the two.
- Feature 1 (Windows, after the branch): load log prints a non-zero "pinned … MiB" line and
  H2D-bound pp improves with `GGML_CUDA_REGISTER_HOST=1`.
- Regression: with both env vars unset, t/s and outputs match the pre-patch build.

---

## Fork consistency check — TurboQuant vs MTP

**TurboQuant: NOT substantially different — effectively the same source.**
- All five Turbo CUDA kernels byte-identical across the forks: `turbo-innerq.cu/.cuh`,
  `turbo-quant.cuh`, `turbo-wht.cu/.cuh`.
- `ggml-common.h` (type defs), `ggml-cuda/convert.cu`, `fattn-common.cuh`, `fattn-vec.cuh` and all
  `*-turbo*_0.cu` template instances identical.
- Only turbo-adjacent deltas: `src/llama-kv-cache.cpp` (+26/−86, largely this fork's own
  context-shift-with-vision + `get_can_shift` work) and `ggml-cpu/ops.cpp` (+150/−16 CPU turbo
  path); `ggml.h` +11 (enum placement). → turbo-level fixes/knowledge transfer cleanly.

**MTP: substantially different implementations (shared base framework only).**
Both expose `COMMON_SPECULATIVE_TYPE_MTP` and a `common/speculative.cpp`, but diverge below that:

| | Atomic fork | Qwen fork (turboquant-mtp) |
|---|---|---|
| Approach | Runtime graph-pass MTP | Dedicated model architectures |
| Graph type | `LLM_GRAPH_TYPE_MTP` (own pass) | none (DECODER/DEFAULT/ENCODER only) |
| Heads | `gemma4-assistant.cpp`, `qwen35-nextn.cpp` loaded into target | `qwen35_mtp` / `qwen35moe_mtp` archs w/ NextN tensors (`blk.N.nextn.*`) |
| Spec types | `MTP` **and** `NEXTN` | `MTP` only |
| Runtime | `decode_mtp` / `sched_mtp` in `llama-context` | arch builds NextN predict layers; `kv_only_nextn` |
| CLI | `--spec-type mtp/nextn`, `--mtp-head`, `--draft-block-size` | generic `--spec-*` |

Implication: the atomic MTP (general spec runtime; Gemma assistant + Qwen NextN) and the Qwen-fork
MTP (arch-based NextN) are **not interchangeable**; unifying them would be real work. It does not
affect the port — both target features live in `ggml-backend.cpp` and the mmap loader, independent
of MTP and TurboQuant, so they drop into either fork identically.
