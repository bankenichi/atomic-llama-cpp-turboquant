# Gemma 4 12B Unified — getting vision working

## TL;DR (the working config)

Run the **heretic finetune LM** + the **official unsloth `gemma4uv` mmproj** (not the
finetune's own mmproj), with vanilla `q8_0` KV:

```
llama-server --model gemma-4-12B-...-heretic.i1-Q4_K_M.gguf \
             --mmproj gemma4-12b-unsloth-mmproj-F16.gguf \
             --n-gpu-layers 99 --ctx-size 32768 --flash-attn on --parallel 1 \
             --cache-type-k q8_0 --cache-type-v q8_0 \
             --temp 1.0 --top-p 0.95 --top-k 64 --jinja
```

In the launcher this is just `run-gemma.ps1 -g12`. Put the **image before the text** in the
prompt (Gemma 4 best practice; audio goes after).

## What Gemma 4 12B Unified is

A June-2026 Google model that is **encoder-free**: the `gemma4uv` projector maps image (and
audio) patches *directly* into the LLM embedding space — there is no SigLIP/ViT tower. So the
mmproj is tiny and has `clip.vision.block_count = 0`, ~11 tensors, ~120–175 MB. **That is
correct and complete**, not a broken/stripped file. (Contrast the 26B MoE, which uses the
older `gemma4v` projector *with* a 27-layer tower, ~1.2 GB.) The fork's runtime has dedicated
graph builders for both: `clip_graph_gemma4uv` (no transformer blocks, just patch-embed +
positional + projection) and `clip_graph_gemma4v`.

## The bug we hit (and what it was NOT)

Symptom: text worked, but the first image turn produced `<unused49>` spam (garbage).

It was **not** the mmproj being "incomplete," and **not** the fork's runtime. It was a
**GGUF conversion bug** in the broader Gemma 4 ecosystem, fixed upstream by
[PR #24118 "Fix Gemma 4 Unified conversion"](https://github.com/ggml-org/llama.cpp/pull/24118)
(June 2026). That PR is conversion-side only (~15 lines in the converter); the runtime did not
change. Its part 3 corrects the **patch-projection weight permutation** (`patch_dense` /
`patch_ln1`), computing `p = patch_size * pooling_kernel_size` when `model_patch_size` is
absent. Get `p` wrong and the image patch columns are permuted into noise → exactly the
`<unused49>` failure. Those tensors live in the **mmproj**, which is why text was fine but
vision was garbage: the finetune's own mmproj (from a community pre-#24118 pipeline) had the
patches permuted wrong.

Our fork's `convert_hf_to_gguf.py` already contains all three parts of #24118
(`:7914`, `:7991`, `:8016-8019`, `:8030-8033`), so the fork can convert these models
correctly — the broken artifact was the third-party mmproj, not our tooling.

## Why swapping in the official mmproj works

The projector is derived from the frozen base Gemma 4 12B and carries **no alignment or refusal
behavior** — it only maps modality patches into the 3840-dim embedding space. The "heretic"
abliteration only perturbs the LM's refusal directions, leaving the input-embedding space
intact. So the correctly-converted official mmproj projects straight into the heretic LM's
space: **uncensored text + working vision, no re-conversion needed.** Verified: image parsing
works with no refusals.

## Also required: the `media_marker` template fix (Fix 5)

This finetune's embedded chat template advertises support for *neither* string nor typed
content, so `render_message_to_json` (`common/chat.cpp`) was handing content over as typed
parts; the template has no `media_marker` branch and silently dropped the image marker
(`number of bitmaps (1) does not match number of markers (0)`). Fix 5 makes the neither-caps
case fall back to string content (marker preserved inline). See
[`mtp-cpumoe-vision-fixes.md`](mtp-cpumoe-vision-fixes.md) Fix 5 and AGENTS.md mod #7. Vision
on the 12B needs both this fix *and* the correct mmproj.

## Plan B — re-convert from the finetune's safetensors (if ever needed)

The finetune is published as the full `Gemma4UnifiedForConditionalGeneration` model in
safetensors (the vision/audio embedder weights are baked in). If a correct official mmproj
weren't available, re-convert one with the fork's (already-#24118) converter:

```
huggingface-cli download llmfan46/gemma-4-12B-...-heretic --local-dir C:\llm-work\heretic-hf
python convert_hf_to_gguf.py C:\llm-work\heretic-hf --mmproj --outfile heretic-mmproj-F16.gguf
```

`--mmproj` emits only the projector (vision+audio); the existing text GGUF is fine as-is.

## Notes

- Avoid going below ~Q4_K_M on the *text* weights for this model — the Gemma 4 tokenizer/quant
  path was fragile early on; Q4_K_M+ is the safe floor.
- KV: the 12B is small (~7 GB weights), so VRAM is not the constraint TurboQuant exists to
  solve. `-g12` uses plain `q8_0` KV (the verified config) rather than TurboQuant.
