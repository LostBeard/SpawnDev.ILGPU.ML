# FLUX 3 Image Klein port: analysis (2026-10-19, model released 2026-10-15)

Source: [black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein) (reference code, Apache-2.0, `src/flux3/`, about 60 KB of Python) and the [Hugging Face weights](https://huggingface.co/black-forest-labs/flux-3-image-klein) (FLUX Kommunity License v1.0; the repo is GATED: the licence has to be accepted on the model page with the account whose token this machine uses before any file, even a config, can be read).

## What it is (read from the code, not the card)

- **DiT** (`dit.py`, 5.6B, no biases anywhere): hidden 3072, 24 heads x 128, RoPE axes (32,32,32,32) with **theta 10000** (FLUX.2 klein: 2000), MLP ratio 3 (9216, SwiGLU). Three TOKEN MODES per sample, in sorted order **image | image_cond | text**: `image` (the picture, 128 ch in), `image_cond` (reference images, 128 ch), `text` (20480 ch in = 8 Qwen3-VL layers x 2560).
  - 5 EARLY blocks per mode (`content_mode_blocks[image]`, `[image_cond]`, `txt_mode_blocks`): each mode has its own stack and attends ONLY within itself (`packed_attention` per sample and mode).
  - then 28 SINGLE blocks over the concatenated sample (joint attention over all modes), per-mode modulation.
  - Block = parallel attention + SwiGLU MLP: separate `q_proj k_proj v_proj`, `mlp_in` (2 x 9216: gate | up), QK RMSNorm + RoPE, `attn_out`, `mlp_out`; input `layernorm(no affine, eps 1e-6) * (1 + scale) + shift`; residual `x + gate * (attn_out(attn) + mlp_out(silu(gate) * up))`. Per-mode `ModeModulation` = Linear(3072, 9216) of silu(time embedding) -> shift, scale, gate (early and single separately).
  - time: sinusoid (256, max_period 10000, t * 1000) -> `MLPEmbedder` (in_layer, silu, out_layer). The distilled model is a FLOW MAP: `ref_time_in` (a second MLPEmbedder) of the time each step jumps TO is ADDED to the time embedding; the base model is not (it uses CFG).
  - `final_layer`: LastLayer on the image tokens only (adaLN shift/scale from the image time embedding, no affine layer norm, Linear 3072 -> 128).
- **Text encoder** (`text_encoder.py`): Qwen3-VL-4B-Instruct, text tower only (hidden 2560, 36 layers, same shape as Qwen3-4B); chat template of the processor; hidden states of layers **4, 8, 12, 16, 20, 24, 28, 32** stacked to 20480; tokens padded to a multiple of 80 (max 4096) by right padding with the attention mask, and ALL padded rows go to the DiT (no mask there). Our `Flux2TextEncoder` is the starting point (it takes layers 9/18/27 of a 27-layer cut and pads to 512).
- **Sampling** (`sampling.py`): schedule `t = a * s / (1 + (a - 1) s)`, s = linspace(1, 0, steps + 1), **alpha 16**; Euler `x += dt * v`. Distilled: 16 steps, flow map, no CFG. Base: 50 steps, CFG 4.0 over a 2-sample pack (empty prompt, prompt).
- **Positions** (`pipeline.py`): `ids = (0, row, col, 0)` with row and column **scaled by 0.5** (`IMAGE_ROPE_SCALE`); reference i gets l = -(1 + i). Text ids (0, 0, 0, position).
- **Autoencoder** (`autoencoder.py`): the FLUX.2 VAE layout (ch 128, mult 1 2 4 4, 32-d latent, 2 x 2 patchify, BatchNorm eps 1e-4, 16x total), `ae.safetensors` 336 MB; whether the weights equal FLUX.2's has to be checked.
- **References**: one reference = fitted to <= 4 MP and reflect-PADDED to a 16-multiple canvas (the output is cropped back), several references = each under 1 MP and centre-cropped; up to 10. Sizes 64-4096 (native 4K).
- **Bounding boxes** are TEXT: the caption gets a JSON list `[{"id", "bbox": [y_min, x_min, y_max, x_max] on 0-1000, "desc"}]` appended (edits: `from`, `src_bbox`, `tgt_bbox`, `<id>` cited in the caption). No extra model input. The authors' 9B prompt upsampler writes these captions; "do not use the model without it": any LLM given their system prompts (`pup_prompts_external_vlm.json`) can do it, and we have local LLMs.
- Weights: `model.safetensors` 11.1 GB bf16, `variants/fp8r` 6.3 GB, `variants/nvfp4` (Blackwell only), text encoder 8.9 GB (Qwen3-VL incl. vision tower; the text tower is about 8 GB bf16), VAE 336 MB. Base (CFG) model is a separate repo (`flux-3-image-klein-base`).

## What we have and what is new

| Piece | Status |
|---|---|
| 4-axis interleaved RoPE, RMS QK norm, SwiGLU, adaLN, Q4/Q8 block quantisation, flash attention, tiled VAE decode, reference tokens in the sequence | in the FLUX.2 port |
| Per-mode early stacks with within-mode attention, per-mode modulation, packed joint attention in the single stream | new (a new `Flux3Transformer`) |
| Flow-map time conditioning (`ref_time_in`) | new, one more MLP and an add |
| Qwen3-VL text tower, 8 layer taps, variable length, padded rows kept | adapt `Flux2TextEncoder` (layer taps, tensor names `model.language_model.*`, length multiple of 80) |
| VAE encode/decode | probably the existing ONNX exports if the weights match, else re-export |
| Bounding-box caption builder + prompt upsampling with a local LLM | new, small (host code, `bbox.py` ported) |
| Memory | 5.6B: Q4 ~3.2 GB, Q8 ~6 GB resident on a 12 GB card; 4K = 65,536 image tokens, attention is the cost: expect minutes per step, so 1K and 2K are the comfortable sizes |

## Plan

1. (needs the gated weights) read configs and tensor names; check whether `ae.safetensors` equals FLUX.2's VAE.
2. Port `dit.py` as `Flux3Transformer`, parameterised (hidden, heads, depths, mlp ratio) so it is gated FIRST against the reference code with RANDOM weights at a small size on this machine (the reference classes run on CPU: no gated file needed), then against the real weights.
3. Text encoder, flow-map sampler, pipeline (references, boxes as text), quantised export (`quantize_flux3.py`), HF repo, tests, docs, demo page with up to 10 references.
