# Qwen-Image Edit 2509 (TextEncodeQwenImageEditPlus) on ILGPU.ML

Started 2026-11-15 (Tuvok), for SpawnGen (Lore): ComfyUI's 2509 edit template, from ComfyUI's own split files. The base Qwen-Image text-to-image port is already
done (`QwenImageTransformer`, Qwen2.5-VL text tower in `Flux2TextEncoder`, `qwen_image_vae`; Docs/flux1.md). Parity target: ComfyUI's tensors (Lore's dumps). ComfyUI is
GPL-3: run and measure it only, never read it. Port sources are Apache-2.0: transformers 5.23.0 `models/qwen2_5_vl/modeling_qwen2_5_vl.py` + `vision_utils.py`, and
diffusers 0.43.0 `pipelines/qwenimage/pipeline_qwenimage_edit_plus.py` + `models/transformers/transformer_qwenimage.py`. Unpacked from the wheels (no install) in
`E:\spawndev-qwenimage\refsrc\src`.

## Status

| Piece | State |
|-------|-------|
| Reference sources read | done 11-15 |
| Lore's dumps (`_DevComms` 2026-11-15-1425 + 1450; fixtures F:/spawndev-qwenimage-edit/comfyui-dumps) | done |
| Vision tower `QwenVisionEncoder` (QwenImage/QwenVisionEncoder.cs; `Flux2Weights.LoadVisionAsync`, shared `Flux2ComfyFormat.TakeF8ScalesAsync`) | GATED 11-15 vs ComfyUI: QwenEdit_VisionTower_MatchesComfyUi CUDA 6.1e-5 / OpenCL 2.5e-5 |
| M-RoPE for picture tokens in the Qwen2.5-VL text tower (`Flux2TextEncoder.Encode(ids, real, pictures)`, `MRopePositions`, `SkinTokensKernels.RopeAxes`) | GATED 11-15 vs ComfyUI: QwenEdit_TextEncoderWithPictures_MatchesComfyUi CUDA 3.1e-4 / OpenCL 8.1e-5 (ComfyUI = transformers' get_rope_index) |
| Reference latents in `QwenImageTransformer` (`OpenAsync(.., references, ..)`, `Ids(text, pictures)`, `InputTokens`) | GATED 11-15 vs ComfyUI: QwenEdit_Transformer_MatchesComfyUiCalls 1 picture 3.2e-4 / 1.9e-4, 2 pictures 3.0e-4 / 3.3e-4 (CUDA); ids = ComfyUI's pe_embedder input exactly. Found with SpawnGen's module taps: raw circular refs, text after the noise latent, odd grids centred -(n/2) |
| Qwen-Image Edit 2511 | later: diffusers' `zero_cond_t` (reference tokens modulated at timestep 0, `modulate_index`) |
| Pipeline / node glue for SpawnGen | after the gates |

## What Lore measured in ComfyUI (11-15 message; the dump note follows)

- System prompt: "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should
  alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
  (the same text as diffusers' `prompt_template_encode`).
- User turn: `Picture 1: <|vision_start|><|vision_end|>` [`Picture 2: ...`] `{prompt}`. The embedding table sees vision_start / vision_end adjacent; the merged vision rows go
  between them.
- Rows dropped: the first 64 (system turn + `<|im_start|>user\n`), as diffusers' `prompt_template_encode_start_idx = 64`. 1 picture: 84 ids + 192 vision rows - 64 = 212
  rows; 2 pictures: 91 + 192 + 198 - 64 = 417.
- Vision input size: A 256x192 -> 336x448 px, grid [1, 24, 32]; B 200x120 -> 308x504 px, grid [1, 22, 36]. About 384^2 area, sides multiples of 28 (diffusers rounds to 32
  first, then the processor to 28: ComfyUI differs, so take sizes from the dumps).
- reference_latents: one per picture at about 1 MP (A [1,16,1,111,148] = 888x1184 px, B [1,16,1,99,165]), on cond AND uncond. Multiples of 8, not diffusers' 32.

## The vision tower (transformers, Qwen2.5-VL-7B vision config)

| Part | Fact |
|------|------|
| Patch embed | Conv3d [1280, 3, 2, 14, 14], no bias, stride = kernel: a linear map of each 2 x 14 x 14 x 3 patch (a still picture is repeated to 2 frames by the processor) |
| Blocks | 32 x (x += attn(rms(x)); x += mlp(rms(x))), RMSNorm eps 1e-6 with weight (bf16), width 1280, 16 heads of 80, qkv Linear 1280->3840 + bias, proj + bias |
| MLP | SwiGLU with biases: down(silu(gate(x)) * up(x)), 3420 wide |
| Attention | non-causal. Blocks 7, 15, 23, 31 attend over the whole picture (one segment per picture); every other block over windows of 112 px = 4 x 4 merged tokens |
| Window order | tokens are kept in 2 x 2 merge blocks (spatial index block-major), and the merge blocks are permuted window by window (`get_vision_window_index`; padded windows at the right and bottom edges are smaller). The merger output is permuted back |
| 2D rotary | head dim 80: inv_freq = 1 / 10000^(i / 40), i = 0, 2, ..., 38 (20 freqs); angle h * f for the first 20, w * f for the next 20, the 40 duplicated (rotate_half layout, not interleaved). (h, w) are the PATCH coords before merging, block-major |
| Merger | RMSNorm(1280) then each 2 x 2 group of 4 consecutive tokens -> 5120 -> Linear + bias -> GELU (exact erf) -> Linear + bias -> 3584 |
| File | ComfyUI `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `visual.*`: matrices F8_E4M3 with scale_weight (exact path as the text tower: 8-bit unscaled, the scale on the input), biases / norms BF16, patch_embed BF16 |

Head dim 80: DiTKernels' tiled attention has a d = 80 kernel (TiledWidths); CUDA's tensor-core flash kernel is d = 128 only, so d = 80 runs the fp32 tiled path. The patch embedding keeps
K = 1176 (each pixel written to both temporal slots): folding the two temporal kernels into one bf16 matrix would round the weights.

## M-RoPE in the text tower (transformers `get_rope_index`)

Three position axes (t, h, w), `mrope_section` [16, 24, 24] (of the 64 rotary frequency pairs: pairs 0-15 rotate by t, 16-39 by h, 40-63 by w). Text token: the same running
position p on all three axes, p += 1. A picture of llm grid (gh, gw) = grid / 2 at running position p: token (y, x) gets (p, p + y, p + x); then p += max(gh, gw). For text-only
prompts this is plain 1D RoPE, which is why the base port did not need it. ComfyUI's conditioning dump decides if it does the same.

## Reference latents in the transformer (diffusers)

The sequence is [text | noise latent tokens | ref 1 tokens | ref 2 tokens ...] (diffusers concatenates on the token axis; the prediction is the first rows*cols rows).
Rotary (16 / 56 / 56, theta 10000): picture k (0 = the noise latent, k >= 1 the refs) has frame index k, h and w centred as in the base (scale_rope). Text starts at the max over
ALL pictures of max(h/2, w/2). ComfyUI's ref-latent positions may differ (Kontext-style offsets): the model-call dumps settle it.

## CFG in diffusers vs ComfyUI

diffusers' edit-plus rescales the CFG result to the conditional prediction's norm per token. ComfyUI's KSampler cfg does not (black box: compare on the dumps). The model
calls are gated one by one, so this only matters for the end-to-end sampler gate.

## Wan 2.1 i2v 14B (item 2 of the 0958 request)

CLIP vision ViT-H: `ClipVision/ClipVisionEncoder.cs`, gated (`ClipVisionH_MatchesComfyUi`: penultimate 4.5e-5 CUDA / 4.4e-5 OpenCL). The 14B model calls have no
ComfyUI reference yet: ComfyUI's CPU load of the 14B fp8 file needs more than 36 GB of RAM (SpawnGen's box 32 GB, Yoda 15.7 GB). Encode-side dumps (umT5, WanImageToVideo) are in
F:/spawndev-wan14b/comfyui-dumps.

## ComfyUI's resize rules (measured by SpawnGen, 11-15; use these in the pipeline)

- **Vision picture** (Qwen2.5-VL input): area-resize to round(side x sqrt(384^2 / (w h))), then bilinear (align_corners false) to the nearest multiples of 28,
  no 8-bit step. Rebuilt from ComfyUI's patch rows: 1.3e-6 (256x192 -> 443x333 -> 448x336; 200x120 -> 496x297 -> 504x308).
- **Reference latent picture**: area-resize to round(side x sqrt(1024^2 / (w h)) / 8) x 8, then VAE encode: 1.2e-6 / 1.4e-6 vs ComfyUI's reference latents
  (bilinear / bicubic / lanczos 0.17-0.31 off, nearest-exact 0.09).
- **Transformer input** (SpawnGen's img_in / pe_embedder taps, _DevComms 2026-11-15-1730): the reference latent as the call carries it (not normalised again), odd
  sides wrapped (circular) to even, packed after the noise latent; rotary: noise latent frame 0, reference k frame k, both centred; TEXT STARTS AFTER THE NOISE
  LATENT's max(rows/2, cols/2) (diffusers: after the largest picture). SpawnGen's 2509 encode side on ML local.7, RX 590: conditioning 4.4e-5 / 4.5e-5 / 7.3e-5.
