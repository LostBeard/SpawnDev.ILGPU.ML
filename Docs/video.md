# Video generation: Wan 2.2 TI2V-5B

Text to video and image to video on ILGPU, desktop (CUDA, OpenCL) and in the browser (WebGPU). `Wan22Pipeline` (namespace `SpawnDev.ILGPU.ML.Wan`) runs the model Alibaba released as
Wan2.2-TI2V-5B (Apache-2.0): the umT5-XXL text encoder, a 5B-parameter diffusion transformer (DiT) with classifier-free guidance and the UniPC flow sampler, and the Wan2.2 video VAE
(4x16x16 compression). One model does both modes: with a first-frame picture the first latent frame is held fixed at every step and the DiT gets per-token timesteps.

**Turbo.** `wan22_ti2v_5b_dit_turbo_q8.safetensors` is the community step-distilled Wan2.2-TI2V-5B-Turbo (quanhaol; trained with Self-Forcing / DMD for 4 steps and guidance 1): one DiT pass per step,
no negative prompt, `WanRequest { Steps = 4, GuidanceScale = 1, Sampler = WanSampler.FewStep }` (predict x0 from the velocity, re-noise it with fresh noise to the next sigma, the last step returns x0).
It is the default of the demo page and the examples (`--turbo`): 1280x704 with 33 frames takes 79 s on an RTX 4070 (the standard model: 203 s for 20 steps, and less sharp), and it runs at 832x480 and 1280x704
in the browser (see Measured). It is sharpest at its native 1280x704; at 640x352 it is rough. The model card names no licence (the base model is Apache-2.0; a mirror of the Turbo weights lists Apache-2.0).

The stages run one after another so only one large model is on the device at a time (the umT5 streams one layer at a time), which is what lets a 12 GB card, and a browser tab, run it.

```csharp
var pipe = new Wan22Pipeline(accelerator, Wan22Paths.InFolder(@"D:\Projects\wan"));                 // files on disk
// or: new Wan22Pipeline(accelerator, modelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo())      // IModelSource: models host + cache, browser OPFS, local folder
var video = await pipe.GenerateAsync(new WanRequest { Prompt = "A red fox walking through fresh snow", Width = 640, Height = 352, Frames = 17, Steps = 30, Seed = 7 });
byte[] rgba = video.GetFrameRgba(0);       // frames are floats in [-1, 1], channel-first; WanVideo.Latent is the denoised latent
```

`WanRequest.ImageRgb` (+ `ImageWidth`, `ImageHeight`) makes it image to video; `InitialNoise` reproduces another implementation's starting noise; `Trace` logs the VAE stage times.

## Files

Exported once from the reference checkpoints with `tools/wan-harness/reference` (`export_dit_bf16.py`, `quantize_streaming.py`, `export_vae_f16.py`, `export_t5_bf16.py`), SafeTensors:

| File | Size | Notes |
|------|------|-------|
| `wan22_ti2v_5b_dit_q8.safetensors` | 5.6 GB | 8-bit blocks (the FLUX.2 block format, dequantised one matrix at a time). Matches the 16-bit model within its own rounding |
| `wan22_ti2v_5b_dit_turbo_q8.safetensors` | 5.6 GB | the Turbo (step-distilled) transformer, 8-bit; exported from the community fp16 checkpoint (`export_dit_bf16.py OUT SRC`, then `quantize_streaming.py`) |
| `wan22_ti2v_5b_dit_turbo_q4.safetensors` | 3.1 GB | the Turbo, 4-bit (`quantize_streaming.py ... 4`): for 8 GB GPUs. One forward is further from the bf16 Turbo than the standard model's quantisations are from theirs (8-bit 1.6e-2 / 5.7e-2, 4-bit 0.15 / 0.37, text / first frame; `Wan_TurboDiT_Quantized_MatchesBf16`), yet its 4-step clips look like the 8-bit one's (same subject and motion at 1280x704 and with a first frame at 832x480; 10/255 mean pixel difference): the few-step sampler predicts x0 every step |
| `wan22_ti2v_5b_dit_q4.safetensors` | 3.1 GB | 4-bit; visibly softer |
| `wan22_ti2v_5b_dit_bf16.safetensors` | 10 GB | as released |
| `wan22_vae_f16.safetensors` | 1.4 GB | |
| `umt5_xxl_enc_q8.safetensors` | 6.4 GB | 8-bit blocks, the embedding table too |
| `umt5_xxl_enc_bf16.safetensors` | 11.4 GB | |
| `google/umt5-xxl/tokenizer.json` | 16 MB | Unigram tokenizer, ported (`UnigramTokenizer`) |

`models.spawndev.com/LostBeard/Wan2.2-TI2V-5B-ILGPU` serves the 8-bit and 4-bit DiT, the VAE, the 8-bit umT5 and the tokenizer (`Wan22Pipeline.DefaultRepo`; the 16-bit files are not hosted, use
`Wan22Paths.InFolder` for a local copy of those).

## Parity with the reference

Every stage is checked against tensors dumped from the reference implementation on identical inputs (`MLTestBase.WanTests.cs`, `HeavyModel`):

| Stage | Result |
|-------|--------|
| DiT noise prediction (bf16 / 8-bit) | 3.5e-3 text to video, 1.0e-2 image to video relative RMS, the reference's own bf16 deviation; 4-bit 3.4e-2 / 7.0e-2 |
| UniPC sampler | exact timesteps; 7e-8 / 2e-6 against the reference when run in its bf16 autocast (`EmulateReference`) |
| 30-step trajectory, teacher-forced | worst 2.6e-2 per step; free-running ends 0.11 away (the reference diverges from itself by about that for a 1.3e-3 perturbation) |
| VAE decode / encode | 1.6e-4 to 4.3e-4 / 1.1e-4 to 2.6e-4 |
| umT5 (bf16 / 8-bit) | 2.3e-2 / 2.6e-2 to 3.4e-2 against the bf16 reference, which is itself 2.2e-2 to 2.8e-2 from exact math |
| tokenizer | identical ids |
| whole pipeline, same prompt and starting noise | decoded video relative RMS 0.11 (8-bit text encoder) to 0.18 against our decode of the reference's final latent |

## Measured

RTX 4070 12 GB, Chrome 156 WebGPU, 640x352, 17 frames, 30 steps, 8-bit DiT and 8-bit umT5: **176 s** (text 33 s, DiT 122 s, VAE decode 21 s). The browser run reads the weights by HTTP range
requests / OPFS and the final latent matches the reference to 0.225 relative RMS (the desktop run: 0.114).
Turbo in the browser (4 steps): 832x480 x33 frames 131 s (text 33 s, DiT 47 s, decode 51 s); 1280x704 x17 frames 166 s (text 27 s, DiT 50 s, decode 89 s).
Desktop CUDA, `Examples/11.VideoGen.Console --ci` (320x192, 9 frames, 8 steps): 38 s, 14 s of it the text encoder.

**MagCache** (`WanRequest.MagCache = new WanMagCache()`, console `--magcache [threshold]`, `/video` on by default for the standard transformer; Ma et al., NeurIPS 2025): a call whose
blocks' residual the calibrated magnitude ratios predict reuses the last one instead of running the 30 blocks. MagCache's own rule, thresholds and TI2V-5B tables
(github.com/Zehong-Ma/MagCache, Apache-2.0); the skip pattern matches MagCache's code for 30 to 50 steps (`WanMagCache_Decisions_MatchMagCache`). Our own calibration
(`WanRequest.MeasureMagRatios`) reproduces MagCache's table to a mean of 0.0017 (step 25: 0.98956 vs 0.98959), so the table holds for this engine and sampler.
MEASURED, RTX 4070 CUDA, 640x352 x17, 50 UniPC steps, guidance 5: plain 75.2 s; threshold 0.04 skips 42 of 100 calls, 54.1 s (1.39x), 28.6 dB PSNR against the plain
clip; 0.12 skips 52, 47.3 s (1.59x), 27.6 dB. In the browser (Chrome WebGPU, `Wan_MagCache_OverHttp`), 30 steps: 26 of 60 calls skipped, 139 s -> 97 s
(1.43x), 24.2 dB (fewer steps: each skip covers more), so `/video` runs the standard transformer at the reference's 50 steps with MagCache, about the cost of 30 plain
steps. One extra [tokens, 3072] buffer per guidance branch. Not for Turbo (4 steps: nothing to skip).

## Limits

- **Memory.** The model's own size, 1280x704 with 121 frames (5 s), runs on a 12 GB card: 2 steps took 224 s (text 23 s, DiT 85 s, about 21 s per forward at 27,280 tokens, VAE decode 116 s), so a
  50-step video is about 35 minutes of DiT. What makes it fit: the decoder keeps its causal caches (two input frames per convolution) as fp16 pairs, decodes the last two stages a frame at a time
  when a chunk is big (`WanVae.MaxSliceBytes`; identical output, tested), and assembles the picture on the host chunk by chunk. VAE decode of 1280x704 x17 frames went from 175 s (VRAM spill) to 17 s.
  In the browser the transformer's time is the limit (about 7 s per forward at 832x480 x33), not memory.
- **Peak GPU memory per stage** (RTX 4070, CUDA, Turbo 8-bit DiT, 8-bit umT5, over the baseline; `tools/ltx-harness/gpumem.py`): 832x480 x49 with a first frame: umT5 1.1 GB, VAE encode
  1.9 GB, DiT 6.9 GB, VAE decode 5.4 GB; 1280x704 x33: umT5 1.4 GB, DiT 7.4 GB, VAE decode 8.4 GB (10.2 GB before the decoder's temporaries were recycled by size, 2026-10-30). So
  832x480 fits an 8 GB card (the 4-bit DiT, 3.1 GB of weights, leaves more room). At 720p the decoder's causal caches (two frames per convolution at full resolution, as the
  reference keeps them) dominate the untiled decode, so `WanRequest.TiledDecode` (console `--tiled-vae`) decodes in tiles as diffusers' `AutoencoderKLWan.enable_tiling()` does
  (256 px tiles, 192 px stride, blended in the 12-channel space before the pixel unshuffle): the 720p decode needs 2.5 GB instead of 8.4 GB (57 s instead of 37 s: the overlap is
  decoded twice). 1280x704 x33 with the 4-bit DiT and the tiled decode peaks at 4.8 GB, so the model's own size runs on an 8 GB card; the 8-bit Turbo DiT alone needs 7.4 GB there,
  the 4-bit Turbo with the tiled decode 4.8 GB (94 s on the RTX 4070: `--turbo --dit turbo_q4 --tiled-vae`). The models host now carries a README with the per-file change notices
  and the Apache-2.0 text (Wan 2.2's licence), which the repo was missing.
  The tiled decode matches diffusers' tiled decode to 2.9e-4 / 2.5e-4 (CUDA / OpenCL; the untiled gate's accuracy), with the same tiling effect as the reference (1.769e-2 from its
  untiled decode, `Wan_VaeDecodeTiled_MatchesReference`; red-checked without the vertical blend: 8.7e-2).
- **Seeds.** At small sizes the model itself gives good clips for some seeds and garbage for others (the reference does too: seed 9 at 640x352 is noise-like where seeds 7 and 8 draw a fox). Judge the
  pipeline only with the reference's noise (`WanRequest.InitialNoise`), as the tests do.
- Our noise is not torch's generator, so a seed here is not a seed there.

## Where it is used

`/video` in the demo (text or image to video, a frame player and WebM download), `Examples/11.VideoGen.Console` (PNG frames and an animated WebP) and `Examples/12.VideoGen.Wpf`.
Tests: `Wan_*` in `MLTestBase.WanTests.cs` (desktop) and `Wan_Pipeline_OverHttp_MatchesReference`, `Wan_Vae_OverHttp_DecodeTimes` in `MLTestBase.WanHttpTests.cs` (WebGPU lane; serve the folder with
`node tools/unirig-harness/serve-files.mjs D:/Projects/wan 8767` first); the demo page by `tools/audit-scenarios/video.mjs`.

# Video generation: LTX-Video 2B (0.9.8 distilled)

`LtxPipeline` (`SpawnDev.ILGPU.ML.Ltx`): text to video, image to video and video conditioning with Lightricks' LTX-Video 2B distilled model: a 2B-parameter DiT (28 blocks, width 2048, every latent position is a token, RMS norm + adaLN,
RoPE over the whole 2048-wide q and k, qk RMS norm, cross-attention to the T5), the T5-v1.1-XXL text encoder (the umT5 encoder class with the shared position bias of block 0), the 7-step distilled flow-matching
Euler sampler (guidance 1: ONE forward per step, the latent never leaves the device), and the timestep-conditioned 32x32x8 video VAE decoder (128 latent channels; the decode adds a little noise at timestep 0.05,
as the recipe does). A 704x480 clip of 97 frames is only 4,290 tokens, which is why this model suits a tab.

**The two-pass recipe** (`LtxRequest.Upscale`, configs/ltxv-2b-0.9.8-distilled.yaml): a first pass at 2/3 of the size (side x 0.6666666, floored to a multiple of 32, as the
official script), the x2 spatial latent upscaler (`LtxLatentUpsampler`: ltxv-spatial-upscaler-0.9.8, 3D convolutions to 512 channels, GroupNorm resblocks, a per-frame 2x2 pixel shuffle,
then AdaIN to the first pass's per-channel statistics), and a second pass at twice the first-pass size from the upsampled latent re-noised to sigma 0.9094 (`LtxRequest.Latents`,
sigmas 0.9094 / 0.7250 / 0.4219). 704x480 comes out at 896x576 (the official script then resizes the frames to the request). The prompt is encoded once; conditions are resized for
each pass. `UpsampleLatentAsync` and `DenoiseAsync` with `Latents` / `Sigmas` expose the stages.

**Conditioning** (diffusers' `LTXConditionPipeline`): `LtxRequest.Conditions` takes pictures or clips (`LtxCondition`: RGB bytes, scaled to cover the size and centre-cropped, or ready pixels;
`FrameIndex`, `Strength`). `LtxCondition.Image(rgb, w, h)` at frame 0 is image to video: the VAE encoder (`LtxVaeEncoder`, the causal 3D encoder, 694M parameters, fp16) turns the picture into the
first latent frame, whose tokens then see timestep 0 (`min(t, (1 - strength) * 1000)` per token) and are reset before every step to their clean latent plus `ImageCondNoiseScale` (0.15) times
(t / 1000)^2 noise; the time positions follow the reference's causal fix (pixel frame max(8 f - 7, 0)). A condition at a later frame becomes extra tokens placed at that time (keyframes); a clip
longer than two latent frames also replaces the latent frames after its first two. Without conditions the pipeline is the text-to-video `LTXPipeline` as before. `PromptEmbedding` takes a T5 output
instead of the prompt (no text encoder loaded), `DenoiseAsync` returns the latent before the decode, and `PosteriorNoise` / `ConditionNoise` / `StepNoise` take another implementation's draws.

Files (`tools/ltx-harness/reference`): `export_dit_bf16.py` (the single-file checkpoint to the Wan layout, q/k/v fused; 3.6 GB; `quantize_streaming.py` makes the 8-bit one, 2.0 GB), `export_t5_bf16.py` (8.9 GB / 8-bit
5.0 GB), `export_vae_f16.py` (the decoder, 1.0 GB), `export_vae_encoder_f16.py` (the encoder, 1.4 GB, loaded only with conditions), the T5 `tokenizer.json`. `export_upsampler_f16.py` (the x2 latent upscaler, 253 MB, loaded only for the two-pass recipe). `LtxPaths.InFolder` / `InRepo` name them. models.spawndev.com serves them as `LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU` (`LtxPipeline.DefaultRepo`: bf16 and 8-bit DiT, 8-bit T5, VAE decoder and encoder, the latent upscaler with its own licence file, tokenizer, the licence and the change notices).

## Parity with the reference (diffusers, same inputs; `MLTestBase.LtxTests.cs`)

| Stage | Result |
|-------|--------|
| tokenizer | identical ids |
| RoPE table | the reference's float32 table to 1.9e-3 (one float32 ulp of a 15,708 rad frequency); exactly as far from the double-precision truth as the reference itself |
| T5 (bf16 / 8-bit) | 1.5e-2 to 3.9e-2 / up to 6.8e-2 against the bf16 reference |
| DiT noise prediction | **4.4e-5 / 7.3e-5 against the float32 reference with the bf16 tensor cores off (exact math), 600 / 2400 tokens**; 1.5e-2 / 8.5e-3 with them on. The bf16 reference is itself 5.2e-2 / 4.0e-2 from exact math (outlier activations of 2500 in the middle blocks), so the bound is the float32 reference. 8-bit DiT: 4.4e-2 |
| sampler loop, teacher-forced | worst 7.5e-3 / 4.5e-3 per step against the float32 reference (7 steps) |
| VAE decode | 1.4e-4 (video), every stage tap 7e-5 to 3.3e-4 |
| VAE encode (float32 reference on the CPU) | latent 1.9e-4 (a 256x192 picture) / 1.7e-4 (a 17-frame 128x96 clip: the temporal downsamplers' first-frame repeat), every stage tap 3e-5 to 1.4e-4 |
| conditioned DiT (`LTXConditionPipeline`'s own calls: 150 tokens at t 0, 450 at t) | **3.5e-5 / 1.0e-5 at steps 0 / 5 with the tensor cores off**; 1.1e-2 / 6.7e-3 with them on |
| keyframe (picture at 0 + picture at frame 16, strength 0.7: 150 extra tokens) / clips (17 frames at frame 8 + 9 frames at 0, strength 0.6: 300 extra tokens, latent frame 3 replaced) | worst latent frame 2.3e-4 / 7.2e-6 (exact math, CUDA and OpenCL; `Ltx_KeyframeAndClipConditions_MatchReference`) |
| x2 latent upscaler (diffusers' LTXLatentUpsamplePipeline, float32, AdaIN 1) | 8.4e-4 / 7.1e-4 CUDA / OpenCL = the fp16 export's own rounding (7.05e-4); **4.1e-6 against the reference with fp16-rounded weights**; every stage tap 2e-4 to 2e-3 |
| two-pass recipe (pipelines chained, 288x192 -> 576x384 x 25, a first frame) | each stage on the reference's input: **upscaler 6.5e-6, second pass 5.9e-5** (browser 6.5e-6 / 5.7e-5); chained from our own first pass 1.4e-3 CUDA / 5.3e-3 OpenCL |
| image to video, the whole sampler (480x320 x 25, the reference's draws) | condition latent 5.6e-6, **final latent 1.4e-4 with the tensor cores off** (OpenCL 1.2e-4, browser WebGPU 1.9e-4); 5.2e-2 with them on (each step 1.1e-2; the reference's own bf16 run is 5e-2 off on one step). The hosted 8-bit DiT: 1.2e-1 after the 7 steps, identical on CUDA and in the browser (the quantisation's cost, not a backend's) |

Red checks: q RoPE off (DiT 0.84), unpatchify offsets swapped (VAE 0.18), a wrong T5 bucket distance (0.14 / 0.28), centred instead of causal time padding in the encoder (0.7), one timestep for every
token run (1.26), the global timestep for conditioned tokens (0.80), keyframe tokens placed at frame 0 instead of their frame (1.35 / 0.51), the upscaler's pixel shuffle transposed (0.72).

At small sizes the sampler amplifies tiny differences along its trajectory: at 288x192 the REFERENCE itself, with only its VAE encoder rounded to fp16 (its condition latent moved
2.5e-6), ends 9.7e-4 from its own float32 run (its step-5 prediction 1.0e-3). Our end-to-end figures at that size (7e-4 to 5e-3) are that sensitivity: each stage fed the
reference's own input matches to 6e-6 to 6e-5, and the same DiT forward on random 1e-5 input noise moves only 4e-5 (`Ltx_ImageToVideo_Steps_MatchReference` records it).

The image-to-video reference (`ref_condition_dump.py`) runs the real `LTXConditionPipeline` and must run with TF32 off: PyTorch's cuDNN convolutions default to TF32 on the GPU, which put 8e-4 of
error into the reference's own encoding of the picture; the model grew that to 8e-3 over the steps, which looked like a port error until the reference was made float32 for real.

## Measured

RTX 4070 12 GB, CUDA, `Examples/11.VideoGen.Console --ltx --models D:\Projects\ltx`: 704x480 x 97 frames **51 s** (T5 streamed from disk 24 s, DiT load 13 s, 7 steps 9 s, VAE decode 5 s); 480x320 x 49 frames 43 s
(30 s with the 8-bit DiT and T5: `--dit q8 --text q8`; the clips look the same).

Browser (Chrome WebGPU, RTX 4070, the hosted 8-bit DiT and T5 read over HTTP): 704x480 x 97 frames **72 s** (T5 26 s, DiT 28 s, decode 18 s; 75 s re-measured after the conditioning changes). Image to video at 480x320 x 25 (the gate, prompt embedding given): denoise incl. the VAE encode 15 s with the 8-bit DiT, 29 s with the bf16 one (`Ltx_ImageToVideo_OverHttp_MatchesReference`).
The two-pass recipe with a first frame, 704x480 x 97 -> 896x576 (the demo's configuration, 8-bit DiT and T5): **103 s** in the browser (T5 27 s, first pass 15 s, upscaler 2 s, second
pass 20 s, decode 29 s; 92 s after the decode's buffer recycling below; `Ltx_MultiScale_OverHttp_MatchesReference`), **35 s** on CUDA (`Examples/11.VideoGen.Console --ltx --upscale --image cat.jpg --dit q8 --text q8`). Every stage gated in the browser against the reference (`Ltx_Stages_OverHttp_MatchReference`: T5 1.5e-2 to 5.1e-2, DiT step 0 vs fp32 2.8e-2, VAE 1.4e-6); `Ltx_Pipeline_OverHttp_Generates` runs the clip (serve the model folder: `node tools/unirig-harness/serve-files.mjs D:/Projects/ltx 8771`).

## GPU memory (smaller devices)

Peak device memory per stage, RTX 4070 via CUDA, the hosted 8-bit DiT and T5, over the memory in use before the run (`tools/ltx-harness/gpumem.py`, nvidia-smi every 100 ms):

| stage | text to video 704x480 x 97 | two-pass with a first frame, 704x480 x 97 -> 896x576 |
|---|---|---|
| T5 (streamed a layer at a time) | 0.7 GB | 0.4 GB |
| VAE encoder (the picture, per pass) | | 1.5 GB |
| DiT (2.2 GB of 8-bit weights + activations) | 2.9 GB | 2.7 GB / 3.2 GB (second pass) |
| x2 latent upscaler | | 1.5 GB |
| VAE decode | 4.5 GB | 6.2 GB |

So both fit an 8 GB card with room for the desktop. The decode was 6.5 / 9.2 GB until its temporaries were recycled by size and freed as soon as they are consumed (three
full-size activations live instead of five; same results bit for bit, and the browser decode got faster: two-pass 103 -> 92 s). Longer or larger clips grow the decode linearly
with the pixels; the denoise stages grow with the token count. Not yet verified on an 8 GB card or measured inside the browser.

## Licence

The weights are under the LTXV Open Weights License 0.X (Lightricks, 2025-04-15; it covers every LTXV release since 0.9.6, so 0.9.8): free use under its use-based restrictions (Attachment A); redistribution of the model or of modified files only with a copy of the licence, the Attachment A restrictions passed on as enforceable terms, prominent notices on changed files and the original notices kept (its section 3). Entities with annual revenues of at least $10,000,000 must obtain a paid commercial licence from Lightricks (its section 2). The x2 latent upscaler (`ltxv-spatial-upscaler-0.9.8`) carries the same terms in a version dated 2025-05-05, hosted beside it as LICENSE-spatial-upscaler.txt. The T5-v1.1-XXL encoder is Apache 2.0.
