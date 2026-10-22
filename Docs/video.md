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
var pipe = new Wan22Pipeline(accelerator, Wan22Paths.InFolder(@"F:\Projects\wan"));                 // files on disk
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

## Limits

- **Memory.** The model's own size, 1280x704 with 121 frames (5 s), runs on a 12 GB card: 2 steps took 224 s (text 23 s, DiT 85 s, about 21 s per forward at 27,280 tokens, VAE decode 116 s), so a
  50-step video is about 35 minutes of DiT. What makes it fit: the decoder keeps its causal caches (two input frames per convolution) as fp16 pairs, decodes the last two stages a frame at a time
  when a chunk is big (`WanVae.MaxSliceBytes`; identical output, tested), and assembles the picture on the host chunk by chunk. VAE decode of 1280x704 x17 frames went from 175 s (VRAM spill) to 17 s.
  In the browser the transformer's time is the limit (about 7 s per forward at 832x480 x33), not memory.
- **Seeds.** At small sizes the model itself gives good clips for some seeds and garbage for others (the reference does too: seed 9 at 640x352 is noise-like where seeds 7 and 8 draw a fox). Judge the
  pipeline only with the reference's noise (`WanRequest.InitialNoise`), as the tests do.
- Our noise is not torch's generator, so a seed here is not a seed there.

## Where it is used

`/video` in the demo (text or image to video, a frame player and WebM download), `Examples/11.VideoGen.Console` (PNG frames and an animated WebP) and `Examples/12.VideoGen.Wpf`.
Tests: `Wan_*` in `MLTestBase.WanTests.cs` (desktop) and `Wan_Pipeline_OverHttp_MatchesReference`, `Wan_Vae_OverHttp_DecodeTimes` in `MLTestBase.WanHttpTests.cs` (WebGPU lane; serve the folder with
`node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767` first); the demo page by `tools/audit-scenarios/video.mjs`.

# Video generation: LTX-Video 2B (0.9.8 distilled)

`LtxPipeline` (`SpawnDev.ILGPU.ML.Ltx`): text to video with Lightricks' LTX-Video 2B distilled model: a 2B-parameter DiT (28 blocks, width 2048, every latent position is a token, RMS norm + adaLN,
RoPE over the whole 2048-wide q and k, qk RMS norm, cross-attention to the T5), the T5-v1.1-XXL text encoder (the umT5 encoder class with the shared position bias of block 0), the 7-step distilled flow-matching
Euler sampler (guidance 1: ONE forward per step, the latent never leaves the device), and the timestep-conditioned 32x32x8 video VAE decoder (128 latent channels; the decode adds a little noise at timestep 0.05,
as the recipe does). A 704x480 clip of 97 frames is only 4,290 tokens, which is why this model suits a tab. Image to video, the VAE encoder and the x2 latent upscaler (the recipe's second pass) are not ported yet.

Files (`tools/ltx-harness/reference`): `export_dit_bf16.py` (the single-file checkpoint to the Wan layout, q/k/v fused; 3.6 GB; `quantize_streaming.py` makes the 8-bit one, 2.0 GB), `export_t5_bf16.py` (8.9 GB / 8-bit
5.0 GB), `export_vae_f16.py` (the decoder, 1.0 GB), the T5 `tokenizer.json`. `LtxPaths.InFolder` / `InRepo` name them.

## Parity with the reference (diffusers, same inputs; `MLTestBase.LtxTests.cs`)

| Stage | Result |
|-------|--------|
| tokenizer | identical ids |
| RoPE table | the reference's float32 table to 1.9e-3 (one float32 ulp of a 15,708 rad frequency); exactly as far from the double-precision truth as the reference itself |
| T5 (bf16 / 8-bit) | 1.5e-2 to 3.9e-2 / up to 6.8e-2 against the bf16 reference |
| DiT noise prediction | **4.4e-5 / 7.3e-5 against the float32 reference with the bf16 tensor cores off (exact math), 600 / 2400 tokens**; 1.5e-2 / 8.5e-3 with them on. The bf16 reference is itself 5.2e-2 / 4.0e-2 from exact math (outlier activations of 2500 in the middle blocks), so the bound is the float32 reference. 8-bit DiT: 4.4e-2 |
| sampler loop, teacher-forced | worst 7.5e-3 / 4.5e-3 per step against the float32 reference (7 steps) |
| VAE decode | 1.4e-4 (video), every stage tap 7e-5 to 3.3e-4 |

Red checks: q RoPE off (DiT 0.84), unpatchify offsets swapped (VAE 0.18), a wrong T5 bucket distance (0.14 / 0.28).

## Measured

RTX 4070 12 GB, CUDA, `Examples/11.VideoGen.Console --ltx --models F:\Projects\ltx`: 704x480 x 97 frames **51 s** (T5 streamed from disk 24 s, DiT load 13 s, 7 steps 9 s, VAE decode 5 s); 480x320 x 49 frames 43 s
(30 s with the 8-bit DiT and T5: `--dit q8 --text q8`; the clips look the same).

## Licence

The weights are under the LTXV Open Weights License 0.X (free for entities under $10M annual revenue, use restrictions in its Attachment A, redistribution only with the licence text, change notices and the use
restrictions passed on). The T5-v1.1-XXL encoder is Apache 2.0.
