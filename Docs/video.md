# Video generation: Wan 2.2 TI2V-5B

Text to video and image to video on ILGPU, desktop (CUDA, OpenCL) and in the browser (WebGPU). `Wan22Pipeline` (namespace `SpawnDev.ILGPU.ML.Wan`) runs the model Alibaba released as
Wan2.2-TI2V-5B (Apache-2.0): the umT5-XXL text encoder, a 5B-parameter diffusion transformer (DiT) with classifier-free guidance and the UniPC flow sampler, and the Wan2.2 video VAE
(4x16x16 compression). One model does both modes: with a first-frame picture the first latent frame is held fixed at every step and the DiT gets per-token timesteps.

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
Desktop CUDA, `Examples/11.VideoGen.Console --ci` (320x192, 9 frames, 8 steps): 38 s, 14 s of it the text encoder.

## Limits

- **Memory.** The VAE decoder keeps two frames of input per causal convolution (the reference's feature cache), so its memory grows with the picture: 640x352 is comfortable on 12 GB and in the
  browser; 1280x704 with 121 frames runs the DiT but runs out of memory in the decoder on a 12 GB card. Fp16 activations and spatial tiling are the planned fixes.
- **Seeds.** At small sizes the model itself gives good clips for some seeds and garbage for others (the reference does too: seed 9 at 640x352 is noise-like where seeds 7 and 8 draw a fox). Judge the
  pipeline only with the reference's noise (`WanRequest.InitialNoise`), as the tests do.
- Our noise is not torch's generator, so a seed here is not a seed there.

## Where it is used

`/video` in the demo (text or image to video, a frame player and WebM download), `Examples/11.VideoGen.Console` (PNG frames and an animated WebP) and `Examples/12.VideoGen.Wpf`.
Tests: `Wan_*` in `MLTestBase.WanTests.cs` (desktop) and `Wan_Pipeline_OverHttp_MatchesReference`, `Wan_Vae_OverHttp_DecodeTimes` in `MLTestBase.WanHttpTests.cs` (WebGPU lane; serve the folder with
`node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767` first); the demo page by `tools/audit-scenarios/video.mjs`.
