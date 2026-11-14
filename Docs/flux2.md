# FLUX.2 klein 4B text to image (`SpawnDev.ILGPU.ML.Flux2`)

FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0) re-implemented on ILGPU: the same C# kernels run it on CUDA, OpenCL and WebGPU, so it draws in a browser tab with no server.
It replaces SD Turbo (2023) as the default picture stage of the text-to-3D demo: sharper, far better anatomy (hands, faces, armour), and it follows long prompts. FLUX 3 Image Klein ([flux3.md](flux3.md), non-commercial weights) is the larger, slower option, and `ImageGenerator` runs either behind one call.

```csharp
var pipe = await Flux2Pipeline.CreateAsync(accelerator, modelSource, "LostBeard/FLUX.2-klein-4B-q4-ILGPU");
var image = await pipe.RunAsync(new Flux2Request { Prompt = "a knight in plate armor, full body", Seed = 7 });   // image.Rgba, 512 x 512
// several seeds of one prompt: the prompt is encoded once and the transformer stays loaded, so each extra picture is only its 4 steps + a VAE decode
var many = await pipe.RunManyAsync(new Flux2Request { Prompt = "..." }, new[] { 7, 8, 9, 10 }, async (i, img) => { /* show it */ });
```

## What runs

| stage | model | weights (4-bit) | notes |
|---|---|---|---|
| tokenizer | Qwen2 byte-level BPE (`Qwen2Tokenizer`) | 11 MB json | token-identical to the reference on ten prompts, accents, CJK, emoji, added tokens |
| text encoder | Qwen3-4B, first 27 of 36 layers (`Flux2TextEncoder`) | 1.95 GB | the pipeline only reads hidden states 9, 18, 27 (stacked to 7680 wide); prompt padded to 512 with the padding keys masked |
| transformer | 3.9B flow DiT, 5 double + 20 single blocks (`Flux2Transformer`) | 2.42 GB | shared adaLN, 4-axis interleaved RoPE, no biases; 4 Euler steps (`Flux2Scheduler`, dynamic shift) |
| VAE decoder | AutoencoderKLFlux2 as ONNX | 199 MB (fp32) | runs on the library's ONNX session, resident |

`KeepTransformerResident = true` keeps the 2.4 GB transformer on the device between runs and the last prompt's embeddings are always kept, so redrawing the same prompt costs only its steps (about 8 s in a browser instead of 75 s); `ReleaseResident()` frees it.

The three big models are never on the GPU together: each is streamed in, run and disposed (`Flux2Weights`), so the peak is the largest stage, about 2.6 GB of weights plus activations.

## 4-bit weights

`tools/unirig-harness/reference/quantize_flux2.py` packs every weight matrix of at least a million elements into blocks of 32 along K: an fp16 scale, an fp16 offset and 32 nibbles
(5 bits per weight), stored as `int32` words. `Flux2Weights.Matrix` dequantises one matrix at a time into a bf16 scratch right before its GEMM, so the ordinary tensor-core / tiled
GEMM runs unchanged; the dequantiser is bit-identical to a CPU decode (`Flux2_Q4Dequant_MatchesCpu`). Per-matrix relative error is about 8%. 4-bit moves the sampling trajectory, so the same seed
gives a different (equally clean) picture than the bf16 weights; the bf16 path stays in the library and is what the reference gates use.

## 8-bit weights, bigger pictures

- **8-bit** (`quantize_flux2.py ... 8`, `Flux2Pipeline.CreateAsync(..., bits: 8)`): the same blocks with a byte per weight, 9 bits in all: 3.5 GB encoder + 4.4 GB transformer, 4.4 GB on the GPU at a time. Per-matrix error 0.5% instead of 8%, and the whole pipeline lands
  where bf16 does (29.2 dB from the diffusers picture against 29.3). For desktop GPUs; the browser demo stays on 4-bit.
- **Sizes above 512**: `Flux2Request.Size` is any multiple of 16 (768 and 1024 are tested). The transformer simply sees more tokens (1024 px = 4096 image tokens + 512 text); the VAE decodes in overlapping 512 px tiles with a 128 px feathered overlap, so its memory never
  grows with the picture (`Flux2Pipeline.VaeTileLatent`, `VaeTileOverlap`). On the RTX 4070 (CUDA, 4-bit): 768 px 19.5 s and 1024 px 12.9 s warm. In Chrome 151 WebGPU on the same card the Image Generation page draws 1024 px in 90 s the first time (weights loading included) and 30 s for the next picture of the prompt, with no memory trouble. The tiled decode is 51 dB from decoding the whole 1024 px picture at once (test) and shows no seam at 100%.

## Numbers

Reference: diffusers `Flux2KleinPipeline` and transformers `Qwen3ForCausalLM`, bf16, same prompt and start noise (`tools/unirig-harness/reference/golden_klein.py`).

- text embeddings: 1.5e-2 relative RMS overall (real tokens 4e-3; the padding rows are chaotic in bf16 itself: the fp32 model differs from the bf16 reference by 2% there at layer 9)
- transformer step 0 prediction 1.3e-2, step 1 2.8e-2; per-block checkpoints 2e-3 to 2e-2
- VAE decoder 2.7e-3
- whole pipeline, bf16 weights: 29.3 dB PSNR / 4.1 grey levels from the reference picture
- RTX 4070, CUDA, 512 px, 4 steps: 42.6 s cold with bf16 weights (35 s of that loading 14 GB), denoise steps 0.4 s each; 4-bit stages: 19 s cold, 2.6 GB of weights at a time
- Chrome 151 WebGPU on the same card: first picture 74 s from a local server (denoise steps ~1.5 s each); in the demo, 427 s for the first run because it includes downloading 4.4 GB, then 7 s for each further picture of the same prompt

## Reproducing

```
python -I tools/unirig-harness/reference/golden_klein.py <klein4b_dir> <out> "<prompt>" 0 512 4     # reference run + per-block checkpoints
python -I tools/unirig-harness/reference/export_flux2.py <klein4b_dir> <weights_dir> all           # bf16, fused q/k/v and gate/up rows
python -I tools/unirig-harness/reference/export_flux2_vae.py <klein4b_dir> <weights_dir>/vae_decoder.onnx
python -I tools/unirig-harness/reference/quantize_flux2.py <weights_dir> transformer|text_encoder  # the 4-bit files
```

Tests: `Flux2_*` in `MLTestBase.Flux2Tests.cs` (desktop lanes; skipped without the checkpoint, `F2_DIR` default `E:/spawndev-flux2`), and `Flux2_OverHttp_DrawsAPicture` on the browser WebGPU lane against `node tools/unirig-harness/serve-files.mjs <hf_upload dir> 8766`.

## Licence

Weights: Apache-2.0 (Black Forest Labs). The 4-bit export is a modified copy, hosted at <https://huggingface.co/LostBeard/FLUX.2-klein-4B-q4-ILGPU> with the licence and a model card that states the change.
Code: MIT like the rest of the library. FLUX.2 klein 9B is non-commercial and is not used.

## Editing with reference images (2026-10-19)

FLUX.2 klein edits and composes pictures: up to 10 REFERENCE images go in next to the prompt (`Flux2Request.References`). Each is prepared like the reference pipeline (`Flux2ReferencePrep`: shrunk to 1024 x 1024 pixels of area with Lanczos when larger, then centre-cropped down to multiples of 16, pixels to [-1, 1]), encoded by the VAE ENCODER (`vae_encoder.onnx`, 138 MB, in the model repo next to the decoder; `Flux2Pipeline.EncodeReferenceAsync`: distribution mode, 2 x 2 patchify, the same batch-norm statistics as the decode path, packed row by row to 128-channel tokens), and appended to the picture's own tokens in the transformer's sequence with their own position (t = 10 + 10 i, row, column, 0). The model predicts only the picture's tokens (the first ones); the schedule's shift depends on the picture's token count alone. The picture has the first reference's size (or `Width` x `Height`), not necessarily square: a non-square picture is decoded by the tiled VAE (`DecodeAsync(vaeInput, rows, columns)`).

`AssetGenerator.EditPictureAsync(picture, "give the knight a tall red plume on the helmet")` is the one-call form; the console's `--edit-instruction` chains it into the 3D region edits (`--edit-region`, `--edit-shape`): FLUX.2 edits the 2D picture, TRELLIS.2 applies it to the 3D model.

Gate against diffusers (`golden_klein_edit.py`, `Flux2_ReferenceEncoder_MatchesTheReference`, `Flux2_ReferenceEdit_MatchesTheReference`, CUDA): preprocessed pixels equal the reference's, encoder latents relRMS 2.3e-2 (the reference encodes in bf16, the export is fp32), the whole edit **44.4 dB PSNR against the reference's edited picture** (mean difference 0.42 grey levels, 46 s with the bf16 weights). On the knight: "Add a tall red feather plume on top of the knight's helmet. Keep everything else exactly the same." returns the same knight with a red plume, about 33 s through the 4-bit weights in the console.

## ComfyUI's files (2026-11-13)

The files ComfyUI's FLUX.2 klein templates load (its split Load Diffusion Model / Load CLIP / Load VAE graph) load as they are, rewritten into this library's layout as they land:

| ComfyUI file | load with | measured against our export |
|---|---|---|
| `flux2-vae.safetensors` (Comfy-Org/flux2-klein: Black Forest Labs' names; Comfy-Org/flux2-dev: diffusers names) | `SdVae.LoadAsync` (the native LDM VAE: encoder + decoder) and `Flux2VaeNorm.FromSafeTensorsAsync` (its batch-norm statistics) | the two namings: the same fp32 weights, bit-identical; they are the fp32 values our bf16 export was rounded from |
| `flux-2-klein-4b.safetensors` (bf16) | `Flux2Weights.LoadAsync` | 149 tensors bit-identical (the final modulation's two halves exchanged: BFL's shift / scale order) |
| `flux-2-klein-4b-fp8.safetensors` (black-forest-labs/FLUX.2-klein-4b-fp8: 8-bit floats, one scale per matrix) | `Flux2Weights.LoadAsync` | 80 matrices within 2.7e-2 relative RMS (the format's rounding), held as 8 bits on the device (4.24 GB) and decoded just before each GEMM |
| `qwen_3_4b.safetensors` (Hugging Face Qwen3-4B names, 36 layers) | `Flux2Weights.LoadAsync` | the 27 layers the pipeline uses: 217 tensors bit-identical (q / k / v and gate / up fused at load); the other 9 layers are never read (6.23 GB on the device) |

**ComfyUI's LATENT for FLUX.2** is `[128, H/16, W/16]` channel-major: the VAE encoder's mean, patchified 2 x 2 (channel `c * 4 + dy * 2 + dx`) and batch-normalised with the file's
statistics, which is the transformer's tokens transposed. MEASURED against ComfyUI's own VAE Encode / Decode (SpawnGen's dumps, float32, CPU): encode 5.8e-4 and 9.3e-4 relative RMS
(a picture not a multiple of 16 is cut to its centred multiple, as ComfyUI does), decode 1.1e-4; and against the diffusers reference's own latents 2.1e-2 (the reference runs bf16).
The device kernels:

```csharp
var vae  = await SdVae.LoadAsync(accelerator, source, repo, "flux2-vae.safetensors");
var norm = await Flux2VaeNorm.FromSafeTensorsAsync(source, repo, "flux2-vae.safetensors");      // eps Flux2VaeNorm.BatchNormEps (1e-4) is not in the file
var k    = new Flux2Kernels(accelerator);
await vae.EncodeAsync(pixels, H, W, mean32);                    // [3, H, W] in [-1, 1] -> [32, H/8, W/8]
k.VaeToLatent(mean32, normMean, normStd, latent, H / 8, W / 8);  // -> LATENT [128, H/16, W/16]
k.LatentToTokens(latent, tokens, H / 16, W / 16);                // -> transformer tokens [cells, 128] (a transpose); TokensToLatent is the inverse
k.LatentToVae(latent, normMean, normStd, z32, H / 16, W / 16);   // LATENT -> the decoder's input
await vae.DecodeAsync(z32, H / 8, W / 8, picture);               // -> [3, H, W] in [-1, 1]
```

The 8-bit float transformer costs picture quality: the whole pipeline lands 24.4 dB from the reference picture, against 29.3 dB for bf16 and 29.2 dB for this library's own 8-bit
block export (0.5% per-matrix error against 2.7%). On a desktop GPU prefer the bf16 file or the hosted 8-bit export. Gates (CUDA, HeavyModel; files in F2_DIR\comfy):
`Flux2_ComfyVae_BothNamingsLoad`, `Flux2_ComfyVae_MatchesTheReference`, `Flux2_ComfyVae_MatchesComfyUiLatents`, `Flux2_ComfyTransformer_Bf16_AsOurExport`,
`Flux2_ComfyTransformer_Fp8_NearOurExport`, `Flux2_ComfyTransformer_Fp8_Pipeline`, `Flux2_ComfyTextEncoder_AsOurExport`. Red-checked: without the quant-convolution names,
with dy and dx exchanged, without the halves exchange, and with k | q | v fused, each fails its gate.

**ComfyUI's call contract for FLUX.2 klein** (MEASURED by SpawnGen against ComfyUI's own graphs, 2026-11-13), for a caller driving these blocks from a ComfyUI-style graph:
the transformer gets the LATENT as it is (it already holds the batch-normalised tokens, transposed: `Flux2Kernels.LatentToTokens`), all 512 text rows with no mask,
and timestep = sigma; BasicScheduler's shift for klein is a constant mu = 2.02; reference pictures (editing) come in as `ref_latents` (VAE Encode's LATENT) and sit at
t = 10 + 10 i, as `Flux2Pipeline` places them (at t = 0 the edit falls to 14 dB). End to end on these blocks, on an AMD RX 590 over OpenCL with the 8-bit float transformer:
the distilled text-to-image template 36.9 dB from ComfyUI's bf16 picture (text encoder 1.3e-6, VAE 1.1e-4 from ComfyUI), the distilled edit template 44.4 dB.
