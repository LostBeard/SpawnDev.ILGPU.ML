# FLUX 3 Image Klein 5.6B text to image and reference editing (`SpawnDev.ILGPU.ML.Flux3`)

FLUX 3 Image Klein (Black Forest Labs, released 2026-10-15) re-implemented on ILGPU: text to image, and edits or compositions of up to 10 reference pictures, on CUDA, OpenCL and WebGPU, so it draws in a browser tab with no server.
The code is ours (MIT); the weights are Black Forest Labs' and stay under the **FLUX Kommunity License** (non-commercial use, or outputs for a Qualifying User; the licence travels with every copy, see below).

```csharp
var pipe = await Flux3Pipeline.CreateAsync(accelerator, modelSource, "LostBeard/flux-3-image-klein-q4-ILGPU");
var image = await pipe.RunAsync(new Flux3Request { Caption = fullSceneCaption, Seed = 1, Width = 1024, Height = 1024 });   // image.Rgba
// an edit or a composition: up to 10 pictures, the caption says what to do with them
var edit = await pipe.RunAsync(new Flux3Request { Caption = "...", Width = 768, Height = 512, References = new[] { new Flux2ReferenceImage(w, h, rgba) } });
```

## What runs

| stage | model | weights (4-bit) | notes |
|---|---|---|---|
| tokenizer | Qwen2 byte-level BPE (`Qwen2Tokenizer`) | 7 MB json | the Qwen3-VL tokenizer, token-identical to the reference |
| text encoder | text tower of Qwen3-VL-4B-Instruct, first 32 of 36 layers (`Flux3TextEncoder`) | 2.26 GB | hidden states after layers 4, 8, ... 32 stacked to 20480 wide; the caption is padded to a multiple of 80 tokens, padding keys masked, padding rows at position 1; rotary theta 5e6 |
| transformer | 5.6B flow-map DiT (`Flux3Transformer`) | 3.47 GB | three token modes (`image`, `image_cond`, `text`): 5 early blocks per mode (attention inside the mode only), then 28 joint blocks; per-mode modulation; the flow-map `ref_time_in` embedding of the next timestep; 16 Euler steps (`Flux3Pipeline.Schedule`, alpha 16 as in the reference code) |
| autoencoder | 16x autoencoder as two ONNX graphs | 199 MB decoder + 138 MB encoder (fp32) | the FLUX.2 layout (32 channels at 1/8, 2 x 2 patches, batch-norm statistics), run on the library's ONNX session; large pictures decode in 512 px tiles |

The text encoder, the transformer and the autoencoder are never on the GPU together: each is streamed in, run and freed (`KeepTransformerResident = true` keeps the 3.5 GB transformer for the next picture).
Position ids are `(0, row / 2, column / 2, 0)` for the picture (the reference halves rows and columns), `(0, 0, 0, index)` for text, and `l = -(1 + i)` for reference picture `i`.

## Reference pictures

`Flux3Request.References` (up to `Flux3Pipeline.MaxReferences` = 10): each picture is prepared like the reference code does (above 1024 x 1024 pixels it is shrunk with Lanczos, then centre-cropped to multiples of 16, `Flux2ReferencePrep`),
encoded by the autoencoder's encoder (`vae_encoder.onnx`, the mean of the latent distribution), patchified and batch-normalised, and fed to the transformer as `image_cond` tokens. The picture's own size is whatever the request says.
Reference tokens cost transformer work in proportion to their area: the demo page shrinks references to 768 px on the longer side.
The encoder path is gated against the reference autoencoder (`Flux3_References_MatchTheReference`: token count and ids exact, values at the bf16 rounding of the reference).

## Weights

`tools/unirig-harness/reference/flux3/` has the scripts: `quantize_flux3.py` (transformer, 4 or 8 bits, fused projections), `quantize_flux3_text.py` (text tower), `export_flux3_vae.py` (decoder, encoder, batch-norm statistics), `golden_flux3.py` and `golden_flux3_refs.py` (reference outputs).
The same block format as FLUX.2 (blocks of 32 along K, fp16 scale and offset). Hosted export: [`LostBeard/flux-3-image-klein-q4-ILGPU`](https://huggingface.co/LostBeard/flux-3-image-klein-q4-ILGPU) with 4-bit and 8-bit files, the licence text, the attribution notice and a statement that the files are a modified copy, not an official release.
The source repository is gated; a hosted copy is what lets a browser fetch the weights without a Hugging Face login.

## Numbers

RTX 4070 (CUDA), 512 px, 16 steps, the reference's caption and start noise, against the reference's own bf16 picture:

| weights | text encoder (vs reference) | picture | time |
|---|---|---|---|
| 8-bit | relRMS 7e-3 | 31.9 dB, mean 2.8 grey levels: the same picture | 32.8 s |
| 4-bit | | 23.2 dB: the same scene and subject, small differences in detail | 26.1 s |

The transformer is also gated on a small random-weight copy against the reference module (OpenCL 1e-5, CUDA 4e-3: the tensor-core GEMM rounds its inputs to bf16).

## Prompts

FLUX 3 Image Klein is trained on full scene captions (medium and style, scene, lighting and camera, every subject with colours, materials and position): a short prompt gives a generic picture. Black Forest Labs' own pipeline always runs a prompt upsampler first (their 9B `flux-3-image-klein-pup`, or any LLM with their system prompt).
`Flux3Request.Caption` is the caption the transformer sees, unchanged.
