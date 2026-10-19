# FLUX.2 klein 4B text to image (`SpawnDev.ILGPU.ML.Flux2`)

FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0) re-implemented on ILGPU: the same C# kernels run it on CUDA, OpenCL and WebGPU, so it draws in a browser tab with no server.
It replaces SD Turbo (2023) as the picture stage of the text-to-3D demo: sharper, far better anatomy (hands, faces, armour), and it follows long prompts.

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

The three big models are never on the GPU together: each is streamed in, run and disposed (`Flux2Weights`), so the peak is the largest stage, about 2.6 GB of weights plus activations.

## 4-bit weights

`tools/unirig-harness/reference/quantize_flux2.py` packs every weight matrix of at least a million elements into blocks of 32 along K: an fp16 scale, an fp16 offset and 32 nibbles
(5 bits per weight), stored as `int32` words. `Flux2Weights.Matrix` dequantises one matrix at a time into a bf16 scratch right before its GEMM, so the ordinary tensor-core / tiled
GEMM runs unchanged; the dequantiser is bit-identical to a CPU decode (`Flux2_Q4Dequant_MatchesCpu`). Per-matrix relative error is about 8%. 4-bit moves the sampling trajectory, so the same seed
gives a different (equally clean) picture than the bf16 weights; the bf16 path stays in the library and is what the reference gates use.

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
