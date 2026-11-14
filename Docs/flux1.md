# FLUX.1, Z-Image and Qwen-Image from ComfyUI's files (`SpawnDev.ILGPU.ML.Flux1`, `.ZImage`, `.QwenImage`)

The models behind ComfyUI's FLUX.1 dev, Z-Image Turbo and Qwen-Image templates, loaded from the files those templates download and checked call by call against ComfyUI itself
(SpawnGen's dumps: ComfyUI's reference build ran each model in float32 on the CPU and recorded its inputs and outputs). These are the blocks a graph runner needs: the text
encoders, the transformer call ComfyUI's sampler makes, and the autoencoder. The sampler loop is the caller's (SpawnGen runs ComfyUI's graphs on these blocks).

## Precision: fp32-class activations on CUDA

ComfyUI's reference runs keep the activations in fp32. CUDA's tensor-core GEMMs round them to bf16, which costs about 5e-3 relative RMS per model call (FLUX.1 4.7e-3,
Z-Image 5.6e-3, Qwen-Image 4.3e-2 over its 60 blocks). `DiTKernels.PreciseActivations` keeps the tensor cores and the fp32-class result: every activation is staged as two bf16
halves (hi = bf16(a), lo = bf16(a - hi)) and each GEMM runs both passes into the same fp32 accumulators (`MmaGemm.KernelBf16x2Dense`; the bf16 weights are exact), and attention
runs the fp32 flash kernel. MEASURED (RTX 4070, against the dumps): FLUX.1 3.3e-5 / 5.5e-5, Z-Image 3.2e-5 / 2.4e-5, Qwen-Image 2.4e-4 / 2.9e-4; the warm call times at 256 px did
not change (the calls are weight-bound). It is ON by default in `Flux1Transformer`, `ZImageTransformer` and `QwenImageTransformer` (their `PreciseActivations` property turns it off)
and opt-in on `Flux2TextEncoder` (FLUX.2's own reference is a bf16 run). The other backends compute in fp32 already: SpawnGen measured the same calls on an RX 590 (OpenCL) at
4.7e-6 to 2.2e-5.

## FLUX.1 dev

| block | ComfyUI file | load with | measured against ComfyUI |
|---|---|---|---|
| transformer (12B MMDiT: 19 double + 38 single blocks) | `flux1-dev-fp8.safetensors` (all in one, 8-bit floats without scales, under `model.diffusion_model.`); the bare `flux1-dev.safetensors` (bf16) and fp8_scaled files load the same way | `Flux1Weights.OpenAsync` + `Flux1Transformer` | the denoised estimate of both model calls of a 2-step 256 px run: 3.32e-5 and 5.49e-5 |
| T5-XXL encoder | the same file, `text_encoders.t5xxl.transformer.` (or `t5xxl_fp16` / `t5xxl_fp8_e4m3fn_scaled.safetensors`) | `Wan.UmT5Encoder.CreateAsync` + `UnigramTokenizer` (t5-v1_1-xxl `tokenizer.json`) | with `PreciseActivations`: the prompt's token rows 3.17e-4, all 256 rows 1.05e-3; the empty prompt 5.6e-6 / 1.5e-4 (bf16 activations: 6.1e-3 / 1.3e-2) |
| CLIP-L pooled vector | the same file, `text_encoders.clip_l.transformer.` (or `clip_l.safetensors`) | `StableDiffusion.SdTextEncoder.LoadAsync` + `PooledAsync` | ComfyUI's `y`: 1.34e-6 with `ExactActivations` (the final-norm end-token row; tensor cores 3.3e-4) |
| autoencoder | `ae.safetensors` (or the checkpoint's `vae.*`) | `SdVae.LoadAsync` | see the ae section below |

The 12B transformer does not fit most cards, so its blocks are split by `StreamedModelWeights`: the blocks that fit in `residentBytes` stay on the device and the rest are read from
the file on every call (RTX 4070, 7 GB budget: 22 of 57 blocks resident, 4.96 GB streamed per call, 1.8 s per call warm). 8-bit matrices stay 8-bit on the device and are
decoded into one shared bf16 scratch just before their GEMM; an fp8_scaled file's `scale_weight` is each matrix's decode scale (a scaled file with an unscaled 8-bit matrix is refused).
ComfyUI computes a scaled Linear as `x (f8 * scale_weight)^T + bias` in fp32 (SpawnGen hooked one in ComfyUI: reproduced to 5e-8; `scale_input` is unused); the 8-bit values decode
exactly and the scale moves to the input. The Krea fp8_scaled file (F16 biases) is 1.3e-3 from ComfyUI until SpawnDev.ILGPU 5.4.33, which rounds the in-register BFloat16 round trip
its exact-bias path needs.

The call ComfyUI's sampler makes, measured:

```csharp
using var w  = await Flux1Weights.OpenAsync(accelerator, source, repo, "flux1-dev-fp8.safetensors", residentBytes: 7_000_000_000);
using var tr = new Flux1Transformer(accelerator, w, textTokens: 256, imageTokens: rows * cols);   // rows, cols = H/16, W/16
tr.SetPositions(rows, cols);                                   // once per picture size
var k = new Flux1Kernels(accelerator);
k.Patchify(x, tokens, rows, cols);                             // x: the MODEL-space latent [16, H/8, W/8] = (LATENT - 0.1159) * 0.3611
await tr.ForwardAsync(tokens, t5Rows, pooledClip, sigma, guidance: 3.5f);   // timestep = sigma; dev reads the guidance (FluxGuidance), schnell has none
k.Denoised(x, tr.Output, denoised, sigma, rows, cols);         // ComfyUI's model output: x - sigma * velocity
```

T5 conditioning is 256 rows: the prompt's tokens, the end token, then pad tokens (id 0) attended like any other (no mask), exactly as ComfyUI pads. An empty prompt is the end
token alone (the tokenizer's whitespace split leaves no pieces).

## Z-Image Turbo

| block | ComfyUI file | load with | measured against ComfyUI |
|---|---|---|---|
| transformer (6B single-stream DiT: 2 + 2 refiners, 30 layers) | `z_image_turbo_bf16.safetensors` | `ZImageTransformer.OpenAsync` | the denoised estimate of both model calls of a 2-step 256 px run: 3.18e-5 and 2.43e-5 |
| text encoder (Qwen3-4B, the state after layer 35 of 36) | `qwen_3_4b.safetensors` (the FLUX.2 klein copy) | `Flux2Weights.LoadAsync(..., textEncoderLayers: 35)` + `new Flux2TextEncoder(accelerator, w, new[] { 35 }) { PreciseActivations = true }` + `Qwen2Tokenizer.ChatPromptNoThinking` | 7.01e-5 (prompt, 15 rows) and 7.07e-5 (empty, 8 rows) with precise activations (bf16: 3.84e-3); the state after layer 34 misses by 0.19 |
| autoencoder | `ae.safetensors` (FLUX.1's) | `SdVae.LoadAsync` | as FLUX.1 |

```csharp
using var tr = await ZImageTransformer.OpenAsync(accelerator, source, repo, "z_image_turbo_bf16.safetensors", rows, cols, maxCaptionTokens, residentBytes);
await tr.ForwardAsync(x, captionRows, modelTime: 1 - sigma, captionTokens: n);   // x: model-space latent [16, H/8, W/8] (FLUX.1's ae space); caption [n, 2560], unpadded;
                                                               // n may change per call (a prompt and its negative under cfg): one open serves both
z.Denoised(x, tr.Output, denoised, sigma, rows, cols);         // ComfyUI's model output: x + sigma * output (the model predicts minus the velocity)
```

Both conventions were measured, not assumed: the other three readings (time = sigma, or the other sign) miss by 1.46 to 2.85 and the gate asserts they lose. The caption is
the prompt in Qwen3's chat template without the thinking block (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`: 15 tokens for a 7-word prompt, 8 for an
empty one, as ComfyUI's), not padded. ComfyUI's scheduler for Z-Image Turbo is its own shift-3 schedule (simple 8 steps: 1.0, 0.95455, 0.9, 0.83333, 0.75, 0.64286, 0.5, 0.3, 0).

## Qwen-Image

| block | ComfyUI file | load with | measured against ComfyUI |
|---|---|---|---|
| transformer (20B MMDiT: 60 double-stream blocks, ported from diffusers' `QwenImageTransformer2DModel`) | `qwen_image_fp8_e4m3fn.safetensors` (8-bit floats without scales) | `QwenImageTransformer.OpenAsync` | the denoised estimate of the prompt's and the negative's calls of a cfg-4 256 px run: 2.42e-4 and 2.90e-4 |
| text encoder (Qwen2.5-VL-7B's language model: 28 layers + final norm) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (8-bit matrices with one scale each, kept 8-bit; the vision tower is not read) | `Flux2Weights.LoadAsync(..., 28, finalNorm: true)` + `new Flux2TextEncoder(accelerator, w, new[] { 28 }, finalNorm: true) { PreciseActivations = true }` | 4.52e-5 (prompt, 12 rows) and 6.95e-5 (empty, 5 rows) (folding the fp8 scales into a bf16 decode: 3.1e-3); without the final norm 1.15, after layer 27 1.95 |
| VAE (the Wan 2.1 VAE: 16 channels, 8x) | `qwen_image_vae.safetensors` (bf16) | `GpuWeights` + `WanComfyFormat.AdaptVaeAsync` + `WanVae` (its Wan 2.1 mode) | encode of a 256 x 192 picture 1.75e-4, decode 5.02e-5 |

```csharp
using var tr = await QwenImageTransformer.OpenAsync(accelerator, source, repo, "qwen_image_fp8_e4m3fn.safetensors", maxTextTokens, rows, cols, residentBytes);
var k = new Flux1Kernels(accelerator);
k.Patchify(x, tokens, rows, cols);                             // x: the model-space latent [16, H/8, W/8] = (LATENT - mean) / std (the 5-D LATENT's one frame; measured)
await tr.ForwardAsync(tokens, cond, timestep: sigma, textTokens: n);   // cond [n, 3584]: the encoder's rows after the template's first 34; n may change per call
k.Denoised(x, tr.Output, denoised, sigma, rows, cols);         // ComfyUI's model output: x - sigma * velocity (FLUX.1's convention)
```

The text is the reference's template (`<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects
and background:<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, ids identical to ComfyUI's); the conditioning is the last hidden state after the final
norm, rows 34 onward (12 rows for a 7-word prompt, 5 for an empty one), no mask. ComfyUI runs the prompt and the negative as separate calls when their lengths differ; one
`QwenImageTransformer` serves both (the gate runs 12 rows, 5 rows, then 12 again: identical to the first). ComfyUI's LATENT is the RAW latent [16, 1, H/8, W/8] (measured); `WanVae`
takes and returns the normalised one, (raw - mean) / std with the constants in `GetLatentNormalizationAsync`, and the transformer works in that normalised space: ComfyUI's
2-step cfg-4 run rebuilt on these blocks lands on its final LATENT at 5.5e-5 when the result is mapped raw = model * std + mean (0.61 when read as the model space itself). On a 12 GB card 20 of the 60 blocks stay resident and 13.6 GB streams per call (~36 s at 256 px: the
disk is the bound).

The text encoder's 8-bit matrices run exact: `Flux2Weights.MatrixExact` decodes the 8-bit values unscaled (every F8_E4M3 value is a bf16 value) and the encoder multiplies the
matrix's input by its scale (s (x W) = (s x) W), where folding the scale into a bf16 decode would round every weight to 8 bits as ComfyUI's fp32 "manual cast" does not.

## Generate Text (ComfyUI's TextGenerate node)

ComfyUI's Generate Text node runs a text-encoder file as a language model. `SafeTensorsGguf.FromHfQwen3Async` presents such a file (`qwen_3_4b.safetensors`: Qwen3-4B,
36 layers, bf16, the head tied to the embedding) as a GGUF model over the same bytes, so the GGUF batch decoder runs it with no conversion and no copy: the tensor
descriptors point into the safetensors data section, names and dimension order become GGUF's, and the tokenizer comes from the model's `tokenizer.json`.

```csharp
await using var fs = File.OpenRead("qwen_3_4b.safetensors");
var model = await SafeTensorsGguf.FromHfQwen3Async(fs, tokenizerJson, chatTemplate);   // tokenizer.json and chat_template.jinja of Qwen3 (ComfyUI's file has neither)
using var gen = await GgufGenerator.CreateAsync(accelerator, fs, maxSeqLen: 4096, model: model);
var tok = SentencePieceTokenizer.FromGGUF(model)!;
var ids = ChatTemplates.BuildChatMLPromptTokens(tok, new[] { ("user", prompt) }, emptyThinking: !thinking);   // ComfyUI's default template
var result = await gen.GenerateAsync(ids, new GenerationConfig { MaxNewTokens = maxLength, Strategy = "greedy" });
```

MEASURED against SpawnGen's black-box token ids of ComfyUI's run (greedy, CPU bf16): the prompt ids are identical, and the generated tokens are identical, 23 of 23
then the end of the turn with thinking off, 159 of 159 with thinking on (ComfyUI's max_length counts the thinking tokens; it never feeds back its last token).
RTX 4070, CUDA: 20 to 26 s to load, the 160-token run in 5.8 s.

**On an 8 GB card** the 8.05 GB of bf16 matrices do not fit: `GgufBatchDecoder.Float16MatrixLoad` (env `ML_GGUF_F16_MATRICES`) converts 16-bit matrices to Q8_0 (ggml's block: 32
weights + one fp16 scale) as they stream in, automatically when they would take more than 75% of the device's memory (`AutoQ8Fraction`), always with `Q8`, never with `Keep`;
the decoder reports it (`ConvertedToQ8`, a console line). Half the memory and the Q8 GEMV path; not the bf16 model: against the same refs, 13 of 23 greedy tokens follow
ComfyUI's with thinking off (then a different wording), 159 of 159 with thinking on.

## The ae (FLUX.1 and Z-Image)

ComfyUI's LATENT for this autoencoder is the raw encoder mean, `[16, H/8, W/8]` channel-major; FLUX.1's scale 0.3611 and shift 0.1159 are applied on the way into the model, not in
LATENT. A picture that is not a multiple of 8 is cut to its centred multiple, as ComfyUI does.

## Gates

CUDA, `HeavyModel` (files: `F1_DIR` default `E:/spawndev-flux1`, `ZI_DIR` default `E:/spawndev-zimage`, `QI_DIR` default `E:/spawndev-qwenimage`, `F2_DIR` for `qwen_3_4b.safetensors`
and the Qwen tokenizer): `Flux1_Transformer_MatchesComfyUiCalls` (control: guidance 1 must miss), `Flux1_T5FromCheckpoint_MatchesComfyUi`, `Flux1_T5Tokenizer_EmptyAndEdgeSpaces`,
`Flux1_ClipLFromCheckpoint_PooledMatchesComfyUi`, `Flux1_ComfyAe_MatchesComfyUi`, `ZImage_Transformer_MatchesComfyUiCalls` (the three other conventions must miss; an 8-row call
between the checked ones), `ZImage_TextEncoder_MatchesComfyUi` (control: the state after layer 34 must miss), `QwenImage_ComfyVae_RoundTrip`, `QwenImage_ComfyVae_MatchesComfyUi`
(control: the normalised reading must miss), `QwenImage_TextEncoder_MatchesComfyUi` (two other hidden states must miss), `QwenImage_Transformer_MatchesComfyUiCalls` (five other
conventions must miss; per-call text length exact), `TextGenerate_ComfyQwen3_MatchesComfyUiTokens`, `TextGenerate_ComfyQwen3_Q8_StaysCloseToComfyUi`.

## Licence

Code: MIT. Weights are the user's own files and are never hosted by this library: FLUX.1 [dev] is Black Forest Labs' non-commercial licence (FLUX.1 [schnell] is Apache-2.0),
Z-Image Turbo is Apache-2.0 (Tongyi), Qwen-Image and Qwen2.5-VL are Apache-2.0 (Qwen), Qwen3-4B and T5 are Apache-2.0. The Z-Image and Qwen-Image ports follow diffusers'
`ZImageTransformer2DModel` and `QwenImageTransformer2DModel` (Apache-2.0).
