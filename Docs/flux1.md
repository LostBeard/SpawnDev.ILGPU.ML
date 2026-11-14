# FLUX.1 and Z-Image from ComfyUI's files (`SpawnDev.ILGPU.ML.Flux1`, `SpawnDev.ILGPU.ML.ZImage`)

The models behind ComfyUI's FLUX.1 dev and Z-Image Turbo templates, loaded from the files those templates download and checked call by call against ComfyUI itself
(SpawnGen's dumps: ComfyUI's reference build ran each model in float32 on the CPU and recorded its inputs and outputs). These are the blocks a graph runner needs: the text
encoders, the transformer call ComfyUI's sampler makes, and the autoencoder. The sampler loop is the caller's (SpawnGen runs ComfyUI's graphs on these blocks).

## FLUX.1 dev

| block | ComfyUI file | load with | measured against ComfyUI |
|---|---|---|---|
| transformer (12B MMDiT: 19 double + 38 single blocks) | `flux1-dev-fp8.safetensors` (all in one, 8-bit floats without scales, under `model.diffusion_model.`) | `Flux1Weights.OpenAsync` + `Flux1Transformer` | the denoised estimate of both model calls of a 2-step 256 px run: 4.71e-3 and 5.34e-3 relative RMS |
| T5-XXL encoder | the same file, `text_encoders.t5xxl.transformer.` | `Wan.UmT5Encoder.CreateAsync` + `UnigramTokenizer` (t5-v1_1-xxl `tokenizer.json`) | the prompt's token rows 6.13e-3, all 256 rows 1.28e-2; the empty prompt 1.26e-3 / 1.99e-2 |
| CLIP-L pooled vector | the same file, `text_encoders.clip_l.transformer.` | `StableDiffusion.SdTextEncoder.LoadAsync` + `PooledAsync` | ComfyUI's `y`: 3.33e-4 (the final-norm end-token row) |
| autoencoder | `ae.safetensors` | `SdVae.LoadAsync` | see the ae section below |

The 12B transformer does not fit most cards, so its blocks are split by `StreamedModelWeights`: the blocks that fit in `residentBytes` stay on the device and the rest are read from
the file on every call (RTX 4070, 7 GB budget: 22 of 57 blocks resident, 4.96 GB streamed per call, 1.8 s per call warm). 8-bit matrices stay 8-bit on the device and are
decoded into one shared bf16 scratch just before their GEMM.

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
| transformer (6B single-stream DiT: 2 + 2 refiners, 30 layers) | `z_image_turbo_bf16.safetensors` | `ZImageTransformer.OpenAsync` | the denoised estimate of both model calls of a 2-step 256 px run: 5.62e-3 and 3.56e-3 |
| text encoder (Qwen3-4B, the state after layer 35 of 36) | `qwen_3_4b.safetensors` (the FLUX.2 klein copy) | `Flux2Weights.LoadAsync(..., textEncoderLayers: 35)` + `new Flux2TextEncoder(accelerator, w, new[] { 35 })` + `Qwen2Tokenizer.ChatPromptNoThinking` | 3.84e-3 (prompt, 15 rows) and 3.86e-3 (empty, 8 rows); the state after layer 34 misses by 0.19 |
| autoencoder | `ae.safetensors` (FLUX.1's) | `SdVae.LoadAsync` | as FLUX.1 |

```csharp
using var tr = await ZImageTransformer.OpenAsync(accelerator, source, repo, "z_image_turbo_bf16.safetensors", rows, cols, captionTokens, residentBytes);
await tr.ForwardAsync(x, captionRows, modelTime: 1 - sigma);   // x: model-space latent [16, H/8, W/8] (FLUX.1's ae space); caption [captionTokens, 2560], unpadded
z.Denoised(x, tr.Output, denoised, sigma, rows, cols);         // ComfyUI's model output: x + sigma * output (the model predicts minus the velocity)
```

Both conventions were measured, not assumed: the other three readings (time = sigma, or the other sign) miss by 1.46 to 2.85 and the gate asserts they lose. The caption is
the prompt in Qwen3's chat template without the thinking block (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`: 15 tokens for a 7-word prompt, 8 for an
empty one, as ComfyUI's), not padded. ComfyUI's scheduler for Z-Image Turbo is its own shift-3 schedule (simple 8 steps: 1.0, 0.95455, 0.9, 0.83333, 0.75, 0.64286, 0.5, 0.3, 0).

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
RTX 4070, CUDA: 26 s to load, the 160-token run in 5.8 s.

## The ae (FLUX.1 and Z-Image)

ComfyUI's LATENT for this autoencoder is the raw encoder mean, `[16, H/8, W/8]` channel-major; FLUX.1's scale 0.3611 and shift 0.1159 are applied on the way into the model, not in
LATENT. A picture that is not a multiple of 8 is cut to its centred multiple, as ComfyUI does.

## Gates

CUDA, `HeavyModel` (files: `F1_DIR` default `E:/spawndev-flux1`, `ZI_DIR` default `E:/spawndev-zimage`, `F2_DIR` for `qwen_3_4b.safetensors`):
`Flux1_Transformer_MatchesComfyUiCalls` (control: guidance 1 must miss), `Flux1_T5FromCheckpoint_MatchesComfyUi`, `Flux1_T5Tokenizer_EmptyAndEdgeSpaces`,
`Flux1_ClipLFromCheckpoint_PooledMatchesComfyUi`, `Flux1_ComfyAe_MatchesComfyUi`, `ZImage_Transformer_MatchesComfyUiCalls` (the three other conventions must miss),
`ZImage_TextEncoder_MatchesComfyUi` (control: the state after layer 34 must miss), `TextGenerate_ComfyQwen3_MatchesComfyUiTokens`.

## Licence

Code: MIT. Weights are the user's own files and are never hosted by this library: FLUX.1 [dev] is Black Forest Labs' non-commercial licence (FLUX.1 [schnell] is Apache-2.0),
Z-Image Turbo is Apache-2.0 (Tongyi), Qwen3-4B and T5 are Apache-2.0. The Z-Image port follows diffusers' `ZImageTransformer2DModel` (Apache-2.0).
