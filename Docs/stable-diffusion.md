# Stable Diffusion checkpoints (SD 1.x, SD 2.x, SDXL)

Namespace `SpawnDev.ILGPU.ML.StableDiffusion`. Loads the files ComfyUI and CivitAI use (single-file `.safetensors` checkpoints, and the standalone UNet / VAE /
text-encoder files of ComfyUI's split loaders) straight onto the GPU and runs the three models: the UNet, the CLIP text encoder(s) and the VAE. The sampler, the
noise, classifier-free guidance and the latents stay with the caller (SpawnGen runs its ComfyUI-compatible samplers on top of it).

Supported: SD 1.x, SD 2.x (512 base and 768-v; Stability's sgm layout such as `sd_turbo.safetensors`), SDXL base. Every tensor is converted to fp16 on the device as
the file streams in (fp32, fp16 and bf16 files), the file is read once, front to back, so browser OPFS and HTTP range streams work.

## Load a checkpoint

```csharp
using SpawnDev.ILGPU.ML.StableDiffusion;

using var ck = await SdCheckpoint.LoadAsync(accelerator, source /* IModelSource */, repo, "v1-5-pruned-emaonly.safetensors",
    new SdCheckpointOptions { /* LoadUNet / LoadTextEncoder / LoadVae, Prediction override */ },
    onProgress: (stage, fraction) => Console.WriteLine($"{stage} {fraction:P0}"));
// or SdCheckpoint.LoadAsync(accelerator, stream, name)

ck.Family;       // SD1 | SD2 | SDXL, read from the tensors
ck.Prediction;   // Epsilon | V (ComfyUI's rule: a v_pred tensor, or SD 2.x's late-norm-bias test)
ck.LatentScale;  // 0.18215 (SD 1.x / 2.x) | 0.13025 (SDXL)
```

## Text

```csharp
int[] ids = tokenizer.EncodeCLIP(prompt);                    // BPETokenizer: [77] start, tokens, end, padding with 49407
int[] ids2 = tokenizer.EncodeCLIP(prompt, padToken: 0);      // SDXL's second tower (OpenCLIP) pads with 0
SdTextEmbedding te = await ck.TextEncoder!.EncodeAsync(ids, ids2, layer: null);
// te.Context [te.Rows, te.Width]: 77 x 768 (SD 1.x) | 1024 (SD 2.x) | 2048 (SDXL);  te.Pooled [1280] for SDXL
```
Defaults follow ComfyUI: SD 1.x takes the last layer, SD 2.x the penultimate, both through the final layer norm; SDXL takes the penultimate layer of each tower
without the norm. `layer: -2` is ComfyUI's "CLIP Set Last Layer -2" (clip skip 2). Long prompts: encode each 77-token chunk and concatenate the rows. A device
overload writes the context and the pooled embedding into GPU buffers.

## UNet (one prediction per call)

```csharp
float[] eps = await ck.UNet!.ForwardAsync(latent /* [4, h, w], the scaled latent */, h, w, timestep /* 0..999 */, te.Context, te.Rows,
    xl: ck.Family == SdFamily.SDXL ? new SdxlConditioning(te.Pooled!, H, W, 0, 0, H, W) : null);
```
Batch 1; call it twice for classifier-free guidance (prompt and negative). `h` and `w` must be multiples of 8. The output is epsilon or v (`ck.Prediction`). A device
overload takes and returns GPU views. `UNet.LabelWidth` is SDXL's label width (2816). SD 2.x runs its attention projections with fp32 activations
(`UpcastAttention`, diffusers' `upcast_attention`): with fp16 tensor cores SD 2.1 768-v drifts 5e-2 from the reference on CUDA.

## VAE

```csharp
float[] rgb = await ck.Vae!.DecodeAsync(latent.Select(v => v / ck.LatentScale).ToArray(), h, w);  // [3, 8h, 8w] in [-1, 1], not clamped
float[] mean = await ck.Vae!.EncodeAsync(image /* [3, H, W] in [-1, 1] */, H, W);                  // posterior mean [4, H/8, W/8], unscaled
```
Both sides take the UNSCALED latent (as ComfyUI's VAE nodes). Every VAE convolution keeps fp32 activations (SDXL's VAE overflows fp16).

## LoRA

```csharp
var report = await ck.ApplyLorasAsync(new[] { SdLoraSpec.From(source, repo, "lcm-lora-sdv1-5.safetensors", strength: 0.8f, clipStrength: 1f) });
await ck.ApplyLorasAsync(Array.Empty<SdLoraSpec>());   // remove (the originals of the patched tensors are kept on the device: bit-identical restore)
```
kohya (`lora_unet_*`, `lora_te_*`, `lora_te1_* / lora_te2_*`), PEFT / diffusers (`unet.* / text_encoder.*` with `lora_A / lora_B`) and LDM-dotted keys; linear
and convolution (LoCon) modules; merged on the device into the fp16 weights (into the fused q|k|v rows where the projection was fused). The report lists the keys
that matched nothing in the checkpoint.

## Merging models (ComfyUI's ModelMerge* / CLIPMerge*)

```csharp
using var other = await SdCheckpoint.LoadAsync(accelerator, source, repo2, file2, new SdCheckpointOptions { LoadVae = false });
await ck.UNet!.MergeAsync(other.UNet!, SdMerge.Simple(0.6f));          // ModelMergeSimple: W = W1 * 0.6 + W2 * 0.4
await ck.TextEncoder!.MergeAsync(other.TextEncoder!, SdMerge.Simple(0.6f));   // CLIPMergeSimple
await ck.UNet.MergeAsync(other.UNet!, SdMerge.Blocks(new KeyValuePair<string, float>[] { new("input", 0.2f), new("middle", 0.5f), new("out", 0.8f) }));  // ModelMergeBlocks
// ModelMergeSD1 / SDXL: Blocks with ComfyUI's argument names ("time_embed.", "label_emb.", "input_blocks.0." ... "middle_block.1." ... "out.") - the longest prefix wins,
// the first entry's ratio when none matches. SdMerge.Add() = ModelAdd (W1 + W2), SdMerge.Subtract(m) = ModelSubtract ((W1 - W2) * m).
other.Dispose();                                                       // the second model is only read
await ck.UNet.ResetAsync();                                            // back to the file's weights (merges and LoRAs)
```
A factor function gets each tensor's internal name (UNet: the LDM name without `model.diffusion_model.`, the q|k|v / k|v projections fused under the
same block prefix; text: `l. / h. / g.` + `tok, pos, layers.N.{ln1, qkv, out, ln2, fc1, fc2}, final_ln, proj`) and returns the factors or null (leave it).
Edits apply in call order, like the graph: LoRAs applied after a merge stay replaceable (`ApplyLorasAsync` takes them off back to the merged weights); a
merge on top of LoRAs bakes them in (LoraLoader -> ModelMergeSimple computes exactly that). The file's weights of every edited tensor are kept on the device
for `ResetAsync` (as much memory again as the merged tensors); `MergeAsync(..., restorable: false)` skips that copy (an SDXL UNet is 5 GB), and `ResetAsync` then throws.

## Saving (ComfyUI's CheckpointSave / ModelSave / CLIPSave / VAESave)

```csharp
await using var file = File.Create("merged.safetensors");          // any writable Stream; in the browser an OPFS / download write stream
await ck.SaveAsync(file, new SdSaveOptions { DType = SdSaveDType.F16, Metadata = new Dictionary<string, string> { ["title"] = "my merge" } });
await ck.UNet!.SaveAsync(unetFile);      // model.diffusion_model.*        (UNETLoader reads it back)
await ck.TextEncoder!.SaveAsync(clipFile); // cond_stage_model.* / conditioner.embedders.*
await ck.Vae!.SaveAsync(vaeFile);         // encoder.* / decoder.* (no prefix, VAELoader)
```
The current weights (merges and LoRAs included) in the LDM single-file layout, fp16 (default, no rounding: the weights are held as fp16), fp32 or bf16, and
ComfyUI's empty `v_pred` tensor for a v-prediction checkpoint (`SdSaveOptions.VPrediction`, default the checkpoint's `Prediction`). Written front to back
(no seeking); each tensor goes back to the file layout on the device and out through `CopyToStreamAsync`, so in the browser the bytes go GPU -> JS -> stream
without the .NET heap. Not written, as in ComfyUI's saves: tensors the loaders do not read (EMA copies, `alphas_cumprod`, `position_ids`, `logit_scale`).

## Split loaders (ComfyUI's UNETLoader / VAELoader / CLIPLoader / DualCLIPLoader)

```csharp
using var unet = await SdUNet.LoadAsync(accelerator, source, repo, "unet.safetensors");        // LDM names (+/- model.diffusion_model.) or diffusers names
using var vae = await SdVae.LoadAsync(accelerator, source, repo, "sdxl_vae.safetensors");      // LDM (+/- first_stage_model.) or diffusers names
using var clip = await SdTextEncoder.LoadAsync(accelerator, source, repo, "clip_l.safetensors"); // one tower: CLIP-L (SD 1.x) or OpenCLIP-H (SD 2.x), HF or OpenCLIP names
using var dual = await SdTextEncoder.LoadSdxlAsync(accelerator, source, repo, "clip_l.safetensors", "clip_g.safetensors");  // SDXL
```
A component loaded alone owns its weights (`Dispose` frees them). `unet.InferPredictionAsync()` gives its prediction type (ComfyUI's rules) and
`unet.ApplyLorasAsync(...)` / `clip.ApplyLorasAsync(...)` merge LoRAs into standalone components (the checkpoint's `ApplyLorasAsync` delegates to both).

## TAESD (tiny previews)

```csharp
using var taesd = await SdTaesd.LoadAsync(accelerator, source, repo, "taesd_decoder.safetensors", "taesd_encoder.safetensors", latentScale: 0.18215f);  // taesdxl: 0.13025
float[] preview = await taesd.DecodeAsync(latent.Select(v => v / 0.18215f).ToArray(), h, w);   // same UNSCALED latents and [-1, 1] pictures as SdVae
```
madebyollin's tiny autoencoder (about 1 M parameters): ComfyUI's live-preview decoder and its VAELoader "taesd" / "taesdxl" entries. Reads the original
`taesd_decoder / taesd_encoder` files, ComfyUI's prefixed form and the diffusers AutoencoderTiny file. The published `taesdxl_decoder.safetensors` is a different
(newer) decoder than the diffusers TAESDXL file, so their pictures differ; both load.

## ControlNet (SD 1.x)

```csharp
using var cn = await SdControlNet.LoadAsync(accelerator, source, repo, "control_v11p_sd15_canny_fp16.safetensors");
var res = await cn.RunAsync(latent, h, w, timestep, te.Context, te.Rows, hint /* [3, 8h, 8w] in [0, 1] */);
res.Strength = 0.8f;
float[] eps = await ck.UNet!.ForwardAsync(latent, h, w, timestep, te.Context, te.Rows, null, new[] { res });   // several ControlNets add up
```
The single-file ControlNet v1.0 / v1.1 layout (`control_model.*`) ComfyUI uses: the UNet encoder copy, the hint encoder and the zero convolutions, 13 residuals
for the skip connections and the middle block, kept on the device.

## How it is checked

`SpawnDev.ILGPU.ML.Demo.Shared/UnitTests/MLTestBase.SdTests.cs`, fixtures from `tools/sd-harness/reference` (diffusers `from_single_file` on the same file, float32
math on fp16-rounded weights), every gate red-checked. RTX 4070:

| | CUDA | OpenCL | WebGPU (browser) |
|---|---|---|---|
| SD 1.5 UNet 512 (eps relRMS) | 5.4e-5, 85 ms/step | 9e-7, 217 ms/step | |
| SD 1.5 20-step DDIM picture, CFG 7.5 | 7.2e-3 | 4.8e-5 | 7.4e-5 (checkpoint over HTTP) |
| SDXL UNet 1024 | 6.2e-5, 436 ms/step | | |
| SD 2.1 768-v UNet 768 | 4.0e-4 | 2.6e-6 | |
| VAE decode 512 / 1024 | 1.1e-5 / 2.2e-4 | 4.8e-6 | 4.8e-6 |
| LCM-LoRA SD 1.5 x0.8 | 1.6e-3 | 1.2e-4 | |
| ControlNet canny SD 1.5 (13 residuals / eps x0.8) | 8.3e-4 / 4.5e-4 | | |
| TAESD / TAESDXL decode (encode) | 1.2e-3 (9e-4) / 1.5e-3 (3.6e-3), ~30 ms at 512 | | |

SpawnGen (Lore) compared it with ComfyUI's own renders on an RX 590: SD 1.5 0.19 % rms, SDXL 1024 0.0001, SD-Turbo 0.0000-0.0002, LCM-LoRA 0.27 %.

Not yet: SDXL ControlNets, diffusers-format ControlNets, Control-LoRA, the SDXL refiner, a tiled VAE decode for small cards (SDXL 1024 in the browser fits an
8 GB RX 590, SpawnGen measured).
