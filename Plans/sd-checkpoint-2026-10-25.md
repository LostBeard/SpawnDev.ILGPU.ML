# SdCheckpoint: single-file Stable Diffusion checkpoints (for SpawnGen's Load Checkpoint)

Requested by Lore (SpawnGen, `_DevComms/2026-10-22-1410-lore-TO-tuvok-model-support-and-parity-needs.md`), API agreed in
`_DevComms/2026-10-22-1445-tuvok-TO-lore-unet-builder-view-and-api.md`. TJ picked it 2026-10-25.

## API (namespace `SpawnDev.ILGPU.ML.StableDiffusion`)
```csharp
using var ck = await SdCheckpoint.LoadAsync(accelerator, source, repo, "model.safetensors", options, (stage, f) => ..., ct);
ck.Family / ck.Prediction / ck.LatentScale          // SD1 | SD2 | SDXL, Epsilon | V, 0.18215 | 0.13025
var te = await ck.TextEncoder!.EncodeAsync(ids /*[77] CLIP ids*/, ids2: null, layer: null);   // te.Context [77, 768|1024|2048], te.Pooled (SDXL)
float[] eps = await ck.UNet!.ForwardAsync(latent /*[4,h,w] scaled*/, h, w, timestep, te.Context, te.Rows, xl /*SDXL only*/);
float[] img = await ck.Vae!.DecodeAsync(latent / ck.LatentScale, h, w);   // [3, 8h, 8w] in [-1, 1], not clamped
float[] mean = await ck.Vae!.EncodeAsync(image, H, W);                   // UNSCALED posterior mean
```
Device-view overloads (`ArrayView1D<float>` in / out) exist for every call; the float[] ones wrap them.
The sampler, noise, CFG and host latents stay in SpawnGen. Tokenizer: `BPETokenizer.EncodeCLIP` (gated against the reference ids).

## Design
- One pass over the stream (browser OPFS / HTTP): `SdTensorStreamer` converts every tensor (F32 / F16 / BF16) to fp16 on the device
  through one reused staging buffer; conv weights are permuted to the gather layout at load; q|k|v (self) and k|v (cross) fused after load.
- Activations channel-last fp32 [pixels, C]; convolutions = the gathered-row implicit GEMM (tensor cores on CUDA for the UNet, the exact
  fp32-activation tiled path for the VAE: SDXL's VAE overflows fp16); attention = portable flash kernels, generator extended to head widths
  40 / 80 / 160 (SD 1.x: 8 heads); the VAE's single 512-wide head = two chunked GEMMs.
- Every conv / linear launch is split into row chunks of at most `SdKernels.MaxFlopsPerDispatch` (never lock the PC); GroupNorm = chunked
  partial sums; the browser awaits `SynchronizeAsync` per block.
- Layout read from the tensors: levels, resnets, transformer depth, linear vs 1x1 projections, SDXL label embedding.
- Prediction type as ComfyUI: `v_pred` tensor, or for SD 2.x the std of `output_blocks.11.1.transformer_blocks.0.norm1.bias` > 0.09.

## Gates (tests `MLTestBase.SdTests.cs`, fixtures `D:\Projects\sd\ref`, reference `tools/sd-harness/reference/ref_sd_dump.py --f16-weights`)
- [x] SD 1.5: tokenizer ids, text encoder (last + penultimate), UNet taps + eps (32x40 and 64x64), VAE decode taps + image, encode mean: CUDA + OpenCL green, red-checked (9b9625f8)
- [x] SD 1.5 on WebGPU (browser, checkpoint over HTTP from serve-files.mjs D:/Projects/sd 8769): 20-step picture 7.4e-5, VAE stages 4.8e-6 after the odd-bias fix (1e374fa9)
- [ ] SDXL on WebGPU (browser): not run yet; 1024 VAE decode memory (tiled decode) for small cards
- [x] a 20-step DDIM picture vs diffusers with the same noise (host sampler): CUDA 7.2e-3, OpenCL 4.8e-5 (e878b7e6)
- [x] SDXL base (two towers, pooled, size ids, VAE fp32 activations): CUDA green at 40x48 and 1024x1024 (UNet 436 ms/step)
- [x] SD 2.x: 2.1 768-v (V inferred), 2.1 base (eps), sgm-layout sd_turbo (Lore's report); UpcastAttention for SD 2.x on CUDA (e878b7e6). Stability deleted its SD 2 repos: files from sd2-community
- [x] LoRA merge (ApplyLorasAsync, kohya/PEFT/LDM keys, restore): LCM-LoRA SD 1.5 gated; [x] a LoRA with text-encoder keys (retrocgi), SDXL LoRAs (LCM-LoRA SDXL, pixel-art-xl), the old diffusers attn-procs format (c3ef2f04)
- [x] Split component loaders (SdUNet/SdVae/SdTextEncoder.LoadAsync, LoadSdxlAsync; LDM/diffusers/HF/OpenCLIP names), gated (8cd376f9); [x] TAESD / TAESDXL (b057bb3b); [x] standalone LoRA + InferPredictionAsync (b057bb3b)
- [x] ControlNet SD 1.x (SdControlNet, control_model.* single files; canny gated 8.3e-4 / 4.5e-4, b057bb3b); [x] SDXL ControlNets, diffusers-format ControlNets (a58ece28); [ ] Control-LoRA, controlnet-union-sdxl
- [ ] diffusers-format single files (second name map), SDXL refiner
- [x] Model merge (Lore #1, 10-30): `SdUNet / SdTextEncoder.MergeAsync(other, factors)`, `SdMerge.Simple / Add / Subtract / Blocks` (ComfyUI's ModelMergeSimple / Add / Subtract / Blocks / SD1 / SDXL, CLIPMerge*); edits in call order with LoRAs (`SdPatchState`), `ResetAsync`; gated: SD 2.1 base + 768-v Blocks 0.2/0.5/0.8 + CLIP 0.3, 1.22e9 values 0 beyond 1 fp16 ulp, reset bit-identical; merge / LoRA order 7 checks (CUDA)
- [x] Save (Lore #1, 10-30): `SdCheckpoint.SaveAsync` (CheckpointSave), `SdUNet / SdTextEncoder / SdVae.SaveAsync` (ModelSave / CLIPSave / VAESave), F16 / F32 / BF16, `v_pred` marker, streams through CopyToStreamAsync; gated: SD 1.5 + SD 2.1 round trip bit-exact vs the source files, reload bit-identical (CUDA)
- [ ] merge / save on OpenCL; a browser save into an OPFS / download stream (CopyToStreamAsync JS path)
- [x] Lore #2 inpainting UNet (10-30): `SdUNet.ConcatChannels`, `ForwardAsync(latent, concat, ...)`; gated vs diffusers 7.5e-5 CUDA / 9.3e-7 OpenCL (Sd15Inpaint_UNet_MatchesReference, fixture ref_sd_inpaint.py)
- [ ] Lore #3 UNet patches: [x] FreeU / FreeU_V2 + PAG (SdUNet.FreeU, PerturbMiddleSelfAttention; Sd15_UNetPatches_FreeUAndPag_MatchComfyUI CUDA 1.3e-3 / OpenCL 3e-6, red-checked); [x] deep shrink + any latent size (SdUNet.DeepShrink; Sd15_DeepShrinkAndOddLatent_MatchComfyUI 6.8e-5 / 9.2e-7, red-checked); [ ] SAG, HyperTile, ToMe
- [ ] Lore #4 SDXL refiner + CLIPTextEncodeSDXLRefiner, #5 unCLIP / CLIP vision / IP-Adapter / GLIGEN
- [ ] memory: SDXL VAE decode at 1024 needs ~4 GB of activations: tiled decode for small cards / browser
- [x] the browser 404 probes (mirrors.json index, 1d5b4b34) + OPFS cache never hit for HF-served files (261047b8)
