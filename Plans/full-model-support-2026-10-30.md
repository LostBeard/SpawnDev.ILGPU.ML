# Full model support - audit and plan (2026-10-30)

TJ, 2026-10-30: "We will need to fully support models (image, video, etc...), not just features and specs that work well on this PC."

A model counts as supported only when ALL of these hold:
1. **Every mode** of the model's reference pipeline is ported and gated against the reference (not just the one that runs well here).
2. **Hosted**: weights on models.spawndev.com (with the licence files the licence requires), so the demo and consumers load them without setup.
3. **Browser**: runs on WebGPU in the demo / a WebGPU-lane test, re-verified on the current SpawnDev.ILGPU.
4. **Smaller devices**: fits and is verified beyond the RTX 4070 12 GB: an 8 GB GPU (Lore's RX 590), browser buffer limits; peak memory measured; streaming / tiling / offload where it does not fit.

Status key: ✅ verified (with evidence) · 🟡 partial · ❌ missing · ❓ to check (not verified either way).

## Generative models (first)

| Model | 1. Modes | 2. Hosted | 3. Browser | 4. Smaller devices |
|---|---|---|---|---|
| **LTX-Video 2B 0.9.8 distilled** (`LtxPipeline`) | ✅ text-to-video ✅ (every stage gated, CUDA); image-to-video + video / keyframe conditioning ✅ 2026-10-30 (LTXConditionPipeline: VAE encoder, per-token timesteps; frame-0 picture gated CUDA / OpenCL / WebGPU; keyframes at later frames and clip conditioning gated CUDA / OpenCL); x2 latent upscaler + two-pass recipe ✅ 2026-10-30 (`LtxRequest.Upscale`; gated CUDA / OpenCL / WebGPU) | ✅ 2026-10-30 `LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU` (8 files verified public incl. the VAE encoder; LICENSE + README with the Attachment A terms and change notices) | ✅ 2026-10-30 WebGPU: stages gated vs reference, 704x480 x97 in 72 s (`Ltx_Stages_OverHttp_MatchReference`, `Ltx_Pipeline_OverHttp_Generates`); found + fixed an aliasing bug in the no-noise VAE decode | 🟡 measured 2026-10-30 on CUDA (RTX 4070): peak 4.5 GB (704x480 x 97) / 6.2 GB (two-pass -> 896x576) after the decode's buffer recycling (was 6.5 / 9.2): fits 8 GB by measurement; ❓ not yet run on an 8 GB card (Lore's RX 590) or measured in the browser |
| **Wan 2.2 TI2V-5B** (`Wan22Pipeline`) | ✅ text-to-video, image-to-video (first frame), Turbo 4-step | ✅ `LostBeard/Wan2.2-TI2V-5B-ILGPU` (q8 / q4 / Turbo DiT, VAE, q8 umT5, tokenizer) | ✅ re-verified 2026-10-30 on ILGPU 5.4.13: 640x352 x17, 30 steps = 176 s (`Wan_Pipeline_OverHttp_MatchesReference`) | 🟡 measured 2026-10-30 on CUDA: 832x480 x49 peak 6.9 GB (the DiT; 4-bit DiT smaller): fits 8 GB; 1280x704: tiled decode (diffusers' enable_tiling, gated) 2.5 GB, with the 4-bit DiT peak 4.8 GB: fits 8 GB; the 4-bit Turbo (hosted 2026-10-30) with the tiled decode 4.8 GB at 720p; ❓ no 8 GB card run |
| Wan 2.2 A14B (T2V / I2V MoE) | ❌ not ported | ❌ | ❌ | ❌ (27B total: needs expert streaming) |
| **Stable Diffusion 1.x / 2.x / SDXL** (`SdCheckpoint`) | 🟡 UNet / CLIP / VAE, LoRA, ControlNet (SD1 + SDXL), TAESD, merge + save, inpainting, FreeU / PAG / HyperTile / deep shrink / SAG ✅ (CUDA + OpenCL); SDXL refiner ❌, unCLIP / IP-Adapter ❌, ToMe ❌, Control-LoRA ❌ | n/a (user checkpoints from HF / CivitAI) | 🟡 SD 1.5 20-step picture over HTTP ✅ (WebGPU, 10-25); SDXL browser by Lore on RX 590 (his check, not our gate); merge / save / inpaint / patches not browser-gated | 🟡 Lore: SD 1.5 + SDXL 1024 on RX 590 8 GB in the browser (10-25, his report) |
| SD-Turbo (ONNX pipeline) | ❓ | ❓ | ✅ `/generate` (`SDTurbo_Generate_E2E` incl. WebGPU) | ❓ `SDTurbo_WebGPU_ImageGen_MultiGen` fails (memory grows gen1 -> gen2): open bug |
| FLUX.2 klein 4B | ❓ text-to-image + reference editing ✅ (Docs/flux2.md) - check the reference pipeline for other modes | ✅ `LostBeard/FLUX.2-klein-4B-q4-ILGPU` | ✅ `/generate` (❓ re-verify on 5.4.13) | ❓ |
| FLUX 3 Image Klein | ❓ (Docs/flux3.md) | ✅ `LostBeard/flux-3-image-klein-q4-ILGPU` | 🟡 Chrome 151 lost the GPU process on picture 3 with resident weights (fine on 153) - ❓ re-verify | ❓ |
| TRELLIS.2 (text / image to 3D) | ❓ (Docs/trellis2.md) | ❓ | ✅ live 10-17 (crown photo -> GLB 148 s) - ❓ re-verify on 5.4.13 | ❓ |
| Kimodo animation, UniRig / SkinTokens rigging | ❓ | 🟡 `LostBeard/SkinTokens-bf16` | ❓ | ❓ |

Then the same audit for the vision / audio / LLM models (Docs/DEMO_AND_MODEL_STATUS.md lists them; column 1 is mostly ✅ there, columns 3 and 4 need the same re-verification).

## Order (editor's choice)

1. ✅ Wan in the browser re-verified on 5.4.13 (2026-10-30).
2. **LTX to full support**: a WebGPU-lane pipeline test over HTTP (like Wan's); host the q8 DiT, q8 T5, VAE, tokenizer + licence files; image-to-video (VAE encoder, the reference's first-frame conditioning); the x2 latent upscaler second pass; peak memory per stage, an 8 GB plan.
3. Smaller-device verification for Wan and LTX (peak memory per stage in the browser; what an 8 GB card needs).
4. Re-verify FLUX.2 / FLUX 3 / TRELLIS.2 in the browser on 5.4.13; fill the ❓ cells from their reference pipelines.
5. SD: ToMe, SDXL refiner, unCLIP / IP-Adapter (Lore's #3 to #5), and browser gates for the new SD features.
6. Wan 2.2 A14B.
