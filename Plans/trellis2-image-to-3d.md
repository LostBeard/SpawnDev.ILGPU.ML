# TRELLIS.2 image-to-3D in SpawnDev.ILGPU.ML

Requested 2026-10-16 (TJ, via the SpawnDarkly session). **Goal: 2D image to PBR 3D model (GLB) on a 12 GB card
(RTX 4070 here), in a reasonable time, on ILGPU.ML.** Model generation runs on the host GPU; clients only get GLB.
Reference source (MIT): https://github.com/microsoft/TRELLIS.2 . Local clone for reading: session scratchpad `ref/trellis2`.

## Licenses (read from primary sources 2026-10-16)

| Piece | License | Consequence |
|---|---|---|
| TRELLIS.2 code + TRELLIS.2-4B weights | MIT (LICENSE file, HF card) | free to port and ship |
| microsoft/TRELLIS-image-large (sparse-structure decoder) | MIT (HF card) | free |
| facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 License: worldwide, royalty-free, redistribution allowed with a copy of the agreement, "Built with DINOv3" attribution, no commercial restriction, trade-control and military-use exclusions | usable; HF repo is GATED (user accepts Meta terms) so the user downloads it |
| briaai/RMBG-2.0 (reference background remover) | CC BY-NC 4.0, commercial needs a Bria agreement | DO NOT ship. Replace (see M0) |
| nvdiffrast / nvdiffrec (reference renderer only) | NVlabs own license | not used |
| Hunyuan3D 2.1 | Tencent community license, excludes EU/UK/South Korea, 1M MAU cap | not building on it |

## Pipeline (from trellis2_image_to_3d.py, pipeline.json, ckpts/*.json)

Types: `512` = SS res 32, shape+tex at 512. `1024_cascade` (reference default) = 512 shape LR, upsample, 1024 shape HR (max 49152 tokens), tex at 1024.
All samplers are Euler flow, 12 steps, CFG with guidance interval, rescale_t 3 to 5. CFG strength: SS 7.5, shape 7.5, tex 1.0.

1. Background removal -> crop to bbox, premultiply alpha (preprocess_image). Caller-supplied RGBA skips the remover.
2. DINOv3 ViT-L/16 features (cond 1024-d tokens, neg cond = zeros), image 512 (and 1024 for HR stage).
3. Sparse-structure flow DiT: 30 blocks, 1536 ch, 12 heads (head dim 128), cross-attn to cond, RoPE 3D on a 16^3 dense grid (4096 tokens), share_mod adaLN, qk RMS norm, mlp 5.3334. Latent 8x16x16x16.
4. SS decoder (conv3d, groupless layer norm, 16 -> 64^3 occupancy) -> argwhere -> sparse coords (max-pool to 32^3 for the 512 path).
5. Shape SLat flow DiT (same arch, sparse tokens, in 32 ch) at 512 (and 1024 for cascade).
6. Shape decoder FlexiDualGridVaeDecoder: SparseConvNeXt blocks [4,16,8,4,0] at channels [1024,512,256,128,64], 4 C2S upsamples (subdiv predicted), 7 out ch (3 dual-vertex offsets, 3 intersect flags, 1 quad split weight).
7. Texture SLat flow DiT: in 64 ch (32 noise + 32 shape slat as concat_cond), out 32.
8. Texture decoder (same net, pred_subdiv=false, guided by the shape decoder's subdivisions) -> 6 ch PBR per voxel (base color 3, metallic, roughness, alpha) * 0.5 + 0.5.
9. flexible_dual_grid_to_mesh (hash lookup of 4 neighbour voxels per intersected edge, quad split by weight) -> mesh. fill_holes.
10. to_glb: simplify, UV unwrap (CuMesh), rasterize UV space, trilinear sample the voxel attribute volume at each texel's 3D position, inpaint gaps, write GLB with PBR material. Reference needs CuMesh + nvdiffrast + trimesh + cv2: all of this is OURS to write.

Checkpoint sizes (bf16/fp16 on disk): each flow DiT 2.58 GB (1.3B params), shape/tex decoder 0.95 GB each, SS decoder 0.15 GB, DINOv3-L ~1.2 GB fp32. The reference already keeps ONE model on the GPU at a time (low_vram). Peak weights = one DiT.

## 12 GB strategy

- One stage's weights resident at a time (load, run, free): peak weights ~2.6 GB (DiT) or ~1 GB (decoder).
- Weights live as fp16 on the GPU (checkpoint is bf16: convert once at load; verify no overflow vs bf16 range on real activations, else fall back to fp32 accumulate / bf16 emulation).
- Activations: DiT tokens are the 512-path occupancy (tens of thousands max) so a flash-style attention kernel is mandatory (no N x N). Decoder upsampling to 512^3/1024^3 sparse voxels is the big one: keep features fp16, build the 27-neighbour map per level only when that level runs and free it after.
- Bulk tensors never leave the GPU (Rule 4). Only the final mesh/texture crosses, as a terminal file sink (GLB).

## What exists in ILGPU.ML already (verified by grep 2026-10-16)

SafeTensors parser/model, fp16 + low-precision weight dispatch, FusedAttentionKernel, RegisterBlockedMatMul, LayerNorm, elementwise kernels,
BackgroundRemovalPipeline (RMBG 1.4 shaped, takes any segmenter session), GLTFExporter (vertex data), Image3DPipeline (TripoSR, vertex colours only), DiffusionPipeline.
NOT present: sparse tensors, submanifold sparse conv, sparse spatial ops, hash map, RoPE-3D sparse attention, flow sampler, DINOv3, UV unwrap, texture bake, PBR GLB.
No tensor-core use anywhere. SpawnDev.ILGPU has inline PTX (`LanguageIntrinsics`), so `mma.sync` fp16 tensor-core GEMM on the CUDA backend is a library-first perf item (Rule 2) once correctness is done.

## Milestones (each: oracle test at production scale, then GPU, local feed only, no full gates per change)

Status 2026-10-17. Every numerical piece below is gated by `tools/trellis2-harness` against tensors dumped from the REAL reference
classes with the real weights (CPU torch; `reference/golden_*.py`). Commands: `header upload mma attn ssflow slat sampler ssdec decoders dualgrid run512 bench ptx`.

- [x] M0 licenses read; TRELLIS.2 + TRELLIS-image-large MIT, DINOv3 usable (gated), RMBG-2.0 NC (not shipped). Background removal replacement: still open (see M0b).
- [ ] M0b background removal: caller RGBA / mask first; permissive segmenter (BiRefNet MIT: verify weights license + port) second.
- [x] M1 streaming SafeTensors reader, native bf16/fp16/fp32 device residency, one stage's weights at a time (peak 3.1 GB whole 512 run).
- [ ] M2 DINOv3 ViT-L/16 feature extractor. BLOCKED on TJ accepting Meta's terms on HF (gated weights).
- [x] M3 flow DiT (adaLN share_mod, RoPE-3D, qk RMS norm, cross attention): SS 6e-6 / SLat shape 1.9e-5 / tex 1.1e-5 in fp32; bf16 tensor cores within the reference's own bf16 noise.
- [x] M4 Euler flow sampler + CFG + guidance interval + rescale: 2e-7 on all three shipped configs.
- [x] M5 sparse-structure decoder + occupancy: logits 6e-7 fp32, occupancy coordinates identical.
- [x] M6 sparse infrastructure: GPU hash neighbour tables, gathered-row implicit-GEMM conv, subdivision, C2S, ConvNeXt.
- [x] M7 shape decoder + dual-grid mesh extraction: exact in fp32, vertices 6e-9, triangles identical.
- [x] M8 texture decoder (guided): exact in fp32.
- [x] M10a `Trellis2Pipeline.Run512` end to end on real checkpoints (stand-in condition): runs, 3.1 GB peak.
- [ ] M9 mesh post: hole fill, simplify, UV unwrap (CuMesh = chart clustering + xatlas; both MIT, we write our own), UV-space rasterize, trilinear attribute bake, gap inpaint, PBR GLB writer.
- [ ] M10 public entry points (managed / TypedArray / ArrayView1D), Trellis2Options, progress; 1024 cascade (shape decoder `upsample`, 49152 token cap).
- [ ] M11 perf (measured RTX 4070): GEMM 3.7 -> 35 TFLOPS (inline-PTX output registers, see below), flash attention 4.3 -> 9-14 TFLOPS. Next: attention (staging, ldmatrix), bf16 activation storage, fused epilogues, non-CUDA backends, Turing (sm_75: fp16 m16n8k8).
- [ ] M12 text-to-3D = existing text-to-image step feeding M10.

### The inline-PTX local-memory finding (2026-10-17)

`CudaAsm.EmitRef` and single-`out` `Emit` made every MMA operand a local-memory round trip: the first GEMM PTX had 232 `ld.local` and
256 `st.local` and ran at 9% of tensor throughput. Fixed IN THE LIBRARY (SpawnDev.ILGPU 5.3.8-local.5, forks 2.3.13-local.3):
(1) `CudaAsm.Emit` overloads with 2-4 `out` operands and up to 16 operands (`CudaAsm.Generated.tt`);
(2) the PTX backend keeps an `out` local that is only an emit output in the register the statement wrote (`PTXCodeGenerator.Values.cs`,
`IsPromotableEmitOutput`). PTX of the GEMM: 0 `ld.local`, 0 `st.local`. Tests: `BackendTestBase.CudaAsmMultiOut` in the ILGPU repo.

## Oracle problem

No torch on this machine and the reference needs flash-attn/flex_gemm/CuMesh (CUDA/Linux-first). So each module is checked against (a) a small straight-C# CPU oracle written from the Python semantics at the REAL dims for one block, and (b) saved intermediate tensors from the reference where we can produce them. Need TJ's call if we want a real PyTorch run (CPU torch install is possible; flash-attn is not required if SDPA is swapped in).
