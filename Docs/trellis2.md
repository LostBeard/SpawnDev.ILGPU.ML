# TRELLIS.2: image and text to 3D, on the desktop and in the browser

`SpawnDev.ILGPU.ML.Trellis2` runs [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (Microsoft, MIT) end to end in C#: a photo, or a
text prompt, becomes a textured PBR 3D model (GLB: positions, normals, UVs, base colour and metallic-roughness textures). Every
network stage runs as SpawnDev.ILGPU kernels, so the same code runs on CUDA (tensor cores), OpenCL, CPU and **in the browser on
WebGPU**. To our knowledge the demo page is the first fully client-side TRELLIS.2 on the web: no server, no upload, the model
weights stream into the browser and everything runs on the visitor's GPU.

**Try it:** [lostbeard.github.io/SpawnDev.ILGPU.ML/text-to-3d](https://lostbeard.github.io/SpawnDev.ILGPU.ML/text-to-3d)
(`/image-to-3d` opens the photo mode).

![Text to 3D in the browser: "a wooden treasure chest with iron bands"](../SpawnDev.ILGPU.ML.Demo/wwwroot/screenshots/2026-10-17-16-55_TextTo3D-Chest.jpg)

## What runs

| Stage | What | Where it runs |
|-------|------|---------------|
| Text to image (text mode) | The demo and `AssetGenerator` draw with FLUX 3 Image Klein (with the Qwen3 caption writer), FLUX.2 klein or SD Turbo, your choice ([flux3.md](flux3.md), [flux2.md](flux2.md)); the library's `Trellis2TextTo3D` draws with SD Turbo (`ImageGenerationPipeline`). A drawing whose cut-out touches the frame is a truncated model, so the next seed is tried | GPU |
| Cut-out | RMBG-1.4 in the demo (`BackgroundRemovalPipeline`); desktop `Trellis2TextTo3D` uses IS-Net (`ForegroundMask`) | GPU |
| Conditioning | DINOv3 ViT-L/16 features of the 512 px image | GPU |
| Sparse structure | 1.3B flow transformer (12 Euler steps, classifier-free guidance) + structure decoder -> occupied cells | GPU |
| Shape | 1.3B shape flow on the sparse voxels + SC-VAE decoder -> FlexiDualGrid mesh (1-3M triangles) | GPU |
| Texture | 1.3B texture flow + SC-VAE decoder -> PBR attributes per voxel | GPU |
| Mesh repair | `MeshRepair` on the dual-grid mesh: non-manifold edges, winding, small holes, floating fragments | host |
| Remesh | `NarrowBandRemesher` (the reference's `to_glb(remesh=True)`): the distance field of the mesh (`MeshDistanceGrid`, on the device) shifted one voxel, extracted by dual contouring on the voxels the band crosses. Closes pinholes and cracks, turns a thin plate into one watertight outward-facing skin; the skin inside a closed solid is dropped | GPU distances, host topology |
| Denoise | `DeviceMeshDenoiser`: bilateral normal filtering, then the vertices move to where the filtered normals put them (4 rounds by default, about 0.1 s on a 1M-face mesh); takes the voxel-scale bumps out of the remesh so a glossy surface does not show facets. On with `Trellis2Options.SmoothNormals` | GPU |
| Mesh | quadric edge collapse in parallel rounds on the device (`GpuMeshSimplifier`) to `Trellis2Options.MaxFaces` triangles (150,000 by default; `AssetGenerator` uses 300,000, the demo offers up to 1M). The host `MeshSimplifier` and `MeshRepair` remain as the reference path (`TRELLIS2_HOSTSIMPLIFY=1`) | GPU |
| UV atlas + bake | `DeviceUvAtlas` aligns the winding, computes the normals and grows and packs the charts on the device (`UvAtlas` is the host path, `TRELLIS2_HOSTATLAS=1`). The charts grow from bilaterally filtered face normals (`UvAtlas.Options.ChartNormalSmoothingIterations`, 3 passes at 30 degrees; folded faces are dissolved one by one), so a smooth dome is a few large charts instead of thousands of small ones: on a 1M-face astronaut 12,343 charts instead of 30,630, 65 % of the atlas used instead of 53 %, 28 % more texels per unit of surface. `TextureBaker` rasterises the atlas, moves every texel onto the nearest point of the ORIGINAL (pre-remesh) mesh (`MeshDistanceGrid.ProjectTexels`, the reference's `bvh.unsigned_distance`) and samples the voxel attributes there, so the colour is read exactly on the surface the decoder painted | GPU |
| GLB | PNG encode + `PbrGlbWriter` | host |

Every network stage was gated against the reference implementation's own classes (CPU torch goldens, `tools/trellis2-harness`).

## Performance (MEASURED, RTX 4070 12 GB)

| Where | Path | Time |
|-------|------|------|
| Desktop, CUDA | photo -> GLB, 512 pipeline (tensor-core GEMM + flash attention, inline PTX) | ~40 s, 3 GB peak |
| Desktop, CUDA | 1024 pipeline (`Trellis2Options.Resolution = 1024`) | plant 73 s, crown 141 s, 3.6 GB |
| Desktop, OpenCL | photo -> GLB (portable kernels) | ~83 s |
| Browser, WebGPU (Chrome, AOT build) | photo -> GLB, weights cached in OPFS, fresh page | 148 s |
| Browser, WebGPU | same, next generation in the session | 134 s |
| Browser, WebGPU | text -> GLB, everything cached | 180 s |
| Browser, WebGPU | text -> GLB (SD Turbo first download included) | 246 s |
| Browser, WebGPU | first ever run (10 GB of weights to download) | a few more minutes, bound by the download |

The cascade pipelines (`Trellis2Options.Cascade = true`, `Resolution` 1024 or 1536; `07.AssetGen.Console --resolution 1024c|1536c`; the Image to 3D page's "Ultra (1024 cascade)" and "Max (1536 cascade)") are the reference's own `1024_cascade` (its default) and `1536_cascade`:
the structure at 32^3 and the shape latent come from the 512 checkpoint, the shape decoder's upsampling gives the high-resolution token set, and the 1024 shape and texture flows run on those tokens. Measured on the same RTX 4070 12 GB with the reference's example image `T.png`, seed 42, 1M faces, 4096 texture:

| Where | Pipeline | Time |
|-------|----------|------|
| Desktop, CUDA | `1024_cascade` | 201 s (the reference's own pipeline on the same card: 461 s) |
| Desktop, CUDA | `1536_cascade` | 1138 s, peak 11.9 GB of the 12 GB (the reference's `1536_cascade` fails on this card with a CUDA driver error) |
| Browser, WebGPU (Chrome 156) | `1024_cascade` | 1403 s, 103.8 MB GLB |

In a browser the cascade's token cap follows the device's buffer limit (20,971 tokens on a 2 GiB adapter), so "Max (1536)" steps down towards 1024 instead of failing; a 1536 decode's finest level is a 4 GB tensor and needs a device with larger buffers. Details: `Plans/trellis2-cascade-2026-10-21.md`.

In the browser almost all of the remaining time is the three flow transformers on the GPU (~97 s of 134 s): the portable kernels reach
~8 TFLOPS (tiled GEMM) and ~5.6 TFLOPS (flash attention) on WebGPU, where CUDA uses tensor cores.

## How the browser version fits

- **Portable kernels.** WebGPU has no tensor cores and no inline assembly: a register-tiled GEMM (one kernel for bf16 / f16 weights and
  dense / gathered rows) and a flash attention (8 lanes x 4 queries per thread, warp shuffles) carry every transformer and sparse
  convolution. Generated by `tools/trellis2-harness/reference/gen_tiled_gemm.py` / `gen_tiled_attention.py`.
- **10 GB of weights.** `Trellis2Weights.FromModelSource` reads checkpoints one stage at a time from any `IModelSource`.
  `RangeStreamModelSource.PlanCacheAsync` caches the files in OPFS and streams only what the browser has no room for, with range
  requests (`FetchRangeStream`: four requests in flight, bytes go JS -> GPU without entering the .NET heap). The quota from
  `navigator.storage.estimate()` is not a limit: Chrome 151 reports `usage + 10 GiB` however much it would take (14 GiB written against a
  10 GiB "quota"), so only an exact headroom bounds the plan (`BrowserStorageEstimate`) and a write the browser really refuses falls back to
  streaming that file (`CacheFallback`).
- **Shader compilation.** The browser compiles a WebGPU shader at its first launch, one at a time with the queue waiting (seconds each
  for these kernels). `WebGPUAccelerator.PrecompileAsync` (SpawnDev.ILGPU 5.3.8) builds the large kernels' pipelines in the background
  and in parallel at the start of a generation.
- **Host work.** The mesh simplification and the UV atlas run in device kernels by default, so what is left on the host is the PNG
  encode, the glTF writing and the host reference paths. The demo is still published with Blazor WebAssembly AOT (`-p:DemoAot=true`):
  the host simplifier and atlas are 30-45x slower in the interpreter (41 s -> 2.6 s and 17 s -> 0.8 s measured on 2026-10-17, before the
  device versions), and the rest of the page benefits too.
- **Mesh size.** The 3M-triangle dual-grid mesh is clustered on the device first (O(n)), so the sequential edge collapse only does the
  last step.

## API

Desktop, local checkpoint files:

```csharp
using SpawnDev.ILGPU.ML.Trellis2;

using var gen = new Trellis2ImageTo3D(accelerator, new Trellis2Paths
{
    ModelDir = @"...\TRELLIS.2-4B",
    SparseStructureDecoder = @"...\TRELLIS-image-large\ckpts\ss_dec_conv3d_16l8_fp16",
    DinoWeights = @"...\dinov3\model.safetensors",
});
byte[] glb = await gen.GenerateGlbAsync(rgbaPixels, width, height,   // RGBA, alpha = foreground
    new Trellis2Options { Seed = 42, MaxFaces = 150_000, TextureSize = 2048 });
```

Text to 3D on the desktop: `new Trellis2TextTo3D(accelerator, paths, imageModelSource, isnetOnnxPath).GenerateGlbAsync(prompt)`.

Browser (or any `IModelSource`): weights from the models host, cached where they fit, streamed otherwise:

```csharp
var ranged = new RangeStreamModelSource(hubModelSource);          // a HubModelSource constructed with an HttpClient
await ranged.PlanCacheAsync(Trellis2Weights.HubWeightFiles(), await hubModelSource.GetStorageEstimateAsync());
using var gen = new Trellis2ImageTo3D(accelerator, Trellis2Weights.FromModelSource(ranged));
byte[] glb = await gen.GenerateGlbAsync(rgbaPixels, width, height);  // StageDone reports each stage and its seconds
```

The demo page (`SpawnDev.ILGPU.ML.Demo/Pages/Image3DPage.razor`) shows the whole flow, including the FLUX 3 / FLUX.2 / SD Turbo picture models and RMBG-1.4 for text mode.

`Trellis2Options` (all `init`): `Seed` (42), `TextureSize` (2048), `Resolution` (512, 1024 or, with `Cascade`, 1536), `Cascade` (false), `MaxFaces` (150,000; 0 keeps the whole dual-grid mesh), `RepairMesh` (true), `Remesh` (true),
`SmoothNormals` (true: the device mesh denoiser and bilaterally filtered export normals; the reference writes plain area-weighted normals, so this is a deliberate difference). `Resolution = 1024` without `Cascade` is the plain `1024` pipeline, whose colours come out paler than 512's: use 512 for colour fidelity, 1024 for fine geometry.

## Tests

`Trellis2_TiledLinear_MatchesHostReference`, `Trellis2_TiledSparseConv_MatchesHostReference`, `Trellis2_FlashAttention_MatchesHostReference`
and `Trellis2_MeshClusterer_SphereStaysOnSurface` run the portable kernels against host double references on WebGPU, CUDA, OpenCL and
CPU; `Remesh_*` (distance grid against a brute-force oracle, the remesh of a sphere, a pinhole and a thin double sheet, the projection) run on CUDA and OpenCL; `Trellis2_UvAtlas_*`, `Trellis2_MeshSimplifier_*`, `Trellis2_PbrGlb_*` cover the host stages; `Trellis2_ImageToGlb_EndToEnd`
(HeavyModel) runs the real model. `Trellis2_DiTKernels_Benchmark` / `Trellis2_KernelCompile_Benchmark` (HeavyModel) measure throughput
and shader compile cost.

## Next step: rig it

A generated character can be given a skeleton and learned skin weights with `SkinTokensRigger` (and the demo's **Rig this model** button): see [rigging.md](rigging.md).

## Licences

TRELLIS.2 code and weights: MIT. DINOv3 weights: Meta's DINOv3 licence ("Built with DINOv3"). SD Turbo: Stability AI's licence.
RMBG-1.4: BRIA's licence. Weights are not shipped in the package; the demo downloads them on first use from the SpawnDev models host (`models.spawndev.com`, Hugging Face as the automatic fallback)
and caches them in the browser. Check each licence before commercial use.

## Why the remesh (measured)

Without it the generated mesh is the raw dual-grid mesh, simplified. A thin plate of armour is two surfaces one voxel apart, and the texture decoder paints the inner one near black; the raw mesh also has voxel-sized pinholes. Baking colour at the simplified mesh's own positions lets that dark inner layer show through wherever the surface sits a little deep, and the pinholes show it directly: a knight came out with dozens of black dots (the "holes"). Measured on that knight: 964 hole loops longer than the hole-fill limit, 34% of the texture voxels near black, 15k dark islands in the baked colour, and none of the cheaper fixes (recolouring voxels, filling texture islands, smoothing normals, flipping triangles) removed the dots. The reference does not have the problem because `to_glb(remesh=True)` rebuilds the surface and reads colour from the original; this is that, ported (`Trellis2Options.Remesh`, default on).

## Editing a region of a generated asset (prototype, 2026-10-19)

`Trellis2Pipeline.RetextureRegionAsync` / `Trellis2ImageTo3D.EditRegionAsync` / `AssetGenerator.EditRegionAsync` re-texture part of a generated asset from a new picture and keep the rest. Training-free, RePaint style: every generation now keeps its structured latents (`Output.Edit`, a few MB on the host). The edit runs the texture flow again, conditioned on the SAME shape latent (so the geometry cannot change) and the new picture's DINOv3 features, and after every Euler step puts the tokens OUTSIDE the region back to the original texture latent noised to the current time (`x_t = (1 - t) x0 + (sigma + (1 - sigma) t) eps`, the flow's own path; `FlowSampler.Sample` has an `afterStep` hook for it). The region is a predicate on a token's centre in the generator's frame (x right, y depth, z up, volume [-0.5, 0.5]^3); `AssetGenerator.RegionFromSpec("z>0.08")`, `box:...`, `sphere:...` and unions make them from text; the console takes `--edit-picture p.png --edit-region "z>0.08"`.

Measured on the knight (CUDA, 512 pipeline, 1193 tokens): upper body (480 tokens) re-textured from a red-armour picture in ~20 s on top of the generation (DINOv3 3.6 s, shape decode 2.8 s, texture flow 1.8 s, texture decode 2.7 s): red enamel and gold trim on head, chest and shoulders, legs and gauntlets still silver, no seam at the cut. Test `Trellis2_RegionRetexture_KeepsTheRestAndChangesTheRegion` (real model, 5553 tokens): voxels far outside the region change by at most 0.0025, voxels inside by 0.74 on average.

### Editing the shape of a region

`Trellis2Pipeline.EditShapeRegionAsync` / `Trellis2ImageTo3D.EditShapeRegionAsync` / `AssetGenerator.EditShapeRegionAsync` (console `--edit-shape`) go one stage earlier: the sparse-structure flow is re-sampled under the new picture with the latent cells outside the region pinned (RePaint again), which decides WHICH tokens exist, so geometry can be added or removed inside the region; the shape flow then runs on the new token set with the tokens that existed before and lie outside the region pinned to their old shape latent, and the texture flow likewise; mesh and texture are decoded. Every generation keeps its structure latent for this (`EditState.SsLatent`). `--edit-picture` takes a prompt as well as a file (the edit picture is drawn with the same image model).

Measured on the knight (CUDA, 512): head region (`z>0.385`, 512 of 4096 structure cells) edited with a picture of a knight in a helmet with a big red plume: a red plume appears on the helmet, the helmet follows the new picture, shoulders, torso and legs are unchanged; 1193 -> 1201 tokens; ~45 s on top of the generation (structure flow 11 s, shape 3 s, texture 2 s, decoders and loads). A cut that runs through the shoulders (`z>0.33`) hollows the top of the pauldrons: choose the region to follow the part you mean. Test `Trellis2_ShapeEdit_ChangesTheRegionsGeometryAndKeepsTheRest` (real model): far outside the region 31 of 642,475 voxels disappear and 26 appear (the decoder's reach at the edges), mean colour change 0.00008; inside, 619,548 of 649,139 voxels change.

Not done: editing an arbitrary mesh (needs the shape and texture ENCODERS, which TRELLIS.2 ships: `shape_enc_next_dc_f16c32_fp16`, `tex_enc_next_dc_f16c32_fp16`), inversion-based preservation (VoxHammer) for harder edits, and a region-edit UI in the demo (the console takes `--edit-picture`, `--edit-region` and `--edit-shape`; a prompt works in place of a picture because the edit picture is drawn with the same image model).
