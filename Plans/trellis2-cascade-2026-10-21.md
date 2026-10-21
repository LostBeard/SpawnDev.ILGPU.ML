# TRELLIS.2 cascade pipelines (2026-10-20 / 21)

State: ported, gated, measured on the desktop (CUDA) and in the browser (Chrome 156 WebGPU). Uncommitted until TJ says push.

## What was built

`Trellis2Options.Cascade` + `Resolution` 1024 / 1536 -> `Trellis2Pipeline.RunCascadeAsync` (`SampleCascadeAsync` + `DecodeCascadeAsync`, `QuantizeUnique`, token cap loop). Dev cache for iterating on the decoders: `TRELLIS2_CASCADE_CACHE=<dir>` (coords / shape / tex latents). Console: `--resolution 512|1024|1024c|1536c`. Page: Detail select.

## Measurements (RTX 4070 12 GB, 15.7 GB RAM, T.png = the reference's `assets/example_image/T.png`, seed 42, 1M faces, 4096 texture)

| pipeline | ours (CUDA) | reference (WSL2, torch 2.6 cu124) |
|---|---|---|
| 1024_cascade | 201 s (shape flow 87, tex flow 50, 7.1M voxels, GLB 101.9 MB) | 461 s (generate 181.5 + to_glb 280), peak 7.39 GiB |
| 1536_cascade | 1138 s, 46,668 tokens, 16.1M voxels, peak 11.9 GB of 12 | FAILED: CUDA driver error: device not ready |
| 1024_cascade in the browser | 1403 s (Chrome 156): shape flow 534 s, tex flow 306 s, GLB 103.8 MB | n/a |

Quality: ours and the reference are the same class at the same seed (different structures: the noise generators differ); renders in `_scratch/steam/` (gitignored).

## The browser hang (root cause: queue depth)

Every 1024 flow in Chrome 156 lost the WebGPU device (`DXGI_ERROR_DEVICE_HUNG`) 3.5 to 4.5 s after the flow started, 6 of 6 runs, plain 1024 included. NOT the cause (each measured): memory (2.5 GB in use), any single op, attention dispatch size (chunked it anyway), element-wise dispatches over 65535 groups, GEMM shapes, one forward (25 s) or two, 134 s of back-to-back flushed forwards in the PMT WebGPU lane. Cause: `FlowSampler.Sample` queued all 12 steps (24 model calls, ~10 min of GPU time) in 0.3 s of CPU; the fix waits per step (`SampleAsync`). Method: skip op families in the real run (query flag, removed), then per-op and per-kernel drains, which showed everything passing and the loss arriving right after the drains stopped.

## Memory in the browser

A WebGPU buffer here is at most 2 GiB. Chunked: the upsample conv output (3.9 GB at 1024) and the ConvNeXt MLP hidden tensor (3.7 GB). At 1024c the largest remaining tensor is [7.1M voxels, 64] = 1.8 GB. 1536 needs [16M, 64] = 4 GB: not possible in one buffer, so on WebGPU the token cap (`DefaultMaxTokens`) steps the resolution down. Lifting that needs segmented tensors or fp16 for the last decoder level.

## Reference in WSL2 (to reproduce the comparison)

Ubuntu, user zed, no sudo; micromamba env `~/t2` (py3.10, torch 2.6.0+cu124, flash-attn 2.7.3 prebuilt wheel, nvdiffrast v0.4.0, CuMesh, FlexGEMM, o-voxel with conda eigen); weights copied from the Windows model cache to `~/t2/w`; runner `_scratch/ref/ref_run.py` (public timm DINOv3 swap, lazy per-model GPU loading to fit 7 GB RAM, rembg skipped because the image has alpha). `~/t2` is about 25 GB.

## Open

- 1536 in a browser (segmented tensors or fp16 last level); desktop 1536 peaks at 11.9 GB, so a 12 GB card with a busy desktop is at the edge (an OOM-driven step-down would help).
- Re-run the heavy CPU group and the rest of the PMT gate before a release (not run for this batch).
- Offer the examples from the 72 official images.

## The faceted gold visor (astronaut GIF), measured against the reference

The astronaut's mirror-gold visor shows square facets in the GIF. Same input (the astronaut cut-out), same seed 21, same `1024_cascade`, 1M-face target; helmet region = the top 12 % of the model; "normal jump" = the angle between the vertex normals at the two ends of an edge (a smooth sphere of the visor's radius would give about 1.4 degrees at 2.6 voxels):

| | edge length | normal jump, mean / p90 | face vs vertex normal, mean |
|---|---|---|---|
| ours, 1M faces | 2.63 voxels | 16.7 / 46.5 deg | 8.8 deg |
| reference, 963k faces | 2.60 voxels | 11.7 / 33.6 deg | 5.8 deg |
| ours, NOT simplified (7.9M faces) | 0.96 voxels | 6.6 / 15.9 deg | 3.4 deg |
| reference, NOT simplified | 0.95 voxels | 4.2 / 10.6 deg | 2.2 deg |

- Mesh density is the same as the reference's at equal face count, and normals are computed on the welded mesh before the UV split in both (reference: `mesh.compute_vertex_normals()` before `uv_unwrap`). Neither GLB has a normal map.
- The roughness is already in the un-simplified surface (voxel-scale bumps, about 4 degrees in the reference and 6.6 in ours) and grows with the edge length, so simplification (8x) only makes it visible. Ours is about 1.5x rougher than the reference at both stages, but the two runs draw different noise (different RNG), so one pair cannot tell implementation from sample. To settle it: several images/seeds, or feed the reference's remesher our decoder mesh.
- Options, in order of cost: (1) feature-preserving smoothing of the vertex normals (average with neighbours within about 35 degrees; a tiny device kernel next to `VertexNormalsImpl`), which removes the facets on glossy surfaces without touching the geometry; (2) a higher face target (2M) when a model is mostly smooth gloss; (3) a tangent-space normal map baked from the 7.9M-face remeshed surface onto the simplified UV mesh (the bake already has a "read from the original mesh" path): restores curvature lost to the decimation, but a normal map cannot come from the latent (it holds colour, metal, roughness, alpha, not normals). None of these is the reference's behaviour (parity); (1) is the cheap one.
