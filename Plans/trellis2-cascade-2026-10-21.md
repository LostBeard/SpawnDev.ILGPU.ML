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
