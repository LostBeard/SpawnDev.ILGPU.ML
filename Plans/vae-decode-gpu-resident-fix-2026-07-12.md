# VAE decode: GPU-resident fix + browser-path data-handling audit

Opened 2026-07-12 (Tuvok, on TJ's directive). Status: **investigation complete, design pending TJ's A-vs-B call.**

TJ's directive: the SD-Turbo WebGPU VAE decode moves whole feature maps into the .NET managed
heap "because SUPPOSEDLY the GPU couldn't handle it." Fix it correctly (data stays GPU/JS-side),
and hunt for other violations of the no-bulk-data-into-.NET rule - suspected root cause of WebGPU
riding ~30% GPU utilization vs ~80-100% on CUDA for the same code.

## The three violations in the SD-Turbo VAE-decode tail (the rest of the pipeline is clean)

Verified: the UNet denoise loop is already GPU-resident (GPU Euler axpy `AddScaledInPlace`,
`ImageGenerationPipeline.cs:317`, no per-step readback). The violations are ALL in the decode tail:

1. **Tiled feature map resident in the managed heap.** `Tiling/TiledFeatureMap.cs` holds the entire
   feature map as `float[][] _tiles`; `TiledVaeUpDecoder` is a hand-rolled reimplementation of the
   VAE up-blocks operating on those managed tiles. This is what OOM'd the WASM managed heap
   (`System.OutOfMemoryException` in `TiledFeatureMap.ReadCore` <- `TiledVaeOps.GroupNorm`), even on
   a SINGLE generation, in the PMT Chromium.
2. **Per-tile upload -> compute -> `SynchronizeAsync` -> `CopyToHostAsync` -> writeback treadmill.**
   `TiledVaeOps.GroupNorm` does this three times per tile across the grid. The GPU stalls on a CPU
   readback every tile -> the ~30% GPU-utilization smoking gun. (Boundary-crossing overhead is NOT
   the cost - TJ already ruled that out; the cost is the bulk bytes crossing + the sync stalls.)
3. **Final image round-trips through .NET too.** `ImageGenerationPipeline.cs:419` `ReadTensorToCpu`
   pulls the whole `[3,512,512]` decode to a managed `float[]`, then a scalar .NET loop over 262,144
   pixels (`:429-440`) packs it to `byte[]`. Should be a GPU kernel (`[-1,1]`->RGBA pack) + zero-copy
   GPU->canvas via `CanvasRendererFactory`.

## Why it happened - MEASURED, not assumed (CUDA, RTX 4070, full-res VAE decode, VAE-only counters)

| Config | peak LIVE (working set) | peak TOTAL (pool-resident) |
|---|---|---|
| full-res, default backlog cap 512 MiB | **896 MiB** | **3224 MiB** |
| full-res, backlog cap 64 MiB (`VAE_BYTECAP_MB=64`) | 896 MiB | **1800 MiB** |

The genuine working set is **896 MiB**. The pool hoarded **3224 MiB** - ~2.3 GiB of freed-but-
retained + deferred buffers sitting resident. THAT is what crossed the browser's per-process D3D12
budget (which is below physical VRAM; the 12 GB card was never the limit). The original "fix"
responded to a POOL-RECLAIM problem by shoveling feature maps into .NET.

Two separable contributors to the bloat:
- **Deferred-release backlog** (`GraphExecutor.MaxPendingReleaseBytes`, default 512 MiB): buffers
  referenced by pending WebGPU dispatches can't be reclaimed until a sync/drain. Bounding it 512->64
  cut TOTAL 3224->1800 MiB.
- **Bucket retention** (`BufferPool._buckets`): Returned buffers are kept in size-buckets for reuse
  and only released via `DisposeBucketedBuffers()`, which fires ONLY under GPU memory pressure
  (`AllocateWithReclaim`). A fat CUDA card never hits pressure -> hoards ~900 MiB beyond the cap.
  `ForceReclaimEveryNRents` already forces proactive `DisposeBucketedBuffers` (made WebGPU-safe in
  preview.12: 238 safe reclaims disposing ~15 GB mid-forward).

## The fix - two options, both keep data GPU-side (the decision is TJ's)

**Option A - full-res GPU decode + proactive reclaim (delete ALL tiling).** If proactive reclaim
(bound `MaxPendingReleaseBytes` + `ForceReclaimEveryNRents`/bucket-trim during VAE decode) drops the
resident total near the 896 MiB working set, and 896 MiB + resident weights fits the browser budget,
then delete `TiledFeatureMap` + `TiledVaeUpDecoder` + `TiledVaeOps` (~600 lines) and run the full-res
ONNX VAE graph GPU-resident. Simplest, removes the whole workaround. **Unproven: does 896 MiB working
set + weights fit the browser per-process budget?**

**Option B - GPU-resident tiling.** If the 896 MiB working set alone still crosses the budget, tile
the decode but keep tiles as WebGPU storage buffers (not managed float[]); halo exchange = GPU
buffer-to-buffer copies (`CopyFrom`, works on all backends); .NET only dispatches. Bounds the working
set to ~250 MiB (2x2) the way the current code does - but GPU-side. Guaranteed to fit; more work than A.

**Decisive experiment (run before cutting):** SD-Turbo full-res on WebGPU (`VaeTileGrid = -1`) with
`ForceReclaimEveryNRents` + low `MaxPendingReleaseBytes` set in the test (WASM can't read env vars).
Survives -> Option A. OOMs -> Option B. Uses existing knobs, no new library code.

Both options also do violation #3: GPU-side RGBA pack kernel + zero-copy GPU->canvas.

## Broader audit - other violations of the no-bulk-data-into-.NET rule

Subagent swept the browser inference path (findings below are the agent's; **each must be re-read/
verified before acting - Rule 6c**). The PRODUCTION-hot paths are largely clean (GGUF decode via
`WebGPUDecodeCapture` reused-pinned-buffer pattern; DAv3 `Dav3Inference`/`AsyncDepthPipeline`;
`GraphExecutor` per-node readback bounded to <=64-element shape metadata). Violations cluster in
secondary pipelines:

1. `Pipelines/DiffusionPipeline.cs:77-124` - generic diffusion (NOT SD-Turbo): per-step full-latent
   readback + CPU DDPM step. (SD-Turbo's `ImageGenerationPipeline` already fixed this pattern.)
2. `Pipelines/AudioPipelines.cs:85-121` - Whisper decode: full ~51,865-float vocab readback per token
   for greedy; should use `GpuArgMax.ArgMaxAsync` (1 int).
3. `Pipelines/NLPPipelines.cs:401-421` - GPT-2 text-gen: fresh vocab alloc + readback + separate fence
   per token; should use the `WebGPUDecodeCapture` reused-buffer/single-fence pattern.
4. `Pipelines/ImageGenerationPipeline.cs:572-626` `TiledVaeDecodeAsync` - the approximate opt-in tiled
   path (same archetype as the exact one; `VAE_TILE_LATENT`).
5. `Operators/EinsumOperator.cs:226-243` general contraction - reads inputs to host, contracts on CPU;
   hot only if a shipped model routes through the general (non-fast) path. NEEDS model-usage evidence.
6. `Pipelines/GgufGenerator.cs:291-294` - non-capture sampled fallback (production uses the clean
   capture path); same reused-buffer fix.
7. `Pipelines/DepthEstimationPipeline.cs:195/365` - full depth readback + CPU min/max/normalize;
   violation only if driven per-frame (the per-frame `AsyncDepthPipeline` is clean).
8. `Operators/MoEOperator.cs:98` - per-layer router-logits readback (small/metadata; fence-count cost).

The unifying fix for 2/3/6 already exists in-repo: `WebGPUDecodeCapture` reused-pinned-buffer +
single-fence, and `GpuArgMax.ArgMaxAsync` for greedy.

## IMPLEMENTED 2026-07-12 (Tuvok) — status

Direction chosen: **Option A** (TJ directive: unnecessary copies = treason, fix all three violations).

- **Violations 1&2 — DONE + VERIFIED.** `ImageGenerationPipeline` VAE-decode default flipped from auto-tile-
  into-.NET to **full-res GPU-resident** + browser-conditional proactive reclaim (`ForceReclaimEveryNRents=8`
  + `MaxPendingReleaseBytes=64MiB`, gated `!EnableGraphCapture`; desktop runs full-res with no reclaim churn).
  VERIFIED: `SDTurbo_WebGPU_ImageGen_MultiGen` GREEN — 4 gens on WebGPU, no OOM, per-gen peak-TOTAL leak guard
  flat. The opt-in exact/latent tiled paths remain reachable via `VaeTileGrid>0` / `VAE_TILE_*` (legacy).
- **Violation 3 (pack) — kernel DONE + PROVEN, integration in progress.** New `ElementWiseKernels.
  NchwDenormToRgba` GPU kernel packs the VAE output NCHW[-1,1] -> RGBA int on-GPU, replacing the 3*px float
  readback + 262K-iteration .NET pixel loop. Regression guard `NchwDenormToRgba_MatchesCpuPixelLoop` GREEN on
  all 8 backend variants (±1/channel vs CPU ref, alpha exact). Integration (full pipeline + pack) being tested
  on CUDA + WebGPU.
- **REMAINING:**
  - One terminal `int[]->byte[]` `Buffer.BlockCopy` bridges the int-packed GPU buffer to the `byte[] ImageRGBA`
    API. Necessary only because the contract is `byte[]`. **Eliminate fully** by delivering browser images
    GPU-buffer -> `IBrowserMemoryBuffer.CopyToHostUint8ArrayAsync` -> canvas (infra already in
    `SpawnDev.ILGPU/Rendering/`, e.g. `CPUCanvasRenderer`) — no .NET image bytes at all. This is an
    `ImageGenerationResult` contract + demo change (needs TJ's call on the shape).
  - Retire the .NET tiling code (`TiledFeatureMap` + `TiledVaeUpDecoder` + `TiledVaeOps`, ~600 lines) now that
    it is opt-in-only — TJ decision (keep opt-in for comparison vs delete outright).
  - The other audit violations (Whisper/GPT-2 per-token vocab readbacks, etc.) are now MANDATORY per the
    treason directive; secondary to the SD-Turbo path.

## preview.13 status
On HOLD. Its GPU-process break-leak fix (CUDA-verified, peak-flat across gens) is a leak-patch on the
tiled-decode workaround this plan removes/reworks. Do not ship it as a standalone band-aid (Rule 1).
Re-evaluate after the VAE decode is fixed correctly. Managed-heap multi-gen OOM tracked separately in
`Plans/wasm-managed-heap-multigen-oom-2026-07-12.md` (that plan is subsumed by this one - the managed
heap should carry NO feature-map data after this fix).
