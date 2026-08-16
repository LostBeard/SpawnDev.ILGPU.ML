# SD-class graph capture: pool priming at scale

**Status: TRACKED GAP** (2026-07-03). `ImageGenerationPipeline.EnableGraphCapture` defaults OFF
until this lands (`SDTURBO_FORCE_CAPTURE=1` opts in for development). Depth (DA-class) captures
fine; SD-Turbo's CLIP captures fine; **UNet and VAE captures crash** with native 0xC0000005.

## Root cause chain (bisect-proven, Examples/03.ImageGen.Console ~2min repro)

1. The capture pass must perform ZERO buffer-pool misses: any miss enters
   `BufferPool.Rent → ILGPU AllocateWithReclaim`, which flushes pending GPU work (a sync on a
   capturing stream invalidates capture) and may `cuMemAlloc`/`cuMemFree` mid-capture - the CUDA
   context is corrupted and the NEXT `cuMemFree` anywhere AVs.
2. DA-class models were pool-primed by the warm passes (d0cd5a6). SD-class activation volumes
   (UNet 1.7GB weights + big feature maps; VAE 512x512 decode) exceed what the warm passes leave
   pooled under the deferred-release byte cap + VRAM pressure on a 12GB card → the capture pass
   misses.

## Fixed already (keep - correct for all captured models)

- `WhereBroadcastND` + `InstanceNorm(/InPlace)` per-call allocs → capture-slot/cached scratch.
- `SessionGraphCapture` owns stable input clones (capture binds input buffers into the graph;
  caller transients being disposed was the first crash).
- `BufferPool.DisposeBucketedBuffers` refuses to free under `SuppressDrains` (mid-capture reclaim).
- Capture attempts are best-effort (`SessionGraphCapture` falls back to direct on failure) -
  but note the CONTEXT-POISONING failures are not catchable, hence the default-off.

## The remaining work

- Make the capture pass provably miss-free: warm passes must leave every size-bucket the capture
  sequence rents available (audit: byte-cap drain between warm and capture; SuppressDrains changes
  release timing so the capture pass's liveness profile differs from warm - align them).
- OR: pre-flight VRAM/pool check before attempting capture (skip capture when the model's measured
  pool profile cannot be held resident) - honest degradation without entering the crash window.
- Verify with `SDTURBO_FORCE_CAPTURE=1` on Examples/03 (CUDA), then the browser E2E; also re-visit
  the WebGPU Range-EMPTY-under-capture-regime issue (CLIP warm pass, browser lane only).

## Update 2026-07-12 (Tuvok) — WebGPU blocker moved to REPLAY: capture-plan buffer vs reclaim conflict

Took a swing after the full-res GPU-resident VAE landed (preview.13). On WebGPU now:
- **Capture of CLIP + UNet SUCCEEDS** (no 0xC0000005 — the CUDA context-poison is CUDA-specific).
- **REPLAY FAILS**: `[WebGPU] 1 GPU error(s) during dispatch: [Buffer (unlabeled)] used in submit while
  destroyed` (at `WebGPUGraphCapture.ReplayAsync` → `Queue.Submit`).

Root cause: the full-res VAE decode's browser buffer RECLAIM (`ForceReclaimEveryNRents` ->
`DisposeBucketedBuffers`) destroys pooled buffers that the captured CLIP/UNet **replay plan still
references**. Next gen's replay submits a destroyed buffer -> GPU error. And the inverse (skip reclaim
under capture, the old `!EnableGraphCapture` gate) OOMs the full-res VAE. So capture/replay and the
memory-bounding reclaim are in **direct conflict** for this pipeline:
- reclaim ON  -> full-res VAE fits, but destroys captured-plan buffers -> replay error.
- reclaim OFF -> captured plans survive, but full-res VAE hoards ~3.2 GiB -> OOM.

**The real fix (either):**
1. Captured plans must OWN / PIN their buffers so `DisposeBucketedBuffers` skips them (pool marks
   capture-plan buffers non-reclaimable), OR
2. Capture the VAE too (so its buffers are plan-owned, not pooled) — which loops back to the pool-miss-
   free capture pass at SD scale (the original tracked work above).

This is a genuine capture/pool lifetime design problem, not a quick fix. Reverted the swing's plumbing
(kept shipped preview.13 clean). The full-res-VAE win (10.2s warm gen on gh-pages) stands independent
of capture; capture/replay would push toward ~2-3s but needs this resolved first.
