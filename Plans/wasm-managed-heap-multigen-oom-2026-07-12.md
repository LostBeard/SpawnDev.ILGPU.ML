# WASM managed-heap OOM in the WebGPU multi-generation SD-Turbo path

Status: **OPEN / tracked** (deferred, NOT gating preview.13). Opened 2026-07-12 (Tuvok).

## What is observed (evidence, not hypothesis)

Running `WebGPUTests.SDTurbo_WebGPU_ElideAB` (a 2-generation A/B: elide-on vs elide-off,
held simultaneously for a pixel-diff compare) on the RTX 4070 under BlazorJS 3.5.25 +
ILGPU 4.17.6 + ML preview.13 fails:

```
System.OutOfMemoryException: Out of memory
  at MLTestBase.<SDTurbo_WebGPU_ElideAB>b__792_0(Accelerator accelerator)
Failed WebGPUTests.SDTurbo_WebGPU_ElideAB [3 m 56 s]
```

This is a **managed .NET / Mono WASM-heap** `OutOfMemoryException` (a CLR allocation failing
inside the test lambda), **NOT** the GPU-process OOM that preview.13 fixes. The two are
distinct failure modes:

| Failure | Layer | Signature | Status |
|---|---|---|---|
| GPU-process OOM (2nd-gen) | Browser GPU process / D3D12 budget | "GPU process died due to out of memory" -> device lost -> "external Instance reference no longer exists" | **FIXED** preview.13 (break-leak reclaim, `GraphExecutor`) |
| Managed-heap OOM (multi-gen) | .NET/Mono WASM managed heap | `System.OutOfMemoryException: Out of memory` | **OPEN** (this doc) |

The prior preview.12 CHANGELOG also references a WebGPU-only 4-generation HeavyModel test
(`SDTurbo_WebGPU_ImageGen_MultiGen`) in the same multi-generation class; whether that test hits
the identical managed-heap ceiling is **not yet confirmed** and should be checked as step 1.

## What is NOT yet known (do not assert until measured)

- Whether the managed OOM is driven by the A/B test **holding two full pipelines/results
  simultaneously** (an artifact of the ElideAB compare), or by **per-generation managed-heap
  growth that does not release between generations** (a real accumulation the production
  multi-gen path would also hit).
- Whether a **single** SD-Turbo WebGPU generation stays comfortably under the managed-heap
  ceiling. (The `SDTurbo_Generate_E2E` WebGPU-only 1-gen run started 2026-07-12 is the first
  data point; update this doc with its result.)
- Whether the growth is `Mono` high-water reporting vs. a true unreclaimed managed allocation
  (recall the 2026-06-14/15 red herring: `usedJSHeapSize` is Mono high-water, not a leak -
  memory `project-ml-backend-priority-fast-first`). Measure the actual managed GC heap, not JS.

## Investigation approach (when this is picked up)

1. Confirm the reproduction on `SDTurbo_WebGPU_ImageGen_MultiGen` (4-gen) and record whether
   the OOM gen index is deterministic (like the GPU-process one was at gen 2).
2. Instrument the managed heap between generations: `GC.GetTotalMemory(true)` before/after each
   generation to separate "grows and never releases" from "single-gen peak too high".
3. If it accumulates: find the managed-side references retained across generations (pipeline,
   executor, capture caches, HeapView-tagged views, result byte[]s). The ImageGen pipeline is
   `using`-scoped per generation in `SDTurbo_Generate_E2E`; check the A/B and MultiGen paths do
   the same and that nothing static (capture/replay caches) pins per-gen buffers.
4. If single-gen peak is the driver: the managed footprint of one SD-Turbo generation itself
   needs reduction (bulk image/tensor bytes should live JS-side per the browser-data rule, not
   the WASM managed heap - memory `feedback-browser-gpu-opfs-data-must-stay-in-js-never-dotnet`).

## Why deferred (not gating preview.13)

preview.13 fixes the GPU-process OOM (the crash TJ actually hit live on gen 2) and is
CUDA-verified peak-flat across 3 gens (bit-identical gen1==gen3). The managed-heap multi-gen
OOM is a separate defect that predates preview.13 and is not introduced by it. TJ's Option A
decision (last session) was: ship the GPU-process break-leak fix now; treat the WebGPU
multi-gen managed-OOM as a tracked follow-up. This file is that tracking.
