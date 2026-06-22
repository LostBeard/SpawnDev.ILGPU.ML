# CUDA-graph decode capture — closing the dispatch-overhead gap to Ollama

Status: PLANNED (next session, with Geordi). In-repo technical companion to memory
`project-tuvok-HANDOFF-cuda-graphs-decode-dispatch-2026-06-22` + DevComms
`_DevComms/SpawnDev.ILGPU/tuvok-cuda-graph-decode-ask-2026-06-22.md`.

## Why
Decode (qwen2.5-coder:7b Q4_K_M / RTX 4070) ≈ **32 ms/tok = ~6 ms GPU + ~25 ms CPU dispatch** (~703 nodes/step,
~35 µs/node). The decode GEMV is already solved (dp4a, 54% of peak). The remaining ~2.7x to Ollama is pure
CPU dispatch overhead. llama.cpp eliminates it with CUDA graphs. Target: decode → ~6-10 ms (beat Ollama ~12 ms).

Reproduce the split (each step prints `wall / drain(GPU) / residual(CPU) / bufs`):
```
GGUF_GEMV_DP4A=1 GGUF_GEN="Write a function." GGUF_GEN_RAW=1 GGUF_GEN_KV=1 GGUF_GEN_N=16 \
  dotnet run --project Examples/04.GGUFTextGen.Console -c Release -- \
  C:/Users/TJ/.ollama/models/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463
```

## Approach
For M=1 decode the kernel sequence/shapes/buffers are identical every token (production recycles output buffers
via `CacheShapeReadbacks` → stable device pointers). So:

1. Run/`cuStreamBeginCapture` ONE warm decode step on the executor's CUDA stream → `cuStreamEndCapture` →
   `cuGraphInstantiate`.
2. Each subsequent token = one `cuGraphLaunch`. ~zero CPU dispatch.
3. Per-step-changing inputs (new token id, KV position `DecodePastLen`): either
   `cuGraphExecKernelNodeSetParams` on the few affected nodes, or store them in a small device buffer the
   captured kernels read (preferred — no graph edit per step). The KV cache writes are in-place at the position;
   the position must be a device-readable value, not a host constant baked into the launch.

## Enabler (already confirmed)
`ILGPU.Runtime.Cuda.CudaStream.StreamPtr` (IntPtr) exposes the raw CUstream (`CudaStream.cs:70`). ILGPU has no
native CUDA-graph API, so add a thin P/Invoke helper over the CUDA Driver API:
`cuStreamBeginCapture / cuStreamEndCapture / cuGraphInstantiate / cuGraphLaunch / cuGraphExecKernelNodeSetParams
/ cuGraphDestroy / cuGraphExecDestroy`. Placement (ILGPU `CudaStream` API vs ML-side helper) = Geordi's call.

## Prerequisites / risks (ask Geordi — see DevComms)
- All decode kernels must launch on ONE capturable stream (no multi-stream, no default stream).
- NO host sync during capture: disable `GraphExecutor.SyncIntervalNodes` drains during the capture step; flag
  any ILGPU-internal `cuCtxSynchronize` / readback. NOTE: the graph executor does shape readbacks (Shape→Gather→
  Concat) — for fixed-shape decode these are cached/warmed (`CacheShapeReadbacks`, `_readbackStable`); confirm no
  live readback remains in the steady-state decode step (a sync during capture aborts it).
- Buffer pointers stable across replays (recycled outputs + cached dp4a activation buffers — verify nothing
  re-allocates mid-decode).
- Correctness gate: graph-replayed decode tokens MUST match the non-graph decode (token-identical). Opt-in flag
  (e.g. `GGUF_DECODE_GRAPH` / `EnableDecodeGraph`), Example 06 opts in after verification.

## Method (do this first — Rule 4b)
CPU-profile decode with dotnet-trace BEFORE refactoring the executor, to confirm the per-node hot path (Rent /
managed alloc / marshalling / dict) and whether GC contributes to the wall variance. Don't touch the critical
executor blind.

## Secondary / complementary
- Kernel FUSION via `Graph/GraphOptimizer.cs` (already has `FuseAttention` + `EliminateDeadNodes`) to cut the
  703-node count — linear dispatch reduction, stacks with graphs, and is pure ML-side (no driver API).
- These are independent: fusion reduces node count; graphs eliminate per-node launch cost. Both help; graphs is
  the bigger win.

## Key files
- `Graph/GraphExecutor.cs` — `RunDecodeStepAsync`; cached-template decode run (~L888, `EnsureRunTemplates`,
  `_baseRefCounts`/`_cleanConstants`); per-node loop; `SyncIntervalNodes=64` (L142); `AllocatedBufferCount` (L248).
- `Pipelines/GgufGenerator.cs` — production decode driver (`CacheShapeReadbacks=true`).
- `Kernels/FusedDequantMatMul.cs` — dp4a GEMVs (`GemvDp4aQ4_KImpl`/`GemvDp4aQ6_KImpl`/`QuantizeActQ8_1Impl`/`Dp4a`).
- `Examples/06.OllamaServer.Console/Program.cs` — server opt-ins (all speed flags).
