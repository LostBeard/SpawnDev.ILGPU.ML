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

---

## 2026-06-22 — MEASURED findings + refined design (Tuvok session)

### Profiling result (Rule 4b — done; dotnet-trace, SampleProfiler @1ms, real qwen decode, 160 tok)
Inside `GraphExecutor.RunAsync`, time classified by stack:
- **Unmanaged non-sync (`cuLaunchKernel` + arg marshalling): 72.5%** ← dominant
- Managed node-walk (Rent / dict / LINQ `.Select().ToArray()` / Execute): **18.0%**
- Sync-wait (SynchronizeAsync drains): 9.5%

**This REFINES the original assumption.** The residual is dominated by per-launch DRIVER cost, NOT the managed
graph-walker (Rent/dict/LINQ is only 18%). So CUDA graphs (collapse ~703 `cuLaunchKernel` → 1 `cuGraphLaunch`)
is the correct primary lever; the managed walker is the secondary one. Caveat: SampleProfiler collapses all
CUDA-driver time into one unmanaged bucket — it cleanly excludes the sync drains (measured 9.5% separately) and
the managed walker (18%), which is enough to confirm the lever. Steady-state wall ~33ms = ~7ms GPU + ~26ms CPU.

### The ILGPU graph API is SHIPPED (Geordi, master `71438a4`) and 4070-verified (6.6× dispatch win)
`CudaStream.BeginCapture/EndCapture/CaptureStatus/SupportsGraphCapture`, `CudaGraph.Instantiate()`,
`CudaGraphExec.Launch/Upload`. Geordi's Part-3 proof validates the EXACT per-token mechanism: a captured kernel
reads its per-step value from a **stable-pointer device buffer**, host mutates the buffer's CONTENTS between
**synced** replays → bit-exact. The decode loop syncs per token anyway (to sample), so per-token updates are free.
**Footgun (Geordi):** un-synced `CopyFromCPU(stream, managedArray)` between launches races (staged pinned host
buffer reused) → wrong. Safe = synced-per-step (natural) OR a dedicated pinned staging buffer.

### Routing decision (ASK to Geordi, pending)
`*StreamKernel` launchers resolve `accelerator.DefaultStream` **at launch time** (`KernelLoaders.cs:401,425,…`),
which wraps the NULL stream (uncapturable). Two routes: (a) Geordi's explicit-stream launchers = change ~295
call sites (miss-one = capture abort); (b) **swap `DefaultStream` to a created non-blocking `CudaStream`** =
reroutes all sites with zero churn, can't-miss, covers prefill. Asked Geordi for a scoped `acc.WithDefaultStream`
(needs ILGPU — `DefaultStream` is `protected set`). With (b), `cuGraphExecKernelNodeSetParams` is NOT needed for
the primary path (device-resident params instead). DevComms `tuvok-to-geordi-graph-profile-routing-package-2026-06-22`.

### Per-step varying state (mapped from the code)
1. **`input_ids` device buffer** — `GgufGenerator` reallocs `inBuf` via `Allocate1D(idf)` EVERY step → pointer
   differs each token. The captured embedding-Gather bakes step-0's pointer → replay reads wrong/freed memory.
   **FIX: a STABLE reused single-token decode input buffer** (write the new token id in per step). Prereq + tiny
   perf win (drops a per-step alloc/free). Decode stepIds is always length 1; prefill (step 0, longer) is not captured.
2. **`DecodePastLen`** (grows +1/token) feeds: RoPE `kv_offset`, KV-write offset, attention `seqKV`/`kvOffset`.
   - **Attention grid is FIXED** (`Index1D` = nHeads×seqQ, seqQ=1); `seqKV` is a kernel **loop bound**, not a grid
     dim. So NO grid update needed across steps. ✅
   - FusedAttention **already passes seqKV/kvOffset/window via a DEVICE buffer** (`RentParamsSlot`→`CopyFromCPU`,
     `FusedAttentionKernel.cs:67`). BUT it **cycles a 64-slot RING** (new device pointer each call) → a captured
     node bakes the capture-step slot, and replay (no host code) never refreshes it. **FIX: a STABLE params slot
     per decode attention node** (keyed by layer/node identity, allocated once), host writes only the 2 dynamic
     ints (`seqKV`=pastLen+1, `kvOffset`=pastLen) per token before `exec.Launch`. The ring's anti-race purpose is
     preserved because each node gets its OWN stable slot (no cross-layer sharing within a forward).
   - **RoPE `kv_offset`** and **KV-write offset**: confirm whether host-baked (kernel scalar) or device-resident;
     if host-baked, make device-resident the same way (stable slot, refresh per token).

### Build order (Stage gating)
- **Stage 0 — mechanics: DONE** (Geordi's 4070 proof, 6.6×, bit-exact).
- **Stage 1 — same-state replay de-risk on the REAL model:** consume the ILGPU local pkg; stable input buffer;
  warm one decode step; `WithDefaultStream(capStream)` + BeginCapture → run one forward (drains disabled,
  readbacks warm, pool warm = no alloc/sync/readback in capture) → EndCapture → Instantiate; replay once, sync,
  compare logits to the non-graph forward at the SAME state (must be identical); measure residual collapse.
  Does NOT need per-step updates — proves capture+replay works on the real decode forward + the win is real.
- **Stage 2 — per-step inputs:** stable input buffer refresh + stable params slots (seqKV/kvOffset/RoPE/KV-write
  device-resident) so successive DIFFERENT tokens are correct; verify token-identical vs non-graph decode over a
  full generation; measure decode ms/tok. Gate opt-in (`GGUF_DECODE_GRAPH`/`EnableDecodeGraph`), Example 06 opts in.
- **Blocker:** ML consumes ILGPU as NuGet `4.15.1` (no graph API) — need Geordi's `-local.N` pkg first.
