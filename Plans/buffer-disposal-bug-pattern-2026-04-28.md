# Buffer Disposal Bug Pattern in Operators (2026-04-28, Data)

## Symptom

`WebGPUTests.Pipeline_Diffusion_DDPM_ProducesImage` fails on WebGPU in isolation (not cascade) with:

```
[WebGPU] 1 GPU error(s) during dispatch:
[Buffer (unlabeled)] used in submit while destroyed.
- While calling [Queue].Submit([[CommandBuffer]])
```

Stack trace lands in `GraphExecutor.RunAsync → _accelerator.SynchronizeAsync()`. The error fires when the WebGPU command queue submits a batched command encoder that references a buffer the .NET side already disposed.

`WasmTests.Pipeline_Diffusion_DDPM_ProducesImage` fails differently (`divide by zero` at Wasm dispatch 19) - separate kernel issue, tracked separately.

WebGL/OpenCL/CUDA/CPU all pass the Diffusion test in isolation (5-13s each).

## Root cause (verified)

`SpawnDev.ILGPU.WebGPU` batches kernel dispatches into a command encoder. The encoder holds buffer handles until `Synchronize()` flushes via `queue.submit`. If a buffer is disposed BETWEEN the dispatch enqueue and the next sync, the submit fires "used while destroyed".

This is the same class of bug already documented in `SpawnDev.ILGPU/SpawnDev.ILGPU/WebGPU/CLAUDE.md`:

> ### WebGPU Command Batching - Never Dispose Buffers Before Flush
> WebGPU batches kernel dispatches into a command encoder. The GPU does NOT execute dispatches immediately - they are submitted together when `Synchronize()`/`Flush()` is called. **Any GPU buffer referenced by a pending dispatch must NOT be disposed until after the flush.**

Commit `49d1ecb` (2026-04-04) fixed this for one variant (`Scale(tempBuf -> output)` -> `CopyFromCPU(result)`). The CURRENT bug is the OTHER variant: operators that legitimately need a GPU temp buffer for aliasing-safe intermediate computation, allocated with `using var` and disposed when the operator's `Execute` method returns.

## The pattern (occurs in many operators)

```csharp
public void Execute(OnnxOpContext ctx)
{
    int total = ctx.Inputs[0].ElementCount;
    using var tempBuf = reg.Accelerator.Allocate1D<float>(total);   // BUG
    reg.ElementWise.Log(ctx.Outputs[0].Data, tempBuf.View, total);  // dispatch enqueued
    reg.ElementWise.Scale(tempBuf.View, ctx.Outputs[0].Data, total, 1f); // dispatch enqueued
    // method exits, `using` disposes tempBuf
    // ... 60-something more nodes execute ...
    // GraphExecutor.SynchronizeAsync fires → submit → "Buffer used while destroyed"
}
```

This works in operator-coverage tests because `AssertCloseGpu` runs `await accelerator.SynchronizeAsync()` immediately, flushing the encoder before the temp buffer goes out of scope. In full graph execution, periodic 64-node syncs are too late - many ops have already returned.

## Confirmed sites (search: `using var.*Allocate1D` in `Operators/` `Kernels/`)

- `Operators/ElementWiseOperators.cs:1243` - LogSoftmax tempBuf
- `Operators/ElementWiseOperators.cs:1277` - Sum tempBuf
- `Operators/ElementWiseOperators.cs:1297` - Mean tempBuf
- `Operators/ElementWiseOperators.cs:1324` - ArgMin negBuf
- `Operators/EinsumOperator.cs:145` - readBuf
- `Operators/RemainingOperators.cs:608-615` - ConvInteger xBufMem/wBufMem/zeroBias/outBufMem
- `Operators/RemainingOperators.cs:742-751` - QLinearConv xBufMem/wBufMem/outBufMem/zeroBias
- `Operators/ShapeOperators.cs:634, 693, 694` - Trilu temp / DynamicQuantize maxBuf/minBuf
- `Operators/StructuredOperators.cs:1641, 1667, 1694, 1743` - LpNormalization absBuf / GlobalLpPool sqBuf / LpPool sqBuf / Multinomial expBuf
- `Kernels/FWHTKernel.cs:94` - padBuf
- `Kernels/QuantizedKVCache.cs:275-279, 304, 369-370` - many TurboQuant temp buffers

## Fix patterns (in order of preference)

### Pattern A: Persistent field with deferred disposal (matches BroadcastBinaryOpND)

`ElementWiseKernels.cs:489-493` already does this for stride buffers:

```csharp
private MemoryBuffer1D<int, Stride1D.Dense>? _lastStridesBuf;
private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _oldStridesBufs = new();

public void BroadcastBinaryOpND(...)
{
    if (_lastStridesBuf != null) _oldStridesBufs.Add(_lastStridesBuf);
    _lastStridesBuf = _accelerator.Allocate1D<int>(paramsSize);
    // ...
}

public void Dispose()
{
    _lastStridesBuf?.Dispose();
    foreach (var buf in _oldStridesBufs) buf.Dispose();
    _oldStridesBufs.Clear();
}
```

For operators, this means giving the operator class persistent fields for its temp buffers. Buffers accumulate during inference but get cleaned up at session disposal.

### Pattern B: Use BufferPool with a reserved name

Some operators already do this (e.g., `MatMulIntegerOperator.cs:641` uses `ctx.Pool.Rent(aShape, "_mmi_a")`). The pool's `_namedBuffers` dict reuses the buffer for subsequent calls with the same name - effectively persistent storage tied to InferenceSession lifetime.

Apply to LogSoftmax, Sum, Mean, ArgMin etc:

```csharp
public void Execute(OnnxOpContext ctx)
{
    int total = ctx.Inputs[0].ElementCount;
    var tempTensor = ctx.Pool.Rent(new[] { total }, "_logsoftmax_temp");
    reg.ElementWise.Log(ctx.Outputs[0].Data, tempTensor.Data, total);
    reg.ElementWise.Scale(tempTensor.Data, ctx.Outputs[0].Data, total, 1f);
    // No dispose - pool owns lifetime, persists with InferenceSession
}
```

### Pattern C: Synchronize before dispose

Simplest but burns a sync per operator dispatch. Not suitable for hot ops.

## Why operator-coverage tests don't catch this

Each operator-coverage test:
1. Allocates input buffers
2. Calls the operator's Execute method
3. Calls `AssertCloseGpu(...)` which calls `await accelerator.SynchronizeAsync()` BEFORE the test scope exits

So the temp buffer is alive at sync time. The bug only fires when many operators chain together without intervening syncs. Need a regression test that runs at least two operators sequentially without any sync between them, then syncs.

## Proposed regression test

Add to `MLTestBase.AllOperatorTests.cs`:

```csharp
[TestMethod]
public async Task ChainedOps_LogSoftmaxThenScale_NoDisposalRace() => await RunTest(async accelerator =>
{
    var input = new float[] { 1, 2, 3, 4 };
    using var inBuf = accelerator.Allocate1D(input);
    using var outBuf = accelerator.Allocate1D<float>(4);

    // Simulate a graph step: chain a LogSoftmax through to subsequent ops
    // without an intervening sync. The current code allocates a using-var
    // temp inside LogSoftmax; if it's disposed before the encoder flushes,
    // WebGPU fires "Buffer used while destroyed" at the next sync.
    var reg = new OperatorRegistry(accelerator);  // operator-level test rig
    // ... invoke LogSoftmax via registry, then chain a follow-up dispatch ...
    // ... only sync once at the end ...

    await AssertCloseGpu(accelerator, outBuf.View, expected, 1e-5f, "Chained:");
});
```

A real chained-graph repro requires graph execution machinery; an alternative is to invoke an operator directly, NOT call sync, then dispatch a separate kernel that also reads from the temp's lifetime, then sync once. Practical implementation: compile a 2-node graph and run it.

## Plan

1. Land the BroadcastBinaryOpND-style persistent-field fix in the smallest operator (ArgMin) as a reference implementation.
2. Verify the WebGPU Diffusion test passes with that one fix in place (probably won't - need to sweep all sites).
3. Sweep the rest of the operator/kernel sites listed above.
4. Add a regression test that exercises chained operators with a single end-of-graph sync.
5. Confirm WebGPU Pipeline_BackgroundRemoval and Pipeline_SemanticSearch behavior - they may share the same root cause (timeout because the post-error state hangs the Playwright UI sync).

## Smoking gun: per-op SynchronizeAsync makes Pipeline_Diffusion pass (2026-04-28)

Confirmed the bug class with a temporary diagnostic: `await _accelerator.SynchronizeAsync()` injected after every `node.Operator.Execute(ctx)` in `GraphExecutor.RunAsync`'s main loop.

| Configuration | Pipeline_Diffusion_DDPM (WebGPU) | Duration |
|---------------|----------------------------------|----------|
| Default batched sync (every 64 nodes) | FAIL (`Buffer used in submit while destroyed`) | 7s |
| Per-op SynchronizeAsync | **PASS** | 15s |

Same model bytes, same kernels, same operators - the only difference is when buffers get a chance to flush before they go out of scope. This isolates the bug class definitively to "an operator's local using-var GPU buffer is `.Dispose()`d before the WebGPU command encoder it was queued into gets `queue.submit`ted."

The diagnostic was reverted (commit pending) once confirmed. Per-op sync is NOT a viable production fix because it amplifies dispatch latency by a constant per-op cost (~150ms on this rig); a 2620-node GPT-2 graph would add 6+ minutes. The right fix is operator-level deferred disposal as already documented above (Pattern A / B).

### What this means for the operator audit

The using-var sites I identified are not the COMPLETE list - none of them are operators DDPM uses (Conv, GroupNorm, SiLU, Concat, Add, Resize). The bug must come from one or more of:

- A using-var I missed in a kernel called by those operators
- A buffer-disposal pattern that doesn't follow the literal `using var Allocate1D` form (e.g., `var x = Allocate(); ... x.Dispose();` written long-form)
- `_pool.Return(...)` returning a buffer to the bucket where the next Rent reuses it for a different tensor while the old tensor's dispatches are still in flight (this would be a BufferPool design issue, not an operator-local one)
- An ILGPU.WebGPU internal that allocates a temp during dispatch and disposes it before submit

Next step is to identify the exact operator triggering the bug. Two approaches:
(A) Bisect: add a static `BreakAtNode` to GraphExecutor; binary search to find the smallest N where the test still fails when stopped at node N.
(B) Instrument: enrich GraphExecutor's per-node logging to flush via Console.Error.WriteLine (PMT captures) and capture stage timings + buffer allocation/disposal events around each op.

Approach (A) is more pragmatic with limited diagnostic infrastructure - identifies the specific failing node without requiring console-message-routing changes.

## What this is NOT

NOT the IsInf bug (closed by `SpawnDev.ILGPU 4.9.2-rc.26` upstream + consumer bump in commit `1be5a2e`). NOT the Wasm divide-by-zero. NOT the DA3Small Playwright timeouts. Each of those is a separate item in `Plans/v4.0.0-checklist.md`.

## Side-finding: AssertCloseGpu uses `Atomic.Max` (WebGL-incompatible) - 2026-04-28

While verifying the rc.26 IsInf fix, WebGL.AllOps_IsInf failed at WGSL/GLSL codegen with `Atomic.Max is not supported on the WebGL backend`. Tracing the throw site:

`SpawnDev.ILGPU.ML/ElementWiseKernels.cs:1799-1800` (in `CompareReduceImpl`, the kernel called by `CompareOnGpuAsync` and through it `AssertCloseGpu`):

```csharp
Atomic.Add(ref results[0], absDiff);
Atomic.Max(ref results[1], absDiff);
```

WebGL fundamentally cannot implement `Atomic.Max` (per `SpawnDev.ILGPU/WebGL/CLAUDE.md`: "No shared memory, atomics, or barriers - fundamentally limited by WebGL 2.0 / GLSL ES 3.0"). rc.10 added `UnsupportedKernelFeatureException` typed throws at the codegen site so this now fails at compile time with a clean message instead of producing wrong output silently.

Implications:
- **Every ML test using `AssertCloseGpu` (~195 sites across 20 unit-test files) is fundamentally broken on WebGL.** The test fails at AssertCloseGpu's reduce-kernel compile step, not at the operator under test. Pre-existing on rc.25; not a rc.26 regression. Confirmed on both rc.25 and rc.26 with identical error.
- All other backends (WebGPU/Wasm/OpenCL/CUDA/CPU) handle `Atomic.Max` natively and pass.

Proposed fix paths (in order of preference):

1. **Multi-pass tree reduction** (no atomics). One kernel writes per-element absDiff; subsequent kernels reduce 2-to-1 in O(log N) dispatches. WebGL-compatible (Transform Feedback friendly). The right long-term fix.
2. **CPU-readback fallback for WebGL only** (`if (accelerator.AcceleratorType == WebGL) { CopyToHostAsync + CPU reduce } else { existing GPU path }`). Gates the cleanest performance regression to one backend.
3. **`AcceleratorRequirements.RequiresAtomics = true` on the ML test class.** Skips AssertCloseGpu-dependent tests on WebGL with a clean reason. Doesn't fix the underlying problem but stops the noise.

Logged here, not fixed today.
