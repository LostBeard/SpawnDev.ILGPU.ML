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

## What this is NOT

NOT the IsInf bug. NOT the Wasm divide-by-zero. NOT the DA3Small Playwright timeouts. Each of those is a separate item in `Plans/v4.0.0-checklist.md`.
