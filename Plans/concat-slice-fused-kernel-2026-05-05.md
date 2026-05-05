# Concat + Slice fused kernels — scope for next session

**Status:** designed, NOT shipped. End-of-session 2026-05-05 — too risky to land cross-backend without dedicated verification cycles.
**Author:** Data
**Date:** 2026-05-05

## Diagnosis recap (already captured)

DA3-Small Wasm `DA3Small_Inference_ProducesDepth` times out at 5min. After the MatMul kernel structural refactor (commits `b54f983` + `5362c7c`, ~10x first-compile win on the worst MatMul), the dominant cost shifted to **per-dispatch overhead in Concat and Slice operators within transformer attention RoPE blocks**.

`WasmTests.DA3Small_FirstNNodes_DiagnosticPerOpSync` with `BREAK_AT = 800` surfaced the top compile-heavy nodes — they are NOT compile-heavy, they are **dispatch-heavy**:

```
2271ms  703_Concat_/backbone/blocks.5/attn/rope_1/Concat_1
2258ms  734_Concat_/backbone/blocks.5/attn/rope/Concat_3
2223ms  699_Concat_/backbone/blocks.5/attn/rope/Concat_1
2151ms  713_Concat_/backbone/blocks.5/attn/rope_1/Concat_2
1962ms  735_Concat_/backbone/blocks.5/attn/rope_1/Concat_3
1775ms  708_Concat_/backbone/blocks.5/attn/rope/Concat_2
... (similar for blocks.4 and likely blocks.0-3 + 6-11)
1133ms  146_MatMul_/backbone/blocks.0/attn/qkv/MatMul    <- after MatMul fix
1104ms  528_Slice_/backbone/blocks.4/attn/rope_1/Slice_6
... 12+ Slice nodes 700-1100ms in blocks.4-5 alone
```

Reading `Operators/StructuredOperators.cs:908+` (ConcatOperator) and `:1393+` (SliceOperator) reveals the cause:

**ConcatOperator.Execute**: nested `for (n)` over inputs and `for (o)` over outer blocks calls `reg.ElementWise.Scale(srcView, dstView, copyLen, 1f)` PER (n, o) tuple. For 4-input Concat with 16 outer blocks = 64 dispatches per Concat node. On Wasm each kernel dispatch has worker-pool round-trip overhead (~30-50ms) regardless of payload size. 64 × 30ms = 1.9s — matches observed 2.2s.

**SliceOperator.SliceGPU** (line 1545+): RECURSIVE function — for each axis it iterates and recurses; on the last axis it dispatches `reg.ElementWise.Scale(...)` per contiguous run. Fan-out is similar.

**ScaleImpl is trivial** (`output[idx] = input[idx] * scalar`, line 140 of `ElementWiseKernels.cs`). Not a compile-time problem; pure dispatch overhead.

## Design

### Goal

Reduce Concat and Slice from many small Scale dispatches to ONE parallel kernel dispatch each. Expected ~60x win per affected node on Wasm; meaningful on every backend (CUDA/OpenCL also pay per-dispatch overhead, just lower per-dispatch cost).

### `SliceKernel`

One thread per output element. Each thread computes its multi-dim output coordinate from its linear index, maps to input coordinate via `start[d] + outCoord[d] * step[d]`, and reads.

Pack `starts`/`steps`/`outShape`/`inStrides` per axis into a single `ArrayView<int>` parameter (rank * 4 ints) to avoid passing variable-length arrays. Pass `rank` as scalar.

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static void SliceKernelImpl(
    Index1D idx,
    ArrayView1D<float, Stride1D.Dense> input,
    ArrayView1D<float, Stride1D.Dense> output,
    ArrayView1D<int, Stride1D.Dense> packedParams,  // [starts(rank), steps(rank), outShape(rank), inStrides(rank)]
    int rank)
{
    int inIdx = 0;
    int remaining = idx;
    int startsBase = 0;
    int stepsBase = rank;
    int outShapeBase = 2 * rank;
    int inStridesBase = 3 * rank;

    for (int d = rank - 1; d >= 0; d--)
    {
        int outShapeD = packedParams[outShapeBase + d];
        int outCoordD = remaining % outShapeD;
        remaining /= outShapeD;
        int inCoordD = packedParams[startsBase + d] + outCoordD * packedParams[stepsBase + d];
        inIdx += inCoordD * packedParams[inStridesBase + d];
    }
    output[idx] = input[inIdx];
}
```

Public API: `SliceKernel.Slice(input, output, starts[], steps[], outShape[], inStrides[], rank, totalOutputElements)` — uploads packed params to a small reusable GPU buffer then dispatches once.

### `ConcatKernel`

For each output element idx, compute its concat-axis coordinate. Find which input contains that coordinate via cumulative offset table. Read from that input.

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static void ConcatKernelImpl(
    Index1D idx,
    ArrayView1D<float, Stride1D.Dense> output,
    ArrayView1D<int, Stride1D.Dense> inputOffsetsAndSizes, // [N+1 cumulative offsets][N input lengths]
    int outer, int totalConcatDim, int inner,
    ArrayView1D<float, Stride1D.Dense> input0,
    ArrayView1D<float, Stride1D.Dense> input1,
    ArrayView1D<float, Stride1D.Dense> input2,
    ArrayView1D<float, Stride1D.Dense> input3,
    int numInputs)  // up to 4 inputs in this signature; >4 needs more variants or struct-of-views
{
    // Compute outer/concat/inner coordinates from linear idx
    int o = idx / (totalConcatDim * inner);
    int rem = idx % (totalConcatDim * inner);
    int c = rem / inner;
    int i = rem % inner;

    // Find which input owns concat-axis coord c via cumulative offsets
    int srcInput = 0;
    int cInInput = c;
    for (int n = 1; n < numInputs; n++)
    {
        if (c >= inputOffsetsAndSizes[n])
        {
            srcInput = n;
            cInInput = c - inputOffsetsAndSizes[n];
        }
    }

    int srcIdx = o * inputOffsetsAndSizes[numInputs + srcInput] * inner + cInInput * inner + i;

    // Branch on srcInput - WGSL likes this less than indexed-into-array, but no struct-of-views support yet
    float v;
    if (srcInput == 0) v = input0[srcIdx];
    else if (srcInput == 1) v = input1[srcIdx];
    else if (srcInput == 2) v = input2[srcIdx];
    else v = input3[srcIdx];
    output[idx] = v;
}
```

**Concat with N>4 inputs:** keep the current per-pair Scale fallback for the rare case (most ML Concat is 2-4 inputs). Or define multi-arity variants.

### Why `[AggressiveInlining]` here, not `[NoInlining]`

These kernel impls are MEDIUM-sized helpers that the JIT may keep as separate fns by default. We want them inlined into the kernel entry point so the codegen produces one tight WGSL/Wasm body — fewer fn boundaries, the optimizer sees the whole computation. `AggressiveInlining` is the safe lock per `feedback_methodimpl_inlining_directives.md` (works on every backend including WebGPU).

## Test plan (mandatory before shipping)

Per Rule 5 + Rule 1:

1. Unit test for `SliceKernel`: every axis as slice axis (0..rank-1), positive + negative starts, step != 1, broadcast-with-Slice combos. Compare GPU vs CPU oracle for known small inputs.

2. Unit test for `ConcatKernel`: 2-input + 4-input cases, axis 0..rank-1, ragged input lengths, axis with size 1 inputs (broadcast pattern).

3. Cross-backend regression: full StyleMosaic 6/6 test must stay GREEN (StyleMosaic doesn't use Slice/Concat in critical path but full path must not regress).

4. DistilBERT WebGPU regression: this model has Slice/Concat in attention path. Must stay GREEN (28s baseline).

5. DA3-Small Wasm `DA3Small_FirstNNodes_DiagnosticPerOpSync` BREAK_AT = 800: re-run, confirm rope-block Concat/Slice nodes drop from 700-2200ms to <100ms each.

6. DA3-Small Wasm full inference: re-run `DA3Small_Inference_ProducesDepth` — target completion in <120s (was 5-min timeout).

## Risks

- Multi-input Concat signature complexity. WGSL doesn't have variadic. The 4-input variant covers most cases; >4 falls back to existing per-pair Scale path — keep both code paths, dispatch by count.
- Slice with rank > 6 or unusual stride patterns might hit unhandled cases. Existing CPU path stays as fallback for `outCount <= 65536`.
- Negative steps. The spec allows `step < 0` (reverse iteration). Existing `SliceGPU` handles `step == 1` only via copy and falls back to per-element Scale for `step != 1`. The new SliceKernel handles arbitrary positive step naturally; negative step needs explicit handling.
- Pre-existing CPU path for small tensors (`outCount <= 65536`). Keep — for small inputs the GPU dispatch overhead might still win, but only if `TryGetInputValues` returns null (data is GPU-resident). Don't disturb the CPU path.

## Files touched (estimated)

- `SpawnDev.ILGPU.ML/Kernels/SliceKernel.cs` — NEW, ~120 lines
- `SpawnDev.ILGPU.ML/Kernels/ConcatKernel.cs` — NEW, ~150 lines
- `SpawnDev.ILGPU.ML/Operators/OperatorRegistry.cs` — register new kernels
- `SpawnDev.ILGPU.ML/Operators/StructuredOperators.cs` — `ConcatOperator.Execute` and `SliceOperator.Execute` route to new kernels for the GPU-resident-data path; keep CPU path for small tensors
- New unit tests in the test harness

## What NOT to do this session (closed by reasoning)

- Don't ship without unit tests covering all axis/step/broadcast combos. Slice has many edge cases the existing operator handles; the new kernel must too.
- Don't drop the existing CPU + per-pair Scale fallbacks. They're correct; keep them as safety nets gated on small-tensor + `TryGetInputValues != null`.
- Don't try to fold Slice + Concat into a single kernel for the rope pattern. Could be a perf win but couples concerns; ship the standalone kernels first.

## Pickup checklist for next session

- [ ] Re-read this scope file
- [ ] Verify current ML state: `git log --oneline -5`, expect head on master
- [ ] Run baseline `DA3Small_FirstNNodes_DiagnosticPerOpSync` BREAK_AT=800 to confirm Concat/Slice timings haven't changed (infra moves under us)
- [ ] Implement `SliceKernel.cs` first (single input, simpler signature)
- [ ] Unit tests + cross-backend smoke
- [ ] Wire into `SliceOperator.Execute`; behind a feature flag if needed
- [ ] Re-run BREAK_AT=800; expect rope-block Slice nodes to drop to <100ms
- [ ] Repeat for `ConcatKernel.cs`
- [ ] Final: full DA3-Small Wasm inference test — target <2min total, <500ms per Concat/Slice node

🖖 Data
