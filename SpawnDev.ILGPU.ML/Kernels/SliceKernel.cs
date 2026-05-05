using ILGPU;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// ONNX Slice operation as a single fused parallel kernel.
///
/// Replaces the recursive per-contiguous-run dispatch pattern in
/// SliceOperator.SliceGPU which on Wasm pays worker-pool round-trip overhead
/// per dispatch and accumulates to seconds per Slice node in transformer
/// attention RoPE blocks (BREAK_AT=800 diagnostic 2026-05-05 showed 12+ Slice
/// nodes at 700-1100ms each on Wasm). One kernel dispatch instead of N runs ~60x
/// per affected node when the per-run overhead dominates.
///
/// Per-thread: maps a linear output index to its multi-dim coordinate, then
/// to an input index via starts[d] + outCoord[d] * steps[d], reads, writes.
/// Supports arbitrary positive steps and arbitrary rank up to MAX_RANK.
/// </summary>
public class SliceKernel : IDisposable
{
    /// <summary>
    /// Maximum supported tensor rank. ONNX models in practice rarely exceed
    /// rank 6 (NCHWZW etc.). 8 leaves headroom without bloating the packed
    /// params buffer.
    /// </summary>
    public const int MAX_RANK = 8;

    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _sliceKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public SliceKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Per-thread Slice: read input at the indexed source location and write to
    /// the output at the linear thread index. AggressiveInlining locks the JIT
    /// into the inlined codegen path which works on every backend (per
    /// feedback_methodimpl_inlining_directives.md). The packed params layout
    /// is [starts(rank), steps(rank), outShape(rank), inStrides(rank)] so this
    /// kernel handles arbitrary rank with a single signature.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SliceImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> packedParams,  // 4 * rank ints
        int rank)
    {
        // packedParams layout per axis d in [0, rank):
        //   starts:    packedParams[0 * rank + d]
        //   steps:     packedParams[1 * rank + d]
        //   outShape:  packedParams[2 * rank + d]
        //   inStrides: packedParams[3 * rank + d]
        int startsBase = 0;
        int stepsBase = rank;
        int outShapeBase = 2 * rank;
        int inStridesBase = 3 * rank;

        // Convert linear output idx -> per-axis coordinate via row-major decomposition,
        // then accumulate the input index via starts + outCoord * step.
        // Decompose from innermost axis (rank-1) outward; remaining contains the still-
        // -to-be-decomposed prefix coordinate.
        int inIdx = 0;
        int remaining = idx;
        for (int d = rank - 1; d >= 0; d--)
        {
            int outShapeD = packedParams[outShapeBase + d];
            int outCoordD = remaining % outShapeD;
            remaining /= outShapeD;
            int inCoordD = packedParams[startsBase + d] + outCoordD * packedParams[stepsBase + d];
            inIdx += inCoordD * packedParams[inStridesBase + d];
        }

        // Defensive bound: malformed params shouldn't crash the kernel.
        // Out-of-range reads return 0 instead of OOB.
        output[idx] = (inIdx >= 0 && (long)inIdx < input.Length) ? input[inIdx] : 0f;
    }

    /// <summary>
    /// Run a fused Slice. Caller supplies pre-resolved per-axis params (starts,
    /// steps, outShape, inStrides) - the same values SliceOperator.Execute
    /// already computes before calling its CPU/GPU paths.
    ///
    /// All arrays must be exactly `rank` long. `rank` must be in [1, MAX_RANK].
    /// `totalOutput` must equal the product of `outShape`.
    ///
    /// Uploads the packed params to a reusable GPU buffer, dispatches one parallel
    /// kernel covering totalOutput elements, returns. No CPU readback.
    /// </summary>
    public void Slice(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] starts, int[] steps, int[] outShape, int[] inStrides,
        int rank, int totalOutput)
    {
        if (rank < 1 || rank > MAX_RANK)
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"rank must be in [1, {MAX_RANK}]");
        if (starts.Length != rank || steps.Length != rank || outShape.Length != rank || inStrides.Length != rank)
            throw new ArgumentException("starts/steps/outShape/inStrides must each be length == rank");
        if (totalOutput <= 0) return;

        EnsureLoaded();

        // Pack: [starts(rank), steps(rank), outShape(rank), inStrides(rank)]
        var packed = new int[4 * rank];
        Array.Copy(starts, 0, packed, 0 * rank, rank);
        Array.Copy(steps, 0, packed, 1 * rank, rank);
        Array.Copy(outShape, 0, packed, 2 * rank, rank);
        Array.Copy(inStrides, 0, packed, 3 * rank, rank);

        // Reusable params buffer - reallocate only on rank growth (rare).
        if (_paramsBuf == null || _paramsBuf.Length < packed.Length)
        {
            _paramsBuf?.Dispose();
            _paramsBuf = _accelerator.Allocate1D<int>(packed.Length);
        }
        _paramsBuf.View.SubView(0, packed.Length).CopyFromCPU(packed);

        _sliceKernel!(totalOutput, input, output, _paramsBuf.View.SubView(0, packed.Length), rank);
    }

    private void EnsureLoaded()
    {
        _sliceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(SliceImpl);
    }

    public void Dispose()
    {
        _paramsBuf?.Dispose();
        _paramsBuf = null;
    }
}
