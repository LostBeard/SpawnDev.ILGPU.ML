using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// General N-dimensional transpose (permutation of axes).
/// Maps ONNX Transpose op.
///
/// For common 2D/3D cases, consider using ElementWiseKernels.TransposeLastTwo
/// which is more efficient (no params buffer needed).
///
/// This kernel handles arbitrary permutations up to 6 dimensions via a params
/// buffer that encodes the shape and permutation.
/// </summary>
public class TransposeKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // params: [rank, shape[0..rank-1], perm[0..rank-1], inStrides[0..rank-1], outStrides[0..rank-1]]
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _transposeKernel;

    // Per-call fresh paramsBuf - eliminates the shared-state race under async dispatch
    // on Wasm where a queued dispatch could read paramsBuf.SharedBuffer AFTER a subsequent
    // CopyFromCPU overwrites it (root cause of YOLOv8 Wasm Softmax mean error 1.85, per
    // Geordi's Tests23_HostWriteVsQueuedDispatchRace 2026-05-04). Same pattern as PadKernel,
    // Conv2DKernel, PoolingKernels, NormalizationKernels. Buffers stay alive in
    // `_allParamsBufs` until Dispose() (typical InferenceSession lifetime). Per-call cost
    // is small (paramsSize ints, e.g. rank=4 -> 17 ints = 68 bytes).
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _allParamsBufs = new();

    public TransposeKernel(Accelerator accelerator) => _accelerator = accelerator;

    public void Dispose()
    {
        foreach (var buf in _allParamsBufs) buf.Dispose();
        _allParamsBufs.Clear();
    }

    /// <summary>
    /// General transpose: for each output element, compute its source index
    /// by reversing the permutation through the strides.
    /// </summary>
    private static void TransposeImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int rank = p[0];
        // p layout: [rank, inShape[rank], perm[rank], inStrides[rank], outStrides[rank]]
        // offsets:   0      1              1+rank       1+2*rank         1+3*rank

        // Decompose output linear index into output coordinates
        int remaining = idx;
        int srcIdx = 0;
        for (int d = 0; d < rank; d++)
        {
            int outStride = p[1 + 3 * rank + d];
            int coord = remaining / outStride;
            remaining = remaining % outStride;

            // This output dim d corresponds to input dim perm[d]
            int srcDim = p[1 + rank + d];
            int inStride = p[1 + 2 * rank + srcDim];
            srcIdx += coord * inStride;
        }

        output[idx] = input[srcIdx];
    }

    /// <summary>
    /// Transpose input with given shape by the specified permutation.
    /// perm[i] = which input dimension maps to output dimension i.
    /// E.g., perm=[2,0,1] on shape [A,B,C] → output shape [C,A,B].
    /// </summary>
    public void Transpose(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] inputShape, int[] perm)
    {
        EnsureLoaded();
        int rank = inputShape.Length;
        if (perm.Length != rank) throw new ArgumentException("Perm length must match rank");

        // Compute strides
        var inStrides = new int[rank];
        var outShape = new int[rank];
        var outStrides = new int[rank];

        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
            inStrides[i] = inStrides[i + 1] * inputShape[i + 1];

        for (int i = 0; i < rank; i++)
            outShape[i] = inputShape[perm[i]];

        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
            outStrides[i] = outStrides[i + 1] * outShape[i + 1];

        int totalElements = 1;
        for (int i = 0; i < rank; i++) totalElements *= inputShape[i];

        // Pack params: [rank, inShape..., perm..., inStrides..., outStrides...]
        // Fresh allocation per call - avoids the cross-call SharedBuffer race on Wasm.
        int paramsSize = 1 + 4 * rank;
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        for (int i = 0; i < rank; i++) paramsData[1 + i] = inputShape[i];
        for (int i = 0; i < rank; i++) paramsData[1 + rank + i] = perm[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 2 * rank + i] = inStrides[i];
        for (int i = 0; i < rank; i++) paramsData[1 + 3 * rank + i] = outStrides[i];
        var paramsBuf = _accelerator.Allocate1D<int>(paramsSize);
        paramsBuf.View.CopyFromCPU(paramsData);
        _allParamsBufs.Add(paramsBuf);

        _transposeKernel!(totalElements, input, output, paramsBuf.View);
    }

    private void EnsureLoaded()
    {
        _transposeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(TransposeImpl);
    }
}
