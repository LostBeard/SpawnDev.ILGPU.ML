using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU padding kernel. Supports constant, edge, and reflect modes.
/// Used by ONNX Pad operator and Conv padding.
/// </summary>
public class PadKernel
{
    private readonly Accelerator _accelerator;

    // params: [rank, mode, inShape[rank], pads[2*rank], outShape[rank], inStrides[rank], outStrides[rank]]
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, float>? _padKernel;
    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public PadKernel(Accelerator accelerator) => _accelerator = accelerator;

    private static void PadImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p,
        float constantValue)
    {
        int rank = p[0];
        int mode = p[1]; // 0=constant, 1=edge, 2=reflect
        // Layout: [rank, mode, inShape[rank], pads[2*rank], inStrides[rank], outStrides[rank]]

        int remaining = idx;
        int srcIdx = 0;
        // WORKAROUND: WGSL codegen bug #3 (SpawnDev.ILGPU PLANS.md) — using int instead of
        // bool and removing break to avoid incorrect PHI merge at loop exit on WebGPU.
        int outOfBounds = 0;

        for (int d = 0; d < rank; d++)
        {
            int outStride = p[2 + 4 * rank + d]; // outStrides offset
            int coord = remaining / outStride;
            remaining = remaining % outStride;

            int inDim = p[2 + d]; // inShape
            int padBefore = p[2 + rank + d]; // pads[d]
            int inStride = p[2 + 3 * rank + d]; // inStrides

            int srcCoord = coord - padBefore;

            if (srcCoord < 0 || srcCoord >= inDim)
            {
                if (mode == 0) // constant — flag and clamp to valid index
                {
                    outOfBounds = 1;
                    srcCoord = 0;
                }
                else if (mode == 1) // edge
                {
                    srcCoord = srcCoord < 0 ? 0 : inDim - 1;
                }
                else // reflect
                {
                    if (srcCoord < 0) srcCoord = -srcCoord;
                    if (srcCoord >= inDim) srcCoord = 2 * (inDim - 1) - srcCoord;
                }
            }

            srcIdx += srcCoord * inStride;
        }

        output[idx] = outOfBounds != 0 ? constantValue : input[srcIdx];
    }

    /// <summary>
    /// Pad a tensor. pads format: [pad_before_dim0, pad_before_dim1, ..., pad_after_dim0, pad_after_dim1, ...].
    /// mode: 0=constant, 1=edge, 2=reflect.
    /// </summary>
    public void Forward(ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int[] inputShape, int[] pads, int mode = 0, float constantValue = 0f)
    {
        EnsureLoaded();
        int rank = inputShape.Length;

        // Compute output shape and strides
        var outShape = new int[rank];
        for (int i = 0; i < rank; i++)
            outShape[i] = inputShape[i] + pads[i] + pads[rank + i];

        var inStrides = new int[rank];
        var outStrides = new int[rank];
        inStrides[rank - 1] = 1; outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
        {
            inStrides[i] = inStrides[i + 1] * inputShape[i + 1];
            outStrides[i] = outStrides[i + 1] * outShape[i + 1];
        }

        int totalOut = 1;
        for (int i = 0; i < rank; i++) totalOut *= outShape[i];

        // Pack params
        int paramsSize = 2 + 5 * rank;
        if (_paramsBuf == null || _paramsBuf.Length < paramsSize)
        {
            _paramsBuf?.Dispose();
            _paramsBuf = _accelerator.Allocate1D<int>(paramsSize);
        }
        var paramsData = new int[paramsSize];
        paramsData[0] = rank;
        paramsData[1] = mode;
        for (int i = 0; i < rank; i++) paramsData[2 + i] = inputShape[i];
        for (int i = 0; i < 2 * rank; i++) paramsData[2 + rank + i] = pads[i];
        for (int i = 0; i < rank; i++) paramsData[2 + 3 * rank + i] = inStrides[i];
        for (int i = 0; i < rank; i++) paramsData[2 + 4 * rank + i] = outStrides[i];
        _paramsBuf.CopyFromCPU(paramsData);

        _padKernel!(totalOut, input, output, _paramsBuf.View, constantValue);
    }

    private void EnsureLoaded()
    {
        _padKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, float>(PadImpl);
    }
}
