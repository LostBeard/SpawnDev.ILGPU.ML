using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Fast Walsh-Hadamard Transform (FWHT) GPU kernel.
/// Core component of TurboQuant — used for data-oblivious KV cache quantization.
///
/// The FWHT maps vectors into the Hadamard basis where they can be quantized
/// with minimal information loss. The transform is its own inverse (up to scaling),
/// making it ideal for fast encode/decode in the attention inner loop.
///
/// Complexity: O(d log d) per vector, where d is the head dimension.
/// Memory: In-place — no additional buffers needed.
/// </summary>
public class FWHTKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int>? _fwhtKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int>? _fwhtBatchKernel;

    public FWHTKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// In-place FWHT on a single vector of length d (must be power of 2).
    /// Normalized: output = H_d @ input / sqrt(d).
    /// </summary>
    public void Forward(ArrayView1D<float, Stride1D.Dense> data, int d)
    {
        // FWHT butterfly: log2(d) sequential passes, each parallelizable
        // Each pass processes d/2 pairs independently
        int numStages = 0;
        for (int s = d; s > 1; s >>= 1) numStages++;

        for (int stage = 0; stage < numStages; stage++)
        {
            int halfSize = 1 << stage;
            _fwhtKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, int>(FWHTStageImpl);
            _fwhtKernel(d / 2, data, halfSize);
        }

        // Normalize by 1/sqrt(d)
        float scale = 1f / MathF.Sqrt(d);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(data, data, d, scale);
    }

    /// <summary>
    /// Batched FWHT: apply FWHT to each vector in a batch.
    /// Input: [batchSize, d] flattened. Each row gets its own FWHT.
    /// </summary>
    public void ForwardBatch(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchSize, int d)
    {
        // Copy input to output (FWHT is in-place)
        output.SubView(0, batchSize * d).CopyFrom(input.SubView(0, batchSize * d));

        int numStages = 0;
        for (int s = d; s > 1; s >>= 1) numStages++;

        for (int stage = 0; stage < numStages; stage++)
        {
            int halfSize = 1 << stage;
            _fwhtBatchKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int>(FWHTBatchStageImpl);
            _fwhtBatchKernel(batchSize * d / 2, output, output, halfSize);
        }

        // Normalize
        float scale = 1f / MathF.Sqrt(d);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output, output, batchSize * d, scale);
    }

    /// <summary>
    /// One butterfly stage of the FWHT. Each thread handles one pair.
    /// </summary>
    private static void FWHTStageImpl(Index1D pairIdx,
        ArrayView1D<float, Stride1D.Dense> data,
        int halfSize)
    {
        int blockSize = halfSize * 2;
        int block = pairIdx / halfSize;
        int offset = pairIdx % halfSize;
        int i = block * blockSize + offset;
        int j = i + halfSize;

        float a = data[i];
        float b = data[j];
        data[i] = a + b;
        data[j] = a - b;
    }

    /// <summary>
    /// Batched butterfly stage. Handles multiple vectors at once.
    /// </summary>
    private static void FWHTBatchStageImpl(Index1D globalPairIdx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int halfSize)
    {
        int blockSize = halfSize * 2;
        int block = globalPairIdx / halfSize;
        int offset = globalPairIdx % halfSize;
        int i = block * blockSize + offset;
        int j = i + halfSize;

        float a = input[i];
        float b = input[j];
        output[i] = a + b;
        output[j] = a - b;
    }
}
