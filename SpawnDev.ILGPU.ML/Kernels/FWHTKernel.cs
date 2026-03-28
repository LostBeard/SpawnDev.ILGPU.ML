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
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>? _scaleInPlaceKernel;

    public void Forward(ArrayView1D<float, Stride1D.Dense> data, int d)
    {
        // FWHT butterfly: log2(d) sequential passes, each parallelizable
        int numStages = 0;
        for (int s = d; s > 1; s >>= 1) numStages++;

        for (int stage = 0; stage < numStages; stage++)
        {
            int halfSize = 1 << stage;
            _fwhtKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, int>(FWHTStageImpl);
            _fwhtKernel(d / 2, data, halfSize);
        }

        // Normalize by 1/sqrt(d) — in-place to avoid WebGPU buffer aliasing
        float scale = 1f / MathF.Sqrt(d);
        _scaleInPlaceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, float>(ScaleInPlaceImpl);
        _scaleInPlaceKernel(d, data, scale);
    }

    private static void ScaleInPlaceImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data, float scale)
    {
        data[idx] *= scale;
    }

    /// <summary>
    /// Batched FWHT: apply FWHT to each vector in a batch.
    /// Input: [batchSize, d] flattened. Each row gets its own FWHT.
    ///
    /// Uses shared memory single-dispatch path for d &lt;= 1024 (fits in one workgroup).
    /// Falls back to multi-dispatch global memory path for larger dimensions.
    /// </summary>
    public void ForwardBatch(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchSize, int d)
    {
        // Use shared memory path when d fits in one workgroup
        // Typical head dims: 64 (GPT-2), 128 (LLaMA), 256 — all fit
        if (d <= _accelerator.MaxNumThreadsPerGroup && d <= 1024 && (d & (d - 1)) == 0)
        {
            ForwardBatchShared(input, output, batchSize, d);
            return;
        }

        // Fallback: multi-dispatch global memory path for large d.
        // FWHT requires power-of-2 dimensions. Pad to next power of 2 if needed.
        int dPad = d;
        if ((dPad & (dPad - 1)) != 0)
        {
            dPad = 1;
            while (dPad < d) dPad <<= 1;
        }

        var ew = new ElementWiseKernels(_accelerator);

        if (dPad != d)
        {
            // Non-power-of-2: allocate padded temp buffer, zero-fill, copy input rows,
            // run FWHT on padded buffer, copy results back to output.
            using var padBuf = _accelerator.Allocate1D<float>(batchSize * dPad);
            ew.Fill(padBuf.View, batchSize * dPad, 0f);
            for (int b = 0; b < batchSize; b++)
                input.SubView(b * d, d).CopyTo(padBuf.View.SubView(b * dPad, d));

            int numStagesPad = 0;
            for (int s = dPad; s > 1; s >>= 1) numStagesPad++;
            for (int stage = 0; stage < numStagesPad; stage++)
            {
                int halfSize = 1 << stage;
                _fwhtBatchKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    int>(FWHTBatchStageImpl);
                _fwhtBatchKernel(batchSize * dPad / 2, padBuf.View, padBuf.View, halfSize);
            }

            // Copy first d elements of each padded row to output, with normalization
            float scalePad = 1f / MathF.Sqrt(d);
            for (int b = 0; b < batchSize; b++)
                ew.Scale(padBuf.View.SubView(b * dPad, d), output.SubView(b * d, d), d, scalePad);
            return;
        }

        // Power-of-2: in-place butterfly
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
        ew.Scale(output, output, batchSize * d, scale);
    }

    // ═══════════════════════════════════════════════════════════
    //  Shared memory single-dispatch FWHT
    //  One workgroup per batch element. All butterfly stages in shared memory.
    //  Reduces log2(d) kernel dispatches to ONE dispatch.
    // ═══════════════════════════════════════════════════════════

    private Action<KernelConfig, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, float>? _fwhtSharedKernel;

    private void ForwardBatchShared(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int batchSize, int d)
    {
        int numStages = 0;
        for (int s = d; s > 1; s >>= 1) numStages++;

        float scale = 1f / MathF.Sqrt(d);

        _fwhtSharedKernel ??= _accelerator.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, float>(FWHTSharedImpl);

        // One workgroup per batch element, d threads per workgroup
        var config = new KernelConfig(
            new Index1D(batchSize),  // grid: one workgroup per vector
            new Index1D(d));         // group: d threads (one per element)

        _fwhtSharedKernel(config, input, output, d, numStages, scale);
    }

    /// <summary>
    /// Shared memory FWHT kernel. Each workgroup processes one vector of length d.
    /// All log2(d) butterfly stages execute in shared memory with Group.Barrier()
    /// between stages. Normalization by 1/sqrt(d) is fused into the final write.
    /// </summary>
    private static void FWHTSharedImpl(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int d, int numStages, float scale)
    {
        // Allocate shared memory for one vector
        var shared = SharedMemory.Allocate<float>(1024);

        int batchIdx = Grid.IdxX;   // which vector in the batch
        int tid = Group.IdxX;       // thread index within the vector

        // Load from global memory into shared memory
        int globalIdx = batchIdx * d + tid;
        if (tid < d)
            shared[tid] = input[globalIdx];

        Group.Barrier();

        // All butterfly stages in shared memory
        for (int stage = 0; stage < numStages; stage++)
        {
            int halfSize = 1 << stage;
            int blockSize = halfSize * 2;

            // Each thread in the lower half of each butterfly block does one pair
            int block = tid / blockSize;
            int offset = tid % blockSize;

            if (offset < halfSize)
            {
                int i = block * blockSize + offset;
                int j = i + halfSize;

                float a = shared[i];
                float b = shared[j];
                shared[i] = a + b;
                shared[j] = a - b;
            }

            Group.Barrier();
        }

        // Write back to global memory with fused normalization
        if (tid < d)
            output[globalIdx] = shared[tid] * scale;
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
