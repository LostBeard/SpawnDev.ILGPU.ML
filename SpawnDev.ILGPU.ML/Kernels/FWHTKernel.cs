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

    // One-store-per-thread out-of-place butterfly (WebGL Transform-Feedback safe; correct on all
    // backends). The old in-place 2-store butterfly silently corrupts on the WebGL TF vertex path
    // (the vertex shader maps a thread's store-site s to output index v*storeCount+s, so a thread's
    // SECOND store lands at the wrong index and only the first survives). The multi-pass path now
    // ping-pongs between a work buffer and a scratch buffer with this single-store kernel.
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _fwhtBatchOutKernel;

    // Pooled padding scratch for the non-power-of-2 batched path. MUST be a member buffer:
    // a method-local `using var` is disposed when ForwardBatch returns — before the CALLER
    // flushes the WebGPU command encoder — so the GPU buffer is destroyed while the FWHT
    // dispatches that read/write it are still pending ("Buffer used in submit while destroyed").
    // Reused across calls; on growth the previous buffer is retained (never disposed mid-flight,
    // since it may still sit in a not-yet-submitted command encoder), mirroring TurboQuantKernels'
    // _oldParamsBufs pattern. Avoids the per-call allocate/free churn too (Rule 4).
    private MemoryBuffer1D<float, Stride1D.Dense>? _padBuf;
    private int _padCapacity;
    private readonly System.Collections.Generic.List<MemoryBuffer1D<float, Stride1D.Dense>> _oldPadBufs = new();

    // Ping-pong scratch for the one-store-per-thread multi-pass FWHT (see _fwhtBatchOutKernel).
    // Same grow-and-retain discipline as _padBuf: never disposed mid-flight (it may still sit in an
    // unsubmitted command encoder), reused across calls, grown only when a larger request arrives.
    private MemoryBuffer1D<float, Stride1D.Dense>? _pingBuf;
    private int _pingCapacity;

    public FWHTKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// (Re)allocate the ping-pong scratch so it covers <paramref name="count"/> floats. Grow-and-retain:
    /// the previous buffer is retained (never freed mid-flight) since it may still sit in a pending
    /// command encoder. Returns a view of exactly <paramref name="count"/> elements.
    /// </summary>
    private ArrayView1D<float, Stride1D.Dense> EnsurePingBuffer(int count)
    {
        if (_pingBuf == null || _pingCapacity < count)
        {
            if (_pingBuf != null) _oldPadBufs.Add(_pingBuf);
            _pingBuf = _accelerator.Allocate1D<float>(count);
            _pingCapacity = count;
        }
        return _pingBuf.View.SubView(0, count);
    }

    /// <summary>
    /// Run all log2(d) FWHT butterfly stages with one-store-per-thread ping-pong between
    /// <paramref name="work"/> and <paramref name="scratch"/> (both must hold <paramref name="total"/>
    /// floats; the initial data must already be in <paramref name="work"/>). Returns true if the
    /// result ended up in <paramref name="work"/>, false if it ended up in <paramref name="scratch"/>.
    /// No normalization is applied — the caller fuses the 1/sqrt(d) scale into its final write.
    /// </summary>
    private bool RunButterflyStagesPingPong(
        ArrayView1D<float, Stride1D.Dense> work,
        ArrayView1D<float, Stride1D.Dense> scratch,
        int d, int total)
    {
        _fwhtBatchOutKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(
                FWHTBatchStageOutOfPlaceImpl);

        int numStages = 0;
        for (int s = d; s > 1; s >>= 1) numStages++;

        var src = work;
        var dst = scratch;
        for (int stage = 0; stage < numStages; stage++)
        {
            int halfSize = 1 << stage;
            _fwhtBatchOutKernel(total, src, dst, halfSize);
            (src, dst) = (dst, src);
        }
        // After numStages swaps starting from src=work: result is in `work` iff numStages is even.
        return (numStages & 1) == 0;
    }

    /// <summary>
    /// In-place FWHT on a single vector of length d (must be power of 2).
    /// Normalized: output = H_d @ input / sqrt(d).
    /// </summary>
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>? _scaleInPlaceKernel;

    public void Forward(ArrayView1D<float, Stride1D.Dense> data, int d)
    {
        // FWHT butterfly: log2(d) sequential passes via one-store-per-thread ping-pong
        // (WebGL Transform-Feedback safe). Initial data is already in `data`; ping-pong with
        // a scratch buffer, then fuse the 1/sqrt(d) normalization into the final write back.
        var scratch = EnsurePingBuffer(d);
        bool inData = RunButterflyStagesPingPong(data, scratch, d, d);

        float scale = 1f / MathF.Sqrt(d);
        if (inData)
        {
            _scaleInPlaceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, float>(ScaleInPlaceImpl);
            _scaleInPlaceKernel(d, data, scale);
        }
        else
        {
            // Result landed in scratch — normalize it back into `data` in one pass.
            new ElementWiseKernels(_accelerator).Scale(scratch, data, d, scale);
        }
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
            // Non-power-of-2: use the pooled padded scratch buffer (see _padBuf field comment),
            // zero-fill, copy input rows, run FWHT on padded buffer, copy results back to output.
            int padCount = batchSize * dPad;
            if (_padBuf == null || _padCapacity < padCount)
            {
                if (_padBuf != null) _oldPadBufs.Add(_padBuf);
                _padBuf = _accelerator.Allocate1D<float>(padCount);
                _padCapacity = padCount;
            }
            var padView = _padBuf.View.SubView(0, padCount);
            ew.Fill(padView, padCount, 0f);
            for (int b = 0; b < batchSize; b++)
                ew.Scale(input.SubView(b * d, d), padView.SubView(b * dPad, d), d, 1f);

            // One-store-per-thread ping-pong butterfly (WebGL TF safe). padView holds the data;
            // ping-pong against a same-size scratch, then copy the first d elements of each padded
            // row to output with the 1/sqrt(d) normalization fused in.
            var padScratch = EnsurePingBuffer(padCount);
            bool inPad = RunButterflyStagesPingPong(padView, padScratch, dPad, padCount);
            var padResult = inPad ? padView : padScratch;

            float scalePad = 1f / MathF.Sqrt(d);
            for (int b = 0; b < batchSize; b++)
                ew.Scale(padResult.SubView(b * dPad, d), output.SubView(b * d, d), d, scalePad);
            return;
        }

        // Power-of-2: copy input → output, then one-store-per-thread ping-pong butterfly
        // (WebGL TF safe). The butterfly writes alternate between `output` and a scratch buffer.
        int total = batchSize * d;
        ew.Scale(input.SubView(0, total), output.SubView(0, total), total, 1f);

        var scratch = EnsurePingBuffer(total);
        bool inOutput = RunButterflyStagesPingPong(output.SubView(0, total), scratch, d, total);

        // Normalize, fusing the final copy-to-output when the result is in scratch.
        float scale = 1f / MathF.Sqrt(d);
        if (inOutput)
        {
            _scaleInPlaceKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, float>(ScaleInPlaceImpl);
            _scaleInPlaceKernel(total, output, scale);
        }
        else
        {
            ew.Scale(scratch, output.SubView(0, total), total, scale);
        }
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
    /// One butterfly stage of the FWHT, ONE store per thread (out-of-place, double-buffered).
    /// Each thread owns exactly one OUTPUT element <c>e</c> and writes <c>dst[e]</c> by reading
    /// its butterfly partner from <paramref name="src"/>. Dispatched over the full element count
    /// (not pairs), so it never relies on a thread writing two locations — which is required on
    /// the WebGL Transform-Feedback path, where a thread's second store silently lands at the
    /// wrong index. The caller ping-pongs <paramref name="src"/>/<paramref name="dst"/> per stage.
    ///
    /// For block size 2*halfSize: the lower half (offset &lt; halfSize) is an "i" position and gets
    /// <c>src[e] + src[e+halfSize]</c>; the upper half is a "j" position and gets
    /// <c>src[e-halfSize] - src[e]</c>. This reproduces the in-place butterfly exactly.
    /// </summary>
    private static void FWHTBatchStageOutOfPlaceImpl(Index1D index,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int halfSize)
    {
        int e = index;
        int blockSize = halfSize * 2;
        int offset = e % blockSize;
        if (offset < halfSize)
            dst[e] = src[e] + src[e + halfSize];   // "i" (lower-half) position
        else
            dst[e] = src[e - halfSize] - src[e];   // "j" (upper-half) position
    }
}
