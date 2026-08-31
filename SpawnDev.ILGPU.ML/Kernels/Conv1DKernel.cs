using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// 1D convolution kernel for audio and sequence models (Whisper, Wav2Vec2, etc.).
/// Supports stride, padding, dilation, and grouped convolution.
/// Layout: [N, C, L] (batch, channels, length).
/// </summary>
public class Conv1DKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input [inC * inL]
        ArrayView1D<float, Stride1D.Dense>,  // weight [outC * inC/groups * kL]
        ArrayView1D<float, Stride1D.Dense>,  // bias [outC] (or empty)
        ArrayView1D<float, Stride1D.Dense>,  // output [outC * outL]
        int, int, int, int, int, int, int, int>?  // inC, inL, outC, outL, kL, stride, padding, dilation, groups
        _conv1dKernel;

    // WORKAROUND: Flatten the triple loop (outC, inC/groups, kL) to single loop
    // to avoid WGSL/GLSL triple-nested loop codegen bug.
    // Remove this workaround once SpawnDev.ILGPU fixes GenerateLoopBody nested loop detection.
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>?   // params packed as ints
        _conv1dFlatKernel;

    // Per-call params buffer + deferred disposal (GatherKernel/SliceKernel pattern). Reusing ONE
    // buffer with CopyFromCPU-per-call is a batching hazard on async backends: a pending dispatch in
    // an un-submitted WebGPU encoder / on the Wasm worker pool still references it, so the next
    // call's overwrite hands the pending dispatch the WRONG params (the DAv3 Slice_4 corruption class).

    /// <summary>One buffer per distinct param set - eighteen Conv nodes share this one kernel instance.</summary>
    private readonly ParamBufferCache<int> _paramsCache = new();

    public Conv1DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Forward pass: 1D convolution.
    /// Input: [inC, inL], Weight: [outC, inC/groups, kL], Bias: [outC] (optional), Output: [outC, outL]
    /// outL = (inL + 2*padding - dilation*(kL-1) - 1) / stride + 1
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inL, int outC, int kL,
        int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
    {
        int outL = (inL + 2 * padding - dilation * (kL - 1) - 1) / stride + 1;
        int inCPerGroup = inC / groups;
        int outCPerGroup = outC / groups;
        int kernelLoopSize = inCPerGroup * kL;
        int totalOutput = outC * outL;

        EnsureLoaded();

        // Pack params into int array to avoid exceeding scalar parameter limits.
        var paramsData = new int[] { inC, inL, outC, outL, kL, stride, padding, dilation, groups, inCPerGroup, outCPerGroup, kernelLoopSize };

        // Reuse the buffer when these values are UNCHANGED, which for a fixed-shape graph means uploading
        // once for the life of the session - every param here is a pure function of the shapes.
        //
        // 🔴 This was "a FRESH buffer per call", and that made CUDA GRAPH CAPTURE IMPOSSIBLE for any graph
        // containing a Conv1D: `Allocate1D` is a cuMemAlloc, which is illegal inside a capture window and
        // faults with an ACCESS VIOLATION (0xC0000005) that no try/catch can see - the process just dies.
        // MEASURED via GraphExecutor.CaptureTraceFile on Silero VAD: the capture pass died at node 13,
        // `Conv /feature_extractor/Conv_output_0`, the graph's first Conv. CudaGraphCapture's own comments
        // predict exactly this failure.
        //
        // ⚠️ Why fresh-per-call existed, and why reuse is still safe: rewriting a params buffer that a
        // PENDING dispatch still reads would corrupt it (the WebGPU command-batching hazard). Reuse here
        // performs NO WRITE at all - the values are already the ones resident - so there is nothing to
        // corrupt. When the values genuinely change we allocate fresh exactly as before and retire the old
        // buffer for deferred disposal, because a queued dispatch may still be reading it.
        var paramsView = _paramsCache.Get(_accelerator, paramsData);

        _conv1dFlatKernel!(totalOutput, input, weight, bias, output, paramsView);
    }

    /// <summary>
    /// 1D convolution kernel with flattened inner loop.
    /// One thread per output element (oc, ox).
    /// Inner loop flattened: for i in 0..inCPerGroup*kL, decompose to (ic, kx).
    /// </summary>
    private static void Conv1DFlatImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inL = p[1]; int outC = p[2]; int outL = p[3];
        int kL = p[4]; int stride = p[5]; int padding = p[6]; int dilation = p[7];
        int groups = p[8]; int inCPerGroup = p[9]; int outCPerGroup = p[10]; int kernelLoopSize = p[11];

        int oc = idx / outL;
        int ox = idx % outL;

        int group = oc / outCPerGroup;
        int icStart = group * inCPerGroup;

        float sum = 0f;

        // Flattened loop over (ic_local, kx)
        for (int i = 0; i < kernelLoopSize; i++)
        {
            int icLocal = i / kL;
            int kx = i % kL;

            int ix = ox * stride + kx * dilation - padding;
            if (ix >= 0 && ix < inL)
            {
                int ic = icStart + icLocal;
                float inputVal = input[ic * inL + ix];
                float weightVal = weight[oc * inCPerGroup * kL + icLocal * kL + kx];
                sum += inputVal * weightVal;
            }
        }

        // Add bias
        if (bias.Length > 0)
            sum += bias[oc];

        output[idx] = sum;
    }

    private void EnsureLoaded()
    {
        _conv1dFlatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(Conv1DFlatImpl);
    }

    public void Dispose()
    {
        _paramsCache.Dispose();
    }
}
