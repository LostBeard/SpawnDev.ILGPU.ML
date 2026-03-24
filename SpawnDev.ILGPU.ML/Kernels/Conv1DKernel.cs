using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// 1D convolution kernel for audio and sequence models (Whisper, Wav2Vec2, etc.).
/// Supports stride, padding, dilation, and grouped convolution.
/// Layout: [N, C, L] (batch, channels, length).
/// </summary>
public class Conv1DKernel
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

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

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

        // Pack params into int array to avoid exceeding scalar parameter limits
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        _paramsBuf ??= _accelerator.Allocate1D<int>(12);
        var paramsData = new int[] { inC, inL, outC, outL, kL, stride, padding, dilation, groups, inCPerGroup, outCPerGroup, kernelLoopSize };
        _paramsBuf.CopyFromCPU(paramsData);

        _conv1dFlatKernel!(totalOutput, input, weight, bias, output, _paramsBuf.View);
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
}
