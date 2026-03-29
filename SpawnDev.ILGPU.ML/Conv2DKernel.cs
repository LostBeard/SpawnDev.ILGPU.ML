using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// 2D Convolution kernel for neural network inference.
/// Supports arbitrary kernel sizes (1×1, 3×3, 14×14), stride, and padding.
/// Group=1 only (standard convolution).
///
/// Layout: NCHW (batch × channels × height × width)
/// Weights: [outChannels, inChannels, kH, kW]
///
/// Parameters are packed into an ArrayView to avoid WebGPU uniform buffer
/// packing issues with high scalar parameter counts.
/// </summary>
public class Conv2DKernel
{
    private readonly Accelerator _accelerator;

    // Conv2D with params packed into a buffer: [inC, inH, inW, outC, kH, kW, stride, padding]
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,  // input
        ArrayView1D<float, Stride1D.Dense>,  // weight
        ArrayView1D<float, Stride1D.Dense>,  // bias
        ArrayView1D<float, Stride1D.Dense>,  // output
        ArrayView1D<int, Stride1D.Dense>>?   // params [8]
        _conv2dKernel;

    // Persistent params buffer (reused across calls)
    // Depthwise Conv2D: group=inC, each channel convolved independently
    // Weight: [C, 1, kH, kW], params: [C, inH, inW, kH, kW, stride, padding]
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>?
        _depthwiseKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public Conv2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Conv2D: one thread per output element. Parameters read from params buffer.
    /// params[0]=inC, [1]=inH, [2]=inW, [3]=outC, [4]=kH, [5]=kW, [6]=stride, [7]=padding
    /// </summary>
    private static void Conv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inH = p[1]; int inW = p[2];
        int outC = p[3]; int kH = p[4]; int kW = p[5];
        int stride = p[6]; int padding = p[7];

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        // Double accumulation: eliminates precision errors across all backends.
        // Dekker f64 emulation on WebGPU/WebGL is fast (90 FPS proven).
        // Always read bias — no conditional branch. ANGLE's HLSL optimizer changes
        // FP evaluation of the accumulation loop when a branch precedes it, causing
        // 0.009 error on WebGL. Callers must always provide a valid bias buffer
        // (zero-filled if no bias). See data-FINAL-ROOT-CAUSE-bias-branch.
        double sum = (double)bias[oc];

        // Triple-nested convolution loop. Requires SpawnDev.ILGPU with the
        // PushPhiValuesTransitive fix (commit 2b6b314) for correct WGSL codegen.
        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky - padding;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx - padding;
                    if (ix < 0 || ix >= inW) continue;

                    sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                }
            }
        }

        output[idx] = (float)sum;
    }

    /// <summary>
    /// Run Conv2D. Input: [inC, inH, inW]. Output: [outC, outH, outW].
    /// Weight: [outC, inC, kH, kW]. Bias: [outC] or empty.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0)
    {
        EnsureLoaded();

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        int totalOutputElements = outC * outH * outW;

        // Pack params into persistent buffer. WebGPU guarantees writeBuffer→dispatch
        // ordering within the same queue, so reusing the buffer is safe.
        // Do NOT use 'using var' — the GPU reads the buffer asynchronously after dispatch.
        _paramsBuf ??= _accelerator.Allocate1D<int>(8);
        _paramsBuf.CopyFromCPU(new int[] { inC, inH, inW, outC, kH, kW, stride, padding });

        _conv2dKernel!(totalOutputElements, input, weight, bias, output, _paramsBuf.View);
    }

    /// <summary>
    /// Depthwise Conv2D: each input channel convolved independently.
    /// Weight: [C, 1, kH, kW]. params: [C, inH, inW, kH, kW, stride, padding]
    /// </summary>
    private static void DepthwiseConv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[0]; int inH = p[1]; int inW = p[2];
        int kH = p[3]; int kW = p[4]; int stride = p[5]; int padding = p[6];

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int c = rem / outH;

        double sum = (double)bias[c]; // Always read — no branch (ANGLE optimizer workaround)

        int inBase = c * inH * inW;
        int wBase = c * kH * kW; // weight [C, 1, kH, kW] = [C, kH*kW]
        for (int ky = 0; ky < kH; ky++)
        {
            int iy = oy * stride + ky - padding;
            if (iy < 0 || iy >= inH) continue;

            for (int kx = 0; kx < kW; kx++)
            {
                int ix = ox * stride + kx - padding;
                if (ix < 0 || ix >= inW) continue;

                sum += (double)input[inBase + iy * inW + ix] * (double)weight[wBase + ky * kW + kx];
            }
        }

        output[idx] = (float)sum;
    }

    /// <summary>
    /// Depthwise Conv2D: group=inC, each channel convolved independently.
    /// Weight: [C, 1, kH, kW]. Bias: [C] or empty.
    /// </summary>
    public void ForwardDepthwise(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW,
        int kH, int kW,
        int stride = 1, int padding = 0)
    {
        EnsureLoaded();
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        _paramsBuf ??= _accelerator.Allocate1D<int>(8);
        _paramsBuf.CopyFromCPU(new int[] { C, inH, inW, kH, kW, stride, padding, 0 });

        _depthwiseKernel!(C * outH * outW, input, weight, bias, output, _paramsBuf.View);
    }

    // ═══ NHWC Variants (TFLite native layout) ═══

    /// <summary>
    /// Conv2D NHWC: input [N,H,W,inC], weight [outC,kH,kW,inC], output [N,outH,outW,outC].
    /// One thread per output element. Native NHWC indexing — zero layout conversion.
    /// </summary>
    private static void Conv2DNHWCImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int inC = p[0]; int inH = p[1]; int inW = p[2];
        int outC = p[3]; int kH = p[4]; int kW = p[5];
        int stride = p[6]; int padding = p[7];

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        // NHWC output: [oy, ox, oc] indexing
        int oc = idx % outC;
        int rem = idx / outC;
        int ox = rem % outW;
        int oy = rem / outW;

        double sum = (double)bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky - padding;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx - padding;
                    if (ix < 0 || ix >= inW) continue;

                    // NHWC input: [iy, ix, ic]
                    int inIdx = (iy * inW + ix) * inC + ic;
                    // NHWC weight (TFLite OHWI): [oc, ky, kx, ic]
                    int wIdx = ((oc * kH + ky) * kW + kx) * inC + ic;
                    sum += (double)input[inIdx] * (double)weight[wIdx];
                }
            }
        }

        output[idx] = (float)sum;
    }

    /// <summary>
    /// Depthwise Conv2D NHWC: input [N,H,W,C], weight [1,kH,kW,C], output [N,outH,outW,C].
    /// </summary>
    private static void DepthwiseConv2DNHWCImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[0]; int inH = p[1]; int inW = p[2];
        int kH = p[3]; int kW = p[4]; int stride = p[5]; int padding = p[6];

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        // NHWC output: [oy, ox, c]
        int c = idx % C;
        int rem = idx / C;
        int ox = rem % outW;
        int oy = rem / outW;

        double sum = (double)bias[c];

        for (int ky = 0; ky < kH; ky++)
        {
            int iy = oy * stride + ky - padding;
            if (iy < 0 || iy >= inH) continue;

            for (int kx = 0; kx < kW; kx++)
            {
                int ix = ox * stride + kx - padding;
                if (ix < 0 || ix >= inW) continue;

                // NHWC input: [iy, ix, c]
                int inIdx = (iy * inW + ix) * C + c;
                // NHWC weight [1, kH, kW, C]: [ky, kx, c]
                int wIdx = (ky * kW + kx) * C + c;
                sum += (double)input[inIdx] * (double)weight[wIdx];
            }
        }

        output[idx] = (float)sum;
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _conv2dNHWCKernel;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _depthwiseNHWCKernel;

    public void ForwardNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride = 1, int padding = 0)
    {
        _conv2dNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(Conv2DNHWCImpl);
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        _paramsBuf ??= _accelerator.Allocate1D<int>(8);
        _paramsBuf.CopyFromCPU(new int[] { inC, inH, inW, outC, kH, kW, stride, padding });
        _conv2dNHWCKernel(outH * outW * outC, input, weight, bias, output, _paramsBuf.View);
    }

    public void ForwardDepthwiseNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW, int stride = 1, int padding = 0)
    {
        _depthwiseNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(DepthwiseConv2DNHWCImpl);
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        _paramsBuf ??= _accelerator.Allocate1D<int>(8);
        _paramsBuf.CopyFromCPU(new int[] { C, inH, inW, kH, kW, stride, padding, 0 });
        _depthwiseNHWCKernel(outH * outW * C, input, weight, bias, output, _paramsBuf.View);
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(Conv2DImpl);
        _depthwiseKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(DepthwiseConv2DImpl);
    }
}
