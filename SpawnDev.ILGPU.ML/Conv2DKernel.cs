using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// 2D Convolution kernel for neural network inference.
/// Supports arbitrary kernel sizes (1×1, 3×3, 14×14), stride, padding, and dilation.
/// Group=1 (standard) and group=inC (depthwise via dedicated entry points).
///
/// Layout: NCHW (input [N,C,H,W], weight [outC,inC,kH,kW]) and NHWC (input [N,H,W,C],
/// weight [outC,kH,kW,inC] — TFLite-native).
///
/// Parameters are captured as scalars per the SpawnDev.ILGPU.ML CLAUDE.md guidance
/// (Lambda Kernels). No shared params buffer = no params-buffer race under async
/// dispatch on Wasm.
/// </summary>
public class Conv2DKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // params: inC, inH, inW, outC, kH, kW, stride, padding, dilationH, dilationW
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _conv2dKernel;

    // params: C, inH, inW, kH, kW, stride, padding, dilationH, dilationW
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int>?
        _depthwiseKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _conv2dNHWCKernel;

    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int>?
        _depthwiseNHWCKernel;

    public Conv2DKernel(Accelerator accelerator) => _accelerator = accelerator;

    private static long _convCallCount;

    /// <summary>
    /// Conv2D NCHW: one thread per output element. inC, inH, inW, outC, kH, kW,
    /// stride, padding, dilationH, dilationW are captured as scalar parameters.
    /// </summary>
    private static void Conv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padding, int dilationH, int dilationW)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        // Double accumulation: eliminates precision errors across all backends.
        // Always read bias — no branch (ANGLE optimizer workaround).
        double sum = (double)bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky * dilationH - padding;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx * dilationW - padding;
                    if (ix < 0 || ix >= inW) continue;

                    sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                }
            }
        }

        output[idx] = (float)sum;
    }

    /// <summary>
    /// Run Conv2D NCHW. Input: [inC, inH, inW]. Output: [outC, outH, outW].
    /// Weight: [outC, inC, kH, kW]. Bias: [outC] or empty.
    /// </summary>
    public void Forward(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();

        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, padding={padding}, dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        int totalOutputElements = outC * outH * outW;
        _convCallCount++;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {totalOutputElements} elements " +
                $"(outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} padding={padding} dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        try
        {
            _conv2dKernel!(totalOutputElements, input, weight, bias, output,
                inC, inH, inW, outC, kH, kW, stride, padding, dilationH, dilationW);
        }
        catch (global::ILGPU.Runtime.OpenCL.CLException clEx)
        {
            throw new InvalidOperationException(
                $"[Conv2DKernel.Forward call #{_convCallCount} {_accelerator.AcceleratorType}] "
                + $"OpenCL {clEx.Error} (CLError) at "
                + $"input=[{inC},{inH},{inW}] outC={outC} k={kH}x{kW} stride={stride} pad={padding} "
                + $"totalOutput={totalOutputElements}", clEx);
        }
    }

    /// <summary>
    /// Depthwise Conv2D NCHW: each input channel convolved independently.
    /// Weight: [C, 1, kH, kW]. Bias: [C].
    /// </summary>
    private static void DepthwiseConv2DImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW,
        int stride, int padding, int dilationH, int dilationW)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;

        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int c = rem / outH;

        double sum = (double)bias[c];

        int inBase = c * inH * inW;
        int wBase = c * kH * kW;
        for (int ky = 0; ky < kH; ky++)
        {
            int iy = oy * stride + ky * dilationH - padding;
            if (iy < 0 || iy >= inH) continue;

            for (int kx = 0; kx < kW; kx++)
            {
                int ix = ox * stride + kx * dilationW - padding;
                if (ix < 0 || ix >= inW) continue;

                sum += (double)input[inBase + iy * inW + ix] * (double)weight[wBase + ky * kW + kx];
            }
        }

        output[idx] = (float)sum;
    }

    public void ForwardDepthwise(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW,
        int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, padding={padding}, dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)C * outH * outW;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(C={C} outH={outH} outW={outW}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} padding={padding} dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");

        _depthwiseKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, kH, kW, stride, padding, dilationH, dilationW);
    }

    // ═══ NHWC Variants (TFLite native layout) ═══

    /// <summary>
    /// Conv2D NHWC: input [N,H,W,inC], weight [outC,kH,kW,inC], output [N,outH,outW,outC].
    /// </summary>
    private static void Conv2DNHWCImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padding, int dilationH, int dilationW)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;

        // NHWC output: [oy, ox, oc] indexing
        int oc = idx % outC;
        int rem = idx / outC;
        int ox = rem % outW;
        int oy = rem / outW;

        double sum = (double)bias[oc];

        int kernelSize = inC * kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ic = k / (kH * kW);
            int rem2 = k % (kH * kW);
            int ky = rem2 / kW;
            int kx = rem2 % kW;

            int iy = oy * stride + ky * dilationH - padding;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * stride + kx * dilationW - padding;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = (iy * inW + ix) * inC + ic;
            int wIdx = ((oc * kH + ky) * kW + kx) * inC + ic;
            sum += (double)input[inIdx] * (double)weight[wIdx];
        }

        output[idx] = (float)sum;
    }

    public void ForwardNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inC={inC}, inH={inH}, inW={inW}, outC={outC}, kH={kH}, kW={kW}, stride={stride}, padding={padding}, dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)outH * outW * outC;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"Conv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} padding={padding} dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _conv2dNHWCKernel!((int)needed, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, padding, dilationH, dilationW);
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
        int C, int inH, int inW, int kH, int kW,
        int stride, int padding, int dilationH, int dilationW)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;

        int c = idx % C;
        int rem = idx / C;
        int ox = rem % outW;
        int oy = rem / outW;

        double sum = (double)bias[c];

        int kernelSize = kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ky = k / kW;
            int kx = k % kW;
            int iy = oy * stride + ky * dilationH - padding;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * stride + kx * dilationW - padding;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = (iy * inW + ix) * C + c;
            int wIdx = (ky * kW + kx) * C + c;
            sum += (double)input[inIdx] * (double)weight[wIdx];
        }

        output[idx] = (float)sum;
    }

    public void ForwardDepthwiseNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, padding={padding}, dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)outH * outW * C;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} C={C}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} padding={padding} dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _depthwiseNHWCKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, kH, kW, stride, padding, dilationH, dilationW);
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>(Conv2DImpl);
        _depthwiseKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int>(DepthwiseConv2DImpl);
        _conv2dNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>(Conv2DNHWCImpl);
        _depthwiseNHWCKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int>(DepthwiseConv2DNHWCImpl);
    }

    public void Dispose() { /* no buffers owned */ }
}
