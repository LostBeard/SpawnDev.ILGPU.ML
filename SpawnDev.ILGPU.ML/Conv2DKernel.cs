using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

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

    // params: inC, inH, inW, outC, kH, kW, stride, padTL(packed), outHW(packed), dilHW(packed)
    private Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int>?
        _conv2dKernel;

    // Native low-precision weights: identical to _conv2dKernel but the WEIGHT (2nd view) is a low-p type T
    // (ILGPU.Half / BFloat16 / Float8E*). One compiled kernel per concrete T, cached; lazily loaded on first
    // use of that type. object-typed because each delegate is T-specific.
    private readonly Dictionary<Type, object> _conv2dLowPWeightKernels = new();

    // params: C, inH, inW, kH, kW, stride, padTL(packed), outHW(packed), dilHW(packed)
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
        int stride, int padTL, int outHW, int dilHW)
    {
        // outH/outW, BEGIN pads, and dilations are passed PACKED to stay within ILGPU's
        // 15-arg kernel limit: padTL=(padTop<<8)|padLeft, outHW=(outH<<16)|outW,
        // dilHW=(dilationH<<8)|dilationW. Recomputing dims here from a single symmetric pad
        // silently truncated stride-2 SAME convs (192->95 instead of 96), shearing every
        // downstream feature map.
        int padTop = padTL >> 8, padLeft = padTL & 0xFF;
        int outH = outHW >> 16, outW = outHW & 0xFFFF;
        int dilationH = dilHW >> 8, dilationW = dilHW & 0xFF;
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        // f32 accumulation (the ML-standard for conv): the rounding error over the inC*kH*kW MACs is ~1e-5
        // relative — imperceptible in an 8-bit image and within the MAC-scaled conv-test tolerance. f64
        // accumulation here was over-cautious "ultimate quality" and is far slower on every GPU backend
        // (consumer cards run f64 at ~1/64 of f32; WebGPU/WebGL EMULATE f64 via Dekker — the conv-heavy UNet/VAE
        // paid that on every MAC). Always read bias — no branch (ANGLE optimizer workaround).
        float sum = bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky * dilationH - padTop;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx * dilationW - padLeft;
                    if (ix < 0 || ix >= inW) continue;

                    sum += input[icBase + iy * inW + ix] * weight[wcBase + ky * kW + kx];
                }
            }
        }

        output[idx] = sum;
    }

    /// <summary>
    /// Conv2D NCHW with NATIVE low-precision WEIGHTS (<typeparamref name="T"/> = ILGPU.Half / BFloat16 /
    /// Float8E*) — identical math to Conv2DImpl, but each filter weight is read NATIVELY and converted to
    /// float in-register (PrecisionConvert) for the f32 accumulation. The weight stays native in
    /// GPU memory (no f32 temp buffer); input/bias/output stay fp32, no accuracy loss. The UNet is mostly
    /// Conv, so this is the bulk of the low-p memory win for SD-Turbo.
    /// </summary>
    private static void Conv2DLowPWeightImpl<T>(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTL, int outHW, int dilHW)
        where T : unmanaged, INumber<T>
    {
        int padTop = padTL >> 8, padLeft = padTL & 0xFF;
        int outH = outHW >> 16, outW = outHW & 0xFFFF;
        int dilationH = dilHW >> 8, dilationW = dilHW & 0xFF;
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int oc = rem / outH;

        float sum = bias[oc];

        for (int ic = 0; ic < inC; ic++)
        {
            int icBase = ic * inH * inW;
            int wcBase = oc * inC * kH * kW + ic * kH * kW;
            for (int ky = 0; ky < kH; ky++)
            {
                int iy = oy * stride + ky * dilationH - padTop;
                if (iy < 0 || iy >= inH) continue;

                for (int kx = 0; kx < kW; kx++)
                {
                    int ix = ox * stride + kx * dilationW - padLeft;
                    if (ix < 0 || ix >= inW) continue;

                    sum += input[icBase + iy * inW + ix] * PrecisionConvert.ConvertToSingle(weight[wcBase + ky * kW + kx]);
                }
            }
        }

        output[idx] = sum;
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
        => ForwardPadded(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>
    /// Conv2D NCHW with explicit asymmetric ONNX pads [padTop, padLeft, padBottom, padRight].
    /// Output dims are computed from the FULL (begin+end) pads — never from a single symmetric
    /// value — so stride-2 SAME convs (ONNX pads like [0,0,1,1]) produce the correct grid
    /// instead of a 1-short, sheared one.
    /// </summary>
    public void ForwardPadded(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();

        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        int totalOutputElements = outC * outH * outW;
        _convCallCount++;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {totalOutputElements} elements " +
                $"(outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        try
        {
            _conv2dKernel!(totalOutputElements, input, weight, bias, output,
                inC, inH, inW, outC, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
        }
        catch (global::ILGPU.Runtime.OpenCL.CLException clEx)
        {
            throw new InvalidOperationException(
                $"[Conv2DKernel.ForwardPadded call #{_convCallCount} {_accelerator.AcceleratorType}] "
                + $"OpenCL {clEx.Error} (CLError) at "
                + $"input=[{inC},{inH},{inW}] outC={outC} k={kH}x{kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] "
                + $"totalOutput={totalOutputElements}", clEx);
        }
    }

    /// <summary>fp16-weight Conv2D NCHW (asymmetric ONNX pads). T=Half wrapper over
    /// <see cref="ForwardPaddedLowPWeight{T}"/> (callers unchanged).</summary>
    public void ForwardPaddedHalfWeight(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
        => ForwardPaddedLowPWeight(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padTop, padLeft, padBottom, padRight, dilationH, dilationW);

    /// <summary>Conv2D NCHW (asymmetric ONNX pads) with NATIVE low-precision weights (<typeparamref name="T"/>
    /// = ILGPU.Half / BFloat16 / Float8E*). Identical to <see cref="ForwardPadded"/> but the weight stays
    /// native in GPU memory (no f32 temp); each weight is converted to float in-register via PrecisionConvert,
    /// fp32/fp64 accumulate.</summary>
    public void ForwardPaddedLowPWeight<T>(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<T, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
        where T : unmanaged, INumber<T>
    {
        var kernel = GetConv2DLowPWeightKernel<T>();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D(low-p) output dims invalid: outH={outH}, outW={outW} (inH={inH}, inW={inW}, kH={kH}, kW={kW}, " +
                $"stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}).");
        int totalOutputElements = outC * outH * outW;
        _convCallCount++;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D(low-p) NCHW output buffer too small: output.Length={output.Length} < {totalOutputElements} elements.");
        kernel(totalOutputElements, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
    }

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int, int, int, int, int, int, int> GetConv2DLowPWeightKernel<T>()
        where T : unmanaged, INumber<T>
    {
        if (!_conv2dLowPWeightKernels.TryGetValue(typeof(T), out var k))
            _conv2dLowPWeightKernels[typeof(T)] = k = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, int, int, int, int, int, int>(Conv2DLowPWeightImpl<T>);
        return (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>)k;
    }

    /// <summary>fp16-weight Conv2D NCHW (symmetric padding). See <see cref="Forward"/>.</summary>
    public void ForwardHalfWeight(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<global::ILGPU.Half, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW,
        int outC, int kH, int kW,
        int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardPaddedHalfWeight(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

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
        int stride, int padTL, int outHW, int dilHW)
    {
        // outH/outW + begin pads + dilations passed packed. See Conv2DImpl.
        int padTop = padTL >> 8, padLeft = padTL & 0xFF;
        int outH = outHW >> 16, outW = outHW & 0xFFFF;
        int dilationH = dilHW >> 8, dilationW = dilHW & 0xFF;
        int ox = idx % outW;
        int rem = idx / outW;
        int oy = rem % outH;
        int c = rem / outH;

        float sum = bias[c];

        int inBase = c * inH * inW;
        int wBase = c * kH * kW;
        for (int ky = 0; ky < kH; ky++)
        {
            int iy = oy * stride + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;

            for (int kx = 0; kx < kW; kx++)
            {
                int ix = ox * stride + kx * dilationW - padLeft;
                if (ix < 0 || ix >= inW) continue;

                sum += input[inBase + iy * inW + ix] * weight[wBase + ky * kW + kx];
            }
        }

        output[idx] = sum;
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
        => ForwardDepthwisePadded(input, weight, bias, output, C, inH, inW, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Depthwise Conv2D NCHW with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardDepthwisePadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)C * outH * outW;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NCHW output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(C={C} outH={outH} outW={outW}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");

        _depthwiseKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
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
        int stride, int padTL, int outHW, int dilHW)
    {
        // NHWC output: [oy, ox, oc] indexing. outH/outW + begin pads + dilations packed. See Conv2DImpl.
        int padTop = padTL >> 8, padLeft = padTL & 0xFF;
        int outH = outHW >> 16, outW = outHW & 0xFFFF;
        int dilationH = dilHW >> 8, dilationW = dilHW & 0xFF;
        int oc = idx % outC;
        int rem = idx / outC;
        int ox = rem % outW;
        int oy = rem / outW;

        float sum = bias[oc];

        int kernelSize = inC * kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ic = k / (kH * kW);
            int rem2 = k % (kH * kW);
            int ky = rem2 / kW;
            int kx = rem2 % kW;

            int iy = oy * stride + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * stride + kx * dilationW - padLeft;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = (iy * inW + ix) * inC + ic;
            int wIdx = ((oc * kH + ky) * kW + kx) * inC + ic;
            sum += input[inIdx] * weight[wIdx];
        }

        output[idx] = sum;
    }

    public void ForwardNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardNHWCPadded(input, weight, bias, output, inC, inH, inW, outC, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Conv2D NHWC with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardNHWCPadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(inC={inC}, inH={inH}, inW={inW}, outC={outC}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)outH * outW * outC;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"Conv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} outC={outC}, inC={inC} inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _conv2dNHWCKernel!((int)needed, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
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
        int stride, int padTL, int outHW, int dilHW)
    {
        // outH/outW + begin pads + dilations passed packed. See Conv2DImpl.
        int padTop = padTL >> 8, padLeft = padTL & 0xFF;
        int outH = outHW >> 16, outW = outHW & 0xFFFF;
        int dilationH = dilHW >> 8, dilationW = dilHW & 0xFF;
        int c = idx % C;
        int rem = idx / C;
        int ox = rem % outW;
        int oy = rem / outW;

        float sum = bias[c];

        int kernelSize = kH * kW;
        for (int k = 0; k < kernelSize; k++)
        {
            int ky = k / kW;
            int kx = k % kW;
            int iy = oy * stride + ky * dilationH - padTop;
            if (iy < 0 || iy >= inH) continue;
            int ix = ox * stride + kx * dilationW - padLeft;
            if (ix < 0 || ix >= inW) continue;

            int inIdx = (iy * inW + ix) * C + c;
            int wIdx = (ky * kW + kx) * C + c;
            sum += input[inIdx] * weight[wIdx];
        }

        output[idx] = sum;
    }

    public void ForwardDepthwiseNHWC(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW, int stride = 1, int padding = 0,
        int dilationH = 1, int dilationW = 1)
        => ForwardDepthwiseNHWCPadded(input, weight, bias, output, C, inH, inW, kH, kW,
            stride, padding, padding, padding, padding, dilationH, dilationW);

    /// <summary>Depthwise Conv2D NHWC with explicit asymmetric ONNX pads [top,left,bottom,right].</summary>
    public void ForwardDepthwiseNHWCPadded(
        ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias, ArrayView1D<float, Stride1D.Dense> output,
        int C, int inH, int inW, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
    {
        EnsureLoaded();
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output dimensions are invalid: outH={outH}, outW={outW} " +
                $"(C={C}, inH={inH}, inW={inW}, kH={kH}, kW={kW}, stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}). " +
                $"This usually means SAME padding was not applied correctly.");
        long needed = (long)outH * outW * C;
        if (output.Length < needed)
            throw new InvalidOperationException(
                $"DepthwiseConv2D NHWC output buffer too small: output.Length={output.Length} but kernel will write {needed} elements " +
                $"(outH={outH} outW={outW} C={C}, inH={inH} inW={inW} kH={kH} kW={kW} stride={stride} pads=[{padTop},{padLeft},{padBottom},{padRight}] dilation={dilationH}x{dilationW}). " +
                $"Upstream shape inference allocated wrong size.");
        _depthwiseNHWCKernel!((int)needed, input, weight, bias, output,
            C, inH, inW, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
    }

    private void EnsureLoaded()
    {
        _conv2dKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>(Conv2DImpl);
        // Low-p-weight conv kernels are lazy per concrete T (see ForwardPaddedLowPWeight / GetConv2DLowPWeightKernel).
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
