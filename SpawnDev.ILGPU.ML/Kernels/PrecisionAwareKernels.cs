using System.Collections.Concurrent;
using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Approach-(i) precision-AWARE op kernels: a SINGLE generic <c>where T : INumber&lt;T&gt;</c> kernel per op
/// that reads its low-precision input DIRECTLY, computes in fp32 (precision), and writes low precision
/// DIRECTLY — NO fp32 temp buffers, NO convert-around-node (that was the workaround that didn't cut memory).
/// One kernel covers float / <see cref="ILGPU.Half"/> / <see cref="ILGPU.BFloat16"/> (and fp8 when its GPU
/// codegen lands), via <see cref="ILGPU.PrecisionConvert"/> (transpilable generic float↔T, ILGPU local.9).
/// This is the Rule-1/Rule-4 path (no workaround, half the activation bytes + bandwidth). The executor stores
/// intermediates as <c>T</c> and dispatches these for <c>T=Half</c> in F16 mode.
///
/// Plan: Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md.
/// </summary>
public sealed class PrecisionAwareKernels : IDisposable
{
    private readonly Accelerator _accelerator;
    // Per-element-type kernel cache (one compiled kernel per T per op). Keyed by typeof(T).
    private readonly ConcurrentDictionary<Type, object> _siluCache = new();
    private readonly ConcurrentDictionary<Type, object> _addCache = new();
    private readonly ConcurrentDictionary<Type, object> _mulCache = new();
    private readonly ConcurrentDictionary<Type, object> _groupNormCache = new();
    private readonly ConcurrentDictionary<Type, object> _conv2dCache = new();

    public PrecisionAwareKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ── SiLU (x * sigmoid(x)) — read T, compute fp32, write T. The gemma/SD activation; tanh/exp force fp32. ──
    private static void SiLUImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output)
        where T : unmanaged, INumber<T>
    {
        float x = PrecisionConvert.ConvertToSingle(input[i]);
        // SiLU = x / (1 + e^-x). Clamp the exponent tail for fp32 stability (matches the fp32 activation kernels).
        float s;
        if (x > 30f) s = x;
        else if (x < -30f) s = 0f;
        else s = x / (1f + XMath.Exp(-x));
        output[i] = PrecisionConvert.ConvertFromSingle<T>(s);
    }

    /// <summary>SiLU (x·sigmoid(x)) in place-of-precision T (float/Half/bf16). out[i] = silu(in[i]).
    /// in/out are <c>ArrayView1D&lt;T&gt;</c> — true low-precision I/O, no fp32 temp.</summary>
    public void SiLU<T>(ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output, int count)
        where T : unmanaged, INumber<T>
    {
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)
            _siluCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>(SiLUImpl<T>));
        k(count, input, output);
    }

    // ── Add (out = a + b) — elementwise residual. Pure PrecisionConvert, no fp32 temp buffers. ──
    private static void AddImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> a,
        ArrayView1D<T, Stride1D.Dense> b, ArrayView1D<T, Stride1D.Dense> output)
        where T : unmanaged, INumber<T>
    {
        float r = PrecisionConvert.ConvertToSingle(a[i]) + PrecisionConvert.ConvertToSingle(b[i]);
        output[i] = PrecisionConvert.ConvertFromSingle<T>(r);
    }

    /// <summary>Elementwise out[i] = a[i] + b[i] in precision T (float/Half/bf16). Low-precision I/O, no fp32 temp.</summary>
    public void Add<T>(ArrayView1D<T, Stride1D.Dense> a, ArrayView1D<T, Stride1D.Dense> b,
        ArrayView1D<T, Stride1D.Dense> output, int count)
        where T : unmanaged, INumber<T>
    {
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)
            _addCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>(AddImpl<T>));
        k(count, a, b, output);
    }

    // ── Mul (out = a * b) — elementwise gate (e.g. SiLU-gate). Pure PrecisionConvert, no fp32 temp. ──
    private static void MulImpl<T>(Index1D i, ArrayView1D<T, Stride1D.Dense> a,
        ArrayView1D<T, Stride1D.Dense> b, ArrayView1D<T, Stride1D.Dense> output)
        where T : unmanaged, INumber<T>
    {
        float r = PrecisionConvert.ConvertToSingle(a[i]) * PrecisionConvert.ConvertToSingle(b[i]);
        output[i] = PrecisionConvert.ConvertFromSingle<T>(r);
    }

    /// <summary>Elementwise out[i] = a[i] * b[i] in precision T (float/Half/bf16). Low-precision I/O, no fp32 temp.</summary>
    public void Mul<T>(ArrayView1D<T, Stride1D.Dense> a, ArrayView1D<T, Stride1D.Dense> b,
        ArrayView1D<T, Stride1D.Dense> output, int count)
        where T : unmanaged, INumber<T>
    {
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>)
            _mulCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>>(MulImpl<T>));
        k(count, a, b, output);
    }

    // ── GroupNorm — read low-p activations, fp32 mean/var + normalize, write low-p. Mirrors the fp32
    //    GroupNormKernel (one thread per output element, each recomputes its (batch,group) stats). The
    //    affine weight/bias stay fp32 (per-channel, tiny — not part of the activation working set). ──
    private static void GroupNormImpl<T>(Index1D idx,
        ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight, ArrayView1D<float, Stride1D.Dense> bias,
        int B, int C, int S, int G, float eps)
        where T : unmanaged, INumber<T>
    {
        int s = idx % S;
        int bc = idx / S;
        int channel = bc % C;
        int batch = bc / C;
        int channelsPerGroup = C / G;
        int group = channel / channelsPerGroup;
        int groupSize = channelsPerGroup * S;
        int groupBase = batch * C * S + group * channelsPerGroup * S;

        // fp32-accumulate mean over the whole (batch, group), reading low-p elements.
        float sum = 0f;
        for (int gi = 0; gi < groupSize; gi++)
            sum += PrecisionConvert.ConvertToSingle(input[groupBase + gi]);
        float mean = sum / groupSize;

        float varSum = 0f;
        for (int gi = 0; gi < groupSize; gi++)
        {
            float diff = PrecisionConvert.ConvertToSingle(input[groupBase + gi]) - mean;
            varSum += diff * diff;
        }
        float invStd = 1f / XMath.Sqrt(varSum / groupSize + eps);

        float x = PrecisionConvert.ConvertToSingle(input[idx]);
        float r = weight[channel] * (x - mean) * invStd + bias[channel];
        output[idx] = PrecisionConvert.ConvertFromSingle<T>(r);
    }

    /// <summary>GroupNorm on a 4D tensor [B, C, S] in precision T (float/Half/bf16). weight/bias are fp32 per-channel
    /// affine. Reads/writes low precision; mean/var accumulate in fp32. Matches the fp32 GroupNormKernel exactly.</summary>
    public void GroupNorm<T>(
        ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> weight, ArrayView1D<float, Stride1D.Dense> bias,
        int batchSize, int channels, int spatial, int numGroups, float epsilon = 1e-5f)
        where T : unmanaged, INumber<T>
    {
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float>)
            _groupNormCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                int, int, int, int, float>(GroupNormImpl<T>));
        k(batchSize * channels * spatial, input, output, weight, bias, batchSize, channels, spatial, numGroups, epsilon);
    }

    // ── Conv2D NCHW (group 1) — read low-p input, DOUBLE accumulate, write low-p. Weight/bias stay fp32
    //    (weights aren't the activation working set; HalfTensor weights are a separate win). Mirrors the
    //    fp32 Conv2DImpl exactly: one thread per output element, packed dims/pads/dilations. ──
    private static void Conv2DImpl<T>(
        Index1D idx,
        ArrayView1D<T, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<T, Stride1D.Dense> output,
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

        double sum = (double)bias[oc];

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

                    sum += (double)PrecisionConvert.ConvertToSingle(input[icBase + iy * inW + ix])
                         * (double)weight[wcBase + ky * kW + kx];
                }
            }
        }

        output[idx] = PrecisionConvert.ConvertFromSingle<T>((float)sum);
    }

    /// <summary>Conv2D NCHW (group 1) with explicit ONNX pads [top,left,bottom,right], in precision T (float/Half/bf16).
    /// Input/output are low precision; weight/bias fp32; double accumulate. Mirrors the fp32 Conv2DKernel.ForwardPadded.</summary>
    public void Conv2D<T>(
        ArrayView1D<T, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<T, Stride1D.Dense> output,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int padBottom, int padRight,
        int dilationH = 1, int dilationW = 1)
        where T : unmanaged, INumber<T>
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + padTop + padBottom - effKH) / stride + 1;
        int outW = (inW + padLeft + padRight - effKW) / stride + 1;
        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException(
                $"Conv2D<{typeof(T).Name}> output dims invalid: outH={outH}, outW={outW} (inH={inH}, inW={inW}, kH={kH}, kW={kW}, " +
                $"stride={stride}, pads=[{padTop},{padLeft},{padBottom},{padRight}], dilation={dilationH}x{dilationW}).");
        int totalOutputElements = outC * outH * outW;
        if (output.Length < totalOutputElements)
            throw new InvalidOperationException(
                $"Conv2D<{typeof(T).Name}> NCHW output buffer too small: output.Length={output.Length} < {totalOutputElements} elements.");
        var k = (Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
            int, int, int, int, int, int, int, int, int, int>)
            _conv2dCache.GetOrAdd(typeof(T), _ => _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<T, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>,
                int, int, int, int, int, int, int, int, int, int>(Conv2DImpl<T>));
        k(totalOutputElements, input, weight, bias, output,
            inC, inH, inW, outC, kH, kW, stride, (padTop << 8) | padLeft, (outH << 16) | outW, (dilationH << 8) | dilationW);
    }

    public void Dispose() { }
}
