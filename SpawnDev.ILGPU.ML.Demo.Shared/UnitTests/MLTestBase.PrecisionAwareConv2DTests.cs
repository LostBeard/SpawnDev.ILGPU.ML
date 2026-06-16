using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Approach-(i) precision-AWARE Conv2D (NCHW, group 1): generic <see cref="PrecisionAwareKernels.Conv2D"/>
/// reads low-p input, accumulates in double, writes low precision DIRECTLY (no fp32 activation temp). Weight
/// and bias stay fp32. Verified vs a CPU fp32 reference (reading the SAME quantized input the kernel sees) for
/// Half and bf16 on EVERY backend. Conv2D is the bulk memory op in the SD VAE decoder.
/// </summary>
public abstract partial class MLTestBase
{
    // CPU double-accumulate Conv2D NCHW group-1, matching Conv2DKernel.Conv2DImpl. xq = quantized input (fp32).
    private static float[] Conv2DCpu(float[] xq, float[] weight, float[] bias,
        int inC, int inH, int inW, int outC, int kH, int kW,
        int stride, int padTop, int padLeft, int outH, int outW)
    {
        var o = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = bias[oc];
                    for (int ic = 0; ic < inC; ic++)
                    {
                        int icBase = ic * inH * inW;
                        int wcBase = oc * inC * kH * kW + ic * kH * kW;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = oy * stride + ky - padTop;
                            if (iy < 0 || iy >= inH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ox * stride + kx - padLeft;
                                if (ix < 0 || ix >= inW) continue;
                                sum += (double)xq[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                            }
                        }
                    }
                    o[(oc * outH + oy) * outW + ox] = (float)sum;
                }
        return o;
    }

    [TestMethod]
    public async Task PrecisionAwareConv2D_Half_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int inC = 3, inH = 8, inW = 8, outC = 4, kH = 3, kW = 3, stride = 1, pad = 1;
        const int outH = (inH + 2 * pad - kH) / stride + 1;   // 8
        const int outW = (inW + 2 * pad - kW) / stride + 1;   // 8
        int nIn = inC * inH * inW, nW = outC * inC * kH * kW, nOut = outC * outH * outW;
        var rng = new Random(301);
        var x = new float[nIn]; for (int i = 0; i < nIn; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var weight = new float[nW]; for (int i = 0; i < nW; i++) weight[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        var bias = new float[outC]; for (int i = 0; i < outC; i++) bias[i] = (float)(rng.NextDouble() * 0.4 - 0.2);

        var xh = new global::ILGPU.Half[nIn]; for (int i = 0; i < nIn; i++) xh[i] = (global::ILGPU.Half)x[i];
        var xq = new float[nIn]; for (int i = 0; i < nIn; i++) xq[i] = (float)xh[i];
        var expected = Conv2DCpu(xq, weight, bias, inC, inH, inW, outC, kH, kW, stride, pad, pad, outH, outW);

        using var inBuf = accelerator.Allocate1D(xh);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(nOut);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Conv2D<global::ILGPU.Half>(inBuf.View, wBuf.View, bBuf.View, outBuf.View,
            inC, inH, inW, outC, kH, kW, stride, pad, pad, pad, pad);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, nOut);

        for (int i = 0; i < nOut; i++)
        {
            float g = (float)got[i];
            if (MathF.Abs(g - expected[i]) > MathF.Max(2e-2f, MathF.Abs(expected[i]) * 2e-2f))
                throw new Exception($"Half precision-aware Conv2D @{i}: got {g}, want {expected[i]} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareConv2D] Half read+double-accum+write matches fp32 on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionAwareConv2D_BFloat16_MatchesFp32_AllBackends() => await RunTest(async accelerator =>
    {
        const int inC = 3, inH = 8, inW = 8, outC = 4, kH = 3, kW = 3, stride = 1, pad = 1;
        const int outH = (inH + 2 * pad - kH) / stride + 1;
        const int outW = (inW + 2 * pad - kW) / stride + 1;
        int nIn = inC * inH * inW, nW = outC * inC * kH * kW, nOut = outC * outH * outW;
        var rng = new Random(303);
        var x = new float[nIn]; for (int i = 0; i < nIn; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var weight = new float[nW]; for (int i = 0; i < nW; i++) weight[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        var bias = new float[outC]; for (int i = 0; i < outC; i++) bias[i] = (float)(rng.NextDouble() * 0.4 - 0.2);

        var xb = new global::ILGPU.BFloat16[nIn]; for (int i = 0; i < nIn; i++) xb[i] = (global::ILGPU.BFloat16)x[i];
        var xq = new float[nIn]; for (int i = 0; i < nIn; i++) xq[i] = (float)xb[i];
        var expected = Conv2DCpu(xq, weight, bias, inC, inH, inW, outC, kH, kW, stride, pad, pad, outH, outW);

        using var inBuf = accelerator.Allocate1D(xb);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(nOut);
        var pa = new PrecisionAwareKernels(accelerator);
        pa.Conv2D<global::ILGPU.BFloat16>(inBuf.View, wBuf.View, bBuf.View, outBuf.View,
            inC, inH, inW, outC, kH, kW, stride, pad, pad, pad, pad);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, nOut);

        for (int i = 0; i < nOut; i++)
        {
            float g = (float)got[i];
            if (MathF.Abs(g - expected[i]) > MathF.Max(6e-2f, MathF.Abs(expected[i]) * 6e-2f))
                throw new Exception($"bf16 precision-aware Conv2D @{i}: got {g}, want {expected[i]} on {BackendName}");
        }
        Console.WriteLine($"[PrecisionAwareConv2D] bf16 read+double-accum+write matches fp32 on {BackendName}");
    });
}
