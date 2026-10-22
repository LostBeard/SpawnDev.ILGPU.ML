using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The implicit-GEMM 2D Conv (<see cref="Conv2DGemm"/>, fp16 weights, the path the SD VAE and UNet convolutions take) against a float64 CPU reference on the same fp16-rounded weights: a 3x3 same
    /// convolution, a stride-2 one with the asymmetric [0,0,1,1] padding of exported downsamplers, a dilated one, a 1x1 and a non-square-stride one, on a size that is not a tile multiple (17 x 23) with a
    /// batch of 2, with fp32 and with fp16 activations in and out. Relative RMS error must stay below 3e-3 (activations are rounded to fp16 inside the tensor-core path). Red check: a table that
    /// forgets the padding or the batch offset is off by far more.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Conv2DGemm_MatchesCpuReference() => await RunTest(async accelerator =>
    {
        using var conv = new Conv2DGemm(accelerator);
        var cases = new (string Name, int N, int C, int H, int W, int O, int KH, int KW, int SH, int SW, int PT, int PL, int PB, int PR, int DH, int DW)[]
        {
            ("3x3 same", 2, 32, 17, 23, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
            ("3x3 stride 2, pads 0,0,1,1", 1, 40, 18, 22, 48, 3, 3, 2, 2, 0, 0, 1, 1, 1, 1),
            ("3x3 dilation 2", 1, 32, 19, 19, 32, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2),
            ("1x1", 2, 64, 9, 11, 32, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1),
            ("3x3 stride 2 x 1", 1, 32, 16, 20, 40, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1),
        };
        var rng = new Random(3);
        var report = new List<string>();
        int ran = 0;
        foreach (var c in cases)
        {
            int taps = c.KH * c.KW, effKH = c.DH * (c.KH - 1) + 1, effKW = c.DW * (c.KW - 1) + 1;
            int outH = (c.H + c.PT + c.PB - effKH) / c.SH + 1, outW = (c.W + c.PL + c.PR - effKW) / c.SW + 1;
            if (!conv.Supports(c.N, c.C, c.H, c.W, c.O, c.KH, c.KW, outH, outW)) continue;
            ran++;
            var x = new float[(long)c.N * c.C * c.H * c.W]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var w = new float[(long)c.O * c.C * taps]; for (int i = 0; i < w.Length; i++) w[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
            var bias = new float[c.O]; for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);
            using var wBuf = accelerator.Allocate1D(w); using var wHalf = accelerator.Allocate1D<global::ILGPU.Half>(w.Length);
            using var biasBuf = accelerator.Allocate1D(bias);
            var cvt = new PrecisionConvertKernels(accelerator);
            cvt.FloatToHalf(wBuf.View, wHalf.View, w.Length);
            // the weights the GEMM sees: fp16-rounded
            var wRounded = new float[w.Length]; for (int i = 0; i < w.Length; i++) wRounded[i] = (float)(System.Half)w[i];
            var xHalfRounded = new float[x.Length]; for (int i = 0; i < x.Length; i++) xHalfRounded[i] = (float)(System.Half)x[i];

            foreach (bool halfAct in new[] { false, true })
            {
                var xs = halfAct ? xHalfRounded : x;
                var want = new float[(long)c.N * c.O * outH * outW];
                for (int b = 0; b < c.N; b++)
                    for (int oc = 0; oc < c.O; oc++)
                        for (int oy = 0; oy < outH; oy++)
                            for (int ox = 0; ox < outW; ox++)
                            {
                                double sum = bias[oc];
                                for (int ic = 0; ic < c.C; ic++)
                                    for (int ky = 0; ky < c.KH; ky++)
                                    {
                                        int iy = oy * c.SH - c.PT + ky * c.DH; if (iy < 0 || iy >= c.H) continue;
                                        for (int kx = 0; kx < c.KW; kx++)
                                        {
                                            int ix = ox * c.SW - c.PL + kx * c.DW; if (ix < 0 || ix >= c.W) continue;
                                            sum += (double)xs[(((long)b * c.C + ic) * c.H + iy) * c.W + ix] * wRounded[((long)oc * c.C + ic) * taps + ky * c.KW + kx];
                                        }
                                    }
                                want[(((long)b * c.O + oc) * outH + oy) * outW + ox] = (float)sum;
                            }
                float[] got;
                if (!halfAct)
                {
                    using var xBuf = accelerator.Allocate1D(x); using var yBuf = accelerator.Allocate1D<float>(want.Length);
                    conv.Run<float>(xBuf.View, wHalf.View, biasBuf.View, yBuf.View, c.N, c.C, c.H, c.W, c.O, c.KH, c.KW, c.SH, c.SW, c.PT, c.PL, c.DH, c.DW, outH, outW);
                    got = await yBuf.View.CopyToHostAsync();
                }
                else
                {
                    using var xf = accelerator.Allocate1D(x); using var xh = accelerator.Allocate1D<global::ILGPU.Half>(x.Length); using var yh = accelerator.Allocate1D<global::ILGPU.Half>(want.Length);
                    using var yf = accelerator.Allocate1D<float>(want.Length);
                    cvt.FloatToHalf(xf.View, xh.View, x.Length);
                    conv.Run<global::ILGPU.Half>(xh.View, wHalf.View, biasBuf.View, yh.View, c.N, c.C, c.H, c.W, c.O, c.KH, c.KW, c.SH, c.SW, c.PT, c.PL, c.DH, c.DW, outH, outW);
                    cvt.HalfToFloat(yh.View, yf.View, want.Length);
                    got = await yf.View.CopyToHostAsync();
                }
                double se = 0, sr = 0; for (int i = 0; i < want.Length; i++) { double d = got[i] - want[i]; se += d * d; sr += (double)want[i] * want[i]; }
                double rel = Math.Sqrt(se / Math.Max(1e-30, sr));
                report.Add($"{c.Name} {(halfAct ? "fp16" : "fp32")} activations: relRMS {rel:E2}");
                if (!(rel < (halfAct ? 6e-3 : 3e-3))) throw new Exception($"{c.Name} ({(halfAct ? "fp16" : "fp32")} activations) on {accelerator.AcceleratorType}: relRMS {rel:E2}");
            }
        }
        if (ran == 0) throw new UnsupportedTestException($"no GEMM path for these convolutions on {accelerator.AcceleratorType}");
        Console.WriteLine($"[ConvGemm] {accelerator.AcceleratorType}: {string.Join("; ", report)}");
        return $"{ran} convolutions";
    });
}
