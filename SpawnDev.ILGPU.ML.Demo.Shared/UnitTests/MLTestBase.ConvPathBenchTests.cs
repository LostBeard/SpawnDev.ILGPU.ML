using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Where the fp32 convolution should leave the direct kernel for the tiled GEMM (<see cref="Kernels.Conv2DGemm.RunF32"/>): both paths timed (median of several, after a warm call) over
    /// MobileNetV2's pointwise convolutions, ResNet-sized 3x3 ones and SD-VAE-sized ones. MEASURED first on MobileNetV2 in the browser (2026-11-25): the tiled path made the model's
    /// second run 94 ms against 30 ms on the direct kernels, and its 308 KB shader is 3.9 s of Chrome's cold compile. A measurement: the rule comes from these numbers.
    /// </summary>
    [TestMethod(Timeout = 1800000)]
    public async Task<string> Conv_DirectVsTiled_Crossover() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("the GPU backends the tiled GEMM runs on");
        var direct = new Conv2DKernel(accelerator);
        using var tiled = new Kernels.Conv2DGemm(accelerator);
        // (inC, outC, H, W, k): MobileNetV2 pointwise, ResNet 3x3, SD VAE 3x3
        var shapes = new (int InC, int OutC, int H, int W, int K)[]
        {
            (32, 16, 112, 112, 1), (96, 24, 56, 56, 1), (144, 32, 28, 28, 1), (192, 64, 14, 14, 1), (384, 96, 14, 14, 1), (576, 160, 7, 7, 1), (960, 320, 7, 7, 1), (320, 1280, 7, 7, 1),
            (64, 64, 56, 56, 3), (128, 128, 28, 28, 3), (256, 256, 14, 14, 3), (512, 512, 7, 7, 3),
            (128, 128, 128, 128, 3), (256, 256, 64, 64, 3), (512, 512, 64, 64, 3), (256, 256, 128, 128, 3),
        };
        const int reps = 11;
        var rng = new Random(11);
        var lines = new List<string>();
        foreach (var (inC, outC, H, W, k) in shapes)
        {
            int pad = k / 2;
            if (!tiled.SupportsExact(1, inC, H, W, outC, k, k, H, W)) { lines.Add($"{inC}->{outC} {H}x{W} k{k}: tiled path does not take it"); continue; }
            var xs = new float[inC * H * W]; for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
            var ws = new float[outC * inC * k * k]; for (int i = 0; i < ws.Length; i++) ws[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            var bs = new float[outC]; for (int i = 0; i < bs.Length; i++) bs[i] = (float)(rng.NextDouble() * 0.1);
            using var x = accelerator.Allocate1D(xs); using var w = accelerator.Allocate1D(ws); using var b = accelerator.Allocate1D(bs);
            using var yd = accelerator.Allocate1D<float>((long)outC * H * W); using var yt = accelerator.Allocate1D<float>((long)outC * H * W);
            void Direct() => direct.ForwardPadded(x.View, w.View, b.View, yd.View, inC, H, W, outC, k, k, 1, pad, pad, pad, pad);
            void Tiled() => tiled.RunF32<float>(x.View, w.View, b.View, yt.View, 1, inC, H, W, outC, k, k, 1, 1, pad, pad, 1, 1, H, W);
            async Task<double> Time(Action run)
            {
                run(); await accelerator.SynchronizeAsync();
                var t = new List<double>();
                for (int i = 0; i < reps; i++) { var sw = Stopwatch.StartNew(); run(); await accelerator.SynchronizeAsync(); t.Add(sw.Elapsed.TotalMilliseconds); }
                t.Sort(); return t[reps / 2];
            }
            // interleaved halves (direct, tiled, direct, tiled) so a clock or load drift does not favour one path
            double td1 = await Time(Direct), tt1 = await Time(Tiled), td2 = await Time(Direct), tt2 = await Time(Tiled);
            double td = Math.Min(td1, td2), tt = Math.Min(tt1, tt2);
            var a = await yd.View.CopyToHostAsync(); var c = await yt.View.CopyToHostAsync();
            // the largest difference against the output's own scale (per-element relative error is meaningless where sums cancel to near zero)
            double maxDiff = 0, maxMag = 0; for (int i = 0; i < a.Length; i++) { maxDiff = Math.Max(maxDiff, Math.Abs(a[i] - c[i])); maxMag = Math.Max(maxMag, Math.Abs(a[i])); }
            double maxRel = maxDiff / Math.Max(maxMag, 1e-30);
            double gflop = 2.0 * H * W * outC * inC * k * k / 1e9;
            lines.Add($"{inC}->{outC} {H}x{W} k{k} ({gflop:F3} GFLOP): direct {td:F2} ms, tiled {tt:F2} ms ({(td / tt):F2}x), max diff {maxRel:E1} of the output's range");
            if (maxRel > 1e-5) throw new Exception($"the two paths disagree: {lines[^1]}");
        }
        var report = $"{accelerator.AcceleratorType} median of {reps}: " + string.Join("; ", lines);
        Console.WriteLine("[ConvPath] " + report);
        return report;
    });
}
