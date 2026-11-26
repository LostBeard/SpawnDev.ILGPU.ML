using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The 32x32-tile convolution kernel (<see cref="Conv2DKernel.SmallTileMaxGroups"/>) gives the 64x64 kernel's results BIT FOR BIT: the same k terms in the same order with
    /// the same fused multiply-adds per output. Shapes that hit every edge: sizes that are not multiples of the tile or of the 16-wide k block, asymmetric padding, stride 2,
    /// dilation 2, a batch of 2, 3x3 and 1x1 kernels, each activation.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ConvSmallTile_MatchesTheLargeTileKernel_BitForBit() => await RunTest(async accelerator =>
    {
        if (accelerator.MaxNumThreadsPerGroup < 256 || accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException("the implicit-GEMM convolution needs 256-thread groups and shared memory");
        var conv = new Conv2DKernel(accelerator);
        // (batch, inC, H, W, outC, k, stride, pads t/l/b/r, dilation, activation)
        var cases = new (int B, int InC, int H, int W, int OutC, int K, int S, int Pt, int Pl, int Pb, int Pr, int D, int Act)[]
        {
            (1, 160, 7, 7, 960, 1, 1, 0, 0, 0, 0, 1, Conv2DKernel.ActNone),       // MobileNetV2's late pointwise: N = 49, the small grid
            (1, 320, 7, 7, 1280, 1, 1, 0, 0, 0, 0, 1, Conv2DKernel.ActRelu6),
            (1, 96, 14, 14, 576, 1, 1, 0, 0, 0, 0, 1, Conv2DKernel.ActRelu6),
            (1, 37, 11, 13, 45, 3, 1, 1, 1, 1, 1, 1, Conv2DKernel.ActRelu),       // nothing a multiple of anything
            (1, 24, 15, 15, 33, 3, 2, 0, 0, 1, 1, 1, Conv2DKernel.ActNone),       // stride 2, SAME pads [0,0,1,1]
            (2, 17, 9, 10, 20, 3, 1, 2, 2, 2, 2, 2, Conv2DKernel.ActRelu6),       // batch 2, dilation 2
            (1, 64, 5, 5, 70, 5, 1, 2, 1, 2, 3, 1, Conv2DKernel.ActRelu),         // 5x5, asymmetric pads
        };
        var rng = new Random(23);
        int savedSmall = Conv2DKernel.SmallTileMaxGroups, savedSmall1x1 = Conv2DKernel.SmallTileMaxGroups1x1;
        Conv2DKernel.SmallTileMaxGroups1x1 = int.MaxValue;   // the 1x1 shapes too
        var report = new List<string>();
        try
        {
            foreach (var (b, inC, h, w, outC, k, s, pt, pl, pb, pr, d, act) in cases)
            {
                int effK = d * (k - 1) + 1;
                int outH = (h + pt + pb - effK) / s + 1, outW = (w + pl + pr - effK) / s + 1;
                var xs = new float[b * inC * h * w]; for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
                var ws = new float[outC * inC * k * k]; for (int i = 0; i < ws.Length; i++) ws[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
                var bs = new float[outC]; for (int i = 0; i < bs.Length; i++) bs[i] = (float)(rng.NextDouble() - 0.5);
                using var x = accelerator.Allocate1D(xs); using var wt = accelerator.Allocate1D(ws); using var bias = accelerator.Allocate1D(bs);
                int outLen = b * outC * outH * outW;
                using var yLarge = accelerator.Allocate1D<float>(outLen); using var ySmall = accelerator.Allocate1D<float>(outLen);
                Conv2DKernel.SmallTileMaxGroups = 0;
                conv.ForwardPadded(x.View, wt.View, bias.View, yLarge.View, inC, h, w, outC, k, k, s, pt, pl, pb, pr, d, d, b, s, act);
                Conv2DKernel.SmallTileMaxGroups = int.MaxValue;
                conv.ForwardPadded(x.View, wt.View, bias.View, ySmall.View, inC, h, w, outC, k, k, s, pt, pl, pb, pr, d, d, b, s, act);
                var large = await yLarge.View.CopyToHostAsync();
                var small = await ySmall.View.CopyToHostAsync();
                int differ = 0, first = -1;
                for (int i = 0; i < outLen; i++)
                    if (BitConverter.SingleToInt32Bits(large[i]) != BitConverter.SingleToInt32Bits(small[i])) { differ++; if (first < 0) first = i; }
                string name = $"b{b} {inC}->{outC} {h}x{w} k{k} s{s} pads {pt},{pl},{pb},{pr} d{d} act{act}";
                if (differ != 0)
                    throw new Exception($"{name}: {differ} of {outLen} outputs differ from the 64x64 kernel (first at {first}: {large[first]:R} vs {small[first]:R})");
                report.Add($"{name}: {outLen} identical");
            }
        }
        finally { Conv2DKernel.SmallTileMaxGroups = savedSmall; Conv2DKernel.SmallTileMaxGroups1x1 = savedSmall1x1; }
        Console.WriteLine("[ConvSmallTile] " + string.Join("; ", report));
    });

    /// <summary>
    /// The 32x32-tile kernel against what MobileNetV2's small-grid convolutions take without it (the 64x64 kernel, or the 1x1 kernel where <see cref="Conv2DKernel.Use1x1Kernel"/>
    /// is on) on the desktop backends: 51 calls per sample queued between two syncs (the sync is amortised), medians of 9 samples, off and on alternated. A measurement.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> ConvSmallTile_DesktopAB() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("a desktop GPU measurement");
        var conv = new Conv2DKernel(accelerator);
        var shapes = new (int InC, int OutC, int HW, int K)[] { (160, 960, 7, 1), (960, 160, 7, 1), (320, 1280, 7, 1), (96, 576, 14, 1), (576, 96, 14, 1), (64, 384, 14, 1), (384, 64, 14, 1), (512, 512, 7, 3) };
        var rng = new Random(29);
        int saved = Conv2DKernel.SmallTileMaxGroups, saved1x1 = Conv2DKernel.SmallTileMaxGroups1x1;
        Conv2DKernel.SmallTileMaxGroups1x1 = int.MaxValue;   // measure every shape on the 32x32 kernel (the 1x1 limit is what this measures)
        var lines = new List<string>();
        try
        {
            foreach (var (inC, outC, hw, k) in shapes)
            {
                int pad = k / 2;
                var xs = new float[inC * hw * hw]; for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
                var ws = new float[outC * inC * k * k]; for (int i = 0; i < ws.Length; i++) ws[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
                using var x = accelerator.Allocate1D(xs); using var w = accelerator.Allocate1D(ws); using var b = accelerator.Allocate1D<float>(outC); using var y = accelerator.Allocate1D<float>((long)outC * hw * hw);
                b.MemSetToZero();
                async Task<double> Sample(int maxGroups)
                {
                    Conv2DKernel.SmallTileMaxGroups = maxGroups;
                    conv.ForwardPadded(x.View, w.View, b.View, y.View, inC, hw, hw, outC, k, k, 1, pad, pad, pad, pad);
                    await accelerator.SynchronizeAsync();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < 51; i++) conv.ForwardPadded(x.View, w.View, b.View, y.View, inC, hw, hw, outC, k, k, 1, pad, pad, pad, pad);
                    await accelerator.SynchronizeAsync();
                    return sw.Elapsed.TotalMilliseconds * 1000 / 51;
                }
                var off = new List<double>(); var on = new List<double>();
                for (int s = 0; s < 9; s++) { off.Add(await Sample(0)); on.Add(await Sample(int.MaxValue)); }
                off.Sort(); on.Sort();
                lines.Add($"{inC}->{outC} {hw}x{hw} k{k}: {off[4]:F1} / {on[4]:F1} us ({off[4] / on[4]:F2}x)");
            }
        }
        finally { Conv2DKernel.SmallTileMaxGroups = saved; Conv2DKernel.SmallTileMaxGroups1x1 = saved1x1; }
        var report = $"{accelerator.AcceleratorType} small-grid convolutions, without / with the 32x32-tile kernel (1x1 kernel {(Conv2DKernel.Use1x1Kernel ? "on" : "off")}): " + string.Join("; ", lines);
        Console.WriteLine("[ConvSmallTile] " + report);
        return report;
    });
}
