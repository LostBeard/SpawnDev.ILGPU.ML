using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="SdKernels.Conv"/> exact (the tiled gather GEMM, 2026-11-20: hoisted gather, and an fp32 weight's fp16 halves fused into one pass by
    /// TiledGemm.GemmGatherF16Split) against a float64 host convolution over the same weights (hi, or hi + lo) and bias, at edge shapes: cin 8 (the tap advances
    /// every k step), cout 130 (a partial column tile), odd picture sizes (partial row tiles, the border's absent neighbours). Within 2e-6 relative RMS.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Sd_ConvExact_MatchesFloat64() => await RunTest(async accelerator =>
    {
        var k = new SdKernels(accelerator);
        var rng = new Random(11);
        var report = new List<string>();
        foreach (var (H, W, cin, cout) in new[] { (12, 12, 64, 64), (9, 7, 8, 24), (13, 11, 128, 130) })
        {
            int M = H * W;
            var xs = new float[(long)M * cin]; for (long i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
            var wHi = new global::ILGPU.Half[(long)cout * 9 * cin]; var wLo = new global::ILGPU.Half[wHi.Length];
            for (long i = 0; i < wHi.Length; i++) { float v = (float)((rng.NextDouble() * 2 - 1) * 0.1); var hi = (global::ILGPU.Half)v; wHi[i] = hi; wLo[i] = (global::ILGPU.Half)(v - (float)hi); }
            var bHi = new global::ILGPU.Half[cout]; var bLo = new global::ILGPU.Half[cout];
            for (int i = 0; i < cout; i++) { float v = (float)(rng.NextDouble() - 0.5); var hi = (global::ILGPU.Half)v; bHi[i] = hi; bLo[i] = (global::ILGPU.Half)(v - (float)hi); }
            using var x = accelerator.Allocate1D(xs); using var y = accelerator.Allocate1D<float>((long)M * cout);
            using var whi = accelerator.Allocate1D(wHi); using var wlo = accelerator.Allocate1D(wLo);
            using var bhi = accelerator.Allocate1D(bHi); using var blo = accelerator.Allocate1D(bLo);
            using var nbr = accelerator.Allocate1D<int>((long)M * 27);
            k.BuildTable(nbr.View, H, W, H, W, 3, 1, 1, 1);
            var table = await nbr.View.CopyToHostAsync();
            foreach (bool lo in new[] { false, true })
            {
                k.Conv(x.View, nbr.View, whi.View, bhi.View, y.View, M, cin, cout, 9, exact: true, lo ? wlo.View : null, lo ? blo.View : null);
                var got = await y.View.CopyToHostAsync();
                double num = 0, den = 0;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < cout; n++)
                    {
                        double acc = (double)(float)bHi[n] + (lo ? (double)(float)bLo[n] : 0);
                        for (int t = 0; t < 9; t++)
                        {
                            int src = table[m * 27 + t]; if (src < 0) continue;
                            for (int c = 0; c < cin; c++)
                            {
                                long wi = ((long)n * 9 + t) * cin + c;
                                double wv = (double)(float)wHi[wi] + (lo ? (double)(float)wLo[wi] : 0);
                                acc += xs[(long)src * cin + c] * wv;
                            }
                        }
                        double d = got[(long)m * cout + n] - acc; num += d * d; den += acc * acc;
                    }
                double rel = Math.Sqrt(num / den);
                report.Add($"{H}x{W} {cin}->{cout} {(lo ? "hi+lo" : "hi")}: {rel:E2}");
                Ur_Expect(rel < 2e-6, $"{accelerator.AcceleratorType}: {report[^1]} (limit 2e-6)");
            }
        }
        string line = $"{accelerator.AcceleratorType}: " + string.Join("; ", report);
        Console.WriteLine("[SdConv] " + line);
        return line;
    });

    /// <summary>
    /// Benchmark (2026-11-20): the whole SD 1.5 VAE decode, 64 x 64 latent -> 512 x 512, in the BROWSER (the fp32 checkpoint over HTTP from the local file server:
    /// <c>node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769</c>; its VAE keeps the low halves). Warm once, then the median of SD_VAE_REPS (default 3).
    /// Against <see cref="Sd_VaeConv_Benchmark"/>'s sum, the share of the decode that is not 3x3 convolution. Prints "[SdConvBench] ...". Reported, not asserted.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_Vae_OverHttp_DecodeBenchmark() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser benchmark (WebGPU)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var source = new SdRangeSource(http);
        try { await http.GetStringAsync(SdServer + "browser/vae_sd15_512/shapes.json"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769"); }
        int reps = int.TryParse(Environment.GetEnvironmentVariable("SD_VAE_REPS"), out var rr) ? rr : 3;
        using var ck = await SdCheckpoint.LoadAsync(accelerator, source, "", Sd15File, new SdCheckpointOptions { LoadUNet = false, LoadTextEncoder = false });
        var vae = ck.Vae!;
        const int h = 64, w = 64;
        var z = Wan.Wan22Pipeline.Gaussian(4L * h * w, 3);
        using var zin = accelerator.Allocate1D(z); using var img = accelerator.Allocate1D<float>(3L * 64 * h * w);
        await vae.DecodeAsync(zin.View, h, w, img.View); await accelerator.SynchronizeAsync();       // warm (compile)
        var times = new List<double>();
        for (int i = 0; i < reps; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await vae.DecodeAsync(zin.View, h, w, img.View);
            await accelerator.SynchronizeAsync();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        string line = $"{accelerator.AcceleratorType} SD 1.5 VAE decode 512x512 (fp32 checkpoint, low halves kept): median {times[times.Count / 2]:F0} ms of {reps} ({string.Join(", ", times.Select(t => t.ToString("F0")))})";
        Console.WriteLine("[SdConvBench] " + line);
        return line;
    });

    /// <summary>
    /// Benchmark (2026-11-20, Lore's queued speed target: the SD VAE decode is 3.3 s on an RX 590 in Chrome, ~70% of it 3x3 convolutions): <see cref="SdKernels.Conv"/>
    /// exact (fp32 activations, the portable tiled GEMM in gather mode, what SdVae runs) at the SD VAE decoder's shapes for a 512 x 512 picture, with fp16 weights only and with
    /// the low halves (an fp32 file's VAE: <see cref="SdWeights.KeepLowHalves"/>). Random data, no files: runs on every fast lane. SD_CONV_SHAPES "H x W x C, ..." overrides,
    /// SD_CONV_REPS the timed repeats (default 3). Prints "[SdConvBench] ..." (PMT_CONSOLE_LOG=SdConvBench in the browser). Reported, not asserted.
    /// </summary>
    [TestMethod(Timeout = 1800000)]
    public async Task Sd_VaeConv_Benchmark() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU)) throw new UnsupportedTestException("a fast-lane benchmark");
        var shapes = (Environment.GetEnvironmentVariable("SD_CONV_SHAPES") is { Length: > 0 } s ? s : "64x64x512,128x128x512,256x256x512,256x256x256,512x512x256,512x512x128")
            .Split(',').Select(t => t.Split('x').Select(int.Parse).ToArray()).ToArray();
        int reps = int.TryParse(Environment.GetEnvironmentVariable("SD_CONV_REPS"), out var r) ? r : 3;
        var k = new SdKernels(accelerator);
        var rng = new Random(7);
        var report = new List<string>();
        double totalHi = 0, totalLo = 0;
        foreach (var sh in shapes)
        {
            int H = sh[0], W = sh[1], C = sh[2], M = H * W;
            var xs = new float[(long)M * C]; for (long i = 0; i < xs.Length; i++) xs[i] = (float)(rng.NextDouble() * 2 - 1);
            var wHi = new global::ILGPU.Half[(long)C * 9 * C]; var wLo = new global::ILGPU.Half[wHi.Length];
            for (long i = 0; i < wHi.Length; i++)
            {
                float v = (float)((rng.NextDouble() * 2 - 1) * 0.05);
                var hi = (global::ILGPU.Half)v; wHi[i] = hi; wLo[i] = (global::ILGPU.Half)(v - (float)hi);
            }
            var bHi = new global::ILGPU.Half[C]; var bLo = new global::ILGPU.Half[C];
            using var x = accelerator.Allocate1D(xs); using var y = accelerator.Allocate1D<float>((long)M * C);
            using var whi = accelerator.Allocate1D(wHi); using var wlo = accelerator.Allocate1D(wLo);
            using var bhi = accelerator.Allocate1D(bHi); using var blo = accelerator.Allocate1D(bLo);
            using var nbr = accelerator.Allocate1D<int>((long)M * 27);
            k.BuildTable(nbr.View, H, W, H, W, 3, 1, 1, 1);
            double flop = 2.0 * M * 9 * C * C;
            async Task<double> Time(bool lo)
            {
                k.Conv(x.View, nbr.View, whi.View, bhi.View, y.View, M, C, C, 9, exact: true, lo ? wlo.View : null, lo ? blo.View : null);   // warm (compile)
                await accelerator.SynchronizeAsync();
                var times = new List<double>();
                for (int i = 0; i < reps; i++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    k.Conv(x.View, nbr.View, whi.View, bhi.View, y.View, M, C, C, 9, exact: true, lo ? wlo.View : null, lo ? blo.View : null);
                    await accelerator.SynchronizeAsync();
                    times.Add(sw.Elapsed.TotalMilliseconds);
                }
                times.Sort();
                return times[times.Count / 2];
            }
            double tHi = await Time(false), tLo = await Time(true);
            totalHi += tHi; totalLo += tLo;
            report.Add($"{H}x{W}x{C}: fp16 {tHi:F1} ms ({flop / tHi / 1e9:F2} TFLOPS), +lo {tLo:F1} ms ({flop / tLo / 1e9:F2} TFLOPS effective)");
        }
        string line = $"{accelerator.AcceleratorType} ({accelerator.Name}) median of {reps}: " + string.Join("; ", report) + $"; sum fp16 {totalHi:F0} ms, +lo {totalLo:F0} ms";
        Console.WriteLine("[SdConvBench] " + line);
        return line;
    });
}
