using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// MagCache's skip rule against MagCache's own code: the skip pattern (one character per call, conditional and unconditional interleaved) that magcache_forward's
    /// decision makes with its TI2V-5B tables, run in Python on MagCache4Wan2.2/magcache_generate.py at df81cb18 (its nearest_interp for 30 and 40 steps). Host only.
    /// </summary>
    [TestMethod]
    public async Task WanMagCache_Decisions_MatchMagCache() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        var cases = new (string Table, int Steps, double Threshold, string Mask)[]
        {
            ("t2v", 50, 0.04, "0000000000000000000011110011110011110011110011110011110011110011001100110011001100110000000000000011"),  // 42 of 100 skipped
            ("t2v", 30, 0.04, "000000000000111100111100111100111100110011001100110000000011"),  // 26 of 60 skipped
            ("i2v", 50, 0.04, "0000000000000000000011110011110011110011110011110011110011110011001100110011001100110000000000000011"),  // 42 of 100 skipped
            ("i2v", 40, 0.08, "00000000000000001111001111001111001111001111001111001111001111001100110011001100"),  // 40 of 80 skipped
            ("t2v", 50, 0.12, "0000000000000000000011110011110011110011110011110011110011110011110011110011110011110011001100110011"),  // 52 of 100 skipped
        };
        foreach (var (table, steps, th, mask) in cases)
        {
            var state = new WanMagCache.State(new WanMagCache(th), table == "t2v" ? WanMagCache.Ti2v5bTextToVideo : WanMagCache.Ti2v5bImageToVideo, steps);
            var got = new System.Text.StringBuilder();
            for (int s = 0; s < steps; s++) for (int b = 0; b < 2; b++) got.Append(state.Skip(s, b) ? '1' : '0');
            if (got.ToString() != mask) throw new Exception($"{table} {steps} steps threshold {th}: got\n{got}\nwant\n{mask}");
        }
        await Task.CompletedTask;
    });

    /// <summary>
    /// The residual cache on the real DiT (q8, the step-0 inputs of the 640x352 x17 reference run): a call with the cache on gives the plain call's output bit-identically, a call that
    /// reuses the residual on the same input gives it to rounding (input + (output - input)), two full calls on the same input measure a ratio of 1, and the other slot's residual
    /// is never used. Desktop GPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_MagCache_ReuseEqualsFullCall() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU");
        var dit8 = Path.Combine(WanDir, Wan22Paths.DitFile("q8")); var fixture = Path.Combine(WanDir, "dump2", "dit_t2v30.npz"); var t5 = Path.Combine(WanDir, "dump", "t5_0.npz");
        if (!File.Exists(dit8) || !File.Exists(fixture) || !File.Exists(t5)) throw new UnsupportedTestException($"Wan files not found in {WanDir}");
        using var fx = ZipFile.OpenRead(fixture); using var t50 = ZipFile.OpenRead(t5);
        var (lat, ls) = WanNpz(fx, "step0_latent_in"); var t = WanNpz(fx, "step0_t").Data[0]; var (ctx, cs) = WanNpz(t50, "context");
        using var weights = GpuWeights.LoadFile(accelerator, dit8);
        using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
        dit.SetGrid(ls[1], ls[2], ls[3]); dit.SetContext(0, ctx, cs[0]); dit.SetContext(1, ctx, cs[0]);
        using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
        async Task<float[]> Call(int slot, bool reuse) { dit.Forward(lIn.View, t, false, slot, lOut.View, reuse); return await lOut.View.CopyToHostAsync(); }
        var plain = await Call(0, false);
        dit.CacheResiduals = true; dit.MeasureResidualRatio = true;
        var cached = await Call(0, false);
        bool first = await dit.ReadResidualRatioAsync() == null;                       // nothing to compare with yet
        var again = await Call(0, false);
        double ratio = await dit.ReadResidualRatioAsync() ?? double.NaN;
        var reused = await Call(0, true);
        bool other = false;
        try { await Call(1, true); } catch (InvalidOperationException) { other = true; }  // slot 1 has no residual: refused
        var mr = WanCompare(reused, plain);
        var report = $"cache on: {(SdSame(plain, cached) ? "plain output" : "DIFFERENT")}; reused residual vs full call relRMS {mr.RelRms:E2}; ratio of two identical calls {ratio:F6} (first call: {(first ? "none" : "UNEXPECTED")}); other slot: {(other ? "refused" : "USED")}";
        Console.WriteLine("[Wan] MagCache " + report);
        if (!(SdSame(plain, cached) && mr.RelRms < 1e-4 && Math.Abs(ratio - 1) < 1e-4 && first && other)) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// What MagCache costs a clip and saves: the reference run's prompt and starting noise (640x352 x17, guidance 5), 50 UniPC steps plain (timed; a second plain run measures the ratios) and with MagCache at thresholds 0.04
    /// (MagCache's default) and 0.12, PSNR of the decoded video against the plain one and the time of the run; plus our own calibration (the plain run measures the ratios)
    /// against MagCache's table. A measurement, logged to out\wan_tests.log; it fails only on a broken clip (PSNR under 20 dB at the default threshold). CUDA only (timings).
    /// </summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task Wan_MagCache_ClipQuality() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("a timing measurement: CUDA only");
        var paths = Wan22Paths.InFolder(WanDir, textEncoder: Wan22Paths.TextEncoderFile("q8")); var fixture = Path.Combine(WanDir, "dump2", "dit_t2v30.npz");
        if (!File.Exists(paths.Dit) || !File.Exists(paths.Vae) || !File.Exists(paths.TextEncoder) || !File.Exists(fixture)) throw new UnsupportedTestException($"Wan files not found in {WanDir}");
        const string prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.";
        using var fx = ZipFile.OpenRead(fixture);
        var (noise, ns) = WanNpz(fx, "noise");
        var pipe = new Wan22Pipeline(accelerator, paths);
        WanRequest Req(WanMagCache? mc, bool measure, int steps = 50) => new()
        {
            Prompt = prompt, Width = ns[3] * 16, Height = ns[2] * 16, Frames = 4 * ns[1] - 3, Steps = steps, InitialNoise = noise, MagCache = mc, MeasureMagRatios = measure,
        };
        await pipe.GenerateAsync(Req(null, false, 2));                    // warm-up: kernels compiled, the prompt's encoding cached
        var report = new List<string>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var plain = await pipe.GenerateAsync(Req(null, false));
        double plainS = sw.Elapsed.TotalSeconds;
        var measured = await pipe.GenerateAsync(Req(null, true));                       // the calibration run: a GPU sync per call, so not the timed one
        var ours = measured.MagRatios!; var theirs = WanMagCache.Ti2v5bTextToVideo;
        double maxD = 0, sumD = 0; int n = 0;
        for (int i = 2; i < Math.Min(ours.Count, theirs.Length); i++) { double d = Math.Abs(ours[i] - theirs[i]); maxD = Math.Max(maxD, d); sumD += d; n++; }
        report.Add($"plain {plainS:F1} s; our ratios (a second plain run) vs MagCache's table: mean |diff| {sumD / n:F5}, max {maxD:F5} (ours step 25 {ours[50]:F5} / theirs {theirs[50]:F5}, last {ours[^1]:F5} / {theirs[^1]:F5})");
        double worst = double.PositiveInfinity;
        foreach (var th in new[] { 0.04, 0.12 })
        {
            sw.Restart();
            var v = await pipe.GenerateAsync(Req(new WanMagCache(th), false));
            double s = sw.Elapsed.TotalSeconds, mse = 0;
            for (int i = 0; i < v.Data.Length; i++) { double d = (Math.Clamp(v.Data[i], -1f, 1f) - Math.Clamp(plain.Data[i], -1f, 1f)) / 2; mse += d * d; }
            double psnr = 10 * Math.Log10(1 / (mse / v.Data.Length));
            if (th == 0.04) worst = psnr;
            report.Add($"MagCache {th}: {v.MagCacheSkips.Skipped} of {v.MagCacheSkips.Calls} calls skipped, {s:F1} s ({plainS / s:F2}x), PSNR {psnr:F1} dB vs plain");
            try { var od = Path.Combine(WanDir, "out"); Directory.CreateDirectory(od); foreach (int f in new[] { 0, v.Frames - 1 }) File.WriteAllBytes(Path.Combine(od, $"magcache_{th}_{f}.png"), Assets.AssetGenerator.EncodePng(v.GetFrameRgba(f), v.Width, v.Height)); } catch (IOException) { }
        }
        foreach (var line in report) Console.WriteLine("[Wan] " + line);
        try { var od = Path.Combine(WanDir, "out"); Directory.CreateDirectory(od); File.AppendAllLines(Path.Combine(od, "wan_tests.log"), report.Select(l => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [MagCache {accelerator.AcceleratorType}] {l}")); } catch (IOException) { }
        if (worst < 20) throw new Exception("MagCache broke the clip: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
