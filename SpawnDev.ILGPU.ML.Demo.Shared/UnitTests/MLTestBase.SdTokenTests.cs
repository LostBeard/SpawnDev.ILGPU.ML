using System.IO.Compression;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Token reduction in SD 1.5's self-attention against diffusers with tomesd's own matching (tools/sd-harness/reference/ref_sd_tome.py -> ref\sd_sd15tome_small_f16w.npz,
    /// 32x40 latent, deterministic destinations): ToMe with ComfyUI's placement (queries merged; ratio 0.5 at the full resolution, 0.3 down to the second), tomesd's placement
    /// (queries, keys and values merged, the block input as the metric) and ToDo (keys and values nearest / nearest-exact / area downsampled). Each must match its reference,
    /// move the prediction well away from the plain one, and switching it off must give the plain prediction bit-identically. Gated with exact fp32 linears on every lane: the
    /// merge choices are discrete (argmax, rank), so CUDA's tensor-core rounding (plain 1.2e-3 off) flips a few of them (measured 1.7e-2 to 5.1e-2 against an effect of 0.25);
    /// on CUDA the tensor-core run is reported and only checked to stay well inside the effect.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_TokenReduction_MatchesTomesdAndToDo() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15tome_small_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"{fxPath} not found (tools/sd-harness/reference/ref_sd_tome.py)");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        Task<float[]> Eps() => unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var report = new List<string>();
        bool ok = true;
        var cases = new (string, Action<bool>)[]
        {
            ("tome_q", on => unet.TokenMerge = on ? new SdTokenMerge(0.5f, RandomDestinations: false) : null),
            ("tome_q2", on => unet.TokenMerge = on ? new SdTokenMerge(0.3f, MaxDownsample: 2, RandomDestinations: false) : null),
            ("tome_tokens", on => unet.TokenMerge = on ? new SdTokenMerge(0.5f, MergeKeysAndValues: true, MaxDownsample: 2, RandomDestinations: false) : null),
            ("todo_nearest", on => unet.TokenDownsample = on ? new SdTokenDownsample(2, 1, SdKvDownsampling.Nearest) : null),
            ("todo_exact", on => unet.TokenDownsample = on ? new SdTokenDownsample(2, 2, SdKvDownsampling.NearestExact) : null),
            ("todo_area", on => unet.TokenDownsample = on ? new SdTokenDownsample(2, 2, SdKvDownsampling.Area) : null),
        };
        bool saved = Trellis2.DiTKernels.EnableTensorCores;
        try
        {
            foreach (bool tc in accelerator.AcceleratorType == global::ILGPU.Runtime.AcceleratorType.Cuda ? new[] { false, true } : new[] { false })
            {
                Trellis2.DiTKernels.EnableTensorCores = tc;
                string tag = tc ? "tensor cores: " : "";
                var plain = await Eps();
                var mp = WanCompare(plain, WanNpz(fx, "eps_plain").Data);
                report.Add($"{tag}plain {mp.RelRms:E2}");
                ok &= mp.RelRms < 2e-3;
                foreach (var (name, setup) in cases)
                {
                    setup(true);
                    var got = await Eps();
                    setup(false);
                    var want = WanNpz(fx, "eps_" + name).Data;
                    var m = WanCompare(got, want);
                    double moved = WanCompare(want, plain).RelRms;
                    bool off = SdSame(plain, await Eps());
                    report.Add($"{tag}{name} {m.RelRms:E2} (maxAbs {m.MaxAbs:E2}; the reference is {moved:E2} from plain; off again: {(off ? "plain" : "DIFFERENT")})");
                    ok &= off && (tc ? m.RelRms < 0.25 * moved : m.RelRms < 2e-3 && moved > 20 * m.RelRms);
                }
            }
        }
        finally { Trellis2.DiTKernels.EnableTensorCores = saved; }
        SdLog("token reduction", report);
        if (!ok) throw new Exception("token reduction: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// What the token-reduction methods cost a picture and save in time: the 512 px DDIM-20 / guidance-7.5 picture of the sample fixture (ref\sample_sd15_ddim20_f16w.npz: its
    /// prompt and starting noise) made plain and with each method (random destinations from a fixed seed), PSNR against the plain picture and the wall time of the loop. A
    /// measurement (the ToDo paper's metric), logged to sd_tests.log and the pictures to out\; it fails only on a broken picture (PSNR under 15 dB). CUDA only (timings).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Sd15_TokenReduction_PictureQuality() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != global::ILGPU.Runtime.AcceleratorType.Cuda) throw new UnsupportedTestException("a timing measurement: CUDA only");
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sample_sd15_ddim20_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"fixture {fxPath} not found");
        using var fx = ZipFile.OpenRead(fxPath);
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions());
        var unet = ck.UNet!;
        int[] Ids(string k) => WanNpz(fx, k).Data.Select(v => (int)v).ToArray();
        var timesteps = Ids("timesteps"); var ac = WanNpz(fx, "alphas_cumprod").Data;
        var (noise, ns) = WanNpz(fx, "noise");
        int h = ns[1], w = ns[2];
        var methods = new (string Name, SdTokenMerge? Tm, SdTokenDownsample? Td)[]
        {
            ("plain", null, null),
            ("tome_q_0.3", new SdTokenMerge(0.3f, Rng: new Random(7)), null),
            ("tome_q_0.5", new SdTokenMerge(0.5f, Rng: new Random(7)), null),
            ("tome_qkv_0.5", new SdTokenMerge(0.5f, MergeKeysAndValues: true, Rng: new Random(7)), null),
            ("todo_2_nearest", null, new SdTokenDownsample(2)),
            ("todo_2_area", null, new SdTokenDownsample(2, 1, SdKvDownsampling.Area)),
            ("todo_2+2_area", null, new SdTokenDownsample(2, 2, SdKvDownsampling.Area)),
        };
        var report = new List<string>();
        float[]? plainImg = null; double plainS = 0, worst = double.PositiveInfinity;
        await SdDdimAsync(ck, Ids("ids"), null, Ids("uncond_ids"), null, timesteps, ac, noise, h, w, 7.5f, null);     // warm-up: kernels compiled, weights resident
        foreach (var (name, tm, td) in methods)
        {
            unet.TokenMerge = tm; unet.TokenDownsample = td;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (_, img) = await SdDdimAsync(ck, Ids("ids"), null, Ids("uncond_ids"), null, timesteps, ac, noise, h, w, 7.5f, null);
            double s = sw.Elapsed.TotalSeconds;
            unet.TokenMerge = null; unet.TokenDownsample = null;
            try { var outDir = Path.Combine(SdDir, "out"); Directory.CreateDirectory(outDir); await File.WriteAllBytesAsync(Path.Combine(outDir, $"tokens_{name}.png"), SdPng(img, h * 8, w * 8)); } catch (IOException) { }
            if (plainImg == null) { plainImg = img; plainS = s; report.Add($"plain {s:F2} s"); continue; }
            double mse = 0;
            for (int i = 0; i < img.Length; i++) { double d = (Math.Clamp(img[i], -1f, 1f) - Math.Clamp(plainImg[i], -1f, 1f)) / 2; mse += d * d; }
            double psnr = 10 * Math.Log10(1 / (mse / img.Length));
            worst = Math.Min(worst, psnr);
            report.Add($"{name} {s:F2} s ({plainS / s:F2}x) PSNR {psnr:F1} dB");
        }
        SdLog("token reduction quality 512", report);
        if (worst < 15) throw new Exception("a broken picture: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
