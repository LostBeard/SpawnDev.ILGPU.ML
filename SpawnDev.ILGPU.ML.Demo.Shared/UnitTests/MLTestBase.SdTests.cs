using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Stable Diffusion single-file checkpoints (SdCheckpoint, for SpawnGen's Load Checkpoint; Plans/sd-checkpoint-2026-10-25.md). Every stage is held to tensors dumped from the
/// REFERENCE (diffusers' from_single_file on the same .safetensors, float32 math on the CPU with the weights rounded to fp16, so the comparison measures our arithmetic and not
/// the weight rounding: tools/sd-harness/reference/ref_sd_dump.py --f16-weights / --f16-storage). Files in SD_DIR (default D:\Projects\sd):
/// <c>v1-5-pruned-emaonly.safetensors</c>, <c>sd_xl_base_1.0.safetensors</c>, <c>ref\sd_*.npz</c>, <c>tokenizer\clip-vit-large-patch14\tokenizer.json</c>; a missing file skips
/// the test. Desktop GPU lanes (the checkpoint is read from disk).
/// </summary>
public abstract partial class MLTestBase
{
    private static string SdDir => Environment.GetEnvironmentVariable("SD_DIR") ?? @"D:\Projects\sd";
    private const string Sd15File = "v1-5-pruned-emaonly.safetensors";
    private const string SdxlFile = "sd_xl_base_1.0.safetensors";

    private static string RequireSdFile(Accelerator accelerator, string file)
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the checkpoint is read from disk on a desktop GPU");
        var path = Path.Combine(SdDir, file);
        if (!File.Exists(path)) throw new UnsupportedTestException($"{file} not found in {SdDir} (set SD_DIR)");
        return path;
    }

    private static async Task<SdCheckpoint> SdLoad(Accelerator accelerator, string path, SdCheckpointOptions options)
    {
        await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        return await SdCheckpoint.LoadAsync(accelerator, fs, Path.GetFileName(path), options);
    }

    /// <summary>Our channel-last tap [H*W, C] against the reference's channel-first [C, H, W].</summary>
    private static (double RelRms, double MaxAbs, double RefMaxAbs) SdCompareTap(float[] gotHwc, float[] wantChw, int C, int HW)
    {
        var chw = new float[(long)C * HW];
        for (int p = 0; p < HW; p++) for (int c = 0; c < C; c++) chw[(long)c * HW + p] = gotHwc[(long)p * C + c];
        return WanCompare(chw, wantChw);
    }

    private static void SdLog(string what, IEnumerable<string> lines)
    {
        var all = lines.ToList();
        try
        {
            var dir = Path.Combine(SdDir, "out"); Directory.CreateDirectory(dir);
            File.AppendAllLines(Path.Combine(dir, "sd_tests.log"), all.Select(l => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [{what}] {l}"));
        }
        catch (IOException) { }
        foreach (var l in all) Console.WriteLine($"[Sd] {what}: {l}");
    }

    /// <summary>The library's CLIP BPE tokenizer (<c>BPETokenizer.EncodeCLIP</c>) against the reference tokenizer's ids for the dumped prompt. Host only.</summary>
    [TestMethod(Timeout = 120000, Category = "HeavyModel")]
    public async Task Sd_ClipTokenizer_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        var tokPath = Path.Combine(SdDir, "tokenizer", "clip-vit-large-patch14", "tokenizer.json");
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15_small_f16w.npz");
        if (!File.Exists(tokPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"SD fixtures not found in {SdDir} (set SD_DIR)");
        var tok = Preprocessing.BPETokenizer.LoadFromTokenizerJson(await File.ReadAllTextAsync(tokPath));
        using var fx = ZipFile.OpenRead(fxPath);
        var (want, _) = WanNpz(fx, "ids");
        var got = tok.EncodeCLIP("a photograph of an astronaut riding a horse on the moon, highly detailed");
        var w = want.Select(v => (int)v).ToArray();
        if (!got.SequenceEqual(w)) throw new Exception($"CLIP ids differ: ours [{string.Join(",", got.Take(20))}] reference [{string.Join(",", w.Take(20))}]");
        return $"77 ids identical ({w.Count(v => v != 49407) + 1} before padding)";
    });

    // -----------------------------------------------------------------------------------------------------------------------------------
    // text encoders
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>The CLIP ViT-L text encoder of the SD 1.5 single file against the reference: the last layer (ComfyUI default) and the penultimate layer with the final norm (clip skip 2).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_TextEncoder_MatchesReference() => await RunTest(accelerator => SdTextCheck(accelerator, Sd15File, SdFamily.SD1, "sd15_small_f16w", 5e-3));

    /// <summary>SDXL base's two text towers (CLIP ViT-L and OpenCLIP bigG, penultimate layers without the final norm, concatenated) and the pooled embedding against the reference.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sdxl_TextEncoder_MatchesReference() => await RunTest(accelerator => SdTextCheck(accelerator, SdxlFile, SdFamily.SDXL, "sdxl_small_f16s", 5e-3));

    private async Task<string> SdTextCheck(Accelerator accelerator, string file, SdFamily family, string tag, double bound)
    {
        var path = RequireSdFile(accelerator, file);
        var fxPath = Path.Combine(SdDir, "ref", $"sd_{tag}.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"SD fixture {fxPath} not found");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadUNet = false, LoadVae = false });
        if (ck.Family != family || ck.TextEncoder == null) throw new Exception($"family {ck.Family} (want {family}), text encoder {(ck.TextEncoder == null ? "missing" : "present")}");
        using var fx = ZipFile.OpenRead(fxPath);
        var ids = WanNpz(fx, "ids").Data.Select(v => (int)v).ToArray();
        int[]? ids2 = fx.GetEntry("ids2.npy") != null ? WanNpz(fx, "ids2").Data.Select(v => (int)v).ToArray() : null;
        var report = new List<string>();
        double worst = 0;
        var cases = family == SdFamily.SD1 ? new[] { ("ctx", (int?)-1), ("ctx_skip2", (int?)-2) } : new[] { ("ctx", (int?)null) };
        foreach (var (key, layer) in cases)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var e = await ck.TextEncoder.EncodeAsync(ids, ids2, layer);
            var m = WanCompare(e.Context, WanNpz(fx, key).Data);
            report.Add($"{key} (layer {layer?.ToString() ?? "default"}): relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:E2} of {m.RefMaxAbs:F2}; {sw.Elapsed.TotalMilliseconds:F0} ms");
            worst = Math.Max(worst, m.RelRms);
            if (family == SdFamily.SDXL)
            {
                if (e.Pooled == null) throw new Exception("SDXL returned no pooled embedding");
                var mp = WanCompare(e.Pooled, WanNpz(fx, "pooled").Data);
                report.Add($"pooled: relRMS {mp.RelRms:E2}, maxAbs {mp.MaxAbs:E2} of {mp.RefMaxAbs:F2}");
                worst = Math.Max(worst, mp.RelRms);
            }
        }
        SdLog($"{family} text encoder", report);
        if (!(worst < bound)) throw new Exception("the text encoder deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // UNets
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// The SD 1.5 UNet against the reference on identical latent / timestep / context, stage by stage (conv_in, the four down levels, mid, the four up levels, the output), at a small
    /// latent (32x40) and at production size (64x64, a 512x512 picture).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_UNet_MatchesReference() => await RunTest(accelerator => SdUNetCheck(accelerator, Sd15File, SdPrediction.Epsilon, new[] { "sd15_small_f16w", "sd15_512_f16w" }, 2e-2));

    /// <summary>The SDXL base UNet (pooled text + size conditioning, three levels, 2 / 10 transformer layers) against the reference, small latent and 1024x1024.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sdxl_UNet_MatchesReference() => await RunTest(accelerator => SdUNetCheck(accelerator, SdxlFile, SdPrediction.Epsilon, new[] { "sdxl_small_f16s", "sdxl_1024_f16s" }, 2e-2));

    private static readonly string[] SdTapNames = { "conv_in", "down_0", "down_1", "down_2", "down_3", "mid", "up_0", "up_1", "up_2", "up_3" };

    private async Task<string> SdUNetCheck(Accelerator accelerator, string file, SdPrediction prediction, string[] tags, double bound)
    {
        var path = RequireSdFile(accelerator, file);
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        if (ck.Prediction != prediction) throw new Exception($"expected a {prediction} model, inferred {ck.Prediction}");
        var report = new List<string>();
        double worstEps = 0, worstTap = 0;
        int ran = 0;
        foreach (var tag in tags)
        {
            var fxPath = Path.Combine(SdDir, "ref", $"sd_{tag}.npz");
            if (!File.Exists(fxPath)) { report.Add($"{tag}: fixture missing, skipped"); continue; }
            ran++;
            using var fx = ZipFile.OpenRead(fxPath);
            var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
            int h = ls[1], w = ls[2];
            SdxlConditioning? xl = null;
            if (unet.IsXL)
            {
                var ti = WanNpz(fx, "time_ids").Data;
                xl = new SdxlConditioning(WanNpz(fx, "pooled").Data, (int)ti[0], (int)ti[1], (int)ti[2], (int)ti[3], (int)ti[4], (int)ti[5]);
            }
            var taps = new Dictionary<string, (float[] Data, int C, int HW)>();
            unet.Tap = (name, view, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = (a, C, H * W); };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var eps = await unet.ForwardAsync(lat, h, w, t, ctx, cs[0], xl);
            double ms = sw.Elapsed.TotalMilliseconds;
            unet.Tap = null;
            var line = new List<string>();
            foreach (var name in SdTapNames)
            {
                if (fx.GetEntry(name + ".npy") == null) continue;
                if (!taps.TryGetValue(name, out var tp)) throw new Exception($"tap {name} was not produced");
                var m = SdCompareTap(tp.Data, WanNpz(fx, name).Data, tp.C, tp.HW);
                line.Add($"{name} {m.RelRms:E1}");
                worstTap = Math.Max(worstTap, m.RelRms);
            }
            var me = WanCompare(eps, WanNpz(fx, "eps").Data);
            worstEps = Math.Max(worstEps, me.RelRms);
            report.Add($"{tag} ({w * 8}x{h * 8}, t={t}): eps relRMS {me.RelRms:E2} maxAbs {me.MaxAbs:E2} of {me.RefMaxAbs:F2}, {ms:F0} ms (first call incl. kernel compile); taps: {string.Join(", ", line)}");
            sw.Restart();
            await unet.ForwardAsync(lat, h, w, t, ctx, cs[0], xl);
            report.Add($"{tag}: warm forward {sw.Elapsed.TotalMilliseconds:F0} ms");
        }
        SdLog($"{ck.Family} UNet", report);
        if (ran == 0) throw new UnsupportedTestException("no UNet fixture found");
        if (!(worstEps < bound && worstTap < bound)) throw new Exception("the UNet deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // VAEs
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>The SD 1.5 VAE against the reference: decode (stage by stage) at 256x320 and 512x512, encode (posterior mean) at both sizes.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_Vae_MatchesReference() => await RunTest(accelerator => SdVaeCheck(accelerator, Sd15File, 0.18215f, new[] { "sd15_small_f16w", "sd15_512_f16w" }, 5e-3));

    /// <summary>The SDXL VAE (fp32 activations: it overflows fp16) against the reference: decode and encode, small and 1024x1024.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sdxl_Vae_MatchesReference() => await RunTest(accelerator => SdVaeCheck(accelerator, SdxlFile, 0.13025f, new[] { "sdxl_small_f16s", "sdxl_1024_f16s" }, 5e-3));

    private async Task<string> SdVaeCheck(Accelerator accelerator, string file, float scale, string[] tags, double bound)
    {
        var path = RequireSdFile(accelerator, file);
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadUNet = false });
        var vae = ck.Vae!;
        if (Math.Abs(ck.LatentScale - scale) > 1e-7) throw new Exception($"latent scale {ck.LatentScale}, want {scale}");
        var report = new List<string>();
        double worst = 0;
        int ran = 0;
        foreach (var tag in tags)
        {
            var fxPath = Path.Combine(SdDir, "ref", $"sd_{tag}.npz");
            if (!File.Exists(fxPath)) { report.Add($"{tag}: fixture missing, skipped"); continue; }
            ran++;
            using var fx = ZipFile.OpenRead(fxPath);
            var (z, zs) = WanNpz(fx, "dec_in");
            int h = zs[1], w = zs[2];
            var taps = new Dictionary<string, (float[] Data, int C, int HW)>();
            vae.Tap = (name, view, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = (a, C, H * W); };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var img = await vae.DecodeAsync(z, h, w);
            double ms = sw.Elapsed.TotalMilliseconds;
            vae.Tap = null;
            var line = new List<string>();
            foreach (var name in new[] { "post_quant_conv", "dec_mid", "dec_up_0", "dec_up_1", "dec_up_2", "dec_up_3" })
            {
                if (!taps.TryGetValue(name, out var tp)) throw new Exception($"tap {name} was not produced");
                var m = SdCompareTap(tp.Data, WanNpz(fx, name).Data, tp.C, tp.HW);
                line.Add($"{name} {m.RelRms:E1}");
                worst = Math.Max(worst, m.RelRms);
            }
            var mi = WanCompare(img, WanNpz(fx, "image").Data);
            worst = Math.Max(worst, mi.RelRms);
            report.Add($"decode {tag} ({w * 8}x{h * 8}): image relRMS {mi.RelRms:E2} maxAbs {mi.MaxAbs:E2}, {ms:F0} ms; taps: {string.Join(", ", line)}");
            var (im, ims) = WanNpz(fx, "enc_image");
            sw.Restart();
            var mean = await vae.EncodeAsync(im, ims[1], ims[2]);
            var me = WanCompare(mean, WanNpz(fx, "enc_mean").Data);
            worst = Math.Max(worst, me.RelRms);
            report.Add($"encode {tag} ({ims[2]}x{ims[1]}): mean relRMS {me.RelRms:E2} maxAbs {me.MaxAbs:E2}, {sw.Elapsed.TotalMilliseconds:F0} ms");
        }
        SdLog($"{ck.Family} VAE", report);
        if (ran == 0) throw new UnsupportedTestException("no VAE fixture found");
        if (!(worst < bound)) throw new Exception("the VAE deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }
}
