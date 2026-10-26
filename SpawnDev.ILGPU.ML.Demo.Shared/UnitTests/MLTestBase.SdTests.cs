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
    private const string Sd21vFile = "v2-1_768-ema-pruned.safetensors", Sd21BaseFile = "v2-1_512-ema-pruned.safetensors";

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

    /// <summary>
    /// The GEMM wrappers with an ODD output width and a bias (the SD VAE's conv_out has 3 channels): the tiled GEMM reads fp16 / bf16 biases as 32-bit words, and an
    /// odd-length bias used to lose its last element (read past the binding: wrong on WebGPU, silently right elsewhere). fp16 Linear (N = 3 and 5), the exact
    /// gathered convolution (cout = 3) and the bf16 Linear (N = 3) against a host reference on the same fp16 / bf16-rounded operands. Light, every backend.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task Sd_OddWidthBias_MatchesHost() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("the tiled GEMM needs shared memory");
        var rng = new Random(5);
        float R() => (float)(rng.NextDouble() * 2 - 1);
        var report = new List<string>();
        double worst = 0;
        var sk = new Trellis2.SparseKernels(accelerator);
        foreach (int N in new[] { 3, 5 })
        {
            const int M = 70, K = 64;
            var x = Enumerable.Range(0, M * K).Select(_ => R()).ToArray();
            var w = Enumerable.Range(0, N * K).Select(_ => (global::ILGPU.Half)R()).ToArray();
            var b = Enumerable.Range(0, N).Select(_ => (global::ILGPU.Half)(R() * 4)).ToArray();
            using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D(w); using var bb = accelerator.Allocate1D(b); using var ob = accelerator.Allocate1D<float>(M * N);
            sk.Linear(xb.View, wb.View, bb.View, ob.View, M, K, N);
            var got = await ob.View.CopyToHostAsync();
            double err = 0;
            for (int m = 0; m < M; m++) for (int n = 0; n < N; n++)
            {
                double acc = (float)b[n]; for (int k = 0; k < K; k++) acc += x[m * K + k] * (float)w[n * K + k];
                err = Math.Max(err, Math.Abs(got[m * N + n] - acc));
            }
            report.Add($"fp16 Linear N={N}: maxErr {err:E1}"); worst = Math.Max(worst, err);
        }
        {
            // exact gathered 3x3 convolution, cin 16 -> cout 3 on an 8x8 picture (weights already in the gather layout [cout][taps][cin])
            int H = 8, W = 8, cin = 16, cout = 3, M = H * W;
            var sd = new StableDiffusion.SdKernels(accelerator);
            var x = Enumerable.Range(0, M * cin).Select(_ => R()).ToArray();
            var w = Enumerable.Range(0, cout * 9 * cin).Select(_ => (global::ILGPU.Half)R()).ToArray();
            var b = Enumerable.Range(0, cout).Select(_ => (global::ILGPU.Half)(R() * 4)).ToArray();
            using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D(w); using var bb = accelerator.Allocate1D(b); using var ob = accelerator.Allocate1D<float>(M * cout);
            using var tb = accelerator.Allocate1D<int>(M * 27);
            sd.BuildTable(tb.View, H, W, H, W, 3, 1, 1, 1);
            sd.Conv(xb.View, tb.View, wb.View, bb.View, ob.View, M, cin, cout, 9, exact: true);
            var got = await ob.View.CopyToHostAsync();
            double err = 0;
            for (int oy = 0; oy < H; oy++) for (int ox = 0; ox < W; ox++) for (int co = 0; co < cout; co++)
            {
                double acc = (float)b[co];
                for (int t = 0; t < 9; t++) { int iy = oy - 1 + t / 3, ix = ox - 1 + t % 3; if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue; for (int c = 0; c < cin; c++) acc += x[(iy * W + ix) * cin + c] * (float)w[(co * 9 + t) * cin + c]; }
                err = Math.Max(err, Math.Abs(got[(oy * W + ox) * cout + co] - acc));
            }
            report.Add($"exact conv cout=3: maxErr {err:E1}"); worst = Math.Max(worst, err);
        }
        {
            const int M = 70, K = 64, N = 3;
            var dk = new Trellis2.DiTKernels(accelerator);
            var x = Enumerable.Range(0, M * K).Select(_ => R()).ToArray();
            var w = Enumerable.Range(0, N * K).Select(_ => (BFloat16)R()).ToArray();
            var b = Enumerable.Range(0, N).Select(_ => (BFloat16)(R() * 4)).ToArray();
            using var xb = accelerator.Allocate1D(x); using var wb = accelerator.Allocate1D(w); using var bb = accelerator.Allocate1D(b); using var ob = accelerator.Allocate1D<float>(M * N);
            dk.Linear(xb.View, wb.View, bb.View, ob.View, M, K, N);
            var got = await ob.View.CopyToHostAsync();
            double err = 0;
            for (int m = 0; m < M; m++) for (int n = 0; n < N; n++)
            {
                double acc = (float)b[n]; for (int k = 0; k < K; k++) acc += x[m * K + k] * (float)w[n * K + k];
                err = Math.Max(err, Math.Abs(got[m * N + n] - acc));
            }
            report.Add($"bf16 Linear N=3: maxErr {err:E1}"); worst = Math.Max(worst, err);      // N < 64: never the tensor-core kernel, the tiled one on every backend
        }
        var line = string.Join("; ", report);
        if (!(worst < 1e-3)) throw new Exception("odd-width bias GEMM deviates: " + line);
        return line;
    });

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

    /// <summary>SD 2.1's OpenCLIP ViT-H (in_proj fused in the file, the penultimate of 24 layers through the final norm: ComfyUI's and diffusers' default) against the reference.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd21_TextEncoder_MatchesReference() => await RunTest(accelerator => SdTextCheck(accelerator, Sd21vFile, SdFamily.SD2, "sd21v_small_f16s", 5e-3));

    /// <summary>
    /// SD-Turbo (<c>sd_turbo.safetensors</c>, Stability's sgm layout: the OpenCLIP-H tower under <c>conditioner.embedders.0.model.*</c>, which a refiner uses for bigG) must load as
    /// SD 2.x (Lore's 10-25 report: it was rejected as a refiner) and its text encoder match the reference.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SdTurbo_TextEncoder_MatchesReference() => await RunTest(accelerator => SdTextCheck(accelerator, SdTurboFile, SdFamily.SD2, "sdturbo_512_f16s", 5e-3));

    /// <summary>The SD-Turbo UNet (sgm-layout SD 2.x) at 512x512, t = 999 (its one-step timestep), against the reference.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task SdTurbo_UNet_MatchesReference() => await RunTest(accelerator => SdUNetCheck(accelerator, SdTurboFile, SdTurboPrediction, new[] { "sdturbo_512_f16s" }, 2e-2));

    private const string SdTurboFile = "sd_turbo.safetensors";
    private const SdPrediction SdTurboPrediction = SdPrediction.Epsilon;     // stabilityai/sd-turbo's scheduler config: epsilon (the V heuristic must not fire)

    private async Task<string> SdTextCheck(Accelerator accelerator, string file, SdFamily family, string tag, double bound)
    {
        var path = RequireSdFile(accelerator, file);
        var fxPath = Path.Combine(SdDir, "ref", $"sd_{tag}.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"SD fixture {fxPath} not found");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadUNet = false, LoadVae = false });
        if (ck.Family != family || ck.TextEncoder == null) throw new Exception($"family {ck.Family} (want {family}), text encoder {(ck.TextEncoder == null ? "missing" : "present")}");
        var report = new List<string>();
        double worst = await SdTextVsFixture(ck.TextEncoder, family, fxPath, report);
        SdLog($"{family} text encoder", report);
        if (!(worst < bound)) throw new Exception("the text encoder deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    private static async Task<double> SdTextVsFixture(SdTextEncoder te, SdFamily family, string fxPath, List<string> report)
    {
        using var fx = ZipFile.OpenRead(fxPath);
        var ids = WanNpz(fx, "ids").Data.Select(v => (int)v).ToArray();
        int[]? ids2 = fx.GetEntry("ids2.npy") != null ? WanNpz(fx, "ids2").Data.Select(v => (int)v).ToArray() : null;
        double worst = 0;
        var cases = family == SdFamily.SD1 ? new[] { ("ctx", (int?)-1), ("ctx_skip2", (int?)-2) } : new[] { ("ctx", (int?)null) };      // SD 2.x / SDXL: the default layer
        foreach (var (key, layer) in cases)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var e = await te.EncodeAsync(ids, ids2, layer);
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
        return worst;
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

    /// <summary>The SD 2.1 768-v UNet (64-wide heads, linear projections, 1024-wide context): the loader must infer V prediction (ComfyUI's norm-bias test); 32x40 and 96x96.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd21v_UNet_MatchesReference() => await RunTest(accelerator => SdUNetCheck(accelerator, Sd21vFile, SdPrediction.V, new[] { "sd21v_small_f16s", "sd21v_768_f16s" }, 2e-2));

    /// <summary>The SD 2.1 base (512) UNet: same layout, an epsilon model; the loader must NOT infer V.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd21Base_UNet_MatchesReference() => await RunTest(accelerator => SdUNetCheck(accelerator, Sd21BaseFile, SdPrediction.Epsilon, new[] { "sd21b_small_f16s" }, 2e-2));

    /// <summary>The SD 2.1 VAE (the SD 1.x layout) decode and encode against the reference.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd21_Vae_MatchesReference() => await RunTest(accelerator => SdVaeCheck(accelerator, Sd21vFile, 0.18215f, new[] { "sd21v_small_f16s" }, 5e-3));

    /// <summary>
    /// LoRA merge on the SD 1.5 UNet: LCM-LoRA (kohya keys; every attention, feed-forward, projection and resnet convolution incl. the 3x3 ones, which sit in the gather layout,
    /// and the fused q|k|v / k|v rows) at strength 0.8 against diffusers' load_lora_weights + fuse_lora; then an empty set must restore the plain model bit for bit (the
    /// originals are kept), and the plain fixture must pass again.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_Lora_MatchesReference() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var loraPath = Path.Combine(SdDir, "lora", "lcm-lora-sdv1-5.safetensors");
        var fxLora = Path.Combine(SdDir, "ref", "sd_sd15lcm_small_f16s.npz"); var fxPlain = Path.Combine(SdDir, "ref", "sd_sd15_small_f16w.npz");
        if (!File.Exists(loraPath) || !File.Exists(fxLora) || !File.Exists(fxPlain)) throw new UnsupportedTestException($"LoRA fixtures not found in {SdDir}");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadVae = false });
        async Task<(double Rel, float[] Eps)> Eps(string fxPath)
        {
            using var fx = ZipFile.OpenRead(fxPath);
            var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
            var eps = await ck.UNet!.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            return (WanCompare(eps, WanNpz(fx, "eps").Data).RelRms, eps);
        }
        var report = new List<string>();
        var (plain0, eps0) = await Eps(fxPlain);
        var spec = new SdLoraSpec(ct => Task.FromResult<Stream>(new FileStream(loraPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true)), "lcm-lora-sdv1-5", 0.8f, 1f);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var rep = await ck.ApplyLorasAsync(new[] { spec });
        double applyMs = sw.Elapsed.TotalMilliseconds;
        var (patched, unmatched) = (rep.Loras[0].Patched, rep.Loras[0].Unmatched);
        var (withLora, _) = await Eps(fxLora);
        var (plainVsLora, _) = await Eps(fxPlain);
        await ck.ApplyLorasAsync(Array.Empty<SdLoraSpec>());
        var (plain1, eps1) = await Eps(fxPlain);
        bool identical = eps0.AsSpan().SequenceEqual(eps1);
        report.Add($"LCM-LoRA x0.8: {patched} modules patched in {applyMs:F0} ms, {unmatched.Count} unmatched{(unmatched.Count > 0 ? " (" + string.Join(", ", unmatched.Take(5)) + ")" : "")}");
        report.Add($"eps vs the LoRA reference relRMS {withLora:E2} (the plain reference is {plainVsLora:E2} away: the LoRA moves the model); removed: plain relRMS {plain1:E2} (before {plain0:E2}), restored output {(identical ? "bit-identical" : "DIFFERENT")}");
        SdLog("SD1.5 LoRA", report);
        if (!(withLora < 2e-2 && unmatched.Count == 0 && patched > 0 && identical && plainVsLora > 10 * withLora)) throw new Exception("LoRA merge failed: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// Standalone components (Lore's 10-25 asks): a UNet loaded ALONE (SdUNet.LoadAsync on a whole checkpoint takes only its UNet) infers its prediction type
    /// (SD 2.1 768-v = V, SD 2.1 base = epsilon), and LoRA merges into a standalone UNet (LCM-LoRA x0.8 vs the reference, then removed bit-identically).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd_StandaloneComponents_PredictionAndLora() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, Sd15File);
        FileStream Open(string f) => new(f, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var report = new List<string>();
        foreach (var (file, want) in new[] { (Sd21vFile, SdPrediction.V), (Sd21BaseFile, SdPrediction.Epsilon) })
        {
            var path = Path.Combine(SdDir, file);
            if (!File.Exists(path)) throw new UnsupportedTestException($"{file} not found");
            await using var s = Open(path);
            using var u = await SdUNet.LoadAsync(accelerator, s, file);
            var got = await u.InferPredictionAsync();
            report.Add($"{file} alone: {u.Family}, {got}");
            if (got != want || u.Family != SdFamily.SD2) throw new Exception($"{file}: inferred {u.Family} / {got}, want SD2 / {want}");
        }
        var loraPath = Path.Combine(SdDir, "lora", "lcm-lora-sdv1-5.safetensors");
        var fxLora = Path.Combine(SdDir, "ref", "sd_sd15lcm_small_f16s.npz");
        if (!File.Exists(loraPath) || !File.Exists(fxLora)) throw new UnsupportedTestException("LoRA fixtures not found");
        await using var us = Open(Path.Combine(SdDir, Sd15File));
        using var unet = await SdUNet.LoadAsync(accelerator, us, Sd15File);
        using var fx = ZipFile.OpenRead(fxLora);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        var plain = await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var rep = await unet.ApplyLorasAsync(new[] { new SdLoraSpec(ct => Task.FromResult<Stream>(Open(loraPath)), "lcm", 0.8f) });
        var withLora = await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var m = WanCompare(withLora, WanNpz(fx, "eps").Data);
        await unet.ApplyLorasAsync(Array.Empty<SdLoraSpec>());
        var back = await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        bool identical = plain.AsSpan().SequenceEqual(back);
        report.Add($"standalone SD 1.5 UNet + LCM-LoRA x0.8: {rep.Loras[0].Patched} modules, {rep.Loras[0].Unmatched.Count} unmatched, eps vs reference {m.RelRms:E2}, removed: {(identical ? "bit-identical" : "DIFFERENT")}");
        SdLog("standalone components", report);
        if (!(m.RelRms < 2e-2 && rep.Loras[0].Patched == 278 && rep.Loras[0].Unmatched.Count == 0 && identical)) throw new Exception("standalone LoRA failed: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// TAESD / TAESDXL (SdTaesd: the original <c>taesd_decoder / taesd_encoder.safetensors</c>, and the diffusers AutoencoderTiny file with both halves) against diffusers'
    /// AutoencoderTiny: decode of an unscaled latent to [-1, 1] and encode back to the unscaled latent, as SdVae does. Files in SD_DIR\taesd, fixtures from
    /// tools/sd-harness/reference/ref_taesd_dump.py.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Sd_Taesd_MatchesReference() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, Sd15File);
        string dir = Path.Combine(SdDir, "taesd");
        FileStream Open(string f) => new(Path.Combine(dir, f), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var report = new List<string>();
        double worst = 0;
        foreach (var (label, dec, enc, scale, tag) in new (string, string, string?, float, string)[]
        {
            ("TAESD (original files)", "taesd_decoder.safetensors", "taesd_encoder.safetensors", 0.18215f, "taesd"),
            ("TAESD (diffusers file)", Path.Combine("diffusers", "diffusion_pytorch_model.safetensors"), null, 0.18215f, "taesd"),
            // the published taesdxl_decoder.safetensors is NOT the diffusers TAESDXL decoder (same encoder): each has its own reference
            ("TAESDXL (original files)", "taesdxl_decoder.safetensors", "taesdxl_encoder.safetensors", 0.13025f, "taesdxl_orig"),
            ("TAESDXL (diffusers file)", Path.Combine("diffusers_xl", "diffusion_pytorch_model.safetensors"), null, 0.13025f, "taesdxl"),
        })
        {
            var fxPath = Path.Combine(SdDir, "ref", $"taesd_{tag}.npz");
            if (!File.Exists(Path.Combine(dir, dec)) || !File.Exists(fxPath)) throw new UnsupportedTestException($"TAESD files not found in {dir}");
            await using var ds = Open(dec);
            await using var es = enc != null ? Open(enc) : null;
            using var t = await SdTaesd.LoadAsync(accelerator, ds, dec, es, enc, scale);
            using var fx = ZipFile.OpenRead(fxPath);
            var (z, zs) = WanNpz(fx, "dec_in");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var img = await t.DecodeAsync(z, zs[1], zs[2]);
            double ms = sw.Elapsed.TotalMilliseconds;
            var md = WanCompare(img, WanNpz(fx, "image").Data);
            var (im, ims) = WanNpz(fx, "enc_image");
            var lat = await t.EncodeAsync(im, ims[1], ims[2]);
            var me = WanCompare(lat, WanNpz(fx, "enc_out").Data);
            report.Add($"{label}: decode {zs[2] * 8}x{zs[1] * 8} relRMS {md.RelRms:E2} ({ms:F0} ms first call), encode relRMS {me.RelRms:E2}");
            worst = Math.Max(worst, Math.Max(md.RelRms, me.RelRms));
        }
        SdLog("TAESD", report);
        if (!(worst < 2e-2)) throw new Exception("TAESD deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// SD 1.5 ControlNet (ComfyUI's single file <c>control_v11p_sd15_canny_fp16.safetensors</c>) against diffusers' ControlNetModel.from_single_file: the 13 residuals for a
    /// line-drawing hint, then the UNet with them at strength 0.8. Fixture from tools/sd-harness/reference/ref_sd_control.py; files in SD_DIR\controlnet.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_ControlNet_MatchesReference() => await RunTest(async accelerator =>
    {
        var ckPath = RequireSdFile(accelerator, Sd15File);
        var cnPath = Path.Combine(SdDir, "controlnet", "control_v11p_sd15_canny_fp16.safetensors");
        var fxPath = Path.Combine(SdDir, "ref", "control_canny_small.npz");
        if (!File.Exists(cnPath) || !File.Exists(fxPath)) throw new UnsupportedTestException("ControlNet files not found");
        using var ck = await SdLoad(accelerator, ckPath, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        await using var cs = new FileStream(cnPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var cn = await SdControlNet.LoadAsync(accelerator, cs, "canny");
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, ctxs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        var hint = WanNpz(fx, "hint").Data; float scale = WanNpz(fx, "scale").Data[0];
        int h = ls[1], w = ls[2];
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var res = await cn.RunAsync(lat, h, w, t, ctx, ctxs[0], hint);
        await accelerator.SynchronizeAsync();
        double cnMs = sw.Elapsed.TotalMilliseconds;
        var report = new List<string>();
        double worstRes = 0;
        if (res.Count != 13) throw new Exception($"{res.Count} residuals, SD 1.5 takes 13");
        var line = new List<string>();
        for (int i = 0; i < res.Count; i++)
        {
            var (want, wsh) = WanNpz(fx, $"res_{i}");
            int C = wsh[0], HW = wsh[1] * wsh[2];
            var got = await res[i].SubView(0, (long)C * HW).CopyToHostAsync();
            var m = SdCompareTap(got, want, C, HW);
            line.Add($"{i}:{m.RelRms:E1}");
            worstRes = Math.Max(worstRes, m.RelRms);
        }
        report.Add($"residuals ({cnMs:F0} ms first call): {string.Join(" ", line)}");
        res.Strength = scale;
        var eps = await ck.UNet!.ForwardAsync(lat, h, w, t, ctx, ctxs[0], null, new[] { res });
        var me = WanCompare(eps, WanNpz(fx, "eps_control").Data);
        var plainVsControl = WanCompare(WanNpz(fx, "eps_plain").Data, WanNpz(fx, "eps_control").Data).RelRms;
        report.Add($"UNet + ControlNet x{scale}: eps relRMS {me.RelRms:E2} (the plain UNet is {plainVsControl:E2} away)");
        SdLog("SD1.5 ControlNet", report);
        if (!(worstRes < 2e-2 && me.RelRms < 2e-2 && plainVsControl > 10 * me.RelRms)) throw new Exception("the ControlNet deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    private static readonly string[] SdTapNames = { "conv_in", "down_0", "down_1", "down_2", "down_3", "mid", "up_0", "up_1", "up_2", "up_3" };

    private async Task<string> SdUNetCheck(Accelerator accelerator, string file, SdPrediction prediction, string[] tags, double bound)
    {
        var path = RequireSdFile(accelerator, file);
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        if (ck.Prediction != prediction) throw new Exception($"expected a {prediction} model, inferred {ck.Prediction}");
        var report = new List<string>();
        var (worstEps, worstTap, ran) = await SdUNetVsFixtures(accelerator, unet, tags, report);
        SdLog($"{ck.Family} UNet", report);
        if (ran == 0) throw new UnsupportedTestException("no UNet fixture found");
        if (!(worstEps < bound && worstTap < bound)) throw new Exception("the UNet deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    private static async Task<(double Eps, double Tap, int Ran)> SdUNetVsFixtures(Accelerator accelerator, SdUNet unet, string[] tags, List<string> report)
    {
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
        return (worstEps, worstTap, ran);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // a whole picture
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// A whole SD 1.5 picture against the reference's (tools/sd-harness/reference/ref_sd_sample.py): text encoder (prompt and the empty negative), 20 DDIM steps (eta 0) with
    /// guidance 7.5 run on the host around our UNet exactly as SpawnGen drives it, from the reference's own starting noise, then the VAE decode. Every step's latent and the
    /// picture are compared; our picture goes to SD_DIR\out\sd15_ddim20_ours.png to look at.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_Ddim20_MatchesReference() => await RunTest(accelerator => SdSampleCheck(accelerator, Sd15File, "sd15_ddim20_f16w", 7.5f));

    private async Task<string> SdSampleCheck(Accelerator accelerator, string file, string tag, float cfg)
    {
        var path = RequireSdFile(accelerator, file);
        var fxPath = Path.Combine(SdDir, "ref", $"sample_{tag}.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"fixture {fxPath} not found");
        using var fx = ZipFile.OpenRead(fxPath);
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions());
        int[] Ids(string k) => WanNpz(fx, k).Data.Select(v => (int)v).ToArray();
        int[]? Opt(string k) => fx.GetEntry(k + ".npy") != null ? Ids(k) : null;
        var timesteps = WanNpz(fx, "timesteps").Data; var ac = WanNpz(fx, "alphas_cumprod").Data;
        var (lats, lsh) = WanNpz(fx, "latents");                    // [steps + 1, 4, h, w]
        int steps = timesteps.Length, h = lsh[2], w = lsh[3], n = 4 * h * w;
        if (lsh[0] != steps + 1) throw new InvalidDataException("latents must hold steps + 1 entries");
        var report = new List<string>();
        double worstStep = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (x, img) = await SdDdimAsync(ck, Ids("ids"), Opt("ids2"), Ids("uncond_ids"), Opt("uncond_ids2"), timesteps.Select(v => (int)v).ToArray(), ac, lats.AsSpan(0, n).ToArray(), h, w, cfg,
            (i, t, lat) =>
            {
                var m = WanCompare(lat, lats.AsSpan((i + 1) * n, n).ToArray());
                worstStep = Math.Max(worstStep, m.RelRms);
                if (i % 5 == 4 || i == steps - 1) report.Add($"step {i + 1} (t={t}): latent relRMS {m.RelRms:E2}");
            });
        double loopS = sw.Elapsed.TotalSeconds;
        var mi = WanCompare(img, WanNpz(fx, "image").Data);
        report.Add($"picture {w * 8}x{h * 8}: relRMS {mi.RelRms:E2} maxAbs {mi.MaxAbs:E2}; text + {steps} steps x 2 UNet calls + decode in {loopS:F1} s");
        try
        {
            var outDir = Path.Combine(SdDir, "out"); Directory.CreateDirectory(outDir);
            await File.WriteAllBytesAsync(Path.Combine(outDir, $"{tag}_ours_{accelerator.AcceleratorType}.png"), SdPng(img, h * 8, w * 8));
        }
        catch (IOException) { }
        SdLog($"{ck.Family} sample {tag}", report);
        // 20 steps of guidance 7.5 amplify a per-call difference; the bound is on the picture a user sees
        if (!(mi.RelRms < 2e-2 && worstStep < 2e-2)) throw new Exception("the picture deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    private static byte[] SdPng(float[] img, int H, int W)
    {
        var rgba = new byte[H * W * 4];
        for (int p = 0; p < H * W; p++) { for (int c = 0; c < 3; c++) rgba[p * 4 + c] = (byte)Math.Clamp(Math.Round((img[(long)c * H * W + p] + 1) * 127.5), 0, 255); rgba[p * 4 + 3] = 255; }
        return Assets.AssetGenerator.EncodePng(rgba, W, H);
    }

    /// <summary>
    /// Text (prompt and negative), <paramref name="timesteps"/>.Length DDIM steps (eta 0, diffusers DDIMScheduler with set_alpha_to_one false) with classifier-free guidance
    /// <paramref name="cfg"/> on the host around the UNet, as SpawnGen drives it, then the VAE decode. <paramref name="onStep"/> sees every step's latent. Returns the final
    /// (scaled) latent and the picture [3, 8h, 8w].
    /// </summary>
    private static async Task<(float[] Latent, float[] Image)> SdDdimAsync(SdCheckpoint ck, int[] ids, int[]? ids2, int[] uncondIds, int[]? uncondIds2, int[] timesteps, float[] ac,
        float[] noise, int h, int w, float cfg, Action<int, int, float[]>? onStep)
    {
        var cond = await ck.TextEncoder!.EncodeAsync(ids, ids2);
        var uncond = await ck.TextEncoder.EncodeAsync(uncondIds, uncondIds2);
        SdxlConditioning? xlC = null, xlU = null;
        if (ck.UNet!.IsXL)
        {
            xlC = new SdxlConditioning(cond.Pooled!, h * 8, w * 8, 0, 0, h * 8, w * 8);
            xlU = new SdxlConditioning(uncond.Pooled!, h * 8, w * 8, 0, 0, h * 8, w * 8);
        }
        var x = (float[])noise.Clone();
        int n = x.Length, stride = 1000 / timesteps.Length;
        for (int i = 0; i < timesteps.Length; i++)
        {
            int t = timesteps[i];
            var ec = await ck.UNet.ForwardAsync(x, h, w, t, cond.Context, cond.Rows, xlC);
            var eu = await ck.UNet.ForwardAsync(x, h, w, t, uncond.Context, uncond.Rows, xlU);
            int tp = t - stride;
            double at = ac[t], ap = tp >= 0 ? ac[tp] : ac[0];
            for (int k = 0; k < n; k++)
            {
                double e = eu[k] + cfg * (ec[k] - eu[k]);
                double x0 = (x[k] - Math.Sqrt(1 - at) * e) / Math.Sqrt(at);
                x[k] = (float)(Math.Sqrt(ap) * x0 + Math.Sqrt(1 - ap) * e);
            }
            onStep?.Invoke(i, t, x);
        }
        var img = await ck.Vae!.DecodeAsync(x.Select(v => v / ck.LatentScale).ToArray(), h, w);
        return (x, img);
    }

    private const string SdServer = "http://127.0.0.1:8769/";

    /// <summary>The local SD file server as a model source: files stream by Range requests that stay in JS (<see cref="Hub.FetchRangeStream"/>).</summary>
    private sealed class SdRangeSource(HttpClient http) : Hub.IModelSource
    {
        public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            using var head = await http.SendAsync(new HttpRequestMessage(HttpMethod.Head, SdServer + filePath), cancellationToken);
            head.EnsureSuccessStatusCode();
            return new Hub.FetchRangeStream(SdServer + filePath, head.Content.Headers.ContentLength ?? throw new InvalidDataException($"{filePath}: the server gave no length"));
        }
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => http.GetByteArrayAsync(SdServer + filePath, cancellationToken);
    }

    // relRMS of a channel-last device tensor [HW, C] against a channel-first reference [C, HW] ON THE DEVICE: chunked partial sums of (got - want)^2 and want^2,
    // only the partials (a few thousand floats) come back
    private static void SdCmpPartialImpl(Index1D chunk, ArrayView1D<float, Stride1D.Dense> got, ArrayView1D<float, Stride1D.Dense> want, ArrayView1D<float, Stride1D.Dense> partial,
        int C, int HW, int per, int channelLast)
    {
        int n = C * HW, i0 = (int)chunk * per, i1 = Math.Min(n, i0 + per);     // 32-bit indices: the largest tap is 33.5M elements (no emulated i64 on WebGPU)
        float se = 0f, sr = 0f;
        for (int i = i0; i < i1; i++)
        {
            int c = i / HW, p = i % HW;                    // reference index i = c * HW + p
            float g = channelLast != 0 ? got[p * C + c] : got[i];
            float d = g - want[i];
            se += d * d; sr += want[i] * want[i];
        }
        partial[2 * chunk] = se; partial[2 * chunk + 1] = sr;
    }

    private static async Task<double> SdDeviceRelRms(Accelerator acc, ArrayView1D<float, Stride1D.Dense> got, ArrayView1D<float, Stride1D.Dense> want, int C, int HW, bool channelLast)
    {
        const int per = 4096;
        long n = (long)C * HW;
        int chunks = (int)((n + per - 1) / per);
        using var partial = acc.Allocate1D<float>(2L * chunks);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int>(SdCmpPartialImpl);
        k(chunks, got, want, partial.View, C, HW, per, channelLast ? 1 : 0);
        await acc.SynchronizeAsync();
        var p = await partial.View.CopyToHostAsync();
        double se = 0, sr = 0;
        for (int i = 0; i < chunks; i++) { se += p[2 * i]; sr += p[2 * i + 1]; }
        return Math.Sqrt(se / Math.Max(1e-30, sr));
    }

    /// <summary>
    /// The SD 1.5 VAE decode and encode in the BROWSER at 512x512, stage by stage: the reference taps stream from the file server straight onto the GPU and every comparison
    /// runs on the device (only the partial sums come back). Localises a backend-only deviation. Same file server as the picture test (D:/Projects/sd on 8769); fixture
    /// files from tools/sd-harness/reference/export_browser_taps.py.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_Vae_OverHttp_StagesMatchReference() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser VAE stage test (the desktop runs Sd15_Vae_MatchesReference)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        const string fx = "browser/vae_sd15_512/";
        var source = new SdRangeSource(http);
        string shapesJson;
        try { shapesJson = await http.GetStringAsync(SdServer + fx + "shapes.json"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769"); }
        using var shapes = System.Text.Json.JsonDocument.Parse(shapesJson);
        int[] Shape(string n) => shapes.RootElement.GetProperty(n).EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var bufs = new List<IDisposable>();
        async Task<ArrayView1D<float, Stride1D.Dense>> Fetch(string n)
        {
            var s = Shape(n); long count = s.Aggregate(1L, (a, b) => a * b);
            var b = accelerator.Allocate1D<float>(count); bufs.Add(b);
            await using var st = await source.OpenAsync("", fx + n + ".f32");
            await b.View.CopyFromStreamAsync(st);
            return b.View;
        }
        try
        {
            using var ck = await SdCheckpoint.LoadAsync(accelerator, source, "", Sd15File, new SdCheckpointOptions { LoadUNet = false, LoadTextEncoder = false });
            var vae = ck.Vae!;
            var zin = await Fetch("dec_in"); var zs = Shape("dec_in");
            int h = zs[1], w = zs[2];
            var taps = new Dictionary<string, (MemoryBuffer1D<float, Stride1D.Dense> Buf, int C, int HW)>();
            vae.Tap = (name, view, H, W, C) => { var b = accelerator.Allocate1D<float>((long)H * W * C); bufs.Add(b); b.View.CopyFrom(view.SubView(0, b.Length)); taps[name] = (b, C, H * W); };
            var img = accelerator.Allocate1D<float>(3L * 64 * h * w); bufs.Add(img);
            await vae.DecodeAsync(zin, h, w, img.View);
            vae.Tap = null;
            var report = new List<string>();
            double worst = 0;
            foreach (var name in new[] { "post_quant_conv", "dec_mid", "dec_up_0", "dec_up_1", "dec_up_2", "dec_up_3" })
            {
                var (b, C, HW) = taps[name];
                double r = await SdDeviceRelRms(accelerator, b.View, await Fetch(name), C, HW, channelLast: true);
                report.Add($"{name} {r:E2}"); worst = Math.Max(worst, r);
            }
            double ri = await SdDeviceRelRms(accelerator, img.View, await Fetch("image"), 3, 64 * h * w, channelLast: false);
            report.Add($"image {ri:E2}"); worst = Math.Max(worst, ri);
            var eimg = await Fetch("enc_image"); var es = Shape("enc_image");
            var mean = accelerator.Allocate1D<float>(4L * (es[1] / 8) * (es[2] / 8)); bufs.Add(mean);
            await vae.EncodeAsync(eimg, es[1], es[2], mean.View);
            double re = await SdDeviceRelRms(accelerator, mean.View, await Fetch("enc_mean"), 4, (es[1] / 8) * (es[2] / 8), channelLast: false);
            report.Add($"encode mean {re:E2}"); worst = Math.Max(worst, re);
            var line = $"{accelerator.AcceleratorType} VAE 512: " + string.Join(", ", report);
            Console.WriteLine("[Sd] " + line);
            if (!(worst < 5e-3)) throw new Exception("the browser VAE deviates from the reference: " + line);
            return line;
        }
        finally
        {
            await accelerator.SynchronizeAsync();
            foreach (var b in bufs) b.Dispose();
        }
    });

    /// <summary>
    /// The same 20-step SD 1.5 picture in the BROWSER: the 4.3 GB fp32 checkpoint streamed by HTTP range requests into fp16 on the GPU (never through the page heap), the same
    /// host DDIM loop, the reference's final latent and picture to compare. Serve SD_DIR first: <c>node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769</c>, fixture
    /// files from tools/sd-harness/reference/export_browser_fixture.py; <c>PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Sd15_Ddim20_OverHttp PMT_LANES=webgpu PMT_CONSOLE_LOG=Sd</c>.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Sd15_Ddim20_OverHttp_MatchesReference() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser picture test (the desktop runs Sd15_Ddim20_MatchesReference)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        const string fx = "browser/sd15_ddim20_f16w/";
        string metaJson; float[] noise, final, want;
        try
        {
            metaJson = await http.GetStringAsync(SdServer + fx + "meta.json");
            noise = WanFloats(await http.GetByteArrayAsync(SdServer + fx + "noise.f32"));
            final = WanFloats(await http.GetByteArrayAsync(SdServer + fx + "final.f32"));
            want = WanFloats(await http.GetByteArrayAsync(SdServer + fx + "image.f32"));
        }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs D:/Projects/sd 8769"); }
        using var meta = System.Text.Json.JsonDocument.Parse(metaJson);
        int[] Ints(string k) => meta.RootElement.GetProperty(k).EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var ac = meta.RootElement.GetProperty("alphas_cumprod").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        var shape = Ints("latent_shape");
        int h = shape[1], w = shape[2];
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var ck = await SdCheckpoint.LoadAsync(accelerator, new SdRangeSource(http), "", Sd15File);
        double loadS = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var (x, img) = await SdDdimAsync(ck, Ints("ids"), null, Ints("uncond_ids"), null, Ints("timesteps"), ac, noise, h, w, 7.5f, null);
        double genS = sw.Elapsed.TotalSeconds;
        var ml = WanCompare(x, final); var mi = WanCompare(img, want);
        var line = $"{accelerator.AcceleratorType}: load {loadS:F1} s, text + 20 steps x 2 + decode {genS:F1} s; final latent relRMS {ml.RelRms:E2}, picture relRMS {mi.RelRms:E2}";
        Console.WriteLine("[Sd] browser picture " + line);
        if (!(ml.RelRms < 2e-2 && mi.RelRms < 2e-2)) throw new Exception("the browser picture deviates from the reference: " + line);
        return line;
    });

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
        var (worst, ran) = await SdVaeVsFixtures(accelerator, vae, tags, report);
        SdLog($"{ck.Family} VAE", report);
        if (ran == 0) throw new UnsupportedTestException("no VAE fixture found");
        if (!(worst < bound)) throw new Exception("the VAE deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    private static async Task<(double Worst, int Ran)> SdVaeVsFixtures(Accelerator accelerator, SdVae vae, string[] tags, List<string> report)
    {
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
        return (worst, ran);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    // split component loaders (ComfyUI UNETLoader / VAELoader / CLIPLoader / DualCLIPLoader)
    // -----------------------------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// SDXL base from its diffusers repo files instead of the single file: <c>unet/diffusion_pytorch_model.fp16.safetensors</c> (diffusers UNet names),
    /// <c>vae/diffusion_pytorch_model.fp16.safetensors</c> (diffusers VAE names), <c>text_encoder</c> + <c>text_encoder_2</c> (Hugging Face CLIP-L and bigG with
    /// text_projection.weight) through SdUNet / SdVae / SdTextEncoder.LoadAsync: the UNet and the text encoders hold the single file weights, so they are held to its
    /// reference and must land on the same numbers; the repo VAE has a different decoder and its own reference (ref_sd_dump.py --vae on the folder). Files in SD_DIR\split\sdxl (stabilityai/stable-diffusion-xl-base-1.0).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sdxl_SplitLoaders_MatchReference() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, SdxlFile);
        string dir = Path.Combine(SdDir, "split", "sdxl");
        string F(string rel) { var f = Path.Combine(dir, rel); if (!File.Exists(f)) throw new UnsupportedTestException($"{f} not found"); return f; }
        FileStream Open(string f) => new(f, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var fx = Path.Combine(SdDir, "ref", "sd_sdxl_small_f16s.npz");
        if (!File.Exists(fx)) throw new UnsupportedTestException("SDXL fixture not found");
        var report = new List<string>();
        double worst = 0;
        {
            await using var l = Open(F("text_encoder/model.fp16.safetensors")); await using var g = Open(F("text_encoder_2/model.fp16.safetensors"));
            using var te = await SdTextEncoder.LoadSdxlAsync(accelerator, l, "text_encoder", g, "text_encoder_2");
            var r = new List<string>();
            worst = Math.Max(worst, await SdTextVsFixture(te, SdFamily.SDXL, fx, r));
            report.Add("DualCLIP (HF L + HF bigG): " + string.Join(", ", r));
        }
        {
            await using var s = Open(F("unet/diffusion_pytorch_model.fp16.safetensors"));
            using var unet = await SdUNet.LoadAsync(accelerator, s, "unet (diffusers)");
            if (!unet.IsXL || unet.LabelWidth != 2816) throw new Exception($"the diffusers UNet loaded as XL={unet.IsXL}, label width {unet.LabelWidth}");
            var r = new List<string>();
            var (e, t, _) = await SdUNetVsFixtures(accelerator, unet, new[] { "sdxl_small_f16s" }, r);
            worst = Math.Max(worst, Math.Max(e, t));
            report.Add("UNETLoader (diffusers names): " + string.Join(", ", r));
        }
        {
            await using var s = Open(F("vae/diffusion_pytorch_model.fp16.safetensors"));
            using var vae = await SdVae.LoadAsync(accelerator, s, "vae (diffusers)");
            var r = new List<string>();
            // the repo VAE is NOT the single file VAE (same encoder, a different decoder: post_quant_conv / decoder weights differ), so it has its own reference
            var (wv, _) = await SdVaeVsFixtures(accelerator, vae, new[] { "sdxlrepovae_small_f16s" }, r);
            worst = Math.Max(worst, wv);
            report.Add("VAELoader (diffusers names): " + string.Join(", ", r));
        }
        SdLog("SDXL split loaders", report);
        if (!(worst < 2e-2)) throw new Exception("a split loader deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// SD 1.x split loaders: CLIPLoader with one Hugging Face CLIP-L file (the SDXL repo's <c>text_encoder</c> is OpenAI CLIP ViT-L, the same encoder SD 1.5 uses: it must
    /// load as SD 1.x and give the SD 1.5 context, last layer and clip skip 2) and VAELoader with an unprefixed LDM VAE (<c>vae-ft-mse-840000-ema-pruned</c>, fp32) against
    /// its own reference (diffusers with that VAE swapped in).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_SplitLoaders_MatchReference() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, Sd15File);
        string clip = Path.Combine(SdDir, "split", "sdxl", "text_encoder", "model.fp16.safetensors"), vaeFile = Path.Combine(SdDir, "split", "vae", "vae-ft-mse-840000-ema-pruned.safetensors");
        string fxText = Path.Combine(SdDir, "ref", "sd_sd15_small_f16w.npz");
        if (!File.Exists(clip) || !File.Exists(vaeFile) || !File.Exists(fxText) || !File.Exists(Path.Combine(SdDir, "ref", "sd_sd15ftmse_small_f16s.npz"))) throw new UnsupportedTestException("split SD 1.5 files not found");
        FileStream Open(string f) => new(f, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var report = new List<string>();
        double worst = 0;
        {
            await using var s = Open(clip);
            using var te = await SdTextEncoder.LoadAsync(accelerator, s, "clip_l (HF)");
            if (te.IsXL || te.Width != 768) throw new Exception($"CLIP-L loaded as XL={te.IsXL}, width {te.Width}");
            var r = new List<string>();
            worst = Math.Max(worst, await SdTextVsFixture(te, SdFamily.SD1, fxText, r));
            report.Add("CLIPLoader (HF CLIP-L): " + string.Join(", ", r));
        }
        {
            await using var s = Open(vaeFile);
            using var vae = await SdVae.LoadAsync(accelerator, s, "vae-ft-mse");
            var r = new List<string>();
            var (wv, _) = await SdVaeVsFixtures(accelerator, vae, new[] { "sd15ftmse_small_f16s" }, r);
            worst = Math.Max(worst, wv);
            report.Add("VAELoader (LDM, unprefixed): " + string.Join(", ", r));
        }
        SdLog("SD1.5 split loaders", report);
        if (!(worst < 2e-2)) throw new Exception("a split loader deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
