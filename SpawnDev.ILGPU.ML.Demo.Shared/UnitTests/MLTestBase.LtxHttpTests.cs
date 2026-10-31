using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// LTX-Video 2B 0.9.8 distilled in the BROWSER, fed by HTTP range requests from a local file server laid out like the model repo on models.spawndev.com
/// (<c>LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU</c>: the 8-bit DiT and T5, the VAE decoder, the tokenizer) plus the reference fixtures in <c>ref/</c>. Serve the folder first:
/// <c>node tools/unirig-harness/serve-files.mjs D:/Projects/ltx 8771</c>, then <c>PMT_EXCLUDE_CATEGORIES=__none__ PMT_FILTER=Ltx_ PMT_LANES=WebGPU PMT_CONSOLE_LOG=Ltx</c>.
/// </summary>
public abstract partial class MLTestBase
{
    private const string LtxServer = "http://127.0.0.1:8771/";

    /// <summary>The local file server as a model source (every repo is the server root); files stream by Range requests that stay in JS.</summary>
    private sealed class LtxRangeSource(HttpClient http) : IModelSource
    {
        public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            using var head = await http.SendAsync(new HttpRequestMessage(HttpMethod.Head, LtxServer + filePath), cancellationToken);
            head.EnsureSuccessStatusCode();
            return new FetchRangeStream(LtxServer + filePath, head.Content.Headers.ContentLength ?? throw new InvalidDataException($"{filePath}: the server gave no length"));
        }
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => http.GetByteArrayAsync(LtxServer + filePath, cancellationToken);
    }

    private async Task<(HttpClient Http, LtxRangeSource Source)> LtxHttp()
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("a browser test (the desktop runs the Ltx_*_MatchesReference gates from disk)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        try { using var r = await http.SendAsync(new HttpRequestMessage(HttpMethod.Head, LtxServer + "ltx_vae_f16.safetensors")); r.EnsureSuccessStatusCode(); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs D:/Projects/ltx 8771"); }
        return (http, new LtxRangeSource(http));
    }

    private static async Task<ZipArchive> LtxFixture(HttpClient http, string file) => new(new MemoryStream(await http.GetByteArrayAsync(LtxServer + "ref/" + file)), ZipArchiveMode.Read);

    /// <summary>
    /// Every LTX stage in the browser against the same reference fixtures as the desktop gates, with the HOSTED files (8-bit T5 and DiT, fp16 VAE decoder) read over HTTP:
    /// the T5 encoder on the three reference prompts (8-bit bound 0.07, as Ltx_T5_Quantized_MatchesReference), the 8-bit DiT's step-0 velocity prediction against the
    /// float32 reference (0.06, as Ltx_DiT_Quantized_MatchesReference) and the VAE decode of the reference latent (0.02, as Ltx_VaeDecode_MatchesReference).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx_Stages_OverHttp_MatchReference() => await RunTest(async accelerator =>
    {
        var (http, source) = await LtxHttp();
        var report = new List<string>();
        bool ok = true;
        var paths = LtxPaths.InRepo(LtxPaths.DitFile("q8"), LtxPaths.TextEncoderFile("q8"));
        // T5
        using (var enc = await UmT5Encoder.CreateAsync(accelerator, source, "", paths.TextEncoder))
        {
            double worst = 0;
            foreach (var name in new[] { "0", "1", "2" })
            {
                using var fx = await LtxFixture(http, $"t5_{name}.npz");
                var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask"); var (want, _) = WanNpz(fx, "context");
                int n = (int)mask.Sum();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var got = await enc.EncodeAsync(ids.Take(n).Select(v => (int)v).ToArray());
                var m = WanCompare(got, want.AsSpan(0, n * 4096).ToArray());
                worst = Math.Max(worst, m.RelRms);
                report.Add($"T5 q8 prompt {name} ({n} tokens) {m.RelRms:E2} in {sw.Elapsed.TotalSeconds:F1} s");
            }
            ok &= worst < 0.07;
        }
        // DiT step 0
        {
            using var t5 = await LtxFixture(http, "t5_0.npz"); using var fx = await LtxFixture(http, "dit_t2v_fp32.npz");
            var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
            var (lat, ls) = WanNpz(fx, "step0_latent_in"); var (tArr, _) = WanNpz(fx, "step0_t");
            await using var ds = await source.OpenAsync("", paths.Dit);
            using var empty = System.Text.Json.JsonDocument.Parse("{}");
            using var weights = await GpuWeights.LoadAsync(accelerator, ds, empty.RootElement, paths.Dit);
            using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
            dit.SetGrid(ls[1], ls[2], ls[3], 8f / LtxFrameRate, 32f, 32f);
            dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
            using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            dit.Forward(lIn.View, tArr[0], 0, lOut.View);
            var got = await lOut.View.CopyToHostAsync();
            var m = WanCompare(got, WanNpz(fx, "step0_pred").Data);
            report.Add($"DiT q8 step 0 [128, {ls[1]}, {ls[2]}, {ls[3]}] vs fp32 {m.RelRms:E2} in {sw.Elapsed.TotalSeconds:F1} s");
            ok &= m.RelRms < 0.06;
        }
        // VAE decode
        {
            using var fx = await LtxFixture(http, "vae_small.npz");
            var (latent, ls) = WanNpz(fx, "latent"); var (tArr, _) = WanNpz(fx, "timestep");
            await using var vs = await source.OpenAsync("", paths.Vae);
            using var empty = System.Text.Json.JsonDocument.Parse("{}");
            using var weights = await GpuWeights.LoadAsync(accelerator, vs, empty.RootElement, paths.Vae);
            using var vae = new LtxVae(accelerator, weights);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var video = await vae.DecodeAsync(latent, ls[1], ls[2], ls[3], tArr[0]);
            var m = WanCompare(video, WanNpz(fx, "video").Data);
            report.Add($"VAE decode {m.RelRms:E2} in {sw.Elapsed.TotalSeconds:F1} s");
            ok &= m.RelRms < 0.02;
        }
        foreach (var l in report) Console.WriteLine($"[Ltx] browser {l}");
        if (!ok) throw new Exception("LTX in the browser: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// Image-to-video in the browser over HTTP (the VAE encoder, the DiT): <see cref="LtxPipeline.DenoiseAsync"/> on the reference's picture, prompt embedding and random draws
    /// (fixture <c>ref/cond_i2v.npz</c>, LTXConditionPipeline) against its final latent. With the bf16 DiT the browser's math is float32 (no tensor cores), so it is held to the
    /// exact-math bound of the desktop gate (1e-3); the hosted 8-bit DiT is then the quantisation's cost, which the desktop gate measures on CUDA with the same weights (bound 0.2).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx_ImageToVideo_OverHttp_MatchesReference() => await RunTest(async accelerator =>
    {
        var (http, source) = await LtxHttp();
        using var t5 = await LtxFixture(http, "t5_0.npz"); using var fx = await LtxFixture(http, "cond_i2v.npz");
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        int steps = LtxSampler.DistilledFirstPass.Length;
        var req = new LtxRequest
        {
            Prompt = "", PromptEmbedding = ctx.AsSpan(0, rows * 4096).ToArray(), Width = 480, Height = 320, Frames = 25, FrameRate = LtxFrameRate,
            Conditions = new[] { new LtxCondition { Pixels = WanNpz(fx, "image").Data } },
            InitialNoise = WanNpz(fx, "noise").Data, PosteriorNoise = new[] { WanNpz(fx, "posterior").Data },
            StepNoise = Enumerable.Range(0, steps).Select(i => WanNpz(fx, $"cond_noise{i}").Data).ToArray(),
        };
        var want = WanNpz(fx, "final").Data;
        var lines = new List<string>();
        double bf16 = 0, q8 = 0;
        foreach (var precision in new[] { "bf16", "q8" })
        {
            var pipe = new LtxPipeline(accelerator, source, "", LtxPaths.InRepo(LtxPaths.DitFile(precision), LtxPaths.TextEncoderFile("q8")));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var m = WanCompare(await pipe.DenoiseAsync(req), want);
            lines.Add($"image to video ({precision} DiT) final latent relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2} in {sw.Elapsed.TotalSeconds:F1} s");
            if (precision == "bf16") bf16 = m.RelRms; else q8 = m.RelRms;
        }
        foreach (var l in lines) Console.WriteLine($"[Ltx] browser {l}");
        var report = string.Join("; ", lines);
        if (!(bf16 < 1e-3 && q8 < 0.2)) throw new Exception("LTX image to video in the browser: " + report);
        return report;
    });

    /// <summary>
    /// The distilled model's two-pass recipe in the browser over HTTP: each stage on the reference's own input (fixture <c>ref/ms_ms.npz</c>, diffusers' pipelines chained; the
    /// browser's math is float32 with the bf16 DiT): the upscaler on the reference's first-pass latent (bound 1e-4) and the second pass on the reference's upsampled latent (5e-3),
    /// as Ltx_MultiScale_MatchesReference on the desktop; then the demo's configuration end to end (<see cref="LtxRequest.Upscale"/>, the 8-bit DiT and T5, a picture as the first
    /// frame, 704x480 x 97 -> 896x576): finite, a picture, the stage times to the console.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx_MultiScale_OverHttp_MatchesReference() => await RunTest(async accelerator =>
    {
        var (http, source) = await LtxHttp();
        using var t5 = await LtxFixture(http, "t5_0.npz"); using var fx = await LtxFixture(http, "ms_ms.npz");
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var sizes = WanNpz(fx, "sizes").Data;
        int w1 = (int)sizes[0], h1 = (int)sizes[1], w2 = (int)sizes[2], h2 = (int)sizes[3];
        var lines = new List<string>();
        var pipe = new LtxPipeline(accelerator, source, "", LtxPaths.InRepo(LtxPaths.DitFile("bf16"), LtxPaths.TextEncoderFile("q8")));
        var refUp = WanNpz(fx, "up").Data;
        var mu = WanCompare(await pipe.UpsampleLatentAsync(WanNpz(fx, "p1_final").Data, 4, h1 / 32, w1 / 32, 1f), refUp);
        var req2 = new LtxRequest
        {
            Prompt = "", PromptEmbedding = ctx.AsSpan(0, rows * 4096).ToArray(), Width = w2, Height = h2, Frames = 25, FrameRate = LtxFrameRate, Sigmas = LtxSampler.DistilledSecondPass, Latents = refUp,
            Conditions = new[] { new LtxCondition { Pixels = WanNpz(fx, "p2_pixels").Data } },
            InitialNoise = WanNpz(fx, "p2_noise").Data, PosteriorNoise = new[] { WanNpz(fx, "p2_posterior").Data },
            StepNoise = Enumerable.Range(0, LtxSampler.DistilledSecondPass.Length).Select(s => WanNpz(fx, $"p2_cond_noise{s}").Data).ToArray(),
        };
        var m2 = WanCompare(await pipe.DenoiseAsync(req2), WanNpz(fx, "p2_final").Data);
        lines.Add($"on the reference's inputs: upscaler {mu.RelRms:E2}, second pass {w2}x{h2} (bf16 DiT) {m2.RelRms:E2}");
        bool ok = mu.RelRms < 1e-4 && m2.RelRms < 5e-3;

        // the demo's configuration end to end: a picture (smooth test content) as the first frame
        int pw = 640, ph = 448;
        var rgb = new byte[pw * ph * 3];
        for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) { int i = (y * pw + x) * 3; rgb[i] = (byte)(128 + 100 * Math.Sin(x / 40.0)); rgb[i + 1] = (byte)(128 + 100 * Math.Cos(y / 30.0)); rgb[i + 2] = (byte)((x + y) % 256); }
        var demo = new LtxPipeline(accelerator, source, "", LtxPaths.InRepo(LtxPaths.DitFile("q8"), LtxPaths.TextEncoderFile("q8")));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string last = ""; var stages = new List<string>();
        var progress = new Progress<WanProgress>(p => { var key = p.Stage + (p.Message.StartsWith("second") ? "2" : ""); if (key != last) { last = key; stages.Add($"{p.Stage}{(key.EndsWith("2") ? " (second pass)" : "")} at {sw.Elapsed.TotalSeconds:F0} s"); } });
        var video = await demo.GenerateAsync(new LtxRequest
        {
            Prompt = "A colourful abstract painting slowly comes alive, the shapes drift and swirl, soft light.", Width = 704, Height = 480, Frames = 97, Seed = 42,
            Upscale = true, Conditions = new[] { LtxCondition.Image(rgb, pw, ph) },
        }, progress);
        double min = double.MaxValue, max = double.MinValue;
        foreach (var v in video.Data) { if (!float.IsFinite(v)) throw new Exception("a pixel is not finite"); min = Math.Min(min, v); max = Math.Max(max, v); }
        lines.Add($"two-pass 704x480 x 97 (8-bit DiT + T5, a first frame): {video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s ({string.Join(", ", stages)}), pixels {min:F2}..{max:F2}");
        ok &= video.Width == 896 && video.Height == 576 && video.Frames == 97 && max <= 3 && min >= -3 && max - min > 0.5;
        foreach (var l in lines) Console.WriteLine($"[Ltx] browser {l}");
        if (!ok) throw new Exception("LTX two-pass in the browser: " + string.Join("; ", lines));
        return string.Join("; ", lines);
    });

    /// <summary>
    /// The whole LTX pipeline in the browser over HTTP with the hosted files (the demo's configuration: 8-bit DiT and T5), at the model's native 704x480 and 97 frames: the
    /// clip must be finite, a picture (spread out, no blow-up beyond the decoder's small overshoot of [-1, 1]); stage times go to the console (each stage is gated against the reference by Ltx_Stages_OverHttp_MatchReference).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Ltx_Pipeline_OverHttp_Generates() => await RunTest(async accelerator =>
    {
        var (_, source) = await LtxHttp();
        var pipe = new LtxPipeline(accelerator, source, "", LtxPaths.InRepo(LtxPaths.DitFile("q8"), LtxPaths.TextEncoderFile("q8")));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string last = "";
        var stages = new List<string>();
        var progress = new Progress<WanProgress>(p => { if (p.Stage != last) { last = p.Stage; stages.Add($"{p.Stage} at {sw.Elapsed.TotalSeconds:F0} s"); Console.WriteLine($"[Ltx] browser stage {p.Stage} at {sw.Elapsed.TotalSeconds:F0} s ({p.Message})"); } });
        var video = await pipe.GenerateAsync(new LtxRequest { Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light.", Width = 704, Height = 480, Frames = 97, Seed = 42 }, progress);
        double min = double.MaxValue, max = double.MinValue;
        foreach (var v in video.Data) { if (!float.IsFinite(v)) throw new Exception("a pixel is not finite"); min = Math.Min(min, v); max = Math.Max(max, v); }
        var report = $"{video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s ({string.Join(", ", stages)}), pixels {min:F2}..{max:F2}";
        Console.WriteLine($"[Ltx] browser pipeline on {accelerator.AcceleratorType}: {report}");
        // the VAE output is not clamped (LtxVae.DecodeAsync: nominally [-1, 1], as diffusers' decoder; the frame conversion clamps): finite, a real picture, no blow-up
        if (video.Frames != 97 || max > 3 || min < -3 || !(max - min > 0.5)) throw new Exception("the LTX browser clip is wrong: " + report);
        return report;
    });
}
