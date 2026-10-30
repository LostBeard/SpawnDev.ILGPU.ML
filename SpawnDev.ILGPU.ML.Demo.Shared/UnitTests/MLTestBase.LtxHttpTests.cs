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
