using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Wan 2.2 TI2V-5B in the BROWSER: the whole pipeline fed by HTTP range requests from a local file server, the way the demo is fed from the model host (the 5.6 GB DiT, the 1.4 GB VAE and
/// the 11.4 GB umT5 never sit in the page memory: each tensor goes from the fetch to the GPU). Serve the folder first:
/// <c>node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767</c>, then <c>PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Wan_Pipeline_OverHttp PMT_LANES=webgpu PMT_CONSOLE_LOG=Wan</c>.
/// </summary>
public abstract partial class MLTestBase
{
    private const string WanServer = "http://127.0.0.1:8767/";

    /// <summary>The local file server as a model source: every repo is the server root; files stream by Range requests that stay in JS (<see cref="FetchRangeStream"/>).</summary>
    private sealed class WanRangeSource(HttpClient http) : IModelSource
    {
        public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            using var head = await http.SendAsync(new HttpRequestMessage(HttpMethod.Head, WanServer + filePath), cancellationToken);
            head.EnsureSuccessStatusCode();
            return new FetchRangeStream(WanServer + filePath, head.Content.Headers.ContentLength ?? throw new InvalidDataException($"{filePath}: the server gave no length"));
        }
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => http.GetByteArrayAsync(WanServer + filePath, cancellationToken);
    }

    private static float[] WanFloats(byte[] raw) { var f = new float[raw.Length / 4]; Buffer.BlockCopy(raw, 0, f, 0, raw.Length); return f; }

    /// <summary>
    /// The pipeline in the browser against the reference 30-step trajectory (the same prompt and starting noise as Wan_Pipeline_MatchesReference; the fixtures are small files exported next to the
    /// models): the final latent must be as close to the reference as the desktop run (relRMS about 0.1 to 0.2: the reference's own sensitivity), the video finite and not constant.
    /// Stage timings go to the console.
    /// </summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_OverHttp_MatchesReference() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser pipeline test (the desktop runs Wan_Pipeline_MatchesReference)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] noiseRaw, wantRaw;
        try { noiseRaw = await http.GetByteArrayAsync(WanServer + "dump2/browser/noise_t2v30.f32"); wantRaw = await http.GetByteArrayAsync(WanServer + "dump2/browser/final_t2v30.f32"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767"); }
        var noise = WanFloats(noiseRaw); var want = WanFloats(wantRaw);
        var pipe = new Wan22Pipeline(accelerator, new WanRangeSource(http), "", Wan22Paths.InRepo(textEncoder: Wan22Paths.TextEncoderFile("q8")));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string last = "";
        var progress = new Progress<WanProgress>(p => { if (p.Stage != last) { last = p.Stage; Console.WriteLine($"[Wan] browser stage {p.Stage} at {sw.Elapsed.TotalSeconds:F0} s ({p.Message})"); } });
        var video = await pipe.GenerateAsync(new WanRequest
        {
            Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.", Width = 640, Height = 352, Frames = 17, Steps = 30, InitialNoise = noise, Trace = true,
        }, progress);
        double se = 0, sr = 0, min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < want.Length; i++) { double d = video.Latent![i] - want[i]; se += d * d; sr += (double)want[i] * want[i]; }
        foreach (var v in video.Data) { if (!float.IsFinite(v)) throw new Exception("a pixel is not finite"); min = Math.Min(min, v); max = Math.Max(max, v); }
        double rel = Math.Sqrt(se / sr);
        Console.WriteLine($"[Wan] browser pipeline on {accelerator.AcceleratorType}: {video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s, final latent relRMS {rel:F3} vs the reference, pixels {min:F2}..{max:F2}");
        if (!(rel < 0.4)) throw new Exception($"the browser pipeline's final latent is not the reference's: relRMS {rel:F3}");
        if (max > 1.0001 || min < -1.0001 || !(max - min > 0.5)) throw new Exception($"the video is out of range or flat: {min}..{max}");
        return $"relRMS {rel:F3} in {sw.Elapsed.TotalSeconds:F0} s";
    });

    /// <summary>
    /// The VAE decoder alone in the browser (the reference's final 30-step latent from the file server, 640x352, 17 frames) with the per-stage device times on the console: the part of the pipeline that
    /// is slow in the browser. Same file server as above.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_Vae_OverHttp_DecodeTimes() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser VAE timing test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        float[] latent;
        try { latent = WanFloats(await http.GetByteArrayAsync(WanServer + "dump2/browser/final_t2v30.f32")); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767"); }
        var source = new WanRangeSource(http);
        await using var stream = await source.OpenAsync("", "wan22_vae_f16.safetensors");
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        using var weights = await SpawnDev.ILGPU.ML.Trellis2.GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, "vae");
        using var vae = new WanVae(accelerator, weights) { Trace = true };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var video = await vae.DecodeAsync(latent, 5, 22, 40);
        double sum = 0, sum2 = 0; foreach (var v in video) { sum += v; sum2 += (double)v * v; }
        Console.WriteLine($"[Wan] vae decode on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s for {video.Length / (3 * 352 * 640)} frames, pixel mean {sum / video.Length:F5}, rms {Math.Sqrt(sum2 / video.Length):F5}");
        return $"{sw.Elapsed.TotalSeconds:F1} s";
    });

    /// <summary>The browser counterpart of Wan_Vae_DecodeSizes: a random latent decoded at 832x480 with 17 and 33 frames (the sizes the /video page offers), timed; a failure is reported, not thrown.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_Vae_OverHttp_DecodeSizes() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser VAE size test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var source = new WanRangeSource(http);
        try { await using var probe = await source.OpenAsync("", "wan22_vae_f16.safetensors"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767"); }
        await using var stream = await source.OpenAsync("", "wan22_vae_f16.safetensors");
        using var empty = System.Text.Json.JsonDocument.Parse("{}");
        using var weights = await SpawnDev.ILGPU.ML.Trellis2.GpuWeights.LoadAsync(accelerator, stream, empty.RootElement, "vae");
        using var vae = new WanVae(accelerator, weights);
        string last = "none";
        foreach (var (w, h, frames) in new[] { (640, 352, 17), (832, 480, 17), (832, 480, 33) })
        {
            int Fz = (frames - 1) / 4 + 1, lw = w / 16, lh = h / 16;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var video = await vae.DecodeAsync(Wan22Pipeline.Gaussian(48L * Fz * lh * lw, 5), Fz, lh, lw);
                Console.WriteLine($"[Wan] browser vae size {w}x{h}x{frames}: decoded in {sw.Elapsed.TotalSeconds:F1} s");
                last = $"{w}x{h}x{frames}";
            }
            catch (Exception e) { Console.WriteLine($"[Wan] browser vae size {w}x{h}x{frames}: FAILED after {sw.Elapsed.TotalSeconds:F1} s: {e.GetType().Name}"); break; }
        }
        return $"largest size that decoded: {last}";
    });

    /// <summary>The largest video the /video page offers, end to end in the browser: 832x480, 33 frames, 8 steps (a draft: only the memory and the stage times matter), seed 3; the video must be finite and not flat.</summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_OverHttp_Large() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser pipeline test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var source = new WanRangeSource(http);
        try { await using var probe = await source.OpenAsync("", "wan22_vae_f16.safetensors"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767"); }
        var pipe = new Wan22Pipeline(accelerator, source, "", Wan22Paths.InRepo(textEncoder: Wan22Paths.TextEncoderFile("q8")));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string last = "";
        var progress = new Progress<WanProgress>(p => { if (p.Stage != last) { last = p.Stage; Console.WriteLine($"[Wan] browser large stage {p.Stage} at {sw.Elapsed.TotalSeconds:F0} s"); } });
        var video = await pipe.GenerateAsync(new WanRequest { Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly.", Width = 832, Height = 480, Frames = 33, Steps = 8, Seed = 3 }, progress);
        double sum = 0, sum2 = 0;
        foreach (var v in video.Data) { if (!float.IsFinite(v)) throw new Exception("a pixel is not finite"); sum += v; sum2 += (double)v * v; }
        double mean = sum / video.Data.Length, std = Math.Sqrt(sum2 / video.Data.Length - mean * mean);
        Console.WriteLine($"[Wan] browser large: {video.Frames} frames {video.Width}x{video.Height} in {sw.Elapsed.TotalSeconds:F0} s, pixel std {std:F3}");
        if (!(std > 0.02)) throw new Exception($"the video is flat (std {std:F4})");
        return $"{sw.Elapsed.TotalSeconds:F0} s";
    });

    /// <summary>
    /// The Turbo transformer (step-distilled, 4 steps, guidance 1, the few-step sampler) in the browser at the sizes the /video page offers: 832x480 x33 frames, then 1280x704 x17 (the model's own size;
    /// a failure there is reported, not thrown). The video must be finite and not flat; stage times go to the console.
    /// </summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_OverHttp_Turbo() => await RunTest(async accelerator =>
    {
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the browser pipeline test");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var source = new WanRangeSource(http);
        try { await using var probe = await source.OpenAsync("", Wan22Paths.DitFile("turbo_q8")); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running or has no Turbo file ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs F:/Projects/wan 8767"); }
        var pipe = new Wan22Pipeline(accelerator, source, "", Wan22Paths.InRepo(Wan22Paths.DitFile("turbo_q8"), Wan22Paths.TextEncoderFile("q8")));
        string summary = "";
        foreach (var (w, h, frames) in new[] { (832, 480, 33), (1280, 704, 17) })
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            string last = "";
            var progress = new Progress<WanProgress>(p => { if (p.Stage != last) { last = p.Stage; Console.WriteLine($"[Wan] browser turbo {w}x{h} stage {p.Stage} at {sw.Elapsed.TotalSeconds:F0} s"); } });
            try
            {
                var video = await pipe.GenerateAsync(new WanRequest { Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.", Width = w, Height = h, Frames = frames, Steps = 4, GuidanceScale = 1.0, Sampler = WanSampler.FewStep, Seed = 8 }, progress);
                double sum = 0, sum2 = 0;
                foreach (var v in video.Data) { if (!float.IsFinite(v)) throw new Exception("a pixel is not finite"); sum += v; sum2 += (double)v * v; }
                double mean = sum / video.Data.Length, std = Math.Sqrt(sum2 / video.Data.Length - mean * mean);
                Console.WriteLine($"[Wan] browser turbo {w}x{h}x{frames}: {sw.Elapsed.TotalSeconds:F0} s, pixel std {std:F3}");
                if (!(std > 0.02)) throw new Exception($"the video is flat (std {std:F4})");
                summary += $"{w}x{h}x{frames} {sw.Elapsed.TotalSeconds:F0} s; ";
            }
            catch (Exception e) when (w == 1280)
            {
                Console.WriteLine($"[Wan] browser turbo {w}x{h}x{frames}: FAILED after {sw.Elapsed.TotalSeconds:F0} s: {e.GetType().Name}: {e.Message}");
                summary += $"{w}x{h}x{frames} failed; ";
            }
        }
        return summary;
    });
}
