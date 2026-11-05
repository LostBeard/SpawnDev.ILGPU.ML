using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="Dinov2EmbeddingPipeline"/> with DINOv2-small against onnxruntime (tools/gen_dinov2_reference.py, references/dinov2/): the
    /// model's tokens from the reference processor's pixels, the pooled descriptor, our GPU preprocessing against Hugging Face's, and the
    /// descriptor end to end from RGBA. Two images (a street, a cat) so the end-to-end cosine is compared with how far apart different images are.
    /// Prints the warm per-frame time. Fast lanes. Added 2026-11-04 (AssistZ place retrieval).
    /// </summary>
    [TestMethod(Timeout = 900000)]
    public async Task Dinov2_Small_MatchesOnnxRuntime() => await RunTest(accelerator => Dinov2Check(accelerator, Dinov2EmbeddingPipeline.SmallRepo, "small"));

    /// <summary>The same gate for DINOv2-base (346 MB). Added 2026-11-04.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Dinov2_Base_MatchesOnnxRuntime() => await RunTest(accelerator => Dinov2Check(accelerator, Dinov2EmbeddingPipeline.BaseRepo, "base"));

    private async Task<string> Dinov2Check(Accelerator accelerator, string repo, string tag)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var dino = await Dinov2EmbeddingPipeline.CreateFromHubAsync(accelerator, new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(modelHttp), repo);
        var notes = new List<string> { $"{accelerator.AcceleratorType} dinov2-{tag}: load {sw.Elapsed.TotalSeconds:F1} s" };
        int wantDim = tag == "small" ? 384 : 768;
        if (dino.Dim != wantDim) throw new Exception($"Dim after load is {dino.Dim}, expected {wantDim} (it must be known before the first embed)");

        static float[] Floats(byte[] b) { var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
        static double Cos(float[] a, float[] b) { double s = 0; for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i]; return s; }
        var ends = new Dictionary<string, float[]>();
        foreach (var img in new[] { "street", "cat" })
        {
            var wantPixels = Floats(await assets.GetByteArrayAsync($"references/dinov2/{tag}_{img}_pixels.bin"));
            var wantTokens = Floats(await assets.GetByteArrayAsync($"references/dinov2/{tag}_{img}_tokens.bin"));
            var wantPooled = Floats(await assets.GetByteArrayAsync($"references/dinov2/{tag}_{img}_pooled.bin"));

            // 1. the model: the reference processor's pixels in, tokens and the pooled descriptor out
            using var px = accelerator.Allocate1D(wantPixels);
            var e = await dino.EmbedPixelsAsync(px.View, includeTokens: true);
            if (e.Tokens!.Length != wantTokens.Length) throw new Exception($"{img}: {e.Tokens.Length} token values, reference {wantTokens.Length}");
            double maxTok = wantTokens.Max(v => Math.Abs((double)v)), tokDiff = 0, tokMean = 0;
            int worst = 0;
            for (int i = 0; i < wantTokens.Length; i++)
            {
                double d = Math.Abs(e.Tokens[i] - wantTokens[i]);
                tokMean += d;
                if (d > tokDiff) { tokDiff = d; worst = i; }
            }
            tokMean /= wantTokens.Length;
            // per token: the worst token's cosine with the reference (a direction error vs a few large values off)
            double minTokCos = 1; int nTok = wantTokens.Length / e.Dim;
            for (int t = 0; t < nTok; t++)
            {
                double ab = 0, aa = 0, bb = 0;
                for (int d = 0; d < e.Dim; d++) { double a = e.Tokens[t * e.Dim + d], b = wantTokens[t * e.Dim + d]; ab += a * b; aa += a * a; bb += b * b; }
                minTokCos = Math.Min(minTokCos, ab / Math.Sqrt(aa * bb));
            }
            double pooledCos = Cos(e.Pooled, wantPooled);
            notes.Add($"{img}: tokens max |diff| {tokDiff:E2} at token {worst / e.Dim} dim {worst % e.Dim} (ref {wantTokens[worst]:F3}, ours {e.Tokens[worst]:F3}), "
                + $"mean |diff| {tokMean:E2}, max |token| {maxTok:F1}, worst token cosine {minTokCos:F6}; pooled cosine {pooledCos:F6}");
            // MEASURED 2026-11-04 CUDA: max 1.2e-4 at max |token| 15.8 (7.6e-6 relative). Red: before the Resize scales fix 0.93 (5%).
            if (tokDiff > 1e-4 * maxTok) throw new Exception($"{img}: the model's tokens differ from onnxruntime: " + string.Join("; ", notes));
            if (pooledCos < 0.99999) throw new Exception($"{img}: the pooled descriptor differs from the reference: " + string.Join("; ", notes));

            // 2. our GPU preprocessing against Hugging Face's (PIL), and 3. the descriptor end to end from RGBA
            var bin = await assets.GetByteArrayAsync($"samples/{img}_rgba.bin");
            int w = BitConverter.ToInt32(bin, 0), h = BitConverter.ToInt32(bin, 4);
            var rgba = new int[w * h];
            Buffer.BlockCopy(bin, 8, rgba, 0, w * h * 4);
            using var rgbaBuf = accelerator.Allocate1D(rgba);
            using var ours = accelerator.Allocate1D<float>(wantPixels.Length);
            dino.Preprocess(rgbaBuf.View, w, h, ours.View);
            var gotPixels = await ours.CopyToHostAsync<float>(0, wantPixels.Length);
            double pxMax = 0, pxMean = 0;
            for (int i = 0; i < wantPixels.Length; i++) { double d = Math.Abs(gotPixels[i] - wantPixels[i]); pxMax = Math.Max(pxMax, d); pxMean += d; }
            pxMean /= wantPixels.Length;
            var end = await dino.EmbedAsync(rgbaBuf.View, w, h);
            double endCos = Cos(end.Pooled, wantPooled);
            ends[img] = end.Pooled;
            notes.Add($"{img} {w}x{h}: preprocess vs PIL mean |diff| {pxMean:E2}, max {pxMax:F3} (normalised units); end to end cosine {endCos:F5}");
            // PIL rounds its intermediate image to 8 bits between the axis passes (1/255 = 0.0171 in these units after /std): a mean of a
            // few hundredths is that rounding; more means a different filter or crop
            // MEASURED 2026-11-04: mean 4.7e-3 / 4.9e-3, end-to-end cosine 0.99997 / 0.99990
            if (pxMean > 0.02) throw new Exception($"{img}: the preprocessing differs from the reference processor: " + string.Join("; ", notes));
            if (endCos < 0.999) throw new Exception($"{img}: the end-to-end descriptor differs from the reference: " + string.Join("; ", notes));
        }
        double between = Cos(ends["street"], ends["cat"]);
        notes.Add($"cosine(street, cat) {between:F4}");
        if (between > 0.5) throw new Exception("two different images embed alike; the descriptor does not discriminate: " + string.Join("; ", notes));

        // warm per-frame time: RGBA already on the device -> pooled descriptor on the host (5 frames, median)
        var bin2 = await assets.GetByteArrayAsync("samples/street_rgba.bin");
        int w2 = BitConverter.ToInt32(bin2, 0), h2 = BitConverter.ToInt32(bin2, 4);
        var rgba2 = new int[w2 * h2];
        Buffer.BlockCopy(bin2, 8, rgba2, 0, w2 * h2 * 4);
        using var frame = accelerator.Allocate1D(rgba2);
        var times = new List<double>();
        for (int r = 0; r < 6; r++)
        {
            sw.Restart();
            await dino.EmbedAsync(frame.View, w2, h2);
            if (r > 0) times.Add(sw.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        notes.Add($"warm frame {w2}x{h2} -> descriptor: median {times[times.Count / 2]:F1} ms (min {times[0]:F1})");
        var line = string.Join("; ", notes);
        Console.WriteLine("[Dinov2] " + line);
        return line;
    }
}
