using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Model delivery with <b>no WebTorrent anywhere</b> - the point of the <see cref="IModelSource"/> /
/// <see cref="IModelStore"/> split.
/// </summary>
/// <remarks>
/// These construct <see cref="HubModelSource"/> (plain HTTP through the hub, cached in OPFS) and never
/// touch <c>WebTorrentClient</c>, <c>HubModelStream</c> or a magnet. If the pipelines ever get re-welded to
/// the torrent transport, these stop compiling or stop passing.
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>The source opens a real hub file, caches it, and hands back a JS-side stream.</summary>
    [TestMethod(Timeout = 120000)]
    public async Task HubModelSource_OpensAndCachesWithoutWebTorrent() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("HubModelSource caches in OPFS - browser only");

        using var source = new HubModelSource(js);
        var key = source.CacheKey("Xenova/distilgpt2", "tokenizer.json");
        await source.Store.RemoveAsync(key);

        if (await source.Store.ExistsAsync(key))
            throw new Exception("Entry reported present before anything was fetched.");

        long length;
        var stream = await source.OpenAsync("Xenova/distilgpt2", "tokenizer.json");
        await using (stream.ConfigureAwait(false))
        {
            length = stream.Length;
            if (length < 100_000)
                throw new Exception($"tokenizer.json came back as {length} bytes - too small to be the real file.");
            if (stream is not SpawnDev.SpawnJS.Toolbox.IJSReadStream)
                throw new Exception($"Source returned {stream.GetType().Name}, not an IJSReadStream - the " +
                                    "zero-copy JS->GPU weight path would not fire.");
            if (!stream.CanSeek)
                throw new Exception("Source stream is not seekable; the ONNX reader could not skip weight blobs.");
        }

        if (!await source.Store.ExistsAsync(key))
            throw new Exception("Nothing was cached after a successful open.");

        // Second open must be served from the store. Same length, and it must still be a JS-side stream.
        var again = await source.OpenAsync("Xenova/distilgpt2", "tokenizer.json");
        await using (again.ConfigureAwait(false))
        {
            if (again.Length != length)
                throw new Exception($"Cached open gave {again.Length}, first open gave {length}.");
        }

        // FetchBytesAsync is the small-file path (structure/tokenizer), and must agree with the stream.
        var bytes = await source.FetchBytesAsync("Xenova/distilgpt2", "tokenizer.json");
        if (bytes.LongLength != length)
            throw new Exception($"FetchBytesAsync returned {bytes.LongLength} bytes, stream reported {length}.");

        Console.WriteLine($"[NoWT] HubModelSource delivered {length:N0} bytes, cached, no WebTorrent: PASS");
        await source.Store.RemoveAsync(key);
    });

    /// <summary>
    /// A real pipeline builds end to end from <see cref="HubModelSource"/> - no WebTorrentClient exists in
    /// this test at all.
    /// </summary>
    /// <remarks>
    /// HeavyModel: downloads and GPU-compiles Depth-Anything-V2-Small. This is the test that would have been
    /// IMPOSSIBLE before the split, because <c>CreateFromHubAsync</c> required a <c>HubModelStream</c> and
    /// therefore a <c>WebTorrentClient</c>. Run with
    /// <c>PMT_EXCLUDE_CATEGORIES=__none__ PMT_FILTER=DepthPipeline_BuildsFromHttpSource</c>.
    /// </remarks>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task DepthPipeline_BuildsFromHttpSource() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("HubModelSource caches in OPFS - browser only");

        using var source = new HubModelSource(js);
        var stages = new List<string>();

        // externalDataFile: "" - DA2-Small is a SINGLE-FILE model (VERIFIED: the hub serves
        // onnx/model.onnx but onnx/model.onnx_data does not exist). That matters beyond correctness: the
        // empty-string branch STREAMS model.onnx, which is the path that keeps weights off the managed
        // heap, whereas the external-data branch pulls the structure file whole via FetchBytesAsync.
        // inputShapes pins the ViT geometry: 518/14 = 37 patches per side, so 37*37+1 = 1370 tokens. Leave
        // it dynamic and the session resolves a tiny spatial size, then Add fails broadcasting
        // [1,2,384] against [1,1370,384] much later.
        var pipe = await DepthEstimationPipeline.CreateFromHubAsync(
            accelerator, source, ModelHub.KnownModels.DepthAnythingV2Small,
            externalDataFile: "",
            onProgress: (stage, pct) => { if (pct == 100) stages.Add(stage); },
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 3, 518, 518 } },
            inputSize: 518);

        using (pipe)
        {
            // Exercise it: a real inference, not just a constructed object. A flat gradient gives the model
            // actual structure to respond to, so a constant output means it did not run.
            const int w = 64, h = 64;
            var rgba = new int[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int r = x * 4, g = y * 4, b = (x + y) * 2;
                    rgba[y * w + x] = unchecked((int)0xFF000000) | (r << 16) | (g << 8) | b;
                }

            var result = await pipe.EstimateAsync(rgba, w, h);
            if (result?.DepthMap == null || result.DepthMap.Length == 0)
                throw new Exception("Depth estimation returned nothing.");

            int finite = 0;
            float min = float.MaxValue, max = float.MinValue;
            foreach (var v in result.DepthMap)
            {
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                finite++; if (v < min) min = v; if (v > max) max = v;
            }
            if (finite == 0) throw new Exception("Every depth value was NaN/Inf.");
            if (min == max) throw new Exception($"Depth map is constant at {min} - the model did not run.");

            Console.WriteLine($"[NoWT] depth {result.DepthMap.Length} values ({result.Width}x{result.Height}), " +
                              $"range {min:F4}..{max:F4}, stages: {string.Join(",", stages)}");
        }

        Console.WriteLine("[NoWT] DepthEstimationPipeline built from HTTP source, no WebTorrent: PASS");
    });
}
