using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;
using BBP = global::SpawnDev.ILGPU.BrowserBufferPolicy;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// "Model weights never enter the .NET heap in the browser", as a regression gate per model: each test loads the way the demo page does, through the page's own source (HubModelSource: OPFS cache, JS-side streams) on the
/// WebGPU lane, and asserts on SpawnDev.ILGPU's census of host -> device copies (<c>BrowserBufferPolicy.HostCopyBytes</c>, every <c>CopyFromCPU</c> on a browser backend) that no more than a few MiB of constants went through the managed
/// heap. The first run of each downloads the model into the browser's OPFS. (FLUX 3 has its own: <c>Flux3_Browser_WeightsStayOutOfTheManagedHeap</c>.)
/// </summary>
public abstract partial class MLTestBase
{
    private const long CensusAllowedBytes = 4L * 1024 * 1024;

    private static HubModelSource CensusSource(global::ILGPU.Runtime.Accelerator accelerator)
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser) throw new UnsupportedTestException("browser-only (OPFS)");
        if (accelerator.AcceleratorType.ToString().IndexOf("WebGPU", StringComparison.OrdinalIgnoreCase) < 0) throw new UnsupportedTestException($"WebGPU only; this lane is {accelerator.AcceleratorType}");
        return new HubModelSource(js);
    }

    /// <summary>Run <paramref name="load"/> with a fresh census, print it and fail when more than <see cref="CensusAllowedBytes"/> went through the managed heap.</summary>
    private static async Task AssertWeightsStayJs(string what, Func<Task> load)
    {
        BBP.ResetHostCopyCensus(); BBP.ResetStreamUploadTiming(); BBP.TraceStreamUploadTiming = true;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try { await load(); }
        finally { BBP.TraceStreamUploadTiming = false; }
        Console.WriteLine($"[WeightCensus] {what}: {sw.Elapsed.TotalSeconds:F1} s; JS->GPU {BBP.StreamBytes / 1048576.0:F1} MiB; through the .NET heap {BBP.HostCopyBytes / 1048576.0:F2} MiB in {BBP.HostCopyCount} copies (largest {BBP.LargestHostCopyBytes / 1024.0:F0} KiB)");
        Ur_Expect(BBP.HostCopyBytes < CensusAllowedBytes, $"{what}: {BBP.HostCopyBytes / 1048576.0:F1} MiB went through the .NET heap (largest single copy {BBP.LargestHostCopyBytes / 1048576.0:F1} MiB)");
    }

    /// <summary>YOLOv8n bundled with the page (Detect page): the URL is cached into OPFS and parsed as a stream.</summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task WeightCensus_Yolov8n_BundledModel() => await RunTest(async accelerator =>
    {
        using var source = CensusSource(accelerator);
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        const int bytes = 12823637;
        var url = new Uri(http.BaseAddress!, $"models/yolov8n/model.onnx?v={bytes}").ToString();
        await AssertWeightsStayJs("YOLOv8n", async () =>
        {
            await using var stream = await source.OpenUrlAsync(url, source.CacheKey("bundled", $"yolov8n-model-{bytes}.onnx"), bytes);
            using var session = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, stream);
        });
    });

    /// <summary>The AI Chat page's smallest model (Qwen2.5 0.5B Q8_0 GGUF) through the page's own source.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task WeightCensus_GgufChat_Qwen05B() => await RunTest(async accelerator =>
    {
        using var source = CensusSource(accelerator);
        await AssertWeightsStayJs("Qwen2.5 0.5B Q8_0", async () =>
        {
            await using var stream = await source.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
            using var pipeline = await GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, stream, maxSeqLen: 512);
        });
    });

    /// <summary>The FLUX.2 klein stages (Generate and Text to 3D pages): the encoder and the transformer through <see cref="Flux2Weights.LoadAsync"/>.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task WeightCensus_Flux2_Stages() => await RunTest(async accelerator =>
    {
        using var source = CensusSource(accelerator);
        const string repo = "LostBeard/FLUX.2-klein-4B-q4-ILGPU";
        foreach (var file in new[] { Flux2Pipeline.EncoderFile, Flux2Pipeline.TransformerFile })
        {
            await using (var warm = await source.OpenAsync(repo, file)) { }            // the download is not what is judged
            await AssertWeightsStayJs($"FLUX.2 {file}", async () => { using var w = await Flux2Weights.LoadAsync(accelerator, source, repo, file); await accelerator.SynchronizeAsync(); });
        }
    });

    /// <summary>SD Turbo (Generate page's other model): the page's own <c>ImageGenerationPipeline.CreateAsync</c> over the hub source.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task WeightCensus_SdTurbo() => await RunTest(async accelerator =>
    {
        using var source = CensusSource(accelerator);
        await AssertWeightsStayJs("SD Turbo", async () => { using var pipeline = await ImageGenerationPipeline.CreateAsync(accelerator, source); });
    });
}
