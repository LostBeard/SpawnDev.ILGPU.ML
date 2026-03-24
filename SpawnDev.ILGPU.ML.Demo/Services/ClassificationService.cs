using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Services;

/// <summary>
/// Demo classification service — wraps InferenceSession + ClassificationPipeline.
/// Manages model loading lifecycle for the demo page.
/// </summary>
public class ClassificationService : IDisposable
{
    private readonly HttpClient _http;
    private InferenceSession? _session;
    private ClassificationPipeline? _pipeline;
    private Accelerator? _accelerator;

    public bool IsModelLoaded => _session != null;
    public string ModelInfo => _session?.ToString() ?? "Not loaded";

    public ClassificationService(HttpClient http) => _http = http;

    /// <summary>Load model for the given accelerator.</summary>
    public async Task LoadModelAsync(string modelUrl, Accelerator accelerator)
    {
        _session?.Dispose();
        _pipeline?.Dispose();
        _accelerator = accelerator;

        _session = await InferenceSession.CreateFromFileAsync(accelerator, _http, modelUrl);
        _pipeline = new ClassificationPipeline(_session, accelerator);
    }

    /// <summary>Classify an RGBA image. Returns results + inference time.</summary>
    public async Task<(ClassificationResult[] predictions, double inferenceMs)> ClassifyAsync(
        int[] rgbaPixels, int width, int height, int topK = 5)
    {
        if (_pipeline == null) throw new InvalidOperationException("Model not loaded");

        var sw = Stopwatch.StartNew();
        var results = await _pipeline.ClassifyAsync(rgbaPixels, width, height, topK);
        sw.Stop();

        return (results, sw.Elapsed.TotalMilliseconds);
    }

    public void Dispose()
    {
        _pipeline?.Dispose();
        _session?.Dispose();
    }
}
