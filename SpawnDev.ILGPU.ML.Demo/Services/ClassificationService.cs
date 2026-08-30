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

    /// <summary>Load model from a URL for the given accelerator.</summary>
    public async Task LoadModelAsync(string modelUrl, Accelerator accelerator)
    {
        _session?.Dispose();
        _pipeline?.Dispose();
        _accelerator = accelerator;

        _session = await InferenceSession.CreateFromFileAsync(accelerator, _http, modelUrl);
        _pipeline = new ClassificationPipeline(_session, accelerator);
    }

    /// <summary>
    /// Adopt an already-created session - the path that lets the caller deliver weights however it likes.
    /// </summary>
    /// <remarks>
    /// Added so a caller can hand over a session built by
    /// <c>InferenceSession.CreateFromHuggingFaceAsync(..., webTorrent:, http:)</c>, i.e. delivered as a
    /// LAZY-HASH torrent. The <c>byte[]</c> overload below forced every caller to materialise the whole
    /// model on the managed heap first, which is what made this service the last holdout when the demo
    /// moved off the superseded delivery path.
    /// </remarks>
    /// <param name="session">A live session; this service takes ownership and disposes it.</param>
    /// <param name="accelerator">The accelerator the session was built on.</param>
    public void UseSession(InferenceSession session, Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session?.Dispose();
        _pipeline?.Dispose();
        _accelerator = accelerator;
        _session = session;
        _pipeline = new ClassificationPipeline(_session, accelerator);
    }

    /// <summary>Load model from raw bytes for the given accelerator.</summary>
    public Task LoadModelAsync(byte[] modelBytes, Accelerator accelerator)
    {
        _session?.Dispose();
        _pipeline?.Dispose();
        _accelerator = accelerator;

        _session = InferenceSession.CreateFromFile(accelerator, modelBytes);
        _pipeline = new ClassificationPipeline(_session, accelerator);
        return Task.CompletedTask;
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
