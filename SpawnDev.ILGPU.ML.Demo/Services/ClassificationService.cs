using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Services;

/// <summary>
/// Image classification service using InferenceSession.
/// Manages model loading, image preprocessing, and inference.
/// </summary>
public class ClassificationService : IDisposable
{
    private readonly HttpClient _http;
    private InferenceSession? _session;
    private Accelerator? _accelerator;
    private Context? _context;
    private BufferPool? _pool;

    public bool IsModelLoaded => _session != null;
    public string? CurrentBackend { get; private set; }

    public ClassificationService(HttpClient http)
    {
        _http = http;
    }

    /// <summary>
    /// Load model for a specific backend. Disposes previous accelerator if switching.
    /// </summary>
    public async Task LoadModelAsync(string modelPath, Accelerator accelerator)
    {
        _session?.Dispose();
        _pool?.Dispose();

        _accelerator = accelerator;
        _pool = new BufferPool(accelerator);

        _session = await InferenceSession.CreateAsync(accelerator, _http, modelPath);
    }

    /// <summary>
    /// Run classification on RGBA pixel data.
    /// Returns top-K predictions sorted by confidence.
    /// </summary>
    public async Task<(ClassificationResult[] predictions, double inferenceMs)> ClassifyAsync(
        int[] rgbaPixels, int width, int height, int topK = 5)
    {
        if (_session == null || _accelerator == null || _pool == null)
            throw new InvalidOperationException("Model not loaded");

        // Upload RGBA pixels to GPU
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);

        // Preprocess: resize to 224x224, normalize with ImageNet stats, NCHW layout
        const int modelSize = 224;
        using var preprocessed = _accelerator.Allocate1D<float>(3 * modelSize * modelSize);
        var preprocess = new SpawnDev.ILGPU.ML.Kernels.ImagePreprocessKernel(_accelerator);
        preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, modelSize, modelSize);

        // Create input tensor
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, modelSize, modelSize }, "input");

        // Run inference
        var sw = Stopwatch.StartNew();

        // Find the actual input name from the session
        var inputName = _session.InputNames[0];
        var outputs = _session.Run(new Dictionary<string, Tensor> { [inputName] = inputTensor });
        await _accelerator.SynchronizeAsync();

        sw.Stop();
        double inferenceMs = sw.Elapsed.TotalMilliseconds;

        // Get output tensor
        var outputName = _session.OutputNames[0];
        var output = outputs[outputName];

        // Read back logits
        var logits = await ReadTensorAsync(output);

        // Softmax on CPU (small array, not worth GPU kernel)
        var probs = CpuSoftmax(logits);

        // Top-K
        var labels = SpawnDev.ILGPU.ML.Data.ImageNetLabels.Labels;
        var predictions = probs
            .Select((p, i) => new ClassificationResult { Label = i < labels.Length ? labels[i] : $"class_{i}", Confidence = p, Index = i })
            .OrderByDescending(p => p.Confidence)
            .Take(topK)
            .ToArray();

        return (predictions, inferenceMs);
    }

    private async Task<float[]> ReadTensorAsync(Tensor tensor)
    {
        using var readBuf = _accelerator!.Allocate1D<float>(tensor.ElementCount);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(tensor.Data.SubView(0, tensor.ElementCount), readBuf.View, tensor.ElementCount, 1f);
        await _accelerator.SynchronizeAsync();
        return await readBuf.CopyToHostAsync<float>(0, tensor.ElementCount);
    }

    private static float[] CpuSoftmax(float[] logits)
    {
        float max = logits.Max();
        var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        float sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    public void Dispose()
    {
        _session?.Dispose();
        _pool?.Dispose();
    }
}

public class ClassificationResult
{
    public string Label { get; set; } = "";
    public float Confidence { get; set; }
    public int Index { get; set; }
}
