using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from image classification — label, confidence, and class index.
/// </summary>
public record ClassificationResult(string Label, float Confidence, int ClassIndex);

/// <summary>
/// High-level image classification pipeline.
/// Wraps InferenceSession with image preprocessing and postprocessing.
///
/// Usage:
///   var pipeline = new ClassificationPipeline(session, accelerator);
///   var results = await pipeline.ClassifyAsync(rgbaPixels, width, height);
///   Console.WriteLine($"Top: {results[0].Label} ({results[0].Confidence:P1})");
/// </summary>
public class ClassificationPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly string[] _labels;
    private readonly int _inputSize;

    public ClassificationPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 224, string[]? labels = null)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _labels = labels ?? Data.ImageNetLabels.Labels;
        _inputSize = inputSize;
    }

    /// <summary>
    /// Classify an RGBA image. Returns top-K predictions sorted by confidence.
    /// </summary>
    public async Task<ClassificationResult[]> ClassifyAsync(
        int[] rgbaPixels, int width, int height, int topK = 5)
    {
        // Upload and preprocess
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        // Create input tensor
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });

        // Run inference
        var outputs = _session.Run(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        // Read logits
        var output = outputs[_session.OutputNames[0]];
        int numClasses = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(numClasses);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, numClasses), readBuf.View, numClasses, 1f);
        await _accelerator.SynchronizeAsync();
        var logits = await readBuf.CopyToHostAsync<float>(0, numClasses);

        // Check if output is already softmax'd (values sum to ~1.0)
        // If so, skip softmax to avoid double-softmax flattening
        float outputSum = logits.Sum();
        bool alreadySoftmaxed = outputSum > 0.9f && outputSum < 1.1f && logits.All(v => v >= 0f);

        return TopK(logits, topK, applySoftmax: !alreadySoftmaxed);
    }

    private ClassificationResult[] TopK(float[] logits, int k, bool applySoftmax = true)
    {
        float[] probs;
        if (applySoftmax)
        {
            float max = logits.Max();
            var exps = new float[logits.Length];
            float sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                exps[i] = MathF.Exp(logits[i] - max);
                sum += exps[i];
            }
            probs = exps.Select(e => e / sum).ToArray();
        }
        else
        {
            probs = logits; // Already probabilities
        }

        var results = new (int Index, float Prob)[probs.Length];
        for (int i = 0; i < probs.Length; i++)
            results[i] = (i, probs[i]);

        Array.Sort(results, (a, b) => b.Prob.CompareTo(a.Prob));

        return results.Take(k).Select(r =>
            new ClassificationResult(
                r.Index < _labels.Length ? _labels[r.Index] : $"class_{r.Index}",
                r.Prob,
                r.Index
            )).ToArray();
    }

    public void Dispose() { }
}
