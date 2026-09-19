using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

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
    /// Classify an RGBA image (managed pixels — desktop / oracle). Returns top-K predictions sorted by confidence.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="ClassifyAsync(TypedArray, int, int, int)"/> or a GPU-view overload.
    /// </remarks>
    public async Task<ClassificationResult[]> ClassifyAsync(
        int[] rgbaPixels, int width, int height, int topK = 5)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await ClassifyAsync(rgbaBuf.View, width, height, topK).ConfigureAwait(false);
    }

    /// <summary>Browser path — JS typed array → GPU without a managed heap crossing.</summary>
    public async Task<ClassificationResult[]> ClassifyAsync(
        TypedArray rgbaPixels, int width, int height, int topK = 5)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await ClassifyAsync(rgbaBuf.View, width, height, topK).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public async Task<ClassificationResult[]> ClassifyAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height, int topK = 5)
    {
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize);

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });

        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        var output = outputs[_session.OutputNames[0]];
        int numClasses = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(numClasses);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, numClasses), readBuf.View, numClasses, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var logits = await readBuf.CopyToHostAsync<float>(0, numClasses).ConfigureAwait(false);

        float outputSum = logits.Sum();
        bool alreadySoftmaxed = outputSum > 0.9f && outputSum < 1.1f && logits.All(v => v >= 0f);

        return TopK(logits, topK, applySoftmax: !alreadySoftmaxed);
    }

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<ClassificationResult[]> ClassifyAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height, int topK = 5)
        => ClassifyAsync(rgbaPixels.View, width, height, topK);

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
            probs = logits;
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
