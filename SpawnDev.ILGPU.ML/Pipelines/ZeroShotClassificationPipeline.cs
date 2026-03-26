using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Zero-shot image classification using CLIP (Contrastive Language-Image Pre-training).
/// Encodes an image and N text labels into a shared embedding space, then ranks by cosine similarity.
///
/// Usage:
///   var pipeline = new ZeroShotClassificationPipeline(visionSession, textSession, accelerator);
///   pipeline.LoadTokenizer(tokenizerJson);
///   var result = await pipeline.ClassifyAsync(rgbaPixels, width, height,
///       new[] { "a photo of a cat", "a photo of a dog", "a sunset" });
///   Console.WriteLine($"Best match: {result.Predictions[0].Label} ({result.Predictions[0].Confidence:P1})");
/// </summary>
public class ZeroShotClassificationPipeline : IDisposable
{
    private readonly InferenceSession _visionSession;
    private readonly InferenceSession _textSession;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private BPETokenizer? _tokenizer;
    private readonly int _inputSize;
    private readonly int _embeddingDim;
    private readonly float _logitScale;

    // CLIP-specific normalization (NOT ImageNet!)
    private static readonly float[] ClipMean = { 0.48145466f, 0.4578275f, 0.40821073f };
    private static readonly float[] ClipStd = { 0.26862954f, 0.26130258f, 0.27577711f };

    public ZeroShotClassificationPipeline(
        InferenceSession visionSession,
        InferenceSession textSession,
        Accelerator accelerator,
        int inputSize = 224,
        int embeddingDim = 512,
        float logitScale = 100f)
    {
        _visionSession = visionSession;
        _textSession = textSession;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _inputSize = inputSize;
        _embeddingDim = embeddingDim;
        _logitScale = logitScale;
    }

    /// <summary>
    /// Load the CLIP tokenizer from HuggingFace tokenizer.json format.
    /// </summary>
    public void LoadTokenizer(string tokenizerJson)
    {
        _tokenizer = BPETokenizer.LoadFromTokenizerJson(tokenizerJson);
    }

    /// <summary>
    /// Classify an image against N text descriptions.
    /// Returns predictions ranked by similarity.
    /// </summary>
    public async Task<ZeroShotResult> ClassifyAsync(
        int[] rgbaPixels, int width, int height,
        string[] textDescriptions)
    {
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer not loaded. Call LoadTokenizer() first.");

        var sw = Stopwatch.StartNew();

        // Encode image
        var imageEmbedding = await EncodeImageAsync(rgbaPixels, width, height);

        // Encode all text descriptions
        var textEmbeddings = new float[textDescriptions.Length][];
        for (int i = 0; i < textDescriptions.Length; i++)
        {
            textEmbeddings[i] = await EncodeTextAsync(textDescriptions[i]);
        }

        // Compute cosine similarities and apply logit scale
        var similarities = new float[textDescriptions.Length];
        for (int i = 0; i < textDescriptions.Length; i++)
        {
            similarities[i] = TextPreprocessor.CosineSimilarity(imageEmbedding, textEmbeddings[i]) * _logitScale;
        }

        // Softmax over similarities
        var probs = TextPreprocessor.Softmax(similarities);

        sw.Stop();

        var predictions = probs
            .Select((p, i) => new ClassPrediction
            {
                Label = textDescriptions[i],
                ClassId = i,
                Confidence = p,
            })
            .OrderByDescending(p => p.Confidence)
            .ToArray();

        return new ZeroShotResult
        {
            Predictions = predictions,
            Similarities = similarities,
            ImageEmbedding = imageEmbedding,
            TextEmbeddings = textEmbeddings,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    /// <summary>
    /// Encode an image to a normalized embedding vector.
    /// </summary>
    public async Task<float[]> EncodeImageAsync(int[] rgbaPixels, int width, int height)
    {
        // Preprocess: RGBA → NCHW with CLIP normalization
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize,
            ClipMean, ClipStd);

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });

        var outputs = await _visionSession.RunAsync(new Dictionary<string, Tensor>
        {
            [_visionSession.InputNames[0]] = inputTensor
        });

        var output = outputs[_visionSession.OutputNames[0]];
        int elems = Math.Min(output.ElementCount, _embeddingDim);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync();
        var embedding = await readBuf.CopyToHostAsync<float>(0, elems);

        // L2 normalize
        return L2Normalize(embedding);
    }

    /// <summary>
    /// Encode a text description to a normalized embedding vector.
    /// </summary>
    public async Task<float[]> EncodeTextAsync(string text)
    {
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer not loaded.");

        // Tokenize with CLIP special tokens: [SOT] + tokens + [EOT] + padding to 77
        var tokenIds = _tokenizer.EncodeCLIP(text, maxLength: 77);

        // Convert to float (our engine uses float32 tensors)
        var tokenFloats = tokenIds.Select(t => (float)t).ToArray();

        using var tokenBuf = _accelerator.Allocate1D(tokenFloats);
        var inputTensor = new Tensor(tokenBuf.View, new[] { 1, 77 });

        var outputs = await _textSession.RunAsync(new Dictionary<string, Tensor>
        {
            [_textSession.InputNames[0]] = inputTensor
        });

        var output = outputs[_textSession.OutputNames[0]];
        int elems = Math.Min(output.ElementCount, _embeddingDim);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync();
        var embedding = await readBuf.CopyToHostAsync<float>(0, elems);

        return L2Normalize(embedding);
    }

    private static float[] L2Normalize(float[] vector)
    {
        float norm = 0;
        for (int i = 0; i < vector.Length; i++)
            norm += vector[i] * vector[i];
        norm = MathF.Sqrt(norm);
        if (norm < 1e-12f) return vector;
        var result = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
            result[i] = vector[i] / norm;
        return result;
    }

    public void Dispose()
    {
        _visionSession?.Dispose();
        _textSession?.Dispose();
    }
}

/// <summary>Result from zero-shot classification.</summary>
public class ZeroShotResult
{
    public ClassPrediction[] Predictions { get; init; } = Array.Empty<ClassPrediction>();
    public float[] Similarities { get; init; } = Array.Empty<float>();
    public float[] ImageEmbedding { get; init; } = Array.Empty<float>();
    public float[][] TextEmbeddings { get; init; } = Array.Empty<float[]>();
    public double InferenceTimeMs { get; init; }
    public string TopLabel => Predictions.Length > 0 ? Predictions[0].Label : "";
    public float TopConfidence => Predictions.Length > 0 ? Predictions[0].Confidence : 0;
}
