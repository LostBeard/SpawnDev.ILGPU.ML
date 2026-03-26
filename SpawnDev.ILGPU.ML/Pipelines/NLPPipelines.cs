using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Text Classification / Sentiment Analysis: text → (label, score) predictions.
/// Models: DistilBERT-SST2 (positive/negative sentiment).
///
/// Usage:
///   var pipeline = new TextClassificationPipeline(session, accelerator);
///   var result = await pipeline.ClassifyAsync("I love this movie!");
///   Console.WriteLine($"{result.TopLabel}: {result.TopConfidence:P1}");
/// </summary>
public class TextClassificationPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly string[] _labels;
    private readonly int _maxLength;

    // DistilBERT BERT-style token IDs (WordPiece)
    private const int CLS_TOKEN = 101;
    private const int SEP_TOKEN = 102;
    private const int PAD_TOKEN = 0;

    public bool IsReady => true;
    public string ModelName { get; init; } = "DistilBERT-SST2";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    public TextClassificationPipeline(InferenceSession session, Accelerator accelerator,
        int maxLength = 128, string[]? labels = null)
    {
        _session = session;
        _accelerator = accelerator;
        _maxLength = maxLength;
        _labels = labels ?? new[] { "NEGATIVE", "POSITIVE" };
    }

    /// <summary>
    /// Classify text with pre-tokenized input (token IDs as ints).
    /// </summary>
    public async Task<TextClassificationResult> ClassifyAsync(int[] tokenIds)
    {
        var sw = Stopwatch.StartNew();

        // Pad or truncate to maxLength
        var padded = TextPreprocessor.PadOrTruncate(tokenIds, _maxLength, PAD_TOKEN);
        var mask = TextPreprocessor.CreateAttentionMask(padded, PAD_TOKEN);

        // Convert to float (our engine uses float32 tensors)
        var idsFloat = padded.Select(t => (float)t).ToArray();
        var maskFloat = mask.Select(m => (float)m).ToArray();

        using var idsBuf = _accelerator.Allocate1D(idsFloat);
        using var maskBuf = _accelerator.Allocate1D(maskFloat);

        var inputs = new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(idsBuf.View, new[] { 1, _maxLength }),
            [_session.InputNames[1]] = new Tensor(maskBuf.View, new[] { 1, _maxLength }),
        };

        var outputs = await _session.RunAsync(inputs);
        var output = outputs[_session.OutputNames[0]];

        // Read logits (num_labels values)
        int numLabels = _labels.Length;
        using var readBuf = _accelerator.Allocate1D<float>(numLabels);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, numLabels), readBuf.View, numLabels, 1f);
        await _accelerator.SynchronizeAsync();
        var logits = await readBuf.CopyToHostAsync<float>(0, numLabels);

        // Softmax
        var probs = TextPreprocessor.Softmax(logits);

        sw.Stop();

        // Build ranked predictions
        var predictions = probs
            .Select((p, i) => new ClassPrediction
            {
                Label = i < _labels.Length ? _labels[i] : $"class_{i}",
                ClassId = i,
                Confidence = p,
            })
            .OrderByDescending(p => p.Confidence)
            .ToArray();

        return new TextClassificationResult
        {
            Predictions = predictions,
            Logits = logits,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    /// <summary>
    /// Classify with raw text using simple whitespace tokenization + BERT special tokens.
    /// For proper tokenization, use ClassifyAsync(int[] tokenIds) with a WordPiece tokenizer.
    /// </summary>
    public async Task<TextClassificationResult> ClassifySimpleAsync(string text, Dictionary<string, int>? vocab = null)
    {
        // Simple fallback: CLS + word tokens + SEP
        // For real use, a WordPiece tokenizer should be used
        var tokens = new List<int> { CLS_TOKEN };
        if (vocab != null)
        {
            foreach (var word in text.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (vocab.TryGetValue(word, out int id))
                    tokens.Add(id);
            }
        }
        tokens.Add(SEP_TOKEN);
        return await ClassifyAsync(tokens.ToArray());
    }

    public async Task<TextClassificationResult> RunAsync(string text) =>
        await ClassifySimpleAsync(text);

    public void Dispose() => _session?.Dispose();
}

/// <summary>Result from text classification with logits.</summary>
public class TextClassificationResult
{
    /// <summary>Ranked predictions, highest confidence first.</summary>
    public ClassPrediction[] Predictions { get; init; } = Array.Empty<ClassPrediction>();
    /// <summary>Raw model logits before softmax.</summary>
    public float[] Logits { get; init; } = Array.Empty<float>();
    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }
    /// <summary>Top prediction label.</summary>
    public string TopLabel => Predictions.Length > 0 ? Predictions[0].Label : "";
    /// <summary>Top prediction confidence.</summary>
    public float TopConfidence => Predictions.Length > 0 ? Predictions[0].Confidence : 0;
}

/// <summary>
/// Feature Extraction / Embeddings: text → dense vector via mean pooling.
/// Works with any BERT-like model that outputs last_hidden_state.
///
/// Usage:
///   var pipeline = new FeatureExtractionPipeline(session, accelerator);
///   var embA = await pipeline.EmbedAsync(new[] { 101, 7592, 2088, 102 });
///   var embB = await pipeline.EmbedAsync(new[] { 101, 3407, 2154, 102 });
///   float sim = TextPreprocessor.CosineSimilarity(embA.Embedding, embB.Embedding);
/// </summary>
public class FeatureExtractionPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private readonly int _maxLength;
    private readonly int _hiddenSize;

    private const int PAD_TOKEN = 0;

    public FeatureExtractionPipeline(InferenceSession session, Accelerator accelerator,
        int maxLength = 128, int hiddenSize = 768)
    {
        _session = session;
        _accelerator = accelerator;
        _maxLength = maxLength;
        _hiddenSize = hiddenSize;
    }

    /// <summary>
    /// Embed pre-tokenized text to a dense vector via mean pooling + L2 normalization.
    /// </summary>
    public async Task<EmbeddingResult> EmbedAsync(int[] tokenIds)
    {
        var sw = Stopwatch.StartNew();

        var padded = TextPreprocessor.PadOrTruncate(tokenIds, _maxLength, PAD_TOKEN);
        var mask = TextPreprocessor.CreateAttentionMask(padded, PAD_TOKEN);
        int realTokenCount = mask.Count(m => m == 1);

        var idsFloat = padded.Select(t => (float)t).ToArray();
        var maskFloat = mask.Select(m => (float)m).ToArray();

        using var idsBuf = _accelerator.Allocate1D(idsFloat);
        using var maskBuf = _accelerator.Allocate1D(maskFloat);

        var inputs = new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(idsBuf.View, new[] { 1, _maxLength }),
            [_session.InputNames[1]] = new Tensor(maskBuf.View, new[] { 1, _maxLength }),
        };

        var outputs = await _session.RunAsync(inputs);
        var output = outputs[_session.OutputNames[0]];

        // Output shape: [1, seq_len, hidden_size]
        // Mean pool over real (non-padded) token positions
        int totalFloats = Math.Min(output.ElementCount, _maxLength * _hiddenSize);
        using var readBuf = _accelerator.Allocate1D<float>(totalFloats);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, totalFloats), readBuf.View, totalFloats, 1f);
        await _accelerator.SynchronizeAsync();
        var hiddenStates = await readBuf.CopyToHostAsync<float>(0, totalFloats);

        // Mean pooling: average hidden states across token positions (masked)
        var embedding = new float[_hiddenSize];
        if (realTokenCount > 0)
        {
            for (int t = 0; t < realTokenCount && t < _maxLength; t++)
            {
                int offset = t * _hiddenSize;
                for (int h = 0; h < _hiddenSize && offset + h < hiddenStates.Length; h++)
                    embedding[h] += hiddenStates[offset + h];
            }
            for (int h = 0; h < _hiddenSize; h++)
                embedding[h] /= realTokenCount;
        }

        // L2 normalize
        float norm = 0;
        for (int i = 0; i < embedding.Length; i++) norm += embedding[i] * embedding[i];
        norm = MathF.Sqrt(norm);
        if (norm > 1e-12f)
            for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;

        sw.Stop();

        return new EmbeddingResult
        {
            Embedding = embedding,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    /// <summary>Compute cosine similarity between two pre-tokenized texts.</summary>
    public async Task<float> SimilarityAsync(int[] tokenIdsA, int[] tokenIdsB)
    {
        var embA = await EmbedAsync(tokenIdsA);
        var embB = await EmbedAsync(tokenIdsB);
        return embA.SimilarityTo(embB);
    }

    public void Dispose() => _session?.Dispose();
}

/// <summary>
/// Text Generation pipeline with autoregressive decoding.
/// Works with any causal LM that outputs [1, seq, vocab] logits.
///
/// Usage:
///   var pipeline = new TextGenerationPipeline(session, accelerator);
///   pipeline.LoadTokenizer(tokenizerJson);
///   var result = await pipeline.GenerateAsync("The cat sat on the", maxNewTokens: 20);
///   Console.WriteLine(result.Text);
/// </summary>
public class TextGenerationPipeline : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly InferenceSession _session;
    private BPETokenizer? _tokenizer;

    public int MaxNewTokens { get; set; } = 50;
    public float Temperature { get; set; } = 1.0f;

    public TextGenerationPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
    }

    public void LoadTokenizer(string tokenizerJson)
    {
        _tokenizer = BPETokenizer.LoadFromTokenizerJson(tokenizerJson);
    }

    /// <summary>
    /// Generate text from a prompt using greedy decoding.
    /// </summary>
    public async Task<TextGenerationResult> GenerateAsync(string prompt, int? maxNewTokens = null)
    {
        if (_tokenizer == null) throw new InvalidOperationException("Tokenizer not loaded.");

        int maxTokens = maxNewTokens ?? MaxNewTokens;
        var sw = Stopwatch.StartNew();

        // Tokenize prompt
        var promptTokens = _tokenizer.Encode(prompt).ToList();
        var allTokens = new List<int>(promptTokens);

        for (int step = 0; step < maxTokens; step++)
        {
            // Create input tensors
            var idsFloat = allTokens.Select(t => (float)t).ToArray();
            var maskFloat = Enumerable.Repeat(1f, allTokens.Count).ToArray();
            var posFloat = Enumerable.Range(0, allTokens.Count).Select(i => (float)i).ToArray();

            using var idsBuf = _accelerator.Allocate1D(idsFloat);
            using var maskBuf = _accelerator.Allocate1D(maskFloat);
            using var posBuf = _accelerator.Allocate1D(posFloat);

            var inputs = new Dictionary<string, Tensor>();
            var inputNames = _session.InputNames;
            inputs[inputNames[0]] = new Tensor(idsBuf.View, new[] { 1, allTokens.Count });
            if (inputNames.Length > 1)
                inputs[inputNames[1]] = new Tensor(maskBuf.View, new[] { 1, allTokens.Count });
            if (inputNames.Length > 2)
                inputs[inputNames[2]] = new Tensor(posBuf.View, new[] { 1, allTokens.Count });

            var outputs = await _session.RunAsync(inputs);
            var output = outputs[_session.OutputNames[0]];

            // Get logits for last position: [1, seq, vocab] → last position
            int vocabSize = output.Shape.Length >= 3 ? output.Shape[^1] : 50257;
            int lastOffset = (allTokens.Count - 1) * vocabSize;

            using var readBuf = _accelerator.Allocate1D<float>(vocabSize);
            new ElementWiseKernels(_accelerator).Scale(
                output.Data.SubView(lastOffset, vocabSize), readBuf.View, vocabSize, 1f);
            await _accelerator.SynchronizeAsync();
            var logits = await readBuf.CopyToHostAsync<float>(0, vocabSize);

            // Greedy argmax
            int nextToken = 0;
            float maxVal = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxVal) { maxVal = logits[i]; nextToken = i; }
            }

            // Check for EOS
            if (nextToken == 50256) break; // GPT-2 EOS token

            allTokens.Add(nextToken);
        }

        sw.Stop();

        // Decode generated tokens
        var generatedTokens = allTokens.Skip(promptTokens.Count).ToArray();
        string generatedText = _tokenizer.Decode(generatedTokens);

        return new TextGenerationResult
        {
            Text = prompt + generatedText,
            GeneratedText = generatedText,
            PromptTokenCount = promptTokens.Count,
            GeneratedTokenCount = generatedTokens.Length,
            TotalTokenCount = allTokens.Count,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            TokensPerSecond = generatedTokens.Length / (sw.Elapsed.TotalSeconds + 1e-9),
        };
    }

    public void Dispose() => _session?.Dispose();
}

/// <summary>Result from text generation.</summary>
public class TextGenerationResult
{
    public string Text { get; init; } = "";
    public string GeneratedText { get; init; } = "";
    public int PromptTokenCount { get; init; }
    public int GeneratedTokenCount { get; init; }
    public int TotalTokenCount { get; init; }
    public double InferenceTimeMs { get; init; }
    public double TokensPerSecond { get; init; }
}
