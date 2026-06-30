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
    /// Classify raw text using a real tokenizer (e.g. WordPiece for BERT/DistilBERT): wraps
    /// <c>[CLS] + tokenize(text) + [SEP]</c> and runs the GPU classifier. This is the honest
    /// production path — prefer it over <see cref="ClassifySimpleAsync"/> (a whitespace fallback that
    /// drops out-of-vocab words). The CLS/SEP ids come from the loaded tokenizer's resolved special
    /// tokens, falling back to BERT's defaults (101/102) when the tokenizer didn't resolve them.
    /// </summary>
    public async Task<TextClassificationResult> ClassifyAsync(string text, LoadedTokenizer tokenizer)
    {
        int cls = tokenizer.BosTokenId >= 0 ? tokenizer.BosTokenId : CLS_TOKEN;
        int sep = tokenizer.EosTokenId >= 0 ? tokenizer.EosTokenId : SEP_TOKEN;
        var ids = new List<int>(_maxLength) { cls };
        ids.AddRange(tokenizer.Encode(text));
        if (ids.Count > _maxLength - 1) ids = ids.Take(_maxLength - 1).ToList(); // leave room for [SEP]
        ids.Add(sep);
        return await ClassifyAsync(ids.ToArray());
    }

    /// <summary>
    /// Classify with raw text using simple whitespace tokenization + BERT special tokens.
    /// For proper tokenization, use <see cref="ClassifyAsync(string, LoadedTokenizer)"/> (WordPiece).
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

    /// <summary>
    /// Embed raw text using a real tokenizer (e.g. WordPiece for BERT): <c>[CLS] + tokenize(text) +
    /// [SEP]</c> then mean-pool + L2-normalize. The honest path — prefer it over a hash/whitespace
    /// tokenizer. CLS/SEP come from the loaded tokenizer's resolved special tokens (BERT 101/102 fallback).
    /// </summary>
    public async Task<EmbeddingResult> EmbedAsync(string text, LoadedTokenizer tokenizer)
    {
        int cls = tokenizer.BosTokenId >= 0 ? tokenizer.BosTokenId : 101;
        int sep = tokenizer.EosTokenId >= 0 ? tokenizer.EosTokenId : 102;
        var ids = new List<int>(_maxLength) { cls };
        ids.AddRange(tokenizer.Encode(text));
        if (ids.Count > _maxLength - 1) ids = ids.Take(_maxLength - 1).ToList(); // leave room for [SEP]
        ids.Add(sep);
        return await EmbedAsync(ids.ToArray());
    }

    /// <summary>Compute cosine similarity between two pre-tokenized texts.</summary>
    public async Task<float> SimilarityAsync(int[] tokenIdsA, int[] tokenIdsB)
    {
        var embA = await EmbedAsync(tokenIdsA);
        var embB = await EmbedAsync(tokenIdsB);
        return embA.SimilarityTo(embB);
    }

    /// <summary>Cosine similarity between two raw texts using a real tokenizer (WordPiece).</summary>
    public async Task<float> SimilarityAsync(string textA, string textB, LoadedTokenizer tokenizer)
    {
        var embA = await EmbedAsync(textA, tokenizer);
        var embB = await EmbedAsync(textB, tokenizer);
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

    /// <summary>Cache shape-derived runtime constants across the fixed-shape decode steps (skips the
    /// ~643 readback GPU round-trips after step 0). Default on. Exposed so a correctness probe can
    /// compare cached vs uncached output.</summary>
    public bool UseShapeReadbackCache { get; set; } = true;

    /// <summary>DIAGNOSTIC: per-step timing lines captured during the most recent <see cref="GenerateAsync"/>
    /// when <see cref="InferenceSession.VerboseLogging"/> is set. A test can read this (the WASM browser
    /// console is not piped into PMT stdout) to inspect the recompile/forward/readback/pool-churn split.</summary>
    public List<string> StepTimings { get; } = new();

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
    /// <param name="config">Optional sampling configuration (strategy, temperature, top-k/top-p,
    /// repetition penalty, seed). When null the pipeline uses pure GREEDY argmax — this keeps the
    /// deterministic GPT-2==ORT reference tests bit-exact. Sampling is OPT-IN and is what the demo
    /// uses to escape greedy's degenerate repetition loops on a small model like DistilGPT-2.</param>
    /// <param name="onToken">Optional progress callback invoked after EACH generated token with
    /// (tokenCount, decodedTextSoFar). Lets a UI stream tokens live instead of showing nothing until
    /// the whole (currently slow, per-step-recompiling) generation completes. Awaited so a Blazor UI
    /// can <c>StateHasChanged</c> between steps.</param>
    public async Task<TextGenerationResult> GenerateAsync(string prompt, int? maxNewTokens = null,
        GenerationConfig? config = null, Func<int, string, Task>? onToken = null)
    {
        if (_tokenizer == null) throw new InvalidOperationException("Tokenizer not loaded.");

        // Token-count precedence: explicit param > config.MaxNewTokens (only if the caller SET it -
        // it's int? defaulting to null) > this pipeline's MaxNewTokens. The config default is null on
        // purpose: passing a GenerationConfig for sampling must NOT silently override an explicitly-set
        // pipeline.MaxNewTokens. (It used to default to 128, which made sampled generations quietly run
        // 128 tokens regardless of pipeline.MaxNewTokens - which on a slow backend looked like a hang.)
        int maxTokens = maxNewTokens ?? config?.MaxNewTokens ?? MaxNewTokens;
        // Seeded RNG → reproducible sampling (a seeded test asserts identical output across two runs);
        // unseeded → Random.Shared for normal generation. Created once so the whole decode shares a stream.
        var rng = config?.Seed is int seed ? new Random(seed) : Random.Shared;
        StepTimings.Clear();
        var sw = Stopwatch.StartNew();

        // Tokenize prompt
        var promptTokens = _tokenizer.Encode(prompt).ToList();
        var allTokens = new List<int>(promptTokens);

        // FIXED-SHAPE decode: run the decoder at a CONSTANT sequence length (prompt + everything we
        // will generate), right-padding the unused positions. Causal attention makes the logits at
        // the current last-real-token position bit-identical to a variable-length forward — an earlier
        // position never attends to right-padding — so greedy output is unchanged. The payoff: every
        // step reuses ONE shape-specialized executor, so BOTH the per-step recompile AND the ~643
        // shape-readback GPU round-trips/step disappear (the latter cached after step 0 via
        // CacheShapeReadbacks). Measured WebGPU DistilGPT-2: ~13s → ~5.8s per step. The growing-shape
        // loop recompiled + re-read every step (the "20-30 min" generations TJ hit).
        const int GPT2_MAX_POSITIONS = 1024;
        int ctx = Math.Min(promptTokens.Count + maxTokens, GPT2_MAX_POSITIONS);
        const float PadToken = 50256f; // GPT-2 EOS as pad — never read (causal), value is irrelevant.
        var inputNames = _session.InputNames;
        // Shape-derived runtime constants are identical across the fixed-shape steps — cache them so
        // steps 1+ skip the readback round-trips. Safe: this loop only ever feeds one shape.
        _session.CacheShapeReadbacks = UseShapeReadbackCache;

        var idsFloat = new float[ctx];
        var maskFloat = new float[ctx];
        var posFloat = new float[ctx];
        for (int i = 0; i < ctx; i++) { maskFloat[i] = 1f; posFloat[i] = i; }

        for (int step = 0; step < maxTokens; step++)
        {
            int valid = allTokens.Count;       // number of real tokens so far
            if (valid >= ctx) break;           // fixed window full
            int lastRealPos = valid - 1;       // position whose logits predict the next token

            // Only input_ids changes per step; mask/pos are constant. Pad the tail.
            for (int i = 0; i < ctx; i++) idsFloat[i] = i < valid ? allTokens[i] : PadToken;

            using var idsBuf = _accelerator.Allocate1D(idsFloat);
            using var maskBuf = _accelerator.Allocate1D(maskFloat);
            using var posBuf = _accelerator.Allocate1D(posFloat);

            var inputs = new Dictionary<string, Tensor>();
            inputs[inputNames[0]] = new Tensor(idsBuf.View, new[] { 1, ctx });
            if (inputNames.Length > 1)
                inputs[inputNames[1]] = new Tensor(maskBuf.View, new[] { 1, ctx });
            if (inputNames.Length > 2)
                inputs[inputNames[2]] = new Tensor(posBuf.View, new[] { 1, ctx });

            var runSw = Stopwatch.StartNew();
            var outputs = await _session.RunAsync(inputs);
            await _accelerator.SynchronizeAsync(); // flush forward GPU work so the timing attributes it to this step
            runSw.Stop();
            var output = outputs[_session.OutputNames[0]];

            // Get logits at the last REAL token's position (not the padded tail): [1, ctx, vocab].
            int vocabSize = output.Shape.Length >= 3 ? output.Shape[^1] : 50257;
            int seqLen = output.Shape.Length >= 3 ? output.Shape[^2] : 1;
            int lastPos = Math.Min(lastRealPos, seqLen - 1);
            int lastOffset = lastPos * vocabSize;
            // Bounds safety: ensure we don't read past the output buffer
            if (lastOffset + vocabSize > output.ElementCount)
                lastOffset = Math.Max(0, output.ElementCount - vocabSize);

            var readSw = Stopwatch.StartNew();
            using var readBuf = _accelerator.Allocate1D<float>(vocabSize);
            new ElementWiseKernels(_accelerator).Scale(
                output.Data.SubView(lastOffset, vocabSize), readBuf.View, vocabSize, 1f);
            await _accelerator.SynchronizeAsync();
            var logits = await readBuf.CopyToHostAsync<float>(0, vocabSize);
            readSw.Stop();

            // DIAGNOSTIC: attribute per-step decode cost to CPU recompile vs GPU forward vs readback.
            // Always recorded into StepTimings (cheap strings, so a test can inspect them in-process —
            // the primary consumption path — without turning on VerboseLogging); only echoed to the
            // console when VerboseLogging is set. Echo via Console.WriteLine (stdout/console.log), NEVER
            // Console.Error — in Blazor WASM stderr makes the #blazor-error-ui bar appear, which PMT
            // flags as a FAILED test even when the body succeeded. (Tests read StepTimings, not this echo.)
            {
                double recompileMs = _session.LastRecompileMs;
                double runMs = runSw.Elapsed.TotalMilliseconds;
                var line = $"[textgen-timing] step {step} ctx={ctx} valid={valid}: " +
                    $"recompile={recompileMs:F0}ms forward={Math.Max(0, runMs - recompileMs):F0}ms " +
                    $"readback={readSw.Elapsed.TotalMilliseconds:F0}ms (run={runMs:F0}ms) " +
                    $"poolBuffers={_session.LastExecutorBufferCount}";
                StepTimings.Add(line);
                if (InferenceSession.VerboseLogging) Console.WriteLine(line);
            }

            // Pick the next token. Default (config == null) is pure greedy argmax — this keeps the
            // GPT-2==ORT reference tests bit-exact. Sampling (top-k / top-p / temperature / repetition
            // penalty) is opt-in via GenerationConfig; it's what the demo uses so DistilGPT-2 doesn't
            // collapse into "the first time I saw the first time I saw" greedy loops.
            int nextToken;
            if (config == null || config.Strategy == "greedy")
            {
                nextToken = TextGenerationSampler.Greedy(logits);
            }
            else
            {
                // Repetition penalty first (in-place on the CPU logits), THEN strategy sampling.
                if (config.RepetitionPenalty != 1.0f)
                    TextGenerationSampler.ApplyRepetitionPenalty(logits, allTokens.ToArray(), config.RepetitionPenalty);
                nextToken = config.Strategy switch
                {
                    "top_k" => TextGenerationSampler.TopK(logits, config.TopK, config.Temperature, rng),
                    "top_p" => TextGenerationSampler.TopP(logits, config.TopP, config.Temperature, rng),
                    _ => TextGenerationSampler.Greedy(logits),
                };
            }

            // Check for EOS — GPT-2's fixed EOS (50256) always stops; an explicit config EOS also stops.
            if (nextToken == 50256 || (config != null && config.EosTokenId >= 0 && nextToken == config.EosTokenId))
                break;

            allTokens.Add(nextToken);

            // Stream progress to the caller (live token count + partial text) so a UI can show
            // generation advancing instead of a frozen "0 tokens" until the whole run finishes.
            if (onToken != null)
            {
                var soFar = _tokenizer.Decode(allTokens.Skip(promptTokens.Count).ToArray());
                await onToken(allTokens.Count - promptTokens.Count, soFar);
            }
        }

        sw.Stop();

        // Decode generated tokens
        var generatedTokens = allTokens.Skip(promptTokens.Count).ToArray();
        string generatedText = _tokenizer.Decode(generatedTokens);

        return new TextGenerationResult
        {
            Text = prompt + generatedText,
            GeneratedText = generatedText,
            GeneratedTokenIds = generatedTokens,
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
    /// <summary>The newly generated token IDs (excludes the prompt), in order. Lets callers compare
    /// against a reference greedy decode (e.g. an ORT-produced fixture) without re-tokenizing text.</summary>
    public int[] GeneratedTokenIds { get; init; } = Array.Empty<int>();
    public int PromptTokenCount { get; init; }
    public int GeneratedTokenCount { get; init; }
    public int TotalTokenCount { get; init; }
    public double InferenceTimeMs { get; init; }
    public double TokensPerSecond { get; init; }
}
