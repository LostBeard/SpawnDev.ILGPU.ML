using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Text Classification / Sentiment Analysis: text → (label, score) predictions.
/// Models: DistilBERT, BERT, RoBERTa.
/// </summary>
public class TextClassificationPipeline : IPipeline<string, ClassificationResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _session;
    private BPETokenizer? _tokenizer;

    public bool IsReady => _session != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    private TextClassificationPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<TextClassificationPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new TextClassificationPipeline(accelerator);
        var path = modelId ?? options.ModelPath ?? "models/distilbert-sst2";
        pipe.ModelName = path;
        pipe._session = await InferenceSession.CreateAsync(accelerator, http, path);
        // TODO: Load tokenizer vocab + merges from model directory
        return pipe;
    }

    public async Task<ClassificationResult> RunAsync(string text)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded");
        // Pipeline: tokenize → pad → create attention mask → inference → softmax → labels
        throw new NotImplementedException("Awaiting tokenizer + InferenceSession integration");
    }

    public void Dispose() => _session?.Dispose();
}

/// <summary>
/// Feature Extraction / Embeddings: text → dense vector.
/// Models: all-MiniLM-L6-v2, BGE, E5.
/// </summary>
public class FeatureExtractionPipeline : IPipeline<string, EmbeddingResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _session;
    private BPETokenizer? _tokenizer;

    public bool IsReady => _session != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();

    private FeatureExtractionPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<FeatureExtractionPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new FeatureExtractionPipeline(accelerator);
        var path = modelId ?? options.ModelPath ?? "models/all-minilm-l6-v2";
        pipe.ModelName = path;
        pipe._session = await InferenceSession.CreateAsync(accelerator, http, path);
        return pipe;
    }

    public async Task<EmbeddingResult> RunAsync(string text)
    {
        if (_session == null) throw new InvalidOperationException("Model not loaded");
        // Pipeline: tokenize → pad → attention mask → inference → mean pooling → L2 normalize
        throw new NotImplementedException("Awaiting tokenizer + InferenceSession integration");
    }

    /// <summary>Compute similarity between two texts.</summary>
    public async Task<float> SimilarityAsync(string textA, string textB)
    {
        var embA = await RunAsync(textA);
        var embB = await RunAsync(textB);
        return embA.SimilarityTo(embB);
    }

    public void Dispose() => _session?.Dispose();
}

/// <summary>
/// Text Generation: prompt → generated text.
/// Models: GPT-2, Qwen, SmolLM, LLaMA.
/// </summary>
public class TextGenerationPipeline : IPipeline<string, TranscriptionResult>
{
    private readonly Accelerator _accelerator;
    private InferenceSession? _session;
    private BPETokenizer? _tokenizer;

    public bool IsReady => _session != null;
    public string ModelName { get; private set; } = "";
    public string BackendName => _accelerator.AcceleratorType.ToString();
    public GenerationConfig Config { get; set; } = new();

    private TextGenerationPipeline(Accelerator accelerator) => _accelerator = accelerator;

    public static async Task<TextGenerationPipeline> CreateAsync(
        Accelerator accelerator, HttpClient http, string? modelId, PipelineOptions options)
    {
        var pipe = new TextGenerationPipeline(accelerator);
        var path = modelId ?? options.ModelPath ?? "models/gpt2";
        pipe.ModelName = path;
        pipe._session = await InferenceSession.CreateAsync(accelerator, http, path);
        return pipe;
    }

    public async Task<TranscriptionResult> RunAsync(string prompt)
    {
        if (_session == null || _tokenizer == null) throw new InvalidOperationException("Model not loaded");
        // Pipeline: tokenize → autoregressive loop (KV cache + sampling) → decode tokens
        throw new NotImplementedException("Awaiting KV cache + autoregressive generation support");
    }

    public void Dispose() => _session?.Dispose();
}
