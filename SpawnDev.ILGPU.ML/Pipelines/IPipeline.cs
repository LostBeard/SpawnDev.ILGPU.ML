namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Common interface for all ML pipelines.
/// TInput is the raw input type (int[] pixels, float[] audio, string text).
/// TOutput is the structured result type.
/// </summary>
public interface IPipeline<TInput, TOutput> : IDisposable
{
    bool IsReady { get; }
    string ModelName { get; }
    string BackendName { get; }
    Task<TOutput> RunAsync(TInput input);
}

/// <summary>
/// Options for pipeline creation. Allows customizing model path,
/// preprocessing, and inference settings.
/// </summary>
public class PipelineOptions
{
    /// <summary>Path to the model directory (relative to HTTP base or filesystem).</summary>
    public string? ModelPath { get; set; }

    /// <summary>Maximum sequence length for text models.</summary>
    public int MaxLength { get; set; } = 512;

    /// <summary>Top-K for classification/generation results.</summary>
    public int TopK { get; set; } = 5;

    /// <summary>Confidence threshold for detection/classification.</summary>
    public float Threshold { get; set; } = 0.5f;

    /// <summary>Temperature for text generation sampling.</summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>Input image size for vision models.</summary>
    public int InputSize { get; set; } = 224;
}
