using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Registry of available demo models with their configurations.
/// Models are loaded from HuggingFace CDN (cached in OPFS) or direct URLs.
/// No models are bundled in the application — everything is fetched on demand.
/// </summary>
public static class ModelRegistry
{
    public static readonly ModelEntry[] AllModels = new[]
    {
        // ── Classification ──
        new ModelEntry
        {
            Id = "squeezenet",
            Name = "SqueezeNet 1.1",
            Task = "image-classification",
            HuggingFaceRepo = ModelHub.KnownModels.SqueezeNet,
            HuggingFaceFile = "squeezenet1.1-7.onnx",
            SizeMB = 4.8,
            Description = "Fast 1000-class ImageNet classification",
            InputWidth = 224, InputHeight = 224,
        },
        new ModelEntry
        {
            Id = "mobilenetv2",
            Name = "MobileNetV2",
            Task = "image-classification",
            HuggingFaceRepo = ModelHub.KnownModels.MobileNetV2,
            HuggingFaceFile = "mobilenetv2-12.onnx",
            SizeMB = 14,
            Description = "Accurate 1000-class ImageNet classification",
            InputWidth = 224, InputHeight = 224,
        },

        // ── Depth Estimation ──
        new ModelEntry
        {
            Id = "depth-anything-v2-small",
            Name = "Depth Anything V2 Small",
            Task = "depth-estimation",
            HuggingFaceRepo = ModelHub.KnownModels.DepthAnythingV2Small,
            HuggingFaceFile = ModelHub.KnownFiles.OnnxModel,
            SizeMB = 95,
            Description = "Monocular depth estimation (26M params)",
            InputWidth = 518, InputHeight = 518,
        },

        // ── Style Transfer (ONNX Model Zoo on HuggingFace) ──
        new ModelEntry
        {
            Id = "style-mosaic",
            Name = "Mosaic",
            Task = "style-transfer",
            HuggingFaceRepo = ModelHub.KnownModels.StyleMosaic,
            HuggingFaceFile = "mosaic-9.onnx",
            SizeMB = 6.5,
            Description = "Mosaic tile artistic style",
        },
        new ModelEntry
        {
            Id = "style-candy",
            Name = "Candy",
            Task = "style-transfer",
            HuggingFaceRepo = ModelHub.KnownModels.StyleCandy,
            HuggingFaceFile = "candy-9.onnx",
            SizeMB = 6.5,
            Description = "Bright candy-colored artistic style",
        },
        new ModelEntry
        {
            Id = "style-rain-princess",
            Name = "Rain Princess",
            Task = "style-transfer",
            HuggingFaceRepo = ModelHub.KnownModels.StyleRainPrincess,
            HuggingFaceFile = "rain-princess-9.onnx",
            SizeMB = 6.5,
            Description = "Impressionist rain scene style",
        },
        new ModelEntry
        {
            Id = "style-udnie",
            Name = "Udnie",
            Task = "style-transfer",
            HuggingFaceRepo = ModelHub.KnownModels.StyleUdnie,
            HuggingFaceFile = "udnie-9.onnx",
            SizeMB = 6.5,
            Description = "Abstract cubist style",
        },
        new ModelEntry
        {
            Id = "style-pointilism",
            Name = "Pointilism",
            Task = "style-transfer",
            HuggingFaceRepo = ModelHub.KnownModels.StylePointilism,
            HuggingFaceFile = "pointilism-9.onnx",
            SizeMB = 6.5,
            Description = "Pointillist dot painting style",
        },

        // ── Super Resolution (ONNX Model Zoo on HuggingFace) ──
        new ModelEntry
        {
            Id = "super-resolution",
            Name = "ESPCN (3x)",
            Task = "super-resolution",
            HuggingFaceRepo = ModelHub.KnownModels.SuperResolution,
            HuggingFaceFile = "super-resolution-10.onnx",
            SizeMB = 0.2,
            Description = "3x upscaling (224x224 → 672x672)",
            InputWidth = 224, InputHeight = 224,
        },

        // ── Pose Estimation ──
        new ModelEntry
        {
            Id = "movenet-lightning",
            Name = "MoveNet Lightning",
            Task = "pose-estimation",
            HuggingFaceRepo = ModelHub.KnownModels.MoveNetLightning,
            HuggingFaceFile = ModelHub.KnownFiles.OnnxModel,
            SizeMB = 9,
            Description = "Fast 17-keypoint pose detection",
            InputWidth = 192, InputHeight = 192,
        },
    };

    /// <summary>Get all models for a specific task.</summary>
    public static ModelEntry[] GetModelsForTask(string task) =>
        AllModels.Where(m => m.Task == task).ToArray();

    /// <summary>Get a model by ID.</summary>
    public static ModelEntry? GetModel(string id) =>
        AllModels.FirstOrDefault(m => m.Id == id);

    /// <summary>Get the default model for a task.</summary>
    public static ModelEntry? GetDefaultForTask(string task) =>
        AllModels.FirstOrDefault(m => m.Task == task);

    /// <summary>Get all unique task types.</summary>
    public static string[] GetAvailableTasks() =>
        AllModels.Select(m => m.Task).Distinct().ToArray();
}

public class ModelEntry
{
    public string Id { get; init; } = "";
    public string Name { get; init; } = "";
    public string Task { get; init; } = "";

    /// <summary>HuggingFace repository ID (e.g., "onnx-community/squeezenet1.1-7")</summary>
    public string? HuggingFaceRepo { get; init; }

    /// <summary>Filename within the HuggingFace repo (e.g., "model.onnx" or "onnx/model.onnx")</summary>
    public string? HuggingFaceFile { get; init; }

    public double SizeMB { get; init; }
    public string Description { get; init; } = "";
    public int InputWidth { get; init; }
    public int InputHeight { get; init; }

    /// <summary>Whether this model is hosted on HuggingFace.</summary>
    public bool IsHuggingFace => HuggingFaceRepo != null;

    /// <summary>
    /// Get the CDN download URL for this model.
    /// </summary>
    public string GetDownloadUrl()
    {
        if (HuggingFaceRepo != null && HuggingFaceFile != null)
            return HuggingFaceClient.GetDownloadUrl(HuggingFaceRepo, HuggingFaceFile);
        throw new InvalidOperationException($"Model '{Id}' has no download source configured.");
    }

    /// <summary>
    /// Load this model's bytes using a ModelHub (with OPFS caching).
    /// </summary>
    public Task<byte[]> LoadAsync(ModelHub hub)
    {
        if (HuggingFaceRepo != null && HuggingFaceFile != null)
            return hub.LoadAsync(HuggingFaceRepo, HuggingFaceFile);
        throw new InvalidOperationException($"Model '{Id}' has no download source configured.");
    }
}
