namespace SpawnDev.ILGPU.ML.Demo.UI.Services;

/// <summary>
/// Registry of available demo models with their configurations.
/// Used by demo pages to discover which models are available and how to configure them.
/// </summary>
public static class ModelRegistry
{
    public static readonly ModelEntry[] AllModels = new[]
    {
        // Classification
        new ModelEntry
        {
            Id = "squeezenet",
            Name = "SqueezeNet 1.1",
            Task = "image-classification",
            Path = "models/squeezenet",
            OnnxPath = "models/squeezenet/model.onnx",
            SizeMB = 4.8,
            Description = "Fast 1000-class ImageNet classification",
            InputWidth = 224, InputHeight = 224,
        },
        new ModelEntry
        {
            Id = "mobilenetv2",
            Name = "MobileNetV2",
            Task = "image-classification",
            Path = "models/mobilenetv2",
            OnnxPath = "models/mobilenetv2/model.onnx",
            SizeMB = 14,
            Description = "Accurate 1000-class ImageNet classification",
            InputWidth = 224, InputHeight = 224,
        },

        // Depth Estimation
        new ModelEntry
        {
            Id = "depth-anything-v2-small",
            Name = "Depth Anything V2 Small",
            Task = "depth-estimation",
            OnnxPath = "models/depth-anything-v2-small/model.onnx",
            SizeMB = 95,
            Description = "Monocular depth estimation (26M params)",
            InputWidth = 518, InputHeight = 518,
        },

        // Style Transfer
        new ModelEntry
        {
            Id = "style-mosaic",
            Name = "Mosaic",
            Task = "style-transfer",
            Path = "models/style-mosaic",
            OnnxPath = "models/style-mosaic/model.onnx",
            SizeMB = 6.5,
            Description = "Mosaic tile artistic style",
        },
        new ModelEntry
        {
            Id = "style-candy",
            Name = "Candy",
            Task = "style-transfer",
            Path = "models/style-candy",
            OnnxPath = "models/style-candy/model.onnx",
            SizeMB = 6.5,
            Description = "Bright candy-colored artistic style",
        },
        new ModelEntry
        {
            Id = "style-rain-princess",
            Name = "Rain Princess",
            Task = "style-transfer",
            Path = "models/style-rain-princess",
            OnnxPath = "models/style-rain-princess/model.onnx",
            SizeMB = 6.5,
            Description = "Impressionist rain scene style",
        },
        new ModelEntry
        {
            Id = "style-udnie",
            Name = "Udnie",
            Task = "style-transfer",
            Path = "models/style-udnie",
            OnnxPath = "models/style-udnie/model.onnx",
            SizeMB = 6.5,
            Description = "Abstract cubist style",
        },
        new ModelEntry
        {
            Id = "style-pointilism",
            Name = "Pointilism",
            Task = "style-transfer",
            Path = "models/style-pointilism",
            OnnxPath = "models/style-pointilism/model.onnx",
            SizeMB = 6.5,
            Description = "Pointillist dot painting style",
        },

        // Super Resolution
        new ModelEntry
        {
            Id = "super-resolution",
            Name = "ESPCN (3x)",
            Task = "super-resolution",
            Path = "models/super-resolution",
            OnnxPath = "models/super-resolution/model.onnx",
            SizeMB = 0.2,
            Description = "3x upscaling (224x224 → 672x672)",
            InputWidth = 224, InputHeight = 224,
        },

        // Pose Estimation
        new ModelEntry
        {
            Id = "movenet-lightning",
            Name = "MoveNet Lightning",
            Task = "pose-estimation",
            OnnxPath = "models/movenet-lightning/model.onnx",
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
    public string? Path { get; init; }
    public string? OnnxPath { get; init; }
    public double SizeMB { get; init; }
    public string Description { get; init; } = "";
    public int InputWidth { get; init; }
    public int InputHeight { get; init; }

    /// <summary>Whether the model has a pre-extracted format (JSON+FP16).</summary>
    public bool HasExtractedFormat => Path != null;

    /// <summary>Whether the model has a direct .onnx file.</summary>
    public bool HasOnnxFormat => OnnxPath != null;
}
