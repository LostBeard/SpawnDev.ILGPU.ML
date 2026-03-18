namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Describes a model's preprocessing requirements.
/// Point at a model config and preprocessing is automatic.
/// </summary>
public class ModelConfig
{
    /// <summary>Model display name.</summary>
    public string Name { get; init; } = "";

    /// <summary>Input tensor name in the ONNX model.</summary>
    public string InputName { get; init; } = "input";

    /// <summary>Output tensor name(s) in the ONNX model.</summary>
    public string[] OutputNames { get; init; } = { "output" };

    /// <summary>Expected input width.</summary>
    public int InputWidth { get; init; } = 224;

    /// <summary>Expected input height.</summary>
    public int InputHeight { get; init; } = 224;

    /// <summary>Number of input channels (3 for RGB).</summary>
    public int InputChannels { get; init; } = 3;

    /// <summary>Tensor layout: NCHW or NHWC.</summary>
    public TensorLayoutType Layout { get; init; } = TensorLayoutType.NCHW;

    /// <summary>Per-channel normalization mean. Null = no normalization.</summary>
    public float[]? NormalizeMean { get; init; }

    /// <summary>Per-channel normalization std. Null = no normalization.</summary>
    public float[]? NormalizeStd { get; init; }

    /// <summary>Whether to scale pixel values to [0,1] (divide by 255).</summary>
    public bool ScaleTo01 { get; init; } = true;

    /// <summary>Whether to use letterbox resize (pad to maintain aspect ratio).</summary>
    public bool UseLetterbox { get; init; }

    /// <summary>Letterbox pad color (RGB bytes).</summary>
    public byte[] LetterboxPadColor { get; init; } = { 114, 114, 114 };

    /// <summary>Whether model expects int32 input instead of float32.</summary>
    public bool InputIsInt32 { get; init; }

    /// <summary>
    /// Preprocess RGBA image bytes for this model, returning a float tensor ready for inference.
    /// </summary>
    public float[] Preprocess(byte[] rgbaPixels, int srcWidth, int srcHeight)
    {
        if (UseLetterbox)
        {
            var (tensor, _) = ImagePreprocessor.PreprocessLetterbox(
                rgbaPixels, srcWidth, srcHeight, InputWidth, InputHeight);
            return tensor;
        }

        if (!ScaleTo01 && NormalizeMean == null)
        {
            return ImagePreprocessor.PreprocessToNCHW255(
                rgbaPixels, srcWidth, srcHeight, InputWidth, InputHeight);
        }

        return ImagePreprocessor.PreprocessToNCHW(
            rgbaPixels, srcWidth, srcHeight, InputWidth, InputHeight,
            NormalizeMean, NormalizeStd, ScaleTo01);
    }

    /// <summary>
    /// Preprocess RGBA image bytes for this model, returning int32 NHWC tensor (for MoveNet etc.).
    /// </summary>
    public int[] PreprocessInt32(byte[] rgbaPixels, int srcWidth, int srcHeight)
    {
        return ImagePreprocessor.PreprocessToNHWCInt32(
            rgbaPixels, srcWidth, srcHeight, InputWidth, InputHeight);
    }
}

/// <summary>
/// Tensor layout format.
/// </summary>
public enum TensorLayoutType
{
    /// <summary>Channel-first: [N, C, H, W] — used by most ONNX models.</summary>
    NCHW,
    /// <summary>Channel-last: [N, H, W, C] — used by TensorFlow-origin models.</summary>
    NHWC,
}

/// <summary>
/// Pre-built model configurations for commonly used models.
/// </summary>
public static class ModelConfigs
{
    /// <summary>MobileNetV2 — ImageNet 1000-class classification.</summary>
    public static readonly ModelConfig MobileNetV2 = new()
    {
        Name = "MobileNetV2",
        InputName = "input",
        OutputNames = new[] { "output" },
        InputWidth = 224,
        InputHeight = 224,
        Layout = TensorLayoutType.NCHW,
        NormalizeMean = new[] { 0.485f, 0.456f, 0.406f },
        NormalizeStd = new[] { 0.229f, 0.224f, 0.225f },
        ScaleTo01 = true,
    };

    /// <summary>Depth Anything V2 Small — monocular depth estimation.</summary>
    public static readonly ModelConfig DepthAnythingV2Small = new()
    {
        Name = "Depth Anything V2 Small",
        InputName = "pixel_values",
        OutputNames = new[] { "predicted_depth" },
        InputWidth = 518,
        InputHeight = 518,
        Layout = TensorLayoutType.NCHW,
        NormalizeMean = new[] { 0.485f, 0.456f, 0.406f },
        NormalizeStd = new[] { 0.229f, 0.224f, 0.225f },
        ScaleTo01 = true,
    };

    /// <summary>YOLOv8-Nano — 80-class COCO object detection.</summary>
    public static readonly ModelConfig YoloV8Nano = new()
    {
        Name = "YOLOv8-Nano",
        InputName = "images",
        OutputNames = new[] { "output0" },
        InputWidth = 640,
        InputHeight = 640,
        Layout = TensorLayoutType.NCHW,
        UseLetterbox = true,
        LetterboxPadColor = new byte[] { 114, 114, 114 },
        ScaleTo01 = true,
    };

    /// <summary>Fast Neural Style Transfer — artistic style transfer.</summary>
    public static readonly ModelConfig FastNeuralStyle = new()
    {
        Name = "Fast Neural Style",
        InputName = "input1",
        OutputNames = new[] { "output1" },
        InputWidth = 224,
        InputHeight = 224,
        Layout = TensorLayoutType.NCHW,
        ScaleTo01 = false, // Expects [0, 255] float range
    };

    /// <summary>RMBG-1.4 — background removal / foreground segmentation.</summary>
    public static readonly ModelConfig RMBG14 = new()
    {
        Name = "RMBG-1.4",
        InputName = "input",
        OutputNames = new[] { "output" },
        InputWidth = 1024,
        InputHeight = 1024,
        Layout = TensorLayoutType.NCHW,
        NormalizeMean = new[] { 0.5f, 0.5f, 0.5f },
        NormalizeStd = new[] { 1.0f, 1.0f, 1.0f },
        ScaleTo01 = true,
    };

    /// <summary>MoveNet Lightning — single-person pose estimation (17 keypoints).</summary>
    public static readonly ModelConfig MoveNetLightning = new()
    {
        Name = "MoveNet Lightning",
        InputName = "input",
        OutputNames = new[] { "output_0" },
        InputWidth = 192,
        InputHeight = 192,
        Layout = TensorLayoutType.NHWC,
        InputIsInt32 = true, // int32 [0,255], not float
    };

    /// <summary>ESPCN — super resolution (3x upscale, Y channel only).</summary>
    public static readonly ModelConfig ESPCN = new()
    {
        Name = "ESPCN (3x)",
        InputName = "input",
        OutputNames = new[] { "output" },
        InputWidth = 224,
        InputHeight = 224,
        InputChannels = 1, // Y luminance only
        Layout = TensorLayoutType.NCHW,
        ScaleTo01 = true,
    };
}
