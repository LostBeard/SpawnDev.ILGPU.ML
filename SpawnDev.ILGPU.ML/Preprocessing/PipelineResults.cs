namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Structured result types for ML inference pipelines.
/// These are the consumer-facing outputs that pipeline abstractions return.
/// </summary>

/// <summary>Result from image classification.</summary>
public class ClassificationResult
{
    /// <summary>Ranked predictions, highest confidence first.</summary>
    public ClassPrediction[] Predictions { get; init; } = Array.Empty<ClassPrediction>();

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Top prediction label.</summary>
    public string TopLabel => Predictions.Length > 0 ? Predictions[0].Label : "";

    /// <summary>Top prediction confidence.</summary>
    public float TopConfidence => Predictions.Length > 0 ? Predictions[0].Confidence : 0;
}

/// <summary>A single class prediction.</summary>
public class ClassPrediction
{
    public string Label { get; init; } = "";
    public int ClassId { get; init; }
    public float Confidence { get; init; }
}

/// <summary>Result from object detection.</summary>
public class DetectionResult
{
    /// <summary>Detected objects.</summary>
    public DetectedObject[] Objects { get; init; } = Array.Empty<DetectedObject>();

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Original image dimensions used for box coordinates.</summary>
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
}

/// <summary>A single detected object.</summary>
public class DetectedObject
{
    public string Label { get; init; } = "";
    public int ClassId { get; init; }
    public float Confidence { get; init; }

    /// <summary>Bounding box in pixel coordinates (top-left origin).</summary>
    public float X { get; init; }
    public float Y { get; init; }
    public float Width { get; init; }
    public float Height { get; init; }

    /// <summary>Center X in pixel coordinates.</summary>
    public float CenterX => X + Width / 2;
    /// <summary>Center Y in pixel coordinates.</summary>
    public float CenterY => Y + Height / 2;
    /// <summary>Box area in pixels.</summary>
    public float Area => Width * Height;
}

/// <summary>Result from depth estimation.</summary>
public class DepthResult
{
    /// <summary>Raw depth values [height, width]. Higher = closer (relative depth).</summary>
    public float[] DepthMap { get; init; } = Array.Empty<float>();

    /// <summary>Depth map dimensions.</summary>
    public int Width { get; init; }
    public int Height { get; init; }

    /// <summary>Minimum depth value in the map.</summary>
    public float MinDepth { get; init; }
    /// <summary>Maximum depth value in the map.</summary>
    public float MaxDepth { get; init; }

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Get normalized depth at a pixel coordinate [0,1].</summary>
    public float GetNormalizedDepth(int x, int y)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height) return 0;
        float raw = DepthMap[y * Width + x];
        float range = MaxDepth - MinDepth;
        return range > 1e-6f ? (raw - MinDepth) / range : 0;
    }

    /// <summary>Generate RGBA colormap image from depth data.</summary>
    public byte[] ToColorMap(string palette = "plasma")
    {
        return DepthColorMaps.ApplyColorMap(DepthMap, Width, Height, palette);
    }
}

/// <summary>Result from pose estimation.</summary>
public class PoseResult
{
    /// <summary>Detected keypoints.</summary>
    public PoseSkeleton.Keypoint[] Keypoints { get; init; } = Array.Empty<PoseSkeleton.Keypoint>();

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Number of keypoints above the confidence threshold.</summary>
    public int DetectedCount(float threshold = 0.3f) =>
        Keypoints.Count(k => k.Confidence >= threshold);

    /// <summary>Get a specific keypoint by name.</summary>
    public PoseSkeleton.Keypoint? GetKeypoint(string name) =>
        Keypoints.FirstOrDefault(k => k.Name == name);
}

/// <summary>Result from face detection.</summary>
public class FaceDetectionResult
{
    /// <summary>Detected faces.</summary>
    public DetectedFace[] Faces { get; init; } = Array.Empty<DetectedFace>();

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Number of faces detected.</summary>
    public int FaceCount => Faces.Length;
}

/// <summary>A single detected face with optional landmarks.</summary>
public class DetectedFace
{
    /// <summary>Bounding box in pixel coordinates.</summary>
    public float X { get; init; }
    public float Y { get; init; }
    public float Width { get; init; }
    public float Height { get; init; }
    public float Confidence { get; init; }

    /// <summary>Facial landmark points (e.g., eyes, nose, mouth).</summary>
    public List<(float X, float Y)> Landmarks { get; init; } = new();
}

/// <summary>Result from segmentation / background removal.</summary>
public class SegmentationResult
{
    /// <summary>Binary mask [height, width] with values in [0,1]. 1 = foreground.</summary>
    public float[] Mask { get; init; } = Array.Empty<float>();

    /// <summary>Mask dimensions.</summary>
    public int Width { get; init; }
    public int Height { get; init; }

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Convert mask to RGBA with transparency (foreground = opaque, background = transparent).</summary>
    public byte[] ToAlphaMask() => ImagePreprocessor.MaskToRGBA(Mask, Width, Height);

    /// <summary>Apply mask to an image, removing the background.</summary>
    public byte[] ApplyToImage(byte[] rgbaImage) =>
        ImagePreprocessor.CompositeWithMask(rgbaImage, ToAlphaMask(), Width, Height);
}

/// <summary>Result from style transfer.</summary>
public class StyleTransferResult
{
    /// <summary>Stylized image as RGBA bytes.</summary>
    public byte[] ImageRGBA { get; init; } = Array.Empty<byte>();

    /// <summary>Image dimensions.</summary>
    public int Width { get; init; }
    public int Height { get; init; }

    /// <summary>Style name applied.</summary>
    public string StyleName { get; init; } = "";

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }
}

/// <summary>Result from super resolution.</summary>
public class SuperResolutionResult
{
    /// <summary>Upscaled image as RGBA bytes.</summary>
    public byte[] ImageRGBA { get; init; } = Array.Empty<byte>();

    /// <summary>Output dimensions.</summary>
    public int Width { get; init; }
    public int Height { get; init; }

    /// <summary>Upscale factor used.</summary>
    public int ScaleFactor { get; init; }

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }
}

/// <summary>Result from speech-to-text.</summary>
public class TranscriptionResult
{
    /// <summary>Transcribed text.</summary>
    public string Text { get; init; } = "";

    /// <summary>Detected language (if available).</summary>
    public string? Language { get; init; }

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Per-segment timestamps (if available).</summary>
    public TranscriptionSegment[] Segments { get; init; } = Array.Empty<TranscriptionSegment>();
}

/// <summary>A segment of transcribed text with timing.</summary>
public class TranscriptionSegment
{
    public string Text { get; init; } = "";
    public double StartTimeSeconds { get; init; }
    public double EndTimeSeconds { get; init; }
}

/// <summary>Result from embedding/feature extraction.</summary>
public class EmbeddingResult
{
    /// <summary>Embedding vector.</summary>
    public float[] Embedding { get; init; } = Array.Empty<float>();

    /// <summary>Embedding dimensionality.</summary>
    public int Dimensions => Embedding.Length;

    /// <summary>Inference time in milliseconds.</summary>
    public double InferenceTimeMs { get; init; }

    /// <summary>Compute cosine similarity with another embedding.</summary>
    public float SimilarityTo(EmbeddingResult other) =>
        TextPreprocessor.CosineSimilarity(Embedding, other.Embedding);
}
