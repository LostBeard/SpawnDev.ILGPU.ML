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

    /// <summary>WHY the encoder's dispatch-plan capture is or is not live.</summary>
    /// <remarks>
    /// ⚠️ Reported for the same reason the ZipVoice decoder reports it: "capture enabled" is a request, not
    /// an outcome. It falls through silently on an ineligible backend, on a control-flow refusal, and on a
    /// TryCapture that returns null - and two of those print nothing at all. Without this, a transcription
    /// that is still slow cannot be told apart from one where capture never engaged, and those call for
    /// opposite work.
    /// </remarks>
    public string EncoderCaptureStatus { get; init; } = "";

    /// <summary>Wall time in the encoder, in ms - ONE run at a fixed shape, and capturable.</summary>
    public double EncoderMs { get; init; }

    /// <summary>Wall time in the decoder prefill (the whole prompt, once), in ms.</summary>
    public double PrefillMs { get; init; }

    /// <summary>Wall time in the autoregressive decode steps together, in ms.</summary>
    /// <remarks>
    /// ⚠️ Split from <see cref="EncoderMs"/> because the two are different KINDS of cost and only one is
    /// addressable the same way. The encoder is one fixed-shape run, so a recorded dispatch plan serves
    /// every transcription for the life of the pipeline. Each decode step's past-K/V is one position longer
    /// than the last, so its shapes change on every call and no <c>SessionGraphCapture</c> recording is
    /// valid twice - that case needs a cursor-patching plan of the kind <c>WebGPUDecodeCapture</c> records
    /// for GGUF decode. Which of the two dominates therefore decides the next piece of work, and an
    /// executor total across all runs cannot say.
    /// </remarks>
    public double DecodeStepsMs { get; init; }

    /// <summary>How many autoregressive decode steps ran.</summary>
    public int DecodeSteps { get; init; }

    /// <summary>Per-token HOST setup inside the decode loop: the ids buffer allocation and input dictionary.</summary>
    public double DecodeSetupMs { get; init; }

    /// <summary>Time inside the decoder graph itself, summed over the decode steps.</summary>
    public double DecodeGraphMs { get; init; }

    /// <summary>Time in the per-token GPU argmax, which is one GPU-to-host round trip per token.</summary>
    public double DecodeArgmaxMs { get; init; }

    /// <summary>
    /// Wall time computing the log-mel spectrogram, in milliseconds.
    /// </summary>
    /// <remarks>
    /// ⚠️ Broken out because it is a CPU cost inside an otherwise GPU pipeline, and because it is FIXED
    /// rather than proportional to the utterance: the audio is padded to a flat 30 s before the STFT runs,
    /// so a four-word turn pays exactly what a full half-minute does. That is why endpointing shortened the
    /// RECORDING without shortening the transcription, and attributing it needs its own number - the
    /// executor counters cannot see work that never reaches the executor.
    /// </remarks>
    public double MelTimeMs { get; init; }

    /// <summary>Wall time in the encoder + decoder, in milliseconds.</summary>
    public double ModelTimeMs { get; init; }
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
