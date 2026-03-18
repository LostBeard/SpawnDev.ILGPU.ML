namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Video preprocessing utilities for ML inference.
/// Handles frame batching, temporal windowing, and frame-rate management
/// for video classification, action recognition, and real-time processing.
/// </summary>
public static class VideoPreprocessor
{
    /// <summary>
    /// Sample N frames uniformly from a list of frames.
    /// Used by video classification models that need a fixed number of frames.
    /// </summary>
    public static T[] UniformSample<T>(T[] frames, int numSamples)
    {
        if (frames.Length <= numSamples)
        {
            // Repeat last frame to fill
            var result = new T[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                result[i] = frames[Math.Min(i, frames.Length - 1)];
            }
            return result;
        }

        var sampled = new T[numSamples];
        float step = (float)(frames.Length - 1) / (numSamples - 1);
        for (int i = 0; i < numSamples; i++)
        {
            sampled[i] = frames[(int)(i * step + 0.5f)];
        }
        return sampled;
    }

    /// <summary>
    /// Extract a temporal window of frames centered on the given index.
    /// Clamps to boundaries (repeats first/last frame if needed).
    /// </summary>
    public static T[] TemporalWindow<T>(T[] frames, int centerIndex, int windowSize)
    {
        var window = new T[windowSize];
        int half = windowSize / 2;
        for (int i = 0; i < windowSize; i++)
        {
            int idx = Math.Clamp(centerIndex - half + i, 0, frames.Length - 1);
            window[i] = frames[idx];
        }
        return window;
    }

    /// <summary>
    /// Compute frame differences (motion detection).
    /// Returns absolute difference between consecutive RGBA frames.
    /// Useful for motion-based preprocessing or skipping static frames.
    /// </summary>
    public static float[] ComputeFrameDifference(byte[] frameA, byte[] frameB)
    {
        int pixelCount = Math.Min(frameA.Length, frameB.Length) / 4;
        var diff = new float[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            float dr = Math.Abs(frameA[i * 4] - frameB[i * 4]);
            float dg = Math.Abs(frameA[i * 4 + 1] - frameB[i * 4 + 1]);
            float db = Math.Abs(frameA[i * 4 + 2] - frameB[i * 4 + 2]);
            diff[i] = (dr + dg + db) / (3f * 255f); // Normalized [0, 1]
        }
        return diff;
    }

    /// <summary>
    /// Compute the mean motion score between two frames (0 = identical, 1 = completely different).
    /// Useful for deciding whether to run inference on a new frame.
    /// </summary>
    public static float ComputeMotionScore(byte[] frameA, byte[] frameB)
    {
        int pixelCount = Math.Min(frameA.Length, frameB.Length) / 4;
        if (pixelCount == 0) return 0;

        float totalDiff = 0;
        for (int i = 0; i < pixelCount; i++)
        {
            totalDiff += Math.Abs(frameA[i * 4] - frameB[i * 4]);
            totalDiff += Math.Abs(frameA[i * 4 + 1] - frameB[i * 4 + 1]);
            totalDiff += Math.Abs(frameA[i * 4 + 2] - frameB[i * 4 + 2]);
        }
        return totalDiff / (pixelCount * 3f * 255f);
    }

    /// <summary>
    /// Stack multiple preprocessed frame tensors into a batch tensor.
    /// Input: array of NCHW tensors [C, H, W]. Output: [N, C, H, W] flattened.
    /// </summary>
    public static float[] StackFrames(float[][] frameTensors)
    {
        if (frameTensors.Length == 0) return Array.Empty<float>();

        int frameSize = frameTensors[0].Length;
        var batch = new float[frameTensors.Length * frameSize];
        for (int i = 0; i < frameTensors.Length; i++)
        {
            Array.Copy(frameTensors[i], 0, batch, i * frameSize, frameSize);
        }
        return batch;
    }
}

/// <summary>
/// Frame rate controller for real-time video inference.
/// Tracks timing and decides whether to run inference on each frame
/// to maintain a target inference rate without overloading the GPU.
/// </summary>
public class InferenceRateController
{
    private readonly double _targetIntervalMs;
    private DateTime _lastInferenceTime = DateTime.MinValue;
    private byte[]? _lastFrame;
    private readonly float _motionThreshold;

    /// <summary>
    /// Create a rate controller.
    /// </summary>
    /// <param name="targetFps">Target inference FPS (e.g., 15 for detection, 30 for pose)</param>
    /// <param name="motionThreshold">Skip frames with motion below this threshold (0 = never skip, 0.01 = skip very static frames)</param>
    public InferenceRateController(float targetFps = 15, float motionThreshold = 0)
    {
        _targetIntervalMs = 1000.0 / targetFps;
        _motionThreshold = motionThreshold;
    }

    /// <summary>
    /// Check if inference should run on this frame.
    /// Returns true if enough time has passed AND the frame has sufficient motion.
    /// </summary>
    public bool ShouldRunInference(byte[]? currentFrame = null)
    {
        var now = DateTime.UtcNow;
        double elapsed = (now - _lastInferenceTime).TotalMilliseconds;

        if (elapsed < _targetIntervalMs)
            return false;

        // Motion check (optional)
        if (_motionThreshold > 0 && currentFrame != null && _lastFrame != null)
        {
            float motion = VideoPreprocessor.ComputeMotionScore(_lastFrame, currentFrame);
            if (motion < _motionThreshold)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Mark that inference was run on this frame. Call after inference completes.
    /// </summary>
    public void MarkInferenceRun(byte[]? frame = null)
    {
        _lastInferenceTime = DateTime.UtcNow;
        _lastFrame = frame;
    }
}
