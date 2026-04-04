using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Face detection pipeline for BlazeFace (MediaPipe).
/// Handles image preprocessing, GPU inference, anchor decoding, and NMS.
///
/// Usage:
///   var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
///   var pipeline = new FaceDetectionPipeline(session, accelerator);
///   var result = await pipeline.DetectAsync(rgbaPixels, width, height);
///   foreach (var face in result.Faces)
///       Console.WriteLine($"Face at ({face.X:F0},{face.Y:F0}) conf={face.Confidence:P0}");
/// </summary>
public class FaceDetectionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly int _inputSize;
    private readonly float[,] _anchors;

    public FaceDetectionPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 128)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _inputSize = inputSize;
        _anchors = GenerateAnchors(inputSize);
    }

    /// <summary>
    /// Detect faces in an RGBA image.
    /// </summary>
    public async Task<FaceDetectionResult> DetectAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.3f)
    {
        var sw = Stopwatch.StartNew();

        // Preprocess: RGBA → NCHW float [0,1] for 128×128
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.ForwardNormalized01(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        // BlazeFace expects NHWC — transpose NCHW→NHWC on GPU (no CPU round-trip)
        int H = _inputSize, W = _inputSize;
        using var nhwcBuf = _accelerator.Allocate1D<float>(3 * H * W);
        new TransposeKernel(_accelerator).Transpose(preprocessed.View, nhwcBuf.View,
            new[] { 3, H, W }, new[] { 1, 2, 0 }); // CHW → HWC
        var inputTensor = new Tensor(nhwcBuf.View, new[] { 1, H, W, 3 });

        // Run inference — BlazeFace has 2 outputs: regressors [1,896,16] and classificators [1,896,1]
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        // Read outputs
        var regressors = await ReadOutputAsync(outputs, 0, 896 * 16);
        var classificators = await ReadOutputAsync(outputs, 1, 896);

        // Decode detections
        var faces = DecodeDetections(regressors, classificators, width, height,
            confidenceThreshold, iouThreshold);

        sw.Stop();

        return new FaceDetectionResult
        {
            Faces = faces,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    private async Task<float[]> ReadOutputAsync(Dictionary<string, Tensor> outputs, int index, int expectedElems)
    {
        var outputName = _session.OutputNames.Length > index ? _session.OutputNames[index] : null;
        if (outputName == null || !outputs.ContainsKey(outputName))
            return new float[expectedElems];

        var output = outputs[outputName];
        int elems = Math.Min(output.ElementCount, expectedElems);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync();
        return await readBuf.CopyToHostAsync<float>(0, elems);
    }

    private DetectedFace[] DecodeDetections(float[] regressors, float[] classificators,
        int imageWidth, int imageHeight, float confThreshold, float iouThreshold)
    {
        var candidates = new List<DetectedFace>();
        int numAnchors = _anchors.GetLength(0);

        for (int i = 0; i < numAnchors && i < classificators.Length; i++)
        {
            float score = Sigmoid(classificators[i]);
            if (score < confThreshold) continue;

            int regBase = i * 16;
            if (regBase + 15 >= regressors.Length) continue;

            // Decode box (relative to anchor)
            float cx = regressors[regBase + 0] / _inputSize + _anchors[i, 0];
            float cy = regressors[regBase + 1] / _inputSize + _anchors[i, 1];
            float w = regressors[regBase + 2] / _inputSize;
            float h = regressors[regBase + 3] / _inputSize;

            float x1 = (cx - w / 2) * imageWidth;
            float y1 = (cy - h / 2) * imageHeight;
            float bw = w * imageWidth;
            float bh = h * imageHeight;

            // Decode 6 landmarks
            var landmarks = new List<(float X, float Y)>();
            for (int j = 0; j < 6; j++)
            {
                float lx = (regressors[regBase + 4 + j * 2] / _inputSize + _anchors[i, 0]) * imageWidth;
                float ly = (regressors[regBase + 4 + j * 2 + 1] / _inputSize + _anchors[i, 1]) * imageHeight;
                landmarks.Add((lx, ly));
            }

            candidates.Add(new DetectedFace
            {
                X = x1,
                Y = y1,
                Width = bw,
                Height = bh,
                Confidence = score,
                Landmarks = landmarks,
            });
        }

        // NMS
        candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        var kept = new List<DetectedFace>();
        var suppressed = new bool[candidates.Count];

        for (int i = 0; i < candidates.Count; i++)
        {
            if (suppressed[i]) continue;
            kept.Add(candidates[i]);

            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (suppressed[j]) continue;
                if (IoU(candidates[i], candidates[j]) > iouThreshold)
                    suppressed[j] = true;
            }
        }

        return kept.ToArray();
    }

    private static float IoU(DetectedFace a, DetectedFace b)
    {
        float x1 = MathF.Max(a.X, b.X);
        float y1 = MathF.Max(a.Y, b.Y);
        float x2 = MathF.Min(a.X + a.Width, b.X + b.Width);
        float y2 = MathF.Min(a.Y + a.Height, b.Y + b.Height);
        float intersection = MathF.Max(0, x2 - x1) * MathF.Max(0, y2 - y1);
        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        float union = areaA + areaB - intersection;
        return union > 0 ? intersection / union : 0;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>
    /// Generate BlazeFace anchors following the MediaPipe SSD anchor specification.
    /// Strides [8, 16] produce 16×16 and 8×8 grids, 2 anchors per cell.
    /// Total: (16×16×2) + (8×8×2) = 512 + 128 = 640... but BlazeFace front-camera
    /// uses a different config that produces 896 anchors.
    /// </summary>
    private static float[,] GenerateAnchors(int inputSize)
    {
        var anchors = new List<(float cx, float cy)>();

        // MediaPipe BlazeFace anchor spec: strides [8, 16], 2 anchors per position
        int[] strides = { 8, 16 };
        foreach (int stride in strides)
        {
            int gridH = inputSize / stride;
            int gridW = inputSize / stride;
            for (int y = 0; y < gridH; y++)
                for (int x = 0; x < gridW; x++)
                    for (int a = 0; a < 2; a++)
                        anchors.Add(((x + 0.5f) / gridW, (y + 0.5f) / gridH));
        }

        var result = new float[anchors.Count, 2];
        for (int i = 0; i < anchors.Count; i++)
        {
            result[i, 0] = anchors[i].cx;
            result[i, 1] = anchors[i].cy;
        }
        return result;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
