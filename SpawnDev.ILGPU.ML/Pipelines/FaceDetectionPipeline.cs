using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Face detection pipeline for BlazeFace (MediaPipe short-range).
/// Handles image preprocessing, GPU inference, anchor decoding, and NMS.
/// </summary>
/// <remarks>
/// MediaPipe short-range contract: input [-1,1] letterboxed 128×128, strides [8,16,16,16] → 896
/// anchors, <c>reverse_output_order</c> yxhw boxes. Decode matches that.
/// <para>
/// Forward parity (5.2.29): TFLite NHWC MaxPool + correct DepthwiseConv fused-activation field.
/// Gate: <c>Pipeline_BlazeFace_Reference_MatchesOnnxRuntime</c> (classificator relRMS ≤ 0.05).
/// </para>
/// </remarks>
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
        if (_anchors.GetLength(0) != 896)
            throw new InvalidOperationException(
                $"BlazeFace short-range expects 896 anchors, GenerateAnchors produced {_anchors.GetLength(0)}");
    }

    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="DetectAsync(TypedArray, int, int, float, float)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<FaceDetectionResult> DetectAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.3f,
        int maxFaces = 1)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await DetectAsync(rgbaBuf.View, width, height, confidenceThreshold, iouThreshold, maxFaces)
            .ConfigureAwait(false);
    }

    public async Task<FaceDetectionResult> DetectAsync(
        TypedArray rgbaPixels, int width, int height,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.3f,
        int maxFaces = 1)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await DetectAsync(rgbaBuf.View, width, height, confidenceThreshold, iouThreshold, maxFaces)
            .ConfigureAwait(false);
    }

    public Task<FaceDetectionResult> DetectAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.3f,
        int maxFaces = 1)
        => DetectCoreAsync(rgbaPixels, width, height, confidenceThreshold, iouThreshold, maxFaces);

    public Task<FaceDetectionResult> DetectAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.3f,
        int maxFaces = 1)
        => DetectCoreAsync(rgbaPixels.View, width, height, confidenceThreshold, iouThreshold, maxFaces);

    private async Task<FaceDetectionResult> DetectCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold, float iouThreshold, int maxFaces)
    {
        var sw = Stopwatch.StartNew();

        // MediaPipe ImageToTensorCalculator: keep_aspect_ratio + float range [-1, 1]
        var (contentW, contentH, padX, padY) = ImagePreprocessKernel.Letterbox(
            width, height, _inputSize, _inputSize);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize,
            mean: new[] { 0.5f, 0.5f, 0.5f },
            std: new[] { 0.5f, 0.5f, 0.5f },
            preserveAspect: true);

        int H = _inputSize, W = _inputSize;
        using var nhwcBuf = _accelerator.Allocate1D<float>(3 * H * W);
        new TransposeKernel(_accelerator).Transpose(preprocessed.View, nhwcBuf.View,
            new[] { 3, H, W }, new[] { 1, 2, 0 }); // CHW → HWC
        var inputTensor = new Tensor(nhwcBuf.View, new[] { 1, H, W, 3 });

        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        var (regName, clsName) = ResolveOutputNames();
        var regressors = await ReadOutputAsync(outputs, regName, 896 * 16).ConfigureAwait(false);
        var classificators = await ReadOutputAsync(outputs, clsName, 896).ConfigureAwait(false);

        var faces = DecodeDetections(regressors, classificators, width, height,
            contentW, contentH, padX, padY, confidenceThreshold, iouThreshold, maxFaces);

        sw.Stop();

        return new FaceDetectionResult
        {
            Faces = faces,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    private (string Regressors, string Classificators) ResolveOutputNames()
    {
        string? reg = null, cls = null;
        foreach (var n in _session.OutputNames)
        {
            var lower = n.ToLowerInvariant();
            if (lower.Contains("regress") || lower.Contains("box"))
                reg ??= n;
            else if (lower.Contains("classif") || lower.Contains("score") || lower.Contains("conf"))
                cls ??= n;
        }
        reg ??= _session.OutputNames.Length > 0 ? _session.OutputNames[0] : "";
        cls ??= _session.OutputNames.Length > 1 ? _session.OutputNames[1] : reg;
        return (reg, cls);
    }

    private async Task<float[]> ReadOutputAsync(Dictionary<string, Tensor> outputs, string outputName, int expectedElems)
    {
        if (string.IsNullOrEmpty(outputName) || !outputs.ContainsKey(outputName))
            return new float[expectedElems];

        var output = outputs[outputName];
        int elems = Math.Min(output.ElementCount, expectedElems);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return await readBuf.CopyToHostAsync<float>(0, elems).ConfigureAwait(false);
    }

    private DetectedFace[] DecodeDetections(float[] regressors, float[] classificators,
        int imageWidth, int imageHeight,
        int contentW, int contentH, int padX, int padY,
        float confThreshold, float iouThreshold, int maxFaces)
    {
        var candidates = new List<DetectedFace>();
        int numAnchors = _anchors.GetLength(0);
        float scale = _inputSize;

        for (int i = 0; i < numAnchors && i < classificators.Length; i++)
        {
            float score = Sigmoid(classificators[i]);
            if (score < confThreshold) continue;

            int regBase = i * 16;
            if (regBase + 15 >= regressors.Length) continue;

            // MediaPipe reverse_output_order: [y_center, x_center, h, w], keypoints (y,x)*6
            float cy = regressors[regBase + 0] / scale + _anchors[i, 1];
            float cx = regressors[regBase + 1] / scale + _anchors[i, 0];
            float h = regressors[regBase + 2] / scale;
            float w = regressors[regBase + 3] / scale;

            float x1 = MapX(cx - w / 2, contentW, padX, imageWidth);
            float y1 = MapY(cy - h / 2, contentH, padY, imageHeight);
            float x2 = MapX(cx + w / 2, contentW, padX, imageWidth);
            float y2 = MapY(cy + h / 2, contentH, padY, imageHeight);

            var landmarks = new List<(float X, float Y)>(6);
            for (int j = 0; j < 6; j++)
            {
                float ly = regressors[regBase + 4 + j * 2] / scale + _anchors[i, 1];
                float lx = regressors[regBase + 4 + j * 2 + 1] / scale + _anchors[i, 0];
                landmarks.Add((
                    Clamp(MapX(lx, contentW, padX, imageWidth), 0, imageWidth),
                    Clamp(MapY(ly, contentH, padY, imageHeight), 0, imageHeight)));
            }

            // Clamp to image — BlazeFace often predicts slightly outside the frame.
            x1 = Clamp(x1, 0, imageWidth);
            y1 = Clamp(y1, 0, imageHeight);
            x2 = Clamp(x2, 0, imageWidth);
            y2 = Clamp(y2, 0, imageHeight);

            candidates.Add(new DetectedFace
            {
                X = x1,
                Y = y1,
                Width = MathF.Max(0, x2 - x1),
                Height = MathF.Max(0, y2 - y1),
                Confidence = score,
                Landmarks = landmarks,
            });
        }

        // MediaPipe Face Detector uses WEIGHTED NMS (merge overlapping boxes by score), not hard
        // suppress. Hard NMS at IoU 0.3 left near-duplicate shifted boxes on portrait (IoU~0.19)
        // plus a hair false-positive → live Faces:2 with wrong overlays.
        var merged = WeightedNms(candidates, iouThreshold);
        if (maxFaces > 0 && merged.Count > maxFaces)
            merged = merged.Take(maxFaces).ToList();
        return merged.ToArray();
    }

    /// <summary>
    /// MediaPipe-style weighted NMS: overlapping detections are blended by confidence instead of
    /// discarding the lower-scoring one. Returns highest-confidence-first.
    /// </summary>
    private static List<DetectedFace> WeightedNms(List<DetectedFace> candidates, float iouThreshold)
    {
        candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        var remaining = new List<DetectedFace>(candidates);
        var kept = new List<DetectedFace>();

        while (remaining.Count > 0)
        {
            var seed = remaining[0];
            remaining.RemoveAt(0);

            float sumW = seed.Confidence;
            float x = seed.X * seed.Confidence;
            float y = seed.Y * seed.Confidence;
            float w = seed.Width * seed.Confidence;
            float h = seed.Height * seed.Confidence;
            float conf = seed.Confidence;
            var landmarks = seed.Landmarks;
            var landmarkAcc = landmarks.Select(p => (p.X * seed.Confidence, p.Y * seed.Confidence)).ToList();

            for (int i = remaining.Count - 1; i >= 0; i--)
            {
                if (IoU(seed, remaining[i]) <= iouThreshold) continue;
                var o = remaining[i];
                float sw = o.Confidence;
                sumW += sw;
                x += o.X * sw;
                y += o.Y * sw;
                w += o.Width * sw;
                h += o.Height * sw;
                if (o.Confidence > conf) conf = o.Confidence;
                if (o.Landmarks.Count == landmarkAcc.Count)
                {
                    for (int k = 0; k < landmarkAcc.Count; k++)
                        landmarkAcc[k] = (landmarkAcc[k].Item1 + o.Landmarks[k].X * sw,
                            landmarkAcc[k].Item2 + o.Landmarks[k].Y * sw);
                }
                remaining.RemoveAt(i);
            }

            kept.Add(new DetectedFace
            {
                X = x / sumW,
                Y = y / sumW,
                Width = w / sumW,
                Height = h / sumW,
                Confidence = conf,
                Landmarks = landmarkAcc.Select(p => (p.Item1 / sumW, p.Item2 / sumW)).ToList(),
            });
        }

        return kept;
    }

    private static float Clamp(float v, float lo, float hi) => v < lo ? lo : (v > hi ? hi : v);

    private float MapX(float nx, int contentW, int padX, int imageWidth)
    {
        float lx = nx * _inputSize - padX;
        if (contentW <= 0) return nx * imageWidth;
        return lx / contentW * imageWidth;
    }

    private float MapY(float ny, int contentH, int padY, int imageHeight)
    {
        float ly = ny * _inputSize - padY;
        if (contentH <= 0) return ny * imageHeight;
        return ly / contentH * imageHeight;
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

    private static float Sigmoid(float x)
    {
        if (x > 100f) x = 100f;
        if (x < -100f) x = -100f;
        return 1f / (1f + MathF.Exp(-x));
    }

    /// <summary>MediaPipe short-range: strides [8,16,16,16], 2 anchors/cell → 896.</summary>
    private static float[,] GenerateAnchors(int inputSize)
    {
        var anchors = new List<(float cx, float cy)>();
        int[] strides = { 8, 16, 16, 16 };
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
