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
/// Handles image preprocessing, GPU inference, anchor decoding, and NMS - all on the GPU: only the final faces
/// (17 floats each, plus a count) are read back, not the 896-anchor output tensors.
/// </summary>
/// <remarks>
/// MediaPipe short-range contract: input [-1,1] letterboxed 128×128, strides [8,16,16,16] → 896
/// anchors. Regressor layout for this TFLite is SSD <c>[x,y,w,h] + (x,y)*6</c> (not
/// calculator <c>reverse_output_order</c>).
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
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _anchorBuffer; // [896 * 2] cx, cy (uploaded once)
    private Kernels.ContentParamBuffers<float>? _params;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _decodeKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _nmsKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _confKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _gatherKernel;

    /// <summary>Floats per decoded anchor / per output face: confidence, x, y, width, height, 6 landmarks (x, y).</summary>
    internal const int FaceStride = 17;
    const int NumAnchors = 896;

    public FaceDetectionPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 128)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _inputSize = inputSize;
        _anchors = GenerateAnchors(inputSize);
        if (_anchors.GetLength(0) != NumAnchors)
            throw new InvalidOperationException(
                $"BlazeFace short-range expects 896 anchors, GenerateAnchors produced {_anchors.GetLength(0)}");
        var flat = new float[NumAnchors * 2];
        for (int i = 0; i < NumAnchors; i++) { flat[i * 2] = _anchors[i, 0]; flat[i * 2 + 1] = _anchors[i, 1]; }
        _anchorBuffer = accelerator.Allocate1D(flat); // 7 KB, once per pipeline
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
        var faces = await DecodeOnGpuAsync(outputs[regName].Data, outputs[clsName].Data, width, height,
            contentW, contentH, padX, padY, confidenceThreshold, iouThreshold, maxFaces).ConfigureAwait(false);

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

    /// <summary>
    /// Anchor decode + weighted NMS on the GPU (<see cref="DecodeKernel"/>, <see cref="WeightedNmsKernel"/>). Reads
    /// back 1 + maxFaces * 17 floats (18 for one face) instead of the 896 * 17 output floats the managed decode needed
    /// (61 KB per detection, ten times a second in a face-following app).
    /// </summary>
    private async Task<DetectedFace[]> DecodeOnGpuAsync(
        ArrayView1D<float, Stride1D.Dense> regressors, ArrayView1D<float, Stride1D.Dense> classificators,
        int imageWidth, int imageHeight, int contentW, int contentH, int padX, int padY,
        float confThreshold, float iouThreshold, int maxFaces)
    {
        if (regressors.Length < NumAnchors * 16 || classificators.Length < NumAnchors)
            throw new InvalidOperationException($"BlazeFace outputs too small: {regressors.Length} / {classificators.Length}");
        _decodeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(DecodeKernel);
        _nmsKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(WeightedNmsKernel);
        _params ??= new Kernels.ContentParamBuffers<float>(_accelerator);

        int maxOut = maxFaces > 0 ? Math.Min(maxFaces, NumAnchors) : NumAnchors;
        var p = _params.Get(new float[] {
            _inputSize, contentW, contentH, padX, padY, imageWidth, imageHeight,
            confThreshold, iouThreshold, maxOut, NumAnchors });

        using var decoded = _accelerator.Allocate1D<float>(NumAnchors * FaceStride);
        _decodeKernel(NumAnchors, regressors.SubView(0, NumAnchors * 16), classificators.SubView(0, NumAnchors),
            _anchorBuffer.View, p, decoded.View);
        if (_accelerator.AcceleratorType == AcceleratorType.WebGL)
            return await NmsGatherOnWebGLAsync(decoded, iouThreshold, maxFaces).ConfigureAwait(false);

        using var output = _accelerator.Allocate1D<float>(1 + maxOut * FaceStride);
        _nmsKernel(1, decoded.View, p, output.View);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);

        // A few faces: one read of the whole (small) output. Unlimited: the count first, then just those faces.
        float[] result;
        if (maxOut <= 16)
        {
            result = await output.CopyToHostAsync<float>(0, 1 + maxOut * FaceStride).ConfigureAwait(false);
        }
        else
        {
            var head = await output.CopyToHostAsync<float>(0, 1).ConfigureAwait(false);
            int n = Math.Clamp((int)head[0], 0, maxOut);
            result = new float[1 + n * FaceStride];
            result[0] = n;
            if (n > 0)
            {
                var body = await output.CopyToHostAsync<float>(1, n * FaceStride).ConfigureAwait(false);
                Array.Copy(body, 0, result, 1, body.Length);
            }
        }
        return ToFaces(result);
    }

    /// <summary>
    /// WebGL path. The NMS kernel needs in-kernel scatter (one thread writes many output slots and marks anchors as
    /// used), which WebGL's Transform Feedback cannot do: it captures one positional record per thread. So: read back
    /// the 896 confidences (3.5 KB), gather only the anchors above the threshold on the GPU (a gather, which WebGL
    /// can do), read those rows back and blend them with the same weighted NMS. Exact, and bounded by the data
    /// actually needed (still far below the 61 KB the managed decode read).
    /// </summary>
    private async Task<DetectedFace[]> NmsGatherOnWebGLAsync(MemoryBuffer1D<float, Stride1D.Dense> decoded,
        float iouThreshold, int maxFaces)
    {
        _confKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(ConfidenceColumnKernel);
        _gatherKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(GatherRowsKernel);

        using var conf = _accelerator.Allocate1D<float>(NumAnchors);
        _confKernel(NumAnchors, decoded.View, conf.View);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var confidences = await conf.CopyToHostAsync<float>(0, NumAnchors).ConfigureAwait(false);
        var passing = new List<int>();
        for (int i = 0; i < NumAnchors; i++) if (confidences[i] >= 0f) passing.Add(i);
        if (passing.Count == 0) return Array.Empty<DetectedFace>();

        using var indices = _accelerator.Allocate1D(passing.ToArray());
        using var rows = _accelerator.Allocate1D<float>(passing.Count * FaceStride);
        _gatherKernel(passing.Count, indices.View, decoded.View, rows.View);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var packed = await rows.CopyToHostAsync<float>(0, passing.Count * FaceStride).ConfigureAwait(false);

        var candidates = new List<DetectedFace>(passing.Count);
        for (int c = 0; c < passing.Count; c++)
        {
            int o = c * FaceStride;
            var landmarks = new List<(float X, float Y)>(6);
            for (int j = 0; j < 6; j++) landmarks.Add((packed[o + 5 + j * 2], packed[o + 6 + j * 2]));
            candidates.Add(new DetectedFace
            {
                Confidence = packed[o], X = packed[o + 1], Y = packed[o + 2],
                Width = packed[o + 3], Height = packed[o + 4], Landmarks = landmarks,
            });
        }
        var merged = WeightedNms(candidates, iouThreshold);
        if (maxFaces > 0 && merged.Count > maxFaces) merged = merged.Take(maxFaces).ToList();
        return merged.ToArray();
    }

    /// <summary>One store per thread: the confidence column of the decoded anchors.</summary>
    private static void ConfidenceColumnKernel(Index1D i,
        ArrayView1D<float, Stride1D.Dense> decoded, ArrayView1D<float, Stride1D.Dense> conf)
        => conf[i] = decoded[i * FaceStride];

    /// <summary>Positional gather (WebGL-safe): row j of the output = decoded row indices[j], 17 stores per thread.</summary>
    private static void GatherRowsKernel(Index1D j, ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> decoded, ArrayView1D<float, Stride1D.Dense> rows)
    {
        // Constant slots, straight-line (see DecodeKernel: loop-computed slots misplace stores on WebGL).
        int src = indices[j] * FaceStride;
        int dst = j * FaceStride;
        rows[dst] = decoded[src];
        rows[dst + 1] = decoded[src + 1];
        rows[dst + 2] = decoded[src + 2];
        rows[dst + 3] = decoded[src + 3];
        rows[dst + 4] = decoded[src + 4];
        rows[dst + 5] = decoded[src + 5];
        rows[dst + 6] = decoded[src + 6];
        rows[dst + 7] = decoded[src + 7];
        rows[dst + 8] = decoded[src + 8];
        rows[dst + 9] = decoded[src + 9];
        rows[dst + 10] = decoded[src + 10];
        rows[dst + 11] = decoded[src + 11];
        rows[dst + 12] = decoded[src + 12];
        rows[dst + 13] = decoded[src + 13];
        rows[dst + 14] = decoded[src + 14];
        rows[dst + 15] = decoded[src + 15];
        rows[dst + 16] = decoded[src + 16];
    }

    internal static DetectedFace[] ToFaces(float[] packed)
    {
        int n = packed.Length == 0 ? 0 : (int)packed[0];
        var faces = new DetectedFace[n];
        for (int f = 0; f < n; f++)
        {
            int o = 1 + f * FaceStride;
            var landmarks = new List<(float X, float Y)>(6);
            for (int j = 0; j < 6; j++) landmarks.Add((packed[o + 5 + j * 2], packed[o + 6 + j * 2]));
            faces[f] = new DetectedFace
            {
                Confidence = packed[o],
                X = packed[o + 1],
                Y = packed[o + 2],
                Width = packed[o + 3],
                Height = packed[o + 4],
                Landmarks = landmarks,
            };
        }
        return faces;
    }

    // params: [inputSize, contentW, contentH, padX, padY, imageW, imageH, confThreshold, iouThreshold, maxOut, numAnchors]

    /// <summary>
    /// One thread per anchor: sigmoid score, then box and landmarks decoded and mapped to image pixels exactly as
    /// <see cref="DecodeDetectionsReference"/> does. Anchors under the threshold get confidence -1.
    /// Output per anchor (<see cref="FaceStride"/> floats): confidence, x, y, width, height, landmarks (x, y) * 6.
    /// </summary>
    private static void DecodeKernel(Index1D i,
        ArrayView1D<float, Stride1D.Dense> regressors,     // [896 * 16]
        ArrayView1D<float, Stride1D.Dense> classificators, // [896]
        ArrayView1D<float, Stride1D.Dense> anchors,        // [896 * 2]
        ArrayView1D<float, Stride1D.Dense> p,
        ArrayView1D<float, Stride1D.Dense> decoded)        // [896 * 17]
    {
        float scale = p[0];
        int contentW = (int)p[1], contentH = (int)p[2], padX = (int)p[3], padY = (int)p[4];
        int imageW = (int)p[5], imageH = (int)p[6];
        float threshold = p[7];
        int o = i * FaceStride;

        float raw = classificators[i];
        if (raw > 100f) raw = 100f;
        if (raw < -100f) raw = -100f;
        float score = 1f / (1f + MathF.Exp(-raw));

        // No early return: all 17 slots are written by every thread (WebGL Transform Feedback captures one
        // positional record per thread; a partly written record is garbage there). Below threshold = confidence -1.
        int r = i * 16;
        float ax = anchors[i * 2], ay = anchors[i * 2 + 1];
        float cx = regressors[r] / scale + ax;
        float cy = regressors[r + 1] / scale + ay;
        float w = regressors[r + 2] / scale;
        float h = regressors[r + 3] / scale;

        float x1 = ClampF(KMap(cx - w / 2, scale, contentW, padX, imageW), 0, imageW);
        float y1 = ClampF(KMap(cy - h / 2, scale, contentH, padY, imageH), 0, imageH);
        float x2 = ClampF(KMap(cx + w / 2, scale, contentW, padX, imageW), 0, imageW);
        float y2 = ClampF(KMap(cy + h / 2, scale, contentH, padY, imageH), 0, imageH);

        // Straight-line stores at constant slots: WebGL maps each output slot of a thread to a Transform Feedback
        // varying, and stores at loop-computed slots landed in the wrong varyings there (measured: a pixel x read
        // back as the confidence). Every other backend is indifferent to the form.
        decoded[o] = score >= threshold ? score : -1f;
        decoded[o + 1] = x1;
        decoded[o + 2] = y1;
        decoded[o + 3] = MathF.Max(0, x2 - x1);
        decoded[o + 4] = MathF.Max(0, y2 - y1);
        decoded[o + 5] = LandmarkX(regressors[r + 4], scale, ax, contentW, padX, imageW);
        decoded[o + 6] = LandmarkX(regressors[r + 5], scale, ay, contentH, padY, imageH);
        decoded[o + 7] = LandmarkX(regressors[r + 6], scale, ax, contentW, padX, imageW);
        decoded[o + 8] = LandmarkX(regressors[r + 7], scale, ay, contentH, padY, imageH);
        decoded[o + 9] = LandmarkX(regressors[r + 8], scale, ax, contentW, padX, imageW);
        decoded[o + 10] = LandmarkX(regressors[r + 9], scale, ay, contentH, padY, imageH);
        decoded[o + 11] = LandmarkX(regressors[r + 10], scale, ax, contentW, padX, imageW);
        decoded[o + 12] = LandmarkX(regressors[r + 11], scale, ay, contentH, padY, imageH);
        decoded[o + 13] = LandmarkX(regressors[r + 12], scale, ax, contentW, padX, imageW);
        decoded[o + 14] = LandmarkX(regressors[r + 13], scale, ay, contentH, padY, imageH);
        decoded[o + 15] = LandmarkX(regressors[r + 14], scale, ax, contentW, padX, imageW);
        decoded[o + 16] = LandmarkX(regressors[r + 15], scale, ay, contentH, padY, imageH);
    }

    /// <summary>One landmark coordinate: regressor offset + anchor, mapped to image pixels and clamped (x or y).</summary>
    private static float LandmarkX(float reg, float scale, float anchor, int content, int pad, int image)
        => ClampF(KMap(reg / scale + anchor, scale, content, pad, image), 0, image);

    /// <summary>
    /// One thread: MediaPipe weighted NMS over the decoded anchors, same arithmetic as <see cref="WeightedNms"/>. The
    /// highest remaining confidence seeds a face; every remaining anchor overlapping it by more than the IoU threshold
    /// is blended in by its confidence and removed. Stops after maxOut faces, which equals the managed version's full
    /// NMS followed by Take(maxOut): each face depends only on the faces before it.
    /// Output: [count, then count * <see cref="FaceStride"/> floats].
    /// </summary>
    private static void WeightedNmsKernel(Index1D index,
        ArrayView1D<float, Stride1D.Dense> decoded, // [896 * 17], confidences consumed (set to -1)
        ArrayView1D<float, Stride1D.Dense> p,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float iouThreshold = p[8];
        int maxOut = (int)p[9];
        int n = (int)p[10];
        int count = 0;
        while (count < maxOut)
        {
            int seed = -1;
            float best = -1f;
            for (int i = 0; i < n; i++)
            {
                float c = decoded[i * FaceStride];
                if (c >= 0f && c > best) { best = c; seed = i; }
            }
            if (seed < 0) break;

            int s = seed * FaceStride;
            float sx = decoded[s + 1], sy = decoded[s + 2], swd = decoded[s + 3], sht = decoded[s + 4];
            decoded[s] = -1f;

            int outBase = 1 + count * FaceStride;
            float sumW = best;
            for (int k = 1; k < FaceStride; k++) output[outBase + k] = decoded[s + k] * best;

            for (int i = 0; i < n; i++)
            {
                int o = i * FaceStride;
                float c = decoded[o];
                if (c < 0f) continue;
                float ox = decoded[o + 1], oy = decoded[o + 2], ow = decoded[o + 3], oh = decoded[o + 4];
                float ix1 = MathF.Max(sx, ox), iy1 = MathF.Max(sy, oy);
                float ix2 = MathF.Min(sx + swd, ox + ow), iy2 = MathF.Min(sy + sht, oy + oh);
                float inter = MathF.Max(0, ix2 - ix1) * MathF.Max(0, iy2 - iy1);
                float union = swd * sht + ow * oh - inter;
                float iou = union > 0 ? inter / union : 0;
                if (iou <= iouThreshold) continue;
                sumW += c;
                for (int k = 1; k < FaceStride; k++) output[outBase + k] += decoded[o + k] * c;
                decoded[o] = -1f;
            }

            output[outBase] = best;
            for (int k = 1; k < FaceStride; k++) output[outBase + k] /= sumW;
            count++;
        }
        output[0] = count;
    }

    private static float KMap(float n, float scale, int content, int pad, int image)
        => content <= 0 ? n * image : (n * scale - pad) / content * image;

    private static float ClampF(float v, float lo, float hi) => v < lo ? lo : (v > hi ? hi : v);

    /// <summary>
    /// Validation only (tests): the managed decode + NMS this pipeline used before the GPU path. Reads the full output
    /// tensors back (61 KB): never use it on a live path.
    /// </summary>
    internal async Task<(DetectedFace[] Gpu, DetectedFace[] Reference)> DetectBothWaysAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold, float iouThreshold, int maxFaces)
    {
        var (contentW, contentH, padX, padY) = ImagePreprocessKernel.Letterbox(width, height, _inputSize, _inputSize);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize,
            mean: new[] { 0.5f, 0.5f, 0.5f }, std: new[] { 0.5f, 0.5f, 0.5f }, preserveAspect: true);
        int H = _inputSize, W = _inputSize;
        using var nhwcBuf = _accelerator.Allocate1D<float>(3 * H * W);
        new TransposeKernel(_accelerator).Transpose(preprocessed.View, nhwcBuf.View, new[] { 3, H, W }, new[] { 1, 2, 0 });
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(nhwcBuf.View, new[] { 1, H, W, 3 })
        }).ConfigureAwait(false);
        var (regName, clsName) = ResolveOutputNames();
        var regressors = await ReadOutputAsync(outputs, regName, NumAnchors * 16).ConfigureAwait(false);
        var classificators = await ReadOutputAsync(outputs, clsName, NumAnchors).ConfigureAwait(false);
        var reference = DecodeDetectionsReference(regressors, classificators, width, height,
            contentW, contentH, padX, padY, confidenceThreshold, iouThreshold, maxFaces);
        var gpu = await DecodeOnGpuAsync(outputs[regName].Data, outputs[clsName].Data, width, height,
            contentW, contentH, padX, padY, confidenceThreshold, iouThreshold, maxFaces).ConfigureAwait(false);
        return (gpu, reference);
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

    private DetectedFace[] DecodeDetectionsReference(float[] regressors, float[] classificators,
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

            // This short-range TFLite emits SSD order [x_center, y_center, w, h] then keypoints
            // (x,y)*6 — NOT MediaPipe calculator reverse_output_order (y,x,h,w). Proved 2026-09-27:
            // yxhw left the box roughly OK (h≈w) but put green dots nowhere on eyes/nose/mouth;
            // xywh matches LiteRT+Comfy decode and lands RE/LE/nose/mouth on the portrait features.
            float cx = regressors[regBase + 0] / scale + _anchors[i, 0];
            float cy = regressors[regBase + 1] / scale + _anchors[i, 1];
            float w = regressors[regBase + 2] / scale;
            float h = regressors[regBase + 3] / scale;

            float x1 = MapX(cx - w / 2, contentW, padX, imageWidth);
            float y1 = MapY(cy - h / 2, contentH, padY, imageHeight);
            float x2 = MapX(cx + w / 2, contentW, padX, imageWidth);
            float y2 = MapY(cy + h / 2, contentH, padY, imageHeight);

            var landmarks = new List<(float X, float Y)>(6);
            for (int j = 0; j < 6; j++)
            {
                float lx = regressors[regBase + 4 + j * 2] / scale + _anchors[i, 0];
                float ly = regressors[regBase + 4 + j * 2 + 1] / scale + _anchors[i, 1];
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
        _anchorBuffer.Dispose();
        _params?.Dispose();
    }
}
