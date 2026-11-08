using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Data;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Object detection for the DETR family exported by Hugging Face transformers: RT-DETR / RT-DETRv2, D-FINE, DEIM (Apache-2.0). Their contract
/// differs from YOLO in every step, so they get their own pipeline:
/// <list type="bullet">
/// <item>Input: the picture STRETCHED to a square (no letterbox), PIL bilinear (antialiased when shrinking), rescaled to [0,1], no mean/std
/// (<c>RTDetrImageProcessor</c>: do_resize, resample=2, do_rescale, do_normalize=false). <see cref="Kernels.ImagePreprocessKernel.ForwardPil"/>
/// matches Pillow bit for bit.</item>
/// <item>Outputs: <c>logits</c> [1, Q, C] and <c>pred_boxes</c> [1, Q, 4], boxes normalised (cx, cy, w, h).</item>
/// <item>Post-process: sigmoid per (query, class) - focal-loss heads, no softmax, no background class - then the top Q (query, class) pairs
/// above the threshold, as <c>post_process_object_detection</c> does. No NMS: the set prediction is already one box per object.</item>
/// </list>
/// The (query, class) scan runs on the GPU: one thread per query writes its best class, score, box and how many OTHER classes clear the
/// threshold, so a frame reads back 7 floats per query instead of the whole [Q, C] logits; only a frame with a query that clears the
/// threshold in two classes reads the logits too (the pair semantics need them).
/// </summary>
/// <remarks>
/// Asked for by Riker (AssistZ, 2026-11-06) as the YOLOv8 replacement: D-FINE-S at bar 0.5 found 190/211 person boxes at 10-30 m vs
/// YOLOv8n's 170, 16/127 at 30-50 m vs 1, with 0 false at rest (onnxruntime CPU, PIL preprocessing).
/// </remarks>
public class DetrDetectionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly string[] _labels;
    private readonly int _inputSize;
    private readonly string _logitsName, _boxesName;
    private Graph.SessionGraphCapture? _capture;
    private MemoryBuffer1D<float, Stride1D.Dense>? _input, _decoded;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float>? _decodeKernel;

    /// <summary>Floats per query in the decoded buffer: best logit, best class, classes above threshold, cx, cy, w, h.</summary>
    internal const int DecodeStride = 7;

    /// <summary>
    /// Record the forward once and replay it per frame (<see cref="Graph.SessionGraphCapture"/>: a WebGPU dispatch plan, a CUDA graph), as
    /// <see cref="ObjectDetectionPipeline.EnableGraphCapture"/>. A D-FINE forward is ~1,100 nodes, so the host work a replay removes is ~4x
    /// YOLOv8n's. Set before the first frame; <see cref="CaptureStatus"/> says whether it replays.
    /// </summary>
    public bool EnableGraphCapture { get; set; }
    /// <summary>Whether the recorded forward replays, and if not why.</summary>
    public string CaptureStatus => _capture?.CaptureStatus ?? "not attempted";

    /// <param name="session">A DETR-family model with outputs <c>logits</c> and <c>pred_boxes</c> (or two outputs in that order).</param>
    /// <param name="inputSize">The square input the model was exported for (640 for the onnx-community D-FINE / RT-DETR exports).</param>
    /// <param name="labels">Class names in logit order; COCO's 80 by default.</param>
    public DetrDetectionPipeline(InferenceSession session, Accelerator accelerator, int inputSize = 640, string[]? labels = null)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _labels = labels ?? CocoLabels.Labels;
        _inputSize = inputSize;
        var outs = session.OutputNames;
        _logitsName = outs.Contains("logits") ? "logits" : outs[0];
        _boxesName = outs.Contains("pred_boxes") ? "pred_boxes" : outs[1];
    }

    /// <summary>Detect objects in an RGBA image (packed, R in the low byte). Pulls the frame onto the managed heap to upload it: prefer the TypedArray or view overloads in a browser.</summary>
    /// <param name="region">Optional source rect (x, y, width, height): detect in that crop only, as PIL <c>crop(box)</c> then resize; boxes come back in full-image pixels.</param>
    public async Task<DetectionResult> DetectAsync(int[] rgbaPixels, int width, int height,
        float threshold = 0.5f, int maxDetections = 300, (int X, int Y, int Width, int Height)? region = null)
    {
        using var buf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await DetectCoreAsync(buf.View, width, height, threshold, maxDetections, region).ConfigureAwait(false);
    }

    /// <summary>Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>); the pixels never enter the .NET heap.</summary>
    public async Task<DetectionResult> DetectAsync(TypedArray rgbaPixels, int width, int height,
        float threshold = 0.5f, int maxDetections = 300, (int X, int Y, int Width, int Height)? region = null)
    {
        using var buf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await DetectCoreAsync(buf.View, width, height, threshold, maxDetections, region).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA: no upload. Tiling one frame is several calls on the same view with different regions.</summary>
    public Task<DetectionResult> DetectAsync(ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold = 0.5f, int maxDetections = 300, (int X, int Y, int Width, int Height)? region = null)
        => DetectCoreAsync(rgbaPixels, width, height, threshold, maxDetections, region);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<DetectionResult> DetectAsync(MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold = 0.5f, int maxDetections = 300, (int X, int Y, int Width, int Height)? region = null)
        => DetectCoreAsync(rgbaPixels.View, width, height, threshold, maxDetections, region);

    private async Task<DetectionResult> DetectCoreAsync(ArrayView1D<int, Stride1D.Dense> rgba, int width, int height,
        float threshold, int maxDetections, (int X, int Y, int Width, int Height)? region)
    {
        if (!(threshold > 0f && threshold < 1f))
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "A sigmoid score threshold lies strictly between 0 and 1.");
        var sw = Stopwatch.StartNew();
        var (rx, ry, rw, rh) = region ?? (0, 0, width, height);

        int n = 3 * _inputSize * _inputSize;
        _input ??= _accelerator.Allocate1D<float>(n);
        _preprocess.ForwardPil(rgba, _input.View, width, height, _inputSize, _inputSize, srcX: rx, srcY: ry, regionW: rw, regionH: rh);

        var feeds = new Dictionary<string, Tensor> { [_session.InputNames[0]] = new Tensor(_input.View, new[] { 1, 3, _inputSize, _inputSize }) };
        Dictionary<string, Tensor> outputs;
        if (EnableGraphCapture)
        {
            _capture ??= new Graph.SessionGraphCapture(_session, _accelerator) { KeepDrains = true };
            outputs = await _capture.RunAsync(feeds).ConfigureAwait(false);
        }
        else outputs = await _session.RunAsync(feeds).ConfigureAwait(false);
        // A direct forward's outputs are pool-rented and returned once read; a replay's belong to the capture (see ObjectDetectionPipeline).
        bool ownOutputs = !EnableGraphCapture || _capture?.LastRunReplayed != true;

        float[] decoded;
        float[]? logits = null;
        int q, c;
        try
        {
            var lt = outputs[_logitsName];
            var bt = outputs[_boxesName];
            c = lt.Shape[^1];
            q = lt.ElementCount / c;
            if (c != _labels.Length)
                throw new InvalidOperationException($"DETR logits [{string.Join(", ", lt.Shape)}] has {c} classes; the pipeline has {_labels.Length} labels.");
            if (bt.ElementCount != q * 4)
                throw new InvalidOperationException($"DETR pred_boxes [{string.Join(", ", bt.Shape)}] is not 4 values per query for {q} queries.");

            float logitThreshold = MathF.Log(threshold / (1f - threshold));
            if (_decoded == null || _decoded.Length < q * DecodeStride)
            {
                _decoded?.Dispose();
                _decoded = _accelerator.Allocate1D<float>(q * DecodeStride);
            }
            _decodeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, float>(DecodeImpl);
            _decodeKernel(q, lt.Data.SubView(0, q * c), bt.Data.SubView(0, q * 4), _decoded.View, q, c, logitThreshold);
            decoded = await _decoded.CopyToHostAsync<float>(0, q * DecodeStride).ConfigureAwait(false);

            bool multi = false;
            for (int i = 0; i < q; i++) if (decoded[i * DecodeStride + 2] >= 2f) { multi = true; break; }
            if (multi)
            {
                // Rare: a query above the threshold in two classes. post_process_object_detection keeps both pairs, which needs that row's logits.
                using var lcopy = _accelerator.Allocate1D<float>(q * c);
                lcopy.View.CopyFrom(lt.Data.SubView(0, q * c));
                logits = await lcopy.CopyToHostAsync<float>(0, q * c).ConfigureAwait(false);
            }
        }
        finally
        {
            if (ownOutputs) _session.ReturnOutputs(outputs);
        }

        var dets = Select(decoded, logits, q, c, threshold);
        int keep = Math.Min(Math.Min(dets.Count, q), maxDetections);
        var objects = new DetectedObject[keep];
        for (int i = 0; i < keep; i++)
        {
            var (score, query, cls) = dets[i];
            int o = query * DecodeStride;
            float cx = decoded[o + 3], cy = decoded[o + 4], bw = decoded[o + 5], bh = decoded[o + 6];
            objects[i] = new DetectedObject
            {
                Label = cls < _labels.Length ? _labels[cls] : $"class_{cls}",
                ClassId = cls,
                Confidence = score,
                X = rx + (cx - bw / 2f) * rw,
                Y = ry + (cy - bh / 2f) * rh,
                Width = bw * rw,
                Height = bh * rh,
            };
        }
        sw.Stop();
        return new DetectionResult { Objects = objects, InferenceTimeMs = sw.Elapsed.TotalMilliseconds, ImageWidth = width, ImageHeight = height };
    }

    /// <summary>
    /// The (score, query, class) pairs <c>post_process_object_detection</c> keeps, highest first: every pair whose sigmoid score is above
    /// <paramref name="threshold"/>. Pairs come from the decoded per-query best class, plus the logits row of any query above it in more than one class.
    /// </summary>
    internal static List<(float Score, int Query, int Class)> Select(float[] decoded, float[]? logits, int q, int c, float threshold)
    {
        var dets = new List<(float, int, int)>();
        for (int i = 0; i < q; i++)
        {
            int o = i * DecodeStride;
            int count = (int)decoded[o + 2];
            if (count == 0) continue;
            if (count == 1 || logits == null) { dets.Add((Sigmoid(decoded[o]), i, (int)decoded[o + 1])); continue; }
            for (int k = 0; k < c; k++)
            {
                float s = Sigmoid(logits[i * c + k]);
                if (s > threshold) dets.Add((s, i, k));
            }
        }
        dets.Sort((a, b) => b.Item1.CompareTo(a.Item1));
        return dets;
    }

    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>One thread per query: best class and its logit, how many classes clear the threshold (in logit space), and the box.</summary>
    private static void DecodeImpl(Index1D qi,
        ArrayView1D<float, Stride1D.Dense> logits, ArrayView1D<float, Stride1D.Dense> boxes, ArrayView1D<float, Stride1D.Dense> outp,
        int q, int c, float logitThreshold)
    {
        int b = qi * c;
        float best = logits[b];
        int bestC = 0;
        int count = 0;
        for (int k = 0; k < c; k++)
        {
            float v = logits[b + k];
            if (v > best) { best = v; bestC = k; }
            if (v > logitThreshold) count++;
        }
        int o = qi * DecodeStride;
        outp[o] = best;
        outp[o + 1] = bestC;
        outp[o + 2] = count;
        outp[o + 3] = boxes[qi * 4];
        outp[o + 4] = boxes[qi * 4 + 1];
        outp[o + 5] = boxes[qi * 4 + 2];
        outp[o + 6] = boxes[qi * 4 + 3];
    }

    public void Dispose()
    {
        _capture?.Dispose();
        _input?.Dispose();
        _decoded?.Dispose();
        _preprocess.Dispose();
        _session?.Dispose();
    }
}
