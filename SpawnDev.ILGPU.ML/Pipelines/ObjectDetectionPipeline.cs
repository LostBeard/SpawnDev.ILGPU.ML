using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Data;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Object detection pipeline for YOLOv8.
/// Handles image preprocessing, GPU inference, and CPU postprocessing (NMS, box decode).
///
/// Usage:
///   var session = await InferenceSession.CreateFromHuggingFaceAsync(accelerator, hub,
///       ModelHub.KnownModels.YOLOv8n, "yolov8n-onnx-web/yolov8n.onnx");
///   var pipeline = new ObjectDetectionPipeline(session, accelerator);
///   var result = await pipeline.DetectAsync(rgbaPixels, width, height);
///   foreach (var obj in result.Objects)
///       Console.WriteLine($"{obj.Label}: {obj.Confidence:P0} at ({obj.X},{obj.Y})");
/// </summary>
public class ObjectDetectionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly string[] _labels;
    private readonly int _inputSize;
    private Graph.SessionGraphCapture? _capture;
    private ElementWiseKernels? _ew;

    /// <summary>
    /// Record the detector's forward once (fixed input size) and replay it per frame: a dispatch plan on WebGPU, a CUDA graph on CUDA
    /// (<see cref="Graph.SessionGraphCapture"/>, as <see cref="DepthEstimationPipeline"/> and the CLAP audio tower use). Opt-in, as for the CLAP tower: a replay
    /// removes the host's per-node dispatch work, which only pays where the host is the bottleneck, and the recording pins its intermediates
    /// (Riker's RX 590 with CLAP, 2026-11: no gain, GPU-bound, +700 MB of Chrome GPU memory). Asked for by Riker (AssistZ, 2026-11-06: 261-node YOLOv8n,
    /// RunAsync 106 ms at 640 and 80 at 320 on the RX 590, the fixed part suspected to be host dispatch). Set before the first frame; the recording
    /// keeps its memory until the pipeline is disposed. <see cref="CaptureStatus"/> says whether it replays.
    /// </summary>
    public bool EnableGraphCapture { get; set; }
    /// <summary>Whether the recorded forward replays, and if not why (<see cref="Graph.SessionGraphCapture.CaptureStatus"/>).</summary>
    public string CaptureStatus => _capture?.CaptureStatus ?? "not attempted";

    public ObjectDetectionPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 640, string[]? labels = null)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _labels = labels ?? CocoLabels.Labels;
        _inputSize = inputSize;
    }

    /// <summary>
    /// Detect objects in an RGBA image.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="DetectAsync(TypedArray, int, int, float, float, int)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<DetectionResult> DetectAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 100)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await DetectAsync(rgbaBuf.View, width, height, confidenceThreshold, iouThreshold, maxDetections)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<DetectionResult> DetectAsync(
        TypedArray rgbaPixels, int width, int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 100)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await DetectAsync(rgbaBuf.View, width, height, confidenceThreshold, iouThreshold, maxDetections)
            .ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public Task<DetectionResult> DetectAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 100)
        => DetectCoreAsync(rgbaPixels, width, height, confidenceThreshold, iouThreshold, maxDetections);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<DetectionResult> DetectAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 100)
        => DetectCoreAsync(rgbaPixels.View, width, height, confidenceThreshold, iouThreshold, maxDetections);

    private async Task<DetectionResult> DetectCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold, float iouThreshold, int maxDetections)
    {
        var sw = Stopwatch.StartNew();

        // YOLOv8 contract: letterbox (aspect-preserving) + NCHW float [0,1] — NOT ImageNet mean/std,
        // NOT stretch-to-square. Comment used to claim letterbox while Forward() defaults did neither;
        // YoloPostProcessor then unmapped as if letterboxed → boxes landed nowhere near the objects
        // (live /detect Street: one mis-placed "traffic light"). Same Letterbox() ints drive both sides.
        var (contentW, contentH, padX, padY) = Kernels.ImagePreprocessKernel.Letterbox(
            width, height, _inputSize, _inputSize);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize,
            mean: new[] { 0f, 0f, 0f },
            std: new[] { 1f, 1f, 1f },
            preserveAspect: true, padValue: 114f / 255f);          // ultralytics letterbox: grey 114 padding (replicated edge pixels smear partly visible people into tall thin boxes)

        // Run inference
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });
        var feeds = new Dictionary<string, Tensor> { [_session.InputNames[0]] = inputTensor };
        Dictionary<string, Tensor> outputs;
        if (EnableGraphCapture)
        {
            // KeepDrains: buffers released during the recorded pass are recycled on a plain forward's schedule instead of all being pinned (CLAP: +132 vs +323 MiB)
            _capture ??= new Graph.SessionGraphCapture(_session, _accelerator) { KeepDrains = true };
            outputs = await _capture.RunAsync(feeds).ConfigureAwait(false);
        }
        else outputs = await _session.RunAsync(feeds).ConfigureAwait(false);

        // Read output to CPU. A direct forward's outputs are pool-rented and nothing else returns them: they go back once read. Before 2026-11-06 they
        // were never returned, one 4 MiB head output leaked per frame (Riker, AssistZ, RX 590: Chrome's GPU memory 65 -> 1,449 MB over 320
        // inferences; gate Pipeline_YOLOv8_DirectFrames_DoNotGrowDeviceMemory). A replay's outputs belong to the capture.
        bool ownOutputs = !EnableGraphCapture || _capture?.LastRunReplayed != true;
        var output = outputs[_session.OutputNames[0]];
        int elems = output.ElementCount;
        float[] outputData;
        try
        {
            using var readBuf = _accelerator.Allocate1D<float>(elems);
            (_ew ??= new ElementWiseKernels(_accelerator)).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            outputData = await readBuf.CopyToHostAsync<float>(0, elems).ConfigureAwait(false);
        }
        finally
        {
            if (ownOutputs) _session.ReturnOutputs(outputs);
        }

        // the head's detection count follows the input size ([1, 4 + classes, N]: 8400 at 640, 2100 at 320). It was left at Process's default of 8400,
        // so any other input size threw (Riker, AssistZ, 2026-11-06).
        int channels = 4 + _labels.Length;
        int numDetections = output.Shape is { Length: 3 } sh && sh[1] == channels ? sh[2] : elems / channels;
        if ((long)numDetections * channels != elems)
            throw new InvalidOperationException($"YOLO output [{string.Join(", ", output.Shape)}] does not hold 4 + {_labels.Length} channels per detection");

        // Postprocess: transpose, filter, NMS — pass the SAME letterbox rect the preprocess used
        var detections = YoloPostProcessor.Process(
            outputData,
            numClasses: _labels.Length,
            numDetections: numDetections,
            confThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            inputWidth: _inputSize,
            inputHeight: _inputSize,
            originalWidth: width,
            originalHeight: height,
            contentW: contentW,
            contentH: contentH,
            padX: padX,
            padY: padY);

        sw.Stop();

        return new DetectionResult
        {
            Objects = detections.Take(maxDetections).Select(d => new DetectedObject
            {
                Label = d.ClassId >= 0 && d.ClassId < _labels.Length ? _labels[d.ClassId] : $"class_{d.ClassId}",
                ClassId = d.ClassId,
                Confidence = d.Confidence,
                X = d.X1,
                Y = d.Y1,
                Width = d.X2 - d.X1,
                Height = d.Y2 - d.Y1,
            }).ToArray(),
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            ImageWidth = width,
            ImageHeight = height,
        };
    }

    public void Dispose()
    {
        _capture?.Dispose();
        _session?.Dispose();
    }
}
