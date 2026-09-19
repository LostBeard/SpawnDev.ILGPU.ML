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

        // Preprocess: RGBA → NCHW float [0,1], letterbox to inputSize
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize);

        // Run inference
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        // Read output to CPU
        var output = outputs[_session.OutputNames[0]];
        int elems = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var outputData = await readBuf.CopyToHostAsync<float>(0, elems).ConfigureAwait(false);

        // Postprocess: transpose, filter, NMS
        var detections = YoloPostProcessor.Process(
            outputData,
            numClasses: _labels.Length,
            confThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            inputWidth: _inputSize,
            inputHeight: _inputSize,
            originalWidth: width,
            originalHeight: height);

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
        _session?.Dispose();
    }
}
