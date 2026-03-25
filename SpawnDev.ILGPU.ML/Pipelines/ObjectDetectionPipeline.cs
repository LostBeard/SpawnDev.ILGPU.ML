using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Data;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
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
    public async Task<DetectionResult> DetectAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 100)
    {
        var sw = Stopwatch.StartNew();

        // Preprocess: RGBA → NCHW float [0,1], letterbox to inputSize
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        // Run inference
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        // Read output to CPU
        var output = outputs[_session.OutputNames[0]];
        int elems = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync();
        var outputData = await readBuf.CopyToHostAsync<float>(0, elems);

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
