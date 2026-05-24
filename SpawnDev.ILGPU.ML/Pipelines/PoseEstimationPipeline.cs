using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Pose estimation pipeline for MoveNet Lightning.
/// Handles image preprocessing, GPU inference, and keypoint decoding.
///
/// Usage:
///   var session = await InferenceSession.CreateFromHuggingFaceAsync(accelerator, hub,
///       ModelHub.KnownModels.MoveNetLightning, ModelHub.KnownFiles.OnnxModel);
///   var pipeline = new PoseEstimationPipeline(session, accelerator);
///   var result = await pipeline.EstimateAsync(rgbaPixels, width, height);
///   foreach (var kp in result.Keypoints.Where(k => k.Confidence > 0.3))
///       Console.WriteLine($"{kp.Name}: ({kp.X:F0}, {kp.Y:F0}) conf={kp.Confidence:P0}");
/// </summary>
public class PoseEstimationPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly int _inputSize;

    public PoseEstimationPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 192)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _inputSize = inputSize;
    }

    /// <summary>
    /// Estimate pose keypoints from an RGBA image.
    /// </summary>
    public async Task<PoseResult> EstimateAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.3f)
    {
        var sw = Stopwatch.StartNew();

        // MoveNet expects NHWC [0,255] — simple bilinear resize on CPU
        var floatInput = new float[_inputSize * _inputSize * 3];
        for (int y = 0; y < _inputSize; y++)
        {
            for (int x = 0; x < _inputSize; x++)
            {
                int srcX = x * width / _inputSize;
                int srcY = y * height / _inputSize;
                srcX = Math.Clamp(srcX, 0, width - 1);
                srcY = Math.Clamp(srcY, 0, height - 1);
                int srcIdx = srcY * width + srcX;
                int rgba = rgbaPixels[srcIdx];
                int dstIdx = (y * _inputSize + x) * 3;
                floatInput[dstIdx + 0] = (rgba & 0xFF);           // R
                floatInput[dstIdx + 1] = ((rgba >> 8) & 0xFF);    // G
                floatInput[dstIdx + 2] = ((rgba >> 16) & 0xFF);   // B
            }
        }
        using var inputBuf = _accelerator.Allocate1D(floatInput);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, _inputSize, _inputSize, 3 });

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        // Read output [1, 1, 17, 3] = 51 floats
        var output = outputs[_session.OutputNames[0]];
        int elems = Math.Min(output.ElementCount, 51);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync();
        var outputData = await readBuf.CopyToHostAsync<float>(0, elems);

        // Decode keypoints
        var keypoints = PoseSkeleton.DecodeMoveNetOutput(outputData, width, height);

        sw.Stop();

        return new PoseResult
        {
            Keypoints = keypoints,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
