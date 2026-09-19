using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
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
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="EstimateAsync(TypedArray, int, int, float)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<PoseResult> EstimateAsync(
        int[] rgbaPixels, int width, int height,
        float confidenceThreshold = 0.3f)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await EstimateAsync(rgbaBuf.View, width, height, confidenceThreshold).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<PoseResult> EstimateAsync(
        TypedArray rgbaPixels, int width, int height,
        float confidenceThreshold = 0.3f)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await EstimateAsync(rgbaBuf.View, width, height, confidenceThreshold).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public Task<PoseResult> EstimateAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.3f)
        => EstimateCoreAsync(rgbaPixels, width, height, confidenceThreshold);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<PoseResult> EstimateAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold = 0.3f)
        => EstimateCoreAsync(rgbaPixels.View, width, height, confidenceThreshold);

    private async Task<PoseResult> EstimateCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float confidenceThreshold)
    {
        var sw = Stopwatch.StartNew();

        // MoveNet expects NHWC [0,255]: GPU resize+unpack via ForwardRaw, then CHW→HWC transpose
        int H = _inputSize, W = _inputSize;
        using var preprocessed = _accelerator.Allocate1D<float>(3 * H * W);
        _preprocess.ForwardRaw(rgbaPixels, preprocessed.View, width, height, W, H);

        using var nhwcBuf = _accelerator.Allocate1D<float>(3 * H * W);
        new TransposeKernel(_accelerator).Transpose(preprocessed.View, nhwcBuf.View,
            new[] { 3, H, W }, new[] { 1, 2, 0 }); // CHW → HWC
        var inputTensor = new Tensor(nhwcBuf.View, new[] { 1, H, W, 3 });

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        // Read output [1, 1, 17, 3] = 51 floats
        var output = outputs[_session.OutputNames[0]];
        int elems = Math.Min(output.ElementCount, 51);
        using var readBuf = _accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var outputData = await readBuf.CopyToHostAsync<float>(0, elems).ConfigureAwait(false);

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
