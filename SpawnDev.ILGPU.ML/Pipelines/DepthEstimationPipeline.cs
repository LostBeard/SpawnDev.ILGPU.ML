using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from depth estimation — depth map as float array.
/// </summary>
public record DepthResult(float[] DepthMap, int Width, int Height, float MinDepth, float MaxDepth);

/// <summary>
/// High-level monocular depth estimation pipeline.
/// Wraps InferenceSession with image preprocessing and depth postprocessing.
///
/// Usage:
///   var pipeline = new DepthEstimationPipeline(session, accelerator);
///   var result = await pipeline.EstimateAsync(rgbaPixels, width, height);
///   // result.DepthMap is [Height × Width] normalized depth values
/// </summary>
public class DepthEstimationPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly int _inputSize;

    public DepthEstimationPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 0)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        // Derive input size from session's compiled input shapes if not specified.
        // Prevents mismatch between preprocessing resolution and compiled graph shapes,
        // which causes silent GPU memory corruption (OOB writes from Conv kernels).
        if (inputSize <= 0)
        {
            var firstShape = session.InputShapes.Values.FirstOrDefault();
            inputSize = firstShape != null && firstShape.Length >= 4 ? firstShape[^1] : 518;
        }
        _inputSize = inputSize;
    }

    /// <summary>
    /// Estimate depth from an RGBA image.
    /// Returns a depth map normalized to [0, 1] (higher = closer).
    /// </summary>
    public async Task<DepthResult> EstimateAsync(int[] rgbaPixels, int width, int height)
    {
        // Upload and preprocess
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        // Create input tensor
        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        // Read depth output to CPU for the CPU-path result
        var output = outputs[_session.OutputNames[0]];
        int depthSize = output.ElementCount;
        Console.WriteLine($"[Depth CPU] Output: shape=[{string.Join(",", output.Shape)}], elements={depthSize}");
        using var readBuf = _accelerator.Allocate1D<float>(depthSize);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, depthSize), readBuf.View, depthSize, 1f);
        await _accelerator.SynchronizeAsync();
        var rawDepth = await readBuf.CopyToHostAsync<float>(0, depthSize);

        // Find min/max on CPU (small cost — depth maps are typically 518×518)
        float min = rawDepth.Min();
        float max = rawDepth.Max();
        float range = max - min;
        var normalized = new float[depthSize];
        if (range > 1e-6f)
        {
            for (int i = 0; i < depthSize; i++)
                normalized[i] = (rawDepth[i] - min) / range;
        }

        int outH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int outW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        return new DepthResult(normalized, outW, outH, min, max);
    }

    /// <summary>
    /// Estimate depth and return a plasma colormap as a GPU MemoryBuffer2D
    /// for zero-copy presentation via ICanvasRenderer.
    /// Entire pipeline stays on GPU — no CPU readback.
    /// Caller owns the returned buffer and must dispose it.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        int[] rgbaPixels, int width, int height)
    {
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize);

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        var output = outputs[_session.OutputNames[0]];
        int depthSize = output.ElementCount;
        int outH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int outW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        Console.WriteLine($"[Depth] Output: shape=[{string.Join(",", output.Shape)}], elements={depthSize}, dataLength={output.Data.Length}");

        // Read output to CPU for min/max
        // Use the actual data view, not SubView(0) which may miss the real offset
        using var tempBuf = _accelerator.Allocate1D<float>(depthSize);
        var ew = new ElementWiseKernels(_accelerator);
        int readSize = Math.Min(depthSize, (int)output.Data.Length);
        ew.Scale(output.Data.SubView(0, readSize), tempBuf.View.SubView(0, readSize), readSize, 1f);
        await _accelerator.SynchronizeAsync();
        var rawDepth = await tempBuf.CopyToHostAsync<float>(0, readSize);
        float minD = rawDepth.Min();
        float maxD = rawDepth.Max();

        Console.WriteLine($"[Depth] Values: min={minD:F4}, max={maxD:F4}, absMax={rawDepth.Max(v => MathF.Abs(v)):F4}, nonZero={rawDepth.Count(v => v != 0)}/{readSize}");

        // GPU: depth → plasma colormap RGBA, output to 2D buffer for zero-copy rendering
        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(outW, outH));
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);
        postprocess.DepthToColormap(output.Data.SubView(0, depthSize), resultBuf.View.BaseView,
            depthSize, minD, maxD);
        await _accelerator.SynchronizeAsync();

        return (resultBuf, outW, outH);
    }

    public void Dispose() { }
}
