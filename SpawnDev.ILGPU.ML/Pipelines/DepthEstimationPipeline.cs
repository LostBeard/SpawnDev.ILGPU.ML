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
        int inputSize = 518)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
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

        // Read depth output
        var output = outputs[_session.OutputNames[0]];
        int depthSize = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(depthSize);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, depthSize), readBuf.View, depthSize, 1f);
        await _accelerator.SynchronizeAsync();
        var rawDepth = await readBuf.CopyToHostAsync<float>(0, depthSize);

        // Normalize to [0, 1]
        float min = rawDepth.Min();
        float max = rawDepth.Max();
        float range = max - min;
        var normalized = new float[depthSize];
        if (range > 1e-6f)
        {
            for (int i = 0; i < depthSize; i++)
                normalized[i] = (rawDepth[i] - min) / range;
        }

        // Determine output spatial dimensions from model output shape
        int outH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int outW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        return new DepthResult(normalized, outW, outH, min, max);
    }

    public void Dispose() { }
}
