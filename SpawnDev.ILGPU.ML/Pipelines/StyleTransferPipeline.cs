using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from style transfer — styled RGBA pixels.
/// </summary>
public record StyleResult(int[] RgbaPixels, int Width, int Height);

/// <summary>
/// High-level neural style transfer pipeline.
/// Input: RGBA image. Output: styled RGBA image (same dimensions).
///
/// Style models expect [0, 255] float input (NOT normalized) and output [0, 255] floats.
///
/// Usage:
///   var pipeline = new StyleTransferPipeline(session, accelerator);
///   var result = await pipeline.TransferAsync(rgbaPixels, width, height);
///   // result.RgbaPixels is the styled image
/// </summary>
public class StyleTransferPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly int _modelH;
    private readonly int _modelW;
    private Kernels.ImagePreprocessKernel? _preprocess;
    private Kernels.ImagePostprocessKernel? _postprocess;

    public StyleTransferPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
        // Use the model's declared input spatial dimensions (typically 224x224)
        var inputShape = session.InputShapes.Values.FirstOrDefault() ?? new[] { 1, 3, 224, 224 };
        _modelH = inputShape.Length >= 4 ? inputShape[2] : 224;
        _modelW = inputShape.Length >= 4 ? inputShape[3] : 224;
        // Replace -1 (dynamic) with default
        if (_modelH <= 0) _modelH = 224;
        if (_modelW <= 0) _modelW = 224;
    }

    /// <summary>
    /// Apply neural style transfer to an RGBA image.
    /// </summary>
    public async Task<StyleResult> TransferAsync(int[] rgbaPixels, int width, int height)
    {
        // Use model's declared input size to match compiled graph shapes
        int modelH = _modelH, modelW = _modelW;

        // GPU preprocessing: RGBA → bilinear resize → NCHW [0, 255]
        // Uses ImagePreprocessKernel.ForwardRaw — all on GPU, no CPU loops
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var inputBuf = _accelerator.Allocate1D<float>(3 * modelH * modelW);
        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        _preprocess.ForwardRaw(rgbaBuf.View, inputBuf.View, width, height, modelW, modelH);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, modelH, modelW });

        // Use RunAsync — includes periodic flush + final SynchronizeAsync
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        var output = outputs[_session.OutputNames[0]];
        int outH = output.Shape.Length >= 4 ? output.Shape[2] : modelH;
        int outW = output.Shape.Length >= 4 ? output.Shape[3] : modelW;

        // GPU postprocessing: NCHW float [0,255] → packed RGBA int on GPU
        _postprocess ??= new Kernels.ImagePostprocessKernel(_accelerator);
        using var rgbaOutBuf = _accelerator.Allocate1D<int>(outH * outW);
        _postprocess.NCHWToRGBA(output.Data, rgbaOutBuf.View, outH, outW);
        await _accelerator.SynchronizeAsync();

        // Single GPU→CPU copy of the final packed RGBA pixels
        var result = await rgbaOutBuf.CopyToHostAsync<int>(0, outH * outW);

        return new StyleResult(result, outW, outH);
    }

    public void Dispose() { }
}
