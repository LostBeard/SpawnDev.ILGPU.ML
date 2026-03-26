using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from super-resolution — upscaled image as RGBA pixels.
/// </summary>
public record SuperResResult(int[] RgbaPixels, int Width, int Height, int UpscaleFactor);

/// <summary>
/// High-level super-resolution pipeline.
/// ESPCN model: input Y luminance channel [1, 1, H, W] → output [1, 1, H*3, W*3].
///
/// Usage:
///   var pipeline = new SuperResolutionPipeline(session, accelerator);
///   var result = await pipeline.UpscaleAsync(rgbaPixels, width, height);
/// </summary>
public class SuperResolutionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly int _upscaleFactor;

    private readonly int _modelH;
    private readonly int _modelW;
    private Kernels.ImagePreprocessKernel? _preprocess;

    public SuperResolutionPipeline(InferenceSession session, Accelerator accelerator,
        int upscaleFactor = 3)
    {
        _session = session;
        _accelerator = accelerator;
        _upscaleFactor = upscaleFactor;
        // Use model's declared input size (graph compiler uses static shapes)
        var inputShape = session.InputShapes.Values.FirstOrDefault() ?? new[] { 1, 1, 224, 224 };
        _modelH = inputShape.Length >= 4 ? inputShape[2] : (inputShape.Length >= 2 ? inputShape[^2] : 224);
        _modelW = inputShape.Length >= 4 ? inputShape[3] : (inputShape.Length >= 1 ? inputShape[^1] : 224);
        if (_modelH <= 0) _modelH = 224;
        if (_modelW <= 0) _modelW = 224;
    }

    /// <summary>
    /// Upscale an RGBA image using the super-resolution model.
    /// </summary>
    public async Task<SuperResResult> UpscaleAsync(int[] rgbaPixels, int width, int height)
    {
        int modelH = _modelH, modelW = _modelW;

        // GPU preprocessing: RGBA → bilinear resize → Y luminance channel
        // BT.601: Y = 0.299*R + 0.587*G + 0.114*B — all on GPU, no CPU loops
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var inputBuf = _accelerator.Allocate1D<float>(modelH * modelW);
        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        _preprocess.ForwardYChannel(rgbaBuf.View, inputBuf.View, width, height, modelW, modelH);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 1, modelH, modelW });

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        // GPU: Y float → grayscale RGBA — stays on GPU, only final result reads to CPU
        var output = outputs[_session.OutputNames[0]];
        int outH = modelH * _upscaleFactor;
        int outW = modelW * _upscaleFactor;
        int outSize = Math.Min(outH * outW, output.ElementCount);

        using var rgbaOutBuf = _accelerator.Allocate1D<int>(outSize);
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);
        postprocess.GrayscaleToRGBA(output.Data.SubView(0, outSize), rgbaOutBuf.View, outSize);
        await _accelerator.SynchronizeAsync();
        var result = await rgbaOutBuf.CopyToHostAsync<int>(0, outSize);

        return new SuperResResult(result, outW, outH, _upscaleFactor);
    }

    /// <summary>
    /// Upscale an RGBA image and return result as GPU MemoryBuffer2D
    /// for zero-copy presentation via ICanvasRenderer.
    /// Caller owns the returned buffer and must dispose it.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> UpscaleGpuAsync(
        int[] rgbaPixels, int width, int height)
    {
        int modelH = _modelH, modelW = _modelW;

        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var inputBuf = _accelerator.Allocate1D<float>(modelH * modelW);
        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        _preprocess.ForwardYChannel(rgbaBuf.View, inputBuf.View, width, height, modelW, modelH);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 1, modelH, modelW });

        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        var output = outputs[_session.OutputNames[0]];
        int outH = modelH * _upscaleFactor;
        int outW = modelW * _upscaleFactor;
        int outSize = Math.Min(outH * outW, output.ElementCount);

        // GPU: Y → grayscale RGBA, output to 2D buffer for zero-copy rendering
        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(outW, outH));
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);
        postprocess.GrayscaleToRGBA(output.Data.SubView(0, outSize), resultBuf.View.BaseView, outSize);
        await _accelerator.SynchronizeAsync();

        return (resultBuf, outW, outH);
    }

    public void Dispose() { }
}
