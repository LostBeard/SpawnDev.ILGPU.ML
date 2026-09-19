using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

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
    private Kernels.ImageTransformKernel? _resize;

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
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="TransferAsync(TypedArray, int, int)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<StyleResult> TransferAsync(int[] rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await TransferAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<StyleResult> TransferAsync(TypedArray rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await TransferAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>
    /// GPU-resident packed RGBA — no upload.
    /// </summary>
    public Task<StyleResult> TransferAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => TransferCoreAsync(rgbaPixels, width, height);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<StyleResult> TransferAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => TransferCoreAsync(rgbaPixels.View, width, height);

    private async Task<StyleResult> TransferCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
    {
        // Use model's declared input size to match compiled graph shapes
        int modelH = _modelH, modelW = _modelW;

        // GPU preprocessing: RGBA → bilinear resize → NCHW [0, 255]
        using var inputBuf = _accelerator.Allocate1D<float>(3 * modelH * modelW);
        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        _preprocess.ForwardRaw(rgbaPixels, inputBuf.View, width, height, modelW, modelH);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, modelH, modelW });

        // Use RunAsync — includes periodic flush + final SynchronizeAsync
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        var output = outputs[_session.OutputNames[0]];
        int outH = output.Shape.Length >= 4 ? output.Shape[2] : modelH;
        int outW = output.Shape.Length >= 4 ? output.Shape[3] : modelW;

        // GPU postprocessing: NCHW float [0,255] → packed RGBA int on GPU
        _postprocess ??= new Kernels.ImagePostprocessKernel(_accelerator);
        using var rgbaOutBuf = _accelerator.Allocate1D<int>(outH * outW);
        _postprocess.NCHWToRGBA(output.Data, rgbaOutBuf.View, outH, outW);

        // Resize styled output back to original dimensions if different
        int finalW = width, finalH = height;
        if (outW != width || outH != height)
        {
            _resize ??= new Kernels.ImageTransformKernel(_accelerator);
            using var resizedBuf = _accelerator.Allocate1D<int>(finalW * finalH);
            _resize.Resize(rgbaOutBuf.View, resizedBuf.View, outW, outH, finalW, finalH);
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            var result = await resizedBuf.CopyToHostAsync<int>(0, finalW * finalH).ConfigureAwait(false);
            return new StyleResult(result, finalW, finalH);
        }

        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        var resultDirect = await rgbaOutBuf.CopyToHostAsync<int>(0, outH * outW).ConfigureAwait(false);
        return new StyleResult(resultDirect, outW, outH);
    }

    /// <summary>
    /// Apply neural style transfer and return result as a GPU MemoryBuffer2D
    /// for zero-copy presentation via ICanvasRenderer.
    /// Caller owns the returned buffer and must dispose it.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="TransferGpuAsync(TypedArray, int, int)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> TransferGpuAsync(
        int[] rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await TransferGpuAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array. Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> TransferGpuAsync(
        TypedArray rgbaPixels, int width, int height)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await TransferGpuAsync(rgbaBuf.View, width, height).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> TransferGpuAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => TransferGpuCoreAsync(rgbaPixels, width, height);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> TransferGpuAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
        => TransferGpuCoreAsync(rgbaPixels.View, width, height);

    private async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> TransferGpuCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height)
    {
        int modelH = _modelH, modelW = _modelW;

        using var inputBuf = _accelerator.Allocate1D<float>(3 * modelH * modelW);
        _preprocess ??= new Kernels.ImagePreprocessKernel(_accelerator);
        _preprocess.ForwardRaw(rgbaPixels, inputBuf.View, width, height, modelW, modelH);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, modelH, modelW });

        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        var output = outputs[_session.OutputNames[0]];
        int outH = output.Shape.Length >= 4 ? output.Shape[2] : modelH;
        int outW = output.Shape.Length >= 4 ? output.Shape[3] : modelW;

        _postprocess ??= new Kernels.ImagePostprocessKernel(_accelerator);

        int finalW = width, finalH = height;
        if (outW != width || outH != height)
        {
            using var rgbaOutBuf = _accelerator.Allocate1D<int>(outH * outW);
            _postprocess.NCHWToRGBA(output.Data, rgbaOutBuf.View, outH, outW);
            _resize ??= new Kernels.ImageTransformKernel(_accelerator);
            // Allocate 2D output (width = X extent, height = Y extent)
            var result2D = _accelerator.Allocate2DDenseX<int>(new Index2D(finalW, finalH));
            _resize.Resize(rgbaOutBuf.View, result2D.View.BaseView, outW, outH, finalW, finalH);
            await _accelerator.SynchronizeAsync().ConfigureAwait(false);
            return (result2D, finalW, finalH);
        }

        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(outW, outH));
        _postprocess.NCHWToRGBA(output.Data, resultBuf.View.BaseView, outH, outW);
        await _accelerator.SynchronizeAsync().ConfigureAwait(false);
        return (resultBuf, outW, outH);
    }

    public void Dispose() { }
}
