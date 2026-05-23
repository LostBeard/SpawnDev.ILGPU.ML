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
    ///
    /// Output dimensions:
    ///   outputWidth = 0 && outputHeight = 0 → match source (width, height) — default,
    ///       preserves aspect ratio so the depth map aligns 1:1 with the input.
    ///   outputWidth > 0 && outputHeight = 0 → use outputWidth, derive height from source aspect.
    ///   outputWidth = 0 && outputHeight > 0 → use outputHeight, derive width from source aspect.
    ///   outputWidth > 0 && outputHeight > 0 → exact size (may not preserve aspect).
    /// Resize is done on the accelerator via bilinear interpolation — no CPU readback of the raw map.
    /// </summary>
    public async Task<DepthResult> EstimateAsync(int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
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

        var output = outputs[_session.OutputNames[0]];
        int rawSize = output.ElementCount;
        int rawH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int rawW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;
        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth CPU] Output: shape=[{string.Join(",", output.Shape)}], elements={rawSize}");

        // Resolve output dimensions (default = source size, preserves source aspect).
        var (outW, outH) = ResolveOutputSize(width, height, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;

        // GPU-side bilinear resize from raw model output (rawW × rawH) → (outW × outH).
        var post = new Kernels.ImagePostprocessKernel(_accelerator);
        using var resized = _accelerator.Allocate1D<float>(outSize);
        post.ResizeBilinear(output.Data.SubView(0, rawSize), resized.View, rawW, rawH, outW, outH);
        await _accelerator.SynchronizeAsync();

        // Read resized depth to CPU for min/max + normalization.
        var rawDepth = await resized.CopyToHostAsync<float>(0, outSize);
        float min = rawDepth.Min();
        float max = rawDepth.Max();
        float range = max - min;
        var normalized = new float[outSize];
        if (range > 1e-6f)
        {
            for (int i = 0; i < outSize; i++)
                normalized[i] = (rawDepth[i] - min) / range;
        }

        return new DepthResult(normalized, outW, outH, min, max);
    }

    /// <summary>
    /// Resolve final output dimensions for a depth result from caller hints.
    ///   (0, 0) → (srcW, srcH) — match source, preserves aspect.
    ///   (w, 0) → (w, w * srcH / srcW) — preserve source aspect, fit width.
    ///   (0, h) → (h * srcW / srcH, h) — preserve source aspect, fit height.
    ///   (w, h) → (w, h) — exact, may distort.
    /// rawW/rawH are used as a fallback when srcW/srcH are non-positive.
    /// </summary>
    private static (int w, int h) ResolveOutputSize(int srcW, int srcH, int rawW, int rawH,
        int outW, int outH)
    {
        if (srcW <= 0 || srcH <= 0) { srcW = rawW; srcH = rawH; }
        if (outW <= 0 && outH <= 0) return (srcW, srcH);
        if (outW > 0 && outH <= 0)
        {
            int h = (int)MathF.Round(outW * (float)srcH / srcW);
            return (outW, Math.Max(1, h));
        }
        if (outH > 0 && outW <= 0)
        {
            int w = (int)MathF.Round(outH * (float)srcW / srcH);
            return (Math.Max(1, w), outH);
        }
        return (outW, outH);
    }

    /// <summary>
    /// Estimate depth and return a plasma colormap as a GPU MemoryBuffer2D for zero-copy
    /// presentation via ICanvasRenderer. The raw depth values stay on the accelerator;
    /// nothing about them leaves the GPU through this contract.
    ///
    /// Output dimensions follow the same convention as <see cref="EstimateAsync"/>:
    ///   (0, 0) → match source (width, height), preserving aspect ratio.
    ///   (w, 0) / (0, h) → fit one axis, derive the other from source aspect.
    ///   (w, h) → exact.
    /// Resize is bilinear, executed on the accelerator before colormap.
    /// Caller owns the returned buffer and must dispose it.
    ///
    /// If you also need the raw depth buffer (to re-apply a different palette later or
    /// run additional postprocessing without re-running inference) use
    /// <see cref="EstimateGpuRawAsync"/> + <see cref="ApplyColormapGpuAsync"/> instead.
    /// </summary>
    public async Task<(MemoryBuffer2D<int, Stride2D.DenseX> Buffer, int Width, int Height)> EstimateGpuAsync(
        int[] rgbaPixels, int width, int height,
        int outputWidth = 0, int outputHeight = 0)
    {
        var (rawDepth, minD, maxD, outW, outH) = await EstimateGpuRawAsync(
            rgbaPixels, width, height, outputWidth, outputHeight);
        try
        {
            var resultBuf = await ApplyColormapGpuAsync(rawDepth.View, outW, outH, minD, maxD,
                Kernels.ImagePostprocessKernel.PalettePlasma);
            return (resultBuf, outW, outH);
        }
        finally
        {
            rawDepth.Dispose();
        }
    }

    /// <summary>
    /// Run depth inference and return the raw normalized-range depth as a GPU buffer
    /// alongside its min/max scalars and dimensions. The buffer stays on the accelerator —
    /// callers can apply <see cref="ApplyColormapGpuAsync"/> as many times as they like
    /// (e.g. when a UI palette toggle changes) without re-running inference.
    ///
    /// Output dimensions follow the same convention as <see cref="EstimateAsync"/>.
    /// Caller owns the returned <see cref="MemoryBuffer1D{T, TStride}"/> and must dispose
    /// it when finished.
    /// </summary>
    public async Task<(MemoryBuffer1D<float, Stride1D.Dense> RawDepth, float MinDepth, float MaxDepth, int Width, int Height)>
        EstimateGpuRawAsync(int[] rgbaPixels, int width, int height,
            int outputWidth = 0, int outputHeight = 0)
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
        int rawSize = output.ElementCount;
        int rawH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int rawW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth] Output: shape=[{string.Join(",", output.Shape)}], elements={rawSize}, dataLength={output.Data.Length}");

        var (outW, outH) = ResolveOutputSize(width, height, rawW, rawH, outputWidth, outputHeight);
        int outSize = outW * outH;

        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);

        // GPU bilinear resize from rawW×rawH → outW×outH. The caller-owned buffer is
        // returned untouched after this; the only readback is a small min/max for the
        // colormap normalization scalar pair (TODO: replace with a GPU reduction so the
        // raw path is entirely host-touch-free).
        int readRawSize = Math.Min(rawSize, (int)output.Data.Length);
        var rawDepth = _accelerator.Allocate1D<float>(outSize);
        postprocess.ResizeBilinear(output.Data.SubView(0, readRawSize), rawDepth.View, rawW, rawH, outW, outH);
        await _accelerator.SynchronizeAsync();

        var resizedHost = await rawDepth.CopyToHostAsync<float>(0, outSize);
        float minD = resizedHost.Min();
        float maxD = resizedHost.Max();

        if (InferenceSession.VerboseLogging) Console.WriteLine($"[Depth] Values: min={minD:F4}, max={maxD:F4}, absMax={resizedHost.Max(v => MathF.Abs(v)):F4}, nonZero={resizedHost.Count(v => v != 0)}/{outSize}");

        return (rawDepth, minD, maxD, outW, outH);
    }

    /// <summary>
    /// Apply a colormap to a raw depth GPU buffer and return a fresh 2D RGBA buffer with
    /// the colored result. Inference is NOT re-run — this is just the postprocess step.
    /// Caller owns the returned buffer and must dispose it.
    ///
    /// Use <see cref="Kernels.ImagePostprocessKernel.PaletteFromName"/> to convert a UI
    /// palette name (plasma / viridis / inferno / grayscale) into the int palette index.
    /// </summary>
    public async Task<MemoryBuffer2D<int, Stride2D.DenseX>> ApplyColormapGpuAsync(
        ArrayView1D<float, Stride1D.Dense> rawDepth, int width, int height,
        float minDepth, float maxDepth, int palette)
    {
        var postprocess = new Kernels.ImagePostprocessKernel(_accelerator);
        var resultBuf = _accelerator.Allocate2DDenseX<int>(new Index2D(width, height));
        postprocess.DepthToColormapPalette(rawDepth, resultBuf.View.BaseView,
            width * height, minDepth, maxDepth, palette);
        await _accelerator.SynchronizeAsync();
        return resultBuf;
    }

    public void Dispose() { }
}
