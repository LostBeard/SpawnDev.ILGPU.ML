using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Background removal pipeline using RMBG 1.4 (or any salient object segmentation model).
/// Input: RGBA image. Output: foreground mask + composited result with transparent background.
///
/// Usage:
///   var pipeline = new BackgroundRemovalPipeline(session, accelerator);
///   var result = await pipeline.RemoveBackgroundAsync(rgbaPixels, width, height);
///   // result.MaskRGBA is the foreground with transparent background
/// </summary>
public class BackgroundRemovalPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly Kernels.ImagePreprocessKernel _preprocess;
    private readonly int _inputSize;

    public BackgroundRemovalPipeline(InferenceSession session, Accelerator accelerator,
        int inputSize = 1024)
    {
        _session = session;
        _accelerator = accelerator;
        _preprocess = new Kernels.ImagePreprocessKernel(accelerator);
        _inputSize = inputSize;
    }

    /// <summary>
    /// Remove background from an RGBA image.
    /// Returns the original image with background pixels made transparent.
    /// </summary>
    /// <remarks>
    /// ⚠️ IN A BROWSER, prefer <see cref="RemoveBackgroundAsync(TypedArray, int, int, float)"/> or a
    /// GPU-resident view overload — this path pulls the frame onto the managed heap solely to upload it.
    /// </remarks>
    public async Task<BackgroundRemovalResult> RemoveBackgroundAsync(
        int[] rgbaPixels, int width, int height,
        float threshold = 0.5f)
    {
        using var rgbaBuf = RgbaUpload.FromManaged(_accelerator, rgbaPixels, width, height);
        return await RemoveBackgroundAsync(rgbaBuf.View, width, height, threshold).ConfigureAwait(false);
    }

    /// <summary>
    /// Browser path: RGBA still a JS typed array (e.g. <c>ImageData.Data</c>). Uploads via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — pixels never enter the .NET managed heap.
    /// </summary>
    public async Task<BackgroundRemovalResult> RemoveBackgroundAsync(
        TypedArray rgbaPixels, int width, int height,
        float threshold = 0.5f)
    {
        using var rgbaBuf = RgbaUpload.FromTypedArray(_accelerator, rgbaPixels, width, height);
        return await RemoveBackgroundAsync(rgbaBuf.View, width, height, threshold).ConfigureAwait(false);
    }

    /// <summary>GPU-resident packed RGBA — no upload.</summary>
    public Task<BackgroundRemovalResult> RemoveBackgroundAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold = 0.5f)
        => RemoveBackgroundCoreAsync(rgbaPixels, width, height, threshold);

    /// <summary>Same as the view overload; accepts an owned buffer.</summary>
    public Task<BackgroundRemovalResult> RemoveBackgroundAsync(
        MemoryBuffer1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold = 0.5f)
        => RemoveBackgroundCoreAsync(rgbaPixels.View, width, height, threshold);

    private async Task<BackgroundRemovalResult> RemoveBackgroundCoreAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold)
    {
        var sw = Stopwatch.StartNew();
        using var cut = await RemoveBackgroundToDeviceAsync(rgbaPixels, width, height, threshold).ConfigureAwait(false);
        int pixelCount = width * height;
        var resultPixels = await cut.View.SubView(0, pixelCount).CopyToHostAsync().ConfigureAwait(false);
        sw.Stop();
        return new BackgroundRemovalResult
        {
            ResultPixels = resultPixels,
            Width = width,
            Height = height,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    // ---- the mask, applied on the device -------------------------------------------------------------------------------------------------------------
    private static void RangeCheckImpl(Index1D i, ArrayView1D<float, Stride1D.Dense> mask, ArrayView1D<int, Stride1D.Dense> flag)
    {
        float v = mask[i];
        if (v < -0.1f || v > 1.1f) Atomic.Max(ref flag[0], 1);
    }

    /// <summary>
    /// One output pixel: the source colour and the mask, resized to the picture with bilinear taps (after the sigmoid when the model output logits), rounded to an 8-bit alpha.
    /// </summary>
    private static void ApplyMaskImpl(Index1D idx, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<float, Stride1D.Dense> mask, ArrayView1D<int, Stride1D.Dense> outPixels, int w, int maskW, int maskH, int h, int sigmoid)
    {
        int x = idx % w, y = idx / w;
        float srcX = (x + 0.5f) * maskW / w - 0.5f, srcY = (y + 0.5f) * maskH / h - 0.5f;
        int x0 = Math.Max(0, Math.Min(maskW - 1, (int)srcX)), y0 = Math.Max(0, Math.Min(maskH - 1, (int)srcY));
        int x1 = Math.Min(x0 + 1, maskW - 1), y1 = Math.Min(y0 + 1, maskH - 1);
        float fx = srcX - x0, fy = srcY - y0;
        float v00 = mask[y0 * maskW + x0], v10 = mask[y0 * maskW + x1], v01 = mask[y1 * maskW + x0], v11 = mask[y1 * maskW + x1];
        if (sigmoid != 0)
        {
            v00 = 1f / (1f + MathF.Exp(-v00)); v10 = 1f / (1f + MathF.Exp(-v10)); v01 = 1f / (1f + MathF.Exp(-v01)); v11 = 1f / (1f + MathF.Exp(-v11));
        }
        float v = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
        int a = (int)(v * 255f + 0.5f);
        a = Math.Max(0, Math.Min(255, a));
        outPixels[idx] = (rgba[idx] & 0x00FFFFFF) | (a << 24);
    }

    /// <summary>
    /// Puts <paramref name="mask"/> (maskWidth x maskHeight, a probability or a logit: logits are told apart by a value outside [-0.1, 1.1]) into the alpha of <paramref name="rgba"/> (width x height) on the device.
    /// The caller disposes the returned buffer.
    /// </summary>
    public static async Task<MemoryBuffer1D<int, Stride1D.Dense>> ApplyMaskAsync(Accelerator accelerator, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<float, Stride1D.Dense> mask, int maskWidth, int maskHeight, int width, int height)
    {
        var rangeCheck = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(RangeCheckImpl);
        var apply = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int>(ApplyMaskImpl);
        using var flag = accelerator.Allocate1D<int>(1);
        flag.MemSetToZero();
        rangeCheck(maskWidth * maskHeight, mask, flag.View);
        int sigmoid = (await flag.View.CopyToHostAsync().ConfigureAwait(false))[0];     // one scalar: whether the model output logits
        var result = accelerator.Allocate1D<int>(width * height);
        apply(width * height, rgba, mask, result.View, width, maskWidth, maskHeight, height, sigmoid);
        await accelerator.SynchronizeAsync().ConfigureAwait(false);
        return result;
    }

    /// <summary>
    /// The picture with the background made transparent, as a device buffer (no mask or pixel crosses to the host). <paramref name="rgbaPixels"/> is packed RGBA, width x height. The caller disposes the result.
    /// </summary>
    public async Task<MemoryBuffer1D<int, Stride1D.Dense>> RemoveBackgroundToDeviceAsync(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels, int width, int height,
        float threshold = 0.5f)
    {
        // Preprocess: RGBA → NCHW float with RMBG normalization
        // RMBG uses mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0] → (pixel/255 - 0.5)
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaPixels, preprocessed.View, width, height, _inputSize, _inputSize,
            new[] { 0.5f, 0.5f, 0.5f }, new[] { 1.0f, 1.0f, 1.0f });

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        }).ConfigureAwait(false);

        // the mask output is [1, 1, H, W] or [1, H, W]
        var output = outputs[_session.OutputNames[0]];
        int maskSize = output.ElementCount;
        int maskH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int maskW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;
        return await ApplyMaskAsync(_accelerator, rgbaPixels, output.Data.SubView(0, maskSize), maskW, maskH, width, height).ConfigureAwait(false);
    }

    /// <summary>
    /// Give back the GPU memory a run leaves parked for the next one (the session's free activation buckets: a 1024 px
    /// pass keeps gigabytes of feature maps). Weights stay; the next run re-allocates. Call it when background removal is
    /// done for now and another model needs the GPU: the text-to-3D page runs RMBG between two SD Turbo decodes, and with
    /// RMBG's buffers still parked the second decode reached 8.7 GB of VRAM and Chrome 151 dropped the device.
    /// Returns the bytes freed.
    /// </summary>
    public long ReleaseWorkingMemory() => _session.ReleaseWorkingMemory();

    public void Dispose()
    {
        _session?.Dispose();
    }
}

/// <summary>Result from background removal.</summary>
public class BackgroundRemovalResult
{
    /// <summary>RGBA pixels with background made transparent.</summary>
    public int[] ResultPixels { get; init; } = Array.Empty<int>();
    private float[]? _mask;
    /// <summary>Foreground mask [0,1] at original resolution: the alpha of <see cref="ResultPixels"/>, worked out when first asked for (the pipeline itself keeps the mask on the device).</summary>
    public float[] Mask
    {
        get
        {
            if (_mask != null) return _mask;
            var m = new float[ResultPixels.Length];
            for (int i = 0; i < m.Length; i++) m[i] = ((ResultPixels[i] >> 24) & 255) / 255f;
            return _mask = m;
        }
        init => _mask = value;
    }
    public int Width { get; init; }
    public int Height { get; init; }
    public double InferenceTimeMs { get; init; }
}
