using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
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
    public async Task<BackgroundRemovalResult> RemoveBackgroundAsync(
        int[] rgbaPixels, int width, int height,
        float threshold = 0.5f)
    {
        var sw = Stopwatch.StartNew();

        // Preprocess: RGBA → NCHW float with RMBG normalization
        // RMBG uses mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0] → (pixel/255 - 0.5)
        using var rgbaBuf = _accelerator.Allocate1D(rgbaPixels);
        using var preprocessed = _accelerator.Allocate1D<float>(3 * _inputSize * _inputSize);
        _preprocess.Forward(rgbaBuf.View, preprocessed.View, width, height, _inputSize, _inputSize,
            new[] { 0.5f, 0.5f, 0.5f }, new[] { 1.0f, 1.0f, 1.0f });

        var inputTensor = new Tensor(preprocessed.View, new[] { 1, 3, _inputSize, _inputSize });

        // Run inference
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });

        // Read mask output — typically [1, 1, H, W] or [1, H, W]
        var output = outputs[_session.OutputNames[0]];
        int maskSize = output.ElementCount;
        using var readBuf = _accelerator.Allocate1D<float>(maskSize);
        new ElementWiseKernels(_accelerator).Scale(output.Data.SubView(0, maskSize), readBuf.View, maskSize, 1f);
        await _accelerator.SynchronizeAsync();
        var rawMask = await readBuf.CopyToHostAsync<float>(0, maskSize);

        // Determine mask spatial dimensions
        int maskH = output.Shape.Length >= 3 ? output.Shape[^2] : _inputSize;
        int maskW = output.Shape.Length >= 3 ? output.Shape[^1] : _inputSize;

        // Apply sigmoid if values are logits (not already in [0,1])
        bool needsSigmoid = rawMask.Any(v => v < -0.1f || v > 1.1f);
        if (needsSigmoid)
        {
            for (int i = 0; i < rawMask.Length; i++)
                rawMask[i] = 1f / (1f + MathF.Exp(-rawMask[i]));
        }

        // Resize mask to original image dimensions
        var resizedMask = ResizeMask(rawMask, maskW, maskH, width, height);

        // Apply mask: set alpha channel based on mask value
        var resultPixels = new int[width * height];
        for (int i = 0; i < width * height; i++)
        {
            int rgba = rgbaPixels[i];
            int r = rgba & 0xFF;
            int g = (rgba >> 8) & 0xFF;
            int b = (rgba >> 16) & 0xFF;
            int a = (int)(resizedMask[i] * 255f + 0.5f);
            if (a < 0) a = 0; if (a > 255) a = 255;
            resultPixels[i] = r | (g << 8) | (b << 16) | (a << 24);
        }

        sw.Stop();

        return new BackgroundRemovalResult
        {
            ResultPixels = resultPixels,
            Mask = resizedMask,
            Width = width,
            Height = height,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    private static float[] ResizeMask(float[] mask, int srcW, int srcH, int dstW, int dstH)
    {
        if (srcW == dstW && srcH == dstH) return mask;

        var result = new float[dstW * dstH];
        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                float srcX = (x + 0.5f) * srcW / dstW - 0.5f;
                float srcY = (y + 0.5f) * srcH / dstH - 0.5f;
                int x0 = Math.Clamp((int)srcX, 0, srcW - 1);
                int y0 = Math.Clamp((int)srcY, 0, srcH - 1);
                int x1 = Math.Clamp(x0 + 1, 0, srcW - 1);
                int y1 = Math.Clamp(y0 + 1, 0, srcH - 1);
                float fx = srcX - x0;
                float fy = srcY - y0;

                float v00 = mask[y0 * srcW + x0];
                float v10 = mask[y0 * srcW + x1];
                float v01 = mask[y1 * srcW + x0];
                float v11 = mask[y1 * srcW + x1];

                result[y * dstW + x] = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                    + v01 * (1 - fx) * fy + v11 * fx * fy;
            }
        }
        return result;
    }

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
    /// <summary>Foreground mask [0,1] at original resolution.</summary>
    public float[] Mask { get; init; } = Array.Empty<float>();
    public int Width { get; init; }
    public int Height { get; init; }
    public double InferenceTimeMs { get; init; }
}
