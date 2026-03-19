using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernels for color space conversion.
/// All operations are one thread per pixel — embarrassingly parallel.
/// Replaces the CPU-side ColorConversion utility with GPU-native execution.
/// </summary>
public class ColorConversionKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _rgbaToYCbCrKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _yCbCrToRGBAKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _rgbaToGrayscaleKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _rgbaToBGRAKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _flipHorizontalKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>? _flipVerticalKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _depthToColormapKernel;

    public ColorConversionKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  RGBA → YCbCr (for super resolution preprocessing)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Convert packed RGBA int pixels to separate Y, Cb, Cr float channels.
    /// One thread per pixel. Output channels are [0, 255] range floats.
    /// </summary>
    private static void RGBAToYCbCrImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> cb,
        ArrayView1D<float, Stride1D.Dense> cr)
    {
        int pixel = rgba[idx];
        float r = (pixel & 0xFF);
        float g = ((pixel >> 8) & 0xFF);
        float b = ((pixel >> 16) & 0xFF);

        y[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        cb[idx] = 128f - 0.168736f * r - 0.331264f * g + 0.5f * b;
        cr[idx] = 128f + 0.5f * r - 0.418688f * g - 0.081312f * b;
    }

    public void RGBAToYCbCr(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> cb,
        ArrayView1D<float, Stride1D.Dense> cr,
        int pixelCount)
    {
        _rgbaToYCbCrKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(RGBAToYCbCrImpl);
        _rgbaToYCbCrKernel(pixelCount, rgba, y, cb, cr);
    }

    // ──────────────────────────────────────────────
    //  YCbCr → RGBA (for super resolution postprocessing)
    // ──────────────────────────────────────────────

    private static void YCbCrToRGBAImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> cb,
        ArrayView1D<float, Stride1D.Dense> cr,
        ArrayView1D<int, Stride1D.Dense> rgba)
    {
        float yf = y[idx];
        float cbf = cb[idx] - 128f;
        float crf = cr[idx] - 128f;

        int r = (int)(yf + 1.402f * crf + 0.5f);
        int g = (int)(yf - 0.344136f * cbf - 0.714136f * crf + 0.5f);
        int b = (int)(yf + 1.772f * cbf + 0.5f);

        // Clamp to [0, 255]
        r = r < 0 ? 0 : (r > 255 ? 255 : r);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);

        rgba[idx] = r | (g << 8) | (b << 16) | (255 << 24);
    }

    public void YCbCrToRGBA(
        ArrayView1D<float, Stride1D.Dense> y,
        ArrayView1D<float, Stride1D.Dense> cb,
        ArrayView1D<float, Stride1D.Dense> cr,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int pixelCount)
    {
        _yCbCrToRGBAKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(YCbCrToRGBAImpl);
        _yCbCrToRGBAKernel(pixelCount, y, cb, cr, rgba);
    }

    // ──────────────────────────────────────────────
    //  RGBA → Grayscale float [0, 1]
    // ──────────────────────────────────────────────

    private static void RGBAToGrayscaleImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> gray)
    {
        int pixel = rgba[idx];
        float r = (pixel & 0xFF) / 255f;
        float g = ((pixel >> 8) & 0xFF) / 255f;
        float b = ((pixel >> 16) & 0xFF) / 255f;
        gray[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    public void RGBAToGrayscale(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> gray,
        int pixelCount)
    {
        _rgbaToGrayscaleKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(RGBAToGrayscaleImpl);
        _rgbaToGrayscaleKernel(pixelCount, rgba, gray);
    }

    // ──────────────────────────────────────────────
    //  RGBA → BGRA (swap R and B channels)
    // ──────────────────────────────────────────────

    private static void RGBAToBGRAImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output)
    {
        int pixel = input[idx];
        int r = pixel & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = (pixel >> 16) & 0xFF;
        int a = (pixel >> 24) & 0xFF;
        output[idx] = b | (g << 8) | (r << 16) | (a << 24);
    }

    public void RGBAToBGRA(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int pixelCount)
    {
        _rgbaToBGRAKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(RGBAToBGRAImpl);
        _rgbaToBGRAKernel(pixelCount, input, output);
    }

    // ──────────────────────────────────────────────
    //  Horizontal flip (mirror)
    // ──────────────────────────────────────────────

    private static void FlipHorizontalImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        int y = idx / width;
        int x = idx % width;
        output[y * width + (width - 1 - x)] = input[idx];
    }

    public void FlipHorizontal(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        // WORKAROUND: Using separate kernel that takes width/height as scalars
        // because we can't use the 2-param version without hitting the scalar param limit
        _flipHorizontalKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                // Can't capture width in kernel lambda — need params buffer approach
                outp[idx] = inp[idx]; // placeholder
            });
        // TODO: Use params buffer like Conv1DKernel for width/height
        _flipHorizontalKernel(width * height, input, output);
    }

    // ──────────────────────────────────────────────
    //  Depth map → colormap RGBA (for depth visualization)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Apply a colormap to a depth map on GPU.
    /// Input: float depth values. Output: packed RGBA int pixels.
    /// Params: [minDepth, maxDepth, paletteId] where paletteId: 0=plasma, 1=viridis, 2=inferno, 3=grayscale
    /// Uses a piecewise linear interpolation with 9 control points per palette.
    /// </summary>
    private static void DepthToColormapImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        float minD = p[0];
        float range = p[1] - p[0];
        int paletteId = (int)p[2];

        float raw = depth[idx];
        float t = range > 0.000001f ? (raw - minD) / range : 0f;
        t = t < 0f ? 0f : (t > 1f ? 1f : t);

        // Inline piecewise linear colormap (9 control points)
        // Plasma: dark purple → pink → yellow
        float r, g, b;

        if (paletteId == 3) // Grayscale
        {
            r = g = b = t;
        }
        else
        {
            // Plasma approximation via piecewise linear
            if (paletteId == 0) // Plasma
            {
                if (t < 0.25f) { float f = t / 0.25f; r = (13 + (126 - 13) * f) / 255f; g = (8 + (3 - 8) * f) / 255f; b = (135 + (168 - 135) * f) / 255f; }
                else if (t < 0.5f) { float f = (t - 0.25f) / 0.25f; r = (126 + (203 - 126) * f) / 255f; g = (3 + (70 - 3) * f) / 255f; b = (168 + (121 - 168) * f) / 255f; }
                else if (t < 0.75f) { float f = (t - 0.5f) / 0.25f; r = (203 + (248 - 203) * f) / 255f; g = (70 + (149 - 70) * f) / 255f; b = (121 + (64 - 121) * f) / 255f; }
                else { float f = (t - 0.75f) / 0.25f; r = (248 + (240 - 248) * f) / 255f; g = (149 + (249 - 149) * f) / 255f; b = (64 + (33 - 64) * f) / 255f; }
            }
            else if (paletteId == 1) // Viridis
            {
                if (t < 0.25f) { float f = t / 0.25f; r = (68 + (65 - 68) * f) / 255f; g = (1 + (68 - 1) * f) / 255f; b = (84 + (135 - 84) * f) / 255f; }
                else if (t < 0.5f) { float f = (t - 0.25f) / 0.25f; r = (65 + (33 - 65) * f) / 255f; g = (68 + (145 - 68) * f) / 255f; b = (135 + (140 - 135) * f) / 255f; }
                else if (t < 0.75f) { float f = (t - 0.5f) / 0.25f; r = (33 + (109 - 33) * f) / 255f; g = (145 + (205 - 145) * f) / 255f; b = (140 + (89 - 140) * f) / 255f; }
                else { float f = (t - 0.75f) / 0.25f; r = (109 + (253 - 109) * f) / 255f; g = (205 + (231 - 205) * f) / 255f; b = (89 + (37 - 89) * f) / 255f; }
            }
            else // Inferno
            {
                if (t < 0.25f) { float f = t / 0.25f; r = (0 + (85 - 0) * f) / 255f; g = (0 + (15 - 0) * f) / 255f; b = (4 + (109 - 4) * f) / 255f; }
                else if (t < 0.5f) { float f = (t - 0.25f) / 0.25f; r = (85 + (186 - 85) * f) / 255f; g = (15 + (54 - 15) * f) / 255f; b = (109 + (85 - 109) * f) / 255f; }
                else if (t < 0.75f) { float f = (t - 0.5f) / 0.25f; r = (186 + (249 - 186) * f) / 255f; g = (54 + (140 - 54) * f) / 255f; b = (85 + (10 - 85) * f) / 255f; }
                else { float f = (t - 0.75f) / 0.25f; r = (249 + (252 - 249) * f) / 255f; g = (140 + (255 - 140) * f) / 255f; b = (10 + (164 - 10) * f) / 255f; }
            }
        }

        int ri = (int)(r * 255f + 0.5f);
        int gi = (int)(g * 255f + 0.5f);
        int bi = (int)(b * 255f + 0.5f);
        ri = ri < 0 ? 0 : (ri > 255 ? 255 : ri);
        gi = gi < 0 ? 0 : (gi > 255 ? 255 : gi);
        bi = bi < 0 ? 0 : (bi > 255 ? 255 : bi);

        rgba[idx] = ri | (gi << 8) | (bi << 16) | (255 << 24);
    }

    /// <summary>
    /// Apply a colormap to a depth map entirely on GPU.
    /// No CPU readback needed — depth stays on GPU, colormap output stays on GPU.
    /// </summary>
    public void DepthToColormap(
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int pixelCount,
        float minDepth, float maxDepth, int paletteId = 0)
    {
        var paramsData = new float[] { minDepth, maxDepth, paletteId };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _depthToColormapKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(DepthToColormapImpl);
        _depthToColormapKernel(pixelCount, depth, rgba, paramsBuf.View);
    }
}
