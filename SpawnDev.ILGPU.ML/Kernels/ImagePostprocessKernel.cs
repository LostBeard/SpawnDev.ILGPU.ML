using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU image postprocessing — converts model output back to displayable RGBA pixels.
/// All operations stay on GPU. Results can be presented directly via ICanvasRenderer
/// for zero-copy GPU→canvas rendering.
/// </summary>
public class ImagePostprocessKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        int, int>? _nchwToRgbaKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>?
        _grayscaleToRgbaKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        float, float>? _depthToColormapKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float, float>? _normalizeKernel;

    public ImagePostprocessKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ═══════════════════════════════════════════════════════════
    //  NCHW → RGBA (style transfer, classification overlay)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Convert NCHW float tensor [0,255] to packed RGBA int pixels on GPU.
    /// </summary>
    public void NCHWToRGBA(
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int height, int width)
    {
        _nchwToRgbaKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int, int>(NCHWToRGBAImpl);
        _nchwToRgbaKernel(height * width, nchw, rgba, height, width);
    }

    private static void NCHWToRGBAImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int H, int W)
    {
        int hw = H * W;
        float rf = nchw[0 * hw + idx];
        float gf = nchw[1 * hw + idx];
        float bf = nchw[2 * hw + idx];
        int r = (int)(rf + 0.5f); if (r < 0) r = 0; if (r > 255) r = 255;
        int g = (int)(gf + 0.5f); if (g < 0) g = 0; if (g > 255) g = 255;
        int b = (int)(bf + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255;
        rgba[idx] = r | (g << 8) | (b << 16) | (0xFF << 24);
    }

    // ═══════════════════════════════════════════════════════════
    //  Grayscale float [0,1] → RGBA (super resolution Y channel)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Convert float [0,1] grayscale values to packed RGBA int pixels on GPU.
    /// Each value becomes (gray, gray, gray, 255).
    /// </summary>
    public void GrayscaleToRGBA(
        ArrayView1D<float, Stride1D.Dense> grayscale,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int count)
    {
        _grayscaleToRgbaKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(GrayscaleToRGBAImpl);
        _grayscaleToRgbaKernel(count, grayscale, rgba);
    }

    private static void GrayscaleToRGBAImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> grayscale,
        ArrayView1D<int, Stride1D.Dense> rgba)
    {
        float v = grayscale[idx] * 255f + 0.5f;
        int g = (int)v;
        if (g < 0) g = 0;
        if (g > 255) g = 255;
        rgba[idx] = g | (g << 8) | (g << 16) | (0xFF << 24);
    }

    // ═══════════════════════════════════════════════════════════
    //  Depth float → Plasma colormap RGBA (depth estimation)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Convert raw depth values to plasma colormap RGBA on GPU.
    /// Normalizes using provided min/max range. Higher values = closer = warmer colors.
    /// </summary>
    public void DepthToColormap(
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int count, float minDepth, float maxDepth)
    {
        _depthToColormapKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            float, float>(DepthToColormapImpl);
        _depthToColormapKernel(count, depth, rgba, minDepth, maxDepth);
    }

    /// <summary>
    /// Plasma colormap on GPU — 5-segment piecewise linear approximation.
    /// Matches matplotlib's plasma colormap closely. All computed per-thread, no LUT needed.
    /// </summary>
    private static void DepthToColormapImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        float minVal, float maxVal)
    {
        float range = maxVal - minVal;
        float t = range > 1e-6f ? (depth[idx] - minVal) / range : 0f;
        if (t < 0f) t = 0f;
        if (t > 1f) t = 1f;

        // Plasma colormap: piecewise linear through 5 control points
        // t=0.0: (13, 8, 135)    dark purple
        // t=0.25: (126, 3, 168)  magenta
        // t=0.5: (204, 71, 120)  pink-red
        // t=0.75: (248, 149, 64) orange
        // t=1.0: (240, 249, 33)  yellow
        float r, g, b;
        if (t < 0.25f)
        {
            float s = t * 4f;
            r = 13f + s * (126f - 13f);
            g = 8f + s * (3f - 8f);
            b = 135f + s * (168f - 135f);
        }
        else if (t < 0.5f)
        {
            float s = (t - 0.25f) * 4f;
            r = 126f + s * (204f - 126f);
            g = 3f + s * (71f - 3f);
            b = 168f + s * (120f - 168f);
        }
        else if (t < 0.75f)
        {
            float s = (t - 0.5f) * 4f;
            r = 204f + s * (248f - 204f);
            g = 71f + s * (149f - 71f);
            b = 120f + s * (64f - 120f);
        }
        else
        {
            float s = (t - 0.75f) * 4f;
            r = 248f + s * (240f - 248f);
            g = 149f + s * (249f - 149f);
            b = 64f + s * (33f - 64f);
        }

        int ri = (int)(r + 0.5f); if (ri < 0) ri = 0; if (ri > 255) ri = 255;
        int gi = (int)(g + 0.5f); if (gi < 0) gi = 0; if (gi > 255) gi = 255;
        int bi = (int)(b + 0.5f); if (bi < 0) bi = 0; if (bi > 255) bi = 255;

        rgba[idx] = ri | (gi << 8) | (bi << 16) | (0xFF << 24);
    }

    // ═══════════════════════════════════════════════════════════
    //  Normalize float array to [0,1] on GPU
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Normalize float values to [0,1] range on GPU using provided min/max.
    /// </summary>
    public void Normalize(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int count, float minVal, float maxVal)
    {
        _normalizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float, float>(NormalizeImpl);
        _normalizeKernel(count, input, output, minVal, maxVal);
    }

    private static void NormalizeImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        float minVal, float maxVal)
    {
        float range = maxVal - minVal;
        float v = range > 1e-6f ? (input[idx] - minVal) / range : 0f;
        if (v < 0f) v = 0f;
        if (v > 1f) v = 1f;
        output[idx] = v;
    }
}
