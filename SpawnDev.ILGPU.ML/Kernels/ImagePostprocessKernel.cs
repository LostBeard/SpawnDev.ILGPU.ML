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
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _resizeFloatKernel;

    public ImagePostprocessKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ═══════════════════════════════════════════════════════════
    //  Float bilinear resize (single-channel maps: depth, mask, heatmap)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Bilinear-resize a single-channel float map [srcH * srcW] → [dstH * dstW] on GPU.
    /// Used to upscale model outputs (518×518) back to source-image dimensions so
    /// downstream rendering aligns 1:1 with the input image (depth maps, masks, etc).
    /// One thread per destination pixel.
    /// </summary>
    public void ResizeBilinear(
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int srcW, int srcH, int dstW, int dstH)
    {
        _resizeFloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, int, int, int>(ResizeBilinearImpl);
        _resizeFloatKernel(dstW * dstH, src, dst, srcW, srcH, dstW, dstH);
    }

    private static void ResizeBilinearImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int srcW, int srcH, int dstW, int dstH)
    {
        int dy = idx / dstW;
        int dx = idx % dstW;

        float fy = ((dy + 0.5f) * srcH / dstH) - 0.5f;
        float fx = ((dx + 0.5f) * srcW / dstW) - 0.5f;

        // Two-statement floor: prevents ILGPU optimizer from eliding floor() before int cast.
        // (int)x truncates toward zero — wrong for negative values.
        float floorY = MathF.Floor(fy); float floorX = MathF.Floor(fx);
        int y0 = (int)floorY; int y1 = y0 + 1;
        int x0 = (int)floorX; int x1 = x0 + 1;
        float ty = fy - floorY; float tx = fx - floorX;

        if (y0 < 0) y0 = 0; if (y0 >= srcH) y0 = srcH - 1;
        if (y1 < 0) y1 = 0; if (y1 >= srcH) y1 = srcH - 1;
        if (x0 < 0) x0 = 0; if (x0 >= srcW) x0 = srcW - 1;
        if (x1 < 0) x1 = 0; if (x1 >= srcW) x1 = srcW - 1;

        float v00 = src[y0 * srcW + x0];
        float v01 = src[y0 * srcW + x1];
        float v10 = src[y1 * srcW + x0];
        float v11 = src[y1 * srcW + x1];

        dst[idx] = v00 * (1f - ty) * (1f - tx) + v01 * (1f - ty) * tx
                 + v10 * ty * (1f - tx) + v11 * ty * tx;
    }

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
