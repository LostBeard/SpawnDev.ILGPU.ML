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
    //  Super-resolution composite (Y from model + Cb/Cr from source RGBA)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int>? _superResCompositeKernel;

    /// <summary>
    /// Composite a super-resolved Y luminance plane with bilinearly-upsampled Cb/Cr
    /// from the original RGBA source to produce a color upscaled image.
    ///
    /// ESPCN-style super-resolution models operate on the Y channel only. A naive
    /// pipeline writes Y back as grayscale (R=G=B=Y), losing all color and shape-locking
    /// the output to the model's square input/output dims. This kernel:
    ///   - Bilinear-samples Y from the model output at the destination resolution
    ///     (handles the case where srcW * scale != modelOutW)
    ///   - Bilinear-samples Cb / Cr from the original RGBA source at the destination
    ///     resolution
    ///   - Combines BT.601 YCbCr → RGB and packs to RGBA
    ///
    /// One thread per destination pixel. dstSize = dstW * dstH.
    /// </summary>
    public void SuperResCompositeYCbCr(
        ArrayView1D<int, Stride1D.Dense> rgbaSrc,
        ArrayView1D<float, Stride1D.Dense> superResY,
        ArrayView1D<int, Stride1D.Dense> rgbaOut,
        int srcW, int srcH,
        int yW, int yH,
        int dstW, int dstH)
    {
        _superResCompositeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int>(SuperResCompositeImpl);
        _superResCompositeKernel(dstW * dstH, rgbaSrc, superResY, rgbaOut,
            srcW, srcH, yW, yH, dstW, dstH);
    }

    private static void SuperResCompositeImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgbaSrc,
        ArrayView1D<float, Stride1D.Dense> superResY,
        ArrayView1D<int, Stride1D.Dense> rgbaOut,
        int srcW, int srcH, int yW, int yH, int dstW, int dstH)
    {
        int dy = idx / dstW;
        int dx = idx % dstW;

        // Sample Y from the super-resolved Y plane (typically yW×yH ≈ modelW*scale)
        float fyY = ((dy + 0.5f) * yH / dstH) - 0.5f;
        float fxY = ((dx + 0.5f) * yW / dstW) - 0.5f;
        float floorYy = MathF.Floor(fyY); float floorYx = MathF.Floor(fxY);
        int y0y = (int)floorYy; int y1y = y0y + 1;
        int x0y = (int)floorYx; int x1y = x0y + 1;
        float tYy = fyY - floorYy; float tYx = fxY - floorYx;
        if (y0y < 0) y0y = 0; if (y0y >= yH) y0y = yH - 1;
        if (y1y < 0) y1y = 0; if (y1y >= yH) y1y = yH - 1;
        if (x0y < 0) x0y = 0; if (x0y >= yW) x0y = yW - 1;
        if (x1y < 0) x1y = 0; if (x1y >= yW) x1y = yW - 1;
        float y00 = superResY[y0y * yW + x0y];
        float y01 = superResY[y0y * yW + x1y];
        float y10 = superResY[y1y * yW + x0y];
        float y11 = superResY[y1y * yW + x1y];
        float yVal = y00 * (1f - tYy) * (1f - tYx) + y01 * (1f - tYy) * tYx
                   + y10 * tYy * (1f - tYx) + y11 * tYy * tYx;
        // Model Y is in [0,1]; convert to [0,255] to match Cb/Cr scale
        float yScaled = yVal * 255f;

        // Sample Cb and Cr from the source RGBA at the destination position.
        // Two-statement floor: ILGPU optimizer can elide MathF.Floor before an int
        // cast that would otherwise truncate toward zero for negative values.
        float fyS = ((dy + 0.5f) * srcH / dstH) - 0.5f;
        float fxS = ((dx + 0.5f) * srcW / dstW) - 0.5f;
        float floorSy = MathF.Floor(fyS); float floorSx = MathF.Floor(fxS);
        int y0s = (int)floorSy; int y1s = y0s + 1;
        int x0s = (int)floorSx; int x1s = x0s + 1;
        float tSy = fyS - floorSy; float tSx = fxS - floorSx;
        if (y0s < 0) y0s = 0; if (y0s >= srcH) y0s = srcH - 1;
        if (y1s < 0) y1s = 0; if (y1s >= srcH) y1s = srcH - 1;
        if (x0s < 0) x0s = 0; if (x0s >= srcW) x0s = srcW - 1;
        if (x1s < 0) x1s = 0; if (x1s >= srcW) x1s = srcW - 1;

        int p00 = rgbaSrc[y0s * srcW + x0s];
        int p01 = rgbaSrc[y0s * srcW + x1s];
        int p10 = rgbaSrc[y1s * srcW + x0s];
        int p11 = rgbaSrc[y1s * srcW + x1s];

        // BT.601: Cb = -0.168736 R - 0.331264 G + 0.5 B + 128
        //         Cr =  0.5 R       - 0.418688 G - 0.081312 B + 128
        float cb00 = -0.168736f * (p00 & 0xFF) - 0.331264f * ((p00 >> 8) & 0xFF) + 0.5f * ((p00 >> 16) & 0xFF) + 128f;
        float cb01 = -0.168736f * (p01 & 0xFF) - 0.331264f * ((p01 >> 8) & 0xFF) + 0.5f * ((p01 >> 16) & 0xFF) + 128f;
        float cb10 = -0.168736f * (p10 & 0xFF) - 0.331264f * ((p10 >> 8) & 0xFF) + 0.5f * ((p10 >> 16) & 0xFF) + 128f;
        float cb11 = -0.168736f * (p11 & 0xFF) - 0.331264f * ((p11 >> 8) & 0xFF) + 0.5f * ((p11 >> 16) & 0xFF) + 128f;
        float cbVal = cb00 * (1f - tSy) * (1f - tSx) + cb01 * (1f - tSy) * tSx
                    + cb10 * tSy * (1f - tSx) + cb11 * tSy * tSx;

        float cr00 = 0.5f * (p00 & 0xFF) - 0.418688f * ((p00 >> 8) & 0xFF) - 0.081312f * ((p00 >> 16) & 0xFF) + 128f;
        float cr01 = 0.5f * (p01 & 0xFF) - 0.418688f * ((p01 >> 8) & 0xFF) - 0.081312f * ((p01 >> 16) & 0xFF) + 128f;
        float cr10 = 0.5f * (p10 & 0xFF) - 0.418688f * ((p10 >> 8) & 0xFF) - 0.081312f * ((p10 >> 16) & 0xFF) + 128f;
        float cr11 = 0.5f * (p11 & 0xFF) - 0.418688f * ((p11 >> 8) & 0xFF) - 0.081312f * ((p11 >> 16) & 0xFF) + 128f;
        float crVal = cr00 * (1f - tSy) * (1f - tSx) + cr01 * (1f - tSy) * tSx
                    + cr10 * tSy * (1f - tSx) + cr11 * tSy * tSx;

        // BT.601 YCbCr → RGB
        float cbShift = cbVal - 128f;
        float crShift = crVal - 128f;
        float r = yScaled + 1.402f * crShift;
        float g = yScaled - 0.344136f * cbShift - 0.714136f * crShift;
        float b = yScaled + 1.772f * cbShift;

        int ri = (int)(r + 0.5f); if (ri < 0) ri = 0; if (ri > 255) ri = 255;
        int gi = (int)(g + 0.5f); if (gi < 0) gi = 0; if (gi > 255) gi = 255;
        int bi = (int)(b + 0.5f); if (bi < 0) bi = 0; if (bi > 255) bi = 255;

        rgbaOut[idx] = ri | (gi << 8) | (bi << 16) | (0xFF << 24);
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
