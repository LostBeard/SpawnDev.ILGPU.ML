using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU image postprocessing — converts model output back to displayable RGBA pixels.
/// All operations stay on GPU. Results can be presented directly via ICanvasRenderer
/// for zero-copy GPU→canvas rendering.
/// </summary>
public class ImagePostprocessKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        int, int>? _nchwToRgbaKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>?
        _grayscaleToRgbaKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        float, float>? _depthToColormapKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        float, float, int>? _depthToColormapPaletteKernel;
    private Action<Index1D, Tensors.TensorView<float>, Tensors.TensorView<int>,
        float, float, int>? _depthToColormapPaletteTensorViewKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float, float>? _normalizeKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int, int, int>? _resizeFloatKernel;
    private Action<Index1D, Tensors.TensorView<float>, Tensors.TensorView<float>>? _resizeFloatTensorViewKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, int>? _minMaxPartialKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int>? _minMaxFinalKernel;
    private MemoryBuffer1D<float, Stride1D.Dense>? _minMaxPartials;   // [2*P]: mins then maxs
    private MemoryBuffer1D<float, Stride1D.Dense>? _minMaxResult;     // [2]: min, max
    private const int MinMaxP = 1024;

    public ImagePostprocessKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Releases the small internal reduction buffers (only allocated if <see cref="MinMaxAsync"/>
    /// was used). Hold ONE instance per pipeline and dispose it - do not construct-and-drop per frame.</summary>
    public void Dispose()
    {
        _minMaxPartials?.Dispose(); _minMaxPartials = null;
        _minMaxResult?.Dispose(); _minMaxResult = null;
    }

    // ═══════════════════════════════════════════════════════════
    //  GPU min/max reduction (colormap normalization scalars)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Min + max of <paramref name="count"/> floats entirely on the GPU - the readback is 8 BYTES
    /// (two floats) instead of the whole buffer. Two dispatches: P=1024 grid-stride threads write
    /// per-thread partials (one store each per array - WebGL-safe), then one thread folds the
    /// partials. Replaces the full-buffer readback + host LINQ Min()/Max() in the video path (the
    /// depth map is ~1MB/frame at 518x518 - the readback dominated the per-frame postprocess).
    /// NaN handling: NaN comparisons are false, so NaNs are SKIPPED (host LINQ Min/Max would
    /// poison to NaN) - for depth maps NaN-freeness is separately asserted by the finite checks.
    /// </summary>
    public async Task<(float Min, float Max)> MinMaxAsync(ArrayView1D<float, Stride1D.Dense> data, int count)
    {
        const int p = MinMaxP;
        _minMaxPartials ??= _accelerator.Allocate1D<float>(2 * p);
        _minMaxResult ??= _accelerator.Allocate1D<float>(2);
        _minMaxPartialKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(MinMaxPartialImpl);
        _minMaxFinalKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(MinMaxFinalImpl);
        _minMaxPartialKernel(p, data, _minMaxPartials.View, count, p);
        _minMaxFinalKernel(1, _minMaxPartials.View, _minMaxResult.View, p);
        var r = await _minMaxResult.View.SubView(0, 2).CopyToHostAsync();
        return (r[0], r[1]);
    }

    private static void MinMaxPartialImpl(Index1D tid,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> partials,
        int count, int stride)
    {
        float mn = float.MaxValue, mx = float.MinValue;
        for (int i = tid; i < count; i += stride)
        {
            float v = data[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        // INTERLEAVED [2*tid]=min, [2*tid+1]=max - the positional v*K+slot multi-store shape, the
        // ONLY multi-store WebGL Transform Feedback captures (a split [tid] + [P+tid] layout
        // silently dropped the max half on WebGL - caught by MinMax_GpuReduction_MatchesHost).
        partials[tid * 2] = mn;
        partials[tid * 2 + 1] = mx;
    }

    private static void MinMaxFinalImpl(Index1D _,
        ArrayView1D<float, Stride1D.Dense> partials,
        ArrayView1D<float, Stride1D.Dense> result,
        int p)
    {
        float mn = float.MaxValue, mx = float.MinValue;
        for (int t = 0; t < p; t++)
        {
            float a = partials[t * 2];
            if (a < mn) mn = a;
            float b = partials[t * 2 + 1];
            if (b > mx) mx = b;
        }
        result[0] = mn;
        result[1] = mx;
    }

    // ═══════════════════════════════════════════════════════════
    //  Float bilinear resize (single-channel maps: depth, mask, heatmap)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Bilinear-resize a single-channel float map on GPU. Tensor shapes are
    /// row-major <c>[H, W]</c> (D0 = height, D1 = width — PyTorch / numpy convention).
    /// One thread per destination pixel.
    ///
    /// <para>
    /// Phase 2 of the Tensor refactor: this overload takes <see cref="Tensors.TensorView{T}"/>
    /// directly instead of unpacking width / height to scalar kernel parameters. The
    /// legacy <c>(ArrayView, srcW, srcH, dstW, dstH)</c> overload is kept for callers
    /// that haven't migrated yet.
    /// </para>
    /// </summary>
    public void ResizeBilinear(Tensors.TensorView<float> src, Tensors.TensorView<float> dst)
    {
        _resizeFloatTensorViewKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            Tensors.TensorView<float>, Tensors.TensorView<float>>(ResizeBilinearTensorViewImpl);
        _resizeFloatTensorViewKernel(dst.ElementCount, src, dst);
    }

    /// <summary>
    /// Bilinear-resize a single-channel float map <c>[srcH * srcW] → [dstH * dstW]</c>
    /// on GPU. Legacy overload — prefer <see cref="ResizeBilinear(Tensors.TensorView{float}, Tensors.TensorView{float})"/>
    /// for new code.
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

    /// <summary>
    /// Phase 2 implementation. Reads <c>src.D0 = srcH, src.D1 = srcW</c> and
    /// <c>dst.D0 = dstH, dst.D1 = dstW</c> from the TensorView dimensions — no
    /// scalar shape parameters needed.
    /// </summary>
    private static void ResizeBilinearTensorViewImpl(Index1D idx,
        Tensors.TensorView<float> src,
        Tensors.TensorView<float> dst)
    {
        int dstW = dst.D1;
        int dy = idx / dstW;
        int dx = idx % dstW;

        float fy = ((dy + 0.5f) * src.D0 / dst.D0) - 0.5f;
        float fx = ((dx + 0.5f) * src.D1 / dst.D1) - 0.5f;

        // Two-statement floor: prevents ILGPU optimizer from eliding floor() before int
        // cast. (int)x truncates toward zero — wrong for negative values.
        float floorY = MathF.Floor(fy); float floorX = MathF.Floor(fx);
        int y0 = (int)floorY; int y1 = y0 + 1;
        int x0 = (int)floorX; int x1 = x0 + 1;
        float ty = fy - floorY; float tx = fx - floorX;

        int srcH = src.D0; int srcW = src.D1;
        if (y0 < 0) y0 = 0; if (y0 >= srcH) y0 = srcH - 1;
        if (y1 < 0) y1 = 0; if (y1 >= srcH) y1 = srcH - 1;
        if (x0 < 0) x0 = 0; if (x0 >= srcW) x0 = srcW - 1;
        if (x1 < 0) x1 = 0; if (x1 >= srcW) x1 = srcW - 1;

        float v00 = src.Get2D(y0, x0);
        float v01 = src.Get2D(y0, x1);
        float v10 = src.Get2D(y1, x0);
        float v11 = src.Get2D(y1, x1);

        dst.Set2D(dy, dx,
            v00 * (1f - ty) * (1f - tx) + v01 * (1f - ty) * tx
          + v10 * ty * (1f - tx) + v11 * ty * tx);
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

    // ─── palette indices (must match DepthColorMaps palette strings) ───
    /// <summary>Plasma palette index for <see cref="DepthToColormapPalette"/>.</summary>
    public const int PalettePlasma = 0;
    /// <summary>Viridis palette index for <see cref="DepthToColormapPalette"/>.</summary>
    public const int PaletteViridis = 1;
    /// <summary>Inferno palette index for <see cref="DepthToColormapPalette"/>.</summary>
    public const int PaletteInferno = 2;
    /// <summary>Grayscale palette index for <see cref="DepthToColormapPalette"/>.</summary>
    public const int PaletteGrayscale = 3;

    /// <summary>Map a palette name to the int parameter accepted by <see cref="DepthToColormapPalette"/>.</summary>
    public static int PaletteFromName(string name) => (name ?? "plasma").ToLowerInvariant() switch
    {
        "viridis" => PaletteViridis,
        "inferno" => PaletteInferno,
        "grayscale" => PaletteGrayscale,
        _ => PalettePlasma,
    };

    /// <summary>
    /// GPU colormap with selectable palette — TensorView overload. Both tensors are
    /// row-major <c>[H, W]</c>; the kernel reads count from <c>depth.ElementCount</c>.
    /// Same palette implementation as the legacy <see cref="DepthToColormapPalette(ArrayView1D{float, Stride1D.Dense}, ArrayView1D{int, Stride1D.Dense}, int, float, float, int)"/>
    /// overload — both call into the same piecewise-linear branches.
    /// </summary>
    public void DepthToColormapPalette(
        Tensors.TensorView<float> depth,
        Tensors.TensorView<int> rgba,
        float minDepth, float maxDepth, int palette)
    {
        _depthToColormapPaletteTensorViewKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            Tensors.TensorView<float>, Tensors.TensorView<int>,
            float, float, int>(DepthToColormapPaletteTensorViewImpl);
        _depthToColormapPaletteTensorViewKernel(depth.ElementCount, depth, rgba, minDepth, maxDepth, palette);
    }

    private static void DepthToColormapPaletteTensorViewImpl(Index1D idx,
        Tensors.TensorView<float> depth,
        Tensors.TensorView<int> rgba,
        float minVal, float maxVal, int palette)
    {
        // Shared implementation: convert flat idx to ArrayView access on both tensors.
        // Both are stored contiguous row-major so Data[idx] is correct regardless of
        // whether the host code described them as [H,W] or [H*W,1].
        DepthToColormapPaletteImpl(idx, depth.Data, rgba.Data, minVal, maxVal, palette);
    }

    /// <summary>
    /// GPU colormap with selectable palette — legacy raw-ArrayView overload.
    /// Prefer the <see cref="DepthToColormapPalette(Tensors.TensorView{float}, Tensors.TensorView{int}, float, float, int)"/>
    /// TensorView overload for new code.
    /// </summary>
    public void DepthToColormapPalette(
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int count, float minDepth, float maxDepth, int palette)
    {
        _depthToColormapPaletteKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            float, float, int>(DepthToColormapPaletteImpl);
        _depthToColormapPaletteKernel(count, depth, rgba, minDepth, maxDepth, palette);
    }

    /// <summary>
    /// Piecewise-linear colormap kernel sharing the 9-point control sets defined in
    /// <see cref="Preprocessing.DepthColorMaps"/>. Branches on palette; each branch
    /// is a sequence of 8 ramps between adjacent control points (matplotlib-derived
    /// plasma / viridis / inferno) or a single grayscale ramp.
    /// </summary>
    private static void DepthToColormapPaletteImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        float minVal, float maxVal, int palette)
    {
        float range = maxVal - minVal;
        float t = range > 1e-6f ? (depth[idx] - minVal) / range : 0f;
        if (t < 0f) t = 0f;
        if (t > 1f) t = 1f;

        float r = 0f, g = 0f, b = 0f;

        // Grayscale: linear ramp 0..255 across all channels.
        if (palette == PaletteGrayscale)
        {
            r = t * 255f; g = r; b = r;
        }
        else if (palette == PaletteViridis)
        {
            // Control points from DepthColorMaps.GenerateViridis (t at 0.13/0.25/0.38/...).
            if (t < 0.13f) { float s = t / 0.13f;
                r = 68f + s * (72f - 68f); g = 1f + s * (36f - 1f); b = 84f + s * (117f - 84f); }
            else if (t < 0.25f) { float s = (t - 0.13f) / 0.12f;
                r = 72f + s * (65f - 72f); g = 36f + s * (68f - 36f); b = 117f + s * (135f - 117f); }
            else if (t < 0.38f) { float s = (t - 0.25f) / 0.13f;
                r = 65f + s * (53f - 65f); g = 68f + s * (95f - 68f); b = 135f + s * (141f - 135f); }
            else if (t < 0.50f) { float s = (t - 0.38f) / 0.12f;
                r = 53f + s * (33f - 53f); g = 95f + s * (145f - 95f); b = 141f + s * (140f - 141f); }
            else if (t < 0.63f) { float s = (t - 0.50f) / 0.13f;
                r = 33f + s * (53f - 33f); g = 145f + s * (183f - 145f); b = 140f + s * (121f - 140f); }
            else if (t < 0.75f) { float s = (t - 0.63f) / 0.12f;
                r = 53f + s * (109f - 53f); g = 183f + s * (205f - 183f); b = 121f + s * (89f - 121f); }
            else if (t < 0.88f) { float s = (t - 0.75f) / 0.13f;
                r = 109f + s * (180f - 109f); g = 205f + s * (222f - 205f); b = 89f + s * (44f - 89f); }
            else { float s = (t - 0.88f) / 0.12f;
                r = 180f + s * (253f - 180f); g = 222f + s * (231f - 222f); b = 44f + s * (37f - 44f); }
        }
        else if (palette == PaletteInferno)
        {
            // Control points from DepthColorMaps.GenerateInferno.
            if (t < 0.13f) { float s = t / 0.13f;
                r = 0f + s * (31f - 0f); g = 0f + s * (12f - 0f); b = 4f + s * (72f - 4f); }
            else if (t < 0.25f) { float s = (t - 0.13f) / 0.12f;
                r = 31f + s * (85f - 31f); g = 12f + s * (15f - 12f); b = 72f + s * (109f - 72f); }
            else if (t < 0.38f) { float s = (t - 0.25f) / 0.13f;
                r = 85f + s * (136f - 85f); g = 15f + s * (34f - 15f); b = 109f + s * (106f - 109f); }
            else if (t < 0.50f) { float s = (t - 0.38f) / 0.12f;
                r = 136f + s * (186f - 136f); g = 34f + s * (54f - 34f); b = 106f + s * (85f - 106f); }
            else if (t < 0.63f) { float s = (t - 0.50f) / 0.13f;
                r = 186f + s * (227f - 186f); g = 54f + s * (89f - 54f); b = 85f + s * (51f - 85f); }
            else if (t < 0.75f) { float s = (t - 0.63f) / 0.12f;
                r = 227f + s * (249f - 227f); g = 89f + s * (140f - 89f); b = 51f + s * (10f - 51f); }
            else if (t < 0.88f) { float s = (t - 0.75f) / 0.13f;
                r = 249f + s * (249f - 249f); g = 140f + s * (201f - 140f); b = 10f + s * (50f - 10f); }
            else { float s = (t - 0.88f) / 0.12f;
                r = 249f + s * (252f - 249f); g = 201f + s * (255f - 201f); b = 50f + s * (164f - 50f); }
        }
        else
        {
            // Plasma (default) — control points from DepthColorMaps.GeneratePlasma.
            if (t < 0.13f) { float s = t / 0.13f;
                r = 13f + s * (75f - 13f); g = 8f + s * (3f - 8f); b = 135f + s * (161f - 135f); }
            else if (t < 0.25f) { float s = (t - 0.13f) / 0.12f;
                r = 75f + s * (126f - 75f); g = 3f + s * (3f - 3f); b = 161f + s * (168f - 161f); }
            else if (t < 0.38f) { float s = (t - 0.25f) / 0.13f;
                r = 126f + s * (168f - 126f); g = 3f + s * (34f - 3f); b = 168f + s * (150f - 168f); }
            else if (t < 0.50f) { float s = (t - 0.38f) / 0.12f;
                r = 168f + s * (203f - 168f); g = 34f + s * (70f - 34f); b = 150f + s * (121f - 150f); }
            else if (t < 0.63f) { float s = (t - 0.50f) / 0.13f;
                r = 203f + s * (229f - 203f); g = 70f + s * (107f - 70f); b = 121f + s * (93f - 121f); }
            else if (t < 0.75f) { float s = (t - 0.63f) / 0.12f;
                r = 229f + s * (248f - 229f); g = 107f + s * (149f - 107f); b = 93f + s * (64f - 93f); }
            else if (t < 0.88f) { float s = (t - 0.75f) / 0.13f;
                r = 248f + s * (253f - 248f); g = 149f + s * (195f - 149f); b = 64f + s * (40f - 64f); }
            else { float s = (t - 0.88f) / 0.12f;
                r = 253f + s * (240f - 253f); g = 195f + s * (249f - 195f); b = 40f + s * (33f - 40f); }
        }

        int ri = (int)(r + 0.5f); if (ri < 0) ri = 0; if (ri > 255) ri = 255;
        int gi = (int)(g + 0.5f); if (gi < 0) gi = 0; if (gi > 255) gi = 255;
        int bi = (int)(b + 0.5f); if (bi < 0) bi = 0; if (bi > 255) bi = 255;

        rgba[idx] = ri | (gi << 8) | (bi << 16) | (0xFF << 24);
    }

    // ═══════════════════════════════════════════════════════════
    //  Tile accumulation (for tile-based super-resolution)
    // ═══════════════════════════════════════════════════════════

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int>? _accumYTileKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>?
        _normYAccKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>>? _clearFloatKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>>? _clearIntKernel;

    /// <summary>
    /// Accumulate a super-resolved Y tile into a destination Y plane + per-pixel count
    /// buffer. One thread per source pixel (srcW * srcH). The thread maps tile-local
    /// (tx, ty) to destination (dstOffsetX + tx, dstOffsetY + ty); if that lands inside
    /// the destination it adds the tile value to <c>dstY</c> and increments <c>dstCount</c>.
    ///
    /// Tile inferences run sequentially (with Synchronize between), so within a single
    /// kernel invocation each thread writes to a unique destination pixel — no atomics
    /// required (which keeps this WebGL-compatible).
    /// </summary>
    public void AccumulateYTile(
        ArrayView1D<float, Stride1D.Dense> srcY,
        ArrayView1D<float, Stride1D.Dense> dstY,
        ArrayView1D<int, Stride1D.Dense> dstCount,
        int srcW, int srcH, int dstOffsetX, int dstOffsetY, int dstW, int dstH)
    {
        _accumYTileKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int>(AccumulateYTileImpl);
        _accumYTileKernel(srcW * srcH, srcY, dstY, dstCount,
            srcW, srcH, dstOffsetX, dstOffsetY, dstW, dstH);
    }

    private static void AccumulateYTileImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> srcY,
        ArrayView1D<float, Stride1D.Dense> dstY,
        ArrayView1D<int, Stride1D.Dense> dstCount,
        int srcW, int srcH, int dstOffsetX, int dstOffsetY, int dstW, int dstH)
    {
        int ty = idx / srcW;
        int tx = idx % srcW;
        int dx = dstOffsetX + tx;
        int dy = dstOffsetY + ty;
        if (dx < 0 || dx >= dstW || dy < 0 || dy >= dstH) return;
        int dstIdx = dy * dstW + dx;
        dstY[dstIdx] = dstY[dstIdx] + srcY[idx];
        dstCount[dstIdx] = dstCount[dstIdx] + 1;
    }

    /// <summary>
    /// Final pass for tile accumulation: divide each destination Y pixel by the number
    /// of tiles that contributed to it. Pixels with count = 0 are left at 0 (should not
    /// happen if tiles cover the destination).
    /// </summary>
    public void NormalizeYAccumulator(
        ArrayView1D<float, Stride1D.Dense> dstY,
        ArrayView1D<int, Stride1D.Dense> dstCount)
    {
        _normYAccKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(NormalizeYAccImpl);
        _normYAccKernel((int)dstY.Length, dstY, dstCount);
    }

    private static void NormalizeYAccImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dstY,
        ArrayView1D<int, Stride1D.Dense> dstCount)
    {
        int n = dstCount[idx];
        if (n > 0) dstY[idx] = dstY[idx] / n;
    }

    /// <summary>GPU-side clear of a float buffer to 0.</summary>
    public void ClearFloat(ArrayView1D<float, Stride1D.Dense> buf)
    {
        _clearFloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>>((Index1D i, ArrayView1D<float, Stride1D.Dense> b) => b[i] = 0f);
        _clearFloatKernel((int)buf.Length, buf);
    }

    /// <summary>GPU-side clear of an int buffer to 0.</summary>
    public void ClearInt(ArrayView1D<int, Stride1D.Dense> buf)
    {
        _clearIntKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>>((Index1D i, ArrayView1D<int, Stride1D.Dense> b) => b[i] = 0);
        _clearIntKernel((int)buf.Length, buf);
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
