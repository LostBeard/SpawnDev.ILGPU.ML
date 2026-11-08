using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

public partial class ImagePreprocessKernel
{
    // ── PIL-exact resize (Image.resize(..., BILINEAR) of an RGB image) ──
    //
    // ⭐ WHY. Hugging Face image processors resize with PIL, and PIL's BILINEAR is NOT 4-tap bilinear when it shrinks: it is a
    // triangle filter whose support widens with the scale (an antialiased resize - 7 taps per axis at 1920 -> 640). Models trained
    // and measured through those processors see THAT input. MEASURED on AssistZ (Riker, 2026-11): every DETR detection number was
    // taken through PIL `crop(box).resize((640, 640), BILINEAR)`, from 1920x1080 game frames, so the plain bilinear Forward feeds the
    // model a sharper, aliased picture than the one the numbers describe.
    //
    // Bit-exact with Pillow (libImaging/Resample.c): coefficients are computed on the host in DOUBLE exactly as precompute_coeffs does,
    // normalised, and converted to 22-bit fixed point as normalize_coeffs_8bpc does; the kernel then runs Pillow's two passes in one:
    // for each vertical tap row it computes the horizontal pass value for that row (rounded and clipped to uint8, as Pillow stores its
    // intermediate image) and accumulates it vertically, rounded and clipped again. Same arithmetic, no intermediate buffer.
    //
    // A source rect (srcX, srcY, regionW, regionH) gives `crop(box).resize(...)` semantics - the crop is the whole image, so taps clamp to
    // its edges - which lets a caller tile one uploaded frame without copying it.

    private const int PilPrecisionBits = 32 - 8 - 2;   // Pillow PRECISION_BITS

    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,    // RGBA pixels [srcH * srcW] packed
        ArrayView1D<float, Stride1D.Dense>,  // output NCHW [3, dstH, dstW]
        ArrayView1D<int, Stride1D.Dense>,    // coefficient table, see PilTable
        ArrayView1D<float, Stride1D.Dense>>? // [mean r,g,b, invStd r,g,b]
        _pilKernel;

    private readonly Dictionary<(int, int, int, int, int, int, int), int[]> _pilTables = new();

    // table header: [dstW, dstH, kx, ky, srcStride, regionX0, regionY0, regionW, regionH, xOff, yOff]
    // then per output x at xOff + dx * (kx + 1): [xmin (region coords), kx fixed-point weights], same for y at yOff.
    private const int PilHeader = 11;

    /// <summary>
    /// Pillow's <c>precompute_coeffs</c> + <c>normalize_coeffs_8bpc</c> for the bilinear (triangle) filter: per output index, the first
    /// input index and <paramref name="ksize"/> 22-bit fixed-point weights (zero past the real taps).
    /// </summary>
    internal static (int[] Bounds, int[] Weights, int Ksize) PilCoeffs(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterscale = scale < 1.0 ? 1.0 : scale;
        double support = 1.0 * filterscale;                  // bilinear filter support 1.0
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize];
        var weights = new int[outSize * ksize];
        var k = new double[ksize];
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            double ww = 0.0, ss = 1.0 / filterscale;
            int xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            for (int x = 0; x < ksize; x++) k[x] = 0;
            for (int x = 0; x < xmax; x++)
            {
                double t = Math.Abs((x + xmin - center + 0.5) * ss);
                double w = t < 1.0 ? 1.0 - t : 0.0;
                k[x] = w; ww += w;
            }
            for (int x = 0; x < xmax; x++) if (ww != 0.0) k[x] /= ww;
            bounds[xx] = xmin;
            for (int x = 0; x < ksize; x++)
                weights[xx * ksize + x] = k[x] < 0
                    ? (int)(-0.5 + k[x] * (1 << PilPrecisionBits))
                    : (int)(0.5 + k[x] * (1 << PilPrecisionBits));
        }
        return (bounds, weights, ksize);
    }

    private int[] PilTable(int srcW, int srcX, int srcY, int regionW, int regionH, int dstW, int dstH)
    {
        var key = (srcW, srcX, srcY, regionW, regionH, dstW, dstH);
        lock (_pilTables)
        {
            if (_pilTables.TryGetValue(key, out var cached)) return cached;
            var (bx, wx, kx) = PilCoeffs(regionW, dstW);
            var (by, wy, ky) = PilCoeffs(regionH, dstH);
            int xOff = PilHeader, yOff = xOff + dstW * (kx + 1);
            var t = new int[yOff + dstH * (ky + 1)];
            t[0] = dstW; t[1] = dstH; t[2] = kx; t[3] = ky; t[4] = srcW;
            t[5] = srcX; t[6] = srcY; t[7] = regionW; t[8] = regionH; t[9] = xOff; t[10] = yOff;
            for (int i = 0; i < dstW; i++) { t[xOff + i * (kx + 1)] = bx[i]; Array.Copy(wx, i * kx, t, xOff + i * (kx + 1) + 1, kx); }
            for (int i = 0; i < dstH; i++) { t[yOff + i * (ky + 1)] = by[i]; Array.Copy(wy, i * ky, t, yOff + i * (ky + 1) + 1, ky); }
            if (_pilTables.Count > 32) _pilTables.Clear();   // a caller cycling through many sizes; the device copies stay content-cached
            _pilTables[key] = t;
            return t;
        }
    }

    private static void PilResizeImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> t,
        ArrayView1D<float, Stride1D.Dense> norm)
    {
        int dstW = t[0]; int dstH = t[1]; int kx = t[2]; int ky = t[3]; int stride = t[4];
        int rx0 = t[5]; int ry0 = t[6]; int rw = t[7]; int rh = t[8];
        int dstHW = dstH * dstW;
        int c = idx / dstHW;
        int rem = idx - c * dstHW;
        int dy = rem / dstW;
        int dx = rem - dy * dstW;
        int xb = t[9] + dx * (kx + 1);
        int yb = t[10] + dy * (ky + 1);
        int xmin = t[xb]; int ymin = t[yb];
        int shift = c * 8;
        int half = 1 << (PilPrecisionBits - 1);
        int accV = half;
        for (int j = 0; j < ky; j++)
        {
            int wv = t[yb + 1 + j];
            // Padded taps carry weight 0: clamp their index into the region so the read stays in bounds.
            int sy = ymin + j;
            if (sy > rh - 1) sy = rh - 1;
            int rowBase = (ry0 + sy) * stride + rx0;
            int accH = half;
            for (int i = 0; i < kx; i++)
            {
                int sx = xmin + i;
                if (sx > rw - 1) sx = rw - 1;
                int px = (rgba[rowBase + sx] >> shift) & 0xFF;
                accH += px * t[xb + 1 + i];
            }
            int h = accH >> PilPrecisionBits;           // clip8: Pillow's intermediate image is uint8
            if (h < 0) h = 0;
            if (h > 255) h = 255;
            accV += h * wv;
        }
        int v = accV >> PilPrecisionBits;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        output[idx] = (v / 255f - norm[c]) * norm[3 + c];
    }

    /// <summary>
    /// Pillow-exact <c>crop(box).resize((dstW, dstH), Image.BILINEAR)</c> of the RGB channels, then <c>(x / 255 - mean) / std</c>, written
    /// NCHW: the preprocessing of a Hugging Face image processor with <c>resample=2</c> (DETR, RT-DETR, D-FINE, ...). Stretches; no letterbox.
    /// </summary>
    /// <param name="rgba">Packed RGBA, <paramref name="srcW"/> pixels per row.</param>
    /// <param name="srcX">Left of the source rect (the crop); 0 with <paramref name="regionW"/> = <paramref name="srcW"/> for the whole image.</param>
    public void ForwardPil(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output,
        int srcW, int srcH, int dstW, int dstH,
        float[]? mean = null, float[]? std = null,
        int srcX = 0, int srcY = 0, int regionW = 0, int regionH = 0)
    {
        if (regionW <= 0) regionW = srcW - srcX;
        if (regionH <= 0) regionH = srcH - srcY;
        if (srcX < 0 || srcY < 0 || srcX + regionW > srcW || srcY + regionH > srcH || regionW <= 0 || regionH <= 0)
            throw new ArgumentOutOfRangeException(nameof(srcX),
                $"Source rect ({srcX}, {srcY}, {regionW}x{regionH}) is not inside the {srcW}x{srcH} image.");
        mean ??= new[] { 0f, 0f, 0f };
        std ??= new[] { 1f, 1f, 1f };
        _pilKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(PilResizeImpl);
        var table = IntParams(PilTable(srcW, srcX, srcY, regionW, regionH, dstW, dstH));
        var norm = FloatParams(new[] { mean[0], mean[1], mean[2], 1f / std[0], 1f / std[1], 1f / std[2] });
        _pilKernel(3 * dstH * dstW, rgba, output, table, norm);
    }
}
