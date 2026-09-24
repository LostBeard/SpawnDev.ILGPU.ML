using ILGPU;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// Depth Anything 3's reference preprocessing, on the accelerator.
///
/// ByteDance-Seed/Depth-Anything-3 <c>utils/io/input_processor.py</c> (<c>upper_bound_resize</c>):
///   1. scale so the LONG side is <c>process_res</c> (aspect kept, NO padding),
///      cv2 INTER_AREA when shrinking, INTER_CUBIC when growing, rounded to uint8;
///   2. resize each side to the NEAREST multiple of the ViT patch (14), same filter rule, uint8;
///   3. /255, ImageNet mean/std, CHW.
///
/// Why it exists: MEASURED 2026-09-23 against COLMAP ground truth (tools/dav3/dav3_quality.py), the
/// letterbox this class also offers costs DAv3 real accuracy - Truck joint depth AbsRel 0.150 vs
/// 0.111, camera-centre error 9.6% vs 3.1%, focal error 44% vs 19% - because the model reads the
/// replicated-border padding as picture. The resize filter alone was worth little; the padding was
/// the cost. This path feeds exactly what the model was built for.
///
/// CUBIC is a port of OpenCV's own uint8 arithmetic (interpolateCubic, 11-bit fixed point, integer
/// accumulation); AREA is the exact rational average OpenCV approximates in float. Against OpenCV (its
/// own algorithm - NOT Intel IPP, whose cubic differs from OpenCV's on ~1.3% of pixels) the tensor is
/// within one uint8 LSB on a few hundredths of a percent of values, identically on every backend.
/// DA3_NativeAspect_Preprocess_MatchesReference gates it; tools/dav3/check_cv2_port.py is the oracle.
/// Everything stays on the device: packed RGBA view in, NCHW float view out, the stage-1 image in a
/// caller-owned device scratch buffer.
/// </summary>
public partial class ImagePreprocessKernel
{
    /// <summary>Resample filter, chosen per stage exactly as the reference does.</summary>
    private const int ModeCopy = 0, ModeArea = 1, ModeCubic = 2;

    /// <summary>
    /// The two sizes the reference resizes through for a <paramref name="srcW"/> x <paramref name="srcH"/>
    /// image: (<c>W1</c>,<c>H1</c>) after fitting the long side to <paramref name="longSide"/>, then
    /// (<c>W</c>,<c>H</c>) = each side rounded to the nearest multiple of <paramref name="patch"/> - the
    /// model input. Python's <c>round</c> is round-half-even, as is <see cref="Math.Round(double)"/>.
    /// One deviation: a side that would round to 0 patches is held at one patch (the reference emits a
    /// 1-pixel side there, which no ViT accepts).
    /// </summary>
    public static (int W1, int H1, int W, int H) NativeAspectSizes(int srcW, int srcH, int longSide, int patch = 14)
    {
        if (srcW <= 0 || srcH <= 0) throw new ArgumentOutOfRangeException(nameof(srcW), "image size must be positive");
        // The exact area average carries 2 * srcW * srcH in an int (AreaRound).
        if ((long)srcW * srcH >= 1L << 30) throw new ArgumentOutOfRangeException(nameof(srcW), $"{srcW}x{srcH} is over 2^30 pixels");
        if (longSide < patch) throw new ArgumentOutOfRangeException(nameof(longSide), $"long side {longSide} is below one {patch}px patch");
        int w1 = srcW, h1 = srcH;
        int longest = Math.Max(srcW, srcH);
        if (longest != longSide)
        {
            double s = longSide / (double)longest;
            w1 = Math.Max(1, (int)Math.Round(srcW * s));
            h1 = Math.Max(1, (int)Math.Round(srcH * s));
        }
        return (w1, h1, Nearest(w1, patch), Nearest(h1, patch));

        static int Nearest(int x, int p)
        {
            int down = x / p * p, up = down + p;
            return Math.Max(p, Math.Abs(up - x) <= Math.Abs(x - down) ? up : down);
        }
    }

    /// <summary>Device scratch (packed RGBA ints) <see cref="ForwardNativeAspect"/> needs for this image; 0 = none.</summary>
    public static int NativeAspectScratchLength(int srcW, int srcH, int longSide, int patch = 14)
    {
        var (w1, h1, _, _) = NativeAspectSizes(srcW, srcH, longSide, patch);
        return Math.Max(srcW, srcH) == longSide ? 0 : w1 * h1;
    }

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>>? _resampleRgbaKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _resampleNchwKernel;
    // One params buffer PER STAGE: both are written before either stage is dispatched, so the pair
    // costs one pending-command flush (WebGPU CopyFromCPU flushes before its writeBuffer), not two.
    private MemoryBuffer1D<int, Stride1D.Dense>? _stage1Params;
    private MemoryBuffer1D<float, Stride1D.Dense>? _stage2Params;

    /// <summary>
    /// DA3 reference preprocessing of one packed-RGBA image into <paramref name="output"/>
    /// (<c>3 * H * W</c> floats, sizes from <see cref="NativeAspectSizes"/>). <paramref name="scratch"/>
    /// must hold <see cref="NativeAspectScratchLength"/> ints and stay alive until the dispatches have
    /// run (the caller awaits the forward anyway). Two dispatches at most, zero host copies.
    /// </summary>
    public (int W, int H) ForwardNativeAspect(
        ArrayView1D<int, Stride1D.Dense> rgba, int srcW, int srcH,
        ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<int, Stride1D.Dense> scratch,
        int longSide, int patch = 14, float[]? mean = null, float[]? std = null)
    {
        mean ??= new[] { 0.485f, 0.456f, 0.406f };
        std ??= new[] { 0.229f, 0.224f, 0.225f };
        var (w1, h1, w, h) = NativeAspectSizes(srcW, srcH, longSide, patch);
        if (output.Length < 3L * w * h)
            throw new ArgumentException($"output holds {output.Length} floats; {w}x{h}x3 = {3L * w * h} needed", nameof(output));

        bool stage1 = Math.Max(srcW, srcH) != longSide;
        if (stage1 && scratch.Length < (long)w1 * h1)
            throw new ArgumentException($"scratch holds {scratch.Length} ints; {w1}x{h1} = {(long)w1 * h1} needed", nameof(scratch));
        // Stage 1: uniform scale, so it either shrinks or grows both axes.
        int mode1 = w1 < srcW ? ModeArea : ModeCubic;
        // Stage 2: cubic if EITHER side grows (the reference tests nw > w or nh > h), else area.
        int s2W = stage1 ? w1 : srcW, s2H = stage1 ? h1 : srcH;
        int mode2 = (w == s2W && h == s2H) ? ModeCopy : (w > s2W || h > s2H) ? ModeCubic : ModeArea;

        _resampleRgbaKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ResampleRgbaImpl);
        _resampleNchwKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(ResampleNchwImpl);
        _stage1Params ??= _accelerator.Allocate1D<int>(5);
        _stage2Params ??= _accelerator.Allocate1D<float>(11);

        if (stage1) _stage1Params.CopyFromCPU(new[] { srcW, srcH, w1, h1, mode1 });
        _stage2Params.CopyFromCPU(new float[] {
            s2W, s2H, w, h, mode2,
            mean[0], mean[1], mean[2], 1f / std[0], 1f / std[1], 1f / std[2] });

        if (stage1) _resampleRgbaKernel(w1 * h1, rgba, scratch, _stage1Params.View);
        _resampleNchwKernel(3 * w * h, stage1 ? scratch : rgba, output, _stage2Params.View);
        return (w, h);
    }

    /// <summary>Stage 1: packed RGBA -> packed RGBA, uint8-rounded like cv2's uint8 output.</summary>
    private static void ResampleRgbaImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> src, ArrayView1D<int, Stride1D.Dense> dst,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int dstW = p[2];
        dst[idx] = SampleRgb(src, p[0], p[1], dstW, p[3], p[4], idx % dstW, idx / dstW) | unchecked((int)0xFF000000);
    }

    /// <summary>
    /// Stage 2: packed RGBA -> normalised NCHW float, ONE THREAD PER OUTPUT ELEMENT. Not one thread
    /// per pixel writing its three planes: WebGL's transform feedback keeps exactly one output record
    /// per thread and silently drops/relocates the rest (SpawnDev.ILGPU WebGL/CLAUDE.md - MEASURED here
    /// 2026-09-23: 99% of values wrong on WebGL, every other backend right). Each thread resamples the
    /// pixel and keeps its channel; the taps are the same memory, so the redundancy is cache hits.
    /// </summary>
    private static void ResampleNchwImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> src, ArrayView1D<float, Stride1D.Dense> dst,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        int srcW = (int)p[0], srcH = (int)p[1], dstW = (int)p[2], dstH = (int)p[3], mode = (int)p[4];
        int plane = dstW * dstH;
        int c = idx / plane, pix = idx - c * plane;
        int rgb = SampleRgb(src, srcW, srcH, dstW, dstH, mode, pix % dstW, pix / dstW);
        // T.ToTensor (/255) then T.Normalize.
        dst[idx] = (((rgb >> (8 * c)) & 0xFF) / 255f - p[5 + c]) * p[8 + c];
    }

    /// <summary>One output pixel of a cv2 resize, as packed 0x00BBGGRR.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SampleRgb(ArrayView1D<int, Stride1D.Dense> src, int srcW, int srcH,
        int dstW, int dstH, int mode, int dx, int dy)
    {
        if (mode == ModeCopy) return src[dy * srcW + dx] & 0xFFFFFF;

        if (mode == ModeArea)
        {
            // Area average in EXACT integer arithmetic. Measure positions in units of 1/(dstW) source
            // pixel horizontally: source pixel i spans [i*dstW, (i+1)*dstW), output cell dx spans
            // [dx*srcW, (dx+1)*srcW), and every overlap is an integer. Likewise vertically. The average
            // is num / den with num = sum(pixel * ox * oy), den = srcW * srcH, rounded half-to-even
            // (cv2's cvRound).
            // Why not cv2's own float weights: they round per backend - WGSL and OpenCL float division
            // is not correctly rounded, which put 0.2% of values a LSB off on WebGPU/OpenCL vs 0.005% on
            // CUDA (MEASURED 2026-09-23). Exact arithmetic is the same on every backend and agrees with
            // cv2 on all but ~0.0005% of values (cv2's own float rounding).
            // ALL 32-BIT: num reaches 255 * srcW * srcH (3.3e9 at 12 MP), so it is carried as
            // srcW * A + B with each row's partial split by srcW; see AreaRound. A 64-bit version of this
            // put stray pixels up to 238 LSB off on WebGPU and WebGL only (i64 emulation, 2026-09-23).
            int ax = dx * srcW, bx = ax + srcW, ay = dy * srcH, by = ay + srcH;
            int x0 = ax / dstW, x1 = (bx - 1) / dstW, y0 = ay / dstH, y1 = (by - 1) / dstH;
            int rA = 0, rB = 0, gA = 0, gB = 0, bA = 0, bB = 0;
            for (int sy = y0; sy <= y1; sy++)
            {
                int oy = Math.Min(by, (sy + 1) * dstH) - Math.Max(ay, sy * dstH);
                int rr = 0, gg = 0, bb = 0;   // <= 255 * srcW
                for (int sx = x0; sx <= x1; sx++)
                {
                    int ox = Math.Min(bx, (sx + 1) * dstW) - Math.Max(ax, sx * dstW);
                    int px = src[sy * srcW + sx];
                    rr += (px & 0xFF) * ox;
                    gg += ((px >> 8) & 0xFF) * ox;
                    bb += ((px >> 16) & 0xFF) * ox;
                }
                // row partial = srcW * q + r; sum(oy * q) <= 255 * srcH, sum(oy * r) < srcW * srcH.
                int rq = rr / srcW, gq = gg / srcW, bq = bb / srcW;
                rA += oy * rq; rB += oy * (rr - rq * srcW);
                gA += oy * gq; gB += oy * (gg - gq * srcW);
                bA += oy * bq; bB += oy * (bb - bq * srcW);
            }
            return AreaRound(rA, rB, srcW, srcH) | (AreaRound(gA, gB, srcW, srcH) << 8) | (AreaRound(bA, bB, srcW, srcH) << 16);
        }

        // cv2 INTER_CUBIC for uint8: Keys a=-0.75 coefficients rounded to 11-bit fixed point, integer
        // accumulation, one rounding shift of 22 bits, clamp. Integer throughout, so this reproduces
        // cv2 exactly; source taps outside the image replicate the border.
        // Source coordinate (d + 0.5) * src / dst - 0.5 = ((2d + 1) * src - dst) / (2 * dst). cv2 rounds it
        // to FLOAT and then takes fx - floor(fx), so the fraction carries float's ulp at the coordinate's
        // magnitude (~3e-5 near 500), and that quantised fraction is what its 11-bit coefficients round
        // from. Reproduce that, from exact integers (one division): the EXACT rational fraction disagrees
        // with cv2 on 0.061% of values, this on 0.014% (tools/dav3/check_cv2_port.py, 2026-09-23).
        float fx = (float)((2 * dx + 1) * srcW - dstW) / (2 * dstW);
        float fy = (float)((2 * dy + 1) * srcH - dstH) / (2 * dstH);
        float flx = MathF.Floor(fx), fly = MathF.Floor(fy);
        int ix = (int)flx, iy = (int)fly;
        float tx = fx - flx, ty = fy - fly;

        // int, like cv2: |coefficient| sums to at most 2816 (x = 0.5), so 255 * 2816 * 2816 = 2.02e9 < 2^31.
        int rs = 0, gs = 0, bs = 0;
        for (int j = 0; j < 4; j++)
        {
            int sy = iy - 1 + j;
            if (sy < 0) sy = 0; if (sy > srcH - 1) sy = srcH - 1;
            int by = CubicFixed(ty, j);
            int rr = 0, gg = 0, bb = 0;
            for (int i = 0; i < 4; i++)
            {
                int sx = ix - 1 + i;
                if (sx < 0) sx = 0; if (sx > srcW - 1) sx = srcW - 1;
                int ax = CubicFixed(tx, i);
                int px = src[sy * srcW + sx];
                rr += (px & 0xFF) * ax;
                gg += ((px >> 8) & 0xFF) * ax;
                bb += ((px >> 16) & 0xFF) * ax;
            }
            rs += rr * by; gs += gg * by; bs += bb * by;
        }
        return ShiftU8(rs) | (ShiftU8(gs) << 8) | (ShiftU8(bs) << 16);
    }

    /// <summary>
    /// round((srcW * A + B) / (srcW * srcH)), ties to even (cv2 cvRound), clamped to a uint8, in 32-bit
    /// integers: B = srcW * bq + br (0 &lt;= br &lt; srcW), C = A + bq = srcH * q + rc, so the exact remainder
    /// is srcW * rc + br &lt; srcW * srcH. Valid while 2 * srcW * srcH fits in an int (&lt; 2^30 pixels,
    /// checked by <see cref="NativeAspectSizes"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int AreaRound(int a, int b, int srcW, int srcH)
    {
        int bq = b / srcW, br = b - bq * srcW;
        int c = a + bq;
        int q = c / srcH, rc = c - q * srcH;
        int r2 = 2 * (srcW * rc + br), den = srcW * srcH;
        if (r2 > den || (r2 == den && (q & 1) == 1)) q++;
        return q > 255 ? 255 : q;
    }

    /// <summary>cv2 interpolateCubic (A = -0.75) tap <paramref name="k"/>, as cv2's saturate_cast&lt;short&gt;(c * 2048).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CubicFixed(float x, int k)
    {
        const float A = -0.75f;
        float c0 = ((A * (x + 1f) - 5f * A) * (x + 1f) + 8f * A) * (x + 1f) - 4f * A;
        float c1 = ((A + 2f) * x - (A + 3f)) * x * x + 1f;
        float c2 = ((A + 2f) * (1f - x) - (A + 3f)) * (1f - x) * (1f - x) + 1f;
        float c = k == 0 ? c0 : k == 1 ? c1 : k == 2 ? c2 : 1f - c0 - c1 - c2;
        return (int)MathF.Round(c * 2048f);
    }

    /// <summary>cv2 FixedPtCast&lt;int, uchar, 22&gt;: add half, shift, clamp.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ShiftU8(int v)
    {
        int i = (v + (1 << 21)) >> 22;
        return i < 0 ? 0 : i > 255 ? 255 : i;
    }
}
