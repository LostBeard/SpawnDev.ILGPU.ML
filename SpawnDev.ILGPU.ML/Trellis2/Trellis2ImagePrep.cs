using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Preprocessing;
using TypedArray = SpawnDev.SpawnJS.JSObjects.TypedArray;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Image preparation of Trellis2ImageTo3DPipeline.preprocess_image + get_cond, on the device, for an RGBA image that already carries its
/// foreground mask in alpha: crop the square around the alpha &gt; 0.8 bounding box (outside = transparent black), premultiply colour by alpha
/// (the reference truncates to 8 bits), Lanczos-resize to S x S as Pillow does (separable, 8-bit rounding between the passes), then
/// ImageNet-normalise into a channel-first float tensor [3, S, S]. Pixels are packed RGBA ints, R in the low byte (the library convention,
/// <see cref="RgbaUpload"/>). A fully opaque image has no foreground mask: background removal must run first.
/// </summary>
public sealed class Trellis2ImagePrep
{
    private readonly Accelerator _acc;
    public Trellis2ImagePrep(Accelerator accelerator) => _acc = accelerator;

    private Action<Index1D, IV, IV, int>? _box;
    private static void BoxImpl(Index1D i, IV rgba, IV box, int w)
    {
        int a = (rgba[i] >> 24) & 255;
        if (a > 204)                                    // alpha > 0.8 * 255 = 204
        {
            int x = i % w, y = i / w;
            Atomic.Min(ref box[0], x); Atomic.Min(ref box[1], y); Atomic.Max(ref box[2], x); Atomic.Max(ref box[3], y);
        }
        if (a != 255) Atomic.Max(ref box[4], 1);        // any transparency at all
    }

    private Action<Index1D, IV, FV, int, int, int, int, int>? _cropPremul;
    private static void CropPremulImpl(Index1D idx, IV rgba, FV o, int w, int h, int x0, int y0, int cw)
    {
        int cx = idx % cw, cy = idx / cw;
        int sx = x0 + cx, sy = y0 + cy;
        float r = 0f, g = 0f, b = 0f;
        if (sx >= 0 && sy >= 0 && sx < w && sy < h)
        {
            int px = rgba[sy * w + sx];
            float a = ((px >> 24) & 255) / 255f;
            // (arr / 255) * (alpha / 255) * 255, truncated to uint8: the reference premultiplies in float32 and casts
            r = MathF.Floor(((px & 255) / 255f) * a * 255f);
            g = MathF.Floor((((px >> 8) & 255) / 255f) * a * 255f);
            b = MathF.Floor((((px >> 16) & 255) / 255f) * a * 255f);
        }
        o[idx * 3] = r; o[idx * 3 + 1] = g; o[idx * 3 + 2] = b;
    }

    private static float Lanczos3(float x)
    {
        if (x <= -3f || x >= 3f) return 0f;
        if (x == 0f) return 1f;
        float px = 3.14159265358979f * x;
        return (MathF.Sin(px) / px) * (MathF.Sin(px / 3f) / (px / 3f));
    }

    // Pillow ImagingResample coefficients, one output sample along one axis (3 interleaved channels)
    private Action<Index1D, FV, FV, int, int, int>? _resizeH;
    private static void ResizeHImpl(Index1D idx, FV src, FV dst, int inW, int outW, int rows)
    {
        int ox = idx % outW, y = idx / outW;
        float scale = (float)inW / outW, fscale = MathF.Max(scale, 1f), support = 3f * fscale;
        float center = (ox + 0.5f) * scale, ss = 1f / fscale;
        int xmin = (int)(center - support + 0.5f); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5f); if (xmax > inW) xmax = inW;
        float wsum = 0f, r = 0f, g = 0f, b = 0f;
        for (int x = xmin; x < xmax; x++)
        {
            float wgt = Lanczos3((x - center + 0.5f) * ss);
            wsum += wgt;
            long s = ((long)y * inW + x) * 3;
            r += wgt * src[s]; g += wgt * src[s + 1]; b += wgt * src[s + 2];
        }
        float inv = wsum != 0f ? 1f / wsum : 0f;
        dst[idx * 3] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(r * inv + 0.5f)));
        dst[idx * 3 + 1] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(g * inv + 0.5f)));
        dst[idx * 3 + 2] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(b * inv + 0.5f)));
    }

    private Action<Index1D, FV, FV, int, int, int>? _resizeV;
    private static void ResizeVImpl(Index1D idx, FV src, FV dst, int inH, int outH, int width)
    {
        int x = idx % width, oy = idx / width;
        float scale = (float)inH / outH, fscale = MathF.Max(scale, 1f), support = 3f * fscale;
        float center = (oy + 0.5f) * scale, ss = 1f / fscale;
        int ymin = (int)(center - support + 0.5f); if (ymin < 0) ymin = 0;
        int ymax = (int)(center + support + 0.5f); if (ymax > inH) ymax = inH;
        float wsum = 0f, r = 0f, g = 0f, b = 0f;
        for (int y = ymin; y < ymax; y++)
        {
            float wgt = Lanczos3((y - center + 0.5f) * ss);
            wsum += wgt;
            long s = ((long)y * width + x) * 3;
            r += wgt * src[s]; g += wgt * src[s + 1]; b += wgt * src[s + 2];
        }
        float inv = wsum != 0f ? 1f / wsum : 0f;
        dst[idx * 3] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(r * inv + 0.5f)));
        dst[idx * 3 + 1] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(g * inv + 0.5f)));
        dst[idx * 3 + 2] = MathF.Min(255f, MathF.Max(0f, MathF.Floor(b * inv + 0.5f)));
    }

    private Action<Index1D, FV, FV, int>? _normalize;
    private static void NormalizeImpl(Index1D idx, FV src, FV o, int size)
    {
        int c = idx / (size * size), p = idx % (size * size);
        float mean = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f, std = c == 0 ? 0.229f : c == 1 ? 0.224f : 0.225f;
        o[idx] = (src[p * 3 + c] / 255f - mean) / std;
    }

    /// <summary>
    /// <paramref name="rgba"/> [width*height] packed RGBA; returns the normalised tensor [3, outSize, outSize] (caller disposes).
    /// Throws when the image has no transparency (no foreground mask).
    /// </summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Prepare(IV rgba, int width, int height, int outSize = 512)
        => Task.Run(() => PrepareAsync(rgba, width, height, outSize)).GetAwaiter().GetResult();

    /// <summary><see cref="Prepare"/>, browser-safe (the crop box is an awaited readback).</summary>
    public async Task<MemoryBuffer1D<float, Stride1D.Dense>> PrepareAsync(IV rgba, int width, int height, int outSize = 512)
    {
        _box ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(BoxImpl);
        _cropPremul ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, int, int, int, int, int>(CropPremulImpl);
        _resizeH ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(ResizeHImpl);
        _resizeV ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, int>(ResizeVImpl);
        _normalize ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int>(NormalizeImpl);
        using var box = _acc.Allocate1D(new[] { int.MaxValue, int.MaxValue, -1, -1, 0 });
        _box(width * height, rgba, box.View, width);
        var b = await box.View.CopyToHostAsync();               // five scalars decide the crop
        if (b[4] == 0) throw new NotSupportedException("the image is fully opaque: it has no foreground mask (run background removal first)");
        if (b[2] < 0) throw new InvalidOperationException("no pixel has alpha > 0.8: the mask is empty");
        // bbox from the pixels, then the square crop exactly as the reference computes it (Pillow rounds the float box half to even)
        double cx = (b[0] + b[2]) / 2.0, cy = (b[1] + b[3]) / 2.0;
        int size = Math.Max(b[2] - b[0], b[3] - b[1]);
        int x0 = (int)Math.Round(cx - size / 2, MidpointRounding.ToEven), y0 = (int)Math.Round(cy - size / 2, MidpointRounding.ToEven);
        int x1 = (int)Math.Round(cx + size / 2, MidpointRounding.ToEven), y1 = (int)Math.Round(cy + size / 2, MidpointRounding.ToEven);
        int cw = x1 - x0, ch = y1 - y0;
        if (cw < 1 || ch < 1) throw new InvalidOperationException("degenerate crop");
        using var cropped = _acc.Allocate1D<float>((long)cw * ch * 3);
        _cropPremul(cw * ch, rgba, cropped.View, width, height, x0, y0, cw);
        using var horiz = _acc.Allocate1D<float>((long)outSize * ch * 3);
        _resizeH(outSize * ch, cropped.View, horiz.View, cw, outSize, ch);
        using var vert = _acc.Allocate1D<float>((long)outSize * outSize * 3);
        _resizeV(outSize * outSize, horiz.View, vert.View, ch, outSize, outSize);
        var result = _acc.Allocate1D<float>(3L * outSize * outSize);
        _normalize(3 * outSize * outSize, vert.View, result.View, outSize);
        _acc.Fence();
        return result;
    }

    /// <summary>Managed pixels (desktop / oracle path): packed RGBA ints, R in the low byte.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Prepare(int[] rgbaPixels, int width, int height, int outSize = 512)
    {
        using var buf = RgbaUpload.FromManaged(_acc, rgbaPixels, width, height);
        return Prepare(buf.View, width, height, outSize);
    }

    /// <summary>Browser path: RGBA bytes still in a JS typed array (<c>ImageData.Data</c>); they never enter the .NET heap.</summary>
    public MemoryBuffer1D<float, Stride1D.Dense> Prepare(TypedArray rgbaBytes, int width, int height, int outSize = 512)
    {
        using var buf = RgbaUpload.FromTypedArray(_acc, rgbaBytes, width, height);
        return Prepare(buf.View, width, height, outSize);
    }
}
