using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Foreground mask for an opaque photo, so it can enter <see cref="Trellis2ImagePrep"/> (which needs the mask in alpha). The reference uses
/// BRIA RMBG-2.0 (CC BY-NC: not shipped here). This runs any IS-Net / U2-Net style ONNX salient-object model through the library's
/// <see cref="InferenceSession"/>: image -> 1024 x 1024 (Lanczos) -> division by the maximum, ImageNet mean, unit std -> network -> first output
/// min-max normalised -> bilinear to the photo size -> alpha. That is the preprocessing / postprocessing of rembg's isnet-general-use session.
/// ⚠ LICENSE: the IS-Net code is Apache-2.0 (xuebinqin/DIS) but the repository does not state a licence for the published
/// isnet-general-use weights (trained with the DIS5K dataset, which has its own terms of use). Treat the model file as a local
/// development dependency until that is settled; a caller that supplies an RGBA image with its own mask never needs this class.
/// </summary>
public sealed class ForegroundMask : IDisposable
{
    private readonly Accelerator _acc;
    private readonly InferenceSession _session;
    private const int Size = 1024;

    public ForegroundMask(Accelerator accelerator, string onnxPath)
    {
        _acc = accelerator;
        _session = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(onnxPath));      // a 170 MB model file: the one host read
    }

    private Action<Index1D, IV, FV, int, int>? _resizeRgb;
    private Action<Index1D, FV, IV, int>? _maxBits;
    private Action<Index1D, FV, FV, IV>? _normalize;
    private Action<Index1D, FV, IV, int>? _minMaxBits;
    private Action<Index1D, IV, FV, IV, IV, int, int, int>? _apply;

    // PIL-equivalent Lanczos resize of the colour channels of a packed RGBA photo to Size x Size, values 0..255, HWC floats
    private static float Lanczos3(float x)
    {
        if (x <= -3f || x >= 3f) return 0f;
        if (x == 0f) return 1f;
        float px = 3.14159265358979f * x;
        return (MathF.Sin(px) / px) * (MathF.Sin(px / 3f) / (px / 3f));
    }
    private static void ResizeRgbImpl(Index1D idx, IV rgba, FV o, int w, int h)
    {
        // direct 2-D evaluation of the separable filter (one thread per output pixel); 8-bit rounding is skipped (the network is not that sensitive)
        int ox = idx % Size, oy = idx / Size;
        float sx = (float)w / Size, sy = (float)h / Size;
        float fx = MathF.Max(sx, 1f), fy = MathF.Max(sy, 1f);
        float cx = (ox + 0.5f) * sx, cy = (oy + 0.5f) * sy;
        int x0 = (int)(cx - 3f * fx + 0.5f); if (x0 < 0) x0 = 0;
        int x1 = (int)(cx + 3f * fx + 0.5f); if (x1 > w) x1 = w;
        int y0 = (int)(cy - 3f * fy + 0.5f); if (y0 < 0) y0 = 0;
        int y1 = (int)(cy + 3f * fy + 0.5f); if (y1 > h) y1 = h;
        float wsum = 0f, r = 0f, g = 0f, b = 0f;
        for (int y = y0; y < y1; y++)
        {
            float wy = Lanczos3((y - cy + 0.5f) / fy);
            for (int x = x0; x < x1; x++)
            {
                float wgt = wy * Lanczos3((x - cx + 0.5f) / fx);
                int px = rgba[y * w + x];
                wsum += wgt; r += wgt * (px & 255); g += wgt * ((px >> 8) & 255); b += wgt * ((px >> 16) & 255);
            }
        }
        float inv = wsum != 0f ? 1f / wsum : 0f;
        o[idx * 3] = MathF.Min(255f, MathF.Max(0f, r * inv)); o[idx * 3 + 1] = MathF.Min(255f, MathF.Max(0f, g * inv)); o[idx * 3 + 2] = MathF.Min(255f, MathF.Max(0f, b * inv));
    }
    private static void MaxBitsImpl(Index1D i, FV v, IV max, int n) => Atomic.Max(ref max[0], (int)Interop.FloatAsInt(v[i]));   // non-negative floats order like their bits
    private static void NormalizeImpl(Index1D idx, FV hwc, FV chw, IV maxBits)
    {
        float m = MathF.Max(Interop.IntAsFloat((uint)maxBits[0]), 1e-6f);
        int c = idx / (Size * Size), p = idx % (Size * Size);
        float mean = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f;
        chw[idx] = hwc[p * 3 + c] / m - mean;                 // std = 1 for this model
    }

    private static void MinMaxImpl(Index1D i, FV v, IV mm, int n)
    {
        // mm[0] = min of the values (as ordered ints), mm[1] = max; v is a sigmoid/probability map or logits: map floats to ordered ints
        int bits = (int)Interop.FloatAsInt(v[i]);
        int ordered = bits >= 0 ? bits : bits ^ 0x7FFFFFFF;
        Atomic.Min(ref mm[0], ordered); Atomic.Max(ref mm[1], ordered);
    }
    private static float FromOrdered(int ordered) => Interop.IntAsFloat((uint)(ordered >= 0 ? ordered : ordered ^ 0x7FFFFFFF));

    private static void ApplyImpl(Index1D idx, IV rgba, FV mask, IV mm, IV o, int w, int h, int maskSize)
    {
        float lo = FromOrdered(mm[0]), hi = FromOrdered(mm[1]);
        float range = MathF.Max(hi - lo, 1e-12f);
        int x = idx % w, y = idx / w;
        float fx = (x + 0.5f) * maskSize / w - 0.5f, fy = (y + 0.5f) * maskSize / h - 0.5f;
        int x0 = (int)MathF.Max(0f, MathF.Min(maskSize - 1f, MathF.Floor(fx))), y0 = (int)MathF.Max(0f, MathF.Min(maskSize - 1f, MathF.Floor(fy)));
        int x1 = Math.Min(x0 + 1, maskSize - 1), y1 = Math.Min(y0 + 1, maskSize - 1);
        float tx = MathF.Max(0f, MathF.Min(1f, fx - x0)), ty = MathF.Max(0f, MathF.Min(1f, fy - y0));
        float m = mask[y0 * maskSize + x0] * (1 - tx) * (1 - ty) + mask[y0 * maskSize + x1] * tx * (1 - ty) + mask[y1 * maskSize + x0] * (1 - tx) * ty + mask[y1 * maskSize + x1] * tx * ty;
        float a = (m - lo) / range;
        int ai = (int)(MathF.Max(0f, MathF.Min(1f, a)) * 255f + 0.5f);
        int px = rgba[idx];
        o[idx] = (px & 0x00FFFFFF) | (ai << 24);
    }


    /// <summary>
    /// Returns a copy of <paramref name="rgba"/> [w*h] whose alpha is the predicted foreground mask (caller disposes).
    /// </summary>
    public MemoryBuffer1D<int, Stride1D.Dense> Apply(IV rgba, int w, int h) => Task.Run(() => ApplyAsync(rgba, w, h)).GetAwaiter().GetResult();

    /// <summary><see cref="Apply"/>, browser-safe.</summary>
    public async Task<MemoryBuffer1D<int, Stride1D.Dense>> ApplyAsync(IV rgba, int w, int h)
    {
        _resizeRgb ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, int, int>(ResizeRgbImpl);
        _maxBits ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, int>(MaxBitsImpl);
        _normalize ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV>(NormalizeImpl);
        _minMaxBits ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, int>(MinMaxImpl);
        _apply ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, IV, int, int, int>(ApplyImpl);
        using var hwc = _acc.Allocate1D<float>((long)Size * Size * 3);
        _resizeRgb(Size * Size, rgba, hwc.View, w, h);
        using var maxBits = _acc.Allocate1D(new[] { 0 });
        _maxBits(Size * Size * 3, hwc.View, maxBits.View, 0);
        using var chw = _acc.Allocate1D<float>((long)3 * Size * Size);
        _normalize(3 * Size * Size, hwc.View, chw.View, maxBits.View);
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = new Tensor(chw.View, new[] { 1, 3, Size, Size }),
        });
        var mask = outputs[_session.OutputNames[0]];
        int n = Size * Size;
        using var mm = _acc.Allocate1D(new[] { int.MaxValue, int.MinValue });
        _minMaxBits(n, mask.Data.SubView(0, n), mm.View, n);
        var result = _acc.Allocate1D<int>((long)w * h);
        _apply(w * h, rgba, mask.Data.SubView(0, n), mm.View, result.View, w, h, Size);
        _acc.Fence();
        return result;
    }

    /// <summary>Managed pixels (packed RGBA ints, R in the low byte).</summary>
    public MemoryBuffer1D<int, Stride1D.Dense> Apply(int[] rgbaPixels, int w, int h)
    {
        using var buf = _acc.Allocate1D(rgbaPixels);
        return Apply(buf.View, w, h);
    }

    public void Dispose() => _session.Dispose();
}
