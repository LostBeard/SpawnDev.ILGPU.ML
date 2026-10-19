using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Puts a cut-out (packed RGBA whose alpha is the foreground mask, as <see cref="Pipelines.BackgroundRemovalPipeline.RemoveBackgroundToDeviceAsync"/> returns it) over a new background on the device:
/// a flat colour or a blurred copy of the picture itself. Only the finished picture has to come back.
/// </summary>
public static class ImageCompositor
{
    // 5-wide box blur of the colour channels, one axis at a time, clamping at the edges (integer mean, as the host loops this replaces)
    private static void BlurHImpl(Index1D i, IV src, IV tmp, int w)
    {
        int x = i % w, row = i - x;
        int r = 0, g = 0, b = 0;
        for (int k = -2; k <= 2; k++)
        {
            int xs = Math.Max(0, Math.Min(w - 1, x + k));
            int p = src[row + xs];
            r += p & 0xFF; g += (p >> 8) & 0xFF; b += (p >> 16) & 0xFF;
        }
        tmp[i] = (r / 5) | ((g / 5) << 8) | ((b / 5) << 16);
    }

    private static void BlurVImpl(Index1D i, IV tmp, IV dst, int w, int h)
    {
        int x = i % w, y = i / w;
        int r = 0, g = 0, b = 0;
        for (int k = -2; k <= 2; k++)
        {
            int ys = Math.Max(0, Math.Min(h - 1, y + k));
            int p = tmp[ys * w + x];
            r += p & 0xFF; g += (p >> 8) & 0xFF; b += (p >> 16) & 0xFF;
        }
        dst[i] = (r / 5) | ((g / 5) << 8) | ((b / 5) << 16) | (0xFF << 24);
    }

    private static void CompositeImpl(Index1D i, IV fg, IV bg, IV o, int bgRgb, int useBg)
    {
        int p = fg[i];
        float a = ((p >> 24) & 0xFF) * (1f / 255f);
        int q = useBg != 0 ? bg[i] : bgRgb;
        int r = (int)((p & 0xFF) * a + (q & 0xFF) * (1f - a) + 0.5f);
        int g = (int)(((p >> 8) & 0xFF) * a + ((q >> 8) & 0xFF) * (1f - a) + 0.5f);
        int b = (int)(((p >> 16) & 0xFF) * a + ((q >> 16) & 0xFF) * (1f - a) + 0.5f);
        o[i] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | (0xFF << 24);
    }

    /// <summary>The cut-out over the flat colour <paramref name="bgRgb"/> (0xBBGGRR as packed here: R in the low byte), alpha 255. The caller disposes the result.</summary>
    public static async Task<MemoryBuffer1D<int, Stride1D.Dense>> OverColorAsync(Accelerator accelerator, IV cut, int width, int height, int bgRgb)
    {
        var composite = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int>(CompositeImpl);
        using var dummy = accelerator.Allocate1D<int>(1);
        var result = accelerator.Allocate1D<int>(width * height);
        composite(width * height, cut, dummy.View, result.View, bgRgb, 0);
        await accelerator.SynchronizeAsync().ConfigureAwait(false);
        return result;
    }

    /// <summary>The cut-out over a 5 x 5 box blur of its own colours, alpha 255. The caller disposes the result.</summary>
    public static async Task<MemoryBuffer1D<int, Stride1D.Dense>> OverBlurAsync(Accelerator accelerator, IV cut, int width, int height)
    {
        var blurH = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(BlurHImpl);
        var blurV = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int, int>(BlurVImpl);
        var composite = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, int, int>(CompositeImpl);
        using var tmp = accelerator.Allocate1D<int>(width * height);
        using var blurred = accelerator.Allocate1D<int>(width * height);
        var result = accelerator.Allocate1D<int>(width * height);
        blurH(width * height, cut, tmp.View, width);
        blurV(width * height, tmp.View, blurred.View, width, height);
        composite(width * height, cut, blurred.View, result.View, 0, 1);
        await accelerator.SynchronizeAsync().ConfigureAwait(false);
        return result;
    }
}
