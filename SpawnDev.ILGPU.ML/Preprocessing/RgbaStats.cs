using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Questions about a packed-RGBA picture on the device (alpha is the top byte), so a page does not read a whole picture into managed memory to loop over it: only the answer crosses.
/// </summary>
public static class RgbaStats
{
    // DeviceCount partials (no atomics, one store per thread: they run on WebGL too)
    private static void NotOpaqueImpl(Index1D p, IV rgba, IV partials, int n)
    {
        int count = 0;
        for (int i = p; i < n; i += DeviceCount.Partials)
            if (((rgba[i] >> 24) & 255) != 255) count++;
        partials[p] = count;
    }

    private static void RingImpl(Index1D p, IV rgba, IV partials, int total, int w, int h, int ring)
    {
        int count = 0;
        for (int i = p; i < total; i += DeviceCount.Partials) count += RingHit(i, rgba, w, h, ring);
        partials[p] = count;
    }

    private static int RingHit(int i, IV rgba, int w, int h, int ring)
    {
        // the pixels within `ring` of the frame: the top band, the bottom band, then the left and right columns of the rows between
        int bandPixels = w * ring;
        int x, y;
        if (i < bandPixels) { x = i % w; y = i / w; }
        else if (i < 2 * bandPixels) { int j = i - bandPixels; x = j % w; y = h - ring + j / w; }
        else { int j = i - 2 * bandPixels, c = j % (2 * ring); x = c < ring ? c : w - 2 * ring + c; y = ring + j / (2 * ring); }
        return ((rgba[y * w + x] >> 24) & 255) > 127 ? 1 : 0;
    }

    private static void OpaqueImpl(Index1D i, IV rgba) { rgba[i] = rgba[i] | unchecked((int)0xFF000000); }

    /// <summary>Sets every pixel's alpha to 255 (a drawing from an image model has no alpha of its own). Queued; the buffer must outlive the launch.</summary>
    public static void MakeOpaque(Accelerator accelerator, IV rgba, int pixelCount)
        => accelerator.LoadAutoGroupedStreamKernel<Index1D, IV>(OpaqueImpl)(pixelCount, rgba);

    /// <summary>True when every pixel's alpha is 255.</summary>
    public static async Task<bool> IsFullyOpaqueAsync(Accelerator accelerator, IV rgba, int width, int height)
    {
        var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int>(NotOpaqueImpl);
        return await DeviceCount.SumAsync(accelerator, partials => k(DeviceCount.Partials, rgba, partials, width * height)).ConfigureAwait(false) == 0;
    }

    /// <summary>The fraction of the pixels within <paramref name="ring"/> of the frame whose alpha is above 127: how much of a cut-out touches the picture's edge (a cropped drawing becomes a truncated model).</summary>
    public static async Task<double> BorderTouchAsync(Accelerator accelerator, IV rgba, int width, int height, int ring = 2)
    {
        if (width <= 2 * ring || height <= 2 * ring) return 0;
        var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, IV, IV, int, int, int, int>(RingImpl);
        long total = 2L * width * ring + 2L * ring * (height - 2 * ring);
        int hit = await DeviceCount.SumAsync(accelerator, partials => k(DeviceCount.Partials, rgba, partials, (int)total, width, height, ring)).ConfigureAwait(false);
        return (double)hit / total;
    }
}
