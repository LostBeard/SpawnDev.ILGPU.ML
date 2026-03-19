using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernel for applying colormaps to depth maps.
/// Converts a float depth map [H, W] normalized to [0, 1] into packed RGBA pixels.
/// Keeps the entire pipeline on GPU — no CPU readback needed for visualization.
///
/// Supports: plasma, viridis, inferno, grayscale colormaps via lookup tables.
/// </summary>
public class DepthColormapKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _colormapKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _lutBuffer;

    public DepthColormapKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Apply a colormap to a depth map on GPU.
    /// Input: float depth [H*W] normalized to [0, 1].
    /// Output: packed RGBA int [H*W].
    /// LUT: 256 packed RGBA colors (the colormap lookup table).
    /// </summary>
    public void Apply(
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int count, string palette = "plasma")
    {
        // Upload colormap LUT to GPU (cached)
        var lut = GetLUT(palette);
        if (_lutBuffer == null || _lutBuffer.Length < 256)
        {
            _lutBuffer?.Dispose();
            _lutBuffer = _accelerator.Allocate1D<int>(256);
        }
        _lutBuffer.CopyFromCPU(lut);

        _colormapKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, int>(ColormapImpl);
        _colormapKernel(count, depth, rgba, _lutBuffer.View, count);
    }

    private static void ColormapImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> depth,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<int, Stride1D.Dense> lut,
        int count)
    {
        float d = depth[idx];
        // Clamp to [0, 1] and map to LUT index [0, 255]
        if (d < 0f) d = 0f;
        if (d > 1f) d = 1f;
        int lutIdx = (int)(d * 255f);
        if (lutIdx > 255) lutIdx = 255;
        rgba[idx] = lut[lutIdx];
    }

    /// <summary>Generate a 256-entry RGBA lookup table for a colormap.</summary>
    private static int[] GetLUT(string palette)
    {
        var lut = new int[256];
        for (int i = 0; i < 256; i++)
        {
            float t = i / 255f;
            var (r, g, b) = palette.ToLowerInvariant() switch
            {
                "plasma" => Plasma(t),
                "viridis" => Viridis(t),
                "inferno" => Inferno(t),
                _ => (t, t, t) // grayscale
            };
            lut[i] = Clamp255(r) | (Clamp255(g) << 8) | (Clamp255(b) << 16) | (0xFF << 24);
        }
        return lut;
    }

    private static int Clamp255(float v)
    {
        int i = (int)(v * 255f + 0.5f);
        return i < 0 ? 0 : (i > 255 ? 255 : i);
    }

    // Approximate plasma colormap
    private static (float R, float G, float B) Plasma(float t)
    {
        float r = 0.0504f + t * (2.02f + t * (-3.34f + t * 1.84f));
        float g = 0.0286f + t * (-0.155f + t * (2.02f + t * (-2.47f + t * 0.87f)));
        float b = 0.533f + t * (1.81f + t * (-4.90f + t * (4.56f + t * (-1.49f))));
        return (r, g, b);
    }

    // Approximate viridis colormap
    private static (float R, float G, float B) Viridis(float t)
    {
        float r = 0.267f + t * (-0.005f + t * (1.77f + t * (-1.22f)));
        float g = 0.004f + t * (1.42f + t * (-0.72f + t * 0.15f));
        float b = 0.329f + t * (1.70f + t * (-3.44f + t * (2.54f + t * (-0.74f))));
        return (r, g, b);
    }

    // Approximate inferno colormap
    private static (float R, float G, float B) Inferno(float t)
    {
        float r = 0.0002f + t * (0.24f + t * (5.07f + t * (-8.12f + t * 3.82f)));
        float g = 0.0003f + t * (-0.09f + t * (2.30f + t * (-3.17f + t * 1.44f)));
        float b = 0.015f + t * (1.88f + t * (-4.77f + t * (4.34f + t * (-1.39f))));
        return (r, g, b);
    }
}
