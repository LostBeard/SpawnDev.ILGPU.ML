namespace SpawnDev.ILGPU.ML.Demo.UI.Data;

/// <summary>
/// Color palettes for depth map visualization.
/// Each palette maps a normalized depth value [0,1] to an RGB color.
/// Palettes sampled at 256 points from matplotlib colormaps.
/// </summary>
public static class DepthColorMaps
{
    /// <summary>
    /// Get RGB color for a normalized depth value using the specified palette.
    /// </summary>
    public static (byte R, byte G, byte B) GetColor(string palette, float normalizedDepth)
    {
        var lut = palette.ToLowerInvariant() switch
        {
            "plasma" => Plasma,
            "viridis" => Viridis,
            "inferno" => Inferno,
            "grayscale" => Grayscale,
            _ => Plasma
        };

        var idx = Math.Clamp((int)(normalizedDepth * 255), 0, 255);
        return lut[idx];
    }

    /// <summary>
    /// Apply a colormap to a float depth array, producing RGBA bytes.
    /// </summary>
    public static byte[] ApplyColorMap(float[] depth, int width, int height, string palette)
    {
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < depth.Length; i++)
        {
            if (depth[i] < min) min = depth[i];
            if (depth[i] > max) max = depth[i];
        }

        float range = max - min;
        if (range < 1e-6f) range = 1f;

        var lut = palette.ToLowerInvariant() switch
        {
            "plasma" => Plasma,
            "viridis" => Viridis,
            "inferno" => Inferno,
            "grayscale" => Grayscale,
            _ => Plasma
        };

        var rgba = new byte[width * height * 4];
        for (int i = 0; i < depth.Length && i < width * height; i++)
        {
            var normalized = (depth[i] - min) / range;
            var idx = Math.Clamp((int)(normalized * 255), 0, 255);
            var (r, g, b) = lut[idx];
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 255;
        }

        return rgba;
    }

    public static readonly string[] AvailablePalettes = { "plasma", "viridis", "inferno", "grayscale" };

    // Grayscale: simple linear ramp
    private static readonly (byte R, byte G, byte B)[] Grayscale = Enumerable.Range(0, 256)
        .Select(i => ((byte)i, (byte)i, (byte)i))
        .ToArray();

    // Plasma colormap (256 samples from matplotlib)
    private static readonly (byte R, byte G, byte B)[] Plasma = GeneratePlasma();

    // Viridis colormap (256 samples from matplotlib)
    private static readonly (byte R, byte G, byte B)[] Viridis = GenerateViridis();

    // Inferno colormap (256 samples from matplotlib)
    private static readonly (byte R, byte G, byte B)[] Inferno = GenerateInferno();

    private static (byte, byte, byte)[] GeneratePlasma()
    {
        // Approximate plasma colormap using key control points and linear interpolation
        var controls = new (float t, byte r, byte g, byte b)[]
        {
            (0.00f, 13, 8, 135),
            (0.13f, 75, 3, 161),
            (0.25f, 126, 3, 168),
            (0.38f, 168, 34, 150),
            (0.50f, 203, 70, 121),
            (0.63f, 229, 107, 93),
            (0.75f, 248, 149, 64),
            (0.88f, 253, 195, 40),
            (1.00f, 240, 249, 33),
        };
        return InterpolateColormap(controls);
    }

    private static (byte, byte, byte)[] GenerateViridis()
    {
        var controls = new (float t, byte r, byte g, byte b)[]
        {
            (0.00f, 68, 1, 84),
            (0.13f, 72, 36, 117),
            (0.25f, 65, 68, 135),
            (0.38f, 53, 95, 141),
            (0.50f, 33, 145, 140),
            (0.63f, 53, 183, 121),
            (0.75f, 109, 205, 89),
            (0.88f, 180, 222, 44),
            (1.00f, 253, 231, 37),
        };
        return InterpolateColormap(controls);
    }

    private static (byte, byte, byte)[] GenerateInferno()
    {
        var controls = new (float t, byte r, byte g, byte b)[]
        {
            (0.00f, 0, 0, 4),
            (0.13f, 31, 12, 72),
            (0.25f, 85, 15, 109),
            (0.38f, 136, 34, 106),
            (0.50f, 186, 54, 85),
            (0.63f, 227, 89, 51),
            (0.75f, 249, 140, 10),
            (0.88f, 249, 201, 50),
            (1.00f, 252, 255, 164),
        };
        return InterpolateColormap(controls);
    }

    private static (byte, byte, byte)[] InterpolateColormap((float t, byte r, byte g, byte b)[] controls)
    {
        var result = new (byte, byte, byte)[256];
        for (int i = 0; i < 256; i++)
        {
            float t = i / 255f;
            int j = 0;
            while (j < controls.Length - 2 && controls[j + 1].t < t) j++;

            var a = controls[j];
            var b = controls[j + 1];
            float frac = (t - a.t) / (b.t - a.t);
            frac = Math.Clamp(frac, 0, 1);

            result[i] = (
                (byte)(a.r + (b.r - a.r) * frac),
                (byte)(a.g + (b.g - a.g) * frac),
                (byte)(a.b + (b.b - a.b) * frac)
            );
        }
        return result;
    }
}
