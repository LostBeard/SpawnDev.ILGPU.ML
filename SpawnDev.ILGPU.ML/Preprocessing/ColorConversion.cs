namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Color space conversions for image processing.
/// All methods work on flat byte or float arrays.
/// </summary>
public static class ColorConversion
{
    /// <summary>
    /// Convert RGBA byte array to RGB byte array (drop alpha).
    /// </summary>
    public static byte[] RGBAToRGB(byte[] rgba)
    {
        int pixelCount = rgba.Length / 4;
        var rgb = new byte[pixelCount * 3];
        for (int i = 0; i < pixelCount; i++)
        {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        return rgb;
    }

    /// <summary>
    /// Convert RGB byte array to RGBA byte array (alpha = 255).
    /// </summary>
    public static byte[] RGBToRGBA(byte[] rgb)
    {
        int pixelCount = rgb.Length / 3;
        var rgba = new byte[pixelCount * 4];
        for (int i = 0; i < pixelCount; i++)
        {
            rgba[i * 4 + 0] = rgb[i * 3 + 0];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        return rgba;
    }

    /// <summary>
    /// Convert RGB to BGR (swap red and blue channels). Works in-place if input == output.
    /// </summary>
    public static byte[] RGBToBGR(byte[] rgb)
    {
        var bgr = new byte[rgb.Length];
        int pixelCount = rgb.Length / 3;
        for (int i = 0; i < pixelCount; i++)
        {
            bgr[i * 3 + 0] = rgb[i * 3 + 2];
            bgr[i * 3 + 1] = rgb[i * 3 + 1];
            bgr[i * 3 + 2] = rgb[i * 3 + 0];
        }
        return bgr;
    }

    /// <summary>
    /// Convert RGB to grayscale using luminosity method (0.2126R + 0.7152G + 0.0722B).
    /// </summary>
    public static byte[] RGBToGrayscale(byte[] rgb)
    {
        int pixelCount = rgb.Length / 3;
        var gray = new byte[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            gray[i] = (byte)(0.2126f * rgb[i * 3] + 0.7152f * rgb[i * 3 + 1] + 0.0722f * rgb[i * 3 + 2]);
        }
        return gray;
    }

    /// <summary>
    /// Convert RGBA to grayscale.
    /// </summary>
    public static byte[] RGBAToGrayscale(byte[] rgba)
    {
        int pixelCount = rgba.Length / 4;
        var gray = new byte[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            gray[i] = (byte)(0.2126f * rgba[i * 4] + 0.7152f * rgba[i * 4 + 1] + 0.0722f * rgba[i * 4 + 2]);
        }
        return gray;
    }

    /// <summary>
    /// Convert RGB to YCbCr. Returns (Y, Cb, Cr) each as separate byte arrays.
    /// Used by ESPCN super resolution (operates on Y channel only).
    /// </summary>
    public static (byte[] Y, byte[] Cb, byte[] Cr) RGBToYCbCr(byte[] rgb)
    {
        int pixelCount = rgb.Length / 3;
        var y = new byte[pixelCount];
        var cb = new byte[pixelCount];
        var cr = new byte[pixelCount];

        for (int i = 0; i < pixelCount; i++)
        {
            float r = rgb[i * 3];
            float g = rgb[i * 3 + 1];
            float b = rgb[i * 3 + 2];

            y[i] = ClampByte(0.299f * r + 0.587f * g + 0.114f * b);
            cb[i] = ClampByte(128f - 0.168736f * r - 0.331264f * g + 0.5f * b);
            cr[i] = ClampByte(128f + 0.5f * r - 0.418688f * g - 0.081312f * b);
        }

        return (y, cb, cr);
    }

    /// <summary>
    /// Convert YCbCr back to RGB. All arrays must be same length.
    /// </summary>
    public static byte[] YCbCrToRGB(byte[] y, byte[] cb, byte[] cr)
    {
        var rgb = new byte[y.Length * 3];
        for (int i = 0; i < y.Length; i++)
        {
            float yf = y[i];
            float cbf = cb[i] - 128f;
            float crf = cr[i] - 128f;

            rgb[i * 3 + 0] = ClampByte(yf + 1.402f * crf);
            rgb[i * 3 + 1] = ClampByte(yf - 0.344136f * cbf - 0.714136f * crf);
            rgb[i * 3 + 2] = ClampByte(yf + 1.772f * cbf);
        }
        return rgb;
    }

    /// <summary>
    /// Convert RGB to HSV. Returns (H, S, V) as float arrays.
    /// H in [0, 360], S in [0, 1], V in [0, 1].
    /// </summary>
    public static (float[] H, float[] S, float[] V) RGBToHSV(byte[] rgb)
    {
        int pixelCount = rgb.Length / 3;
        var h = new float[pixelCount];
        var s = new float[pixelCount];
        var v = new float[pixelCount];

        for (int i = 0; i < pixelCount; i++)
        {
            float r = rgb[i * 3] / 255f;
            float g = rgb[i * 3 + 1] / 255f;
            float b = rgb[i * 3 + 2] / 255f;

            float max = MathF.Max(r, MathF.Max(g, b));
            float min = MathF.Min(r, MathF.Min(g, b));
            float delta = max - min;

            v[i] = max;
            s[i] = max == 0 ? 0 : delta / max;

            if (delta == 0)
                h[i] = 0;
            else if (max == r)
                h[i] = 60f * (((g - b) / delta) % 6);
            else if (max == g)
                h[i] = 60f * ((b - r) / delta + 2);
            else
                h[i] = 60f * ((r - g) / delta + 4);

            if (h[i] < 0) h[i] += 360;
        }

        return (h, s, v);
    }

    private static byte ClampByte(float v) => (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
}
