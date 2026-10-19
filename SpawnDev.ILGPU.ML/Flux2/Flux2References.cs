namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>A reference picture for editing: 8-bit RGBA, any size (the alpha channel is ignored).</summary>
public sealed record Flux2ReferenceImage(int Width, int Height, byte[] Rgba);

/// <summary>
/// What the reference's <c>Flux2Pipeline</c> does to a reference image before the VAE encoder sees it (diffusers <c>Flux2ImageProcessor</c>): a picture above 1024 x 1024 pixels is shrunk to that area with
/// Lanczos (sides truncated to integers), then its sides are floored to multiples of 16 by a CENTRE CROP (no resize), and the pixels go to [-1, 1].
/// </summary>
public static class Flux2ReferencePrep
{
    public const int MaxArea = 1024 * 1024, Multiple = 16;

    /// <summary>The size (width, height) a picture of this size is prepared to: shrunk to 1024 x 1024 pixels of area if larger, then floored to multiples of 16.</summary>
    public static (int Width, int Height) PreparedSize(int w, int h)
    {
        if ((long)w * h > MaxArea) { double scale = Math.Sqrt((double)MaxArea / ((long)w * h)); (w, h) = ((int)(w * scale), (int)(h * scale)); }
        return (w / Multiple * Multiple, h / Multiple * Multiple);
    }

    /// <summary>The prepared image as planar CHW floats in [-1, 1], with its width and height (multiples of 16).</summary>
    public static (float[] Chw, int Width, int Height) Prepare(Flux2ReferenceImage image)
    {
        int w = image.Width, h = image.Height;
        if (w <= 0 || h <= 0 || image.Rgba.Length < w * h * 4) throw new ArgumentException("The reference image is empty or its pixel buffer is too short.");
        var px = image.Rgba;
        if ((long)w * h > MaxArea)
        {
            double scale = Math.Sqrt((double)MaxArea / ((long)w * h));
            int nw = (int)(w * scale), nh = (int)(h * scale);
            px = Lanczos(px, w, h, nw, nh); w = nw; h = nh;
        }
        int tw = w / Multiple * Multiple, th = h / Multiple * Multiple;
        if (tw < Multiple || th < Multiple) throw new ArgumentException("The reference image is smaller than 16 pixels.");
        int left = (w - tw) / 2, top = (h - th) / 2;
        var chw = new float[3 * tw * th];
        for (int y = 0; y < th; y++)
            for (int x = 0; x < tw; x++)
            {
                int s = ((top + y) * w + left + x) * 4;
                for (int c = 0; c < 3; c++) chw[(c * th + y) * tw + x] = px[s + c] / 255f * 2f - 1f;
            }
        return (chw, tw, th);
    }

    // ---- Lanczos (the Pillow algorithm: a separable filter whose support grows with the shrink factor) --------------------------------------------------------------

    private static double LanczosKernel(double x)
    {
        if (x <= -3.0 || x >= 3.0) return 0.0;
        if (x == 0.0) return 1.0;
        double px = Math.PI * x;
        return Math.Sin(px) / px * (Math.Sin(px / 3.0) / (px / 3.0));
    }

    private static (int[] Start, float[][] Weights) Coefficients(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize, filterScale = Math.Max(scale, 1.0), support = 3.0 * filterScale;
        var start = new int[outSize]; var weights = new float[outSize][];
        for (int i = 0; i < outSize; i++)
        {
            double center = (i + 0.5) * scale;
            int min = Math.Max((int)(center - support + 0.5), 0), max = Math.Min((int)(center + support + 0.5), inSize) - min;
            var k = new double[max]; double sum = 0;
            for (int j = 0; j < max; j++) { k[j] = LanczosKernel((j + min - center + 0.5) / filterScale); sum += k[j]; }
            var wt = new float[max];
            for (int j = 0; j < max; j++) wt[j] = (float)(k[j] / sum);
            start[i] = min; weights[i] = wt;
        }
        return (start, weights);
    }

    /// <summary>Lanczos resize of RGBA bytes (the alpha channel is resized with the rest).</summary>
    public static byte[] Lanczos(byte[] rgba, int w, int h, int nw, int nh)
    {
        if (w == nw && h == nh) return rgba;
        var (hs, hw) = Coefficients(w, nw);
        var tmp = new byte[nw * h * 4];                                                // horizontal pass first, rounded to 8 bits like Pillow's
        for (int y = 0; y < h; y++)
            for (int x = 0; x < nw; x++)
                for (int c = 0; c < 4; c++)
                {
                    float acc = 0; var k = hw[x]; int s = hs[x];
                    for (int j = 0; j < k.Length; j++) acc += k[j] * rgba[((long)y * w + s + j) * 4 + c];
                    tmp[((long)y * nw + x) * 4 + c] = (byte)Math.Clamp((int)MathF.Floor(acc + 0.5f), 0, 255);
                }
        var (vs, vw) = Coefficients(h, nh);
        var o = new byte[nw * nh * 4];
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++)
                for (int c = 0; c < 4; c++)
                {
                    float acc = 0; var k = vw[y]; int s = vs[y];
                    for (int j = 0; j < k.Length; j++) acc += k[j] * tmp[((long)(s + j) * nw + x) * 4 + c];
                    o[((long)y * nw + x) * 4 + c] = (byte)Math.Clamp((int)MathF.Floor(acc + 0.5f), 0, 255);
                }
        return o;
    }
}
