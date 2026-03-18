namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Postprocessing for ESPCN super resolution model.
/// Handles the Y-channel-only workflow: split YCbCr, upscale Y on GPU,
/// upscale CbCr with bicubic on CPU, merge back to RGB.
/// </summary>
public static class SuperResPostProcessor
{
    /// <summary>
    /// Prepare ESPCN input: extract Y (luminance) channel as float [0,1] from RGBA image.
    /// Also returns Cb and Cr channels for postprocessing.
    /// </summary>
    /// <param name="rgbaPixels">Source RGBA pixels</param>
    /// <param name="srcWidth">Source width</param>
    /// <param name="srcHeight">Source height</param>
    /// <param name="targetWidth">Model input width (e.g., 224)</param>
    /// <param name="targetHeight">Model input height (e.g., 224)</param>
    /// <returns>Y channel as float [1, 1, H, W] for model input, plus Cb/Cr for later merging</returns>
    public static (float[] YTensor, byte[] Cb, byte[] Cr) PrepareInput(
        byte[] rgbaPixels, int srcWidth, int srcHeight, int targetWidth, int targetHeight)
    {
        // Resize to target size first
        var resized = ImageOps.Resize(rgbaPixels, srcWidth, srcHeight, targetWidth, targetHeight);

        // Convert to RGB
        var rgb = ColorConversion.RGBAToRGB(resized);

        // Convert to YCbCr
        var (y, cb, cr) = ColorConversion.RGBToYCbCr(rgb);

        // Y channel to float [0,1] in NCHW format [1, 1, H, W]
        var yFloat = new float[targetWidth * targetHeight];
        for (int i = 0; i < yFloat.Length; i++)
        {
            yFloat[i] = y[i] / 255f;
        }

        return (yFloat, cb, cr);
    }

    /// <summary>
    /// Merge ESPCN output (upscaled Y channel) with bicubic-upscaled Cb/Cr to produce final RGB.
    /// </summary>
    /// <param name="yOutput">Model output: upscaled Y channel as float [1, 1, outH, outW] in [0,1]</param>
    /// <param name="cb">Original Cb channel at input resolution</param>
    /// <param name="cr">Original Cr channel at input resolution</param>
    /// <param name="inputWidth">Original input width (before upscaling)</param>
    /// <param name="inputHeight">Original input height (before upscaling)</param>
    /// <param name="outputWidth">Upscaled output width</param>
    /// <param name="outputHeight">Upscaled output height</param>
    /// <returns>RGBA byte array at output resolution</returns>
    public static byte[] MergeOutput(
        float[] yOutput, byte[] cb, byte[] cr,
        int inputWidth, int inputHeight, int outputWidth, int outputHeight)
    {
        // Convert Y output to bytes
        var yBytes = new byte[outputWidth * outputHeight];
        for (int i = 0; i < yBytes.Length; i++)
        {
            yBytes[i] = (byte)Math.Clamp((int)(yOutput[i] * 255f + 0.5f), 0, 255);
        }

        // Upscale Cb and Cr using bilinear interpolation
        var cbUpscaled = ResizeChannel(cb, inputWidth, inputHeight, outputWidth, outputHeight);
        var crUpscaled = ResizeChannel(cr, inputWidth, inputHeight, outputWidth, outputHeight);

        // Merge Y, Cb, Cr back to RGB
        var rgb = ColorConversion.YCbCrToRGB(yBytes, cbUpscaled, crUpscaled);

        // Convert to RGBA
        return ColorConversion.RGBToRGBA(rgb);
    }

    /// <summary>
    /// Bilinear resize a single-channel byte array.
    /// </summary>
    private static byte[] ResizeChannel(byte[] channel, int srcW, int srcH, int dstW, int dstH)
    {
        var resized = new byte[dstW * dstH];
        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                float srcY = (y + 0.5f) * scaleY - 0.5f;

                int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, srcW - 1);
                int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, srcH - 1);
                int x1 = Math.Min(x0 + 1, srcW - 1);
                int y1 = Math.Min(y0 + 1, srcH - 1);
                float fx = srcX - x0;
                float fy = srcY - y0;

                float v = Lerp(Lerp(channel[y0 * srcW + x0], channel[y0 * srcW + x1], fx),
                               Lerp(channel[y1 * srcW + x0], channel[y1 * srcW + x1], fx), fy);
                resized[y * dstW + x] = (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
            }
        }

        return resized;
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
