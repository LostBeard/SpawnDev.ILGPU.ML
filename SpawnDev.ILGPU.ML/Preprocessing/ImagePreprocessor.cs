namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// CPU-side image preprocessing for ML model input.
/// Converts raw image bytes into normalized float tensors in NCHW or NHWC layout.
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>
    /// Standard ImageNet normalization (used by MobileNet, ViT, Depth Anything, etc.)
    /// </summary>
    public static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    public static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// RMBG normalization (mean=0.5, std=1.0 per channel)
    /// </summary>
    public static readonly float[] RmbgMean = { 0.5f, 0.5f, 0.5f };
    public static readonly float[] RmbgStd = { 1.0f, 1.0f, 1.0f };

    /// <summary>
    /// Preprocess RGBA pixel data into a float32 NCHW tensor with normalization.
    /// </summary>
    /// <param name="rgbaPixels">Input RGBA bytes [H, W, 4]</param>
    /// <param name="srcWidth">Source image width</param>
    /// <param name="srcHeight">Source image height</param>
    /// <param name="targetWidth">Target tensor width</param>
    /// <param name="targetHeight">Target tensor height</param>
    /// <param name="mean">Per-channel mean (RGB). Null = no normalization.</param>
    /// <param name="std">Per-channel std (RGB). Null = no normalization.</param>
    /// <param name="scaleTo01">If true, divide by 255 before normalization</param>
    /// <returns>Float array of shape [1, 3, targetHeight, targetWidth]</returns>
    public static float[] PreprocessToNCHW(
        byte[] rgbaPixels, int srcWidth, int srcHeight,
        int targetWidth, int targetHeight,
        float[]? mean = null, float[]? std = null,
        bool scaleTo01 = true)
    {
        var output = new float[3 * targetHeight * targetWidth];
        float scaleX = (float)srcWidth / targetWidth;
        float scaleY = (float)srcHeight / targetHeight;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Bilinear sample from source
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                float srcY = (y + 0.5f) * scaleY - 0.5f;
                var (r, g, b) = BilinearSample(rgbaPixels, srcWidth, srcHeight, srcX, srcY);

                float rf = r, gf = g, bf = b;
                if (scaleTo01) { rf /= 255f; gf /= 255f; bf /= 255f; }

                if (mean != null && std != null)
                {
                    rf = (rf - mean[0]) / std[0];
                    gf = (gf - mean[1]) / std[1];
                    bf = (bf - mean[2]) / std[2];
                }

                // NCHW: [C, H, W]
                int hw = targetHeight * targetWidth;
                output[0 * hw + y * targetWidth + x] = rf;
                output[1 * hw + y * targetWidth + x] = gf;
                output[2 * hw + y * targetWidth + x] = bf;
            }
        }

        return output;
    }

    /// <summary>
    /// Preprocess RGBA pixel data into an int32 NHWC tensor (for MoveNet).
    /// </summary>
    public static int[] PreprocessToNHWCInt32(
        byte[] rgbaPixels, int srcWidth, int srcHeight,
        int targetWidth, int targetHeight)
    {
        var output = new int[3 * targetHeight * targetWidth];
        float scaleX = (float)srcWidth / targetWidth;
        float scaleY = (float)srcHeight / targetHeight;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                float srcY = (y + 0.5f) * scaleY - 0.5f;
                var (r, g, b) = BilinearSample(rgbaPixels, srcWidth, srcHeight, srcX, srcY);

                // NHWC: [H, W, C]
                int idx = (y * targetWidth + x) * 3;
                output[idx + 0] = r;
                output[idx + 1] = g;
                output[idx + 2] = b;
            }
        }

        return output;
    }

    /// <summary>
    /// Preprocess for style transfer: NCHW float [0, 255] range, no normalization.
    /// </summary>
    public static float[] PreprocessToNCHW255(
        byte[] rgbaPixels, int srcWidth, int srcHeight,
        int targetWidth, int targetHeight)
    {
        return PreprocessToNCHW(rgbaPixels, srcWidth, srcHeight,
            targetWidth, targetHeight, mean: null, std: null, scaleTo01: false);
    }

    /// <summary>
    /// Preprocess for YOLO: letterbox resize to target size with gray padding, NCHW [0,1].
    /// Returns the preprocessed tensor and the letterbox parameters for postprocessing.
    /// </summary>
    public static (float[] Tensor, LetterboxInfo Info) PreprocessLetterbox(
        byte[] rgbaPixels, int srcWidth, int srcHeight,
        int targetWidth, int targetHeight)
    {
        float scale = Math.Min((float)targetWidth / srcWidth, (float)targetHeight / srcHeight);
        int newW = (int)(srcWidth * scale);
        int newH = (int)(srcHeight * scale);
        int padX = (targetWidth - newW) / 2;
        int padY = (targetHeight - newH) / 2;

        var output = new float[3 * targetHeight * targetWidth];
        int hw = targetHeight * targetWidth;

        // Fill with gray (114/255)
        float gray = 114f / 255f;
        for (int i = 0; i < hw; i++)
        {
            output[0 * hw + i] = gray;
            output[1 * hw + i] = gray;
            output[2 * hw + i] = gray;
        }

        // Draw resized image into letterbox
        for (int y = 0; y < newH; y++)
        {
            for (int x = 0; x < newW; x++)
            {
                float srcX = (x + 0.5f) * srcWidth / newW - 0.5f;
                float srcY = (y + 0.5f) * srcHeight / newH - 0.5f;
                var (r, g, b) = BilinearSample(rgbaPixels, srcWidth, srcHeight, srcX, srcY);

                int outX = x + padX;
                int outY = y + padY;
                if (outX >= 0 && outX < targetWidth && outY >= 0 && outY < targetHeight)
                {
                    output[0 * hw + outY * targetWidth + outX] = r / 255f;
                    output[1 * hw + outY * targetWidth + outX] = g / 255f;
                    output[2 * hw + outY * targetWidth + outX] = b / 255f;
                }
            }
        }

        return (output, new LetterboxInfo
        {
            Scale = scale,
            PadX = padX,
            PadY = padY,
            ResizedWidth = newW,
            ResizedHeight = newH,
        });
    }

    /// <summary>
    /// Convert NCHW float output [0,255] to RGBA byte array for display.
    /// </summary>
    public static byte[] NCHWToRGBA(float[] nchw, int width, int height)
    {
        int hw = width * height;
        var rgba = new byte[hw * 4];
        for (int i = 0; i < hw; i++)
        {
            rgba[i * 4 + 0] = ClampByte(nchw[0 * hw + i]);
            rgba[i * 4 + 1] = ClampByte(nchw[1 * hw + i]);
            rgba[i * 4 + 2] = ClampByte(nchw[2 * hw + i]);
            rgba[i * 4 + 3] = 255;
        }
        return rgba;
    }

    /// <summary>
    /// Convert a single-channel float output to an RGBA mask (for segmentation).
    /// Values are expected in [0,1] range.
    /// </summary>
    public static byte[] MaskToRGBA(float[] mask, int width, int height, bool invert = false)
    {
        var rgba = new byte[width * height * 4];
        for (int i = 0; i < width * height && i < mask.Length; i++)
        {
            float v = Math.Clamp(mask[i], 0f, 1f);
            if (invert) v = 1f - v;
            byte val = (byte)(v * 255);
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = val; // Alpha = mask value
        }
        return rgba;
    }

    /// <summary>
    /// Composite foreground over a background using an alpha mask.
    /// </summary>
    public static byte[] CompositeWithMask(byte[] fgRGBA, byte[] maskRGBA, int width, int height, byte bgR = 0, byte bgG = 0, byte bgB = 0)
    {
        var result = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            float alpha = maskRGBA[i * 4 + 3] / 255f;
            result[i * 4 + 0] = (byte)(fgRGBA[i * 4 + 0] * alpha + bgR * (1 - alpha));
            result[i * 4 + 1] = (byte)(fgRGBA[i * 4 + 1] * alpha + bgG * (1 - alpha));
            result[i * 4 + 2] = (byte)(fgRGBA[i * 4 + 2] * alpha + bgB * (1 - alpha));
            result[i * 4 + 3] = 255;
        }
        return result;
    }

    private static (byte R, byte G, byte B) BilinearSample(byte[] rgba, int w, int h, float x, float y)
    {
        int x0 = Math.Clamp((int)MathF.Floor(x), 0, w - 1);
        int y0 = Math.Clamp((int)MathF.Floor(y), 0, h - 1);
        int x1 = Math.Min(x0 + 1, w - 1);
        int y1 = Math.Min(y0 + 1, h - 1);
        float fx = x - x0;
        float fy = y - y0;

        int i00 = (y0 * w + x0) * 4;
        int i10 = (y0 * w + x1) * 4;
        int i01 = (y1 * w + x0) * 4;
        int i11 = (y1 * w + x1) * 4;

        float r = Lerp(Lerp(rgba[i00], rgba[i10], fx), Lerp(rgba[i01], rgba[i11], fx), fy);
        float g = Lerp(Lerp(rgba[i00 + 1], rgba[i10 + 1], fx), Lerp(rgba[i01 + 1], rgba[i11 + 1], fx), fy);
        float b = Lerp(Lerp(rgba[i00 + 2], rgba[i10 + 2], fx), Lerp(rgba[i01 + 2], rgba[i11 + 2], fx), fy);

        return (ClampByte(r), ClampByte(g), ClampByte(b));
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
    private static byte ClampByte(float v) => (byte)Math.Clamp((int)(v + 0.5f), 0, 255);

    public class LetterboxInfo
    {
        public float Scale { get; set; }
        public int PadX { get; set; }
        public int PadY { get; set; }
        public int ResizedWidth { get; set; }
        public int ResizedHeight { get; set; }
    }
}
