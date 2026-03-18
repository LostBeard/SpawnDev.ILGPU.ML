namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Common image operations: crop, flip, pad, resize, and statistics.
/// All methods operate on RGBA byte arrays unless otherwise noted.
/// </summary>
public static class ImageOps
{
    /// <summary>
    /// Center crop an image to the specified size.
    /// </summary>
    public static byte[] CenterCrop(byte[] rgba, int srcW, int srcH, int cropW, int cropH)
    {
        int startX = (srcW - cropW) / 2;
        int startY = (srcH - cropH) / 2;
        return Crop(rgba, srcW, srcH, startX, startY, cropW, cropH);
    }

    /// <summary>
    /// Crop a rectangular region from an RGBA image.
    /// </summary>
    public static byte[] Crop(byte[] rgba, int srcW, int srcH, int x, int y, int cropW, int cropH)
    {
        var cropped = new byte[cropW * cropH * 4];
        for (int row = 0; row < cropH; row++)
        {
            int srcRow = Math.Clamp(y + row, 0, srcH - 1);
            for (int col = 0; col < cropW; col++)
            {
                int srcCol = Math.Clamp(x + col, 0, srcW - 1);
                int srcIdx = (srcRow * srcW + srcCol) * 4;
                int dstIdx = (row * cropW + col) * 4;
                cropped[dstIdx + 0] = rgba[srcIdx + 0];
                cropped[dstIdx + 1] = rgba[srcIdx + 1];
                cropped[dstIdx + 2] = rgba[srcIdx + 2];
                cropped[dstIdx + 3] = rgba[srcIdx + 3];
            }
        }
        return cropped;
    }

    /// <summary>
    /// Flip an RGBA image horizontally (left-right mirror).
    /// </summary>
    public static byte[] FlipHorizontal(byte[] rgba, int width, int height)
    {
        var flipped = new byte[rgba.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int srcIdx = (y * width + x) * 4;
                int dstIdx = (y * width + (width - 1 - x)) * 4;
                flipped[dstIdx + 0] = rgba[srcIdx + 0];
                flipped[dstIdx + 1] = rgba[srcIdx + 1];
                flipped[dstIdx + 2] = rgba[srcIdx + 2];
                flipped[dstIdx + 3] = rgba[srcIdx + 3];
            }
        }
        return flipped;
    }

    /// <summary>
    /// Flip an RGBA image vertically (top-bottom mirror).
    /// </summary>
    public static byte[] FlipVertical(byte[] rgba, int width, int height)
    {
        var flipped = new byte[rgba.Length];
        int rowBytes = width * 4;
        for (int y = 0; y < height; y++)
        {
            Array.Copy(rgba, y * rowBytes, flipped, (height - 1 - y) * rowBytes, rowBytes);
        }
        return flipped;
    }

    /// <summary>
    /// Pad an RGBA image with a constant color.
    /// </summary>
    public static byte[] PadConstant(byte[] rgba, int srcW, int srcH,
        int padTop, int padBottom, int padLeft, int padRight,
        byte padR = 0, byte padG = 0, byte padB = 0, byte padA = 255)
    {
        int dstW = srcW + padLeft + padRight;
        int dstH = srcH + padTop + padBottom;
        var padded = new byte[dstW * dstH * 4];

        // Fill with pad color
        for (int i = 0; i < dstW * dstH; i++)
        {
            padded[i * 4 + 0] = padR;
            padded[i * 4 + 1] = padG;
            padded[i * 4 + 2] = padB;
            padded[i * 4 + 3] = padA;
        }

        // Copy source image
        for (int y = 0; y < srcH; y++)
        {
            int srcOffset = y * srcW * 4;
            int dstOffset = ((y + padTop) * dstW + padLeft) * 4;
            Array.Copy(rgba, srcOffset, padded, dstOffset, srcW * 4);
        }

        return padded;
    }

    /// <summary>
    /// Resize an RGBA image using bilinear interpolation.
    /// </summary>
    public static byte[] Resize(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
    {
        var resized = new byte[dstW * dstH * 4];
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

                int i00 = (y0 * srcW + x0) * 4;
                int i10 = (y0 * srcW + x1) * 4;
                int i01 = (y1 * srcW + x0) * 4;
                int i11 = (y1 * srcW + x1) * 4;

                int dstIdx = (y * dstW + x) * 4;
                for (int c = 0; c < 4; c++)
                {
                    float v = Lerp(Lerp(rgba[i00 + c], rgba[i10 + c], fx),
                                   Lerp(rgba[i01 + c], rgba[i11 + c], fx), fy);
                    resized[dstIdx + c] = (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
                }
            }
        }

        return resized;
    }

    /// <summary>
    /// Resize preserving aspect ratio, fitting within maxW x maxH.
    /// Returns the resized image and the actual dimensions.
    /// </summary>
    public static (byte[] Data, int Width, int Height) ResizePreserveAspect(
        byte[] rgba, int srcW, int srcH, int maxW, int maxH)
    {
        float scale = Math.Min((float)maxW / srcW, (float)maxH / srcH);
        int newW = (int)(srcW * scale);
        int newH = (int)(srcH * scale);
        return (Resize(rgba, srcW, srcH, newW, newH), newW, newH);
    }

    /// <summary>
    /// Compute per-channel statistics for an RGBA image.
    /// </summary>
    public static ImageStats ComputeStats(byte[] rgba, int width, int height)
    {
        int pixelCount = width * height;
        float[] sumR = { 0 }, sumG = { 0 }, sumB = { 0 };
        float minR = 255, minG = 255, minB = 255;
        float maxR = 0, maxG = 0, maxB = 0;

        for (int i = 0; i < pixelCount; i++)
        {
            float r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
            sumR[0] += r; sumG[0] += g; sumB[0] += b;
            if (r < minR) minR = r; if (r > maxR) maxR = r;
            if (g < minG) minG = g; if (g > maxG) maxG = g;
            if (b < minB) minB = b; if (b > maxB) maxB = b;
        }

        float meanR = sumR[0] / pixelCount;
        float meanG = sumG[0] / pixelCount;
        float meanB = sumB[0] / pixelCount;

        float varR = 0, varG = 0, varB = 0;
        for (int i = 0; i < pixelCount; i++)
        {
            float dr = rgba[i * 4] - meanR;
            float dg = rgba[i * 4 + 1] - meanG;
            float db = rgba[i * 4 + 2] - meanB;
            varR += dr * dr; varG += dg * dg; varB += db * db;
        }

        return new ImageStats
        {
            Width = width,
            Height = height,
            MeanR = meanR / 255f, MeanG = meanG / 255f, MeanB = meanB / 255f,
            StdR = MathF.Sqrt(varR / pixelCount) / 255f,
            StdG = MathF.Sqrt(varG / pixelCount) / 255f,
            StdB = MathF.Sqrt(varB / pixelCount) / 255f,
            MinR = minR / 255f, MinG = minG / 255f, MinB = minB / 255f,
            MaxR = maxR / 255f, MaxG = maxG / 255f, MaxB = maxB / 255f,
        };
    }

    /// <summary>
    /// Compute PSNR (Peak Signal-to-Noise Ratio) between two RGBA images.
    /// Higher is better. Useful for evaluating super-resolution quality.
    /// </summary>
    public static float ComputePSNR(byte[] imageA, byte[] imageB)
    {
        int pixelCount = Math.Min(imageA.Length, imageB.Length) / 4;
        float mse = 0;
        for (int i = 0; i < pixelCount; i++)
        {
            for (int c = 0; c < 3; c++) // RGB only, skip alpha
            {
                float diff = imageA[i * 4 + c] - imageB[i * 4 + c];
                mse += diff * diff;
            }
        }
        mse /= pixelCount * 3;
        if (mse < 1e-10f) return float.PositiveInfinity;
        return 10f * MathF.Log10(255f * 255f / mse);
    }

    /// <summary>
    /// Compute SSIM (Structural Similarity Index) between two RGBA images.
    /// Returns a value in [0, 1] where 1 = identical. Uses 8x8 window.
    /// </summary>
    public static float ComputeSSIM(byte[] imageA, byte[] imageB, int width, int height)
    {
        const float C1 = 6.5025f;   // (0.01 * 255)^2
        const float C2 = 58.5225f;  // (0.03 * 255)^2
        const int windowSize = 8;

        int blocksX = width / windowSize;
        int blocksY = height / windowSize;
        if (blocksX == 0 || blocksY == 0) return 0;

        float totalSSIM = 0;
        int blockCount = 0;

        for (int by = 0; by < blocksY; by++)
        {
            for (int bx = 0; bx < blocksX; bx++)
            {
                float meanA = 0, meanB = 0;
                int startX = bx * windowSize;
                int startY = by * windowSize;

                // Compute means (grayscale approximation)
                for (int dy = 0; dy < windowSize; dy++)
                {
                    for (int dx = 0; dx < windowSize; dx++)
                    {
                        int idx = ((startY + dy) * width + (startX + dx)) * 4;
                        float grayA = 0.299f * imageA[idx] + 0.587f * imageA[idx + 1] + 0.114f * imageA[idx + 2];
                        float grayB = 0.299f * imageB[idx] + 0.587f * imageB[idx + 1] + 0.114f * imageB[idx + 2];
                        meanA += grayA;
                        meanB += grayB;
                    }
                }
                int n = windowSize * windowSize;
                meanA /= n;
                meanB /= n;

                float varA = 0, varB = 0, covAB = 0;
                for (int dy = 0; dy < windowSize; dy++)
                {
                    for (int dx = 0; dx < windowSize; dx++)
                    {
                        int idx = ((startY + dy) * width + (startX + dx)) * 4;
                        float grayA = 0.299f * imageA[idx] + 0.587f * imageA[idx + 1] + 0.114f * imageA[idx + 2];
                        float grayB = 0.299f * imageB[idx] + 0.587f * imageB[idx + 1] + 0.114f * imageB[idx + 2];
                        float da = grayA - meanA;
                        float db = grayB - meanB;
                        varA += da * da;
                        varB += db * db;
                        covAB += da * db;
                    }
                }
                varA /= n - 1;
                varB /= n - 1;
                covAB /= n - 1;

                float ssim = ((2 * meanA * meanB + C1) * (2 * covAB + C2)) /
                             ((meanA * meanA + meanB * meanB + C1) * (varA + varB + C2));
                totalSSIM += ssim;
                blockCount++;
            }
        }

        return totalSSIM / blockCount;
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    public class ImageStats
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public float MeanR { get; set; }
        public float MeanG { get; set; }
        public float MeanB { get; set; }
        public float StdR { get; set; }
        public float StdG { get; set; }
        public float StdB { get; set; }
        public float MinR { get; set; }
        public float MinG { get; set; }
        public float MinB { get; set; }
        public float MaxR { get; set; }
        public float MaxG { get; set; }
        public float MaxB { get; set; }
    }
}
