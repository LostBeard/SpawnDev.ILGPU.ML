using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU image operations using lambda kernels with captured scalars.
/// Takes advantage of SpawnDev.ILGPU's capturing lambda support —
/// scalar values (width, height, etc.) are captured at dispatch time
/// and automatically passed to the GPU. No params buffer boilerplate.
///
/// Also uses DelegateSpecialization for operations that share the same
/// kernel structure but differ only in the per-pixel math.
/// </summary>
public class ImageOpsKernel
{
    private readonly Accelerator _accelerator;

    public ImageOpsKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  Lambda kernel: Bilinear resize (captures dimensions)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Bilinear resize using a capturing lambda kernel.
    /// srcW, srcH, dstW, dstH are captured scalars — no params buffer needed.
    /// </summary>
    public void Resize(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int srcW, int srcH, int dstW, int dstH)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int dy = idx / dstW;
                int dx = idx % dstW;

                float srcX = (dx + 0.5f) * srcW / dstW - 0.5f;
                float srcY = (dy + 0.5f) * srcH / dstH - 0.5f;

                // Two-statement floor: prevents ILGPU optimizer from eliding floor() before int cast.
                // (int)x truncates toward zero — wrong for negative values (e.g., -0.25 → 0, should be -1).
                float floorX = MathF.Floor(srcX); float floorY = MathF.Floor(srcY);
                int x0 = (int)floorX; int x1 = x0 + 1;
                int y0 = (int)floorY; int y1 = y0 + 1;
                float fx = srcX - floorX; float fy = srcY - floorY;

                x0 = x0 < 0 ? 0 : (x0 >= srcW ? srcW - 1 : x0);
                x1 = x1 < 0 ? 0 : (x1 >= srcW ? srcW - 1 : x1);
                y0 = y0 < 0 ? 0 : (y0 >= srcH ? srcH - 1 : y0);
                y1 = y1 < 0 ? 0 : (y1 >= srcH ? srcH - 1 : y1);

                int p00 = inp[y0 * srcW + x0], p10 = inp[y0 * srcW + x1];
                int p01 = inp[y1 * srcW + x0], p11 = inp[y1 * srcW + x1];

                int r = (int)(Lerp(Lerp(p00 & 0xFF, p10 & 0xFF, fx), Lerp(p01 & 0xFF, p11 & 0xFF, fx), fy) + 0.5f);
                int g = (int)(Lerp(Lerp((p00 >> 8) & 0xFF, (p10 >> 8) & 0xFF, fx), Lerp((p01 >> 8) & 0xFF, (p11 >> 8) & 0xFF, fx), fy) + 0.5f);
                int b = (int)(Lerp(Lerp((p00 >> 16) & 0xFF, (p10 >> 16) & 0xFF, fx), Lerp((p01 >> 16) & 0xFF, (p11 >> 16) & 0xFF, fx), fy) + 0.5f);
                int a = (int)(Lerp(Lerp((p00 >> 24) & 0xFF, (p10 >> 24) & 0xFF, fx), Lerp((p01 >> 24) & 0xFF, (p11 >> 24) & 0xFF, fx), fy) + 0.5f);

                outp[idx] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24);
            });

        kernel((Index1D)(dstW * dstH), input, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Center crop
    // ──────────────────────────────────────────────

    public void CenterCrop(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int srcW, int srcH, int cropW, int cropH)
    {
        int startX = (srcW - cropW) / 2;
        int startY = (srcH - cropH) / 2;

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int dy = idx / cropW;
                int dx = idx % cropW;
                outp[idx] = inp[(startY + dy) * srcW + (startX + dx)];
            });

        kernel((Index1D)(cropW * cropH), input, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Flip horizontal
    // ──────────────────────────────────────────────

    public void FlipHorizontal(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int y = idx / width;
                int x = idx % width;
                outp[y * width + (width - 1 - x)] = inp[idx];
            });

        kernel((Index1D)(width * height), input, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Flip vertical
    // ──────────────────────────────────────────────

    public void FlipVertical(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int y = idx / width;
                int x = idx % width;
                outp[(height - 1 - y) * width + x] = inp[idx];
            });

        kernel((Index1D)(width * height), input, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: RGBA → Grayscale float
    // ──────────────────────────────────────────────

    public void RGBAToGrayscale(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> gray,
        int pixelCount)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<float, Stride1D.Dense> outp) =>
            {
                int pixel = inp[idx];
                float r = (pixel & 0xFF) / 255f;
                float g = ((pixel >> 8) & 0xFF) / 255f;
                float b = ((pixel >> 16) & 0xFF) / 255f;
                outp[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            });

        kernel((Index1D)pixelCount, rgba, gray);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Apply alpha mask compositing
    // ──────────────────────────────────────────────

    /// <summary>
    /// Composite foreground over background using an alpha mask.
    /// All on GPU — foreground pixels, mask, and output stay in GPU memory.
    /// </summary>
    public void CompositeWithMask(
        ArrayView1D<int, Stride1D.Dense> foreground,
        ArrayView1D<float, Stride1D.Dense> mask,
        ArrayView1D<int, Stride1D.Dense> output,
        int pixelCount,
        int bgR = 0, int bgG = 0, int bgB = 0)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> fg, ArrayView1D<float, Stride1D.Dense> m, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int pixel = fg[idx];
                float alpha = m[idx];
                alpha = alpha < 0f ? 0f : (alpha > 1f ? 1f : alpha);

                int r = (int)((pixel & 0xFF) * alpha + bgR * (1f - alpha) + 0.5f);
                int g = (int)(((pixel >> 8) & 0xFF) * alpha + bgG * (1f - alpha) + 0.5f);
                int b = (int)(((pixel >> 16) & 0xFF) * alpha + bgB * (1f - alpha) + 0.5f);

                outp[idx] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | (255 << 24);
            });

        kernel((Index1D)pixelCount, foreground, mask, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Brightness/Contrast adjustment
    // ──────────────────────────────────────────────

    public void AdjustBrightnessContrast(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int pixelCount,
        float brightness, float contrast)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int pixel = inp[idx];
                float r = (pixel & 0xFF) / 255f;
                float g = ((pixel >> 8) & 0xFF) / 255f;
                float b = ((pixel >> 16) & 0xFF) / 255f;

                // Apply brightness and contrast
                r = (r - 0.5f) * contrast + 0.5f + brightness;
                g = (g - 0.5f) * contrast + 0.5f + brightness;
                b = (b - 0.5f) * contrast + 0.5f + brightness;

                // Clamp
                r = r < 0f ? 0f : (r > 1f ? 1f : r);
                g = g < 0f ? 0f : (g > 1f ? 1f : g);
                b = b < 0f ? 0f : (b > 1f ? 1f : b);

                int ri = (int)(r * 255f + 0.5f);
                int gi = (int)(g * 255f + 0.5f);
                int bi = (int)(b * 255f + 0.5f);

                outp[idx] = ri | (gi << 8) | (bi << 16) | (pixel & unchecked((int)0xFF000000));
            });

        kernel((Index1D)pixelCount, input, output);
    }

    // ──────────────────────────────────────────────
    //  Lambda kernel: Gaussian blur (3x3 approximation)
    // ──────────────────────────────────────────────

    public void GaussianBlur3x3(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<int, Stride1D.Dense> inp, ArrayView1D<int, Stride1D.Dense> outp) =>
            {
                int y = idx / width;
                int x = idx % width;

                // 3x3 Gaussian weights: 1/16 [1,2,1; 2,4,2; 1,2,1]
                float rSum = 0, gSum = 0, bSum = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int sx = x + dx; int sy = y + dy;
                        sx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
                        sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);

                        float w = (dx == 0 && dy == 0) ? 4f : ((dx == 0 || dy == 0) ? 2f : 1f);
                        int p = inp[sy * width + sx];
                        rSum += (p & 0xFF) * w;
                        gSum += ((p >> 8) & 0xFF) * w;
                        bSum += ((p >> 16) & 0xFF) * w;
                    }
                }

                int r = (int)(rSum / 16f + 0.5f);
                int g = (int)(gSum / 16f + 0.5f);
                int b = (int)(bSum / 16f + 0.5f);

                outp[idx] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | (inp[idx] & unchecked((int)0xFF000000));
            });

        kernel((Index1D)(width * height), input, output);
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
