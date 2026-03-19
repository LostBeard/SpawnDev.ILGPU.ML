using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernels for image spatial transformations.
/// Replaces CPU-side ImageOps (resize, crop, flip, pad) with GPU-native execution.
/// All operations keep data on GPU — no CPU readback.
/// </summary>
public class ImageTransformKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _resizeKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _centerCropKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _flipHKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _flipVKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _padKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _resizeFloatKernel;

    public ImageTransformKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  Bilinear resize (RGBA int pixels)
    // ──────────────────────────────────────────────

    /// <summary>
    /// GPU bilinear resize for packed RGBA int pixels.
    /// One thread per output pixel. Stays entirely on GPU.
    /// params: [srcW, srcH, dstW, dstH]
    /// </summary>
    private static void ResizeImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int srcW = p[0]; int srcH = p[1]; int dstW = p[2]; int dstH = p[3];

        int dy = idx / dstW;
        int dx = idx % dstW;

        float srcX = (dx + 0.5f) * srcW / dstW - 0.5f;
        float srcY = (dy + 0.5f) * srcH / dstH - 0.5f;

        int x0 = (int)srcX; int x1 = x0 + 1;
        int y0 = (int)srcY; int y1 = y0 + 1;
        float fx = srcX - x0; float fy = srcY - y0;

        x0 = x0 < 0 ? 0 : (x0 >= srcW ? srcW - 1 : x0);
        x1 = x1 < 0 ? 0 : (x1 >= srcW ? srcW - 1 : x1);
        y0 = y0 < 0 ? 0 : (y0 >= srcH ? srcH - 1 : y0);
        y1 = y1 < 0 ? 0 : (y1 >= srcH ? srcH - 1 : y1);

        int p00 = input[y0 * srcW + x0];
        int p10 = input[y0 * srcW + x1];
        int p01 = input[y1 * srcW + x0];
        int p11 = input[y1 * srcW + x1];

        int r = (int)(Lerp(Lerp(p00 & 0xFF, p10 & 0xFF, fx), Lerp(p01 & 0xFF, p11 & 0xFF, fx), fy) + 0.5f);
        int g = (int)(Lerp(Lerp((p00 >> 8) & 0xFF, (p10 >> 8) & 0xFF, fx), Lerp((p01 >> 8) & 0xFF, (p11 >> 8) & 0xFF, fx), fy) + 0.5f);
        int b = (int)(Lerp(Lerp((p00 >> 16) & 0xFF, (p10 >> 16) & 0xFF, fx), Lerp((p01 >> 16) & 0xFF, (p11 >> 16) & 0xFF, fx), fy) + 0.5f);
        int a = (int)(Lerp(Lerp((p00 >> 24) & 0xFF, (p10 >> 24) & 0xFF, fx), Lerp((p01 >> 24) & 0xFF, (p11 >> 24) & 0xFF, fx), fy) + 0.5f);

        output[idx] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24);
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    public void Resize(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int srcW, int srcH, int dstW, int dstH)
    {
        var paramsData = new int[] { srcW, srcH, dstW, dstH };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _resizeKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ResizeImpl);
        _resizeKernel(dstW * dstH, input, output, paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Bilinear resize (NCHW float tensor)
    // ──────────────────────────────────────────────

    /// <summary>
    /// GPU bilinear resize for NCHW float tensors.
    /// One thread per output element. Handles multi-channel.
    /// params: [channels, srcH, srcW, dstH, dstW]
    /// </summary>
    private static void ResizeFloatImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[0]; int srcH = p[1]; int srcW = p[2]; int dstH = p[3]; int dstW = p[4];

        int dstHW = dstH * dstW;
        int c = idx / dstHW;
        int rem = idx % dstHW;
        int dy = rem / dstW;
        int dx = rem % dstW;

        float srcX = (dx + 0.5f) * srcW / dstW - 0.5f;
        float srcY = (dy + 0.5f) * srcH / dstH - 0.5f;

        int x0 = (int)srcX; int x1 = x0 + 1;
        int y0 = (int)srcY; int y1 = y0 + 1;
        float fx = srcX - x0; float fy = srcY - y0;

        x0 = x0 < 0 ? 0 : (x0 >= srcW ? srcW - 1 : x0);
        x1 = x1 < 0 ? 0 : (x1 >= srcW ? srcW - 1 : x1);
        y0 = y0 < 0 ? 0 : (y0 >= srcH ? srcH - 1 : y0);
        y1 = y1 < 0 ? 0 : (y1 >= srcH ? srcH - 1 : y1);

        int srcHW = srcH * srcW;
        int channelOffset = c * srcHW;
        float v00 = input[channelOffset + y0 * srcW + x0];
        float v10 = input[channelOffset + y0 * srcW + x1];
        float v01 = input[channelOffset + y1 * srcW + x0];
        float v11 = input[channelOffset + y1 * srcW + x1];

        output[idx] = Lerp(Lerp(v00, v10, fx), Lerp(v01, v11, fx), fy);
    }

    public void ResizeFloat(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int channels, int srcH, int srcW, int dstH, int dstW)
    {
        var paramsData = new int[] { channels, srcH, srcW, dstH, dstW };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _resizeFloatKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ResizeFloatImpl);
        _resizeFloatKernel(channels * dstH * dstW, input, output, paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Center crop
    // ──────────────────────────────────────────────

    /// <summary>
    /// GPU center crop for packed RGBA int pixels.
    /// params: [srcW, srcH, cropW, cropH]
    /// </summary>
    private static void CenterCropImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int srcW = p[0]; int srcH = p[1]; int cropW = p[2]; int cropH = p[3];
        int startX = (srcW - cropW) / 2;
        int startY = (srcH - cropH) / 2;

        int dy = idx / cropW;
        int dx = idx % cropW;
        int srcX = startX + dx;
        int srcY = startY + dy;

        if (srcX >= 0 && srcX < srcW && srcY >= 0 && srcY < srcH)
            output[idx] = input[srcY * srcW + srcX];
        else
            output[idx] = 0;
    }

    public void CenterCrop(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int srcW, int srcH, int cropW, int cropH)
    {
        var paramsData = new int[] { srcW, srcH, cropW, cropH };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _centerCropKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(CenterCropImpl);
        _centerCropKernel(cropW * cropH, input, output, paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Horizontal flip
    // ──────────────────────────────────────────────

    private static void FlipHImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int width = p[0];
        int y = idx / width;
        int x = idx % width;
        output[y * width + (width - 1 - x)] = input[idx];
    }

    public void FlipHorizontal(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        var paramsData = new int[] { width };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _flipHKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FlipHImpl);
        _flipHKernel(width * height, input, output, paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Vertical flip
    // ──────────────────────────────────────────────

    private static void FlipVImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int width = p[0]; int height = p[1];
        int y = idx / width;
        int x = idx % width;
        output[(height - 1 - y) * width + x] = input[idx];
    }

    public void FlipVertical(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int width, int height)
    {
        var paramsData = new int[] { width, height };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _flipVKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(FlipVImpl);
        _flipVKernel(width * height, input, output, paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Constant padding
    // ──────────────────────────────────────────────

    /// <summary>
    /// GPU constant pad for packed RGBA int pixels.
    /// params: [srcW, srcH, padTop, padLeft, dstW, dstH, padColor]
    /// </summary>
    private static void PadConstantImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int srcW = p[0]; int srcH = p[1];
        int padTop = p[2]; int padLeft = p[3];
        int dstW = p[4]; int padColor = p[6];

        int dy = idx / dstW;
        int dx = idx % dstW;
        int srcY = dy - padTop;
        int srcX = dx - padLeft;

        if (srcX >= 0 && srcX < srcW && srcY >= 0 && srcY < srcH)
            output[idx] = input[srcY * srcW + srcX];
        else
            output[idx] = padColor;
    }

    public void PadConstant(
        ArrayView1D<int, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output,
        int srcW, int srcH,
        int padTop, int padBottom, int padLeft, int padRight,
        int padColor = unchecked((int)0xFF000000)) // Default: transparent black
    {
        int dstW = srcW + padLeft + padRight;
        int dstH = srcH + padTop + padBottom;

        var paramsData = new int[] { srcW, srcH, padTop, padLeft, dstW, dstH, padColor };
        using var paramsBuf = _accelerator.Allocate1D(paramsData);

        _padKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(PadConstantImpl);
        _padKernel(dstW * dstH, input, output, paramsBuf.View);
    }
}
