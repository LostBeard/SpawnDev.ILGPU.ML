using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernels for tensor layout conversions.
/// Replaces CPU-side TensorLayout utility with GPU-native execution.
/// All operations keep data on GPU.
/// </summary>
public class TensorLayoutKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _nchwToNhwcKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _nhwcToNchwKernel;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _interleavedToPlanarKernel;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>? _planarToInterleavedKernel;

    private MemoryBuffer1D<int, Stride1D.Dense>? _paramsBuf;

    public TensorLayoutKernel(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  NCHW → NHWC
    // ──────────────────────────────────────────────

    /// <summary>
    /// Convert NCHW [C, H, W] float tensor to NHWC [H, W, C] on GPU.
    /// One thread per element.
    /// params: [channels, height, width]
    /// </summary>
    private static void NCHWToNHWCImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<float, Stride1D.Dense> nhwc,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[0]; int H = p[1]; int W = p[2];
        int HW = H * W;

        // Input index in NCHW: c * H * W + y * W + x
        int c = idx / HW;
        int rem = idx % HW;
        int y = rem / W;
        int x = rem % W;

        // Output index in NHWC: y * W * C + x * C + c
        nhwc[y * W * C + x * C + c] = nchw[idx];
    }

    public void NCHWToNHWC(
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<float, Stride1D.Dense> nhwc,
        int channels, int height, int width)
    {
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        _paramsBuf ??= _accelerator.Allocate1D<int>(3);
        var paramsData = new int[] { channels, height, width };
        _paramsBuf.CopyFromCPU(paramsData);

        _nchwToNhwcKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(NCHWToNHWCImpl);
        _nchwToNhwcKernel(channels * height * width, nchw, nhwc, _paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  NHWC → NCHW
    // ──────────────────────────────────────────────

    private static void NHWCToNCHWImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> nhwc,
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int C = p[0]; int H = p[1]; int W = p[2];

        // Input index in NHWC: y * W * C + x * C + c
        int y = idx / (W * C);
        int rem = idx % (W * C);
        int x = rem / C;
        int c = rem % C;

        // Output index in NCHW: c * H * W + y * W + x
        nchw[c * H * W + y * W + x] = nhwc[idx];
    }

    public void NHWCToNCHW(
        ArrayView1D<float, Stride1D.Dense> nhwc,
        ArrayView1D<float, Stride1D.Dense> nchw,
        int channels, int height, int width)
    {
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        _paramsBuf ??= _accelerator.Allocate1D<int>(3);
        var paramsData = new int[] { channels, height, width };
        _paramsBuf.CopyFromCPU(paramsData);

        _nhwcToNchwKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(NHWCToNCHWImpl);
        _nhwcToNchwKernel(channels * height * width, nhwc, nchw, _paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Interleaved RGBA int → Planar float [3, H, W] in [0, 1]
    // ──────────────────────────────────────────────

    /// <summary>
    /// Convert packed RGBA int pixels to planar NCHW float [3, H, W] in [0, 1].
    /// Drops alpha channel. One thread per output element.
    /// params: [width, height]
    /// </summary>
    private static void InterleavedToPlanarImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> planar,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int W = p[0]; int H = p[1];
        int HW = H * W;

        int c = idx / HW;
        int pixelIdx = idx % HW;

        int pixel = rgba[pixelIdx];
        int shift = c * 8; // R=0, G=8, B=16
        float value = ((pixel >> shift) & 0xFF) / 255f;
        planar[idx] = value;
    }

    public void InterleavedRGBAToPlanarFloat(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> planar,
        int width, int height)
    {
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        _paramsBuf ??= _accelerator.Allocate1D<int>(3);
        var paramsData = new int[] { width, height, 0 };
        _paramsBuf.CopyFromCPU(paramsData);

        _interleavedToPlanarKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(InterleavedToPlanarImpl);
        _interleavedToPlanarKernel(3 * width * height, rgba, planar, _paramsBuf.View);
    }

    // ──────────────────────────────────────────────
    //  Planar float [3, H, W] → Interleaved RGBA int
    // ──────────────────────────────────────────────

    /// <summary>
    /// Convert planar NCHW float [3, H, W] in [0, 1] to packed RGBA int pixels.
    /// Alpha = 255. One thread per pixel.
    /// params: [width, height]
    /// </summary>
    private static void PlanarToInterleavedImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> planar,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int W = p[0]; int H = p[1];
        int HW = H * W;

        float rf = planar[0 * HW + idx];
        float gf = planar[1 * HW + idx];
        float bf = planar[2 * HW + idx];

        int r = (int)(rf * 255f + 0.5f);
        int g = (int)(gf * 255f + 0.5f);
        int b = (int)(bf * 255f + 0.5f);
        r = r < 0 ? 0 : (r > 255 ? 255 : r);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);

        rgba[idx] = r | (g << 8) | (b << 16) | (255 << 24);
    }

    public void PlanarFloatToInterleavedRGBA(
        ArrayView1D<float, Stride1D.Dense> planar,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int width, int height)
    {
        // Persistent buffer avoids use-after-dispose on async backends (WebGPU, Wasm)
        _paramsBuf ??= _accelerator.Allocate1D<int>(3);
        var paramsData = new int[] { width, height, 0 };
        _paramsBuf.CopyFromCPU(paramsData);

        _planarToInterleavedKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(PlanarToInterleavedImpl);
        _planarToInterleavedKernel(width * height, planar, rgba, _paramsBuf.View);
    }
}
