using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU image postprocessing — converts model output back to displayable RGBA pixels.
/// Inverse of ImagePreprocessKernel: NCHW float → packed RGBA int.
///
/// Keeps data on GPU through the entire pipeline. The result can be:
/// - Read back to CPU as int[] for display via data URL
/// - Or in the future, rendered directly via WebGPU render pass (zero-copy to canvas)
/// </summary>
public class ImagePostprocessKernel
{
    private readonly Accelerator _accelerator;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        int, int>? _nchwToRgbaKernel;

    public ImagePostprocessKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Convert NCHW float tensor to packed RGBA int pixels on GPU.
    /// Input: [3, H, W] or [1, 3, H, W] in [0, 255] range.
    /// Output: [H * W] packed RGBA (R | G<<8 | B<<16 | 0xFF<<24).
    /// One thread per pixel.
    /// </summary>
    public void NCHWToRGBA(
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int height, int width)
    {
        _nchwToRgbaKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            int, int>(NCHWToRGBAImpl);
        _nchwToRgbaKernel(height * width, nchw, rgba, height, width);
    }

    private static void NCHWToRGBAImpl(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> nchw,
        ArrayView1D<int, Stride1D.Dense> rgba,
        int H, int W)
    {
        int hw = H * W;

        // Read RGB channels from NCHW layout
        float rf = nchw[0 * hw + idx]; // R channel
        float gf = nchw[1 * hw + idx]; // G channel
        float bf = nchw[2 * hw + idx]; // B channel

        // Clamp to [0, 255] and round
        int r = (int)(rf + 0.5f); if (r < 0) r = 0; if (r > 255) r = 255;
        int g = (int)(gf + 0.5f); if (g < 0) g = 0; if (g > 255) g = 255;
        int b = (int)(bf + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255;

        // Pack as RGBA (little-endian: R in lowest byte, A=255)
        rgba[idx] = r | (g << 8) | (b << 16) | (0xFF << 24);
    }
}
