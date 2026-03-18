using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU image preprocessing for neural network inference.
/// Converts RGBA uint8 pixels to normalized NCHW float tensor.
///
/// Operations (fused into single kernel for efficiency):
/// 1. Bilinear resize from source dimensions to model input dimensions
/// 2. Convert uint8 [0,255] → float [0,1]
/// 3. Normalize with per-channel mean and std
/// 4. Convert HWC (pixel order) → NCHW (channel-first)
///
/// This replaces 4 separate kernels with one GPU pass — no intermediate buffers.
/// </summary>
public class ImagePreprocessKernel
{
    private readonly Accelerator _accelerator;

    // params: [srcW, srcH, dstW, dstH] + mean[3] + invStd[3] packed as float bits in int buffer
    private Action<Index1D,
        ArrayView1D<int, Stride1D.Dense>,    // RGBA pixels [srcH * srcW] as packed uint32
        ArrayView1D<float, Stride1D.Dense>,  // output NCHW [3, dstH, dstW]
        ArrayView1D<float, Stride1D.Dense>>? // params [10]: srcW, srcH, dstW, dstH, meanR, meanG, meanB, invStdR, invStdG, invStdB
        _preprocessKernel;

    public ImagePreprocessKernel(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>
    /// Fused preprocess: bilinear resize + normalize + HWC→NCHW.
    /// One thread per output element (3 * dstH * dstW total).
    /// </summary>
    private static void PreprocessImpl(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        int srcW = (int)p[0]; int srcH = (int)p[1];
        int dstW = (int)p[2]; int dstH = (int)p[3];

        // Decompose output index → (channel, y, x) in NCHW layout
        int dstHW = dstH * dstW;
        int c = idx / dstHW;
        int rem = idx % dstHW;
        int dy = rem / dstW;
        int dx = rem % dstW;

        // Bilinear sample coordinates (half-pixel centered)
        float fy = ((dy + 0.5f) * srcH / dstH) - 0.5f;
        float fx = ((dx + 0.5f) * srcW / dstW) - 0.5f;

        int y0 = (int)fy; int y1 = y0 + 1;
        int x0 = (int)fx; int x1 = x0 + 1;
        float ty = fy - y0; float tx = fx - x0;

        // Clamp to source bounds
        if (y0 < 0) y0 = 0; if (y1 >= srcH) y1 = srcH - 1;
        if (x0 < 0) x0 = 0; if (x1 >= srcW) x1 = srcW - 1;

        // Read 4 RGBA pixels and extract channel c (R=0, G=1, B=2)
        int shift = c * 8; // R=bits 0-7, G=bits 8-15, B=bits 16-23
        float v00 = ((rgba[y0 * srcW + x0] >> shift) & 0xFF) / 255f;
        float v01 = ((rgba[y0 * srcW + x1] >> shift) & 0xFF) / 255f;
        float v10 = ((rgba[y1 * srcW + x0] >> shift) & 0xFF) / 255f;
        float v11 = ((rgba[y1 * srcW + x1] >> shift) & 0xFF) / 255f;

        // Bilinear interpolation
        float pixel = v00 * (1f - ty) * (1f - tx) + v01 * (1f - ty) * tx
                    + v10 * ty * (1f - tx) + v11 * ty * tx;

        // Normalize: (pixel - mean) / std = (pixel - mean) * invStd
        float mean = p[4 + c];     // meanR, meanG, meanB
        float invStd = p[7 + c];   // invStdR, invStdG, invStdB
        output[idx] = (pixel - mean) * invStd;
    }

    private MemoryBuffer1D<float, Stride1D.Dense>? _paramsBuf;

    /// <summary>
    /// Preprocess RGBA image for model inference.
    /// Input: RGBA pixels as packed int32 [srcH * srcW].
    /// Output: NCHW float [3, dstH, dstW].
    /// Uses ImageNet default normalization unless overridden.
    /// </summary>
    public void Forward(
        ArrayView1D<int, Stride1D.Dense> rgba,
        ArrayView1D<float, Stride1D.Dense> output,
        int srcW, int srcH, int dstW, int dstH,
        float[]? mean = null, float[]? std = null)
    {
        EnsureLoaded();

        mean ??= new[] { 0.485f, 0.456f, 0.406f }; // ImageNet RGB mean
        std ??= new[] { 0.229f, 0.224f, 0.225f };   // ImageNet RGB std

        _paramsBuf ??= _accelerator.Allocate1D<float>(10);
        _paramsBuf.CopyFromCPU(new float[] {
            srcW, srcH, dstW, dstH,
            mean[0], mean[1], mean[2],
            1f / std[0], 1f / std[1], 1f / std[2]
        });

        int totalOutput = 3 * dstH * dstW;
        _preprocessKernel!(totalOutput, rgba, output, _paramsBuf.View);
    }

    private void EnsureLoaded()
    {
        _preprocessKernel ??= _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(PreprocessImpl);
    }
}
