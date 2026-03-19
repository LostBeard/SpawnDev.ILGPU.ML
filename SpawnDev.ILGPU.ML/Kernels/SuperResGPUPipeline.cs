using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU-native super resolution pre/postprocessing pipeline.
/// Replaces CPU-side SuperResPostProcessor with GPU kernels.
/// Full pipeline stays on GPU: RGBA → YCbCr split → Y to model →
/// Cb/Cr resize → merge → RGBA. Zero CPU readback.
/// </summary>
public class SuperResGPUPipeline
{
    private readonly Accelerator _accelerator;
    private readonly ColorConversionKernel _colorKernel;
    private readonly ImageTransformKernel _transformKernel;

    public SuperResGPUPipeline(Accelerator accelerator)
    {
        _accelerator = accelerator;
        _colorKernel = new ColorConversionKernel(accelerator);
        _transformKernel = new ImageTransformKernel(accelerator);
    }

    /// <summary>
    /// Prepare ESPCN input: extract Y channel from RGBA pixels on GPU.
    /// Returns GPU buffers for Y (model input), Cb, Cr (for postprocessing).
    /// Everything stays on GPU.
    /// </summary>
    public (MemoryBuffer1D<float, Stride1D.Dense> Y,
            MemoryBuffer1D<float, Stride1D.Dense> Cb,
            MemoryBuffer1D<float, Stride1D.Dense> Cr) PrepareInput(
        ArrayView1D<int, Stride1D.Dense> rgbaPixels,
        int width, int height)
    {
        int pixelCount = width * height;

        // Allocate Y, Cb, Cr buffers on GPU
        var yBuf = _accelerator.Allocate1D<float>(pixelCount);
        var cbBuf = _accelerator.Allocate1D<float>(pixelCount);
        var crBuf = _accelerator.Allocate1D<float>(pixelCount);

        // RGBA → YCbCr on GPU
        _colorKernel.RGBAToYCbCr(rgbaPixels, yBuf.View, cbBuf.View, crBuf.View, pixelCount);

        // Normalize Y to [0, 1] for model input
        NormalizeY(yBuf.View, pixelCount);

        return (yBuf, cbBuf, crBuf);
    }

    /// <summary>
    /// Merge ESPCN output (upscaled Y) with resized Cb/Cr to produce final RGBA.
    /// All on GPU — no CPU readback until final display.
    /// </summary>
    /// <param name="yUpscaled">Model output: upscaled Y channel [outH * outW] in [0, 1]</param>
    /// <param name="cb">Original Cb at input resolution [inH * inW]</param>
    /// <param name="cr">Original Cr at input resolution [inH * inW]</param>
    /// <param name="inW">Input width</param>
    /// <param name="inH">Input height</param>
    /// <param name="outW">Output width (3x input for ESPCN)</param>
    /// <param name="outH">Output height (3x input for ESPCN)</param>
    /// <returns>RGBA output buffer on GPU</returns>
    public MemoryBuffer1D<int, Stride1D.Dense> MergeOutput(
        ArrayView1D<float, Stride1D.Dense> yUpscaled,
        ArrayView1D<float, Stride1D.Dense> cb,
        ArrayView1D<float, Stride1D.Dense> cr,
        int inW, int inH, int outW, int outH)
    {
        int outPixels = outW * outH;

        // Denormalize Y: [0, 1] → [0, 255]
        using var yDenorm = _accelerator.Allocate1D<float>(outPixels);
        DenormalizeY(yUpscaled, yDenorm.View, outPixels);

        // Resize Cb and Cr to output resolution on GPU
        using var cbResized = _accelerator.Allocate1D<float>(outPixels);
        using var crResized = _accelerator.Allocate1D<float>(outPixels);
        _transformKernel.ResizeFloat(cb, cbResized.View, 1, inH, inW, outH, outW);
        _transformKernel.ResizeFloat(cr, crResized.View, 1, inH, inW, outH, outW);

        // Merge YCbCr → RGBA on GPU
        var rgbaOut = _accelerator.Allocate1D<int>(outPixels);
        _colorKernel.YCbCrToRGBA(yDenorm.View, cbResized.View, crResized.View, rgbaOut.View, outPixels);

        return rgbaOut;
    }

    // Y normalization: [0, 255] → [0, 1]
    private void NormalizeY(ArrayView1D<float, Stride1D.Dense> y, int count)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                d[idx] /= 255f;
            });
        kernel((Index1D)count, y);
    }

    // Y denormalization: [0, 1] → [0, 255]
    private void DenormalizeY(ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int count)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> inp, ArrayView1D<float, Stride1D.Dense> outp) =>
            {
                float v = inp[idx] * 255f;
                outp[idx] = v < 0f ? 0f : (v > 255f ? 255f : v);
            });
        kernel((Index1D)count, input, output);
    }
}
