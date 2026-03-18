using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from style transfer — styled RGBA pixels.
/// </summary>
public record StyleResult(int[] RgbaPixels, int Width, int Height);

/// <summary>
/// High-level neural style transfer pipeline.
/// Input: RGBA image. Output: styled RGBA image (same dimensions).
///
/// Style models expect [0, 255] float input (NOT normalized) and output [0, 255] floats.
///
/// Usage:
///   var pipeline = new StyleTransferPipeline(session, accelerator);
///   var result = await pipeline.TransferAsync(rgbaPixels, width, height);
///   // result.RgbaPixels is the styled image
/// </summary>
public class StyleTransferPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;

    public StyleTransferPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
    }

    /// <summary>
    /// Apply neural style transfer to an RGBA image.
    /// </summary>
    public async Task<StyleResult> TransferAsync(int[] rgbaPixels, int width, int height)
    {
        // Convert RGBA int[] to [1, 3, H, W] float tensor in [0, 255] range
        // Style models use RGB [0,255] WITHOUT ImageNet normalization
        var inputData = new float[3 * height * width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int pixel = rgbaPixels[y * width + x];
                int r = pixel & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = (pixel >> 16) & 0xFF;
                inputData[0 * height * width + y * width + x] = r; // R channel
                inputData[1 * height * width + y * width + x] = g; // G channel
                inputData[2 * height * width + y * width + x] = b; // B channel
            }

        using var inputBuf = _accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, height, width });

        // Run style transfer
        var outputs = _session.Run(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        // Read output [1, 3, H, W] → clip to [0, 255] → pack to RGBA int[]
        var output = outputs[_session.OutputNames[0]];
        int outH = output.Shape.Length >= 4 ? output.Shape[2] : height;
        int outW = output.Shape.Length >= 4 ? output.Shape[3] : width;
        int outSize = 3 * outH * outW;

        using var readBuf = _accelerator.Allocate1D<float>(outSize);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, outSize), readBuf.View, outSize, 1f);
        await _accelerator.SynchronizeAsync();
        var raw = await readBuf.CopyToHostAsync<float>(0, outSize);

        // Pack NCHW float [0,255] → RGBA int[]
        var result = new int[outH * outW];
        for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
            {
                int r = Clamp255(raw[0 * outH * outW + y * outW + x]);
                int g = Clamp255(raw[1 * outH * outW + y * outW + x]);
                int b = Clamp255(raw[2 * outH * outW + y * outW + x]);
                result[y * outW + x] = r | (g << 8) | (b << 16) | (0xFF << 24);
            }

        return new StyleResult(result, outW, outH);
    }

    private static int Clamp255(float v)
    {
        int i = (int)(v + 0.5f);
        return i < 0 ? 0 : (i > 255 ? 255 : i);
    }

    public void Dispose() { }
}
