using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Result from super-resolution — upscaled image as RGBA pixels.
/// </summary>
public record SuperResResult(int[] RgbaPixels, int Width, int Height, int UpscaleFactor);

/// <summary>
/// High-level super-resolution pipeline.
/// ESPCN model: input Y luminance channel [1, 1, H, W] → output [1, 1, H*3, W*3].
///
/// Usage:
///   var pipeline = new SuperResolutionPipeline(session, accelerator);
///   var result = await pipeline.UpscaleAsync(rgbaPixels, width, height);
/// </summary>
public class SuperResolutionPipeline : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly int _upscaleFactor;

    public SuperResolutionPipeline(InferenceSession session, Accelerator accelerator,
        int upscaleFactor = 3)
    {
        _session = session;
        _accelerator = accelerator;
        _upscaleFactor = upscaleFactor;
    }

    /// <summary>
    /// Upscale an RGBA image using the super-resolution model.
    /// </summary>
    public async Task<SuperResResult> UpscaleAsync(int[] rgbaPixels, int width, int height)
    {
        // Extract Y luminance channel from RGBA
        // Y = 0.299R + 0.587G + 0.114B (BT.601)
        var yChannel = new float[height * width];
        for (int i = 0; i < width * height; i++)
        {
            int pixel = rgbaPixels[i];
            float r = (pixel & 0xFF) / 255f;
            float g = ((pixel >> 8) & 0xFF) / 255f;
            float b = ((pixel >> 16) & 0xFF) / 255f;
            yChannel[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }

        // Upload as [1, 1, H, W]
        using var inputBuf = _accelerator.Allocate1D(yChannel);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 1, height, width });

        // Run inference
        var outputs = _session.Run(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        await _accelerator.SynchronizeAsync();

        // Read upscaled Y channel
        var output = outputs[_session.OutputNames[0]];
        int outH = height * _upscaleFactor;
        int outW = width * _upscaleFactor;
        int outSize = outH * outW;

        using var readBuf = _accelerator.Allocate1D<float>(outSize);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, Math.Min(outSize, output.ElementCount)),
            readBuf.View.SubView(0, Math.Min(outSize, output.ElementCount)),
            Math.Min(outSize, output.ElementCount), 1f);
        await _accelerator.SynchronizeAsync();
        var upscaledY = await readBuf.CopyToHostAsync<float>(0, outSize);

        // Convert Y back to grayscale RGBA (simple version — full version would merge with bicubic Cb/Cr)
        var result = new int[outH * outW];
        for (int i = 0; i < outH * outW; i++)
        {
            int gray = Math.Clamp((int)(upscaledY[i] * 255f + 0.5f), 0, 255);
            result[i] = gray | (gray << 8) | (gray << 16) | (0xFF << 24);
        }

        return new SuperResResult(result, outW, outH, _upscaleFactor);
    }

    public void Dispose() { }
}
