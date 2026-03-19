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
    private readonly int _modelH;
    private readonly int _modelW;

    public StyleTransferPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
        // Use the model's declared input spatial dimensions (typically 224x224)
        var inputShape = session.InputShapes.Values.FirstOrDefault() ?? new[] { 1, 3, 224, 224 };
        _modelH = inputShape.Length >= 4 ? inputShape[2] : 224;
        _modelW = inputShape.Length >= 4 ? inputShape[3] : 224;
        // Replace -1 (dynamic) with default
        if (_modelH <= 0) _modelH = 224;
        if (_modelW <= 0) _modelW = 224;
    }

    /// <summary>
    /// Apply neural style transfer to an RGBA image.
    /// </summary>
    public async Task<StyleResult> TransferAsync(int[] rgbaPixels, int width, int height)
    {
        // Use model's declared input size to match compiled graph shapes
        int modelH = _modelH, modelW = _modelW;

        // Convert RGBA int[] to [1, 3, modelH, modelW] float tensor in [0, 255] range
        // Style models use RGB [0,255] WITHOUT ImageNet normalization
        // Simple nearest-neighbor resize from source to model dimensions
        var inputData = new float[3 * modelH * modelW];
        for (int y = 0; y < modelH; y++)
            for (int x = 0; x < modelW; x++)
            {
                int sy = y * height / modelH;
                int sx = x * width / modelW;
                if (sy >= height) sy = height - 1;
                if (sx >= width) sx = width - 1;
                int pixel = rgbaPixels[sy * width + sx];
                int r = pixel & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = (pixel >> 16) & 0xFF;
                inputData[0 * modelH * modelW + y * modelW + x] = r;
                inputData[1 * modelH * modelW + y * modelW + x] = g;
                inputData[2 * modelH * modelW + y * modelW + x] = b;
            }

        using var inputBuf = _accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, modelH, modelW });

        // Use RunAsync — includes periodic flush + final SynchronizeAsync
        Console.WriteLine("[StylePipeline] Running inference via RunAsync...");
        Graph.GraphExecutor.MaxNodeCount = 0; // Full run
        var outputs = await _session.RunAsync(new Dictionary<string, Tensor>
        {
            [_session.InputNames[0]] = inputTensor
        });
        // Check if GPU device survived the inference workload
        if (_accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator webGpu)
            Console.WriteLine($"[StylePipeline] Device lost: {webGpu.IsDeviceLost}");
        Console.WriteLine("[StylePipeline] Inference done, reading output...");

        var output = outputs[_session.OutputNames[0]];
        int outH = output.Shape.Length >= 4 ? output.Shape[2] : modelH;
        int outW = output.Shape.Length >= 4 ? output.Shape[3] : modelW;
        int outSize = 3 * outH * outW;
        int actualRead = Math.Min(outSize, (int)output.Data.Length);

        using var readBuf = _accelerator.Allocate1D<float>(actualRead);
        var ew = new ElementWiseKernels(_accelerator);
        ew.Scale(output.Data.SubView(0, actualRead), readBuf.View, actualRead, 1f);
        _accelerator.Synchronize();
        Console.WriteLine("[StylePipeline] Scale done, reading back...");
        var raw2 = await readBuf.CopyToHostAsync<float>(0, actualRead);
        Console.WriteLine($"[StylePipeline] Read {raw2.Length} floats");

        // Pack NCHW float [0,255] → RGBA int[]
        var result = new int[outH * outW];
        for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
            {
                int idx = y * outW + x;
                if (idx < raw2.Length / 3)
                {
                    int r = Clamp255(raw2[0 * outH * outW + idx]);
                    int g = Clamp255(raw2[1 * outH * outW + idx]);
                    int b = Clamp255(raw2[2 * outH * outW + idx]);
                    result[idx] = r | (g << 8) | (b << 16) | (0xFF << 24);
                }
            }

        Graph.GraphExecutor.MaxNodeCount = 0; // Reset
        // For partial runs (debugging), just return what we have
        return new StyleResult(result, outW, outH);
    }

    private static int Clamp255(float v)
    {
        int i = (int)(v + 0.5f);
        return i < 0 ? 0 : (i > 255 ? 255 : i);
    }

    public void Dispose() { }
}
