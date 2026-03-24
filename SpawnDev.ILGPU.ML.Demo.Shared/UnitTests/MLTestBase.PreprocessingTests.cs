using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // ──────────────────────────────────────────────────────────────
    //  Preprocessing tests: catch bilinear/normalization bugs
    //  BEFORE they corrupt model input.
    // ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Verify GPU preprocessing matches ONNX Runtime reference input.
    /// Uses style-mosaic reference (ForwardRaw = [0,255] range, no ImageNet norm).
    /// This catches the class of bug that cost us a full day (bilinear floor truncation).
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task Preprocessing_ForwardRaw_MatchesReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load cat image
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);

        // Load reference preprocessed tensor (ONNX Runtime's input)
        var refBytes = await http.GetByteArrayAsync("references/style-mosaic/cat_input_nchw.bin");
        var refInput = new float[refBytes.Length / 4];
        Buffer.BlockCopy(refBytes, 0, refInput, 0, refBytes.Length);

        // GPU preprocess: RGBA → NCHW [0, 255] (ForwardRaw)
        int dstW = 224, dstH = 224;
        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var nchwBuf = accelerator.Allocate1D<float>(3 * dstH * dstW);
        var preprocess = new ImagePreprocessKernel(accelerator);
        preprocess.ForwardRaw(rgbaBuf.View, nchwBuf.View, width, height, dstW, dstH);
        await accelerator.SynchronizeAsync();

        var actual = await nchwBuf.CopyToHostAsync<float>(0, 3 * dstH * dstW);

        // Compare first 100 values
        int compareCount = Math.Min(100, actual.Length);
        float maxErr = 0;
        double sumErr = 0;
        for (int i = 0; i < compareCount; i++)
        {
            float err = MathF.Abs(actual[i] - refInput[i]);
            sumErr += err;
            if (err > maxErr) maxErr = err;
        }
        float meanErr = (float)(sumErr / compareCount);

        // Tolerance: 1.0 per element (bilinear resize can differ slightly between implementations)
        if (meanErr > 1.0f)
            throw new Exception($"Preprocessing diverges: mean error {meanErr:F4} > 1.0. " +
                $"First actual=[{actual[0]:F2},{actual[1]:F2},{actual[2]:F2}] " +
                $"ref=[{refInput[0]:F2},{refInput[1]:F2},{refInput[2]:F2}]");
    });

    /// <summary>
    /// Verify ImageNet preprocessing (classification models).
    /// Uses SqueezeNet reference input.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task Preprocessing_ImageNet_MatchesReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Load cat image
        var binData = await http.GetByteArrayAsync("samples/cat_rgba.bin");
        int width = BitConverter.ToInt32(binData, 0);
        int height = BitConverter.ToInt32(binData, 4);
        var pixels = new int[width * height];
        Buffer.BlockCopy(binData, 8, pixels, 0, width * height * 4);

        // Load SqueezeNet reference input
        var refBytes = await http.GetByteArrayAsync("references/squeezenet/cat_input_nchw.bin");
        var refInput = new float[refBytes.Length / 4];
        Buffer.BlockCopy(refBytes, 0, refInput, 0, refBytes.Length);

        // GPU preprocess: RGBA → NCHW with ImageNet normalization
        int dstW = 224, dstH = 224;
        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var nchwBuf = accelerator.Allocate1D<float>(3 * dstH * dstW);
        var preprocess = new ImagePreprocessKernel(accelerator);
        preprocess.Forward(rgbaBuf.View, nchwBuf.View, width, height, dstW, dstH);
        await accelerator.SynchronizeAsync();

        var actual = await nchwBuf.CopyToHostAsync<float>(0, 3 * dstH * dstW);

        int compareCount = Math.Min(100, actual.Length);
        float maxErr = 0;
        double sumErr = 0;
        for (int i = 0; i < compareCount; i++)
        {
            float err = MathF.Abs(actual[i] - refInput[i]);
            sumErr += err;
            if (err > maxErr) maxErr = err;
        }
        float meanErr = (float)(sumErr / compareCount);

        // Note: reference input may use different preprocessing (raw [0,255] vs ImageNet-normalized).
        // If reference is raw pixels (values > 1.0), compare our normalized output to manual normalization of ref.
        bool refIsRaw = refInput[0] > 1.0f;
        if (refIsRaw)
        {
            // Reference is raw [0,255] — apply ImageNet normalization for comparison
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };
            int hw = dstH * dstW;
            sumErr = 0; maxErr = 0;
            for (int i = 0; i < compareCount; i++)
            {
                int c = i / hw;
                float refNorm = (refInput[i] / 255f - mean[c]) / std[c];
                float err = MathF.Abs(actual[i] - refNorm);
                sumErr += err;
                if (err > maxErr) maxErr = err;
            }
            meanErr = (float)(sumErr / compareCount);
        }

        if (meanErr > 0.1f)
            throw new Exception($"ImageNet preprocessing diverges: mean error {meanErr:F4} > 0.1. " +
                $"First actual=[{actual[0]:F4},{actual[1]:F4},{actual[2]:F4}] " +
                $"ref=[{refInput[0]:F4},{refInput[1]:F4},{refInput[2]:F4}]" +
                (refIsRaw ? " (ref was raw pixels, manually normalized for comparison)" : ""));
    });
}
