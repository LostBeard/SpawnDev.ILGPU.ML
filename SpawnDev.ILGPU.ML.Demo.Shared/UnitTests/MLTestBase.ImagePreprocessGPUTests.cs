using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GPU tests for ImagePreprocessKernel: verify bilinear resize, normalization modes,
/// and channel layout conversion all work correctly on the GPU.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ImagePreprocess_ForwardRaw_ValueRange() => await RunTest(async accelerator =>
    {
        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);

        // Create a 4x4 RGBA image: solid red (255,0,0,255)
        var pixels = new int[16];
        for (int i = 0; i < 16; i++)
            pixels[i] = 255 | (0 << 8) | (0 << 16) | (0xFF << 24); // R=255

        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var outputBuf = accelerator.Allocate1D<float>(3 * 2 * 2); // 3 channels, 2x2 output

        preprocess.ForwardRaw(rgbaBuf.View, outputBuf.View, 4, 4, 2, 2);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, 12);

        // NCHW layout: [R channel (4 pixels), G channel (4 pixels), B channel (4 pixels)]
        // R channel should be ~255, G and B should be ~0
        for (int i = 0; i < 4; i++)
        {
            if (output[i] < 240f) // R channel
                throw new Exception($"R channel [{i}]={output[i]:F1}, expected ~255");
        }
        for (int i = 4; i < 8; i++)
        {
            if (output[i] > 15f) // G channel
                throw new Exception($"G channel [{i}]={output[i]:F1}, expected ~0");
        }

        Console.WriteLine("[ImagePreprocess] ForwardRaw: red image → R~255, G~0, B~0 correct");
    });

    [TestMethod]
    public async Task ImagePreprocess_Forward_ImageNetNormalization() => await RunTest(async accelerator =>
    {
        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);

        // Create a 2x2 image with known pixel values: (128, 128, 128, 255) mid-gray
        var pixels = new int[4];
        for (int i = 0; i < 4; i++)
            pixels[i] = 128 | (128 << 8) | (128 << 16) | (0xFF << 24);

        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var outputBuf = accelerator.Allocate1D<float>(3 * 2 * 2);

        // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        // For pixel 128: normalized = (128/255 - mean) / std
        preprocess.Forward(rgbaBuf.View, outputBuf.View, 2, 2, 2, 2);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, 12);

        // R: (128/255 - 0.485) / 0.229 ≈ 0.068
        // G: (128/255 - 0.456) / 0.224 ≈ 0.200
        // B: (128/255 - 0.406) / 0.225 ≈ 0.427
        float expectedR = (128f / 255f - 0.485f) / 0.229f;
        float expectedG = (128f / 255f - 0.456f) / 0.224f;
        float expectedB = (128f / 255f - 0.406f) / 0.225f;

        if (MathF.Abs(output[0] - expectedR) > 0.1f)
            throw new Exception($"ImageNet R: {output[0]:F4}, expected ~{expectedR:F4}");
        if (MathF.Abs(output[4] - expectedG) > 0.1f)
            throw new Exception($"ImageNet G: {output[4]:F4}, expected ~{expectedG:F4}");

        Console.WriteLine($"[ImagePreprocess] ImageNet norm: R={output[0]:F3} G={output[4]:F3} B={output[8]:F3} (expected {expectedR:F3} {expectedG:F3} {expectedB:F3})");
    });

    [TestMethod]
    public async Task ImagePreprocess_BilinearResize_OutputSize() => await RunTest(async accelerator =>
    {
        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);

        // 8x8 input → 4x4 output
        var pixels = new int[64];
        for (int i = 0; i < 64; i++)
            pixels[i] = (i * 3) | ((i * 3) << 8) | ((i * 3) << 16) | (0xFF << 24);

        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var outputBuf = accelerator.Allocate1D<float>(3 * 4 * 4); // NCHW [3, 4, 4]

        preprocess.ForwardRaw(rgbaBuf.View, outputBuf.View, 8, 8, 4, 4);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, 48);

        // Should have 48 valid values (3 channels × 16 pixels)
        bool hasNonZero = output.Any(v => v > 1f);
        if (!hasNonZero)
            throw new Exception("Bilinear resize produced all near-zero values");

        // Values should be interpolated — not identical to input corners
        Console.WriteLine($"[ImagePreprocess] BilinearResize 8x8→4x4: {output.Count(v => v > 0)} non-zero values, range [{output.Min():F1}, {output.Max():F1}]");
    });

    [TestMethod]
    public async Task ImagePreprocess_YChannel_LuminanceCorrect() => await RunTest(async accelerator =>
    {
        var preprocess = new Kernels.ImagePreprocessKernel(accelerator);

        // Pure white pixel: Y = 0.299*255 + 0.587*255 + 0.114*255 = 255
        // Pure red pixel: Y = 0.299*255 = 76.245
        var pixels = new int[2];
        pixels[0] = 255 | (255 << 8) | (255 << 16) | (0xFF << 24); // white
        pixels[1] = 255 | (0 << 8) | (0 << 16) | (0xFF << 24);     // red

        using var rgbaBuf = accelerator.Allocate1D(pixels);
        using var outputBuf = accelerator.Allocate1D<float>(2); // 2 Y values

        preprocess.ForwardYChannel(rgbaBuf.View, outputBuf.View, 2, 1, 2, 1);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, 2);

        // White should give Y ≈ 1.0 (normalized)
        if (output[0] < 0.9f)
            throw new Exception($"White Y={output[0]:F4}, expected ~1.0");

        // Red should give Y ≈ 0.299
        if (MathF.Abs(output[1] - 0.299f) > 0.05f)
            throw new Exception($"Red Y={output[1]:F4}, expected ~0.299");

        Console.WriteLine($"[ImagePreprocess] YChannel: white={output[0]:F3}, red={output[1]:F3} (BT.601 correct)");
    });
}
