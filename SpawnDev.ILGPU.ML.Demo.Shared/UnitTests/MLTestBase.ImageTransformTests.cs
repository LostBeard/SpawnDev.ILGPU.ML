using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ImageResize_Bilinear_2x2to4x4() => await RunTest(async accelerator =>
    {
        // 2x2 input: red, green, blue, white (packed RGBA as int)
        int srcW = 2, srcH = 2;
        var input = new int[]
        {
            unchecked((int)0xFF0000FF), unchecked((int)0xFF00FF00),  // red, green
            unchecked((int)0xFFFF0000), unchecked((int)0xFFFFFFFF)   // blue, white
        };

        int dstW = 4, dstH = 4;
        var expected = new int[dstW * dstH];

        // CPU bilinear reference
        for (int dy = 0; dy < dstH; dy++)
        for (int dx = 0; dx < dstW; dx++)
        {
            float srcX = (dx + 0.5f) * srcW / dstW - 0.5f;
            float srcY = (dy + 0.5f) * srcH / dstH - 0.5f;
            int x0 = Math.Clamp((int)srcX, 0, srcW - 1);
            int x1 = Math.Clamp(x0 + 1, 0, srcW - 1);
            int y0 = Math.Clamp((int)srcY, 0, srcH - 1);
            int y1 = Math.Clamp(y0 + 1, 0, srcH - 1);
            float fx = srcX - (int)srcX; float fy = srcY - (int)srcY;
            if (fx < 0) fx = 0; if (fy < 0) fy = 0;

            int p00 = input[y0 * srcW + x0], p10 = input[y0 * srcW + x1];
            int p01 = input[y1 * srcW + x0], p11 = input[y1 * srcW + x1];

            int r = (int)(Lerp(Lerp(p00 & 0xFF, p10 & 0xFF, fx), Lerp(p01 & 0xFF, p11 & 0xFF, fx), fy) + 0.5f);
            int g = (int)(Lerp(Lerp((p00 >> 8) & 0xFF, (p10 >> 8) & 0xFF, fx), Lerp((p01 >> 8) & 0xFF, (p11 >> 8) & 0xFF, fx), fy) + 0.5f);
            int b = (int)(Lerp(Lerp((p00 >> 16) & 0xFF, (p10 >> 16) & 0xFF, fx), Lerp((p01 >> 16) & 0xFF, (p11 >> 16) & 0xFF, fx), fy) + 0.5f);
            int a = (int)(Lerp(Lerp((p00 >> 24) & 0xFF, (p10 >> 24) & 0xFF, fx), Lerp((p01 >> 24) & 0xFF, (p11 >> 24) & 0xFF, fx), fy) + 0.5f);
            expected[dy * dstW + dx] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24);
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<int>(dstW * dstH);

        var transform = new ImageTransformKernel(accelerator);
        transform.Resize(inBuf.View, outBuf.View, srcW, srcH, dstW, dstH);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<int>(0, dstW * dstH);

        // Compare each pixel channel
        for (int i = 0; i < expected.Length; i++)
        {
            int er = expected[i] & 0xFF, ar = actual[i] & 0xFF;
            int eg = (expected[i] >> 8) & 0xFF, ag = (actual[i] >> 8) & 0xFF;
            int eb = (expected[i] >> 16) & 0xFF, ab = (actual[i] >> 16) & 0xFF;
            if (Math.Abs(er - ar) > 1 || Math.Abs(eg - ag) > 1 || Math.Abs(eb - ab) > 1)
                throw new Exception($"Pixel [{i}] mismatch: expected=({er},{eg},{eb}), actual=({ar},{ag},{ab})");
        }
    });

    [TestMethod]
    public async Task ColorConversion_RGBtoGray() => await RunTest(async accelerator =>
    {
        // 4 pixels: red, green, blue, white (packed RGBA as int)
        int count = 4;
        var input = new int[]
        {
            unchecked((int)0xFF0000FF),  // R=255, G=0,   B=0
            unchecked((int)0xFF00FF00),  // R=0,   G=255, B=0
            unchecked((int)0xFFFF0000),  // R=0,   G=0,   B=255
            unchecked((int)0xFFFFFFFF)   // R=255, G=255, B=255
        };

        // CPU reference: BT.709 luminance Y = 0.2126*R + 0.7152*G + 0.0722*B
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float r = (input[i] & 0xFF) / 255f;
            float g = ((input[i] >> 8) & 0xFF) / 255f;
            float b = ((input[i] >> 16) & 0xFF) / 255f;
            expected[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);

        var color = new ColorConversionKernel(accelerator);
        color.RGBAToGrayscale(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, count);
        AssertClose(expected, actual, 1e-3f, "RGB→Gray: ");
    });

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
