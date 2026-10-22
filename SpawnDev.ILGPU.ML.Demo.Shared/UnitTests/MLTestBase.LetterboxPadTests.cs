using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Letterbox padding: with <c>padValue</c> the band outside the content rect is that constant (YOLO's grey 114/255), with the default (negative) it replicates the border pixel as before.
    /// A 4 x 2 picture into 8 x 8 (content 8 x 4, pad 2 rows above and below) with mean 0 / std 1: pad rows must read 114/255 in every channel, content rows must be untouched by the option, and the
    /// default must still repeat the first and last content rows (the behaviour the depth models rely on). Reported by SpawnGen: replicated edge pixels made YOLOv8n lose partly visible people.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task ImagePreprocess_Letterbox_PadValueFillsThePadding() => await RunTest(async accelerator =>
    {
        var px = new int[4 * 2];
        for (int y = 0; y < 2; y++) for (int x = 0; x < 4; x++) px[y * 4 + x] = (60 + 40 * x) | ((20 + 30 * y) << 8) | (200 << 16) | (255 << 24);     // R by column, G by row, B constant
        using var src = accelerator.Allocate1D(px);
        using var withPad = accelerator.Allocate1D<float>(3 * 64); using var replicate = accelerator.Allocate1D<float>(3 * 64);
        var k = new ImagePreprocessKernel(accelerator);
        var mean = new[] { 0f, 0f, 0f }; var std = new[] { 1f, 1f, 1f };
        k.Forward(src.View, withPad.View, 4, 2, 8, 8, mean, std, preserveAspect: true, padValue: 114f / 255f);
        k.Forward(src.View, replicate.View, 4, 2, 8, 8, mean, std, preserveAspect: true);
        var a = await withPad.View.CopyToHostAsync(); var b = await replicate.View.CopyToHostAsync();
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                {
                    int i = c * 64 + y * 8 + x; bool pad = y < 2 || y >= 6;
                    if (pad && Math.Abs(a[i] - 114f / 255f) > 1e-6f) throw new Exception($"pad pixel c{c} ({x},{y}) = {a[i]}, expected 114/255");
                    if (!pad && Math.Abs(a[i] - b[i]) > 1e-6f) throw new Exception($"the content pixel c{c} ({x},{y}) changed with padValue: {a[i]} vs {b[i]}");
                    if (y < 2 && Math.Abs(b[i] - b[c * 64 + 2 * 8 + x]) > 1e-6f) throw new Exception($"the default no longer replicates the first content row at c{c} ({x},{y})");
                    if (y >= 6 && Math.Abs(b[i] - b[c * 64 + 5 * 8 + x]) > 1e-6f) throw new Exception($"the default no longer replicates the last content row at c{c} ({x},{y})");
                }
        return "pad value and border replication both correct";
    });
}
