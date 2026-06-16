using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Resize OPERATOR tests — the production node path (Resolve -> Execute) vs a CPU oracle. These exist
/// because <see cref="ResizeOperator"/> used to IGNORE the ONNX <c>mode</c> attribute and ALWAYS
/// bilinear-resize. SD-Turbo's VAE decoder (and depth/style/super-res models) upsample with
/// <c>mode="nearest"</c>; bilinear-smoothing a nearest Resize low-passes the whole image, so EVERY
/// generated image came out blurry. The discriminator test below is the regression guard: it uses a 2x2
/// input whose nearest and bilinear upsamples DIFFER at every interior pixel, so a regression to "always
/// bilinear" fails loudly. (Found by diffing our VAE decode against the ONNX Runtime oracle on an identical
/// latent — ORT sharp, ours soft — then reading the op.)
/// </summary>
public abstract partial class MLTestBase
{
    // CPU oracle: ONNX Resize mode="nearest", asymmetric/floor (the SD/diffusers upsample). Each output
    // pixel copies the input pixel at floor(outIdx * inDim/outDim) per axis. NCHW.
    private static float[] ResizeNearestCpu(float[] x, int C, int inH, int inW, int outH, int outW)
    {
        var o = new float[C * outH * outW];
        for (int c = 0; c < C; c++)
            for (int oy = 0; oy < outH; oy++)
            {
                int iy = (int)((long)oy * inH / outH); if (iy >= inH) iy = inH - 1;
                for (int ox = 0; ox < outW; ox++)
                {
                    int ix = (int)((long)ox * inW / outW); if (ix >= inW) ix = inW - 1;
                    o[(c * outH + oy) * outW + ox] = x[(c * inH + iy) * inW + ix];
                }
            }
        return o;
    }

    [TestMethod]
    public async Task Resize_NearestMode_MatchesCpu_NotBilinear() => await RunTest(async accelerator =>
    {
        // 1 channel, 2x2 -> 4x4 nearest. Distinct corner values so nearest != bilinear at interior pixels.
        const int C = 1, inH = 2, inW = 2, outH = 4, outW = 4;
        var x = new float[] { 0f, 10f, 20f, 30f }; // [[0,10],[20,30]]
        var expected = ResizeNearestCpu(x, C, inH, inW, outH, outW);

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(C * outH * outW);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("Resize");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, C, inH, inW }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, outH, outW }) },
            Attributes = new Dictionary<string, object> { ["mode"] = "nearest" },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, C * outH * outW);

        AssertClose(expected, got, 1e-4f, "Resize nearest ");

        // Discriminator: a "0.5" interior pixel (bilinear avg of 0 and 10 = 5) would mean a regression to
        // always-bilinear. Nearest gives exactly 0 or 10 there — never 5.
        // got[row0] must be [0,0,10,10] for nearest (NOT [0, ~3.3, ~6.6, 10] bilinear).
        if (MathF.Abs(got[1] - 0f) > 1e-4f)
            throw new Exception($"Resize nearest regressed to bilinear: out[0,1]={got[1]} (nearest=0, bilinear≈3.3). " +
                "ResizeOperator must honor mode=\"nearest\".");
        Console.WriteLine($"[Resize] nearest mode matches CPU oracle + is NOT bilinear ({BackendName})");
    });

    [TestMethod]
    public async Task Resize_LinearMode_IsBilinear() => await RunTest(async accelerator =>
    {
        // mode="linear" must still bilinear-interpolate (the non-default path stays correct).
        const int C = 1, inH = 2, inW = 2, outH = 4, outW = 4;
        var x = new float[] { 0f, 10f, 20f, 30f };

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(C * outH * outW);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("Resize");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { 1, C, inH, inW }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { 1, C, outH, outW }) },
            Attributes = new Dictionary<string, object> { ["mode"] = "linear" },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, C * outH * outW);

        // Linear must produce at least one interior value strictly between the input corners (smoothing) —
        // i.e. NOT the pure nearest set {0,10,20,30}. Proves the linear path is still wired.
        bool anyInterpolated = got.Any(v => v > 0.01f && v < 9.99f) || got.Any(v => v > 10.01f && v < 19.99f);
        if (!anyInterpolated)
            throw new Exception("Resize mode=\"linear\" produced no interpolated values — bilinear path broken.");
        Console.WriteLine($"[Resize] linear mode still bilinear-interpolates ({BackendName})");
    });
}
