using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A 1x1 Conv over a TALL, one-pixel-wide map at production size (2026-10-01): RaCo-ALIKED's descriptor head
    /// samples 2048 keypoints x 16 offsets into [2,128,32768,1] and runs sf_conv (128->128, 1x1) on it; the desktop
    /// test process crashed there. models/tests/conv1x1_tall.onnx: W[o,c] = ((o*128+c) % 13 - 6) / 64, zero bias.
    /// Every output position is checked against a CPU dot product.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Conv1x1_Tall_ProductionShape() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/conv1x1_tall.onnx");
        const int N = 2, C = 128, H = 32768;
        var x = new float[N * C * H];
        for (int i = 0; i < x.Length; i++) x[i] = ((i * 7) % 17 - 8) / 8f;
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { N, C, H, 1 }) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { N, C, H, 1 }))
            throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [{N},{C},{H},1]");
        using var yH = accelerator.Allocate1D<float>(x.Length);
        await yH.View.CopyFromAsync(y.Data.SubView(0, x.Length));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, x.Length);
        int bad = 0; string first = "";
        for (int n = 0; n < N; n++)
            for (int o = 0; o < C; o++)
                for (int p = 0; p < H; p++)
                {
                    float want = 0;
                    for (int c = 0; c < C; c++) want += ((o * 128 + c) % 13 - 6) / 64f * x[(n * C + c) * H + p];
                    float got = yv[(n * C + o) * H + p];
                    if (MathF.Abs(got - want) > 1e-3f * (1 + MathF.Abs(want)))
                        if (bad++ == 0) first = $"y[{n},{o},{p}] = {got}, expected {want}";
                }
        if (bad > 0) throw new Exception($"{bad} of {yv.Length} outputs wrong, first {first}");
    });

    /// <summary>
    /// Runs models/tests/{model} (one NCHW Conv: W = ((i % 11) - 5) / 16, b[o] = ((o % 5) - 2) / 8, x = ((i*7 % 17) - 8)
    /// / 8, all built by the same formulas in the generator) and checks every output against a direct convolution.
    /// </summary>
    async Task RunConvCase(Accelerator accelerator, string model, int[] xs, int[] ws, int group,
        int[] pads, int strideH, int strideW, int dilH, int dilW)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync($"models/tests/{model}");
        int N = xs[0], C = xs[1], H = xs[2], W = xs[3], O = ws[0], Cg = ws[1], KH = ws[2], KW = ws[3];
        int OH = (H + pads[0] + pads[2] - (dilH * (KH - 1) + 1)) / strideH + 1;
        int OW = (W + pads[1] + pads[3] - (dilW * (KW - 1) + 1)) / strideW + 1;
        var x = new float[N * C * H * W];
        for (int i = 0; i < x.Length; i++) x[i] = ((i * 7) % 17 - 8) / 8f;
        float Wt(int i) => ((i % 11) - 5) / 16f;
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, xs) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { N, O, OH, OW }))
            throw new Exception($"{model}: y shape [{string.Join(",", y.Shape)}], expected [{N},{O},{OH},{OW}]");
        int total = N * O * OH * OW;
        using var yH = accelerator.Allocate1D<float>(total);
        await yH.View.CopyFromAsync(y.Data.SubView(0, total));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, total);
        int oPerG = O / group, bad = 0; string first = "";
        for (int n = 0; n < N; n++)
            for (int o = 0; o < O; o++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++)
                    {
                        int g = o / oPerG;
                        float want = ((o % 5) - 2) / 8f;
                        for (int c = 0; c < Cg; c++)
                            for (int ky = 0; ky < KH; ky++)
                                for (int kx = 0; kx < KW; kx++)
                                {
                                    int iy = oy * strideH + ky * dilH - pads[0], ix = ox * strideW + kx * dilW - pads[1];
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                                    want += Wt(((o * Cg + c) * KH + ky) * KW + kx) * x[((n * C + g * Cg + c) * H + iy) * W + ix];
                                }
                        float got = yv[((n * O + o) * OH + oy) * OW + ox];
                        if (MathF.Abs(got - want) > 1e-4f * (1 + MathF.Abs(want)))
                            if (bad++ == 0) first = $"y[{n},{o},{oy},{ox}] = {got}, expected {want}";
                    }
        if (bad > 0) throw new Exception($"{model}: {bad} of {total} outputs wrong, first {first}");
    }

    /// <summary>Depthwise Conv with batch 2: the depthwise kernel computed image 0 only (2026-10-01).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Conv_Depthwise_Batch2() => await RunTest(async accelerator =>
        await RunConvCase(accelerator, "conv_depthwise_batch2.onnx", new[] { 2, 3, 9, 7 }, new[] { 3, 1, 3, 3 }, 3,
            new[] { 1, 1, 1, 1 }, 1, 1, 1, 1));

    /// <summary>Grouped (non-depthwise) Conv with batch 2, stride (2,1), dilation (1,2), asymmetric pads: the
    /// per-group slices covered image 0 only (2026-10-01).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Conv_Grouped_Batch2() => await RunTest(async accelerator =>
        await RunConvCase(accelerator, "conv_grouped_batch2.onnx", new[] { 2, 4, 6, 5 }, new[] { 6, 2, 3, 3 }, 2,
            new[] { 1, 0, 1, 2 }, 2, 1, 1, 2));

    /// <summary>Grouped Conv with batch 2 and square stride - the batch loop alone (the test above also needs
    /// per-axis strides, and fails on those first without them).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Conv_GroupedSquare_Batch2() => await RunTest(async accelerator =>
        await RunConvCase(accelerator, "conv_grouped_square_batch2.onnx", new[] { 2, 4, 6, 5 }, new[] { 6, 2, 3, 3 }, 2,
            new[] { 1, 1, 1, 1 }, 1, 1, 1, 1));

    /// <summary>A begin pad of 256: the kernels packed pads 8 bits each, so 256 decoded as 0 (2026-10-01).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Conv_WidePad256() => await RunTest(async accelerator =>
        await RunConvCase(accelerator, "conv_wide_pad256.onnx", new[] { 1, 1, 1, 600 }, new[] { 1, 1, 1, 513 }, 1,
            new[] { 0, 256, 0, 256 }, 1, 1, 1, 1));
}
