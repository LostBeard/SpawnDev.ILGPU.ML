using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>CPU reference Conv2D: output[oc, oy, ox] = bias[oc] + sum over ic,ky,kx.</summary>
    protected static float[] CpuConv2D(float[] input, float[] weight, float[] bias,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride, int padding)
    {
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        var output = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
        {
            for (int oy = 0; oy < outH; oy++)
            {
                for (int ox = 0; ox < outW; ox++)
                {
                    float sum = bias.Length > 0 ? bias[oc] : 0f;
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oy * stride + ky - padding;
                                int ix = ox * stride + kx - padding;
                                if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                    sum += input[ic * inH * inW + iy * inW + ix]
                                         * weight[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                            }
                    output[oc * outH * outW + oy * outW + ox] = sum;
                }
            }
        }
        return output;
    }

    [TestMethod]
    public async Task Conv2D_1x1Projection() => await RunTest(async accelerator =>
    {
        // DPT head projection: Conv1x1 [768 → 48] at 37×37
        int inC = 768, inH = 37, inW = 37, outC = 48, kH = 1, kW = 1;
        var input = RandomFloats(inC * inH * inW, seed: 90, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 91, scale: 0.05f);
        var bias = RandomFloats(outC, seed: 92, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, 1, 0);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * inH * inW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, 1, 0);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * inH * inW);
        AssertClose(expected, actual, inC * 2e-5f, "Conv2D 1x1: ");
    });

    [TestMethod]
    public async Task Conv2D_3x3WithPadding() => await RunTest(async accelerator =>
    {
        // RefineNet conv: Conv3x3 pad=1 [64 → 64] at 37×37
        int inC = 64, inH = 37, inW = 37, outC = 64, kH = 3, kW = 3;
        var input = RandomFloats(inC * inH * inW, seed: 93, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 94, scale: 0.02f);
        var bias = RandomFloats(outC, seed: 95, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, 1, 1);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * inH * inW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, 1, 1);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * inH * inW);
        AssertClose(expected, actual, inC * kH * kW * 2e-5f, "Conv2D 3x3 pad=1: ");
    });

    [TestMethod]
    public async Task Conv2D_PatchEmbed14x14() => await RunTest(async accelerator =>
    {
        // Patch embedding: Conv 14×14 stride 14 [3 → 384] on 518×518
        // Use smaller input to keep test fast
        int inC = 3, inH = 56, inW = 56, outC = 16, kH = 14, kW = 14, stride = 14;
        int outH = (inH - kH) / stride + 1; // 3
        int outW = (inW - kW) / stride + 1; // 3
        var input = RandomFloats(inC * inH * inW, seed: 96, scale: 1f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 97, scale: 0.02f);
        var bias = RandomFloats(outC, seed: 98, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, stride, 0);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, 0);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);
        AssertClose(expected, actual, inC * kH * kW * 2e-5f, "Conv2D 14x14 patch: ");
    });

    [TestMethod]
    public async Task Conv2D_3x3NoPadding() => await RunTest(async accelerator =>
    {
        // 3x3 with NO padding — isolates whether padding logic causes the WebGPU failure
        int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3;
        int outH = inH - kH + 1; // 8
        int outW = inW - kW + 1; // 8
        var input = RandomFloats(inC * inH * inW, seed: 110, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 111, scale: 0.1f);
        var bias = RandomFloats(outC, seed: 112, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, 1, 0);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, 1, 0);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);
        AssertClose(expected, actual, inC * kH * kW * 2e-5f, "Conv2D 3x3 no-pad: ");
    });

    [TestMethod]
    public async Task Conv2D_3x3Stride2() => await RunTest(async accelerator =>
    {
        // 3x3 stride=2 pad=1 — tests stride > 1 with padding
        int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3, stride = 2, padding = 1;
        int outH = (inH + 2 * padding - kH) / stride + 1; // 5
        int outW = (inW + 2 * padding - kW) / stride + 1; // 5
        var input = RandomFloats(inC * inH * inW, seed: 113, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 114, scale: 0.1f);
        var bias = RandomFloats(outC, seed: 115, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, stride, padding);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, padding);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);
        AssertClose(expected, actual, inC * kH * kW * 2e-5f, "Conv2D 3x3 s2p1: ");
    });

    [TestMethod]
    public async Task ConvTranspose2D_Stride4() => await RunTest(async accelerator =>
    {
        // DPT resize_layer: ConvTranspose [48,48,4,4] stride=4 → 37→148
        int inC = 4, inH = 5, inW = 5, outC = 4, kH = 4, kW = 4, stride = 4, padding = 0;
        int outH = (inH - 1) * stride + kH; // 20
        int outW = (inW - 1) * stride + kW; // 20
        var input = RandomFloats(inC * inH * inW, seed: 120, scale: 0.5f);
        var weight = RandomFloats(inC * outC * kH * kW, seed: 121, scale: 0.1f);
        var bias = RandomFloats(outC, seed: 122, scale: 0.01f);

        // CPU reference (gather direction)
        var expected = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    float sum = bias[oc];
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int diffY = oy + padding - ky;
                            if (diffY < 0 || diffY % stride != 0) continue;
                            int iy = diffY / stride;
                            if (iy >= inH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int diffX = ox + padding - kx;
                                if (diffX < 0 || diffX % stride != 0) continue;
                                int ix = diffX / stride;
                                if (ix >= inW) continue;
                                sum += input[ic * inH * inW + iy * inW + ix]
                                     * weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
                            }
                        }
                    expected[oc * outH * outW + oy * outW + ox] = sum;
                }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);

        var convT = new ConvTranspose2DKernel(accelerator);
        convT.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, padding);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);
        AssertClose(expected, actual, inC * kH * kW * 2e-5f, "ConvTranspose2D s4: ");
    });

    [TestMethod]
    public async Task AttentionSplitMerge_RoundTrip() => await RunTest(async accelerator =>
    {
        // SplitHeads → MergeHeads should recover the original data
        int T = 1370, C = 384, H = 6, D = 64;
        // Create QKV data [T, 3*C] with known pattern
        var qkvData = RandomFloats(T * 3 * C, seed: 80);

        using var qkvBuf = accelerator.Allocate1D(qkvData);
        using var qBuf = accelerator.Allocate1D<float>(H * T * D);
        using var kBuf = accelerator.Allocate1D<float>(H * T * D);
        using var vBuf = accelerator.Allocate1D<float>(H * T * D);
        using var mergedBuf = accelerator.Allocate1D<float>(T * C);

        var attn = new AttentionKernels(accelerator);

        // Split
        attn.SplitHeads(qkvBuf.View, qBuf.View, kBuf.View, vBuf.View, T);
        // Merge Q back — should recover the Q portion of QKV
        attn.MergeHeads(qBuf.View, mergedBuf.View, T);
        await accelerator.SynchronizeAsync();

        // Expected: merged[t, c] = qkv[t, c] for c in [0, C) (Q portion)
        var merged = await mergedBuf.CopyToHostAsync<float>(0, T * C);
        var expected = new float[T * C];
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                expected[t * C + c] = qkvData[t * 3 * C + c]; // Q is first C values per row

        AssertClose(expected, merged, 0f, "Attention split/merge round-trip: ");
    });
}
