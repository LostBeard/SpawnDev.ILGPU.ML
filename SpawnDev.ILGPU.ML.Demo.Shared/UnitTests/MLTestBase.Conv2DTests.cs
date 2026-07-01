using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>CPU reference Conv2D: output[oc, oy, ox] = bias[oc] + sum over ic,ky,kx.</summary>
    protected static float[] CpuConv2D(float[] input, float[] weight, float[] bias,
        int inC, int inH, int inW, int outC, int kH, int kW, int stride, int padding,
        int dilationH = 1, int dilationW = 1)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        var output = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
        {
            for (int oy = 0; oy < outH; oy++)
            {
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = bias.Length > 0 ? (double)bias[oc] : 0.0;
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oy * stride + ky * dilationH - padding;
                                int ix = ox * stride + kx * dilationW - padding;
                                if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                    sum += (double)input[ic * inH * inW + iy * inW + ix]
                                         * (double)weight[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                            }
                    output[oc * outH * outW + oy * outW + ox] = (float)sum;
                }
            }
        }
        return output;
    }

    /// <summary>CPU reference DepthwiseConv2D NCHW: weight [C, 1, kH, kW].</summary>
    protected static float[] CpuDepthwiseConv2D(float[] input, float[] weight, float[] bias,
        int C, int inH, int inW, int kH, int kW, int stride, int padding,
        int dilationH = 1, int dilationW = 1)
    {
        int effKH = dilationH * (kH - 1) + 1;
        int effKW = dilationW * (kW - 1) + 1;
        int outH = (inH + 2 * padding - effKH) / stride + 1;
        int outW = (inW + 2 * padding - effKW) / stride + 1;
        var output = new float[C * outH * outW];
        for (int c = 0; c < C; c++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = bias.Length > 0 ? (double)bias[c] : 0.0;
                    for (int ky = 0; ky < kH; ky++)
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int iy = oy * stride + ky * dilationH - padding;
                            int ix = ox * stride + kx * dilationW - padding;
                            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                sum += (double)input[c * inH * inW + iy * inW + ix]
                                     * (double)weight[c * kH * kW + ky * kW + kx];
                        }
                    output[c * outH * outW + oy * outW + ox] = (float)sum;
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

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * inH * inW), expected, inC * 2e-5f, "Conv2D 1x1: ");
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

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * inH * inW), expected, inC * kH * kW * 2e-5f, "Conv2D 3x3 pad=1: ");
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

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * outH * outW), expected, inC * kH * kW * 2e-5f, "Conv2D 14x14 patch: ");
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

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * outH * outW), expected, inC * kH * kW * 2e-5f, "Conv2D 3x3 no-pad: ");
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

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * outH * outW), expected, inC * kH * kW * 2e-5f, "Conv2D 3x3 s2p1: ");
    });

    [TestMethod]
    public async Task Conv2D_Depthwise3x3() => await RunTest(async accelerator =>
    {
        int C = 32, inH = 8, inW = 8, kH = 3, kW = 3, stride = 1, padding = 1;
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        var input = RandomFloats(C * inH * inW, seed: 125, scale: 0.5f);
        var weight = RandomFloats(C * kH * kW, seed: 126, scale: 0.1f);
        var bias = RandomFloats(C, seed: 127, scale: 0.01f);

        var expected = new float[C * outH * outW];
        for (int c = 0; c < C; c++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = (double)bias[c];
                    for (int ky = 0; ky < kH; ky++)
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int iy = oy * stride + ky - padding;
                            int ix = ox * stride + kx - padding;
                            if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                sum += (double)input[c * inH * inW + iy * inW + ix] * (double)weight[c * kH * kW + ky * kW + kx];
                        }
                    expected[c * outH * outW + oy * outW + ox] = (float)sum;
                }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(C * outH * outW);

        var conv = new Conv2DKernel(accelerator);
        conv.ForwardDepthwise(inBuf.View, wBuf.View, bBuf.View, outBuf.View, C, inH, inW, kH, kW, stride, padding);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, C * outH * outW), expected, kH * kW * 2e-5f, "Depthwise Conv2D: ");
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
                    double sum = (double)bias[oc];
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
                                sum += (double)input[ic * inH * inW + iy * inW + ix]
                                     * (double)weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
                            }
                        }
                    expected[oc * outH * outW + oy * outW + ox] = (float)sum;
                }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);

        var convT = new ConvTranspose2DKernel(accelerator);
        convT.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, padding);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * outH * outW), expected, inC * kH * kW * 2e-5f, "ConvTranspose2D s4: ");
    });

    [TestMethod]
    public async Task Conv2D_Batch2_ComputesBothViews() => await RunTest(async accelerator =>
    {
        // REGRESSION (DAv3 multi-view, num_images>1): a batched Conv MUST compute EVERY view. Conv2DImpl once
        // decoded idx→(oc,oy,ox) with no batch offset on the input read → view 1+ was left stale garbage. Feed
        // two DIFFERENT views sharing one weight; both must match the per-view CPU reference.
        int inC = 3, inH = 16, inW = 16, outC = 8, kH = 3, kW = 3, stride = 1, pad = 1, batch = 2;
        var in0 = RandomFloats(inC * inH * inW, seed: 400, scale: 0.5f);
        var in1 = RandomFloats(inC * inH * inW, seed: 401, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 402, scale: 0.1f);
        var bias = RandomFloats(outC, seed: 403, scale: 0.01f);
        var input = in0.Concat(in1).ToArray();
        var expected = CpuConv2D(in0, weight, bias, inC, inH, inW, outC, kH, kW, stride, pad)
            .Concat(CpuConv2D(in1, weight, bias, inC, inH, inW, outC, kH, kW, stride, pad)).ToArray();

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);
        var conv = new Conv2DKernel(accelerator);
        conv.ForwardPadded(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW,
            stride, pad, pad, pad, pad, 1, 1, batch);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, expected.Length), expected, inC * 2e-5f, "Conv2D batch2: ");
    });

    [TestMethod]
    public async Task ConvTranspose2D_Batch2_ComputesBothViews() => await RunTest(async accelerator =>
    {
        // REGRESSION (DAv3 multi-view): the DPT head resize_layers ConvTranspose must compute EVERY view. The
        // kernel once had no batch offset → view 1 stale, which corrupted the entire head downstream.
        int inC = 4, inH = 5, inW = 5, outC = 4, kH = 4, kW = 4, stride = 4, padding = 0, batch = 2;
        int outH = (inH - 1) * stride + kH, outW = (inW - 1) * stride + kW;
        var in0 = RandomFloats(inC * inH * inW, seed: 410, scale: 0.5f);
        var in1 = RandomFloats(inC * inH * inW, seed: 411, scale: 0.5f);
        var weight = RandomFloats(inC * outC * kH * kW, seed: 412, scale: 0.1f);
        var bias = RandomFloats(outC, seed: 413, scale: 0.01f);
        float[] CpuCT(float[] input)
        {
            var e = new float[outC * outH * outW];
            for (int oc = 0; oc < outC; oc++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++)
                    {
                        double sum = (double)bias[oc];
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int diffY = oy + padding - ky; if (diffY < 0 || diffY % stride != 0) continue;
                                int iy = diffY / stride; if (iy >= inH) continue;
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int diffX = ox + padding - kx; if (diffX < 0 || diffX % stride != 0) continue;
                                    int ix = diffX / stride; if (ix >= inW) continue;
                                    sum += (double)input[ic * inH * inW + iy * inW + ix]
                                         * (double)weight[ic * outC * kH * kW + oc * kH * kW + ky * kW + kx];
                                }
                            }
                        e[oc * outH * outW + oy * outW + ox] = (float)sum;
                    }
            return e;
        }
        var input = in0.Concat(in1).ToArray();
        var expected = CpuCT(in0).Concat(CpuCT(in1)).ToArray();

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);
        var convT = new ConvTranspose2DKernel(accelerator);
        convT.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, padding, batch);
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, expected.Length), expected, inC * kH * kW * 2e-5f, "ConvTranspose2D batch2: ");
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
        var expected = new float[T * C];
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                expected[t * C + c] = qkvData[t * 3 * C + c]; // Q is first C values per row

        await AssertCloseGpu(accelerator, mergedBuf.View.SubView(0, T * C), expected, 0f, "Attention split/merge round-trip: ");
    });

    // ── Dilation regression tests (commit bbaca6d) ──
    // Conv2DKernel.{Forward,ForwardNHWC,ForwardDepthwise,ForwardDepthwiseNHWC} did not
    // honor dilations until 2026-05-04. RMBG, DDPM, DepthAnything, MoveNet, SqueezeNet,
    // YOLOv8, Whisper-tiny, Style transfers, SuperResolution all use dilation>=2 convs.
    // These tests lock down dilation correctness across all 4 kernel families.

    [TestMethod]
    public async Task Conv2D_NCHW_Dilation2() => await RunTest(async accelerator =>
    {
        // RMBG-style: Conv3x3 stride=1 padding=2 dilation=2 — outH = (inH + 4 - 5)/1 + 1 = inH
        int inC = 32, inH = 16, inW = 16, outC = 32, kH = 3, kW = 3;
        var input = RandomFloats(inC * inH * inW, seed: 110, scale: 0.5f);
        var weight = RandomFloats(outC * inC * kH * kW, seed: 111, scale: 0.05f);
        var bias = RandomFloats(outC, seed: 112, scale: 0.01f);
        var expected = CpuConv2D(input, weight, bias, inC, inH, inW, outC, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * inH * inW);

        var conv = new Conv2DKernel(accelerator);
        conv.Forward(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * inH * inW), expected, inC * 5e-5f, "Conv2D NCHW dilation=2: ");
    });

    [TestMethod]
    public async Task Conv2D_NHWC_Dilation2() => await RunTest(async accelerator =>
    {
        // NHWC Conv3x3 stride=1 padding=2 dilation=2
        int inC = 16, inH = 12, inW = 12, outC = 24, kH = 3, kW = 3;
        // NHWC input layout: [N=1, H, W, inC]
        var inputNHWC = RandomFloats(inH * inW * inC, seed: 113, scale: 0.5f);
        // NHWC weight layout: [outC, kH, kW, inC]
        var weightNHWC = RandomFloats(outC * kH * kW * inC, seed: 114, scale: 0.05f);
        var bias = RandomFloats(outC, seed: 115, scale: 0.01f);

        // CPU reference: convert to NCHW, run NCHW conv, convert back
        var inputNCHW = new float[inC * inH * inW];
        for (int h = 0; h < inH; h++)
            for (int w = 0; w < inW; w++)
                for (int c = 0; c < inC; c++)
                    inputNCHW[c * inH * inW + h * inW + w] = inputNHWC[h * inW * inC + w * inC + c];
        // weight NHWC [oc, kh, kw, ic] → NCHW [oc, ic, kh, kw]
        var weightNCHW = new float[outC * inC * kH * kW];
        for (int oc = 0; oc < outC; oc++)
            for (int ky = 0; ky < kH; ky++)
                for (int kx = 0; kx < kW; kx++)
                    for (int ic = 0; ic < inC; ic++)
                        weightNCHW[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx] = weightNHWC[oc * kH * kW * inC + ky * kW * inC + kx * inC + ic];
        var expectedNCHW = CpuConv2D(inputNCHW, weightNCHW, bias, inC, inH, inW, outC, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);
        // Convert expected back to NHWC
        var expected = new float[outC * inH * inW];
        for (int h = 0; h < inH; h++)
            for (int w = 0; w < inW; w++)
                for (int c = 0; c < outC; c++)
                    expected[h * inW * outC + w * outC + c] = expectedNCHW[c * inH * inW + h * inW + w];

        using var inBuf = accelerator.Allocate1D(inputNHWC);
        using var wBuf = accelerator.Allocate1D(weightNHWC);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * inH * inW);

        var conv = new Conv2DKernel(accelerator);
        conv.ForwardNHWC(inBuf.View, wBuf.View, bBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, outC * inH * inW), expected, inC * 5e-5f, "Conv2D NHWC dilation=2: ");
    });

    [TestMethod]
    public async Task DepthwiseConv2D_NCHW_Dilation2() => await RunTest(async accelerator =>
    {
        // Depthwise NCHW kernel 3x3 stride=1 padding=2 dilation=2
        int C = 24, inH = 14, inW = 14, kH = 3, kW = 3;
        var input = RandomFloats(C * inH * inW, seed: 116, scale: 0.5f);
        // Weight [C, 1, kH, kW] = [C * kH * kW] flat
        var weight = RandomFloats(C * kH * kW, seed: 117, scale: 0.1f);
        var bias = RandomFloats(C, seed: 118, scale: 0.01f);
        var expected = CpuDepthwiseConv2D(input, weight, bias, C, inH, inW, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(weight);
        using var bBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(C * inH * inW);

        var conv = new Conv2DKernel(accelerator);
        conv.ForwardDepthwise(inBuf.View, wBuf.View, bBuf.View, outBuf.View, C, inH, inW, kH, kW, stride: 1, padding: 2, dilationH: 2, dilationW: 2);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, C * inH * inW), expected, kH * kW * 5e-5f, "DepthwiseConv2D NCHW dilation=2: ");
    });
}
