using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Pad_Constant_2D() => await RunTest(async accelerator =>
    {
        // Input [3, 4], pad [1, 1, 1, 1] → output [5, 6]
        var input = new float[]
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        int[] inputShape = { 3, 4 };
        int[] pads = { 1, 1, 1, 1 }; // before_dim0, before_dim1, after_dim0, after_dim1
        int rank = 2;

        // CPU reference: constant pad with value 0
        int[] outShape = { 5, 6 };
        var expected = new float[outShape[0] * outShape[1]];
        for (int y = 0; y < outShape[0]; y++)
        for (int x = 0; x < outShape[1]; x++)
        {
            int sy = y - pads[0];
            int sx = x - pads[1];
            expected[y * outShape[1] + x] = (sy >= 0 && sy < 3 && sx >= 0 && sx < 4)
                ? input[sy * 4 + sx] : 0f;
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(expected.Length);

        var pad = new PadKernel(accelerator);
        pad.Forward(inBuf.View, outBuf.View, inputShape, pads, mode: 0, constantValue: 0f);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, expected.Length), expected, 1e-6f, "Pad constant 2D: ");
    });

    [TestMethod]
    public async Task Pad_Reflect_4D_StyleTransfer() => await RunTest(async accelerator =>
    {
        // Style transfer uses reflect padding on [N, C, H, W] with spatial pads only
        // Input [1, 1, 4, 4], pad [0, 0, 1, 1, 0, 0, 1, 1] → output [1, 1, 6, 6]
        int N = 1, C = 1, H = 4, W = 4;
        var input = new float[N * C * H * W];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1; // 1..16

        int[] inputShape = { N, C, H, W };
        int[] pads = { 0, 0, 1, 1, 0, 0, 1, 1 }; // before: [0,0,1,1], after: [0,0,1,1]
        int rank = 4;

        int oH = H + 2, oW = W + 2;
        int[] outShape = { N, C, oH, oW };
        int totalOut = N * C * oH * oW;
        var expected = new float[totalOut];

        // CPU reference for reflect padding
        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int oy = 0; oy < oH; oy++)
        for (int ox = 0; ox < oW; ox++)
        {
            int sy = oy - 1; // pad_before for H = 1
            int sx = ox - 1; // pad_before for W = 1

            // Reflect
            if (sy < 0) sy = -sy;
            if (sy >= H) sy = 2 * (H - 1) - sy;
            if (sx < 0) sx = -sx;
            if (sx >= W) sx = 2 * (W - 1) - sx;

            int outIdx = ((n * C + c) * oH + oy) * oW + ox;
            int inIdx = ((n * C + c) * H + sy) * W + sx;
            expected[outIdx] = input[inIdx];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(totalOut);

        var pad = new PadKernel(accelerator);
        pad.Forward(inBuf.View, outBuf.View, inputShape, pads, mode: 2);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, totalOut), expected, 1e-6f, "Pad reflect 4D: ");
    });

    [TestMethod]
    public async Task Pad_Edge_2D() => await RunTest(async accelerator =>
    {
        // Input [3, 3], pad [1, 1, 1, 1] → output [5, 5]
        var input = new float[]
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int[] inputShape = { 3, 3 };
        int[] pads = { 1, 1, 1, 1 };

        int[] outShape = { 5, 5 };
        var expected = new float[25];
        for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++)
        {
            int sy = Math.Clamp(y - 1, 0, 2);
            int sx = Math.Clamp(x - 1, 0, 2);
            expected[y * 5 + x] = input[sy * 3 + sx];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(25);

        var pad = new PadKernel(accelerator);
        pad.Forward(inBuf.View, outBuf.View, inputShape, pads, mode: 1);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, 25), expected, 1e-6f, "Pad edge 2D: ");
    });

    [TestMethod]
    public async Task NearestUpsample_2x() => await RunTest(async accelerator =>
    {
        // Input [1, 2, 3, 3] → output [1, 2, 6, 6] (2x upsample)
        int N = 1, C = 2, inH = 3, inW = 3;
        int outH = 6, outW = 6;
        var input = RandomFloats(N * C * inH * inW, seed: 200);

        int inC = N * C; // channels including batch
        int totalOut = inC * outH * outW;
        var expected = new float[totalOut];

        // CPU reference: nearest-neighbor
        for (int c = 0; c < inC; c++)
        for (int oy = 0; oy < outH; oy++)
        for (int ox = 0; ox < outW; ox++)
        {
            int iy = oy * inH / outH;
            int ix = ox * inW / outW;
            if (iy >= inH) iy = inH - 1;
            if (ix >= inW) ix = inW - 1;
            expected[c * outH * outW + oy * outW + ox] = input[c * inH * inW + iy * inW + ix];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(totalOut);

        var ew = new ElementWiseKernels(accelerator);
        int[] inputShape = { N, C, inH, inW };
        int[] outputShape = { N, C, outH, outW };
        ew.NearestUpsample(inBuf.View, outBuf.View, inputShape, outputShape);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, totalOut), expected, 1e-6f, "NearestUpsample 2x: ");
    });

    [TestMethod]
    public async Task NearestUpsample_StyleTransferDims() => await RunTest(async accelerator =>
    {
        // Style transfer upsamples: [1, 64, 56, 56] → [1, 64, 112, 112]
        int N = 1, C = 64, inH = 56, inW = 56;
        int outH = 112, outW = 112;
        var input = RandomFloats(N * C * inH * inW, seed: 210, scale: 5f);

        int inC = N * C;
        int totalOut = inC * outH * outW;
        var expected = new float[totalOut];

        for (int c = 0; c < inC; c++)
        for (int oy = 0; oy < outH; oy++)
        for (int ox = 0; ox < outW; ox++)
        {
            int iy = oy * inH / outH;
            int ix = ox * inW / outW;
            if (iy >= inH) iy = inH - 1;
            if (ix >= inW) ix = inW - 1;
            expected[c * outH * outW + oy * outW + ox] = input[c * inH * inW + iy * inW + ix];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(totalOut);

        var ew = new ElementWiseKernels(accelerator);
        ew.NearestUpsample(inBuf.View, outBuf.View,
            new[] { N, C, inH, inW }, new[] { N, C, outH, outW });
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, totalOut), expected, 1e-6f, "NearestUpsample 56→112: ");
    });

    [TestMethod]
    public async Task Div_ElementWise_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 1024;
        var a = RandomFloats(count, seed: 220, scale: 10f);
        var b = RandomFloats(count, seed: 221, scale: 5f);
        // Avoid division by zero
        for (int i = 0; i < count; i++)
            if (MathF.Abs(b[i]) < 0.01f) b[i] = 1f;

        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = a[i] / b[i];

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var outBuf = accelerator.Allocate1D<float>(count);

        var ew = new ElementWiseKernels(accelerator);
        ew.Div(aBuf.View, bBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, count), expected, 1e-4f, "Div element-wise: ");
    });

    [TestMethod]
    public async Task Floor_MatchesCpu() => await RunTest(async accelerator =>
    {
        int count = 512;
        var input = RandomFloats(count, seed: 230, scale: 100f);

        var expected = new float[count];
        for (int i = 0; i < count; i++)
            expected[i] = MathF.Floor(input[i]);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(count);

        var ew = new ElementWiseKernels(accelerator);
        ew.Floor(inBuf.View, outBuf.View, count);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, count), expected, 1e-6f, "Floor: ");
    });
}
