using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task BatchNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 2, C = 64, H = 7, W = 7;
        int spatial = H * W;
        var input = RandomFloats(N * C * spatial, seed: 140);
        var scale = RandomFloats(C, seed: 141, scale: 0.5f);
        var bias = RandomFloats(C, seed: 142, scale: 0.1f);
        var mean = RandomFloats(C, seed: 143, scale: 2f);
        var variance = RandomFloats(C, seed: 144, scale: 1f);
        for (int i = 0; i < C; i++) variance[i] = MathF.Abs(variance[i]) + 0.1f; // positive

        // CPU reference
        float eps = 1e-5f;
        var expected = new float[N * C * spatial];
        for (int i = 0; i < expected.Length; i++)
        {
            int c = (i / spatial) % C;
            float invStd = 1f / MathF.Sqrt(variance[c] + eps);
            expected[i] = scale[c] * (input[i] - mean[c]) * invStd + bias[c];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(N * C * spatial);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);
        using var mBuf = accelerator.Allocate1D(mean);
        using var vBuf = accelerator.Allocate1D(variance);

        var norm = new NormalizationKernels(accelerator);
        norm.BatchNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, mBuf.View, vBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, N * C * spatial), expected, 1e-4f, "BatchNorm: ");
    });

    [TestMethod]
    public async Task InstanceNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int N = 1, C = 3, H = 8, W = 8;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 150, scale: 5f);
        var scale = RandomFloats(C, seed: 151, scale: 1f);
        var bias = RandomFloats(C, seed: 152, scale: 0.5f);
        for (int i = 0; i < C; i++) scale[i] = MathF.Abs(scale[i]) + 0.5f;

        // CPU reference: normalize each (n,c) slice independently
        float eps = 1e-5f;
        var expected = new float[total];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                int sliceBase = (n * C + c) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[sliceBase + i];
                float mean = sum / spatial;
                float varSum = 0;
                for (int i = 0; i < spatial; i++)
                {
                    float d = input[sliceBase + i] - mean;
                    varSum += d * d;
                }
                float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);
                for (int i = 0; i < spatial; i++)
                    expected[sliceBase + i] = scale[c] * (input[sliceBase + i] - mean) * invStd + bias[c];
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);
        norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, total), expected, 1e-4f, "InstanceNorm small: ");
    });

    [TestMethod]
    public async Task InstanceNorm_StyleTransferDims_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Style transfer dimensions: [1, 3, 224, 224]
        int N = 1, C = 3, H = 224, W = 224;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 160, scale: 255f);
        var scale = new float[] { 1f, 1f, 1f };
        var bias = new float[] { 0f, 0f, 0f };

        // CPU reference
        float eps = 1e-5f;
        var expected = new float[total];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                int sliceBase = (n * C + c) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[sliceBase + i];
                float mean = sum / spatial;
                float varSum = 0;
                for (int i = 0; i < spatial; i++)
                {
                    float d = input[sliceBase + i] - mean;
                    varSum += d * d;
                }
                float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);
                for (int i = 0; i < spatial; i++)
                    expected[sliceBase + i] = scale[c] * (input[sliceBase + i] - mean) * invStd + bias[c];
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);
        norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, total), expected, 1e-3f, "InstanceNorm 224x224: ");
    });

    [TestMethod]
    public async Task InstanceNorm_StyleMosaicShape_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Tight repro for Data's WebGL StyleMosaic mean-error 47.29 (2026-05-04).
        // First InstanceNorm call in StyleMosaic graph: N=1, C=32, H=W=224 -> spatial=50176.
        // Pass 1 capture (means + invStds) was bit-identical WebGPU vs WebGL; Pass 2 suspect.
        // This test exercises the kernel directly so a failure pins the bug at codegen, not graph.
        int N = 1, C = 32, H = 224, W = 224;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 170, scale: 5f);
        var scale = RandomFloats(C, seed: 171, scale: 1f);
        var bias = RandomFloats(C, seed: 172, scale: 0.5f);
        for (int i = 0; i < C; i++) scale[i] = MathF.Abs(scale[i]) + 0.5f;

        float eps = 1e-5f;
        var expected = new float[total];
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                int sliceBase = (n * C + c) * spatial;
                float sum = 0;
                for (int i = 0; i < spatial; i++) sum += input[sliceBase + i];
                float mean = sum / spatial;
                float varSum = 0;
                for (int i = 0; i < spatial; i++)
                {
                    float d = input[sliceBase + i] - mean;
                    varSum += d * d;
                }
                float invStd = 1f / MathF.Sqrt(varSum / spatial + eps);
                for (int i = 0; i < spatial; i++)
                    expected[sliceBase + i] = scale[c] * (input[sliceBase + i] - mean) * invStd + bias[c];
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);
        norm.InstanceNorm(inBuf.View, outBuf.View, sBuf.View, bBuf.View, N, C, spatial);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, total), expected, 1e-3f, "InstanceNorm 1x32x224x224: ");
    });

    [TestMethod]
    public async Task Softmax_YOLOv8DflShape_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Tight repro for Data's YOLOv8 Wasm node-220 Softmax divergence (2026-05-04).
        // YOLOv8 DFL head softmax: 16 bins per row over 4 * num_anchors rows.
        // For 640x640 input num_anchors=8400, so rows=33600, cols=16.
        // Wasm bin0=0.088 vs WebGPU bin0=0.520 — first-divergent node confirmed by Data.
        int rows = 33600, cols = 16;
        var input = RandomFloats(rows * cols, seed: 180, scale: 4f);

        var expected = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            int rb = r * cols;
            float max = float.MinValue;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, input[rb + c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += MathF.Exp(input[rb + c] - max);
            float invSum = (float)(1.0 / sum);
            for (int c = 0; c < cols; c++) expected[rb + c] = MathF.Exp(input[rb + c] - max) * invSum;
        }

        var inputCopy = (float[])input.Clone();
        using var dataBuf = accelerator.Allocate1D(inputCopy);
        var sm = new SoftmaxKernel(accelerator);
        sm.Forward(dataBuf.View, rows, cols);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, dataBuf.View.SubView(0, rows * cols), expected, 1e-4f, "Softmax 33600x16 (YOLOv8 DFL): ");
    });

    [TestMethod]
    public async Task Softmax_YOLOv8DflFullPath_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Reproduces the SoftmaxOperator general-axis path used by YOLOv8 node 220
        // (DFL head): input shape [4, 16, 8400], axis=1 (= the 16-bin dim).
        // Operator does: Transpose [4,16,8400]->[4,8400,16] (perm=[0,2,1]) →
        //                Softmax rows=33600 cols=16 →
        //                Transpose [4,8400,16]->[4,16,8400] (perm=[0,2,1]).
        // Data captured Wasm bin0=0.088 vs WebGPU bin0=0.520 at node 220 output.
        // SoftmaxKernel-direct (rows=33600, cols=16) passed on every backend, so
        // the bug must live in the transpose/softmax COMPOSITION on Wasm.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 181, scale: 4f);

        // CPU reference: softmax over the axisDim axis (axis=1 for [outer,axisDim,inner]).
        var expected = new float[total];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int baseIdx = o * axisDim * inner + i;
                float max = float.MinValue;
                for (int a = 0; a < axisDim; a++)
                    max = MathF.Max(max, input[baseIdx + a * inner]);
                double sum = 0;
                for (int a = 0; a < axisDim; a++)
                    sum += MathF.Exp(input[baseIdx + a * inner] - max);
                float invSum = (float)(1.0 / sum);
                for (int a = 0; a < axisDim; a++)
                    expected[baseIdx + a * inner] = MathF.Exp(input[baseIdx + a * inner] - max) * invSum;
            }
        }

        using var inOutBuf = accelerator.Allocate1D((float[])input.Clone());
        using var transposedBuf = accelerator.Allocate1D<float>(total);

        // Step 1: Transpose [outer,axisDim,inner] -> [outer,inner,axisDim] (perm=[0,2,1])
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);
        transpose.Transpose(inOutBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });

        // Step 2: Softmax on rows of axisDim
        var sm = new SoftmaxKernel(accelerator);
        sm.Forward(transposedBuf.View, outer * inner, axisDim);

        // Step 3: Transpose back [outer,inner,axisDim] -> [outer,axisDim,inner]
        transpose.Transpose(transposedBuf.View, inOutBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, inOutBuf.View.SubView(0, total), expected, 1e-4f, "Softmax YOLOv8 DFL [4,16,8400] axis=1: ");
    });

    [TestMethod]
    public async Task ShapeGatherScale_StyleMosaicShape_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Reproduces the StyleMosaic node 55 Gather_117 path that Data identified as
        // first-divergent on WebGL (returns 0 instead of 56). The model:
        //   Shape(input)        -> [1, 32, 56, 56]   stored as floats
        //   Gather(shape, [2])  -> [56]              extracts H dim
        // GatherOperator's CPU-index path calls Scale(data.SubView(srcOffset=2, 1),
        // output.SubView(0, 1), 1, 1f) — a 1-element SubView read. If the SubView
        // offset isn't applied to texelFetch on WebGL, the read returns data[0]=1
        // (or, with stale texture data, 0).
        var op = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var shapeData = new float[] { 1f, 32f, 56f, 56f };
        using var dataBuf = accelerator.Allocate1D(shapeData);
        using var outBuf = accelerator.Allocate1D<float>(1);

        // Mimic GatherOperator inner loop for axis=0, idx=2.
        op.Scale(
            dataBuf.View.SubView(2, 1),
            outBuf.View.SubView(0, 1),
            1, 1f);
        await accelerator.SynchronizeAsync();

        var got = await outBuf.CopyToHostAsync<float>();
        if (Math.Abs(got[0] - 56f) > 1e-5f)
            throw new Exception($"SubView Scale: expected output[0]=56 got {got[0]}. " +
                "If WebGL returns 0/1, the SubView elementOffset isn't being applied to the input texelFetch. " +
                "Surfaced 2026-05-04 by Data's StyleMosaic node 55 Gather_117 first-divergent capture.");
    });

    [TestMethod]
    public async Task ShapeGather_GpuRuntimeIdx_StyleMosaicPath_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Same StyleMosaic-shape Shape->Gather scenario, but exercising the GPU runtime-indices
        // path (GatherAxis0Float) that fires when GatherOperator's TryGetInputValues(1) returns null
        // (indices are runtime tensors, not pre-read constants). This is the path my 2026-05-03
        // rc.2 view.Length fix targeted; verifying it still works for this specific shape combo.
        var gather = new SpawnDev.ILGPU.ML.Kernels.GatherKernel(accelerator);
        var shapeData = new float[] { 1f, 32f, 56f, 56f };
        var indicesData = new float[] { 2f };
        using var dataBuf = accelerator.Allocate1D(shapeData);
        using var idxBuf = accelerator.Allocate1D(indicesData);
        using var outBuf = accelerator.Allocate1D<float>(1);

        gather.GatherAxis0Float(dataBuf.View, idxBuf.View, outBuf.View,
            numIndices: 1, innerSize: 1, dataRows: 4);
        await accelerator.SynchronizeAsync();

        var got = await outBuf.CopyToHostAsync<float>();
        if (Math.Abs(got[0] - 56f) > 1e-5f)
            throw new Exception($"GatherAxis0Float (StyleMosaic-shape: data[4], idx[1]=2, axis=0): expected 56 got {got[0]}. " +
                "If WebGL returns 0, view.Length is still returning 0 for this specific Dense-1D shape, OR float-to-int index cast breaks for this layout.");
    });

    [TestMethod]
    public async Task RMSNorm_MatchesCpu() => await RunTest(async accelerator =>
    {
        int rows = 100, C = 384;
        var input = RandomFloats(rows * C, seed: 145);
        var weight = RandomFloats(C, seed: 146, scale: 0.5f);
        for (int i = 0; i < C; i++) weight[i] = MathF.Abs(weight[i]) + 0.5f;

        // CPU reference
        float eps = 1e-6f;
        var expected = new float[rows * C];
        for (int r = 0; r < rows; r++)
        {
            float sumSq = 0;
            for (int i = 0; i < C; i++)
            {
                float v = input[r * C + i];
                sumSq += v * v;
            }
            float rms = MathF.Sqrt(sumSq / C + eps);
            float invRms = 1f / rms;
            for (int i = 0; i < C; i++)
                expected[r * C + i] = input[r * C + i] * invRms * weight[i];
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(rows * C);
        using var wBuf = accelerator.Allocate1D(weight);

        var norm = new NormalizationKernels(accelerator);
        norm.RMSNorm(inBuf.View, outBuf.View, wBuf.View, rows, C);
        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outBuf.View.SubView(0, rows * C), expected, 1e-4f, "RMSNorm: ");
    });
}
