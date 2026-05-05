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

    [TestMethod(Timeout = 90000)]
    public async Task InstanceNorm_Chain24_NoHang() => await RunTest(async accelerator =>
    {
        // 2026-05-04 Data: tight repro for StyleMosaic Wasm hang at rc.16+.
        // StyleMosaic graph runs ~24 sequential InstanceNorm calls. Each call
        // allocates fresh per-call mean/invStd temp buffers, runs Pass1 then Pass2.
        //
        // Geordi's generic Pass1-write/Pass2-read pattern test (Tests23_StyleMosaicShape_PairedDispatchesScale)
        // PASSES on Wasm at rc.16 — so the GENERIC dispatch pattern is fine.
        // This test calls the actual InstanceNorm operator in a 24-deep chain
        // matching StyleMosaic. If THIS hangs on Wasm, the bug is specific to
        // either NormalizationKernels.InstanceNorm itself or its per-call temp
        // buffer allocation pattern (NOT the generic dispatcher).
        //
        // Full StyleMosaic spatial (224x224 = 50176) since 56x56 passed (2s) — testing
        // size-dependence hypothesis. If 24 chained InstanceNorms at full size hangs
        // on Wasm, the bug is shape/size scaling not pattern. If passes, the bug
        // needs other ops (Conv2D / Pad / ReLU) interleaved.
        int N = 1, C = 32, H = 224, W = 224;
        int spatial = H * W;
        int total = N * C * spatial;
        var input = RandomFloats(total, seed: 200, scale: 1f);
        var scale = RandomFloats(C, seed: 201, scale: 1f);
        var bias = RandomFloats(C, seed: 202, scale: 0.5f);
        for (int i = 0; i < C; i++) scale[i] = MathF.Abs(scale[i]) + 0.5f;

        using var bufA = accelerator.Allocate1D(input);
        using var bufB = accelerator.Allocate1D<float>(total);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var norm = new NormalizationKernels(accelerator);

        // 24 sequential calls, ping-pong A↔B as input↔output. Track which buffer
        // holds the final output via parity (24 swaps -> bufA holds the output)
        var src = bufA.View;
        var dst = bufB.View;
        for (int iter = 0; iter < 24; iter++)
        {
            norm.InstanceNorm(src, dst, sBuf.View, bBuf.View, N, C, spatial);
            (src, dst) = (dst, src); // swap
        }
        await accelerator.SynchronizeAsync();

        // After 24 swaps the FINAL src points to the most recent output. With even
        // iter count (24), src ends pointing to bufA.View (back to start). Confirm
        // by reading bufA.
        var finalBuf = bufA;
        var firstFew = await finalBuf.CopyToHostAsync<float>(0, 8);
        bool anyFinite = false;
        for (int i = 0; i < firstFew.Length; i++)
            if (!float.IsNaN(firstFew[i]) && !float.IsInfinity(firstFew[i])) anyFinite = true;
        if (!anyFinite)
            throw new Exception($"InstanceNorm chain output all NaN/Inf: [{string.Join(",", firstFew)}]");
        Console.WriteLine($"[InstanceNorm_Chain24] passed without hang. firstFew=[{string.Join(",", firstFew.Take(4).Select(v => v.ToString("F4")))}]");
    });

    [TestMethod(Timeout = 90000)]
    public async Task ConvInstanceNormReluChain24_NoHang() => await RunTest(async accelerator =>
    {
        // 2026-05-04 Data: tighter repro for StyleMosaic Wasm hang at rc.16+.
        // Chain24 of just InstanceNorm passes (3s on Wasm at full size). Adding
        // Conv2D 3x3 + ReLU between InstanceNorms tests whether the bug is in
        // the Conv2D + InstanceNorm + ReLU interaction at kernel level.
        // If THIS hangs, kernel-level interaction is the issue. If passes,
        // bug needs the GraphExecutor/InferenceSession/BufferPool runtime.
        int N = 1, C = 32, H = 56, W = 56;  // smaller for faster verify, still triggering
        int spatial = H * W;
        int total = N * C * spatial;
        int kSize = 9; // 3*3 kernel
        int weightCount = C * C * kSize;

        var input = RandomFloats(total, seed: 300, scale: 1f);
        var convWeight = RandomFloats(weightCount, seed: 301, scale: 0.1f);
        var convBias = RandomFloats(C, seed: 302, scale: 0.01f);
        var scale = RandomFloats(C, seed: 303, scale: 1f);
        var bias = RandomFloats(C, seed: 304, scale: 0.5f);
        for (int i = 0; i < C; i++) scale[i] = MathF.Abs(scale[i]) + 0.5f;

        using var bufA = accelerator.Allocate1D(input);
        using var bufB = accelerator.Allocate1D<float>(total);
        using var convOut = accelerator.Allocate1D<float>(total);
        using var wBuf = accelerator.Allocate1D(convWeight);
        using var convBiasBuf = accelerator.Allocate1D(convBias);
        using var sBuf = accelerator.Allocate1D(scale);
        using var bBuf = accelerator.Allocate1D(bias);

        var conv = new Conv2DKernel(accelerator);
        var norm = new NormalizationKernels(accelerator);
        var ew = new ElementWiseKernels(accelerator);

        var src = bufA.View;
        var dst = bufB.View;
        for (int iter = 0; iter < 24; iter++)
        {
            // Conv2D 3x3 stride=1 pad=1 (same-shape)
            conv.Forward(src, wBuf.View, convBiasBuf.View, convOut.View,
                C, H, W, C, 3, 3, 1, 1);
            // InstanceNorm
            norm.InstanceNorm(convOut.View, dst, sBuf.View, bBuf.View, N, C, spatial);
            // ReLU in-place on dst
            ew.ReLUInPlace(dst, total);
            (src, dst) = (dst, src);
        }
        await accelerator.SynchronizeAsync();

        var finalBuf = bufA;
        var firstFew = await finalBuf.CopyToHostAsync<float>(0, 8);
        bool anyFinite = false;
        for (int i = 0; i < firstFew.Length; i++)
            if (!float.IsNaN(firstFew[i]) && !float.IsInfinity(firstFew[i])) anyFinite = true;
        if (!anyFinite)
            throw new Exception($"Conv-IN-ReLU chain output all NaN/Inf: [{string.Join(",", firstFew)}]");
        Console.WriteLine($"[ConvInstanceNormReluChain24] passed without hang. firstFew=[{string.Join(",", firstFew.Take(4).Select(v => v.ToString("F4")))}]");
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
    public async Task ShapeGatherScale_AfterReuse_StyleMosaicShape_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Repro for Data's WebGL Gather bug (2026-05-04). His full-element capture pinned
        // the divergence to Gather receiving a buffer-pool-reused buffer whose CopyFromCPU
        // path on WebGL doesn't update the texture properly when followed by a SubView read
        // at non-zero offset. My fresh-Allocate1D ShapeGatherScale test passed; this one
        // mimics the pool reuse pattern: pre-fill the buffer (simulating a prior op),
        // overwrite via CopyFromCPU (Shape's path), then Scale(SubView(2, 1), ...).
        var op = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);

        // Step 1: Allocate buffer and pre-populate with non-zero values (simulates a prior op
        // having written data, possibly through a kernel dispatch + texture upload).
        using var dataBuf = accelerator.Allocate1D(new float[] { 99f, 99f, 99f, 99f });
        await accelerator.SynchronizeAsync(); // force the prior write/upload to settle

        // Step 2: Write Shape's [1, 32, 56, 56] over the same buffer via CopyFromCPU.
        var shapeData = new float[] { 1f, 32f, 56f, 56f };
        dataBuf.View.SubView(0, 4).CopyFromCPU(shapeData);

        using var outBuf = accelerator.Allocate1D<float>(1);

        // Step 3: Mimic Gather: Scale(buf.SubView(2, 1), output, 1, 1f).
        op.Scale(
            dataBuf.View.SubView(2, 1),
            outBuf.View.SubView(0, 1),
            1, 1f);
        await accelerator.SynchronizeAsync();

        var got = await outBuf.CopyToHostAsync<float>();
        if (Math.Abs(got[0] - 56f) > 1e-5f)
            throw new Exception($"After CopyFromCPU over a pre-populated buffer, SubView Scale: " +
                $"expected 56 got {got[0]}. " +
                $"If WebGL returns 99 (stale), the texture upload after CopyFromCPU isn't firing. " +
                $"If WebGL returns 0, the SubView elementOffset path is broken.");
    });

    [TestMethod]
    public async Task ShapeGatherScale_AfterKernelWrite_StyleMosaicShape_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Tighter repro for Data's WebGL Gather bug. The pre-CopyFromCPU step here is
        // a KERNEL DISPATCH that writes via Transform Feedback (the path Conv/Shape
        // upstream of Gather actually exercises in production). After that the GL
        // worker's entry.data + texture are populated by the worker-side TF readback.
        // Then ShapeOp does CopyFromCPU which writes the .NET _backingArray and sets
        // NeedsUpload=true, expecting the next EnsureBufferInWorker to re-upload.
        // If the kernel-write + CopyFromCPU sequence breaks the texture state on
        // WebGL, this reproduces the production bug.
        var op = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);

        // Step 1: Allocate buffer and pre-populate via KERNEL (Scale by 99 of zeros).
        using var donor = accelerator.Allocate1D(new float[] { 1f, 1f, 1f, 1f });
        using var dataBuf = accelerator.Allocate1D<float>(4);
        op.Scale(donor.View, dataBuf.View, 4, 99f); // dataBuf = [99,99,99,99] via kernel
        await accelerator.SynchronizeAsync();

        // Step 2: ShapeOp's CopyFromCPU pattern — overwrite via .NET CPU path.
        var shapeData = new float[] { 1f, 32f, 56f, 56f };
        dataBuf.View.SubView(0, 4).CopyFromCPU(shapeData);

        using var outBuf = accelerator.Allocate1D<float>(1);

        // Step 3: GatherOp's Scale with non-zero SubView offset.
        op.Scale(
            dataBuf.View.SubView(2, 1),
            outBuf.View.SubView(0, 1),
            1, 1f);
        await accelerator.SynchronizeAsync();

        var got = await outBuf.CopyToHostAsync<float>();
        if (Math.Abs(got[0] - 56f) > 1e-5f)
            throw new Exception($"Kernel-write -> CopyFromCPU -> SubView Scale: " +
                $"expected 56 got {got[0]}. " +
                $"99 = stale kernel-write data; 0 = SubView offset path broken; other = surprise.");
    });

    [TestMethod]
    public async Task SubViewOffset_NonZeroThenZero_NoLeakAcrossDispatches() => await RunTest(async accelerator =>
    {
        // Specific repro for the WebGL glWorker.js bug surfaced 2026-05-04 by Data's
        // StyleMosaic node 55 Gather first-divergent capture. The dispatcher's offset
        // uniform was only being set when elementOffset != 0; for elementOffset == 0
        // it was skipped, leaving the uniform at whatever the previous dispatch with
        // the same kernel program had set. Two same-program dispatches: first with
        // SubView(2, 1) sets uniform to 2; second with SubView(0, 1) on a different
        // buffer reads at offset 2 instead of 0 because the uniform leaked.
        var op = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);

        // Two distinct buffers, deterministic content.
        using var bufA = accelerator.Allocate1D(new float[] { 100f, 200f, 300f, 400f });
        using var bufB = accelerator.Allocate1D(new float[] { 11f, 22f, 33f, 44f });
        using var outA = accelerator.Allocate1D<float>(1);
        using var outB = accelerator.Allocate1D<float>(1);

        // Dispatch 1: Scale(bufA.SubView(2, 1), outA, 1, 1f) — reads bufA[2]=300.
        // Sets the Scale program's u_paramX_offset uniform to 2.
        op.Scale(bufA.View.SubView(2, 1), outA.View.SubView(0, 1), 1, 1f);

        // Dispatch 2: Scale(bufB.SubView(0, 1), outB, 1, 1f) — should read bufB[0]=11.
        // If glWorker skips setting the offset uniform when elementOffset==0,
        // the uniform retains 2 from Dispatch 1. The kernel then reads bufB[0+2]=33.
        op.Scale(bufB.View.SubView(0, 1), outB.View.SubView(0, 1), 1, 1f);

        await accelerator.SynchronizeAsync();

        var gotA = await outA.CopyToHostAsync<float>();
        var gotB = await outB.CopyToHostAsync<float>();

        if (Math.Abs(gotA[0] - 300f) > 1e-5f)
            throw new Exception($"Dispatch 1 (SubView(2,1) of bufA): expected 300 got {gotA[0]}");

        if (Math.Abs(gotB[0] - 11f) > 1e-5f)
            throw new Exception($"Dispatch 2 (SubView(0,1) of bufB) after a non-zero-offset dispatch: " +
                $"expected 11 got {gotB[0]}. " +
                $"If WebGL returns 33, the offset uniform from Dispatch 1 leaked into Dispatch 2 " +
                $"because glWorker.js skipped setting u_paramX_offset when elementOffset==0. " +
                $"Surfaced by Data 2026-05-04 in StyleMosaic Gather (node 55).");
    });

    [TestMethod]
    public async Task Softmax_YOLOv8DflFullPathWithExtraScale_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Data's verify experiment 2026-05-04 ~13:30: production `SoftmaxOperator.Execute` does
        // an EXTRA `reg.ElementWise.Scale(inputs, outputs, count, 1f)` BEFORE the
        // Transpose+Softmax+Transpose chain. My existing `Softmax_YOLOv8DflFullPath_MatchesCpu`
        // test (which passed on Wasm) skipped that step. If the Scale leaves transient state
        // that affects the subsequent Transpose dispatch on Wasm, this test reproduces the
        // YOLOv8 Wasm node 220 first-divergent symptom Data captured.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 182, scale: 4f);

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

        // Production-mimicking flow: separate `inputs` and `outputs` buffers (rented in real
        // graph; here freshly-allocated to keep the bug surface narrow), Scale to copy
        // input→output, then Transpose+Softmax+Transpose all over the SAME `outputs` buffer.
        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var outputBuf = accelerator.Allocate1D<float>(total);
        using var transposedBuf = accelerator.Allocate1D<float>(total);

        var ew = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);
        var sm = new SoftmaxKernel(accelerator);

        // Step 1 — extra Scale that the production SoftmaxOperator does (line 82).
        ew.Scale(inputBuf.View, outputBuf.View, total, 1f);

        // Step 2 — first Transpose READS outputBuf (just-written by Scale).
        transpose.Transpose(outputBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });

        // Step 3 — Softmax on rows of axisDim.
        sm.Forward(transposedBuf.View, outer * inner, axisDim);

        // Step 4 — second Transpose WRITES outputBuf (same buffer the first Transpose read from).
        transpose.Transpose(transposedBuf.View, outputBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outputBuf.View.SubView(0, total), expected, 1e-4f,
            "Softmax YOLOv8 DFL [4,16,8400] axis=1 with extra Scale: ");
    });

    [TestMethod]
    public async Task ScaleThenTranspose_LargeBuffer_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Bisect Data's Wasm Softmax-with-Scale failure (2026-05-04). Production
        // SoftmaxOperator chains: Scale(input -> output) -> Transpose(output -> tmp) ->
        // Softmax(tmp) -> Transpose(tmp -> output). The ExtraScale variant of my Softmax
        // test FAILS on Wasm. This narrows further: skip the Softmax+secondTranspose,
        // do JUST Scale + first Transpose. If THIS fails, the bug is in
        // Wasm's Scale-then-Transpose sequencing on a shared buffer.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 183, scale: 4f);

        // CPU reference: identity Scale (1f) then transpose [outer, axisDim, inner] -> [outer, inner, axisDim]
        var expected = new float[total];
        for (int o = 0; o < outer; o++)
            for (int a = 0; a < axisDim; a++)
                for (int i = 0; i < inner; i++)
                {
                    int srcIdx = o * axisDim * inner + a * inner + i;
                    int dstIdx = o * inner * axisDim + i * axisDim + a;
                    expected[dstIdx] = input[srcIdx]; // Scale by 1 = identity
                }

        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var scaledBuf = accelerator.Allocate1D<float>(total);
        using var transposedBuf = accelerator.Allocate1D<float>(total);

        var ew = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);

        ew.Scale(inputBuf.View, scaledBuf.View, total, 1f);
        transpose.Transpose(scaledBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, transposedBuf.View.SubView(0, total), expected, 1e-5f,
            "Scale-then-Transpose [4,16,8400]: ");
    });

    [TestMethod]
    public async Task Scale_Transpose_Softmax_NoSecondTranspose_MatchesCpu() => await RunTest(async accelerator =>
    {
        // Bisect step 2: Scale + Transpose1 + Softmax. Stops short of the second
        // Transpose. If THIS fails on Wasm but ScaleThenTranspose passes, the bug is
        // triggered by Softmax-after-Transpose. If THIS passes, the bug needs the
        // second Transpose writing back to outputBuf (which is also Scale's output).
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 184, scale: 4f);

        // CPU reference: transposed [outer, inner, axisDim] then row-wise softmax over axisDim.
        var expected = new float[total];
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++)
            {
                int rowBase = o * inner * axisDim + i * axisDim;
                float max = float.MinValue;
                for (int a = 0; a < axisDim; a++)
                    max = MathF.Max(max, input[o * axisDim * inner + a * inner + i]);
                double sum = 0;
                for (int a = 0; a < axisDim; a++)
                    sum += MathF.Exp(input[o * axisDim * inner + a * inner + i] - max);
                float invSum = (float)(1.0 / sum);
                for (int a = 0; a < axisDim; a++)
                    expected[rowBase + a] = MathF.Exp(input[o * axisDim * inner + a * inner + i] - max) * invSum;
            }

        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var scaledBuf = accelerator.Allocate1D<float>(total);
        using var transposedBuf = accelerator.Allocate1D<float>(total);

        var ew = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);
        var sm = new SoftmaxKernel(accelerator);

        ew.Scale(inputBuf.View, scaledBuf.View, total, 1f);
        transpose.Transpose(scaledBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });
        sm.Forward(transposedBuf.View, outer * inner, axisDim);

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, transposedBuf.View.SubView(0, total), expected, 1e-4f,
            "Scale-Transpose-Softmax (no second Transpose) [4,16,8400]: ");
    });

    [TestMethod]
    public async Task Scale_T1_T2_NoSoftmax_OutputBufReusedAcrossDispatches() => await RunTest(async accelerator =>
    {
        // Bisect step 3: Scale + T1 + T2 (no Softmax). All other variants narrowed:
        //   Scale + T1 (no Softmax, no T2):       PASS — Scale-then-Transpose works
        //   Scale + T1 + Softmax (no T2):         PASS — adding Softmax doesn't break it
        //   Scale + T1 + Softmax + T2:            FAIL on Wasm — production triggers the bug
        // If THIS variant (T2 directly after T1, no Softmax) FAILS, the bug is in
        // two Transpose dispatches sharing outputBuf as both write target and prior input.
        // If it PASSES, the bug needs Softmax-then-Transpose specifically.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 185, scale: 4f);

        // CPU reference: identity Scale, transpose [outer,axisDim,inner] -> [outer,inner,axisDim],
        // then transpose back [outer,inner,axisDim] -> [outer,axisDim,inner] = identity overall.
        // So expected = input.
        var expected = (float[])input.Clone();

        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var outputBuf = accelerator.Allocate1D<float>(total);
        using var transposedBuf = accelerator.Allocate1D<float>(total);

        var ew = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);

        ew.Scale(inputBuf.View, outputBuf.View, total, 1f);
        transpose.Transpose(outputBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });
        // Skip Softmax — go directly to second Transpose
        transpose.Transpose(transposedBuf.View, outputBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outputBuf.View.SubView(0, total), expected, 1e-5f,
            "Scale + T1 + T2 (no Softmax) [4,16,8400]: ");
    });

    [TestMethod]
    public async Task T1_T2_SharedOutputBuf_NoScale() => await RunTest(async accelerator =>
    {
        // Bisect step 4: just T1 + T2 (no Scale, no Softmax). T1 reads inputBuf and
        // writes transposedBuf. T2 reads transposedBuf and writes outputBuf. Different
        // buffers all the way - no buffer is reused as input AND output across the chain.
        // If THIS fails on Wasm, the bug is in two consecutive Transpose dispatches with
        // shared paramsBuf (the only buffer touched by both).
        // If THIS passes, the bug requires a buffer to be in BOTH dispatches' bufferInfos
        // (the outputBuf in Scale + T1 + T2).
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 186, scale: 4f);

        // CPU reference: transpose [outer, axisDim, inner] -> [outer, inner, axisDim] then
        // transpose back -> identity.
        var expected = (float[])input.Clone();

        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var transposedBuf = accelerator.Allocate1D<float>(total);
        using var outputBuf = accelerator.Allocate1D<float>(total);

        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);

        transpose.Transpose(inputBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });
        transpose.Transpose(transposedBuf.View, outputBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outputBuf.View.SubView(0, total), expected, 1e-5f,
            "T1+T2 (no Scale, no shared buffer reuse) [4,16,8400]: ");
    });

    [TestMethod]
    public async Task Scale_T1_T2_FreshOutput_NoSharedBuffer() => await RunTest(async accelerator =>
    {
        // Bisect step 5: Scale + T1 + T2 but T2 writes to a FRESH 4th buffer (not the
        // Scale-output / T1-input buffer). All the same dispatches, but no buffer is
        // shared across all 3.
        //
        // Prior bisect:
        //   Scale + T1 + T2 (T2 -> outputBuf which is also Scale's output): FAIL maxErr 8.0
        //   T1 + T2 alone (no Scale, no shared buffer): PASS
        //
        // If THIS variant (Scale + T1 + T2_to_fresh_buffer) PASSES, the bug REQUIRES
        // T2 writing to outputBuf which is also in Scale's bufferInfos.
        // If FAILS, the bug is just "3 dispatches sharing a buffer" regardless of
        // where T2 writes.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 187, scale: 4f);

        var expected = (float[])input.Clone(); // identity

        using var inputBuf = accelerator.Allocate1D((float[])input.Clone());
        using var scaledBuf = accelerator.Allocate1D<float>(total);     // Scale writes here
        using var transposedBuf = accelerator.Allocate1D<float>(total); // T1 writes here, T2 reads
        using var freshOutBuf = accelerator.Allocate1D<float>(total);   // T2 writes HERE (NEW buffer)

        var ew = new SpawnDev.ILGPU.ML.ElementWiseKernels(accelerator);
        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);

        ew.Scale(inputBuf.View, scaledBuf.View, total, 1f);
        transpose.Transpose(scaledBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });
        // T2 writes to freshOutBuf instead of scaledBuf
        transpose.Transpose(transposedBuf.View, freshOutBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, freshOutBuf.View.SubView(0, total), expected, 1e-5f,
            "Scale + T1 + T2(fresh output) [4,16,8400]: ");
    });

    [TestMethod]
    public async Task CopyFromCPU_T1_T2_NoKernelBeforeT1() => await RunTest(async accelerator =>
    {
        // Bisect step 6: replace Scale with CopyFromCPU (host write), then T1 + T2.
        //
        // Prior:
        //   T1 + T2 alone (T1 reads inputBuf, written via Allocate1D ctor):  PASS
        //   Scale + T1 + T2 (T1 reads scaledBuf, written by Scale kernel):   FAIL maxErr 8.0
        //
        // If CopyFromCPU + T1 + T2 (host writes scaledBuf, T1 reads it) PASSES,
        // the bug is specifically about KERNEL-WRITTEN buffer being read by next dispatch.
        // If FAILS, the bug is just "3 dispatches with shared buffer state" regardless.
        int outer = 4, axisDim = 16, inner = 8400;
        int total = outer * axisDim * inner;
        var input = RandomFloats(total, seed: 188, scale: 4f);

        var expected = (float[])input.Clone();

        using var scaledBuf = accelerator.Allocate1D<float>(total);
        using var transposedBuf = accelerator.Allocate1D<float>(total);
        using var outputBuf = accelerator.Allocate1D<float>(total);

        var transpose = new SpawnDev.ILGPU.ML.Kernels.TransposeKernel(accelerator);

        // Replace Scale with direct host CopyFromCPU
        scaledBuf.View.CopyFromCPU(input);

        transpose.Transpose(scaledBuf.View, transposedBuf.View,
            new[] { outer, axisDim, inner }, new[] { 0, 2, 1 });
        transpose.Transpose(transposedBuf.View, outputBuf.View,
            new[] { outer, inner, axisDim }, new[] { 0, 2, 1 });

        await accelerator.SynchronizeAsync();

        await AssertCloseGpu(accelerator, outputBuf.View.SubView(0, total), expected, 1e-5f,
            "CopyFromCPU + T1 + T2 [4,16,8400]: ");
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
