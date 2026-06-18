using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// f16-native weights/compute tests (task: f16). The spike (2026-06-05) settled the storage architecture
/// across all 6 backends: native <c>ILGPU.Half</c> storage + fp32 compute WORKS everywhere (incl.
/// WebGPU/WGSL) — store weights as ILGPU.Half (half the bytes), read + upconvert to float, accumulate fp32
/// (ORT-style mixed precision; no accuracy loss).
///
/// UPDATE (2026-06-17): the original spike's "generic-math kernels FAIL everywhere (BitCast intrinsic)"
/// finding is SUPERSEDED by Geordi's <c>ILGPU.PrecisionConvert</c> (4.13.0-local.9+) — a single generic
/// <c>where T : unmanaged, INumber&lt;T&gt;</c> kernel using <c>PrecisionConvert.ConvertToSingle(B[i])</c>
/// now transpiles + runs bit-exact on all 6 backends for Half / BFloat16 / Float8E*. The half-weight matmul
/// kernels are now ONE generic <c>MatMulLowPWeight&lt;T&gt;</c> (MatMulHalfWeight = the T=Half wrapper), so
/// bf16/fp8 weights stay native too (no f32 temp). See <c>F16_MatMulBFloat16Weight_MatchesFp32Reference</c>.
/// </summary>
public abstract partial class MLTestBase
{
    // f16 matmul foundation primitive: ILGPU.Half storage in, fp32 compute + fp32 out.
    private static void F16_HalfToFloatMul(
        Index1D i, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> a, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> r)
        => r[i] = (float)a[i] * (float)b[i] + (float)a[i];

    /// <summary>Capability guard: ILGPU.Half storage + fp32 compute works on every backend (the f16 foundation).</summary>
    [TestMethod]
    public Task F16_IlgpuHalf_StorageAndFp32Compute() => RunTest(async accelerator =>
    {
        // a=[1,2,3,4] b=[10,20,30,40] -> a*b+a = [11,42,93,164] (all exact in fp16, integers < 2048).
        var expected = new[] { 11f, 42f, 93f, 164f };
        using (var a = accelerator.Allocate1D(new[] { (global::ILGPU.Half)1f, (global::ILGPU.Half)2f, (global::ILGPU.Half)3f, (global::ILGPU.Half)4f }))
        using (var b = accelerator.Allocate1D(new[] { (global::ILGPU.Half)10f, (global::ILGPU.Half)20f, (global::ILGPU.Half)30f, (global::ILGPU.Half)40f }))
        using (var r = accelerator.Allocate1D<float>(4))
        {
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
                F16_HalfToFloatMul);
            k((int)r.Length, a.View, b.View, r.View);
            await accelerator.SynchronizeAsync();
            var got = await r.CopyToHostAsync<float>(0, 4);
            for (int i = 0; i < 4; i++)
                if (MathF.Abs(got[i] - expected[i]) > 1e-3f)
                    throw new Exception($"ILGPU.Half storage+fp32 compute [{i}]={got[i]}, expected {expected[i]}");
        }
    });

    /// <summary>
    /// Production path: MatMulKernel.MatMulHalfWeight (fp16 weights, fp32 accumulate) matches a fp32
    /// reference computed with the SAME fp16-rounded weights — so this isolates KERNEL correctness from
    /// the (expected, separate) fp16 rounding cost. Proves the f16-weight matmul is numerically right.
    /// </summary>
    [TestMethod]
    public Task F16_MatMulHalfWeight_MatchesFp32Reference() => RunTest(async accelerator =>
    {
        int M = 8, K = 16, N = 8;
        var rng = new Random(42);
        var a = new float[M * K];
        var bf = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bf.Length; i++) bf[i] = (float)(rng.NextDouble() * 2 - 1);

        // fp16-rounded weights (what the GPU actually reads), and the fp32 reference using THOSE weights.
        var bHalf = new global::ILGPU.Half[bf.Length];
        for (int i = 0; i < bf.Length; i++) bHalf[i] = (global::ILGPU.Half)bf[i];
        var cpuC = new float[M * N];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                float s = 0f;
                for (int k = 0; k < K; k++)
                    s += a[r * K + k] * (float)bHalf[k * N + c];
                cpuC[r * N + c] = s;
            }

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(bHalf);
        using var cBuf = accelerator.Allocate1D<float>(M * N);
        var mm = new MatMulKernel(accelerator);
        mm.MatMulHalfWeight(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0f;
        for (int i = 0; i < cpuC.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
        if (maxErr > 1e-3f)
            throw new Exception($"MatMulHalfWeight maxErr={maxErr:E3} vs fp16-weight fp32 reference (expected < 1e-3)");
    });

    /// <summary>
    /// The generic native-low-p weight path on a SECOND type: MatMulKernel.MatMulLowPWeight&lt;BFloat16&gt;
    /// (the same generic kernel MatMulHalfWeight uses with T=Half) reads bf16 weights NATIVELY and matches a
    /// fp32 reference computed with the SAME bf16-rounded weights — isolating kernel correctness from bf16
    /// rounding. Proves the no-needless-conversion generalization: bf16 weights stay native (no f32 temp),
    /// converted in-register via PrecisionConvert, on all 6 backends. (Pre-PrecisionConvert this generic
    /// kernel could not compile; see the class doc.)
    /// </summary>
    [TestMethod]
    public Task F16_MatMulBFloat16Weight_MatchesFp32Reference() => RunTest(async accelerator =>
    {
        int M = 8, K = 16, N = 8;
        var rng = new Random(42);
        var a = new float[M * K];
        var bf = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bf.Length; i++) bf[i] = (float)(rng.NextDouble() * 2 - 1);

        // bf16-rounded weights (what the GPU actually reads), and the fp32 reference using THOSE weights.
        var bBf16 = new global::ILGPU.BFloat16[bf.Length];
        for (int i = 0; i < bf.Length; i++) bBf16[i] = (global::ILGPU.BFloat16)bf[i];
        var cpuC = new float[M * N];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                float s = 0f;
                for (int k = 0; k < K; k++)
                    s += a[r * K + k] * (float)bBf16[k * N + c];
                cpuC[r * N + c] = s;
            }

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(bBf16);
        using var cBuf = accelerator.Allocate1D<float>(M * N);
        var mm = new MatMulKernel(accelerator);
        mm.MatMulLowPWeight(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

        // bf16->f32 widening is lossless and accumulation is fp32, so the GPU must match the CPU ref that used
        // the same bf16-rounded weights to ~kernel precision (same bar as the fp16 test).
        float maxErr = 0f;
        for (int i = 0; i < cpuC.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
        if (maxErr > 1e-3f)
            throw new Exception($"MatMulLowPWeight<BFloat16> maxErr={maxErr:E3} vs bf16-weight fp32 reference (expected < 1e-3)");
    });

    /// <summary>
    /// Slice 2+3: BufferPool.AllocateHalfWeightFromStreamAsync loads an fp16 weight from a stream (like an
    /// ONNX fp16 initializer) into a HalfTensor (half the GPU bytes), then MatMulHalfWeight consumes it.
    /// Verifies the LOAD+COMPUTE path matches a fp32 reference using the same fp16-rounded weights.
    /// </summary>
    [TestMethod]
    public Task F16_BufferPoolHalfWeight_LoadAndMatMul() => RunTest(async accelerator =>
    {
        int M = 4, K = 8, N = 4;
        var rng = new Random(7);
        var a = new float[M * K];
        var w = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        // Serialize the weight as fp16 bytes (dtype 10), exactly like an ONNX FLOAT16 initializer.
        var wBytes = new byte[w.Length * 2];
        for (int i = 0; i < w.Length; i++)
        {
            var bits = BitConverter.GetBytes((System.Half)w[i]);
            wBytes[i * 2] = bits[0];
            wBytes[i * 2 + 1] = bits[1];
        }
        using var ms = new System.IO.MemoryStream(wBytes);

        var pool = new SpawnDev.ILGPU.ML.Tensors.BufferPool(accelerator);
        try
        {
            var halfW = await pool.AllocateHalfWeightFromStreamAsync(ms, 0, wBytes.Length, 10, new[] { K, N });

            // CPU reference: A × (fp16-rounded W), fp32 accumulate.
            var cpuC = new float[M * N];
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < K; k++)
                        s += a[r * K + k] * (float)(System.Half)w[k * N + c];
                    cpuC[r * N + c] = s;
                }

            using var aBuf = accelerator.Allocate1D(a);
            using var cBuf = accelerator.Allocate1D<float>(M * N);
            var mm = new MatMulKernel(accelerator);
            mm.MatMulHalfWeight(aBuf.View, halfW.Data, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();
            var gpuC = await cBuf.CopyToHostAsync<float>(0, M * N);

            float maxErr = 0f;
            for (int i = 0; i < cpuC.Length; i++)
                maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
            if (maxErr > 1e-3f)
                throw new Exception($"BufferPool half-weight load+matmul maxErr={maxErr:E3} (expected < 1e-3)");
        }
        finally { pool.Dispose(); }
    });

    /// <summary>
    /// Slice 4: the MatMul OPERATOR routes a half-backed weight (Tensor.FromHalf — fp16, no float buffer)
    /// to the half-weight kernel. Verifies the executor-carries-half-tensors path end to end: fp16 weight
    /// -> HalfTensor -> Tensor.FromHalf -> MatMulOperator.Execute -> MatMulHalfWeight, matching a fp32
    /// reference with the same fp16-rounded weights.
    /// </summary>
    [TestMethod]
    public Task F16_MatMulOperator_RoutesHalfWeight() => RunTest(async accelerator =>
    {
        int M = 4, K = 8, N = 4;
        var rng = new Random(11);
        var a = new float[M * K];
        var w = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        var pool = new BufferPool(accelerator);
        try
        {
            var wBytes = new byte[w.Length * 2];
            for (int i = 0; i < w.Length; i++) { var bb = BitConverter.GetBytes((System.Half)w[i]); wBytes[i * 2] = bb[0]; wBytes[i * 2 + 1] = bb[1]; }
            using var ms = new System.IO.MemoryStream(wBytes);
            var halfTensor = await pool.AllocateHalfWeightFromStreamAsync(ms, 0, wBytes.Length, 10, new[] { K, N });
            var bHalf = Tensor.FromHalf(halfTensor);
            if (!bHalf.IsHalf) throw new Exception("Tensor.FromHalf should set IsHalf=true");

            using var aBuf = accelerator.Allocate1D(a);
            using var outBuf = accelerator.Allocate1D<float>(M * N);
            var aT = new Tensor(aBuf.View, new[] { M, K }, "a");
            var outT = new Tensor(outBuf.View, new[] { M, N }, "out");

            var registry = new OperatorRegistry(accelerator);
            var op = new MatMulOperator(registry);
            var ctx = new OnnxOpContext
            {
                Inputs = new[] { aT, bHalf },
                Outputs = new[] { outT },
                Attributes = new Dictionary<string, object>(),
                Pool = pool,
                InputNames = new[] { "a", "b" },
                Registry = registry,
            };
            op.Execute(ctx);
            await accelerator.SynchronizeAsync();
            var gpuC = await outBuf.CopyToHostAsync<float>(0, M * N);

            var cpuC = new float[M * N];
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < K; k++)
                        s += a[r * K + k] * (float)(System.Half)w[k * N + c];
                    cpuC[r * N + c] = s;
                }
            float maxErr = 0f;
            for (int i = 0; i < cpuC.Length; i++)
                maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
            if (maxErr > 1e-3f)
                throw new Exception($"MatMul operator half-weight routing maxErr={maxErr:E3} (expected < 1e-3)");
        }
        finally { pool.Dispose(); }
    });

    /// <summary>
    /// The op-dispatch generalization: MatMulOperator.Execute routes a bf16-backed weight (Tensor.FromLowP&lt;
    /// BFloat16&gt;, DType=BFloat16) through LowPWeightDispatch -> MatMulLowPWeight&lt;BFloat16&gt;, NOT the
    /// fp32 path (which would read the empty .Data). Proves dispatch keys on the real DType, not the old
    /// IsHalf bool - so a non-fp16 low-p weight stays native end to end. Matches a fp32 ref with the same
    /// bf16-rounded weights.
    /// </summary>
    [TestMethod]
    public Task F16_MatMulOperator_RoutesBFloat16Weight() => RunTest(async accelerator =>
    {
        int M = 4, K = 8, N = 4;
        var rng = new Random(11);
        var a = new float[M * K];
        var w = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        var pool = new BufferPool(accelerator);
        try
        {
            var wBf16 = new global::ILGPU.BFloat16[w.Length];
            for (int i = 0; i < w.Length; i++) wBf16[i] = (global::ILGPU.BFloat16)w[i];
            using var wBuf = accelerator.Allocate1D(wBf16);
            var bLowP = Tensor.FromLowP(wBuf.View, TensorDataType.BFloat16, new[] { K, N }, "b");
            if (bLowP.DType != TensorDataType.BFloat16) throw new Exception("FromLowP should set DType=BFloat16");
            if (bLowP.IsHalf) throw new Exception("a bf16 weight must NOT report IsHalf (that is fp16-only)");

            using var aBuf = accelerator.Allocate1D(a);
            using var outBuf = accelerator.Allocate1D<float>(M * N);
            var aT = new Tensor(aBuf.View, new[] { M, K }, "a");
            var outT = new Tensor(outBuf.View, new[] { M, N }, "out");

            var registry = new OperatorRegistry(accelerator);
            var op = new MatMulOperator(registry);
            var ctx = new OnnxOpContext
            {
                Inputs = new[] { aT, bLowP },
                Outputs = new[] { outT },
                Attributes = new Dictionary<string, object>(),
                Pool = pool,
                InputNames = new[] { "a", "b" },
                Registry = registry,
            };
            op.Execute(ctx);
            await accelerator.SynchronizeAsync();
            var gpuC = await outBuf.CopyToHostAsync<float>(0, M * N);

            var cpuC = new float[M * N];
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < K; k++)
                        s += a[r * K + k] * (float)wBf16[k * N + c];
                    cpuC[r * N + c] = s;
                }
            float maxErr = 0f;
            for (int i = 0; i < cpuC.Length; i++)
                maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
            if (maxErr > 1e-3f)
                throw new Exception($"MatMul operator bf16-weight routing maxErr={maxErr:E3} (expected < 1e-3)");
        }
        finally { pool.Dispose(); }
    });

    /// <summary>
    /// Slice 5: Conv2DKernel.ForwardHalfWeight (fp16 filter, fp32 accumulate) — Conv is the UNet's bulk, so
    /// this is the big SD-Turbo memory win. Matches a fp32 reference conv computed with the same fp16-rounded
    /// weights (isolates kernel correctness). NCHW, asymmetric-pad-capable, double accumulate like the float kernel.
    /// </summary>
    [TestMethod]
    public Task F16_Conv2DHalfWeight_MatchesFp32Reference() => RunTest(async accelerator =>
    {
        int inC = 2, inH = 5, inW = 5, outC = 3, kH = 3, kW = 3, stride = 1, pad = 1;
        int outH = (inH + 2 * pad - kH) / stride + 1;
        int outW = (inW + 2 * pad - kW) / stride + 1;
        var rng = new Random(13);
        var input = new float[inC * inH * inW];
        var wf = new float[outC * inC * kH * kW];
        var bias = new float[outC];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 2 - 1);

        var wHalfBytes = new byte[wf.Length * 2];
        var wRounded = new float[wf.Length];
        for (int i = 0; i < wf.Length; i++)
        {
            var hb = (System.Half)wf[i];
            var bb = BitConverter.GetBytes(hb);
            wHalfBytes[i * 2] = bb[0]; wHalfBytes[i * 2 + 1] = bb[1];
            wRounded[i] = (float)hb;
        }

        var cpuOut = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = bias[oc];
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= inH || ix < 0 || ix >= inW) continue;
                                sum += (double)input[ic * inH * inW + iy * inW + ix]
                                     * (double)wRounded[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                            }
                    cpuOut[oc * outH * outW + oy * outW + ox] = (float)sum;
                }

        var pool = new BufferPool(accelerator);
        try
        {
            using var ms = new System.IO.MemoryStream(wHalfBytes);
            var halfW = await pool.AllocateHalfWeightFromStreamAsync(ms, 0, wHalfBytes.Length, 10, new[] { outC, inC, kH, kW });
            using var inBuf = accelerator.Allocate1D(input);
            using var biasBuf = accelerator.Allocate1D(bias);
            using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);
            var conv = new Conv2DKernel(accelerator);
            conv.ForwardHalfWeight(inBuf.View, halfW.Data, biasBuf.View, outBuf.View, inC, inH, inW, outC, kH, kW, stride, pad);
            await accelerator.SynchronizeAsync();
            var gpuOut = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);

            float maxErr = 0f;
            for (int i = 0; i < cpuOut.Length; i++)
                maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - cpuOut[i]));
            if (maxErr > 1e-3f)
                throw new Exception($"Conv2D half-weight maxErr={maxErr:E3} vs fp16-weight fp32 reference (expected < 1e-3)");
        }
        finally { pool.Dispose(); }
    });

    /// <summary>
    /// The generic native-low-p Conv weight path on a SECOND type: Conv2DKernel.ForwardPaddedLowPWeight&lt;
    /// BFloat16&gt; (the same generic kernel ForwardHalfWeight uses with T=Half) reads bf16 filter weights
    /// NATIVELY and matches a fp32 reference conv computed with the SAME bf16-rounded weights. Proves the
    /// no-needless-conversion generalization on Conv: bf16 weights stay native (no f32 temp), converted
    /// in-register via PrecisionConvert, on all 6 backends.
    /// </summary>
    [TestMethod]
    public Task F16_Conv2DBFloat16Weight_MatchesFp32Reference() => RunTest(async accelerator =>
    {
        int inC = 2, inH = 5, inW = 5, outC = 3, kH = 3, kW = 3, stride = 1, pad = 1;
        int outH = (inH + 2 * pad - kH) / stride + 1;
        int outW = (inW + 2 * pad - kW) / stride + 1;
        var rng = new Random(13);
        var input = new float[inC * inH * inW];
        var wf = new float[outC * inC * kH * kW];
        var bias = new float[outC];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 2 - 1);

        // bf16-rounded weights (what the GPU reads) and the fp32 reference using THOSE weights.
        var wBf16 = new global::ILGPU.BFloat16[wf.Length];
        var wRounded = new float[wf.Length];
        for (int i = 0; i < wf.Length; i++) { wBf16[i] = (global::ILGPU.BFloat16)wf[i]; wRounded[i] = (float)wBf16[i]; }

        var cpuOut = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = bias[oc];
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= inH || ix < 0 || ix >= inW) continue;
                                sum += (double)input[ic * inH * inW + iy * inW + ix]
                                     * (double)wRounded[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                            }
                    cpuOut[oc * outH * outW + oy * outW + ox] = (float)sum;
                }

        using var inBuf = accelerator.Allocate1D(input);
        using var wBuf = accelerator.Allocate1D(wBf16);
        using var biasBuf = accelerator.Allocate1D(bias);
        using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);
        var conv = new Conv2DKernel(accelerator);
        conv.ForwardPaddedLowPWeight(inBuf.View, wBuf.View, biasBuf.View, outBuf.View,
            inC, inH, inW, outC, kH, kW, stride, pad, pad, pad, pad);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);

        float maxErr = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - cpuOut[i]));
        if (maxErr > 1e-3f)
            throw new Exception($"Conv2D ForwardPaddedLowPWeight<BFloat16> maxErr={maxErr:E3} vs bf16-weight fp32 reference (expected < 1e-3)");
    });

    /// <summary>
    /// Slice 5: the Conv OPERATOR routes a half-backed weight (NCHW group-1) to the half kernel. Builds a
    /// half Conv weight via Tensor.FromHalf and runs ConvOperator.Execute end to end, matching a fp32 conv
    /// reference using the same fp16-rounded weights (isolates the operator routing + the half kernel).
    /// </summary>
    [TestMethod]
    public Task F16_ConvOperator_RoutesHalfWeight() => RunTest(async accelerator =>
    {
        int inC = 2, inH = 5, inW = 5, outC = 3, kH = 3, kW = 3, stride = 1, pad = 1;
        int outH = (inH + 2 * pad - kH) / stride + 1;
        int outW = (inW + 2 * pad - kW) / stride + 1;
        var rng = new Random(17);
        var input = new float[inC * inH * inW];
        var wf = new float[outC * inC * kH * kW];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)(rng.NextDouble() * 2 - 1);

        var wBytes = new byte[wf.Length * 2];
        var wRounded = new float[wf.Length];
        for (int i = 0; i < wf.Length; i++)
        {
            var hb = (System.Half)wf[i];
            var bb = BitConverter.GetBytes(hb);
            wBytes[i * 2] = bb[0]; wBytes[i * 2 + 1] = bb[1];
            wRounded[i] = (float)hb;
        }

        // CPU reference conv (NCHW, zero bias since no bias input).
        var cpuOut = new float[outC * outH * outW];
        for (int oc = 0; oc < outC; oc++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++)
                {
                    double sum = 0;
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= inH || ix < 0 || ix >= inW) continue;
                                sum += (double)input[ic * inH * inW + iy * inW + ix]
                                     * (double)wRounded[oc * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                            }
                    cpuOut[oc * outH * outW + oy * outW + ox] = (float)sum;
                }

        var pool = new BufferPool(accelerator);
        try
        {
            using var ms = new System.IO.MemoryStream(wBytes);
            var halfW = await pool.AllocateHalfWeightFromStreamAsync(ms, 0, wBytes.Length, 10, new[] { outC, inC, kH, kW });
            var wT = Tensor.FromHalf(halfW);
            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(outC * outH * outW);
            var xT = new Tensor(inBuf.View, new[] { 1, inC, inH, inW }, "x");
            var outT = new Tensor(outBuf.View, new[] { 1, outC, outH, outW }, "out");

            var registry = new OperatorRegistry(accelerator);
            var op = new ConvOperator(registry);
            var ctx = new OnnxOpContext
            {
                Inputs = new[] { xT, wT },
                Outputs = new[] { outT },
                Attributes = new Dictionary<string, object>
                {
                    ["strides"] = new long[] { stride, stride },
                    ["pads"] = new long[] { pad, pad, pad, pad },
                    ["group"] = (long)1,
                    ["dilations"] = new long[] { 1, 1 },
                },
                Pool = pool,
                InputNames = new[] { "x", "w" },
                Registry = registry,
            };
            op.Execute(ctx);
            await accelerator.SynchronizeAsync();
            var gpuOut = await outBuf.CopyToHostAsync<float>(0, outC * outH * outW);

            float maxErr = 0f;
            for (int i = 0; i < cpuOut.Length; i++)
                maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - cpuOut[i]));
            if (maxErr > 1e-3f)
                throw new Exception($"Conv operator half-weight routing maxErr={maxErr:E3} (expected < 1e-3)");
        }
        finally { pool.Dispose(); }
    });

    /// <summary>
    /// Slice 6: batched MatMul with fp16 weights (SD-Turbo attention projections — 2D weight, rank-3
    /// activation). Matches a fp32 per-batch reference using the same fp16-rounded weights.
    /// </summary>
    [TestMethod]
    public Task F16_BatchedMatMulHalfWeight_MatchesFp32Reference() => RunTest(async accelerator =>
    {
        int batch = 2, M = 3, K = 8, N = 4;
        var rng = new Random(23);
        var a = new float[batch * M * K];
        var bf = new float[batch * K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bf.Length; i++) bf[i] = (float)(rng.NextDouble() * 2 - 1);
        var bHalf = new global::ILGPU.Half[bf.Length];
        var bRounded = new float[bf.Length];
        for (int i = 0; i < bf.Length; i++) { bHalf[i] = (global::ILGPU.Half)bf[i]; bRounded[i] = (float)bHalf[i]; }

        var cpuC = new float[batch * M * N];
        for (int bt = 0; bt < batch; bt++)
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                {
                    float s = 0f;
                    for (int k = 0; k < K; k++)
                        s += a[bt * M * K + r * K + k] * bRounded[bt * K * N + k * N + c];
                    cpuC[bt * M * N + r * N + c] = s;
                }

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(bHalf);
        using var cBuf = accelerator.Allocate1D<float>(batch * M * N);
        var mm = new MatMulKernel(accelerator);
        mm.BatchedMatMulHalfWeight(aBuf.View, bBuf.View, cBuf.View, batch, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuC = await cBuf.CopyToHostAsync<float>(0, batch * M * N);

        float maxErr = 0f;
        for (int i = 0; i < cpuC.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuC[i] - cpuC[i]));
        if (maxErr > 1e-3f)
            throw new Exception($"BatchedMatMulHalfWeight maxErr={maxErr:E3} (expected < 1e-3)");
    });

    // NOTE (2026-06-05): a GENERIC-WEIGHT kernel (generic over TW:INumber<TW>, read TW->float via
    // float.CreateTruncating, fp32 accumulate) was spiked here and FAILED on all 6 backends —
    // NotSupportedException "Class type 'System.Type' is not supported" (float.CreateTruncating inspects
    // typeof(TOther) internally; ILGPU can't transpile it). Geordi's ILGPU.Half:INumber enables generic
    // PURE-T arithmetic (all operands T), but NOT the mixed-precision fp16-weight->fp32-accumulate convert.
    // So the DEDICATED half kernels (ILGPU.Half's (float) operator [transpilable] + fp32 accumulate) are the
    // ORT-parity approach. Spike removed; do NOT reintroduce float.CreateTruncating in a kernel.
}
