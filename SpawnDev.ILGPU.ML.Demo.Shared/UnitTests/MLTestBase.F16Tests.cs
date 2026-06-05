using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// f16-native weights/compute tests (task: f16). The spike (2026-06-05) settled the kernel architecture
/// across all 6 backends: a GENERIC-MATH kernel (System.Half + INumber&lt;T&gt;) FAILS everywhere
/// ("Not supported intrinsic type 'BitCast'"), but native <c>ILGPU.Half</c> storage + fp32 compute WORKS
/// everywhere (incl. WebGPU/WGSL). So f16 = store weights as ILGPU.Half (half the bytes), read + upconvert
/// to float, accumulate fp32 (ORT-style mixed precision; no accuracy loss). Dedicated half-input kernels,
/// no generics.
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
}
