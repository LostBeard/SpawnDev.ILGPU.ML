using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// PROOF for approach-(i) precision-aware ops (the Rule-1/Rule-4 design: ops read+write fp16/bf16 DIRECTLY,
/// fp32 compute, NO fp32 temp buffers, NO convert-around-node). Such an op (Conv/GroupNorm/MatMul/…) must
/// read its low-p input, accumulate in FLOAT (precision over many terms), and write low-p — i.e. it needs
/// <c>float ↔ T</c> conversion INSIDE a generic <c>where T : INumber&lt;T&gt;</c> kernel
/// (<c>float.CreateChecked(T)</c> / <c>T.CreateChecked(float)</c>). Geordi's GenericPrecision proved generic
/// arithmetic/compare/scalar-params, but NOT in-kernel float↔T conversion — that's what this checks. If it
/// transpiles+runs on all backends, every approach-(i) op is ONE generic kernel for float/Half/bf16. If a
/// backend chokes, this is a clean repro for Geordi (ILGPU codegen) and we use per-type kernels meanwhile.
/// </summary>
public abstract partial class MLTestBase
{
    // Per-row mean: read T, accumulate FLOAT, write T. The reduction shape Conv/GroupNorm/MatMul share.
    private static void MixedMeanGeneric<T>(Index1D row,
        ArrayView1D<T, Stride1D.Dense> input, ArrayView1D<T, Stride1D.Dense> output, int C)
        where T : unmanaged, INumber<T>
    {
        int b = row * C;
        float acc = 0f;
        // ILGPU.PrecisionConvert (local.9): transpilable generic float<->T conversion ([ConvertIntrinsic],
        // lowers to the same IR as the concrete (float)Half / (Half)float cast — no System.Type). This is
        // what makes approach-(i) precision-aware ops a SINGLE generic kernel for float/Half/bf16/fp8.
        for (int c = 0; c < C; c++) acc += global::ILGPU.PrecisionConvert.ConvertToSingle(input[b + c]); // T -> float
        output[row] = global::ILGPU.PrecisionConvert.ConvertFromSingle<T>(acc / C);                      // float -> T
    }

    private static float[] MeanCpu(float[] x, int rows, int C)
    {
        var o = new float[rows];
        for (int r = 0; r < rows; r++) { float a = 0; for (int c = 0; c < C; c++) a += x[r * C + c]; o[r] = a / C; }
        return o;
    }

    [TestMethod]
    public async Task GenericMixedCompute_Half_FloatAccumulate_AllBackends() => await RunTest(async accelerator =>
    {
        const int rows = 33, C = 80; // C large enough that Half accumulation would drift → proves FLOAT accumulate
        var rng = new Random(19);
        var x = new float[rows * C];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var expected = MeanCpu(x, rows, C);

        var xh = new global::ILGPU.Half[x.Length];
        for (int i = 0; i < x.Length; i++) xh[i] = (global::ILGPU.Half)x[i];
        using var inBuf = accelerator.Allocate1D(xh);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.Half>(rows);
        var k = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, int>(
            MixedMeanGeneric<global::ILGPU.Half>);
        k(rows, inBuf.View, outBuf.View, C);
        await accelerator.SynchronizeAsync();
        var gotH = await outBuf.CopyToHostAsync<global::ILGPU.Half>(0, rows);

        for (int r = 0; r < rows; r++)
        {
            float got = (float)gotH[r];
            if (MathF.Abs(got - expected[r]) > MathF.Max(5e-3f, MathF.Abs(expected[r]) * 5e-3f))
                throw new Exception($"Half generic mixed-compute @row{r}: got {got}, want {expected[r]} — float↔T in-kernel conversion broken on {BackendName}");
        }
        Console.WriteLine($"[GenericMixedCompute] Half read+float-accumulate+write transpiles + matches CPU ({BackendName})");
    });

    [TestMethod]
    public async Task GenericMixedCompute_BFloat16_FloatAccumulate_AllBackends() => await RunTest(async accelerator =>
    {
        const int rows = 33, C = 80;
        var rng = new Random(23);
        var x = new float[rows * C];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var expected = MeanCpu(x, rows, C);

        var xb = new global::ILGPU.BFloat16[x.Length];
        for (int i = 0; i < x.Length; i++) xb[i] = (global::ILGPU.BFloat16)x[i];
        using var inBuf = accelerator.Allocate1D(xb);
        using var outBuf = accelerator.Allocate1D<global::ILGPU.BFloat16>(rows);
        var k = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>, ArrayView1D<global::ILGPU.BFloat16, Stride1D.Dense>, int>(
            MixedMeanGeneric<global::ILGPU.BFloat16>);
        k(rows, inBuf.View, outBuf.View, C);
        await accelerator.SynchronizeAsync();
        var gotB = await outBuf.CopyToHostAsync<global::ILGPU.BFloat16>(0, rows);

        for (int r = 0; r < rows; r++)
        {
            float got = (float)gotB[r];
            if (MathF.Abs(got - expected[r]) > MathF.Max(2e-2f, MathF.Abs(expected[r]) * 2e-2f))
                throw new Exception($"BFloat16 generic mixed-compute @row{r}: got {got}, want {expected[r]} — float↔T in-kernel conversion broken on {BackendName}");
        }
        Console.WriteLine($"[GenericMixedCompute] BFloat16 read+float-accumulate+write transpiles + matches CPU ({BackendName})");
    });
}
