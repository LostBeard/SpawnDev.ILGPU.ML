using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Foundational test for mixed-precision activations: the fp32↔fp16/bf16 GPU convert kernels
/// (<see cref="PrecisionConvertKernels"/>) — the boundary primitive every low-precision-activation op needs —
/// must round-trip correctly on every backend. Uses the HAND-WRITTEN per-type cast kernels (the generic
/// INumber path has cross-backend codegen gaps, tracked to Geordi). A round-trip
/// fp32 → low-p store → fp32 must match the input within that format's representable precision, which ALSO
/// confirms Half/bf16 activation STORAGE + readback works on each backend (isolating it from the generic
/// path's Wasm value-corruption failure).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task PrecisionConvert_FloatHalfRoundTrip_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 1024;
        var rng = new Random(31);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 8 - 4); // O(1-4), fp16-representable

        var conv = new PrecisionConvertKernels(accelerator);
        using var fIn = accelerator.Allocate1D(x);
        using var hMid = accelerator.Allocate1D<global::ILGPU.Half>(n);
        using var fOut = accelerator.Allocate1D<float>(n);
        conv.FloatToHalf(fIn.View, hMid.View, n);
        conv.HalfToFloat(hMid.View, fOut.View, n);
        await accelerator.SynchronizeAsync();
        var got = await fOut.CopyToHostAsync<float>(0, n);

        // Reference: the EXACT value fp16 can store = (float)(Half)x. Round-trip must equal it bit-for-bit
        // (both directions are pure casts), so tolerance is just fp16 quantization of the input.
        for (int i = 0; i < n; i++)
        {
            float want = (float)(global::ILGPU.Half)x[i];
            if (MathF.Abs(got[i] - want) > MathF.Max(1e-3f, MathF.Abs(want) * 1e-3f))
                throw new Exception($"fp16 round-trip @{i}: got {got[i]}, want {want} (input {x[i]})");
        }
        Console.WriteLine($"[PrecisionConvert] fp32<->fp16 round-trip OK on {BackendName}");
    });

    [TestMethod]
    public async Task PrecisionConvert_FloatBFloat16RoundTrip_AllBackends() => await RunTest(async accelerator =>
    {
        const int n = 1024;
        var rng = new Random(53);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 8 - 4);

        var conv = new PrecisionConvertKernels(accelerator);
        using var fIn = accelerator.Allocate1D(x);
        using var bMid = accelerator.Allocate1D<global::ILGPU.BFloat16>(n);
        using var fOut = accelerator.Allocate1D<float>(n);
        conv.FloatToBFloat16(fIn.View, bMid.View, n);
        conv.BFloat16ToFloat(bMid.View, fOut.View, n);
        await accelerator.SynchronizeAsync();
        var got = await fOut.CopyToHostAsync<float>(0, n);

        for (int i = 0; i < n; i++)
        {
            float want = (float)(global::ILGPU.BFloat16)x[i];
            if (MathF.Abs(got[i] - want) > MathF.Max(1e-2f, MathF.Abs(want) * 1e-2f))
                throw new Exception($"bf16 round-trip @{i}: got {got[i]}, want {want} (input {x[i]})");
        }
        Console.WriteLine($"[PrecisionConvert] fp32<->bf16 round-trip OK on {BackendName}");
    });
}
