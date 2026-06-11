using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// RoPEKernel correctness - both pairing styles (NeoX split-half, GPT-J interleaved),
/// per-call base (gemma4 runs 10000 on local layers and 1000000 on global layers),
/// partial rotary, and decode-time startPosition. Oracle = independent CPU rotation.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task RopeKernel_NeoX_DefaultApi_MatchesCPU() => await RunTest(async accelerator =>
        await RopeOracle(accelerator, numPos: 16, headDim: 64, startPos: 0,
            ropeBase: 10000f, rotaryDim: 64, interleaved: false, useLegacyApi: true));

    [TestMethod]
    public async Task RopeKernel_DualBase_Gemma4Bases_MatchCPU() => await RunTest(async accelerator =>
    {
        // The two gemma4 per-layer bases through the SAME kernel instance.
        await RopeOracle(accelerator, 16, 64, 5, 10000f, 64, interleaved: false, useLegacyApi: false);
        await RopeOracle(accelerator, 16, 64, 5, 1000000f, 64, interleaved: false, useLegacyApi: false);
    });

    [TestMethod]
    public async Task RopeKernel_Interleaved_MatchesCPU() => await RunTest(async accelerator =>
        await RopeOracle(accelerator, 12, 32, 3, 10000f, 32, interleaved: true, useLegacyApi: false));

    [TestMethod]
    public async Task RopeKernel_PartialRotary_TailPassThrough() => await RunTest(async accelerator =>
        await RopeOracle(accelerator, 8, 64, 7, 10000f, rotaryDim: 32, interleaved: false, useLegacyApi: false));

    private static async Task RopeOracle(Accelerator accelerator,
        int numPos, int headDim, int startPos, float ropeBase, int rotaryDim,
        bool interleaved, bool useLegacyApi)
    {
        var rng = new Random(53 + headDim + (interleaved ? 1 : 0) + (int)(ropeBase / 10000));
        var input = new float[numPos * headDim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU oracle: rotate pairs per style; pass through dims >= rotaryDim.
        var expected = (float[])input.Clone();
        int half = rotaryDim / 2;
        for (int p = 0; p < numPos; p++)
        {
            long pos = startPos + p;
            for (int i = 0; i < half; i++)
            {
                double theta = pos / Math.Pow(ropeBase, 2.0 * i / rotaryDim);
                double c = Math.Cos(theta), s = Math.Sin(theta);
                int i0 = interleaved ? 2 * i : i;
                int i1 = interleaved ? 2 * i + 1 : i + half;
                double x0 = input[p * headDim + i0];
                double x1 = input[p * headDim + i1];
                expected[p * headDim + i0] = (float)(x0 * c - x1 * s);
                expected[p * headDim + i1] = (float)(x0 * s + x1 * c);
            }
        }

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(input.Length);
        var rope = new RoPEKernel(accelerator);
        if (useLegacyApi)
            rope.Apply(inBuf.View, outBuf.View, numPos, headDim, startPos);
        else
            rope.Apply(inBuf.View, outBuf.View, numPos, headDim, startPos, ropeBase, rotaryDim, interleaved);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, input.Length);

        AssertCloseQuant(got, expected, 2e-3f,
            $"RoPE base={ropeBase} rotDim={rotaryDim} interleaved={interleaved}");

        // Partial rotary: the tail must be EXACTLY the input (pure pass-through).
        for (int p = 0; p < numPos; p++)
            for (int k = rotaryDim; k < headDim; k++)
                if (got[p * headDim + k] != input[p * headDim + k])
                    throw new Exception($"RoPE partial pass-through violated at pos {p} dim {k}");

        Console.WriteLine($"[RoPE] base={ropeBase} rotDim={rotaryDim}/{headDim} " +
            $"interleaved={interleaved} startPos={startPos}: matches CPU oracle");
    }
}
