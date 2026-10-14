using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// FusedAttention is right for EVERY head dimension real models use, not only 64/128. The warp-cooperative register path
    /// splits a query across headDim/16 lanes and reduces with an xor butterfly, which only works when that is a power of two:
    /// ReDimNet2's head dimension 48 (3 lanes) produced a 67% wrong attention output with no error (measured against
    /// onnxruntime), and 80 / 96 / 112 would have done the same. The existing tests used D=4 and 8, which never reach that
    /// path at all. Realistic sizes: 4 heads x 149 positions (a 3 s speech clip through a conformer-style block), non-causal.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task FusedAttention_EveryHeadDim_MatchesCPU() => await RunTest(async accelerator =>
    {
        const int BH = 4, S = 149;
        var failures = new List<string>();
        foreach (var D in new[] { 16, 32, 48, 64, 80, 96, 112, 128 })
        {
            var rng = new Random(1000 + D);
            var Q = new float[BH * S * D]; var K = new float[BH * S * D]; var V = new float[BH * S * D];
            for (int i = 0; i < Q.Length; i++) { Q[i] = (float)(rng.NextDouble() * 2 - 1); K[i] = (float)(rng.NextDouble() * 2 - 1); V[i] = (float)(rng.NextDouble() * 2 - 1); }
            float scale = 1f / MathF.Sqrt(D);

            var expected = new float[BH * S * D];
            for (int bh = 0; bh < BH; bh++)
                for (int sq = 0; sq < S; sq++)
                {
                    var scores = new double[S];
                    double maxS = double.MinValue;
                    for (int kv = 0; kv < S; kv++)
                    {
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += Q[(bh * S + sq) * D + d] * K[(bh * S + kv) * D + d];
                        scores[kv] = dot * scale; maxS = Math.Max(maxS, scores[kv]);
                    }
                    double sum = 0;
                    for (int kv = 0; kv < S; kv++) { scores[kv] = Math.Exp(scores[kv] - maxS); sum += scores[kv]; }
                    for (int d = 0; d < D; d++)
                    {
                        double val = 0;
                        for (int kv = 0; kv < S; kv++) val += scores[kv] / sum * V[(bh * S + kv) * D + d];
                        expected[(bh * S + sq) * D + d] = (float)val;
                    }
                }

            using var qBuf = accelerator.Allocate1D(Q); using var kBuf = accelerator.Allocate1D(K); using var vBuf = accelerator.Allocate1D(V);
            using var outBuf = accelerator.Allocate1D<float>(BH * S * D);
            var fused = new FusedAttentionKernel(accelerator);
            // Non-causal, explicit scale: what FuseAttention emits (causal=0, window=0).
            fused.Forward(qBuf.View, kBuf.View, vBuf.View, outBuf.View, BH, BH, S, S, D, causal: false, window: int.MaxValue, kvOffset: 0, scale: scale);
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, BH * S * D);

            float maxErr = 0;
            for (int i = 0; i < expected.Length; i++) maxErr = MathF.Max(maxErr, MathF.Abs(got[i] - expected[i]));
            Console.WriteLine($"[FusedAttention] headDim {D}: max |err| {maxErr:E2} on {BackendName}");
            if (!(maxErr < 1e-3f)) failures.Add($"headDim {D}: max |err| {maxErr:E2}");
        }
        if (failures.Count > 0)
            throw new Exception($"FusedAttention is wrong on {BackendName} for: {string.Join("; ", failures)}");
    });
}
