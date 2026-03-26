using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for fused attention kernel: correctness vs CPU reference.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task FusedAttention_MatchesCPU() => await RunTest(async accelerator =>
    {
        int BH = 1, SQ = 2, SKV = 3, D = 4;
        var rng = new Random(42);

        var Q = new float[BH * SQ * D];
        var K = new float[BH * SKV * D];
        var V = new float[BH * SKV * D];
        for (int i = 0; i < Q.Length; i++) Q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < K.Length; i++) K[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < V.Length; i++) V[i] = (float)(rng.NextDouble() * 2 - 1);

        float scale = 1f / MathF.Sqrt(D);

        // CPU reference
        var expected = new float[BH * SQ * D];
        for (int bh = 0; bh < BH; bh++)
            for (int sq = 0; sq < SQ; sq++)
            {
                // Compute scores
                var scores = new float[SKV];
                float maxS = float.MinValue;
                for (int kv = 0; kv < SKV; kv++)
                {
                    float dot = 0;
                    for (int d = 0; d < D; d++)
                        dot += Q[(bh * SQ + sq) * D + d] * K[(bh * SKV + kv) * D + d];
                    scores[kv] = dot * scale;
                    if (scores[kv] > maxS) maxS = scores[kv];
                }
                // Softmax
                float sumE = 0;
                for (int kv = 0; kv < SKV; kv++) { scores[kv] = MathF.Exp(scores[kv] - maxS); sumE += scores[kv]; }
                for (int kv = 0; kv < SKV; kv++) scores[kv] /= sumE;
                // Weighted V
                for (int d = 0; d < D; d++)
                {
                    float val = 0;
                    for (int kv = 0; kv < SKV; kv++)
                        val += scores[kv] * V[(bh * SKV + kv) * D + d];
                    expected[(bh * SQ + sq) * D + d] = val;
                }
            }

        // GPU fused attention
        using var qBuf = accelerator.Allocate1D(Q);
        using var kBuf = accelerator.Allocate1D(K);
        using var vBuf = accelerator.Allocate1D(V);
        using var outBuf = accelerator.Allocate1D<float>(BH * SQ * D);

        var fused = new FusedAttentionKernel(accelerator);
        fused.Forward(qBuf.View, kBuf.View, vBuf.View, outBuf.View, BH, SQ, SKV, D);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outBuf.CopyToHostAsync<float>(0, BH * SQ * D);

        float maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - expected[i]));

        Console.WriteLine($"[FusedAttention] maxErr={maxErr:E3} vs CPU reference");
        if (maxErr > 0.01f)
            throw new Exception($"FusedAttention maxErr={maxErr:E3} exceeds tolerance 0.01");
    });
}
