using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// FusedLinearKernel split-K (few output tiles, long K: the K reduction is split across groups and the partials summed in a fixed order)
    /// against a double-precision CPU reference, on the shapes that engage it - CLAP HTSAT's layers.2/3 Linears (M=64/256 tokens, K up to 3072)
    /// plus ragged edges (M, N, K not multiples of the 64x64 tile or the 16-wide K tile) - and on shapes that must NOT engage it. Each case
    /// also runs with split-K off: both paths must match the reference, and SplitKCalls must count exactly the engaged cases.
    /// Added 2026-11-04.
    /// </summary>
    [TestMethod]
    public async Task FusedLinear_SplitK_MatchesCpu() => await RunTest(async accelerator =>
    {
        if (accelerator.MaxNumThreadsPerGroup < 256)
            throw new UnsupportedTestException("the register-blocked path needs 256-thread groups (split-K lives on it)");
        var cases = new (int M, int K, int N, FusedActivation Act, bool Engage)[]
        {
            (64, 3072, 768, FusedActivation.None, true),     // CLAP layers.3 output dense
            (64, 768, 3072, FusedActivation.GELU, true),     // CLAP layers.3 intermediate (48 tiles)
            (256, 1536, 384, FusedActivation.None, true),    // CLAP layers.2 output dense (24 tiles)
            (70, 1000, 200, FusedActivation.ReLU, true),     // ragged M, N, K
            (65, 520, 65, FusedActivation.SiLU, true),       // one row / column past a tile, K just past the minimum
            (64, 256, 768, FusedActivation.None, false),     // K below the minimum
            (512, 1024, 512, FusedActivation.None, false),   // 64 tiles: enough to fill the GPU unsplit
        };
        bool saved = FusedLinearKernel.SplitK;
        var notes = new List<string>();
        try
        {
            using var fused = new FusedLinearKernel(accelerator);
            foreach (var (M, K, N, act, engage) in cases)
            {
                var x = RandomFloats(M * K, seed: 900 + M);
                var w = RandomFloats(K * N, seed: 901 + N, scale: 0.05f);
                var b = RandomFloats(N, seed: 902, scale: 0.2f);
                var want = new float[M * N];
                for (int r = 0; r < M; r++)
                    for (int c = 0; c < N; c++)
                    {
                        double sum = 0;
                        for (int k = 0; k < K; k++) sum += (double)x[r * K + k] * w[k * N + c];
                        want[r * N + c] = Act((float)sum + b[c], act);
                    }
                using var xb = accelerator.Allocate1D(x);
                using var wb = accelerator.Allocate1D(w);
                using var bb = accelerator.Allocate1D(b);
                using var yb = accelerator.Allocate1D<float>(M * N);
                foreach (bool split in new[] { true, false })
                {
                    FusedLinearKernel.SplitK = split;
                    long calls0 = fused.SplitKCalls;
                    fused.Forward(xb.View, wb.View, bb.View, yb.View, M, K, N, act);
                    await accelerator.SynchronizeAsync();
                    bool engaged = fused.SplitKCalls > calls0;
                    if (engaged != (split && engage))
                        throw new Exception($"M={M} K={K} N={N} split={split}: split-K {(engaged ? "engaged" : "did not engage")} (S={FusedLinearKernel.SplitKCount(M, K, N)})");
                    var got = await yb.View.CopyToHostAsync();
                    double worst = 0;
                    for (int i = 0; i < got.Length; i++)
                        worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / (1.0 + Math.Abs(want[i])));
                    if (worst > 1e-4)
                        throw new Exception($"M={M} K={K} N={N} {act} split={split}: max relative error {worst:E2}");
                    if (split) notes.Add($"{M}x{K}x{N} {act} S={FusedLinearKernel.SplitKCount(M, K, N)} err {worst:E1}");
                }
            }
        }
        finally { FusedLinearKernel.SplitK = saved; }
        var line = $"[SplitK] {accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine(line);
        return line;

        static float Act(float v, FusedActivation a) => a switch
        {
            FusedActivation.ReLU => v > 0 ? v : 0,
            FusedActivation.SiLU => v / (1f + MathF.Exp(-v)),
            FusedActivation.GELU => v > 10f ? v : v < -10f ? 0f : 0.5f * v * (1f + ErfApprox(v * 0.7071067811865475f)),
            _ => v,
        };
    });
}
