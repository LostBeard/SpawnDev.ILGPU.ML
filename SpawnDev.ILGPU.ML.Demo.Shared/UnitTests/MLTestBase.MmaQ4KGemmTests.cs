using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The tensor-core Q4_K GEMM (<see cref="MmaQ4KGemm"/>, a prefill's M &gt; 1 rows on CUDA sm_80+) against the CPU dequant oracle: shapes whose M and N do not fill
    /// the 128 x 128 tiles (M = 200 / N = 300, M = 16 / N = 130, M = 1000 / N = 257) over K = 4096, 1024, 2048 (several super-blocks: every sub-block scale
    /// form j &lt; 4 and j &gt;= 4). The operands are rounded to f16, so each output's error is bounded by 2e-3 x sum |a_k w_k| (f16 keeps 11 significant bits;
    /// the bound is ~4x the two roundings). It must be the kernel that ran; a 16-byte-misaligned input must take the SIMT kernels and still match. Added 2026-11-02.
    /// Both variants: the sm_80+ kernel (m16n8k16) where the card has it, and the Turing kernel (sm_75: two m16n8k8 per 16 k, scalar f16 packs) on any sm_75+
    /// card (<see cref="FusedDequantMatMul.ForceSm75Mma"/> runs it on a newer one).
    /// </summary>
    [TestMethod]
    public async Task Gemm_MmaQ4K_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("the tensor-core GEMM runs on CUDA sm_75+");
        var variants = CudaCaps.TensorCores(accelerator) ? new[] { (false, "mma-q4k"), (true, "mma-q4k-sm75") } : new[] { (true, "mma-q4k-sm75") };
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma;
        var type = GGMLType.Q4_K;
        using var fused = new FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        try
        {
        foreach (var (force, want) in variants)
        foreach (var (M, K, N) in new[] { (200, 4096, 300), (16, 1024, 130), (1000, 2048, 257) })
        {
            FusedDequantMatMul.ForceSm75Mma = force;
            var rng = new Random(13 + M + K + N);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[M * K + 4];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            (float[] Want, float[] Bound) Oracle(int off)
            {
                var want = new float[M * N]; var bound = new float[M * N];
                Parallel.For(0, M, m =>
                {
                    for (int n = 0; n < N; n++)
                    {
                        double s = 0, a = 0;
                        for (int k = 0; k < K; k++) { double p = input[off + m * K + k] * wRows[n][k]; s += p; a += Math.Abs(p); }
                        want[m * N + n] = (float)s; bound[m * N + n] = (float)(2e-3 * a + 1e-5);
                    }
                });
                return (want, bound);
            }
            static int Bad(float[] got, (float[] Want, float[] Bound) o, out float worst)
            {
                int bad = 0; worst = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    float e = Math.Abs(got[i] - o.Want[i]);
                    if (!(e <= o.Bound[i])) bad++;
                    worst = Math.Max(worst, e / o.Bound[i]);
                }
                return bad;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>((long)M * N);
            // aligned: the tensor-core kernel
            fused.Forward(inputBuf.View.SubView(0, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
            string route = fused.LastGemmKernel;
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
            if (route != want) throw new Exception($"M={M} K={K} N={N}: an aligned Q4_K GEMM ran on '{route}', not {want}");
            int bad = Bad(got, Oracle(0), out float worst);
            if (bad > 0) throw new Exception($"M={M} K={K} N={N} {want}: {bad} of {got.Length} outputs past 2e-3 x sum|a w| (worst {worst:F2}x the bound)");
            // misaligned by one float: the SIMT kernels, the same answer
            fused.Forward(inputBuf.View.SubView(1, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
            string route2 = fused.LastGemmKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
            if (route2.StartsWith("mma")) throw new Exception($"M={M} K={K} N={N}: a misaligned input took the tensor-core kernel (its loads assume 16-byte alignment)");
            int bad2 = Bad(got2, Oracle(1), out float worst2);
            if (bad2 > 0) throw new Exception($"M={M} K={K} N={N}: SIMT Q4_K GEMM (misaligned input), {bad2} of {got2.Length} outputs past the bound (worst {worst2:F2}x)");
            notes.Add($"M={M} K={K} N={N}: {route} worst {worst:F3}x the bound, {route2} {worst2:F3}x");
        }
        }
        finally { FusedDequantMatMul.ForceSm75Mma = forceSaved; }
        return string.Join("; ", notes);
    });
}
