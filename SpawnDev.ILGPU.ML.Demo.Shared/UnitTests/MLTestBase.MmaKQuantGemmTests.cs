using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The tensor-core K-quant GEMMs (<see cref="MmaKQuantGemm"/>: a prefill's M &gt; 1 rows of Q4_K and Q6_K on CUDA sm_75+) against the CPU dequant oracle: shapes
    /// whose M and N do not fill the 128 x 128 tiles (M = 200 / N = 300, M = 16 / N = 130, M = 1000 / N = 257) over K = 4096, 1024, 2048 (several super-blocks:
    /// every Q4_K sub-block scale form j &lt; 4 and j &gt;= 4, every Q6_K half and quarter). The operands are rounded to f16, so each output's error is bounded by
    /// 2e-3 x sum |a_k w_k| (f16 keeps 11 significant bits; the bound is ~4x the two roundings). It must be the kernel that ran; a 16-byte-misaligned input must
    /// take the SIMT kernels and still match. Both variants: the sm_80+ kernels (m16n8k16) where the card has them, and the Turing kernels (sm_75: two m16n8k8 per
    /// 16 k, scalar f16 packs) on any sm_75+ card (<see cref="FusedDequantMatMul.ForceSm75Mma"/> runs them on a newer one). Added 2026-11-02 (Q4_K), Q6_K the
    /// same day.
    /// </summary>
    [TestMethod]
    public async Task Gemm_MmaKQuant_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (!CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("the tensor-core GEMMs run on CUDA sm_75+");
        var forces = CudaCaps.TensorCores(accelerator) ? new[] { false, true } : new[] { true };
        bool forceSaved = FusedDequantMatMul.ForceSm75Mma, v2Saved = FusedDequantMatMul.EnableMmaV2, v3Saved = FusedDequantMatMul.EnableMmaV3; bool v5Saved = FusedDequantMatMul.EnableMmaV5;
        bool scrSaved = FusedDequantMatMul.EnableMmaScratch;
        int v3MinSaved = FusedDequantMatMul.MmaV3MinRows, scrMinSaved = FusedDequantMatMul.MmaScratchMinRows;
        FusedDequantMatMul.MmaV3MinRows = 1;   // every shape on V3 too
        FusedDequantMatMul.MmaScratchMinRows = 1;   // and on the scratch-operand GEMM
        using var fused = new FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        try
        {
        foreach (var type in new[] { GGMLType.Q4_K, GGMLType.Q6_K })
        foreach (bool force in forces)
        foreach (var (M, K, N) in new[] { (200, 4096, 300), (16, 1024, 130), (1000, 2048, 257) })
        {
            FusedDequantMatMul.ForceSm75Mma = force;
            string want = (type == GGMLType.Q4_K ? "mma-q4k" : "mma-q6k") + (force ? "-sm75" : "");
            var rng = new Random(13 + M + K + N + (int)type);
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
                var o = new float[M * N]; var bound = new float[M * N];
                Parallel.For(0, M, m =>
                {
                    for (int n = 0; n < N; n++)
                    {
                        double s = 0, a = 0;
                        for (int k = 0; k < K; k++) { double p = input[off + m * K + k] * wRows[n][k]; s += p; a += Math.Abs(p); }
                        o[m * N + n] = (float)s; bound[m * N + n] = (float)(2e-3 * a + 1e-5);
                    }
                });
                return (o, bound);
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
            // aligned: the tensor-core kernel, V1, V2 (double-buffered, 64-bit fragment loads) and V3 (256-row tiles): the same rounding and MMA
            // order per output, so V2 and V3 must be bit-identical to V1
            float[] got = Array.Empty<float>(); string route = "";
            foreach (var (v2, v3, scr, v5, sfx0) in new[] { (false, false, false, false, ""), (true, false, false, false, "-v2"), (true, true, false, false, "-v3"),
                (true, true, true, false, "-scratch"), (true, true, true, true, "-scratch-v5") })
            {
                // V5 (64 x 64 warp tiles) is sm_80+: the Turing scratch path keeps its V2 GEMM
                string sfx = v5 && want.Contains("sm75") ? "-scratch" : sfx0;
                FusedDequantMatMul.EnableMmaV2 = v2;
                FusedDequantMatMul.EnableMmaV3 = v3;
                FusedDequantMatMul.EnableMmaScratch = scr;
                FusedDequantMatMul.EnableMmaV5 = v5;
                fused.Forward(inputBuf.View.SubView(0, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
                string r = fused.LastGemmKernel;
                await accelerator.SynchronizeAsync();
                var g = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
                if (r != want + sfx) throw new Exception($"{type} M={M} K={K} N={N}: an aligned GEMM ran on '{r}', not {want}{sfx}");
                if (sfx != "")
                {
                    int diff = 0; for (int i = 0; i < g.Length; i++) if (BitConverter.SingleToInt32Bits(g[i]) != BitConverter.SingleToInt32Bits(got[i])) diff++;
                    if (diff > 0) throw new Exception($"{type} M={M} K={K} N={N} {want}{sfx}: differs from V1 in {diff} of {g.Length} outputs");
                }
                else got = g;
                route = r;
            }
            int bad = Bad(got, Oracle(0), out float worst);
            if (bad > 0) throw new Exception($"{type} M={M} K={K} N={N} {want}: {bad} of {got.Length} outputs past 2e-3 x sum|a w| (worst {worst:F2}x the bound)");
            // misaligned by one float: the SIMT kernels, the same answer
            FusedDequantMatMul.EnableMmaScratch = false;
            fused.Forward(inputBuf.View.SubView(1, (long)M * K), weightBuf.View, outBuf.View, M, K, N, type);
            string route2 = fused.LastGemmKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
            if (route2.StartsWith("mma")) throw new Exception($"{type} M={M} K={K} N={N}: a misaligned input took the tensor-core kernel (its loads assume 16-byte alignment)");
            int bad2 = Bad(got2, Oracle(1), out float worst2);
            if (bad2 > 0) throw new Exception($"{type} M={M} K={K} N={N}: SIMT GEMM (misaligned input), {bad2} of {got2.Length} outputs past the bound (worst {worst2:F2}x)");
            notes.Add($"{type} M={M} K={K} N={N}: {route} worst {worst:F3}x the bound, {route2} {worst2:F3}x");
        }
        }
        finally
        {
            FusedDequantMatMul.ForceSm75Mma = forceSaved; FusedDequantMatMul.EnableMmaV2 = v2Saved; FusedDequantMatMul.EnableMmaV3 = v3Saved;
            FusedDequantMatMul.MmaV3MinRows = v3MinSaved; FusedDequantMatMul.EnableMmaScratch = scrSaved; FusedDequantMatMul.MmaScratchMinRows = scrMinSaved;
            FusedDequantMatMul.EnableMmaV5 = v5Saved;
        }
        return string.Join("; ", notes);
    });
}
