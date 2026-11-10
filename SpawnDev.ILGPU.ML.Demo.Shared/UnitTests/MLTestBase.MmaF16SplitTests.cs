using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <c>FusedDequantMatMul.EnableMmaF16</c> (2026-11-10): M &gt; 1 of F16 weights on the tensor-core GEMM with the activations split as A = hi + lo * 2^-11
    /// (both f16, two GEMMs, the products added in f32), against an f64 CPU reference, and the fp32 path (the tiled GEMM the route replaces) measured on the
    /// same data. Error relative to sum |x w| within 4e-6 (measured 0.7-1.3e-6, the fp32 path 0.3-0.4e-6; rounding A to f16 alone gives ~2e-4, so a
    /// missing or broken lo half fails). Shapes: qwen3 1.7B's attn_v (K 2048, N 1024) at 16, 64 and 806 rows, a ragged M and N, and K 4096. Inputs span
    /// |x| from 1e-3 to 1 (where an unscaled lo half would be subnormal).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task<string> Gemm_F16_MmaSplit_MatchesFp32() => await RunTest(async accelerator =>
    {
        if (!SpawnDev.ILGPU.ML.Kernels.CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("CUDA sm_75+ (tensor cores)");
        using var fused = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(accelerator);
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableMmaF16;
        int savedRows = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.MmaF16MinRows;
        var notes = new List<string>();
        try
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.MmaF16MinRows = 2;
            foreach (var (M, K, N) in new[] { (16, 2048, 1024), (64, 2048, 1024), (806, 2048, 1024), (77, 2048, 136), (130, 4096, 256) })
            {
                var rng = new Random(9 + M + K + N);
                var x = new float[M * K];
                for (int i = 0; i < x.Length; i++)
                {
                    double mag = Math.Pow(10, -3 + 3 * rng.NextDouble());                 // 1e-3 .. 1
                    x[i] = (float)(mag * (rng.Next(2) == 0 ? -1 : 1));
                }
                var wBytes = new byte[N * K * 2];
                var w = new double[N * K];
                for (int i = 0; i < N * K; i++)
                {
                    var h = (System.Half)(rng.NextDouble() * 2 - 1);
                    w[i] = (double)h;
                    BitConverter.TryWriteBytes(new Span<byte>(wBytes, 2 * i, 2), BitConverter.HalfToUInt16Bits(h));
                }
                var want = new double[M * N]; var scale = new double[M * N];
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                    {
                        double sum = 0, abs = 0;
                        for (int k = 0; k < K; k++) { double p = x[m * K + k] * w[n * K + k]; sum += p; abs += Math.Abs(p); }
                        want[m * N + n] = sum; scale[m * N + n] = abs;
                    }
                using var xBuf = accelerator.Allocate1D(x);
                using var wBuf = AllocatePadded(accelerator, wBytes);
                using var oBuf = accelerator.Allocate1D<float>(M * N);
                async Task<(string Route, double Worst)> Run(bool mma)
                {
                    SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableMmaF16 = mma;
                    oBuf.MemSetToZero();
                    fused.Forward(xBuf.View, wBuf.View, oBuf.View, M, K, N, GGMLType.F16);
                    string route = fused.LastGemmKernel;
                    await accelerator.SynchronizeAsync();
                    var got = await oBuf.CopyToHostAsync<float>(0, M * N);
                    double worst = 0;
                    for (int i = 0; i < got.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / Math.Max(1e-30, scale[i]));
                    return (route, worst);
                }
                var fp32 = await Run(false);
                var split = await Run(true);
                string note = $"M={M} K={K} N={N}: {split.Route} worst {split.Worst:E1}, {fp32.Route} (fp32 path) worst {fp32.Worst:E1}";
                notes.Add(note);
                if (!split.Route.StartsWith("mma-f16-split")) throw new Exception($"{note} - the F16 GEMM did not take the split tensor-core route");
                if (!(split.Worst < 4e-6)) throw new Exception($"{note} - the split route lost accuracy (limit 4e-6 of sum |x w|)");
            }
        }
        finally
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.EnableMmaF16 = saved;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.MmaF16MinRows = savedRows;
        }
        return string.Join("; ", notes);
    });
}
