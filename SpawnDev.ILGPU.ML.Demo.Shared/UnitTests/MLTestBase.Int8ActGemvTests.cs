using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// OPT-IN <c>FusedDequantMatMul.Int8ActGemv</c> (2026-11-10, llama.cpp / Ollama's MMVQ scheme): the one-row Q4_K GEMV on the activations quantized to
    /// ggml's q8_1 (32-value blocks, d = amax / 127, q = round(x / d)) and dotted with dp4a. Two checks per shape:
    /// (1) FIDELITY to the scheme: against an f64 CPU dot of the dequantized weights with the q8_1-dequantized activations (the ggml formula
    /// d * sc * d8 * dot - dmin * m * d8 * sum q equals exactly that dot), within 1e-5 of sum |w x_q| - float rounding only;
    /// (2) the scheme's ACCURACY COST, reported (not gated): the same output against the f64 fp32-activation dot, relative to sum |w x|.
    /// Shapes: qwen3 1.7B's q+k (2048 x 3072), o (2048 x 2048), ffn_down (6144 x 2048), and ragged ones; activations spanning three decades per block.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task<string> Gemv_Q4K_Int8Act_MatchesGgmlQ8_1Scheme() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("dp4a (CUDA)");
        var type = GGMLType.Q4_K;
        using var fused = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(accelerator);
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        var notes = new List<string>();
        try
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = true;
            foreach (var (K, N) in new[] { (2048, 3072), (2048, 2048), (6144, 2048), (4096, 203), (256, 9) })
            {
                var rng = new Random(23 + K + N);
                int bytesPerRow = RowBytes(type, K);
                var weightBytes = new byte[N * bytesPerRow];
                var wRows = new float[N][];
                for (int n = 0; n < N; n++)
                {
                    var rowBytes = MakeBlocks(type, K, rng);
                    Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                    wRows[n] = ReferenceDequant(type, rowBytes, K);
                }
                var x = new float[K];
                for (int i = 0; i < K; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, -2 + 2 * rng.NextDouble()));
                // ggml's q8_1 on the CPU, in the kernel's float32 arithmetic
                var xq = new double[K];
                for (int b = 0; b < K / 32; b++)
                {
                    float amax = 0f;
                    for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(x[b * 32 + j]));
                    float d = amax / 127f;
                    for (int j = 0; j < 32; j++)
                    {
                        int q = 0;
                        if (amax > 0f) { float t = x[b * 32 + j] / d; q = (int)(t + (t >= 0f ? 0.5f : -0.5f)); }
                        xq[b * 32 + j] = (double)d * q;
                    }
                }
                using var weightBuf = AllocatePadded(accelerator, weightBytes);
                using var inputBuf = accelerator.Allocate1D(x);
                using var outBuf = accelerator.Allocate1D<float>(N);
                outBuf.MemSetToZero();
                fused.ForwardRows(inputBuf.View, weightBuf.View, outBuf.View, 1, K, N, type);
                string route = fused.LastRowsKernel;
                await accelerator.SynchronizeAsync();
                var got = await outBuf.CopyToHostAsync<float>(0, N);
                if (route != "q4k-dp4a-vec") throw new Exception($"K={K} N={N}: the int8-activation route did not run (ran '{route}')");
                double worstFid = 0, worstCost = 0, sumCost = 0;
                for (int n = 0; n < N; n++)
                {
                    double refQ = 0, absQ = 0, refF = 0, absF = 0;
                    for (int k = 0; k < K; k++)
                    {
                        double w = wRows[n][k];
                        refQ += w * xq[k]; absQ += Math.Abs(w * xq[k]);
                        refF += w * x[k]; absF += Math.Abs(w * x[k]);
                    }
                    worstFid = Math.Max(worstFid, Math.Abs(got[n] - refQ) / Math.Max(1e-30, absQ));
                    double cost = Math.Abs(got[n] - refF) / Math.Max(1e-30, absF);
                    worstCost = Math.Max(worstCost, cost); sumCost += cost;
                }
                string note = $"K={K} N={N}: vs the q8_1 scheme worst {worstFid:E1}; scheme cost vs fp32 activations mean {sumCost / N:E1} worst {worstCost:E1}";
                notes.Add(note);
                if (!(worstFid < 1e-5)) throw new Exception(note + " - the kernel does not compute ggml's q8_1 scheme (limit 1e-5)");
            }
        }
        finally { SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = saved; }
        return string.Join("; ", notes);
    });
}
