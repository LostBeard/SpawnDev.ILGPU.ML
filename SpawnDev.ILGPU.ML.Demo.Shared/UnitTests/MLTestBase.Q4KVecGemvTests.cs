using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The one-row Q4_K decode GEMV on 128-bit loads (<c>GemvQ4_KVecImpl</c>: a lane per t-unit, four outputs a warp, the super-block header as one 16-byte load,
    /// its scales decoded in registers) against the CPU dequant oracle: K = 4096 with N = 203 and K = 1536 with N = 9 (outputs that do not fill the last warp or group),
    /// and it must be the kernel that ran. A 16-byte-misaligned input view must take the word-wise kernel and still match. Added 2026-11-02 with the kernel
    /// (llama3.1:8b decode GEMVs 135-144 -> 411-460 GB/s on an RTX 4070).
    /// </summary>
    [TestMethod]
    public async Task Gemv_Q4KVec_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("ForwardRows takes the per-element kernel on WebGL (no shared memory)");
        var type = GGMLType.Q4_K;
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        foreach (var (K, N) in new[] { (4096, 203), (1536, 9), (512, 4100), (256, 8200) })
        {
            var rng = new Random(71 + K + N);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[K + 1];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            float[] Expected(int off)
            {
                var e = new float[N];
                for (int n = 0; n < N; n++) { double s = 0; for (int k = 0; k < K; k++) s += input[off + k] * wRows[n][k]; e[n] = (float)s; }
                return e;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(N);
            // aligned (offset 0): the vector kernel
            fused.ForwardRows(inputBuf.View.SubView(0, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, N);
            if (route != "q4k-vec") throw new Exception($"K={K} N={N}: an aligned one-row Q4_K GEMV ran on '{route}', not the vector kernel");
            AssertCloseQuant(got, Expected(0), 2e-3f, $"Q4_K vector GEMV K={K} N={N}");
            // misaligned by one float: the word-wise kernel, same answer
            fused.ForwardRows(inputBuf.View.SubView(1, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route2 = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, N);
            if (route2 == "q4k-vec") throw new Exception($"K={K} N={N}: a misaligned input view took the vector kernel (its loads assume 16-byte alignment)");
            AssertCloseQuant(got2, Expected(1), 2e-3f, $"Q4_K word-wise GEMV (misaligned input) K={K} N={N}");
            notes.Add($"K={K} N={N}: {route} and {route2} match the oracle");
        }
        return string.Join("; ", notes);
    });

    /// <summary>
    /// The one-row Q8_0 decode GEMV (<c>GemvQ8_0VecImpl</c>: a lane per block pair, 17 aligned word loads, four outputs a warp) against the CPU dequant oracle:
    /// K = 2048 with N = 203, K = 192 with N = 9 (fewer pairs than lanes), K = 6144 with N = 5; it must be the kernel that ran, and a misaligned input view
    /// must take the word-wise kernel and still match. Added 2026-11-05 with the kernel (qwen3:1.7b Q8_0 decode: its GEMVs ran ~200 GB/s on the RTX 4070).
    /// </summary>
    [TestMethod]
    public async Task Gemv_Q8Vec_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("ForwardRows takes the per-element kernel on WebGL (no shared memory)");
        var type = GGMLType.Q8_0;
        // off by default on WebGPU (no gain measured there); the kernel itself is gated on every lane that can run it
        bool q8WebSaved = Kernels.FusedDequantMatMul.Q8VecWebGPU;
        Kernels.FusedDequantMatMul.Q8VecWebGPU = true;
        try
        {
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        foreach (var (K, N) in new[] { (2048, 203), (192, 9), (6144, 5), (512, 4100), (256, 8200) })
        {
            var rng = new Random(71 + K + N);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[K + 1];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            float[] Expected(int off)
            {
                var e = new float[N];
                for (int n = 0; n < N; n++) { double s = 0; for (int k = 0; k < K; k++) s += input[off + k] * wRows[n][k]; e[n] = (float)s; }
                return e;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(N);
            // aligned (offset 0): the vector kernel
            fused.ForwardRows(inputBuf.View.SubView(0, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, N);
            if (route != "q8-vec") throw new Exception($"K={K} N={N}: an aligned one-row Q8_0 GEMV ran on '{route}', not the vector kernel");
            AssertCloseQuant(got, Expected(0), 2e-3f, $"Q8_0 vector GEMV K={K} N={N}");
            // misaligned by one float: the word-wise kernel, same answer
            fused.ForwardRows(inputBuf.View.SubView(1, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route2 = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, N);
            if (route2 == "q8-vec") throw new Exception($"K={K} N={N}: a misaligned input view took the vector kernel (its loads assume 16-byte alignment)");
            AssertCloseQuant(got2, Expected(1), 2e-3f, $"Q8_0 word-wise GEMV (misaligned input) K={K} N={N}");
            notes.Add($"K={K} N={N}: {route} and {route2} match the oracle");
        }
        return string.Join("; ", notes);
        }
        finally { Kernels.FusedDequantMatMul.Q8VecWebGPU = q8WebSaved; }
    });

    /// <summary>
    /// The one-row F16 GEMV (<c>GemvF16VecImpl</c>: a lane per 8 values, four outputs a warp) against the CPU oracle: K = 2048 with N = 203, K = 40 with N = 9
    /// (fewer chunks than lanes), K = 6144 with N = 5; it must be the kernel that ran, and a misaligned input must take the rows kernel and still match.
    /// Added 2026-11-06 (Ollama's qwen3:1.7b Q8_0 file has an F16 attn_v: 0.78 ms of a 6.5 ms decode token on the per-element kernel).
    /// </summary>
    [TestMethod]
    public async Task Gemv_F16Vec_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("ForwardRows takes the per-element kernel on WebGL (no shared memory)");
        var type = GGMLType.F16;
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        foreach (var (K, N) in new[] { (2048, 203), (40, 9), (6144, 5), (512, 4100), (256, 8200) })
        {
            var rng = new Random(71 + K + N);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[K + 1];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            float[] Expected(int off)
            {
                var e = new float[N];
                for (int n = 0; n < N; n++) { double s = 0; for (int k = 0; k < K; k++) s += input[off + k] * wRows[n][k]; e[n] = (float)s; }
                return e;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(N);
            // aligned (offset 0): the vector kernel
            fused.ForwardRows(inputBuf.View.SubView(0, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, N);
            if (route != "f16-vec") throw new Exception($"K={K} N={N}: an aligned one-row F16 GEMV ran on '{route}', not the vector kernel");
            AssertCloseQuant(got, Expected(0), 2e-3f, $"F16 vector GEMV K={K} N={N}");
            // misaligned by one float: the word-wise kernel, same answer
            fused.ForwardRows(inputBuf.View.SubView(1, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route2 = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, N);
            if (route2 == "f16-vec") throw new Exception($"K={K} N={N}: a misaligned input view took the vector kernel (its loads assume 16-byte alignment)");
            AssertCloseQuant(got2, Expected(1), 2e-3f, $"F16 rows GEMV (misaligned input) K={K} N={N}");
            notes.Add($"K={K} N={N}: {route} and {route2} match the oracle");
        }
        return string.Join("; ", notes);
    });

    /// <summary>
    /// The one-row Q6_K GEMV (<c>GemvQ6_KVecImpl</c>: a lane per 16 values, four outputs a warp) against the CPU oracle: K = 2048 with N = 203, K = 256 with N = 9
    /// (fewer units than lanes), K = 6144 with N = 5; it must be the kernel that ran, and a misaligned input must take the word-wise kernel and still match.
    /// Added 2026-11-06 (Q4_K_M files keep ffn_down and the LM head in Q6_K).
    /// </summary>
    [TestMethod]
    public async Task Gemv_Q6KVec_MatchesOracle_AndIsTaken() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("ForwardRows takes the per-element kernel on WebGL (no shared memory)");
        var type = GGMLType.Q6_K;
        bool q6WebSaved = Kernels.FusedDequantMatMul.Q6VecWebGPU;
        Kernels.FusedDequantMatMul.Q6VecWebGPU = true;   // gated on every lane that can run it
        try
        {
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        foreach (var (K, N) in new[] { (2048, 203), (256, 9), (6144, 5), (512, 4100), (256, 8200) })
        {
            var rng = new Random(71 + K + N);
            int bytesPerRow = RowBytes(type, K);
            var weightBytes = new byte[N * bytesPerRow];
            var wRows = new float[N][];
            for (int n = 0; n < N; n++)
            {
                var rowBytes = MakeBlocks(type, K, rng);
                Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                wRows[n] = ReferenceDequant(type, rowBytes, K);
            }
            var input = new float[K + 1];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            float[] Expected(int off)
            {
                var e = new float[N];
                for (int n = 0; n < N; n++) { double s = 0; for (int k = 0; k < K; k++) s += input[off + k] * wRows[n][k]; e[n] = (float)s; }
                return e;
            }
            using var weightBuf = AllocatePadded(accelerator, weightBytes);
            using var inputBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(N);
            // aligned (offset 0): the vector kernel
            fused.ForwardRows(inputBuf.View.SubView(0, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, N);
            if (route != "q6k-vec") throw new Exception($"K={K} N={N}: an aligned one-row Q6_K GEMV ran on '{route}', not the vector kernel");
            AssertCloseQuant(got, Expected(0), 2e-3f, $"Q6_K vector GEMV K={K} N={N}");
            // misaligned by one float: the word-wise kernel, same answer
            fused.ForwardRows(inputBuf.View.SubView(1, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route2 = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, N);
            if (route2 == "q6k-vec") throw new Exception($"K={K} N={N}: a misaligned input view took the vector kernel (its loads assume 16-byte alignment)");
            AssertCloseQuant(got2, Expected(1), 2e-3f, $"Q6_K rows GEMV (misaligned input) K={K} N={N}");
            notes.Add($"K={K} N={N}: {route} and {route2} match the oracle");
        }
        return string.Join("; ", notes);
        }
        finally { Kernels.FusedDequantMatMul.Q6VecWebGPU = q6WebSaved; }
    });
}
