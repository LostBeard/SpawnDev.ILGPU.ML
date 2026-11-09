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
        bool? shuffleSaved = Kernels.FusedDequantMatMul.Q4KVecShuffle;
        try
        {
        foreach (var (K, N) in new[] { (4096, 203), (1536, 9), (512, 4100), (256, 8200), (2048, 3072) })
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
            // aligned (offset 0): the vector kernel, with the shared-memory reduction and (32-wide warps) with the shuffle reduction
            string route = "";
            foreach (bool shuffle in new[] { false, true })
            {
                Kernels.FusedDequantMatMul.Q4KVecShuffle = shuffle;
                string want = shuffle && fused.CanShuffle32 ? "q4k-vec-shuffle" : "q4k-vec";
                outBuf.MemSetToZero();   // a stale result from the other reduction must not pass
                fused.ForwardRows(inputBuf.View.SubView(0, K), weightBuf.View, outBuf.View, 1, K, N, type);
                string r = fused.LastRowsKernel;
                await accelerator.SynchronizeAsync();
                var got = await outBuf.CopyToHostAsync<float>(0, N);
                if (r != want) throw new Exception($"K={K} N={N}: an aligned one-row Q4_K GEMV ran on '{r}', not '{want}'");
                AssertCloseQuant(got, Expected(0), 2e-3f, $"Q4_K vector GEMV ({r}) K={K} N={N}");
                route += (route.Length > 0 ? " + " : "") + r;
            }
            Kernels.FusedDequantMatMul.Q4KVecShuffle = shuffleSaved;
            // misaligned by one float: the word-wise kernel, same answer
            fused.ForwardRows(inputBuf.View.SubView(1, K), weightBuf.View, outBuf.View, 1, K, N, type);
            string route2 = fused.LastRowsKernel;
            await accelerator.SynchronizeAsync();
            var got2 = await outBuf.CopyToHostAsync<float>(0, N);
            if (route2.StartsWith("q4k-vec")) throw new Exception($"K={K} N={N}: a misaligned input view took the vector kernel (its loads assume 16-byte alignment)");
            AssertCloseQuant(got2, Expected(1), 2e-3f, $"Q4_K word-wise GEMV (misaligned input) K={K} N={N}");
            notes.Add($"K={K} N={N}: {route} and {route2} match the oracle");
        }
        }
        finally { Kernels.FusedDequantMatMul.Q4KVecShuffle = shuffleSaved; }
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

    /// <summary>
    /// Q8_0 regrouped in place (<see cref="GGMLType.Q8_0G"/>, FusedDequantMatMul.RepackQ8_0Grouped) on every route that reads it, against the CPU oracle of the
    /// native rows: the device regrouping byte-exact against a CPU one; the one-row vector GEMV (q8g-vec), a misaligned input on the word-wise kernel, three rows
    /// (word-wise), and a 200-row prefill (CUDA sm_75+: the f16 tensor-core GEMM; elsewhere the tiled GEMM through DecodeOf). Each route must be the one that ran.
    /// Added 2026-11-06 with the layout (TJ: "Measure twice, cut once").
    /// </summary>
    [TestMethod]
    public async Task Gemm_Q8_0G_MatchesOracle_AllRoutes() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("ForwardRows takes the per-element kernel on WebGL (no shared memory)");
        int scrMinSaved = Kernels.FusedDequantMatMul.MmaScratchMinRows;
        Kernels.FusedDequantMatMul.MmaScratchMinRows = 1;
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        var notes = new List<string>();
        try
        {
            foreach (var (K, N) in new[] { (2048, 203), (256, 9), (512, 4100) })
            {
                var rng = new Random(91 + K + N);
                int rb = RowBytes(GGMLType.Q8_0, K);
                var bytes = new byte[N * rb];
                var wRows = new float[N][];
                for (int n = 0; n < N; n++)
                {
                    var row = MakeBlocks(GGMLType.Q8_0, K, rng);
                    Buffer.BlockCopy(row, 0, bytes, n * rb, rb);
                    wRows[n] = ReferenceDequant(GGMLType.Q8_0, row, K);
                }
                // the CPU regrouping: per 272 bytes, 8 scales then 256 values
                var want = new byte[bytes.Length];
                for (int g = 0; g < bytes.Length / 272; g++)
                    for (int b = 0; b < 8; b++)
                    {
                        want[g * 272 + 2 * b] = bytes[g * 272 + 34 * b]; want[g * 272 + 2 * b + 1] = bytes[g * 272 + 34 * b + 1];
                        Buffer.BlockCopy(bytes, g * 272 + 34 * b + 2, want, g * 272 + 16 + 32 * b, 32);
                    }
                using var wBuf = AllocatePadded(accelerator, bytes);
                fused.RepackQ8_0Grouped(wBuf.View, bytes.Length / 272);
                await accelerator.SynchronizeAsync();
                var got = await wBuf.View.SubView(0, bytes.Length).CopyToHostAsync();
                for (int i = 0; i < want.Length; i++)
                    if (got[i] != want[i]) throw new Exception($"K={K} N={N}: regrouped byte {i} = {got[i]}, expected {want[i]}");
                const int Mmax = 200;
                var input = new float[Mmax * K + 1];
                for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
                using var inBuf = accelerator.Allocate1D(input);
                using var outBuf = accelerator.Allocate1D<float>((long)Mmax * N);
                async Task Check(int M, int off, string wantRoute, bool gemm)
                {
                    if (gemm) fused.Forward(inBuf.View.SubView(off, (long)M * K), wBuf.View, outBuf.View.SubView(0, (long)M * N), M, K, N, GGMLType.Q8_0G);
                    else fused.ForwardRows(inBuf.View.SubView(off, (long)M * K), wBuf.View, outBuf.View.SubView(0, (long)M * N), M, K, N, GGMLType.Q8_0G);
                    string route = gemm ? fused.LastGemmKernel : fused.LastRowsKernel;
                    await accelerator.SynchronizeAsync();
                    var o = await outBuf.View.SubView(0, (long)M * N).CopyToHostAsync();
                    if (wantRoute.Length > 0 && !route.StartsWith(wantRoute)) throw new Exception($"K={K} N={N} M={M}: ran on '{route}', not '{wantRoute}'");
                    int bad = 0; double worst = 0;
                    for (int m = 0; m < M; m++)
                        for (int n = 0; n < N; n++)
                        {
                            double sum = 0, a = 0;
                            for (int k = 0; k < K; k++) { double pr = input[off + m * K + k] * wRows[n][k]; sum += pr; a += Math.Abs(pr); }
                            double e = Math.Abs(o[m * N + n] - sum), bound = (gemm ? 2e-3 : 2e-5) * a + 1e-5;
                            if (!(e <= bound)) bad++;
                            worst = Math.Max(worst, e / bound);
                        }
                    if (bad > 0) throw new Exception($"{route} K={K} N={N} M={M}: {bad} outputs off (worst {worst:F2} of the bound)");
                    notes.Add($"{route} K={K} N={N} M={M} worst {worst:F2}");
                }
                await Check(1, 0, "q8g-vec", false);
                await Check(1, 1, "rows-Q8_0G", false);                 // misaligned input
                await Check(3, 0, "rows-Q8_0G", false);
                bool tc = accelerator.AcceleratorType == AcceleratorType.Cuda && Kernels.CudaCaps.TuringTensorCores(accelerator);
                if (N <= 2048) await Check(Mmax, 0, tc ? "mma-q8_0g" : "", true);
            }
        }
        finally { Kernels.FusedDequantMatMul.MmaScratchMinRows = scrMinSaved; }
        return string.Join("; ", notes);
    });
}
