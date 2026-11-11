using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
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
        await Int8ActSchemeBody(accelerator, GGMLType.Q4_K, new[] { (2048, 3072), (2048, 2048), (6144, 2048), (4096, 203), (256, 9) }, (K, N) => "q4k-dp4a-vec"));

    /// <summary>
    /// <see cref="Gemv_Q4K_Int8Act_MatchesGgmlQ8_1Scheme"/> for Q6_K (ggml's vec_dot_q6_K_q8_1 in dp4a; 2026-11-10): the split kernel's shapes (N &lt;= 8192:
    /// qwen3 1.7B's ffn_down 6144 x 2048, v 2048 x 1024, ragged ones). Larger N (the LM head) keeps fp32 activations (the dp4a kernel was slower there).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task<string> Gemv_Q6K_Int8Act_MatchesGgmlQ8_1Scheme() => await RunTest(async accelerator =>
        await Int8ActSchemeBody(accelerator, GGMLType.Q6_K, new[] { (6144, 2048), (2048, 1024), (4096, 203), (256, 9), (512, 8191) }, (K, N) => "q6k-dp4a-split"));

    private async Task<string> Int8ActSchemeBody(Accelerator accelerator, GGMLType type, (int K, int N)[] shapes, Func<int, int, string> expectRoute)
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("dp4a (CUDA)");
        using var fused = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(accelerator);
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        var notes = new List<string>();
        try
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = true;
            foreach (var (K, N) in shapes)
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
                if (route != expectRoute(K, N)) throw new Exception($"K={K} N={N}: the int8-activation route did not run (ran '{route}')");
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
    }

    /// <summary>
    /// The producers' fused q8_1 (2026-11-10): <c>NormalizationKernels.AddRMSNormFusedQ8</c> writes the normed row and its q8_1 form in one launch, and a
    /// <c>ForwardRows</c> given that form (<c>FusedDequantMatMul.Q8Act</c>) skips its own quantize. Both must be BIT-IDENTICAL to the unfused pair (the plain fused norm,
    /// then the GEMV quantizing its input itself): the residual, the normed floats and the GEMV outputs. Widths 1,536 / 2,048 / 4,096 (qwen2.5 1.5B, qwen3 1.7B, 4B).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task<string> AddRmsNormQ8_Fused_BitIdentical() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("dp4a (CUDA)");
        var type = GGMLType.Q4_K;
        using var fused = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(accelerator);
        using var norm = new SpawnDev.ILGPU.ML.Kernels.NormalizationKernels(accelerator);
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        var notes = new List<string>();
        try
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = true;
            foreach (var (C, N) in new[] { (1536, 2048), (2048, 3072), (4096, 512) })
            {
                if (!fused.Int8ActFor(1, C)) throw new Exception($"C={C}: Int8ActFor is false with Int8ActGemv on (CUDA, dp4a)");
                var rng = new Random(7 + C);
                int bytesPerRow = RowBytes(type, C);
                var weightBytes = new byte[N * bytesPerRow];
                for (int n = 0; n < N; n++) Buffer.BlockCopy(MakeBlocks(type, C, rng), 0, weightBytes, n * bytesPerRow, bytesPerRow);
                float[] Rand(float s) { var v = new float[C]; for (int i = 0; i < C; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * s); return v; }
                using var aBuf = accelerator.Allocate1D(Rand(3f));
                using var bBuf = accelerator.Allocate1D(Rand(0.5f));
                using var wBuf = accelerator.Allocate1D(Rand(2f));
                using var weightBuf = AllocatePadded(accelerator, weightBytes);
                using var res0 = accelerator.Allocate1D<float>(C); using var nrm0 = accelerator.Allocate1D<float>(C); using var out0 = accelerator.Allocate1D<float>(N);
                using var res1 = accelerator.Allocate1D<float>(C); using var nrm1 = accelerator.Allocate1D<float>(C); using var out1 = accelerator.Allocate1D<float>(N);
                using var qs = accelerator.Allocate1D<byte>(C); using var ds = accelerator.Allocate1D<float>(C / 16);
                // unfused: the plain fused norm, then the GEMV quantizes its input itself
                if (!norm.AddRMSNormFused(aBuf.View, bBuf.View, res0.View, nrm0.View, wBuf.View, 1, C, 1e-6f, 1)) throw new Exception("fused norm refused");
                fused.ForwardRows(nrm0.View, weightBuf.View, out0.View, 1, C, N, type);
                if (fused.LastRowsKernel != "q4k-dp4a-vec") throw new Exception($"C={C}: unfused arm ran '{fused.LastRowsKernel}'");
                // fused: the norm writes q8_1 too, the GEMV reads it
                if (!norm.AddRMSNormFusedQ8(aBuf.View, bBuf.View, res1.View, nrm1.View, wBuf.View, qs.View, ds.View, 1, C, 1e-6f)) throw new Exception("AddRMSNormFusedQ8 refused on CUDA");
                if (norm.LastAddRmsKernel != "shuffle-q8") throw new Exception($"norm ran '{norm.LastAddRmsKernel}'");
                fused.ForwardRows(nrm1.View, weightBuf.View, out1.View, 1, C, N, type, new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Q8Act(qs.View, ds.View));
                if (fused.LastRowsKernel != "q4k-dp4a-vec-pre") throw new Exception($"C={C}: fused arm ran '{fused.LastRowsKernel}'");
                await accelerator.SynchronizeAsync();
                static int Diff(float[] x, float[] y) { int d = 0; for (int i = 0; i < x.Length; i++) if (BitConverter.SingleToInt32Bits(x[i]) != BitConverter.SingleToInt32Bits(y[i])) d++; return d; }
                int dr = Diff(await res0.CopyToHostAsync<float>(0, C), await res1.CopyToHostAsync<float>(0, C));
                int dn = Diff(await nrm0.CopyToHostAsync<float>(0, C), await nrm1.CopyToHostAsync<float>(0, C));
                var o0 = await out0.CopyToHostAsync<float>(0, N);
                int dout = Diff(o0, await out1.CopyToHostAsync<float>(0, N));
                int nonzero = o0.Count(v => v != 0f);
                string note = $"C={C} N={N}: residual {dr} / normed {dn} / GEMV {dout} values differ in bits ({nonzero} of {N} outputs nonzero)";
                notes.Add(note);
                if (dr != 0 || dn != 0 || dout != 0 || nonzero < N / 2) throw new Exception(note);
            }
        }
        finally { SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = saved; }
        return string.Join("; ", notes);
    });

    /// <summary>
    /// The decoder's fused q8_1 producers (AddNorm's <c>AddRMSNormFusedQ8</c>, the SwiGLU's <c>SwigluJoinedQ8Impl</c>) against the GEMVs quantizing their own inputs
    /// (<c>FusedDequantMatMul.Int8ActFuse</c> off): one sequence (the int8 route is one-row), qwen2.5 1.5B Q4_K_M, step-1 and step-4 logits BIT-IDENTICAL and 32 greedy
    /// tokens the same, with CUDA graph capture on. The int8 route itself is checked to be live (its logits differ from fp32 activations').
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task<string> GgufBatchDecoder_Int8ActFused_BitIdentical() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("dp4a (CUDA)");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var hub = new ModelsHostStreamLoader(http);
        var src = await hub.OpenAsync("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", deselect: false);
        using var dec = await GgufBatchDecoder.CreateAsync(accelerator, src.Stream, maxSeq: 512, maxBatch: 1);
        var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
        var (ids, stops) = ChatTemplates.BuildChatPrompt(dec.Model, tok,
            new List<(string, string)> { ("user", "Name three things a lighthouse keeper does each evening.") }, thinking: false);
        var stopSet = stops.ToHashSet(); if (tok.EosId >= 0) stopSet.Add(tok.EosId);
        async Task<(float[] S1, float[] S4, int[] Greedy, bool Fused)> Run(bool int8, bool fuse)
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = int8;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActFuse = fuse;
            dec.InvalidateCapturedSteps();
            await dec.PrefillAsync(ids, 0f);
            await dec.StepAsync(new[] { 11 }, 0f);
            bool f = dec.LastStepInt8Fused;
            var s1 = await dec.ReadLogitsAsync();
            for (int s = 1; s < 4; s++) await dec.StepAsync(new[] { 11 + s }, 0f);
            var s4 = await dec.ReadLogitsAsync();
            var greedy = (await dec.GenerateAsync(ids, 1, 32, 0f, 1f, 1, stopSet))[0];
            return (s1, s4, greedy, f);
        }
        static int Diff(float[] a, float[] b) { if (a.Length != b.Length) return -1; int n = 0; for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++; return n; }
        bool savedI = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv, savedF = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActFuse;
        try
        {
            var fp = await Run(false, true);
            var own = await Run(true, false);
            var fu = await Run(true, true);
            if (fp.Fused || own.Fused) throw new Exception("the fused route ran with it off");
            if (!fu.Fused) throw new Exception("the fused route did not run (LastStepInt8Fused false)");
            int live = Diff(fp.S1, own.S1);
            int d1 = Diff(own.S1, fu.S1), d4 = Diff(own.S4, fu.S4);
            bool same = own.Greedy.SequenceEqual(fu.Greedy);
            string note = $"int8 vs fp32 activations: {live} of {fp.S1.Length} step-1 logits differ (route live); fused vs own quantize: step 1 {d1}, step 4 {d4} logits differ in bits, 32 greedy tokens {(same ? "same" : "DIFFER")}";
            Console.WriteLine($"[Int8ActFused] {note}");
            if (live == 0 || d1 != 0 || d4 != 0 || !same) throw new Exception(note);
            return note;
        }
        finally
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = savedI;
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActFuse = savedF;
            dec.InvalidateCapturedSteps();
        }
    });

    /// <summary>
    /// The prefill half of the OPT-IN int8 activations (2026-11-10, <c>FusedDequantMatMul.Int8ActGemm</c>: MmaS8Gemm, llama.cpp / Ollama's MMQ scheme): M &gt; 1
    /// rows of Q4_K and Q6_K on the int8 tensor-core GEMMs, A in ggml's q8_1. Per shape, on 3,000 sampled outputs: (1) FIDELITY to the scheme against an f64
    /// CPU dot of the dequantized weights with each row's q8_1-dequantized activations, within 1e-5 of sum |w x_q|; (2) the scheme's cost against fp32
    /// activations, reported. Shapes: ragged M and N around the 128 x 128 tiles, qwen3 1.7B's q+k (2048 x 3072) and down (6144 x 2048) at a prefill's rows.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task<string> Gemm_Int8Act_MatchesGgmlQ8_1Scheme() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("int8 tensor cores (CUDA)");
        if (!SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator)) throw new UnsupportedTestException("sm_80+");
        using var fused = new SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul(accelerator);
        bool saved = SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv;
        var notes = new List<string>();
        try
        {
            SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = true;
            foreach (var (type, M, K, N) in new[] { (GGMLType.Q4_K, 9, 256, 8), (GGMLType.Q4_K, 130, 2048, 3072), (GGMLType.Q4_K, 300, 6144, 2048), (GGMLType.Q4_K, 77, 512, 200),
                (GGMLType.Q6_K, 9, 256, 8), (GGMLType.Q6_K, 130, 6144, 2048), (GGMLType.Q6_K, 257, 2048, 1000) })
            {
                var rng = new Random(31 + M + K + N);
                int bytesPerRow = RowBytes(type, K);
                var weightBytes = new byte[N * bytesPerRow];
                var wRows = new float[N][];
                for (int n = 0; n < N; n++)
                {
                    var rowBytes = MakeBlocks(type, K, rng);
                    Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
                    wRows[n] = ReferenceDequant(type, rowBytes, K);
                }
                var x = new float[M * K];
                for (int i = 0; i < x.Length; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, -2 + 2 * rng.NextDouble()));
                var xq = new double[M * K];
                for (int b = 0; b < M * K / 32; b++)
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
                using var outBuf = accelerator.Allocate1D<float>((long)M * N);
                // Q4_K on both kernels: S8Q4KDirect (raw rows, the default) and the unpack + S8Q4K pair
                foreach (bool direct in type == GGMLType.Q4_K ? new[] { true, false } : new[] { true })
                {
                SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.S8Direct = direct;
                outBuf.MemSetToZero();
                fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
                string route = fused.LastGemmKernel;
                await accelerator.SynchronizeAsync();
                var got = await outBuf.CopyToHostAsync<float>(0, (long)M * N);
                // grids of at most half the GPU's SM count take split-K (every int8 kernel; floor(SMs / tiles) slices = one wave): e.g. 77 x 200 (2 tiles), 9 x 8
                bool small = ((M + 127) / 128) * ((N + 127) / 128) * 2 <= accelerator.NumMultiprocessors;   // split only when one wave holds 2+ slices per tile
                string want = (type == GGMLType.Q4_K ? (direct ? "mma-s8-q4k-direct" : "mma-s8-q4k") : "mma-s8-q6k") + (small ? "-splitk" : "");
                if (route != want) throw new Exception($"{type} M={M} K={K} N={N}: the int8 GEMM did not run (ran '{route}', wanted '{want}')");
                if (type == GGMLType.Q4_K && direct && M == 77 && !route.Contains("-splitk")) throw new Exception($"M=77 N=200 (2 tiles) did not take split-K ('{route}')");
                double worstFid = 0, worstCost = 0, sumCost = 0;
                int samples = Math.Min(3000, M * N);
                for (int s = 0; s < samples; s++)
                {
                    int m = samples == M * N ? s / N : rng.Next(M), n = samples == M * N ? s % N : rng.Next(N);
                    double refQ = 0, absQ = 0, refF = 0, absF = 0;
                    for (int k = 0; k < K; k++)
                    {
                        double wv = wRows[n][k];
                        refQ += wv * xq[m * K + k]; absQ += Math.Abs(wv * xq[m * K + k]);
                        refF += wv * x[m * K + k]; absF += Math.Abs(wv * x[m * K + k]);
                    }
                    double g = got[(long)m * N + n];
                    worstFid = Math.Max(worstFid, Math.Abs(g - refQ) / Math.Max(1e-30, absQ));
                    double cost = Math.Abs(g - refF) / Math.Max(1e-30, absF);
                    worstCost = Math.Max(worstCost, cost); sumCost += cost;
                }
                string note = $"{route} M={M} K={K} N={N}: vs the q8_1 scheme worst {worstFid:E1}; cost vs fp32 activations mean {sumCost / samples:E1} worst {worstCost:E1}";
                notes.Add(note);
                if (!(worstFid < 1e-5)) throw new Exception(note + " - the GEMM does not compute ggml's q8_1 scheme (limit 1e-5)");
                }
            }
        }
        finally { SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.Int8ActGemv = saved; SpawnDev.ILGPU.ML.Kernels.FusedDequantMatMul.S8Direct = true; }
        return string.Join("; ", notes);
    });
}
