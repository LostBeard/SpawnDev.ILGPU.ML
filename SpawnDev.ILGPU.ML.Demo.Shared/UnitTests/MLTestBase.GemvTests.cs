using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GEMV (M == 1) decode-shape correctness - locks the M == 1 routing in
/// MatMulKernel.MatMul / BatchedMatMul (simple coalesced thread-per-output kernel
/// instead of the 16x16 tiled kernel that pads the single row to a 16-row tile,
/// idling 15/16 of every group) and the quantized fused path at the decode shape.
/// LLM decode is M == 1 everywhere: x @ W for projections, q @ K^T / attn @ V per
/// head for attention. Shapes here mirror that (large K, batched heads).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Gemv_M1_MatchesCPU_LargeK() => await RunTest(async accelerator =>
    {
        const int M = 1, K = 4096, N = 512;
        var rng = new Random(31);
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += a[k] * b[k * N + n];
            expected[n] = sum;
        }

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var cBuf = accelerator.Allocate1D<float>(M * N);
        var mm = new MatMulKernel(accelerator);
        mm.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var got = await cBuf.CopyToHostAsync<float>(0, M * N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv M=1");
        Console.WriteLine($"[Gemv] M=1 K={K} N={N}: matches CPU reference");
    });

    [TestMethod]
    public async Task Gemv_BatchedM1_DecodeAttentionShape_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Decode-time attention: per head, one query row against K^T - batch = heads,
        // M = 1, K = headDim, N = sequence length.
        const int batch = 8, M = 1, K = 64, N = 512;
        var rng = new Random(37);
        var a = new float[batch * M * K];
        var b = new float[batch * K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[batch * N];
        for (int bi = 0; bi < batch; bi++)
            for (int n = 0; n < N; n++)
            {
                float sum = 0f;
                for (int k = 0; k < K; k++) sum += a[bi * K + k] * b[bi * K * N + k * N + n];
                expected[bi * N + n] = sum;
            }

        using var aBuf = accelerator.Allocate1D(a);
        using var bBuf = accelerator.Allocate1D(b);
        using var cBuf = accelerator.Allocate1D<float>(batch * N);
        var mm = new MatMulKernel(accelerator);
        mm.BatchedMatMul(aBuf.View, bBuf.View, cBuf.View, batch, M, K, N);
        await accelerator.SynchronizeAsync();
        var got = await cBuf.CopyToHostAsync<float>(0, batch * N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv batched M=1");
        Console.WriteLine($"[Gemv] batched M=1 heads={batch} K={K} N={N}: matches CPU reference");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ6K_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Quantized projection at the decode shape: x[1,K] @ W(Q6_K, stored [N][K]).
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q6_K;
        var rng = new Random(41);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv quantized Q6_K M=1");
        Console.WriteLine($"[Gemv] quantized Q6_K M=1 K={K} N={N}: matches oracle");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ4K_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Exercises the M==1 coalesced GEMV path (group-per-column + shared-mem reduction) added to
        // FusedDequantMatMul for Q4_K — the gemma4 decode hot path (96.9% of decode time). Same oracle
        // shape as the Q6_K case; must match the CPU dequant·GEMV reference bit-for-tolerance.
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q4_K;
        var rng = new Random(43);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv quantized Q4_K M=1");
        Console.WriteLine($"[Gemv] quantized Q4_K M=1 K={K} N={N}: matches oracle");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ4K_Warp_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Exercises the VECTORIZED warp-cooperative Q4_K GEMV (EnableWarpGemv / GGUF_GEMV_V2): one 32-lane warp
        // per output column, each lane loading its nibble word ONCE and decoding all 8 nibbles, scales + reduction
        // via Warp.Shuffle (no Group.Barrier / no shared mem). ~2.5x the default GEMV's bandwidth on a 4070,
        // argmax-identical on qwen2.5-coder. On a backend whose warp size != 32 (CPU/Wasm) it falls back to the
        // default GEMV (still correct). Must match the CPU dequant·GEMV reference to GEMV-reduction tolerance.
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q4_K;
        var rng = new Random(43);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        bool saved = Kernels.FusedDequantMatMul.EnableWarpGemv;
        try
        {
            Kernels.FusedDequantMatMul.EnableWarpGemv = true;
            fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
            await accelerator.SynchronizeAsync();
        }
        finally { Kernels.FusedDequantMatMul.EnableWarpGemv = saved; }
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv warp Q4_K M=1");
        Console.WriteLine($"[Gemv] WARP Q4_K M=1 K={K} N={N} (warpSize={accelerator.WarpSize}): matches oracle");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ6K_Warp_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Vectorized warp-cooperative Q6_K GEMV (EnableWarpGemv / GGUF_GEMV_V2): one 32-lane warp per output
        // column, lane==l reads each block's qh[l]+ql[l]+ql[l+32] ONCE and decodes all 4 variants (vs the default
        // kernel's per-element re-reads), reducing via Warp.ShuffleDown. CUDA runs the warp kernel; other backends
        // fall back to the portable GEMV. Must match the CPU dequant·GEMV reference to GEMV-reduction tolerance.
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q6_K;
        var rng = new Random(61);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        bool saved = Kernels.FusedDequantMatMul.EnableWarpGemv;
        try
        {
            Kernels.FusedDequantMatMul.EnableWarpGemv = true;
            fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
            await accelerator.SynchronizeAsync();
        }
        finally { Kernels.FusedDequantMatMul.EnableWarpGemv = saved; }
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv warp Q6_K M=1");
        Console.WriteLine($"[Gemv] WARP Q6_K M=1 K={K} N={N} (warpSize={accelerator.WarpSize}): matches oracle");
    });

    [TestMethod]
    public async Task CudaAsm_Dp4a_InlinePtx_Works() => await RunTest(async accelerator =>
    {
        // Validates ILGPU inline-PTX (CudaAsm.Emit) + the dp4a (4x int8 dot-product) instruction in THIS build —
        // the enabler for an int8-dot-product decode GEMV (the llama.cpp MMVQ technique: keep weights int, dot
        // via dp4a, no float upconvert). CUDA-only intrinsic; other backends skip (the kernel is only loaded on
        // CUDA, so they never compile the PTX). a=int8[1,2,3,4], b=int8[5,6,7,8] → 1*5+2*6+3*7+4*8 = 70.
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
        {
            Console.WriteLine($"[CudaAsm] dp4a skipped on {accelerator.AcceleratorType} (CUDA-only intrinsic)");
            return;
        }
        int a = (1 & 0xFF) | (2 << 8) | (3 << 16) | (4 << 24);
        int b = (5 & 0xFF) | (6 << 8) | (7 << 16) | (8 << 24);
        using var outBuf = accelerator.Allocate1D<int>(1);
        var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(Dp4aKernel);
        k(1, outBuf.View, a, b);
        await accelerator.SynchronizeAsync();
        var got = (await outBuf.CopyToHostAsync<int>(0, 1))[0];
        if (got != 70) throw new Exception($"dp4a expected 70, got {got} — inline-PTX dp4a path is NOT usable in this build");
        Console.WriteLine("[CudaAsm] dp4a.s32.s32 = 70 ✓ — inline-PTX path validated (int8-dot decode GEMV is buildable in ILGPU today)");
    });

    private static void Dp4aKernel(Index1D i, ArrayView<int> outp, int a, int b)
    {
        global::ILGPU.Runtime.Cuda.CudaAsm.Emit("dp4a.s32.s32 %0, %1, %2, %3;", out int r, a, b, 0);
        outp[0] = r;
    }

    [TestMethod]
    public async Task Gemv_M1_Dp4a_Q4K_MatchesInt8ActReference() => await RunTest(async accelerator =>
    {
        // The dp4a int8-activation Q4_K decode GEMV (the llama.cpp/Ollama MMVQ path, EnableDp4aGemv). It int8-
        // quantizes the activation (block_q8_1) and dots in the integer domain via dp4a. The output is int8-
        // APPROXIMATE (not float-exact) — exactly Ollama's approximation — so the oracle is an int8-activation
        // CPU reference (quantize the activation the SAME way, requantize, dot with the float weights), which the
        // kernel must match to float-reduction tolerance. We also report the loss vs the float-exact dot.
        // CUDA-only (dp4a inline-PTX); other backends skip (the path is gated to CUDA, they never load the kernel).
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
        {
            Console.WriteLine($"[Gemv] dp4a Q4_K: CUDA-only, skipped on {accelerator.AcceleratorType}");
            return;
        }
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q4_K;
        var rng = new Random(91);
        var input = new float[K];
        for (int i = 0; i < K; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rb = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rb, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rb, K);
        }

        // int8-activation reference: quantize the activation per 32-block EXACTLY as QuantizeActQ8_1Impl does
        // (amax/127 scale, round-half-up, clamp [-127,127]), requantize (q·d), dot with the float weights.
        var aq = new float[K];
        for (int blk = 0; blk < K / 32; blk++)
        {
            float amax = 0f;
            for (int j = 0; j < 32; j++) { float a = MathF.Abs(input[blk * 32 + j]); if (a > amax) amax = a; }
            float d = amax / 127f, invd = amax > 0f ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++)
            {
                float v = input[blk * 32 + j] * invd;
                int q = (int)(v + (v >= 0f ? 0.5f : -0.5f));
                q = Math.Clamp(q, -127, 127);
                aq[blk * 32 + j] = q * d;
            }
        }
        var expected = new float[N];
        var exact = new float[N];
        for (int n = 0; n < N; n++)
        {
            float s = 0f, e = 0f;
            for (int k = 0; k < K; k++) { s += aq[k] * wRows[n][k]; e += input[k] * wRows[n][k]; }
            expected[n] = s; exact[n] = e;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        bool saved = Kernels.FusedDequantMatMul.EnableDp4aGemv;
        try
        {
            Kernels.FusedDequantMatMul.EnableDp4aGemv = true;
            fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
            await accelerator.SynchronizeAsync();
        }
        finally { Kernels.FusedDequantMatMul.EnableDp4aGemv = saved; }
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 1e-2f, "dp4a Q4_K vs int8-activation reference");
        float maxRel = 0f;
        for (int n = 0; n < N; n++) { float r = MathF.Abs(got[n] - exact[n]) / (MathF.Abs(exact[n]) + 1e-3f); if (r > maxRel) maxRel = r; }
        Console.WriteLine($"[Gemv] dp4a Q4_K M=1 K={K} N={N}: matches int8-act reference; max rel err vs FLOAT-exact = {maxRel:P2} (Ollama-style activation-quant loss)");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ8_0_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Exercises the M==1 coalesced GEMV path for Q8_0 (34B/32: [d][32 int8]).
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q8_0;
        var rng = new Random(44);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv quantized Q8_0 M=1");
        Console.WriteLine($"[Gemv] quantized Q8_0 M=1 K={K} N={N}: matches oracle");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedQ4_0_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Exercises the M==1 coalesced GEMV path for Q4_0 (18B/32: [d][16 nibble bytes]).
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.Q4_0;
        var rng = new Random(45);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv quantized Q4_0 M=1");
        Console.WriteLine($"[Gemv] quantized Q4_0 M=1 K={K} N={N}: matches oracle");
    });

    [TestMethod]
    public async Task Gemv_M1_QuantizedMXFP4_MatchesOracle() => await RunTest(async accelerator =>
    {
        // Exercises the M==1 coalesced GEMV path for MXFP4 (17B/32: [e:E8M0][16 nibble bytes]).
        const int M = 1, K = 4096, N = 256;
        var type = GGMLType.MXFP4;
        var rng = new Random(46);
        var input = new float[K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = RowBytes(type, K);
        var weightBytes = new byte[N * bytesPerRow];
        var wRows = new float[N][];
        for (int n = 0; n < N; n++)
        {
            var rowBytes = MakeBlocks(type, K, rng);
            Buffer.BlockCopy(rowBytes, 0, weightBytes, n * bytesPerRow, bytesPerRow);
            wRows[n] = ReferenceDequant(type, rowBytes, K);
        }

        var expected = new float[N];
        for (int n = 0; n < N; n++)
        {
            float sum = 0f;
            for (int k = 0; k < K; k++) sum += input[k] * wRows[n][k];
            expected[n] = sum;
        }

        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = AllocatePadded(accelerator, weightBytes);
        using var outBuf = accelerator.Allocate1D<float>(N);
        using var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outBuf.View, M, K, N, type);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, N);

        AssertCloseQuant(got, expected, 2e-3f, "Gemv quantized MXFP4 M=1");
        Console.WriteLine($"[Gemv] quantized MXFP4 M=1 K={K} N={N}: matches oracle");
    });
}
