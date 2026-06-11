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
}
