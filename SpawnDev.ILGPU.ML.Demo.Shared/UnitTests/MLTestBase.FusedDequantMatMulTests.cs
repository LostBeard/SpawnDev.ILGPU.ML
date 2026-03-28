using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for FusedDequantMatMul — Q4_0 weight dequantization inside MatMul.
/// Validates that dequantize-in-register produces same result as explicit dequant + MatMul.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task FusedDequantMatMul_SmallMatrix_MatchesCPU() => await RunTest(async accelerator =>
    {
        // 2×32 input × 32×2 Q4_0 weight = 2×2 output
        // Using block size of 32 (one Q4_0 block per weight row)
        int M = 2, K = 32, N = 2;

        var rng = new Random(42);
        var input = new float[M * K];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        // Create Q4_0 packed weights for a [K, N] matrix
        // Q4_0: each block of 32 values = 18 bytes (2 scale FP16 + 16 data)
        int blocksPerRow = (K + 31) / 32;
        int bytesPerRow = blocksPerRow * 18;
        var weightQ4 = new byte[N * bytesPerRow];
        var weightFloat = new float[K * N]; // CPU reference weights

        for (int n = 0; n < N; n++)
        {
            for (int blockIdx = 0; blockIdx < blocksPerRow; blockIdx++)
            {
                int blockStart = blockIdx * 32;
                int blockOffset = n * bytesPerRow + blockIdx * 18;

                // Generate 32 quantized values: nibble ∈ [0,15], dequant = (nibble - 8) * scale
                float scale = 0.1f + (float)rng.NextDouble() * 0.5f;

                // Write scale as FP16
                int scaleFP16 = FloatToHalf(scale);
                weightQ4[blockOffset] = (byte)(scaleFP16 & 0xFF);
                weightQ4[blockOffset + 1] = (byte)((scaleFP16 >> 8) & 0xFF);

                for (int i = 0; i < 32 && blockStart + i < K; i++)
                {
                    int nibble = rng.Next(0, 16);
                    int byteIdx = blockOffset + 2 + (i / 2);
                    if (i % 2 == 0)
                        weightQ4[byteIdx] = (byte)((weightQ4[byteIdx] & 0xF0) | (nibble & 0xF));
                    else
                        weightQ4[byteIdx] = (byte)((weightQ4[byteIdx] & 0x0F) | ((nibble & 0xF) << 4));

                    // Store dequantized reference
                    float readBackScale = HalfToFloatCPU(scaleFP16);
                    weightFloat[(blockStart + i) * N + n] = (nibble - 8) * readBackScale;
                }
            }
        }

        // CPU reference MatMul: input[M,K] × weightFloat[K,N] = expected[M,N]
        var expected = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += input[m * K + k] * weightFloat[k * N + n];
                expected[m * N + n] = sum;
            }

        // GPU fused dequant MatMul
        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = accelerator.Allocate1D(weightQ4);
        using var outputBuf = accelerator.Allocate1D<float>(M * N);

        var fused = new FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, outputBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var gpuOut = await outputBuf.CopyToHostAsync<float>(0, M * N);

        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuOut[i] - expected[i]));

        Console.WriteLine($"[FusedDequantMatMul] 2×32×2: maxErr={maxErr:E3}");
        if (maxErr > 0.1f)
            throw new Exception($"FusedDequantMatMul maxErr={maxErr:E3} exceeds tolerance 0.1");
    });

    /// <summary>FP32 → FP16 conversion for test data generation.</summary>
    private static int FloatToHalf(float f)
    {
        // Simple conversion: handle normal range only (sufficient for scale factors)
        if (f == 0) return 0;
        int sign = f < 0 ? 1 : 0;
        f = MathF.Abs(f);
        int exp = (int)MathF.Floor(MathF.Log2(f));
        float frac = f / MathF.Pow(2, exp) - 1f;
        int biasedExp = exp + 15;
        if (biasedExp <= 0) return (sign << 15); // underflow → zero
        if (biasedExp >= 31) return (sign << 15) | 0x7C00; // overflow → inf
        int mant = (int)(frac * 1024f + 0.5f);
        if (mant > 1023) mant = 1023;
        return (sign << 15) | (biasedExp << 10) | mant;
    }

    /// <summary>
    /// Validates the Q4 routing through MatMulOperator — same as above but exercises
    /// the QuantizedWeights → MatMulOperator → FusedDequantMatMul pipeline.
    /// </summary>
    [TestMethod]
    public async Task Q4MatMulRouting_ViaOperator_MatchesDirect() => await RunTest(async accelerator =>
    {
        int M = 2, K = 32, N = 2;
        var rng = new Random(42);

        var input = new float[M * K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // Build Q4 weight bytes
        int blocksPerRow = (K + 31) / 32;
        int bytesPerRow = blocksPerRow * 18;
        var weightQ4 = new byte[N * bytesPerRow];
        for (int n = 0; n < N; n++)
            for (int blockIdx = 0; blockIdx < blocksPerRow; blockIdx++)
            {
                int blockOffset = n * bytesPerRow + blockIdx * 18;
                float scale = 0.1f + (float)rng.NextDouble() * 0.5f;
                int scaleFP16 = FloatToHalf(scale);
                weightQ4[blockOffset] = (byte)(scaleFP16 & 0xFF);
                weightQ4[blockOffset + 1] = (byte)((scaleFP16 >> 8) & 0xFF);
                for (int i = 0; i < 32; i++)
                {
                    int nibble = rng.Next(0, 16);
                    int byteIdx = blockOffset + 2 + (i / 2);
                    if (i % 2 == 0) weightQ4[byteIdx] = (byte)((weightQ4[byteIdx] & 0xF0) | (nibble & 0xF));
                    else weightQ4[byteIdx] = (byte)((weightQ4[byteIdx] & 0x0F) | ((nibble & 0xF) << 4));
                }
            }

        // Get reference output from direct FusedDequantMatMul
        using var inputBuf = accelerator.Allocate1D(input);
        using var weightBuf = accelerator.Allocate1D(weightQ4);
        using var directOut = accelerator.Allocate1D<float>(M * N);
        var fused = new Kernels.FusedDequantMatMul(accelerator);
        fused.Forward(inputBuf.View, weightBuf.View, directOut.View, M, K, N);
        await accelerator.SynchronizeAsync();
        var directResult = await directOut.CopyToHostAsync<float>(0, M * N);

        // Now test via MatMulOperator with QuantizedWeights routing
        var registry = new OperatorRegistry(accelerator);
        var matmulOp = registry.Resolve("MatMul");

        // Create dummy float tensor for shape tracking (data doesn't matter — Q4 route used)
        using var dummyWeight = accelerator.Allocate1D<float>(K * N);
        using var routedOut = accelerator.Allocate1D<float>(M * N);

        var ctx = new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensors.Tensor(inputBuf.View, new[] { M, K }),
                new Tensors.Tensor(dummyWeight.View, new[] { K, N }),
            },
            Outputs = new[] { new Tensors.Tensor(routedOut.View, new[] { M, N }) },
            Attributes = new Dictionary<string, object>(),
            Pool = new Tensors.BufferPool(accelerator),
            InputNames = new[] { "input", "weight_q4" },
            QuantizedWeights = new Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>
            {
                ["weight_q4"] = weightBuf.View
            },
        };

        matmulOp.Execute(ctx);
        await accelerator.SynchronizeAsync();
        var routedResult = await routedOut.CopyToHostAsync<float>(0, M * N);

        // Direct and routed should match exactly
        float maxErr = 0;
        for (int i = 0; i < M * N; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(routedResult[i] - directResult[i]));

        Console.WriteLine($"[Q4Routing] Direct vs Routed maxErr={maxErr:E3}");
        Console.WriteLine($"[Q4Routing] Direct: [{string.Join(", ", directResult.Select(v => v.ToString("F4")))}]");
        Console.WriteLine($"[Q4Routing] Routed: [{string.Join(", ", routedResult.Select(v => v.ToString("F4")))}]");

        if (maxErr > 1e-6f)
            throw new Exception($"Q4 routing mismatch: maxErr={maxErr:E3}. Direct and routed paths should be identical.");

        Console.WriteLine("[Q4Routing] PASS — MatMulOperator correctly routes to FusedDequantMatMul");
    });

    /// <summary>FP16 → FP32 conversion (CPU reference, matches kernel's HalfToFloat).</summary>
    private static float HalfToFloatCPU(int h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) return mant == 0 ? (sign == 1 ? -0f : 0f) : (sign == 1 ? -1 : 1) * mant / 1024f * (1f / 16384f);
        if (exp == 31) return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;
        float result = (1f + mant / 1024f) * MathF.Pow(2, exp - 15);
        return sign == 1 ? -result : result;
    }
}
