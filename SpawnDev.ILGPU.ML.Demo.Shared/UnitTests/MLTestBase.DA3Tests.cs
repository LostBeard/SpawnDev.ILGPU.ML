using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for Depth Anything V3 components: RoPE + QKNorm kernels.
/// Model-level tests require DA3-Small ONNX (deferred until model available).
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task DA3_RoPE_Position0_Identity() => await RunTest(async accelerator =>
    {
        // Position 0: theta=0 for all dims → cos(0)=1, sin(0)=0 → identity
        int headDim = 64;
        var input = new float[headDim];
        var rng = new Random(42);
        for (int i = 0; i < headDim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        using var inputBuf = accelerator.Allocate1D(input);
        using var outputBuf = accelerator.Allocate1D<float>(headDim);

        var rope = new RoPEKernel(accelerator);
        rope.Apply(inputBuf.View, outputBuf.View, 1, headDim, startPosition: 0);
        await accelerator.SynchronizeAsync();
        var output = await outputBuf.CopyToHostAsync<float>(0, headDim);

        float maxDiff = 0;
        for (int i = 0; i < headDim; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(output[i] - input[i]));

        if (maxDiff > 1e-5f)
            throw new Exception($"RoPE position 0 should be identity: maxDiff={maxDiff:F6}");

        Console.WriteLine($"[DA3] RoPE position 0 identity: maxDiff={maxDiff:E3}");
    });

    [TestMethod]
    public async Task DA3_RoPE_DotProduct_PositionInvariant() => await RunTest(async accelerator =>
    {
        // Key property: dot(RoPE(q,p), RoPE(k,p)) = dot(q,k) for same position
        int headDim = 64;
        var rng = new Random(42);
        var q = new float[headDim];
        var k = new float[headDim];
        for (int i = 0; i < headDim; i++)
        {
            q[i] = (float)(rng.NextDouble() * 2 - 1);
            k[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Original dot product
        float origDot = 0;
        for (int i = 0; i < headDim; i++) origDot += q[i] * k[i];

        // Apply RoPE at same position
        using var qBuf = accelerator.Allocate1D(q);
        using var kBuf = accelerator.Allocate1D(k);
        using var qOutBuf = accelerator.Allocate1D<float>(headDim);
        using var kOutBuf = accelerator.Allocate1D<float>(headDim);

        var rope = new RoPEKernel(accelerator);
        rope.Apply(qBuf.View, qOutBuf.View, 1, headDim, startPosition: 5);
        rope.Apply(kBuf.View, kOutBuf.View, 1, headDim, startPosition: 5);
        await accelerator.SynchronizeAsync();

        var qRot = await qOutBuf.CopyToHostAsync<float>(0, headDim);
        var kRot = await kOutBuf.CopyToHostAsync<float>(0, headDim);

        float rotDot = 0;
        for (int i = 0; i < headDim; i++) rotDot += qRot[i] * kRot[i];

        float relErr = MathF.Abs(rotDot - origDot) / (MathF.Abs(origDot) + 1e-10f);

        if (relErr > 0.01f)
            throw new Exception($"RoPE dot product not preserved: orig={origDot:F4}, rotated={rotDot:F4}, relErr={relErr:F4}");

        Console.WriteLine($"[DA3] RoPE dot product invariance: orig={origDot:F4}, rotated={rotDot:F4}, relErr={relErr:E3}");
    });

    [TestMethod]
    public async Task DA3_QKNorm_PreservesDirection() => await RunTest(async accelerator =>
    {
        // Normalized vectors should point in same direction (positive cosine with original)
        int dim = 64;
        var rng = new Random(42);
        var data = new float[dim];
        for (int i = 0; i < dim; i++) data[i] = (float)(rng.NextDouble() * 10 - 5);

        using var inputBuf = accelerator.Allocate1D(data);
        using var outputBuf = accelerator.Allocate1D<float>(dim);

        var qkNorm = new QKNormKernel(accelerator);
        qkNorm.NormalizeRows(inputBuf.View, outputBuf.View, 1, dim);
        await accelerator.SynchronizeAsync();
        var normalized = await outputBuf.CopyToHostAsync<float>(0, dim);

        // Cosine similarity with original should be 1.0 (same direction)
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < dim; i++)
        {
            dot += data[i] * normalized[i];
            normA += data[i] * data[i];
            normB += normalized[i] * normalized[i];
        }
        float cosine = dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);

        if (cosine < 0.999f)
            throw new Exception($"QKNorm changed direction: cosine={cosine:F6}");

        Console.WriteLine($"[DA3] QKNorm preserves direction: cosine={cosine:F6}");
    });
}
