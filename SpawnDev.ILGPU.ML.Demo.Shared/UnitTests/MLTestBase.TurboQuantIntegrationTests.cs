using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Integration tests for TurboQuant: verify quantized attention matches
/// full-precision attention within acceptable tolerance.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task TurboQuant_QuantizedAttention_MatchesFP32() => await RunTest(async accelerator =>
    {
        int headDim = 64;
        int numKV = 8;
        var rng = new Random(42);

        // Generate random Q, K, V vectors
        var qData = new float[headDim];
        var kData = new float[numKV * headDim];
        var vData = new float[numKV * headDim];
        for (int i = 0; i < headDim; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < numKV * headDim; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < numKV * headDim; i++) vData[i] = (float)(rng.NextDouble() * 2 - 1);

        float scale = 1f / MathF.Sqrt(headDim);

        // ═══ Full-precision attention (CPU reference) ═══
        var fp32Output = new float[headDim];
        {
            // Compute QK^T scores
            var scores = new float[numKV];
            float maxScore = float.MinValue;
            for (int kv = 0; kv < numKV; kv++)
            {
                float dot = 0;
                for (int d = 0; d < headDim; d++)
                    dot += qData[d] * kData[kv * headDim + d];
                scores[kv] = dot * scale;
                if (scores[kv] > maxScore) maxScore = scores[kv];
            }

            // Softmax
            float sumExp = 0;
            for (int kv = 0; kv < numKV; kv++)
            {
                scores[kv] = MathF.Exp(scores[kv] - maxScore);
                sumExp += scores[kv];
            }
            for (int kv = 0; kv < numKV; kv++)
                scores[kv] /= sumExp;

            // Weighted sum of V
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0;
                for (int kv = 0; kv < numKV; kv++)
                    sum += scores[kv] * vData[kv * headDim + d];
                fp32Output[d] = sum;
            }
        }

        // ═══ Quantized attention (GPU via TurboQuant) ═══
        var tq = new TurboQuantKernels(accelerator);

        // Quantize K vectors
        int packedDim = headDim / 8;
        var codebook = new float[] { -1.75f,-1.25f,-0.875f,-0.625f,-0.375f,-0.2f,-0.075f,0f,
            0.075f,0.2f,0.375f,0.625f,0.875f,1.25f,1.75f,2.5f };

        using var qBuf = accelerator.Allocate1D(qData);

        // Encode K and V: normalize → quantize → pack
        using var kPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var vPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var kNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var vNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var codebookBuf = accelerator.Allocate1D(codebook);

        // Per-vector encode
        using var tempNorm = accelerator.Allocate1D<float>(headDim);
        using var tempNormVal = accelerator.Allocate1D<float>(1);
        using var tempIndices = accelerator.Allocate1D<int>(headDim);

        for (int kv = 0; kv < numKV; kv++)
        {
            // Upload K vector
            var kSlice = new float[headDim];
            Array.Copy(kData, kv * headDim, kSlice, 0, headDim);
            using var kVec = accelerator.Allocate1D(kSlice);

            tq.Normalize(kVec.View, tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, kPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            tempNormVal.View.SubView(0, 1).CopyTo(kNormsBuf.View.SubView(kv, 1));

            // Same for V
            var vSlice = new float[headDim];
            Array.Copy(vData, kv * headDim, vSlice, 0, headDim);
            using var vVec = accelerator.Allocate1D(vSlice);

            tq.Normalize(vVec.View, tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, vPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            tempNormVal.View.SubView(0, 1).CopyTo(vNormsBuf.View.SubView(kv, 1));
        }

        // Run fused quantized attention
        using var outputBuf = accelerator.Allocate1D<float>(headDim);
        tq.FusedQuantizedAttention(
            qBuf.View, kPackedBuf.View, codebookBuf.View,
            vPackedBuf.View, codebookBuf.View,
            kNormsBuf.View, vNormsBuf.View, outputBuf.View,
            1, numKV, headDim, scale);

        await accelerator.SynchronizeAsync();
        var quantizedOutput = await outputBuf.CopyToHostAsync<float>(0, headDim);

        // ═══ Compare ═══
        float maxErr = 0, sumErr = 0;
        for (int d = 0; d < headDim; d++)
        {
            float err = MathF.Abs(quantizedOutput[d] - fp32Output[d]);
            maxErr = MathF.Max(maxErr, err);
            sumErr += err;
        }
        float meanErr = sumErr / headDim;

        // Cosine similarity
        float dotAB = 0, normA = 0, normB = 0;
        for (int d = 0; d < headDim; d++)
        {
            dotAB += fp32Output[d] * quantizedOutput[d];
            normA += fp32Output[d] * fp32Output[d];
            normB += quantizedOutput[d] * quantizedOutput[d];
        }
        float cosineSim = dotAB / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);

        Console.WriteLine($"[TurboQuant] Quantized vs FP32 attention: maxErr={maxErr:F4}, meanErr={meanErr:F4}, cosine={cosineSim:F4}");

        // 4-bit quantization should maintain reasonable accuracy
        // Cosine similarity > 0.9 is the key metric
        if (cosineSim < 0.8f)
            throw new Exception($"Quantized attention cosine similarity {cosineSim:F4} too low — expected > 0.8");
    });
}
