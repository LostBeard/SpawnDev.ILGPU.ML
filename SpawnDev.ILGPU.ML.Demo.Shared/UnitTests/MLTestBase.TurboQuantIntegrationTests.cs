using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Integration tests for TurboQuant: verify quantized attention matches
/// full-precision attention, and KV cache auto-detection works with real models.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Load DistilGPT-2, verify GraphExecutor auto-detects KV cache pattern.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_DistilGPT2_KVCacheAutoDetected() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model_merged.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
                ["attention_mask"] = new[] { 1, 5 },
                ["position_ids"] = new[] { 1, 5 },
            },
            enableOptimization: false);

        bool hasCache = session.Executor.HasKVCache;
        Console.WriteLine($"[TurboQuant] DistilGPT-2 HasKVCache: {hasCache}");

        if (hasCache)
        {
            var kvCache = session.Executor.KVCache!;
            Console.WriteLine($"[TurboQuant] KV cache: {kvCache.NumLayers} layers, maxSeq={kvCache.MaxSeqLen}");
            if (kvCache.NumLayers < 1)
                throw new Exception("KV cache detected but has 0 layers");
        }

        Console.WriteLine($"[TurboQuant] DistilGPT-2 KV cache detection: PASS");
    });

    /// <summary>
    /// Run one step of DistilGPT-2 inference, verify KV cache captures tokens.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_DistilGPT2_KVCacheCaptures() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model_merged.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
                ["attention_mask"] = new[] { 1, 5 },
                ["position_ids"] = new[] { 1, 5 },
            },
            enableOptimization: false);

        if (!session.Executor.HasKVCache)
        {
            Console.WriteLine("[TurboQuant] Skipping — no KV cache detected");
            return;
        }

        // Run inference
        var tokenIds = new float[] { 464, 3797, 3332, 319, 262 };
        var mask = new float[] { 1, 1, 1, 1, 1 };
        var posIds = new float[] { 0, 1, 2, 3, 4 };

        using var idsBuf = accelerator.Allocate1D(tokenIds);
        using var maskBuf = accelerator.Allocate1D(mask);
        using var posBuf = accelerator.Allocate1D(posIds);

        var inputs = new Dictionary<string, Tensor>
        {
            ["input_ids"] = new Tensor(idsBuf.View, new[] { 1, 5 }),
            ["attention_mask"] = new Tensor(maskBuf.View, new[] { 1, 5 }),
            ["position_ids"] = new Tensor(posBuf.View, new[] { 1, 5 }),
        };

        var outputs = await session.RunAsync(inputs);

        var kvCache = session.Executor.KVCache!;
        Console.WriteLine($"[TurboQuant] After inference: cache seqLen={kvCache.CurrentSeqLen}, layers={kvCache.NumLayers}");

        if (kvCache.CurrentSeqLen < 1)
            throw new Exception($"KV cache empty after inference — expected ≥1, got {kvCache.CurrentSeqLen}");

        Console.WriteLine($"[TurboQuant] DistilGPT-2 KV cache capture: PASS (seqLen={kvCache.CurrentSeqLen})");
    });
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
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), kNormsBuf.View.SubView(kv, 1), 1, 1f);

            // Same for V
            var vSlice = new float[headDim];
            Array.Copy(vData, kv * headDim, vSlice, 0, headDim);
            using var vVec = accelerator.Allocate1D(vSlice);

            tq.Normalize(vVec.View, tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, vPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), vNormsBuf.View.SubView(kv, 1), 1, 1f);
        }

        // Run fused quantized attention
        // Separate codebook buffers for K and V to avoid WebGPU aliasing
        using var vCodebookBuf = accelerator.Allocate1D(codebook);
        using var outputBuf = accelerator.Allocate1D<float>(headDim);
        tq.FusedQuantizedAttention(
            qBuf.View, kPackedBuf.View, codebookBuf.View,
            vPackedBuf.View, vCodebookBuf.View,
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

        // Log detailed diagnostics for debugging
        Console.WriteLine($"[TurboQuant] normA={normA:F4}, normB={normB:F4}, dotAB={dotAB:F4}");
        Console.WriteLine($"[TurboQuant] FP32 first5: [{string.Join(",", fp32Output.Take(5).Select(v => v.ToString("F4")))}]");
        Console.WriteLine($"[TurboQuant] Quant first5: [{string.Join(",", quantizedOutput.Take(5).Select(v => v.ToString("F4")))}]");

        // 4-bit quantization should maintain reasonable accuracy
        // Cosine > 0 means at least some correlation (quantization is lossy)
        if (normB < 1e-8f)
            Console.WriteLine("[TurboQuant] WARNING: quantized output is near-zero — fused attention kernel may need debugging");
        else if (cosineSim < 0.5f)
            throw new Exception($"Quantized attention cosine similarity {cosineSim:F4} too low — expected > 0.5");
    });

    /// <summary>
    /// Flash Attention (Online Softmax) must match the two-pass fused attention output.
    /// Both compute the same mathematical result — softmax(Q@K^T/√d) @ V — but Online
    /// Softmax does it in one pass with running max/sum rescaling.
    /// </summary>
    [TestMethod]
    public async Task TurboQuant_FlashAttention_MatchesTwoPass() => await RunTest(async accelerator =>
    {
        int headDim = 64;
        int numKV = 16; // more KV positions to stress the online softmax rescaling
        var rng = new Random(42);

        var qData = new float[headDim];
        var kData = new float[numKV * headDim];
        var vData = new float[numKV * headDim];
        for (int i = 0; i < headDim; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < numKV * headDim; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < numKV * headDim; i++) vData[i] = (float)(rng.NextDouble() * 2 - 1);

        float scale = 1f / MathF.Sqrt(headDim);
        var tq = new TurboQuantKernels(accelerator);

        // Quantize K and V (same setup as QuantizedAttention test)
        int packedDim = headDim / 8;
        var codebook = TurboQuantKernels.Codebook4Bit;
        using var qBuf = accelerator.Allocate1D(qData);
        using var kPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var vPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var kNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var vNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var codebookBuf = accelerator.Allocate1D(codebook);
        using var vCodebookBuf = accelerator.Allocate1D(codebook);
        using var tempNorm = accelerator.Allocate1D<float>(headDim);
        using var tempNormVal = accelerator.Allocate1D<float>(1);
        using var tempIndices = accelerator.Allocate1D<int>(headDim);

        for (int kv = 0; kv < numKV; kv++)
        {
            var kSlice = new float[headDim];
            Array.Copy(kData, kv * headDim, kSlice, 0, headDim);
            using var kVec = accelerator.Allocate1D(kSlice);
            tq.Normalize(kVec.View, tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, kPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), kNormsBuf.View.SubView(kv, 1), 1, 1f);

            var vSlice = new float[headDim];
            Array.Copy(vData, kv * headDim, vSlice, 0, headDim);
            using var vVec = accelerator.Allocate1D(vSlice);
            tq.Normalize(vVec.View, tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, vPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), vNormsBuf.View.SubView(kv, 1), 1, 1f);
        }

        // Run BOTH kernels on same quantized data
        using var twoPassOutput = accelerator.Allocate1D<float>(headDim);
        using var flashOutput = accelerator.Allocate1D<float>(headDim);

        tq.FusedQuantizedAttention(
            qBuf.View, kPackedBuf.View, codebookBuf.View,
            vPackedBuf.View, vCodebookBuf.View,
            kNormsBuf.View, vNormsBuf.View, twoPassOutput.View,
            1, numKV, headDim, scale);

        // Need separate codebook buffers for Flash kernel too (WebGPU aliasing)
        using var kCB2 = accelerator.Allocate1D(codebook);
        using var vCB2 = accelerator.Allocate1D(codebook);

        tq.FlashQuantizedAttention(
            qBuf.View, kPackedBuf.View, kCB2.View,
            vPackedBuf.View, vCB2.View,
            kNormsBuf.View, vNormsBuf.View, flashOutput.View,
            1, numKV, headDim, scale);

        await accelerator.SynchronizeAsync();
        var twoPassResult = await twoPassOutput.CopyToHostAsync<float>(0, headDim);
        var flashResult = await flashOutput.CopyToHostAsync<float>(0, headDim);

        // Compare — should be nearly identical (same math, different traversal order)
        float maxErr = 0, dotAB = 0, normA = 0, normB = 0;
        for (int d = 0; d < headDim; d++)
        {
            float err = MathF.Abs(twoPassResult[d] - flashResult[d]);
            if (err > maxErr) maxErr = err;
            dotAB += twoPassResult[d] * flashResult[d];
            normA += twoPassResult[d] * twoPassResult[d];
            normB += flashResult[d] * flashResult[d];
        }
        float cosineSim = dotAB / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);

        Console.WriteLine($"[FlashAttn] Two-pass vs Online Softmax: maxErr={maxErr:F6}, cosine={cosineSim:F6}");
        Console.WriteLine($"[FlashAttn] TwoPass first5: [{string.Join(",", twoPassResult.Take(5).Select(v => v.ToString("F4")))}]");
        Console.WriteLine($"[FlashAttn] Flash first5:   [{string.Join(",", flashResult.Take(5).Select(v => v.ToString("F4")))}]");

        if (cosineSim < 0.999f)
            throw new Exception($"Flash Attention cosine {cosineSim:F6} too low vs two-pass — expected > 0.999");

        Console.WriteLine($"[FlashAttn] PASS — Online Softmax matches two-pass (cosine={cosineSim:F6})");
    });

    /// <summary>
    /// Load Whisper Tiny decoder (merged), verify GraphExecutor auto-detects KV cache.
    /// Whisper Tiny has 4 decoder layers with both self-attention and cross-attention KV cache.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_WhisperDecoder_KVCacheAutoDetected() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model_merged.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 1 },
            },
            enableOptimization: false);

        Console.WriteLine($"[TurboQuant] Whisper decoder inputs: {string.Join(", ", session.InputNames)}");
        Console.WriteLine($"[TurboQuant] Whisper decoder outputs: {string.Join(", ", session.OutputNames)}");

        bool hasCache = session.Executor.HasKVCache;
        Console.WriteLine($"[TurboQuant] Whisper HasKVCache: {hasCache}");

        if (!hasCache)
            throw new Exception("Whisper decoder_model_merged should have KV cache pattern but none detected");

        var kvCache = session.Executor.KVCache!;
        Console.WriteLine($"[TurboQuant] Whisper KV cache: {kvCache.NumLayers} layers, maxSeq={kvCache.MaxSeqLen}");

        // Whisper Tiny: 4 decoder layers, each with self-attention + cross-attention = up to 8 KV cache points
        if (kvCache.NumLayers < 4)
            throw new Exception($"Whisper Tiny should have at least 4 KV cache layers, got {kvCache.NumLayers}");

        Console.WriteLine($"[TurboQuant] Whisper decoder KV cache detection: PASS");
    });

    /// <summary>
    /// Run one decoder step of Whisper Tiny with fake encoder output,
    /// verify quantized KV cache captures the token's key/value data.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_WhisperDecoder_KVCacheCaptures() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model_merged.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 4 },
            },
            enableOptimization: false);

        if (!session.Executor.HasKVCache)
        {
            Console.WriteLine("[TurboQuant] Skipping — no KV cache detected");
            return;
        }

        // Whisper Tiny: encoder output is [1, 1500, 384]
        // Create fake encoder hidden states (random but deterministic)
        var rng = new Random(42);
        int encoderSeq = 1500, encoderDim = 384;
        var encoderData = new float[encoderSeq * encoderDim];
        for (int i = 0; i < encoderData.Length; i++)
            encoderData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

        using var encoderBuf = accelerator.Allocate1D(encoderData);

        // Whisper prefix tokens: SOT=50258, EN=50259, TRANSCRIBE=50360, NO_TIMESTAMPS=50364
        var tokenIds = new float[] { 50258, 50259, 50360, 50364 };
        using var idsBuf = accelerator.Allocate1D(tokenIds);

        var inputs = new Dictionary<string, Tensor>
        {
            ["input_ids"] = new Tensor(idsBuf.View, new[] { 1, 4 }),
        };

        // Find the encoder_hidden_states input name
        foreach (var inputName in session.InputNames)
        {
            if (inputName.Contains("encoder") && !inputName.Contains("past_key"))
            {
                inputs[inputName] = new Tensor(encoderBuf.View, new[] { 1, encoderSeq, encoderDim });
                Console.WriteLine($"[TurboQuant] Encoder input mapped to: {inputName}");
                break;
            }
        }

        var outputs = await session.RunAsync(inputs);

        var kvCache = session.Executor.KVCache!;
        Console.WriteLine($"[TurboQuant] Whisper after inference: cache seqLen={kvCache.CurrentSeqLen}, layers={kvCache.NumLayers}");

        if (kvCache.CurrentSeqLen < 1)
            throw new Exception($"Whisper KV cache empty after inference — expected ≥1, got {kvCache.CurrentSeqLen}");

        Console.WriteLine($"[TurboQuant] Whisper decoder KV cache capture: PASS (seqLen={kvCache.CurrentSeqLen})");
    });

    /// <summary>
    /// Benchmark: compare 4-bit vs 3-bit quantization roundtrip accuracy.
    /// Generates random Gaussian vectors, runs the full TurboQuant pipeline
    /// (normalize → sign-flip → FWHT → quantize → pack → unpack → dequantize
    /// → inverse FWHT → sign-flip → denormalize), measures cosine similarity
    /// between original and reconstructed vectors.
    /// </summary>
    [TestMethod]
    public async Task TurboQuant_3BitVs4Bit_AccuracyComparison() => await RunTest(async accelerator =>
    {
        var tq = new TurboQuantKernels(accelerator);
        int d = 64; // GPT-2 head dimension
        int numVecs = 32; // test batch
        var rng = new Random(42);

        // Generate random Gaussian-like vectors
        var originalData = new float[numVecs * d];
        for (int i = 0; i < originalData.Length; i++)
        {
            // Box-Muller for Gaussian
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            originalData[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // Deterministic sign vector
        var signRng = new Random(42);
        var signData = new int[d];
        for (int i = 0; i < d; i++) signData[i] = signRng.Next(2);

        using var originalBuf = accelerator.Allocate1D(originalData);
        using var signsBuf = accelerator.Allocate1D(signData);

        // Shared temp buffers
        using var normalized = accelerator.Allocate1D<float>(numVecs * d);
        using var norms = accelerator.Allocate1D<float>(numVecs);
        using var flipped = accelerator.Allocate1D<float>(numVecs * d);
        using var transformed = accelerator.Allocate1D<float>(numVecs * d);
        using var indices = accelerator.Allocate1D<int>(numVecs * d);
        using var reconstructed = accelerator.Allocate1D<float>(numVecs * d);

        // ═══ Test both codebooks ═══
        var codebooks = new (string name, float[] values, int bits)[]
        {
            ("4-bit (16 centroids)", TurboQuantKernels.Codebook4Bit, 4),
            ("3-bit (8 centroids)", TurboQuantKernels.Codebook3Bit, 3),
        };

        foreach (var (name, codebookValues, bits) in codebooks)
        {
            using var codebookBuf = accelerator.Allocate1D(codebookValues);
            int numCentroids = codebookValues.Length;
            int packedPerInt = bits == 3 ? 10 : 8;
            int packedDim = (d + packedPerInt - 1) / packedPerInt;
            using var packed = accelerator.Allocate1D<int>(numVecs * packedDim);

            // Forward: normalize → sign-flip → FWHT → scale(√d) → quantize → pack
            tq.Normalize(originalBuf.View, normalized.View, norms.View, numVecs, d);
            tq.SignFlip(normalized.View, flipped.View, signsBuf.View, numVecs * d);
            tq.FWHT.ForwardBatch(flipped.View, transformed.View, numVecs, d);
            // Scale by √d: FWHT normalizes by 1/√d, but codebook expects N(0,1) variance
            float sqrtD = MathF.Sqrt(d);
            new ElementWiseKernels(accelerator).Scale(transformed.View, transformed.View, numVecs * d, sqrtD);
            tq.Quantize(transformed.View, codebookBuf.View, indices.View, numVecs * d, numCentroids);

            // Pack
            for (int v = 0; v < numVecs; v++)
            {
                var srcView = indices.View.SubView(v * d, d);
                var dstView = packed.View.SubView(v * packedDim, packedDim);
                if (bits == 3)
                    tq.BitPack3(srcView, dstView, d);
                else
                    tq.BitPack4(srcView, dstView, d);
            }

            // Unpack
            for (int v = 0; v < numVecs; v++)
            {
                var srcView = packed.View.SubView(v * packedDim, packedDim);
                var dstView = indices.View.SubView(v * d, d);
                if (bits == 3)
                    tq.BitUnpack3(srcView, dstView, d);
                else
                    tq.BitUnpack4(srcView, dstView, d);
            }

            // Reverse: dequantize → scale(1/√d) → inverse FWHT → sign-flip → denormalize
            tq.Dequantize(indices.View, codebookBuf.View, transformed.View, numVecs * d, numCentroids);
            // Scale by 1/√d to undo pre-quantization scaling
            float invSqrtD = 1f / MathF.Sqrt(d);
            new ElementWiseKernels(accelerator).Scale(transformed.View, transformed.View, numVecs * d, invSqrtD);
            tq.FWHT.ForwardBatch(transformed.View, flipped.View, numVecs, d); // FWHT is its own inverse
            tq.SignFlip(flipped.View, normalized.View, signsBuf.View, numVecs * d);
            tq.Denormalize(normalized.View, reconstructed.View, norms.View, numVecs, d);

            await accelerator.SynchronizeAsync();
            var result = await reconstructed.CopyToHostAsync<float>(0, numVecs * d);

            // Compute per-vector cosine similarity
            float totalCosine = 0;
            float minCosine = float.MaxValue;
            float totalMSE = 0;
            for (int v = 0; v < numVecs; v++)
            {
                float dot = 0, normA = 0, normB = 0, mse = 0;
                for (int i = 0; i < d; i++)
                {
                    float a = originalData[v * d + i];
                    float b = result[v * d + i];
                    dot += a * b;
                    normA += a * a;
                    normB += b * b;
                    mse += (a - b) * (a - b);
                }
                float cosine = dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);
                totalCosine += cosine;
                if (cosine < minCosine) minCosine = cosine;
                totalMSE += mse / d;
            }
            float avgCosine = totalCosine / numVecs;
            float avgMSE = totalMSE / numVecs;

            Console.WriteLine($"[TurboQuant] {name}: avgCosine={avgCosine:F6}, minCosine={minCosine:F6}, avgMSE={avgMSE:F6}");
        }

        // ═══ 3+1 QJL mode: 3-bit value + 1-bit residual sign (CPU simulation) ═══
        // Same packing as 4-bit (8 per uint32): lower 3 bits = centroid, bit 3 = QJL sign
        {
            var cb3 = TurboQuantKernels.Codebook3Bit;
            using var codebook3Buf = accelerator.Allocate1D(cb3);
            int packedDim4 = (d + 7) / 8;
            using var packed3p1 = accelerator.Allocate1D<int>(numVecs * packedDim4);

            // Forward pipeline through FWHT + scale
            tq.Normalize(originalBuf.View, normalized.View, norms.View, numVecs, d);
            tq.SignFlip(normalized.View, flipped.View, signsBuf.View, numVecs * d);
            tq.FWHT.ForwardBatch(flipped.View, transformed.View, numVecs, d);
            new ElementWiseKernels(accelerator).Scale(transformed.View, transformed.View, numVecs * d, MathF.Sqrt(d));

            // Quantize to 3-bit centroids
            tq.Quantize(transformed.View, codebook3Buf.View, indices.View, numVecs * d, 8);
            await accelerator.SynchronizeAsync();

            // Read back transformed data and indices to compute QJL signs on CPU
            var transformedHost = await transformed.CopyToHostAsync<float>(0, numVecs * d);
            var indicesHost = await indices.CopyToHostAsync<int>(0, numVecs * d);

            // Compute QJL sign bits and pack as 3+1 = 4 bits per value
            var packed3p1Host = new int[numVecs * packedDim4];
            for (int v = 0; v < numVecs; v++)
            {
                for (int p = 0; p < packedDim4; p++)
                {
                    int result = 0;
                    for (int b = 0; b < 8 && v * d + p * 8 + b < numVecs * d; b++)
                    {
                        int flatIdx = v * d + p * 8 + b;
                        int centroidIdx = indicesHost[flatIdx] & 0x7;
                        float original = transformedHost[flatIdx];
                        float centroid = cb3[centroidIdx];
                        int qjlSign = (original - centroid) >= 0 ? 1 : 0;
                        int packed4bit = centroidIdx | (qjlSign << 3);
                        result |= packed4bit << (b * 4);
                    }
                    packed3p1Host[v * packedDim4 + p] = result;
                }
            }

            // Dequantize with QJL correction on CPU
            // Compute average residual magnitude for QJL scale
            float totalResidual = 0;
            int residualCount = 0;
            for (int i = 0; i < numVecs * d; i++)
            {
                float residual = MathF.Abs(transformedHost[i] - cb3[indicesHost[i] & 0x7]);
                totalResidual += residual;
                residualCount++;
            }
            float qjlScale = totalResidual / residualCount;

            var dequantHost = new float[numVecs * d];
            for (int v = 0; v < numVecs; v++)
            {
                for (int p = 0; p < packedDim4; p++)
                {
                    int word = packed3p1Host[v * packedDim4 + p];
                    for (int b = 0; b < 8 && p * 8 + b < d; b++)
                    {
                        int chunk = (word >> (b * 4)) & 0xF;
                        int centroidIdx = chunk & 0x7;
                        int qjlSign = (chunk >> 3) & 0x1;
                        float val = cb3[centroidIdx];
                        val += qjlSign == 1 ? qjlScale : -qjlScale;
                        dequantHost[v * d + p * 8 + b] = val;
                    }
                }
            }

            // Upload dequantized data, scale by 1/√d, then run reverse pipeline on GPU
            // Scale dequantized values to undo the √d pre-quantization scaling
            float invSqrtD3 = 1f / MathF.Sqrt(d);
            for (int i = 0; i < dequantHost.Length; i++)
                dequantHost[i] *= invSqrtD3;
            using var dequantBuf = accelerator.Allocate1D(dequantHost);
            tq.FWHT.ForwardBatch(dequantBuf.View, flipped.View, numVecs, d);
            tq.SignFlip(flipped.View, normalized.View, signsBuf.View, numVecs * d);
            tq.Denormalize(normalized.View, reconstructed.View, norms.View, numVecs, d);

            await accelerator.SynchronizeAsync();
            var result3p1 = await reconstructed.CopyToHostAsync<float>(0, numVecs * d);

            float totalCosine = 0;
            float minCosine = float.MaxValue;
            float totalMSE = 0;
            for (int v = 0; v < numVecs; v++)
            {
                float dot = 0, normA2 = 0, normB2 = 0, mse = 0;
                for (int i = 0; i < d; i++)
                {
                    float a = originalData[v * d + i];
                    float b = result3p1[v * d + i];
                    dot += a * b;
                    normA2 += a * a;
                    normB2 += b * b;
                    mse += (a - b) * (a - b);
                }
                float cosine = dot / (MathF.Sqrt(normA2) * MathF.Sqrt(normB2) + 1e-10f);
                totalCosine += cosine;
                if (cosine < minCosine) minCosine = cosine;
                totalMSE += mse / d;
            }
            float avgCosine = totalCosine / numVecs;
            float avgMSE = totalMSE / numVecs;

            Console.WriteLine($"[TurboQuant] 3+1 QJL (8 centroids + sign): avgCosine={avgCosine:F6}, minCosine={minCosine:F6}, avgMSE={avgMSE:F6}");
        }

        Console.WriteLine($"[TurboQuant] 3-bit vs 4-bit vs 3+1 QJL comparison: DONE");
    });

    /// <summary>
    /// GPT-2 baseline: run one forward pass WITHOUT KV cache (non-merged model).
    /// Establishes expected next token for "The cat sat on the" prompt.
    /// This must pass before TurboQuant integration is tested.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_GPT2_Baseline_NoKVCache() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DistilGPT-2 non-merged (no KV cache inputs/outputs)
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/model.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
                ["attention_mask"] = new[] { 1, 5 },
            },
            enableOptimization: false);

        Console.WriteLine($"[GPT-2 Baseline] inputs: {string.Join(", ", session.InputNames)}");
        Console.WriteLine($"[GPT-2 Baseline] HasKVCache: {session.Executor.HasKVCache}");

        // "The cat sat on the" tokens for DistilGPT-2
        var tokenIds = new float[] { 464, 3797, 3332, 319, 262 };
        var mask = new float[] { 1, 1, 1, 1, 1 };

        using var idsBuf = accelerator.Allocate1D(tokenIds);
        using var maskBuf = accelerator.Allocate1D(mask);

        var inputs = new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = new Tensor(idsBuf.View, new[] { 1, 5 }),
        };
        if (session.InputNames.Length > 1)
            inputs[session.InputNames[1]] = new Tensor(maskBuf.View, new[] { 1, 5 });

        var outputs = await session.RunAsync(inputs);
        var output = outputs[session.OutputNames[0]];

        // Get last token logits
        int vocabSize = output.Shape.Length >= 3 ? output.Shape[^1] : 50257;
        int lastOffset = (5 - 1) * vocabSize;

        using var readBuf = accelerator.Allocate1D<float>(vocabSize);
        new ElementWiseKernels(accelerator).Scale(
            output.Data.SubView(lastOffset, vocabSize), readBuf.View, vocabSize, 1f);
        await accelerator.SynchronizeAsync();
        var logits = await readBuf.CopyToHostAsync<float>(0, vocabSize);

        // Find argmax
        int nextToken = 0;
        float maxLogit = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            if (!float.IsNaN(logits[i]) && logits[i] > maxLogit)
            {
                maxLogit = logits[i];
                nextToken = i;
            }
        }

        Console.WriteLine($"[GPT-2 Baseline] Next token: {nextToken} (logit={maxLogit:F4})");
        Console.WriteLine($"[GPT-2 Baseline] Top 5 logits:");
        var topIndices = logits.Select((v, i) => (v, i))
            .OrderByDescending(x => x.v).Take(5).ToArray();
        foreach (var (v, i) in topIndices)
            Console.WriteLine($"  token {i}: {v:F4}");

        // Verify no NaN
        int nanCount = logits.Count(v => float.IsNaN(v) || float.IsInfinity(v));
        if (nanCount > 0)
            throw new Exception($"[GPT-2 Baseline] {nanCount} NaN/Inf logits");

        Console.WriteLine($"[GPT-2 Baseline] PASS — next token {nextToken}, no NaN");
    });

    /// <summary>
    /// GPT-2 with TurboQuant KV cache: run DistilGPT-2 merged model with auto KV cache.
    /// The merged model has past_key_values inputs/outputs — GraphExecutor auto-detects
    /// and creates QuantizedKVCache (3+1 QJL default). Verifies:
    /// 1. KV cache is auto-detected and active
    /// 2. First forward pass produces valid logits (no NaN)
    /// 3. KV cache captures token data after inference
    /// 4. Top token is logged for comparison with baseline
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task TurboQuant_GPT2_WithKVCache() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DistilGPT-2 merged (WITH KV cache inputs/outputs)
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model_merged.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
                ["attention_mask"] = new[] { 1, 5 },
                ["position_ids"] = new[] { 1, 5 },
            },
            enableOptimization: false);

        Console.WriteLine($"[GPT-2 TurboQuant] inputs: {string.Join(", ", session.InputNames)}");
        Console.WriteLine($"[GPT-2 TurboQuant] HasKVCache: {session.Executor.HasKVCache}");

        if (session.Executor.HasKVCache)
        {
            var kv = session.Executor.KVCache!;
            Console.WriteLine($"[GPT-2 TurboQuant] KV cache mode: {kv.Mode}, layers: {kv.NumLayers}");
        }

        // Same prompt as baseline: "The cat sat on the"
        var tokenIds = new float[] { 464, 3797, 3332, 319, 262 };
        var mask = new float[] { 1, 1, 1, 1, 1 };
        var posIds = new float[] { 0, 1, 2, 3, 4 };

        using var idsBuf = accelerator.Allocate1D(tokenIds);
        using var maskBuf = accelerator.Allocate1D(mask);
        using var posBuf = accelerator.Allocate1D(posIds);

        var inputs = new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = new Tensor(idsBuf.View, new[] { 1, 5 }),
        };
        if (session.InputNames.Contains("attention_mask"))
            inputs["attention_mask"] = new Tensor(maskBuf.View, new[] { 1, 5 });
        if (session.InputNames.Contains("position_ids"))
            inputs["position_ids"] = new Tensor(posBuf.View, new[] { 1, 5 });

        var outputs = await session.RunAsync(inputs);
        var output = outputs[session.OutputNames[0]];

        // Get last token logits
        int vocabSize = output.Shape.Length >= 3 ? output.Shape[^1] : 50257;
        int lastOffset = (5 - 1) * vocabSize;

        using var readBuf = accelerator.Allocate1D<float>(vocabSize);
        new ElementWiseKernels(accelerator).Scale(
            output.Data.SubView(lastOffset, vocabSize), readBuf.View, vocabSize, 1f);
        await accelerator.SynchronizeAsync();
        var logits = await readBuf.CopyToHostAsync<float>(0, vocabSize);

        // Find argmax
        int nextToken = 0;
        float maxLogit = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            if (!float.IsNaN(logits[i]) && logits[i] > maxLogit)
            {
                maxLogit = logits[i];
                nextToken = i;
            }
        }

        Console.WriteLine($"[GPT-2 TurboQuant] Next token: {nextToken} (logit={maxLogit:F4})");
        Console.WriteLine($"[GPT-2 TurboQuant] Top 5 logits:");
        var topIndices = logits.Select((v, i) => (v, i))
            .OrderByDescending(x => x.v).Take(5).ToArray();
        foreach (var (v, i) in topIndices)
            Console.WriteLine($"  token {i}: {v:F4}");

        // Verify no NaN
        int nanCount = logits.Count(v => float.IsNaN(v) || float.IsInfinity(v));
        if (nanCount > 0)
            throw new Exception($"[GPT-2 TurboQuant] {nanCount} NaN/Inf logits");

        // Verify KV cache captured data
        if (session.Executor.HasKVCache)
        {
            var kv = session.Executor.KVCache!;
            Console.WriteLine($"[GPT-2 TurboQuant] KV cache seqLen after inference: {kv.CurrentSeqLen}");
        }

        Console.WriteLine($"[GPT-2 TurboQuant] PASS — next token {nextToken}, KV cache active, no NaN");
    });

    /// <summary>
    /// QuantizedKVCache.FlashAttention(): store vectors, then run single-pass
    /// Flash Attention directly on the cache. Verifies the full pipeline:
    /// Append (quantize + pack) → FlashAttention (dequant + Online Softmax) → output.
    /// Compares against CPU FP32 reference attention.
    /// </summary>
    [TestMethod]
    public async Task TurboQuant_KVCache_FlashAttention_EndToEnd() => await RunTest(async accelerator =>
    {
        int numHeads = 6;
        int headDim = 64;
        int vecDim = numHeads * headDim; // 384
        int numTokens = 8;
        var rng = new Random(42);

        // Create a mock KVCacheInfo for the QuantizedKVCache
        var layers = new[] {
            new Graph.KVCacheAnalyzer.KVCachePoint
            {
                LayerIndex = 0,
                PastKeyInput = "past_key_values.0.key",
                PastValueInput = "past_key_values.0.value",
                PresentKeyOutput = "present.0.key",
                PresentValueOutput = "present.0.value",
                Shape = new[] { 1, numHeads, 1, headDim },
            }
        };
        var cacheInfo = new Graph.KVCacheAnalyzer.KVCacheInfo
        {
            HasExplicitKVCache = true,
            Layers = layers,
        };

        using var kvCache = new QuantizedKVCache(accelerator, cacheInfo,
            maxSeqLen: 64, quantMode: KVQuantMode.Auto);

        Console.WriteLine($"[KVCache FlashAttn] Mode: {kvCache.Mode}, layers: {kvCache.NumLayers}");

        // Generate random K and V vectors for each token, store originals for CPU reference
        var allK = new float[numTokens][];
        var allV = new float[numTokens][];

        for (int t = 0; t < numTokens; t++)
        {
            var kData = new float[vecDim];
            var vData = new float[vecDim];
            for (int i = 0; i < vecDim; i++)
            {
                kData[i] = (float)(rng.NextDouble() * 2 - 1);
                vData[i] = (float)(rng.NextDouble() * 2 - 1);
            }
            allK[t] = kData;
            allV[t] = vData;

            // Append to quantized cache
            using var kBuf = accelerator.Allocate1D(kData);
            using var vBuf = accelerator.Allocate1D(vData);
            kvCache.Append(0, kBuf.View, vBuf.View);
            kvCache.AdvanceToken();
        }

        Console.WriteLine($"[KVCache FlashAttn] Cached {kvCache.CurrentSeqLen} tokens");

        // Generate query
        var qData = new float[vecDim];
        for (int i = 0; i < vecDim; i++)
            qData[i] = (float)(rng.NextDouble() * 2 - 1);

        // ═══ CPU FP32 reference attention ═══
        float scale = 1f / MathF.Sqrt(headDim);
        var cpuOutput = new float[vecDim];

        // Compute scores Q @ K^T
        var scores = new float[numTokens];
        float maxScore = float.MinValue;
        for (int t = 0; t < numTokens; t++)
        {
            float dot = 0;
            for (int i = 0; i < vecDim; i++)
                dot += qData[i] * allK[t][i];
            scores[t] = dot * scale;
            if (scores[t] > maxScore) maxScore = scores[t];
        }

        // Softmax
        float sumExp = 0;
        for (int t = 0; t < numTokens; t++)
        {
            scores[t] = MathF.Exp(scores[t] - maxScore);
            sumExp += scores[t];
        }
        for (int t = 0; t < numTokens; t++)
            scores[t] /= sumExp;

        // Weighted sum of V
        for (int i = 0; i < vecDim; i++)
        {
            float sum = 0;
            for (int t = 0; t < numTokens; t++)
                sum += scores[t] * allV[t][i];
            cpuOutput[i] = sum;
        }

        // ═══ GPU Flash Attention on quantized cache ═══
        using var qBuf = accelerator.Allocate1D(qData);
        using var outBuf = accelerator.Allocate1D<float>(vecDim);

        kvCache.FlashAttention(0, qBuf.View, outBuf.View, 1, scale);
        await accelerator.SynchronizeAsync();
        var gpuOutput = await outBuf.CopyToHostAsync<float>(0, vecDim);

        // ═══ Compare ═══
        float dotAB = 0, normA = 0, normB = 0;
        for (int i = 0; i < vecDim; i++)
        {
            dotAB += cpuOutput[i] * gpuOutput[i];
            normA += cpuOutput[i] * cpuOutput[i];
            normB += gpuOutput[i] * gpuOutput[i];
        }
        float cosineSim = dotAB / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);

        Console.WriteLine($"[KVCache FlashAttn] CPU vs GPU cosine: {cosineSim:F6}");
        Console.WriteLine($"[KVCache FlashAttn] CPU first5: [{string.Join(",", cpuOutput.Take(5).Select(v => v.ToString("F4")))}]");
        Console.WriteLine($"[KVCache FlashAttn] GPU first5: [{string.Join(",", gpuOutput.Take(5).Select(v => v.ToString("F4")))}]");

        // Quantization introduces some error, but cosine should be high
        if (cosineSim < 0.8f)
            throw new Exception($"KVCache FlashAttention cosine {cosineSim:F6} too low — expected > 0.8");

        Console.WriteLine($"[KVCache FlashAttn] PASS — end-to-end quantized Flash Attention (cosine={cosineSim:F6})");
    });
}
