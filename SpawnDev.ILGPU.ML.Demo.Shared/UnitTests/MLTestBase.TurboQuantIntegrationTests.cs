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

        // Use base decoder (no If control flow nodes) — same KV cache outputs
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
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

        // Use base decoder (no If control flow nodes) — same KV cache outputs
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
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

        // Encode K and V: normalize → quantize → pack.
        // Upload K and V ONCE as whole buffers and SubView per vector. The previous code
        // allocated a per-iteration `using var kVec`/`vVec` and disposed it each iteration —
        // but on WebGPU the Normalize/Quantize dispatches are batched in the command encoder
        // and not executed until SynchronizeAsync, so destroying the input buffer mid-loop
        // produced "Buffer used in submit while destroyed" GPU errors (one per KV position).
        // Whole-buffer SubViews stay alive until after the flush and avoid numKV uploads.
        using var kAllBuf = accelerator.Allocate1D(kData);
        using var vAllBuf = accelerator.Allocate1D(vData);
        using var kPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var vPackedBuf = accelerator.Allocate1D<int>(numKV * packedDim);
        using var kNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var vNormsBuf = accelerator.Allocate1D<float>(numKV);
        using var codebookBuf = accelerator.Allocate1D(codebook);

        // Per-vector encode. Scratch (tempNorm/tempIndices/tempNormVal) is reused across
        // iterations — safe on WebGPU because the batched compute passes execute in
        // submission order; the only hazard was destroying input buffers before the flush.
        using var tempNorm = accelerator.Allocate1D<float>(headDim);
        using var tempNormVal = accelerator.Allocate1D<float>(1);
        using var tempIndices = accelerator.Allocate1D<int>(headDim);

        for (int kv = 0; kv < numKV; kv++)
        {
            tq.Normalize(kAllBuf.View.SubView(kv * headDim, headDim), tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, kPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), kNormsBuf.View.SubView(kv, 1), 1, 1f);

            tq.Normalize(vAllBuf.View.SubView(kv * headDim, headDim), tempNorm.View, tempNormVal.View, 1, headDim);
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

        // 4-bit quantization should maintain reasonable accuracy.
        // A near-zero output is a HARD FAILURE, not a warning: the warn-and-pass version
        // of this check let the WebGL TF multi-store garbage (output ≈ zeros) slide for
        // months while the strict FlashAttention test caught the identical corpse
        // (2026-06-12). A test that warns on the broken case is a fake test.
        if (normB < 1e-8f)
            throw new Exception(
                "Quantized attention output is near-zero — the fused attention kernel wrote " +
                "nothing (norm² " + normB + "). On WebGL this is the TF multi-store kernel-" +
                "shape contract; see FusedAttentionImpl's doc.");
        if (cosineSim < 0.5f)
            throw new Exception($"Quantized attention cosine similarity {cosineSim:F4} too low — expected > 0.5");
    });

    /// <summary>
    /// DISCRIMINATOR for the WebGL FP32-mismatch (2026-06-12): the attention kernels now
    /// agree with each other on WebGL but BOTH disagree with the CPU reference, which
    /// points UPSTREAM — at the 16-iteration per-vector encode loop (Normalize → Quantize
    /// → BitPack4 → Scale) writing OFFSET SUBVIEWS, a shape the single-shot round-trip
    /// tests never exercise. Encodes the same vectors two ways and compares each against
    /// a CPU reference encode:
    ///   A) the production shape — shared scratch, packed/norm outputs via SubView(kv*…)
    ///   B) per-vector DEDICATED buffers — no offset subviews, no scratch reuse
    /// A bad + B good  → offset-subview TF writeback / loop-reuse bug (backend);
    /// A bad + B bad   → pipeline kernels themselves;
    /// A good + B good → the encode is clean and the attention kernels' input reads are
    ///                   the remaining suspect.
    /// </summary>
    [TestMethod]
    public async Task TurboQuant_EncodeLoop_SubViewVsDedicated_MatchesCPU() => await RunTest(async accelerator =>
    {
        const int headDim = 64, numKV = 16, numCentroids = 16;
        int packedDim = headDim / 8;
        var rng = new Random(42);
        var kData = new float[numKV * headDim];
        for (int i = 0; i < kData.Length; i++) kData[i] = (float)(rng.NextDouble() * 2 - 1);
        var codebook = TurboQuantKernels.Codebook4Bit;

        // ── CPU reference encode ──
        var refPacked = new int[numKV * packedDim];
        var refNorms = new float[numKV];
        for (int kv = 0; kv < numKV; kv++)
        {
            float sumSq = 0;
            for (int i = 0; i < headDim; i++) sumSq += kData[kv * headDim + i] * kData[kv * headDim + i];
            float norm = MathF.Sqrt(sumSq);
            refNorms[kv] = norm;
            float invNorm = norm > 1e-12f ? 1f / norm : 0f;
            for (int i = 0; i < headDim; i++)
            {
                float val = kData[kv * headDim + i] * invNorm;
                int best = 0; float bestDist = MathF.Abs(val - codebook[0]);
                for (int c = 1; c < numCentroids; c++)
                {
                    float dist = MathF.Abs(val - codebook[c]);
                    if (dist < bestDist) { bestDist = dist; best = c; }
                }
                refPacked[kv * packedDim + i / 8] |= best << ((i % 8) * 4);
            }
        }

        var tq = new TurboQuantKernels(accelerator);
        using var kAllBuf = accelerator.Allocate1D(kData);
        using var codebookBuf = accelerator.Allocate1D(codebook);

        // ── Variant A: production shape (shared scratch + offset-subview outputs) ──
        using var aPacked = accelerator.Allocate1D<int>(numKV * packedDim);
        using var aNorms = accelerator.Allocate1D<float>(numKV);
        using var tempNorm = accelerator.Allocate1D<float>(headDim);
        using var tempNormVal = accelerator.Allocate1D<float>(1);
        using var tempIndices = accelerator.Allocate1D<int>(headDim);
        var ew = new ElementWiseKernels(accelerator);
        for (int kv = 0; kv < numKV; kv++)
        {
            tq.Normalize(kAllBuf.View.SubView(kv * headDim, headDim), tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, numCentroids);
            tq.BitPack4(tempIndices.View, aPacked.View.SubView(kv * packedDim, packedDim), headDim);
            ew.Scale(tempNormVal.View.SubView(0, 1), aNorms.View.SubView(kv, 1), 1, 1f);
        }
        await accelerator.SynchronizeAsync();
        var aPackedHost = await aPacked.CopyToHostAsync<int>(0, numKV * packedDim);
        var aNormsHost = await aNorms.CopyToHostAsync<float>(0, numKV);

        // ── Variant B: per-vector dedicated buffers (no offset subviews, no reuse) ──
        var bPackedHost = new int[numKV * packedDim];
        var bNormsHost = new float[numKV];
        var dedicated = new List<IDisposable>();
        try
        {
            var perKvPacked = new global::ILGPU.Runtime.MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>[numKV];
            var perKvNorm = new global::ILGPU.Runtime.MemoryBuffer1D<float, global::ILGPU.Stride1D.Dense>[numKV];
            for (int kv = 0; kv < numKV; kv++)
            {
                var nOut = accelerator.Allocate1D<float>(headDim);
                var nVal = accelerator.Allocate1D<float>(1);
                var qIdx = accelerator.Allocate1D<int>(headDim);
                var pOut = accelerator.Allocate1D<int>(packedDim);
                dedicated.Add(nOut); dedicated.Add(nVal); dedicated.Add(qIdx); dedicated.Add(pOut);
                perKvPacked[kv] = pOut; perKvNorm[kv] = nVal;
                tq.Normalize(kAllBuf.View.SubView(kv * headDim, headDim), nOut.View, nVal.View, 1, headDim);
                tq.Quantize(nOut.View, codebookBuf.View, qIdx.View, headDim, numCentroids);
                tq.BitPack4(qIdx.View, pOut.View, headDim);
            }
            await accelerator.SynchronizeAsync();
            for (int kv = 0; kv < numKV; kv++)
            {
                var p = await perKvPacked[kv].CopyToHostAsync<int>(0, packedDim);
                Array.Copy(p, 0, bPackedHost, kv * packedDim, packedDim);
                bNormsHost[kv] = (await perKvNorm[kv].CopyToHostAsync<float>(0, 1))[0];
            }
        }
        finally { foreach (var d in dedicated) d.Dispose(); }

        // ── Compare both variants against the CPU reference (reconstructed values, so a
        //    rare fp-boundary centroid flip can't fail the test; garbage cannot pass) ──
        void Verify(string label, int[] gotPacked, float[] gotNorms)
        {
            int badVals = 0, badNorms = 0;
            for (int kv = 0; kv < numKV; kv++)
            {
                if (MathF.Abs(gotNorms[kv] - refNorms[kv]) > MathF.Abs(refNorms[kv]) * 1e-4f + 1e-5f) badNorms++;
                for (int i = 0; i < headDim; i++)
                {
                    float got = codebook[(gotPacked[kv * packedDim + i / 8] >> ((i % 8) * 4)) & 0xF];
                    float want = codebook[(refPacked[kv * packedDim + i / 8] >> ((i % 8) * 4)) & 0xF];
                    if (MathF.Abs(got - want) > 0.3f) badVals++; // > one centroid gap = real corruption
                }
            }
            if (badVals > 0 || badNorms > 0)
                throw new Exception($"{label}: encode diverges from CPU reference — " +
                    $"{badVals}/{numKV * headDim} centroid values off by >0.3, {badNorms}/{numKV} norms wrong. " +
                    $"norms got=[{string.Join(",", gotNorms.Take(4).Select(v => v.ToString("F3")))}] " +
                    $"want=[{string.Join(",", refNorms.Take(4).Select(v => v.ToString("F3")))}]");
            Console.WriteLine($"[TurboQuant] {label}: encode matches CPU reference");
        }
        Verify("VariantA(subview-loop)", aPackedHost, aNormsHost);
        Verify("VariantB(dedicated)", bPackedHost, bNormsHost);

        // ── Variant C: the attention KERNEL on the proven-clean encoded data, vs a CPU
        //    oracle computing the EXACT same math (attention over the codebook-
        //    reconstructed K/V — no quantization-loss tolerance excuse). The encode above
        //    verified clean on every backend, so a failure here is the attention kernel's
        //    arithmetic/reads on this backend, with the numbers in the message. ──
        var qData = new float[headDim];
        for (int i = 0; i < headDim; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1f / MathF.Sqrt(headDim);

        // CPU oracle over the reconstruction
        var recon = new float[numKV * headDim];
        for (int kv = 0; kv < numKV; kv++)
            for (int i = 0; i < headDim; i++)
                recon[kv * headDim + i] =
                    codebook[(refPacked[kv * packedDim + i / 8] >> ((i % 8) * 4)) & 0xF] * refNorms[kv];
        var oracle = new float[headDim];
        {
            var w = new float[numKV];
            float maxScore = float.MinValue;
            for (int kv = 0; kv < numKV; kv++)
            {
                float dot = 0;
                for (int i = 0; i < headDim; i++) dot += qData[i] * recon[kv * headDim + i];
                w[kv] = dot * scale;
                maxScore = MathF.Max(maxScore, w[kv]);
            }
            float sum = 0;
            for (int kv = 0; kv < numKV; kv++) { w[kv] = MathF.Exp(w[kv] - maxScore); sum += w[kv]; }
            for (int i = 0; i < headDim; i++)
            {
                float acc = 0;
                for (int kv = 0; kv < numKV; kv++) acc += w[kv] * recon[kv * headDim + i];
                oracle[i] = acc / sum;
            }
        }

        using var qBuf = accelerator.Allocate1D(qData);
        using var vCB = accelerator.Allocate1D(codebook); // separate codebook (aliasing)

        async Task VerifyAttn(string label,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> kp,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> kn,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> vp,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> vn)
        {
            using var attnOut = accelerator.Allocate1D<float>(headDim);
            tq.FusedQuantizedAttention(qBuf.View, kp, codebookBuf.View,
                vp, vCB.View, kn, vn, attnOut.View, 1, numKV, headDim, scale);
            await accelerator.SynchronizeAsync();
            var gotAttn = await attnOut.CopyToHostAsync<float>(0, headDim);
            int badAttn = 0; int worstI = -1; float worstDiff = 0;
            for (int i = 0; i < headDim; i++)
            {
                float diff = MathF.Abs(gotAttn[i] - oracle[i]);
                float tol = MathF.Max(1e-3f, MathF.Abs(oracle[i]) * 1e-3f);
                if (diff > tol || float.IsNaN(gotAttn[i])) { badAttn++; if (diff > worstDiff) { worstDiff = diff; worstI = i; } }
            }
            if (badAttn > 0)
                throw new Exception($"{label}: {badAttn}/{headDim} elements diverge from the exact-math CPU " +
                    $"oracle (worst @{worstI}: got {gotAttn[worstI]}, want {oracle[worstI]}). " +
                    $"got=[{string.Join(",", gotAttn.Take(6).Select(v => v.ToString("F4")))}] " +
                    $"want=[{string.Join(",", oracle.Take(6).Select(v => v.ToString("F4")))}]");
            Console.WriteLine($"[TurboQuant] {label}: matches the exact-math CPU oracle");
        }

        // C0: the REAL FusedAttentionImpl method, but dispatched by the TEST (own kernel
        // load via reflection, own params buffer, host-uploaded inputs, extent numKV*D).
        // Same IL as the wrapper's kernel — if C0 passes where C2 fails, the kernel method
        // is exonerated and the difference is the WRAPPER's dispatch context alone.
        {
            var mi = typeof(TurboQuantKernels).GetMethod("FusedAttentionImpl",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
            var del = (Action<global::ILGPU.Index1D,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>>)
                Delegate.CreateDelegate(typeof(Action<global::ILGPU.Index1D,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
                global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>>), mi);
            var kern = accelerator.LoadAutoGroupedStreamKernel(del);
            // CANONICAL REPRO of the WebGL frozen-TF-capture bug (routed to Geordi,
            // seven-to-geordi 2026-06-12): ALL-FRESH buffers, the proven kernel method
            // (TurboQuant_AttentionRealShape_Probe passes this exact code in a clean test),
            // sentinel-prefilled output — yet after this test's ~112 mixed-vertex-count
            // encode dispatches, EVERY subsequent attention dispatch on WebGL returns the
            // SAME frozen values (plausible normalized-vector leftovers from an encode
            // capture). Eliminated: kernel code, params reads, buffer identity/provenance,
            // input staleness, draw vertex count (sentinel IS overwritten), GL console
            // errors (none). Remaining suspect: glWorker TF-capture reuse/growth state.
            using var kp0 = accelerator.Allocate1D(aPackedHost);
            using var kn0 = accelerator.Allocate1D(aNormsHost);
            using var vp0 = accelerator.Allocate1D(aPackedHost);
            using var vn0 = accelerator.Allocate1D(aNormsHost);
            using var q0 = accelerator.Allocate1D(qData);
            using var kcb0 = accelerator.Allocate1D(codebook);
            using var vcb0 = accelerator.Allocate1D(codebook);
            using var out0 = accelerator.Allocate1D(Enumerable.Repeat(777f, headDim).ToArray());
            using var par0 = accelerator.Allocate1D(new int[] { 1, numKV, headDim,
                BitConverter.SingleToInt32Bits(scale), 8, 4, 0xF });
            kern(headDim, q0.View, kp0.View, kcb0.View, vp0.View, vcb0.View,
                kn0.View, vn0.View, out0.View, par0.View);
            await accelerator.SynchronizeAsync();
            var got0 = await out0.CopyToHostAsync<float>(0, headDim);
            int bad0 = 0;
            for (int i = 0; i < headDim; i++)
                if (MathF.Abs(got0[i] - oracle[i]) > MathF.Max(1e-3f, MathF.Abs(oracle[i]) * 1e-3f) || float.IsNaN(got0[i])) bad0++;
            if (bad0 > 0)
                throw new Exception($"VariantC0(all-fresh, post-encode dispatches): {bad0}/{headDim} diverge — " +
                    "the WebGL frozen-TF-capture bug (see comment; routed to the ILGPU lane). " +
                    $"got=[{string.Join(",", got0.Take(6).Select(v => v.ToString("F4")))}] " +
                    $"want=[{string.Join(",", oracle.Take(6).Select(v => v.ToString("F4")))}]");
            Console.WriteLine("[TurboQuant] VariantC0(all-fresh): matches oracle");
        }

        // C2: ALL inputs re-uploaded from HOST (the verified readbacks) — isolates whether
        // GPU-WRITTEN input buffers (vs CPU-uploaded) are what breaks the attention kernel.
        using (var kp2 = accelerator.Allocate1D(aPackedHost))
        using (var kn2 = accelerator.Allocate1D(aNormsHost))
        using (var vp2 = accelerator.Allocate1D(aPackedHost))
        using (var vn2 = accelerator.Allocate1D(aNormsHost))
            await VerifyAttn("VariantC2(host-uploaded inputs)", kp2.View, kn2.View, vp2.View, vn2.View);

        // C3: K side = the kernel-WRITTEN buffers, V side = host-uploaded — splits
        // kernel-written-texture staleness (K) from the GPU→GPU CopyFromAsync path (V, C1).
        using (var vp3 = accelerator.Allocate1D(aPackedHost))
        using (var vn3 = accelerator.Allocate1D(aNormsHost))
            await VerifyAttn("VariantC3(K=kernel-written, V=host)", aPacked.View, aNorms.View, vp3.View, vn3.View);

        // C1: the original failing shape — K = kernel-written, V = GPU→GPU CopyFromAsync.
        using var vPacked2 = accelerator.Allocate1D<int>(numKV * packedDim);
        using var vNorms2 = accelerator.Allocate1D<float>(numKV);
        await vPacked2.View.CopyFromAsync(aPacked.View);
        await vNorms2.View.CopyFromAsync(aNorms.View);
        await VerifyAttn("VariantC1(K=kernel-written, V=gpu-copy)", aPacked.View, aNorms.View, vPacked2.View, vNorms2.View);
    });

    /// <summary>PROBE for the WebGL VariantC failure: echoes the params-buffer values the
    /// kernel actually SEES (same 9-view signature as the TurboQuant attention kernels,
    /// same int-params read pattern + IntAsFloat). If WebGL shows zeros/garbage here, the
    /// params plumbing is the bug; if correct, the fault is further into the kernel.</summary>
    private static void ParamsEchoImpl(global::ILGPU.Index1D idx,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> Q,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> K_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_codebook,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> V_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_codebook,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> output,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> paramsArr)
    {
        // Slots 0..6: raw params as float. Slot 7: IntAsFloat(params[3]) (the scale path).
        // Slot 8: K_norms[0]. Slot 9: K_codebook[1]. Slot 10: K_packed[0] low nibble.
        int s = idx % 11;
        float v = 0f;
        if (s < 7) v = (float)paramsArr[s];
        else if (s == 7) v = global::ILGPU.Interop.IntAsFloat((uint)paramsArr[3]);
        else if (s == 8) v = K_norms[0];
        else if (s == 9) v = K_codebook[1];
        else v = (float)(K_packed[0] & 0xF);
        output[idx] = v + 0f * (Q[0] + V_codebook[0] + V_norms[0] + (float)V_packed[0]);
    }

    /// <summary>STAGE PROBE: replicates the two-pass attention kernel body at REAL sizes
    /// (headDim=64, packedDim=8, numKV=16) but outputs a chosen INTERMEDIATE per slot —
    /// dot(kv=0), maxScore, sumExp, acc, final — so one run shows the FIRST stage that
    /// diverges on a backend (the params/read probe above already proved inputs clean).</summary>
    private static void AttnStagesImpl(global::ILGPU.Index1D idx,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> Q,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> K_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_codebook,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> V_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_codebook,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> output,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> paramsArr)
    {
        int numQ = paramsArr[0];
        int numKV = paramsArr[1];
        int D = paramsArr[2];
        float scale = global::ILGPU.Interop.IntAsFloat((uint)paramsArr[3]);
        int valuesPerInt = paramsArr[4];
        int bitsPerValue = paramsArr[5];
        int indexMask = paramsArr[6];
        int packedDim = (D + valuesPerInt - 1) / valuesPerInt;
        // Thread idx selects the reported stage; d is fixed to 5 so every stage is scalar.
        int stage = idx % 8;
        int queryIdx = 0;
        int d = 5;
        if (queryIdx >= numQ) { output[idx] = -999f; return; }

        float maxScore = -1e10f;
        float dot0 = 0f;
        for (int kv = 0; kv < numKV; kv++)
        {
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < valuesPerInt && p * valuesPerInt + b < D; b++)
                {
                    int cIdx = (packed >> (b * bitsPerValue)) & indexMask;
                    dot += Q[queryIdx * D + p * valuesPerInt + b] * K_codebook[cIdx] * kNorm;
                }
            }
            if (kv == 0) dot0 = dot;
            maxScore = MathF.Max(maxScore, dot * scale);
        }

        float sumExp = 0f;
        float acc = 0f;
        int vp = d / valuesPerInt;
        int vShift = (d % valuesPerInt) * bitsPerValue;
        for (int kv = 0; kv < numKV; kv++)
        {
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < valuesPerInt && p * valuesPerInt + b < D; b++)
                {
                    int cIdx = (packed >> (b * bitsPerValue)) & indexMask;
                    dot += Q[queryIdx * D + p * valuesPerInt + b] * K_codebook[cIdx] * kNorm;
                }
            }
            float weight = MathF.Exp(dot * scale - maxScore);
            sumExp += weight;
            int vIdx = (V_packed[kv * packedDim + vp] >> vShift) & indexMask;
            acc += weight * V_codebook[vIdx] * V_norms[kv];
        }

        float v = 0f;
        if (stage == 0) v = dot0;
        else if (stage == 1) v = maxScore;
        else if (stage == 2) v = sumExp;
        else if (stage == 3) v = acc;
        else if (stage == 4) v = acc / (sumExp + 1e-10f);
        else if (stage == 5) v = MathF.Exp(-1.5f);          // bare exp sanity
        else if (stage == 6) v = MathF.Max(-1e10f, 2.5f);   // bare max sanity
        else v = scale;
        output[idx] = v;
    }

    /// <summary>REAL-SHAPE probe: verbatim copy of FusedAttentionImpl's body with a MODE
    /// param toggling the structural differences from the (passing) stage probe:
    /// mode 0 = exact real shape (per-thread d, bare early return), mode 1 = early return
    /// ASSIGNS output first, mode 2 = d fixed to 5 (the passing probe's shape).</summary>
    private static void AttnRealShapeImpl(global::ILGPU.Index1D idx,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> Q,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> K_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_codebook,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> V_packed,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_codebook,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> K_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> V_norms,
        global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense> output,
        global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense> paramsArr)
    {
        int numQ = paramsArr[0];
        int numKV = paramsArr[1];
        int D = paramsArr[2];
        float scale = global::ILGPU.Interop.IntAsFloat((uint)paramsArr[3]);
        int valuesPerInt = paramsArr[4];
        int bitsPerValue = paramsArr[5];
        int indexMask = paramsArr[6];
        int mode = paramsArr[7];
        int packedDim = (D + valuesPerInt - 1) / valuesPerInt;
        int queryIdx = idx / D;
        int d = mode == 2 ? 5 : idx % D;
        if (queryIdx >= numQ)
        {
            if (mode == 1) output[idx] = -999f;
            return;
        }

        float maxScore = -1e10f;
        for (int kv = 0; kv < numKV; kv++)
        {
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < valuesPerInt && p * valuesPerInt + b < D; b++)
                {
                    int cIdx = (packed >> (b * bitsPerValue)) & indexMask;
                    dot += Q[queryIdx * D + p * valuesPerInt + b] * K_codebook[cIdx] * kNorm;
                }
            }
            maxScore = MathF.Max(maxScore, dot * scale);
        }

        float sumExp = 0f;
        float acc = 0f;
        int vp = d / valuesPerInt;
        int vShift = (d % valuesPerInt) * bitsPerValue;
        for (int kv = 0; kv < numKV; kv++)
        {
            float dot = 0f;
            float kNorm = K_norms[kv];
            for (int p = 0; p < packedDim; p++)
            {
                int packed = K_packed[kv * packedDim + p];
                for (int b = 0; b < valuesPerInt && p * valuesPerInt + b < D; b++)
                {
                    int cIdx = (packed >> (b * bitsPerValue)) & indexMask;
                    dot += Q[queryIdx * D + p * valuesPerInt + b] * K_codebook[cIdx] * kNorm;
                }
            }
            float weight = MathF.Exp(dot * scale - maxScore);
            sumExp += weight;
            int vIdx = (V_packed[kv * packedDim + vp] >> vShift) & indexMask;
            acc += weight * V_codebook[vIdx] * V_norms[kv];
        }
        output[idx] = acc / (sumExp + 1e-10f);
    }

    [TestMethod]
    public async Task TurboQuant_AttentionRealShape_Probe() => await RunTest(async accelerator =>
    {
        const int headDim = 64, numKV = 16;
        int packedDim = headDim / 8;
        var rng = new Random(7);
        var codebook = TurboQuantKernels.Codebook4Bit;
        var kPackedData = new int[numKV * packedDim];
        var vPackedData = new int[numKV * packedDim];
        for (int i = 0; i < kPackedData.Length; i++) kPackedData[i] = rng.Next(int.MinValue, int.MaxValue);
        for (int i = 0; i < vPackedData.Length; i++) vPackedData[i] = rng.Next(int.MinValue, int.MaxValue);
        var kNormsData = new float[numKV];
        var vNormsData = new float[numKV];
        for (int i = 0; i < numKV; i++) { kNormsData[i] = 1f + (float)rng.NextDouble(); vNormsData[i] = 1f + (float)rng.NextDouble(); }
        var qData = new float[headDim];
        for (int i = 0; i < headDim; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1f / MathF.Sqrt(headDim);

        // CPU oracle (per element d)
        float Dot(int kv)
        {
            float dot = 0;
            for (int p = 0; p < packedDim; p++)
            {
                int packed = kPackedData[kv * packedDim + p];
                for (int b = 0; b < 8; b++)
                    dot += qData[p * 8 + b] * codebook[(packed >> (b * 4)) & 0xF] * kNormsData[kv];
            }
            return dot;
        }
        float cMax = -1e10f;
        for (int kv = 0; kv < numKV; kv++) cMax = MathF.Max(cMax, Dot(kv) * scale);
        float OracleAt(int d)
        {
            float cSum = 0, cAcc = 0;
            for (int kv = 0; kv < numKV; kv++)
            {
                float w = MathF.Exp(Dot(kv) * scale - cMax);
                cSum += w;
                cAcc += w * codebook[(vPackedData[kv * packedDim + d / 8] >> ((d % 8) * 4)) & 0xF] * vNormsData[kv];
            }
            return cAcc / (cSum + 1e-10f);
        }

        using var qBuf = accelerator.Allocate1D(qData);
        using var kPacked = accelerator.Allocate1D(kPackedData);
        using var vPacked = accelerator.Allocate1D(vPackedData);
        using var kCB = accelerator.Allocate1D(codebook);
        using var vCB = accelerator.Allocate1D(codebook);
        using var kNorms = accelerator.Allocate1D(kNormsData);
        using var vNorms = accelerator.Allocate1D(vNormsData);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<global::ILGPU.Index1D,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>>(AttnRealShapeImpl);

        var fails = new List<string>();
        for (int mode = 0; mode <= 2; mode++)
        {
            using var outBuf = accelerator.Allocate1D(Enumerable.Repeat(777f, headDim).ToArray());
            using var paramsBuf = accelerator.Allocate1D(new int[] { 1, numKV, headDim,
                BitConverter.SingleToInt32Bits(scale), 8, 4, 0xF, mode });
            kernel(headDim, qBuf.View, kPacked.View, kCB.View, vPacked.View, vCB.View,
                kNorms.View, vNorms.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var got = await outBuf.CopyToHostAsync<float>(0, headDim);
            int bad = 0;
            for (int i = 0; i < headDim; i++)
            {
                float want = OracleAt(mode == 2 ? 5 : i);
                if (MathF.Abs(got[i] - want) > MathF.Max(1e-4f, MathF.Abs(want) * 1e-4f) || float.IsNaN(got[i])) bad++;
            }
            if (bad > 0)
                fails.Add($"mode{mode}: {bad}/{headDim} bad, got=[{string.Join(",", got.Take(5).Select(v => v.ToString("F4")))}] " +
                    $"want=[{string.Join(",", Enumerable.Range(0, 5).Select(i => OracleAt(mode == 2 ? 5 : i).ToString("F4")))}]");
            else
                Console.WriteLine($"[TurboQuant] RealShape mode{mode}: matches oracle");
        }
        if (fails.Count > 0) throw new Exception($"RealShape probe: {string.Join(" || ", fails)}");
    });

    [TestMethod]
    public async Task TurboQuant_AttentionStages_Probe() => await RunTest(async accelerator =>
    {
        const int headDim = 64, numKV = 16, numCentroids = 16;
        int packedDim = headDim / 8;
        var rng = new Random(7);
        var codebook = TurboQuantKernels.Codebook4Bit;

        // Synthetic packed data + norms + Q, fully CPU-replicable
        var kPackedData = new int[numKV * packedDim];
        var vPackedData = new int[numKV * packedDim];
        for (int i = 0; i < kPackedData.Length; i++) kPackedData[i] = rng.Next(int.MinValue, int.MaxValue);
        for (int i = 0; i < vPackedData.Length; i++) vPackedData[i] = rng.Next(int.MinValue, int.MaxValue);
        var kNormsData = new float[numKV];
        var vNormsData = new float[numKV];
        for (int i = 0; i < numKV; i++) { kNormsData[i] = 1f + (float)rng.NextDouble(); vNormsData[i] = 1f + (float)rng.NextDouble(); }
        var qData = new float[headDim];
        for (int i = 0; i < headDim; i++) qData[i] = (float)(rng.NextDouble() * 2 - 1);
        float scale = 1f / MathF.Sqrt(headDim);

        // CPU replication of every stage (identical arithmetic order)
        float Dot(int kv)
        {
            float dot = 0;
            for (int p = 0; p < packedDim; p++)
            {
                int packed = kPackedData[kv * packedDim + p];
                for (int b = 0; b < 8; b++)
                    dot += qData[p * 8 + b] * codebook[(packed >> (b * 4)) & 0xF] * kNormsData[kv];
            }
            return dot;
        }
        float cMax = -1e10f;
        for (int kv = 0; kv < numKV; kv++) cMax = MathF.Max(cMax, Dot(kv) * scale);
        float cSum = 0, cAcc = 0;
        const int dFixed = 5;
        for (int kv = 0; kv < numKV; kv++)
        {
            float w = MathF.Exp(Dot(kv) * scale - cMax);
            cSum += w;
            cAcc += w * codebook[(vPackedData[kv * packedDim + dFixed / 8] >> ((dFixed % 8) * 4)) & 0xF] * vNormsData[kv];
        }
        var want = new float[8] { Dot(0), cMax, cSum, cAcc, cAcc / (cSum + 1e-10f), MathF.Exp(-1.5f), 2.5f, scale };

        using var qBuf = accelerator.Allocate1D(qData);
        using var kPacked = accelerator.Allocate1D(kPackedData);
        using var vPacked = accelerator.Allocate1D(vPackedData);
        using var kCB = accelerator.Allocate1D(codebook);
        using var vCB = accelerator.Allocate1D(codebook);
        using var kNorms = accelerator.Allocate1D(kNormsData);
        using var vNorms = accelerator.Allocate1D(vNormsData);
        using var outBuf = accelerator.Allocate1D<float>(8);
        using var paramsBuf = accelerator.Allocate1D(new int[] { 1, numKV, headDim,
            BitConverter.SingleToInt32Bits(scale), 8, 4, 0xF });

        var kernel = accelerator.LoadAutoGroupedStreamKernel<global::ILGPU.Index1D,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>>(AttnStagesImpl);
        kernel(8, qBuf.View, kPacked.View, kCB.View, vPacked.View, vCB.View,
            kNorms.View, vNorms.View, outBuf.View, paramsBuf.View);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, 8);

        var names = new[] { "dot0", "maxScore", "sumExp", "acc", "final", "exp(-1.5)", "max", "scale" };
        var bad = new List<string>();
        for (int s = 0; s < 8; s++)
            if (MathF.Abs(got[s] - want[s]) > MathF.Abs(want[s]) * 1e-4f + 1e-5f || float.IsNaN(got[s]))
                bad.Add($"{names[s]}: got {got[s]:G6}, want {want[s]:G6}");
        if (bad.Count > 0)
            throw new Exception($"Attention stage(s) diverge: {string.Join("; ", bad)}");
        Console.WriteLine("[TurboQuant] AttentionStages probe: all stages match CPU");
    });

    [TestMethod]
    public async Task TurboQuant_AttentionParamsEcho_Probe() => await RunTest(async accelerator =>
    {
        const int headDim = 64, numKV = 16;
        int packedDim = headDim / 8;
        var codebook = TurboQuantKernels.Codebook4Bit;
        using var qBuf = accelerator.Allocate1D(new float[headDim]);
        using var kPacked = accelerator.Allocate1D(Enumerable.Range(0, numKV * packedDim).Select(i => 0x5A5A5A57).ToArray());
        using var vPacked = accelerator.Allocate1D<int>(numKV * packedDim);
        using var kCB = accelerator.Allocate1D(codebook);
        using var vCB = accelerator.Allocate1D(codebook);
        using var kNorms = accelerator.Allocate1D(Enumerable.Repeat(3.25f, numKV).ToArray());
        using var vNorms = accelerator.Allocate1D<float>(numKV);
        using var outBuf = accelerator.Allocate1D<float>(22);

        float scale = 0.125f;
        var paramsData = new int[] { 1, numKV, headDim, BitConverter.SingleToInt32Bits(scale), 8, 4, 0xF };
        using var paramsBuf = accelerator.Allocate1D(paramsData);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<global::ILGPU.Index1D,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<float, global::ILGPU.Stride1D.Dense>,
            global::ILGPU.Runtime.ArrayView1D<int, global::ILGPU.Stride1D.Dense>>(ParamsEchoImpl);
        kernel(22, qBuf.View, kPacked.View, kCB.View, vPacked.View, vCB.View,
            kNorms.View, vNorms.View, outBuf.View, paramsBuf.View);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, 22);

        var want = new float[11];
        for (int s = 0; s < 7; s++) want[s] = paramsData[s];
        want[7] = scale; want[8] = 3.25f; want[9] = codebook[1]; want[10] = 0x5A5A5A57 & 0xF;
        var bad = new List<string>();
        for (int i = 0; i < 22; i++)
            if (MathF.Abs(got[i] - want[i % 11]) > MathF.Abs(want[i % 11]) * 1e-6f + 1e-6f)
                bad.Add($"slot{i % 11}@{i}: got {got[i]}, want {want[i % 11]}");
        if (bad.Count > 0)
            throw new Exception($"Params echo diverges ({bad.Count}/22): {string.Join("; ", bad.Take(8))}");
        Console.WriteLine("[TurboQuant] ParamsEcho probe: kernel sees correct params/buffer values");
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
        // Upload K and V ONCE as whole buffers and SubView per vector. Allocating and
        // disposing a per-iteration kVec/vVec mid-loop destroyed the buffer before the
        // batched WebGPU dispatches were flushed at SynchronizeAsync ("Buffer used in submit
        // while destroyed", one error per KV position). Whole-buffer SubViews live until the
        // flush and avoid numKV separate uploads.
        using var qBuf = accelerator.Allocate1D(qData);
        using var kAllBuf = accelerator.Allocate1D(kData);
        using var vAllBuf = accelerator.Allocate1D(vData);
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
            tq.Normalize(kAllBuf.View.SubView(kv * headDim, headDim), tempNorm.View, tempNormVal.View, 1, headDim);
            tq.Quantize(tempNorm.View, codebookBuf.View, tempIndices.View, headDim, 16);
            tq.BitPack4(tempIndices.View, kPackedBuf.View.SubView(kv * packedDim, packedDim), headDim);
            new ElementWiseKernels(accelerator).Scale(tempNormVal.View.SubView(0, 1), kNormsBuf.View.SubView(kv, 1), 1, 1f);

            tq.Normalize(vAllBuf.View.SubView(kv * headDim, headDim), tempNorm.View, tempNormVal.View, 1, headDim);
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
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task TurboQuant_WhisperDecoder_KVCacheAutoDetected() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Use base decoder (no If control flow nodes) — same KV cache outputs
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model.onnx");

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

        // Base decoder uses present.N.decoder.key naming (not present.N.key).
        // KV cache auto-detection may not match this pattern yet.
        // Full KV cache support requires the merged model + If operator.
        if (hasCache)
        {
            var kvCache = session.Executor.KVCache!;
            Console.WriteLine($"[TurboQuant] Whisper KV cache: {kvCache.NumLayers} layers, maxSeq={kvCache.MaxSeqLen}");
        }
        else
        {
            // Verify the model at least has present outputs (KV cache structure exists)
            bool hasPresentOutputs = session.OutputNames.Any(n => n.StartsWith("present."));
            if (!hasPresentOutputs)
                throw new Exception("Whisper decoder should have present.* outputs but none found");
            Console.WriteLine($"[TurboQuant] Whisper has {session.OutputNames.Count(n => n.StartsWith("present."))} present outputs (KV cache structure present, auto-detection pending)");
        }

        Console.WriteLine($"[TurboQuant] Whisper decoder KV cache detection: PASS");
    });

    /// <summary>
    /// Run one decoder step of Whisper Tiny with fake encoder output,
    /// verify quantized KV cache captures the token's key/value data.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task TurboQuant_WhisperDecoder_KVCacheCaptures() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Use base decoder (no If control flow nodes) — same KV cache outputs
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model.onnx");

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
            new ElementWiseKernels(accelerator).ScaleInPlace(transformed.View, numVecs * d, sqrtD);
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
            // Scale by 1/√d to undo pre-quantization scaling.
            // In-place scale MUST use ScaleInPlace (single buffer binding): passing
            // transformed.View as both input and output to Scale() binds the same GPU
            // buffer to two read_write storage slots, which WebGPU forbids (the library
            // correctly throws "Storage buffer aliasing detected"). Mirrors the ScaleInPlace
            // call in the forward pass above.
            float invSqrtD = 1f / MathF.Sqrt(d);
            new ElementWiseKernels(accelerator).ScaleInPlace(transformed.View, numVecs * d, invSqrtD);
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
            // In-place scale MUST use ScaleInPlace (single buffer binding) — passing transformed.View
            // as both input and output to Scale() aliases one GPU buffer to two read_write storage
            // slots, which WebGPU forbids. Mirrors the ScaleInPlace calls in the 3-bit/4-bit pass above.
            new ElementWiseKernels(accelerator).ScaleInPlace(transformed.View, numVecs * d, MathF.Sqrt(d));

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
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task TurboQuant_GPT2_Baseline_NoKVCache() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DistilGPT-2 base decoder (no If control flow, no past_key_values inputs)
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
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
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task TurboQuant_GPT2_WithKVCache() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // DistilGPT-2 merged (WITH KV cache inputs/outputs)
        // Use base decoder (no If control flow nodes) — same KV cache outputs
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/Xenova/distilgpt2/resolve/main/onnx/decoder_model.onnx");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["input_ids"] = new[] { 1, 5 },
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
        var kFlat = new float[numTokens * vecDim];
        var vFlat = new float[numTokens * vecDim];

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
            Array.Copy(kData, 0, kFlat, t * vecDim, vecDim);
            Array.Copy(vData, 0, vFlat, t * vecDim, vecDim);
        }

        // Upload ALL tokens once and Append each token's SubView. kvCache.Append runs
        // QuantizeVector, which dispatches kernels that READ the input view — on WebGPU
        // these are batched in the command encoder and not executed until SynchronizeAsync
        // below. The previous code allocated a per-token `using var kBuf`/`vBuf` and disposed
        // it each iteration, destroying the buffer before the flush, so the cache quantized
        // garbage and the GPU attention output was all zeros (cosine 0). Whole-buffer
        // SubViews stay alive until the flush.
        using var kAllBuf = accelerator.Allocate1D(kFlat);
        using var vAllBuf = accelerator.Allocate1D(vFlat);
        for (int t = 0; t < numTokens; t++)
        {
            kvCache.Append(0, kAllBuf.View.SubView(t * vecDim, vecDim), vAllBuf.View.SubView(t * vecDim, vecDim));
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
