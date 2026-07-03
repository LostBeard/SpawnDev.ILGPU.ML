using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Tensors;
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

    // ── DA3 Model Tests (require ONNX from HuggingFace) ──

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task DA3Small_ONNX_Loads() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Download model + external data
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

        Console.WriteLine($"[DA3] model.onnx: {onnxBytes.Length / 1024}KB, model.onnx_data: {extDataBytes.Length / 1024 / 1024}MB");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            },
            externalData: extDataBytes);

        Console.WriteLine($"[DA3] Loaded: inputs=[{string.Join(",", session.InputNames)}], outputs=[{string.Join(",", session.OutputNames)}]");

        if (session.InputNames.Length == 0)
            throw new Exception("DA3 model has no inputs");
        if (session.OutputNames.Length == 0)
            throw new Exception("DA3 model has no outputs");

        Console.WriteLine($"[DA3] DA3-Small ONNX load: PASS");
    });

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task DA3Small_Inference_ProducesDepth() => await RunTest(async accelerator =>
    {
        // Baseline measurement (fold OFF): where does DAv3's per-inference wall-clock actually go?
        // readbackMs vs syncDrainMs vs run tells us whether the ~1400 shape readbacks are the real cost.
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        long tDownload, tCreate, tRun, tVerify;

        Console.WriteLine($"[DA3] start at t=0ms");

        // Download model + external data
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        Console.WriteLine($"[DA3] onnx={onnxBytes.Length / 1024}KB downloaded at t={sw.ElapsedMilliseconds}ms");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        tDownload = sw.ElapsedMilliseconds;
        Console.WriteLine($"[DA3] ext_data={extDataBytes.Length / 1024 / 1024}MB downloaded at t={tDownload}ms (download phase done)");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            },
            externalData: extDataBytes);
        tCreate = sw.ElapsedMilliseconds - tDownload;
        Console.WriteLine($"[DA3] session created in {tCreate}ms (parse + compile + weight upload), nodes={session.NodeCount}");

        // Generate test input: random normalized image [1, 3, 224, 224]
        var rng = new Random(42);
        int pixelCount = 3 * 224 * 224;
        var inputData = new float[pixelCount];
        for (int i = 0; i < pixelCount; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, 224, 224 });

        long tRunStart = sw.ElapsedMilliseconds;
        Console.WriteLine($"[DA3] starting inference at t={tRunStart}ms");
        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });
        await accelerator.SynchronizeAsync();
        tRun = sw.ElapsedMilliseconds - tRunStart;
        Console.WriteLine($"[DA3] inference + sync done in {tRun}ms");

        var output = outputs[session.OutputNames[0]];
        int elems = output.ElementCount;

        if (elems < 100)
            throw new Exception($"DA3 output too small: {elems} elements (shape=[{string.Join(",", output.Shape)}])");

        // GPU-side finite check + reduction. Only 3 floats read back on
        // atomics-capable backends; no per-element CPU loop. Per project CLAUDE.md:
        // "Tests must not waste resources. Use GPU-side verification when it exists."
        long tVerifyStart = sw.ElapsedMilliseconds;
        Console.WriteLine($"[DA3] starting GPU verify at t={tVerifyStart}ms (elems={elems})");
        var ew = new ElementWiseKernels(accelerator);
        var (nanCount, absSum, absMax) = await ew.FiniteCheckOnGpuAsync(
            output.Data.SubView(0, elems), elems);
        tVerify = sw.ElapsedMilliseconds - tVerifyStart;
        float meanAbs = absSum / Math.Max(1, elems - nanCount);

        Console.WriteLine($"[DA3] verify done in {tVerify}ms");
        Console.WriteLine($"[DA3] timing: download={tDownload}ms, create={tCreate}ms, run={tRun}ms, verify={tVerify}ms, total={sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"[DA3] output: shape=[{string.Join(",", output.Shape)}], elems={elems}, absMax={absMax:F4}, meanAbs={meanAbs:F4}, NaN={nanCount}/{elems}");

        if (nanCount > elems / 10)
            throw new Exception($"DA3 output has {nanCount}/{elems} NaN values (timing: dl={tDownload}ms create={tCreate}ms run={tRun}ms verify={tVerify}ms)");
        if (absMax == 0)
            throw new Exception($"DA3 output is all zeros (timing: dl={tDownload}ms create={tCreate}ms run={tRun}ms verify={tVerify}ms)");

        // Throw-on-pass surfaces the timing breakdown in the test result so we can
        // see WHERE wall-clock budget went (download / compile / inference / verify).
        // folded/readbacks measure the compile-time shape-subgraph fold (2026-07-01): folded = shape nodes
        // removed at compile (target ~1400 for DAv3), readbacks = per-inference GPU->CPU shape drains remaining.
        throw new Exception($"PASSED. nodes={session.NodeCount} folded={Graph.GraphCompiler.LastCompileFoldedNodeCount} readbacks={Graph.GraphExecutor.LastRunReadbackCount} readbackMs={Graph.GraphExecutor.LastRunReadbackMs:F0} drains={Graph.GraphExecutor.LastRunSyncDrainCount} drainMs={Graph.GraphExecutor.LastRunSyncDrainMs:F0}; timing: create={tCreate}ms run={tRun}ms verify={tVerify}ms total={sw.ElapsedMilliseconds}ms; absMax={absMax:F4} NaN={nanCount}/{elems}");
    });

    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task DA3Small_DepthMap_NotFlat() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Download model + external data
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            },
            externalData: extDataBytes);

        // Generate structured input: gradient image (left dark, right bright)
        int pixelCount = 3 * 224 * 224;
        var inputData = new float[pixelCount];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 224; y++)
                for (int x = 0; x < 224; x++)
                    inputData[c * 224 * 224 + y * 224 + x] = (x / 223f) * 2f - 1f;

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, 224, 224 });

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        var output = outputs[session.OutputNames[0]];
        int elems = output.ElementCount;

        using var readBuf = accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await accelerator.SynchronizeAsync();
        var actual = await readBuf.CopyToHostAsync<float>(0, elems);

        // Verify depth map has spatial variation (not flat)
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < actual.Length; i++)
        {
            if (float.IsNaN(actual[i])) continue;
            min = MathF.Min(min, actual[i]);
            max = MathF.Max(max, actual[i]);
        }
        float range = max - min;
        Console.WriteLine($"[DA3] Depth range: [{min:F4}, {max:F4}], range={range:F4}");

        if (range < 0.01f)
            throw new Exception($"DA3 depth map is flat (range={range:F6}). Model not computing correctly.");

        Console.WriteLine($"[DA3] DA3-Small depth map variation: PASS (range={range:F4})");
    });

    /// <summary>
    /// Data's EXACT SpawnScene path: DAv3-Small at its native 5-D input [1,1,3,518,518] driven through the
    /// public <see cref="Pipelines.DepthEstimationPipeline.EstimateGpuRawAsync"/> (preprocess → forward → GPU
    /// resize), NOT the 4-D [1,3,224,224] RunAsync the other DA3 tests use. This resolution + rank + entry point
    /// is what exposed the WebGPU-only "buffer used in submit while destroyed" at node 177
    /// (/backbone/blocks.4/attn/rope/Add). ROOT CAUSE (fixed): GatherKernel.GatherGenericFloat destroyed the
    /// previous call's params buffer INLINE every call; the RoPE dynamic-shape subgraph issues several Gather
    /// calls that batch into one un-submitted WebGPU command encoder, so the 2nd Gather destroyed the 1st's
    /// params buffer while the 1st's dispatch was still pending → the next Queue.Submit failed. Synchronous
    /// CUDA/OpenCL submit eagerly so never hit it. Fix = defer the old params buffer (GatherKernel _oldGenericParams).
    /// This test runs the real pipeline on WebGPU (past the multi-Gather RoPE region) + desktop refs and proves
    /// the forward pass is finite + spatially varying — the ONLY 5-D-pipeline-on-WebGPU regression guard.
    /// (Desktop-only for now; WebGPU re-enabled after the compile-time shape-subgraph fold — see skip below.)
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DA3Small_Pipeline_5D_WebGPU_ProducesDepth() => await RunTest(async accelerator =>
    {
        // CUDA/OpenCL/CPU are the fast desktop refs. WebGL can't compile the DAv3 vertex shader; Wasm is non-AOT.
        // WebGPU is TEMPORARILY skipped here too: the crash is fixed (verified separately), but until the
        // compile-time shape-subgraph fold lands, the 1416 per-inference shape readbacks make a WebGPU forward
        // take minutes (would time out this HeavyModel test). Re-enable WebGPU once the shape-fold perf fix lands.
        // Desktop lanes (CUDA/OpenCL/CPU) — interpreter now gated to browser GPU only, so CPU should no longer
        // fault and CUDA/OpenCL show Seven's kernel wins without the interpreter's CopyFromCPU overhead.
        // WebGL/Wasm too slow; WebGPU measured separately.
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: DAv3 depth pipeline skipped here (see comment)");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // Runtime CPU shape interpreter ON (browser-gated readback-skip). Dispatch-elide OFF — it corrupts the
        // executor's runtime shape cascade (downstream Concat mis-sized) and needs a deeper fix (shape resolution
        // must read runtimeConstants for elided outputs). Deferred; readback-skip + Seven's kernels are the wins.
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = false;

        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

        // Native DAv3 shape: 5-D [1, num_images=1, 3, 518, 518] — the rank the graph is compiled for.
        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 1, 3, 518, 518 }
            },
            externalData: extDataBytes);
        Console.WriteLine($"[DA3-5D] session created, nodes={session.NodeCount}, input={string.Join(",", session.InputShapes[session.InputNames[0]])}");

        using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator); // inputSize auto-derives to 518

        // Structured 518x518 RGBA gradient (left dark → right bright) so the depth map must vary spatially.
        const int W = 518, H = 518;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int v = (int)(x / (float)(W - 1) * 255f);
                rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v; // A R G B packed
            }

        var (rawDepth, minD, maxD, outW, outH) = await pipeline.EstimateGpuRawAsync(rgba, W, H);
        using (rawDepth)
        {
            int outSize = outW * outH;
            var (nanCount, absSum, absMax) = await new ElementWiseKernels(accelerator)
                .FiniteCheckOnGpuAsync(rawDepth.View.SubView(0, outSize), outSize);
            float range = maxD - minD;
            if (nanCount > outSize / 10)
                throw new Exception($"DA3 5-D pipeline output has {nanCount}/{outSize} NaN values");
            if (range < 0.01f)
                throw new Exception($"DA3 5-D pipeline depth map is flat (range={range:F6}) — forward pass wrong");
            // Return normally = TestResult.Success (UnitTestRunner: throw => Error). Metrics to console/log.
            Console.WriteLine($"[DA3-5D] PASSED. nodes={session.NodeCount} interpResolved={Graph.GraphExecutor.LastRunShapeInterpResolved} "
                + $"readbacks={Graph.GraphExecutor.LastRunReadbackCount} totalMs={Graph.GraphExecutor.LastRunTotalMs:F0}; "
                + $"{outW}x{outH} range={range:F6}");
        }
    });

    /// <summary>
    /// DIAGNOSTIC + regression guard for DISPATCH-ELIDE (the CUDA ~1200ms orchestration lever).
    /// Runs the real DAv3 5-D pipeline on the fast desktop lanes with the CPU shape interpreter AND
    /// dispatch-elide ON (ShapeInterpElideDispatch): interpreter-resolved shape ops are NOT dispatched to
    /// the GPU at all, removing their per-node orchestration on every backend. Known to CRASH before the
    /// fix — an elided shape-op output consumed as a GPU tensor is materialized on-demand as rank-1
    /// [cval.Length], breaking a downstream rank-matched Concat (GraphExecutor Concat runtime override).
    /// On failure this dumps GraphExecutor.LastRunOpLog (per-node trace; elided nodes tagged ~cpu-elided)
    /// so the exact producer/consumer is named. On success it reports range + totalMs + resolved + elided
    /// so the elide win is measurable. GATE (post-fix): range must match the elide-off reference (Seven's
    /// desktop 0.1365) — elide is a pure orchestration removal, so depth must be byte-similar.
    /// CUDA/OpenCL only (interpreter can fault on the CPU backend; browser lanes measured separately).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DA3Small_Pipeline_5D_ElideDispatch() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: dispatch-elide diag runs on CUDA/OpenCL only");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);

            using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);

            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int v = (int)(x / (float)(W - 1) * 255f);
                    rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
                }

            // REFERENCE run: interpreter ON, dispatch-elide OFF (the committed, Seven-verified path, range≈0.1365).
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
            var refR = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            int outSize = refR.Width * refR.Height;
            float refRange = refR.MaxDepth - refR.MinDepth;
            float[] refDepth;
            using (refR.RawDepth) refDepth = await refR.RawDepth.CopyToHostAsync<float>(0, outSize);

            // TEST run: dispatch-elide ON (rank-safe elide). This is where the pre-fix code crashed on the
            // RoPE Concat_12 rank mismatch; on the op-log dump names the elided producer/failing consumer.
            Graph.GraphExecutor.ShapeInterpElideDispatch = true;
            (MemoryBuffer1D<float, Stride1D.Dense> rawDepth, float minD, float maxD, int outW, int outH) tR;
            try { tR = await pipeline.EstimateGpuRawAsync(rgba, W, H); }
            catch (Exception ex)
            {
                var log = Graph.GraphExecutor.LastRunOpLog;
                int start = Math.Max(0, log.Count - 130);
                // No inner exception: UnitTestRunner shows InnerException when present, so drop it to surface
                // THIS message (with the elided-op trace) as the test error.
                throw new Exception($"[ELIDE] crashed: {ex.Message}\n"
                    + $"resolved={Graph.GraphExecutor.LastRunShapeInterpResolved} totalNodes~{session.NodeCount}\n"
                    + $"last {log.Count - start} ops:\n  " + string.Join("\n  ", log.GetRange(start, log.Count - start)));
            }

            int elided = Graph.GraphExecutor.LastRunOpLog.Count(s => s.Contains("~cpu-elided"));
            long elideOnMs = (long)Graph.GraphExecutor.LastRunTotalMs;
            float testRange = tR.maxD - tR.minD;
            float[] testDepth;
            using (tR.rawDepth) testDepth = await tR.rawDepth.CopyToHostAsync<float>(0, outSize);

            // Dispatch-elide only removes GPU dispatch of shape ops whose value the CPU interpreter already
            // computed - it must NOT change any real tensor math, so the depth map must match the reference
            // bit-for-bit (allow a tiny epsilon for any GPU float non-determinism).
            float maxAbsDiff = 0f;
            for (int i = 0; i < outSize; i++) maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(testDepth[i] - refDepth[i]));

            if (elided < 1)
                throw new Exception("[ELIDE] no shape ops were elided (elided=0) - the elide path was not exercised");
            if (testRange < 0.01f)
                throw new Exception($"[ELIDE] flat depth with elide on (range={testRange:F6}) - forward pass wrong");
            if (maxAbsDiff > 1e-3f)
                throw new Exception($"[ELIDE] elide CHANGED the depth map: maxAbsDiff={maxAbsDiff:E3} "
                    + $"(ref range={refRange:F6}, test range={testRange:F6}) - elide must be a pure orchestration no-op");
            // Return normally = TestResult.Success (UnitTestRunner: throw => Error, return => Success).
            Console.WriteLine($"[ELIDE] PASSED. elided={elided} maxAbsDiff={maxAbsDiff:E3} "
                + $"refRange={refRange:F6} testRange={testRange:F6} elideOnTotalMs={elideOnMs} ({tR.outW}x{tR.outH})");
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// CUDA-GRAPH CAPTURE probe for the DAv3 forward — the 270ms per-node CPU launch-prep lever (param-buffer
    /// alloc + H2D upload + cuLaunchKernel per dispatched op; ~68% of the warm 724ms elide-on forward). Records
    /// ONE warm forward into a CUDA graph and replays it with a single cuGraphLaunch, collapsing per-node host
    /// dispatch toward the ~40ms GPU-kernel floor. Mirrors the proven GGUF decode capture (Example 04):
    ///   • FusedAttentionKernel.UseStableCaptureSlots — captured attention nodes read a FIXED device pointer
    ///     the warm pass populated (the production ring hands a different slot each call → replay would bake a
    ///     stale slot). ForwardGeneration ticks per RunAsync → the per-forward slot counter auto-resets.
    ///   • CacheShapeReadbacks — the warm pass finalizes a stable readback cache so the capture pass performs
    ///     ZERO GPU-syncing readbacks (a sync mid-capture is illegal).
    ///   • GraphExecutor.SuppressDrains — no periodic drain / final synchronize / buffer-return aborts capture.
    ///   • 3-pass warm: (A drains-on) populate slots + prime JIT + finalize readback cache; (B drains-suppressed)
    ///     grow the pool to the no-drain working set so the capture pass allocates NOTHING (a cuMemAlloc
    ///     mid-capture crashes); (capture) record the forward.
    /// On capture failure this dumps LastRunOpLog so the exact breaking op (unstable alloc / freed param buffer /
    /// in-operator sync) is NAMED rather than guessed. GATE: replay output must be bit-identical to a fresh
    /// non-graph forward at the same input. CUDA-only (graph capture is a CUDA driver feature).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DA3Small_Pipeline_5D_CudaGraphCapture() => await RunTest(async accelerator =>
    {
        // WIP capture probe — env-gated (same pattern as GGUF_DECODE_GRAPH_PROBE). The DAv3 shape subgraph has
        // host-sync sites (CopyFromCPU of compile-time-constant shape values) still being made capture-clean;
        // until that lands, this crashes the CUDA lane, so it is OFF by default. Set DA3_CAPTURE_PROBE=1 to run.
        if (Environment.GetEnvironmentVariable("DA3_CAPTURE_PROBE") != "1")
            throw new UnsupportedTestException("DA3 CUDA-graph capture probe (WIP) — set DA3_CAPTURE_PROBE=1 to run");
        if (accelerator is not CudaAccelerator)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: CUDA graph capture is CUDA-only");
        if (!CudaStream.SupportsGraphCapture)
            throw new UnsupportedTestException("driver does not expose the CUDA graph API");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // ELIDE ON is the capturable regime: interpreter-resolved shape ops (Shape/Gather/Cast/... on shape
        // vectors) are NOT dispatched, so none of their host-side CopyFromCPU fast-paths run during capture.
        // The ONE remaining host-sync site is the on-demand materialization of an elided shape value consumed
        // by a GPU op — made capture-safe by skipping the H2D under SuppressDrains (the deterministic pool
        // buffer already holds the value the warm pass wrote). Non-elide would instead dispatch every shape op
        // and expose dozens of host-upload sites — the wrong regime for capture.
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = true;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");

            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);
            session.CacheShapeReadbacks = true;   // finalize a stable readback cache → capture pass syncs nothing

            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int v = (int)(x / (float)(W - 1) * 255f);
                    rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v;
                }

            // Stable preprocessed input (NOT `using` — a captured input-consuming node bakes this device
            // pointer, so it must not move across warm/capture/replay). Preprocess ONCE.
            var preprocess = new ImagePreprocessKernel(accelerator);
            using var rgbaBuf = accelerator.Allocate1D(rgba);
            var inputBuf = accelerator.Allocate1D<float>(3 * W * H);
            var capStream = (CudaStream)accelerator.CreateStream();
            try
            {
                preprocess.Forward(rgbaBuf.View, inputBuf.View, W, H, W, H);
                await accelerator.SynchronizeAsync();
                var inputTensor = new Tensor(inputBuf.View, new[] { 1, 1, 3, W, H }, session.InputNames[0]);
                var inputs = new Dictionary<string, Tensor> { [session.InputNames[0]] = inputTensor };

                float[] ReadOut(Tensor t) { var h = new float[t.ElementCount]; t.Data.CopyToCPU(h); return h; }

                // DIAGNOSTIC: crash-surviving per-node trace across ALL passes. A native access violation
                // leaves NO managed stack; after the crash this file's last line names the exact faulting op
                // and the pass marker names which pass it was in.
                var traceFile = @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\_mldump\capture-trace.txt";
                void Mark(string s) { try { System.IO.File.AppendAllText(traceFile, $"== {s} ==\n"); } catch { } }
                try { System.IO.File.WriteAllText(traceFile, ""); } catch { }
                Graph.GraphExecutor.CaptureTraceFile = traceFile;

                // REFERENCE: non-graph forward at this input (the correctness oracle).
                Mark("REFERENCE");
                var refOut = await session.RunAsync(inputs);
                await accelerator.SynchronizeAsync();
                float[] refData = ReadOut(refOut[session.OutputNames[0]]);
                int outCount = refData.Length;

                const int R = 10;
                double directMs = 0, graphMs = 0; float maxAbsDiff = -1f;
                try
                {
                    FusedAttentionKernel.UseStableCaptureSlots = true;
                    Graph.GraphExecutor.UseCaptureParamSlots = true;   // per-op params → stable capture slots
                    using (accelerator.WithDefaultStream(capStream))   // reroute *StreamKernel launches → capStream
                    {
                        // Warm A (drains ON): populate stable attention slots, prime JIT, finalize readback cache.
                        Mark("WARM A");
                        await session.RunAsync(inputs);
                        await accelerator.SynchronizeAsync();

                        // Warm B (drains ON = NORMAL): grows the buffer pool to the (higher) deferred-release peak
                        // AND fully returns every intermediate to its size-bucket at the final drain — so the pool
                        // is OVER-provisioned relative to the capture pass's lower immediate-return peak, and the
                        // capture pass finds a warm buffer in every bucket instead of a cuMemAlloc (illegal mid-
                        // capture — the node-58 [1,1,2] output Rent crash). Priming the pool matters more here than
                        // matching the capture footprint.
                        Mark("WARM B");
                        await session.RunAsync(inputs);
                        await accelerator.SynchronizeAsync();

                        // CAPTURE one forward (drains suppressed = capture-clean).
                        Mark("CAPTURE");
                        Graph.GraphExecutor.SuppressDrains = true;
                        capStream.BeginCapture(CudaStreamCaptureMode.Global);
                        var capOut = await session.RunAsync(inputs);
                        using var graph = capStream.EndCapture();
                        Graph.GraphExecutor.CaptureTraceFile = null;   // capture pass survived — stop tracing
                        Graph.GraphExecutor.SuppressDrains = false;
                        var capTensor = capOut[session.OutputNames[0]];

                        using var gexec = graph.Instantiate();
                        gexec.Upload(capStream);

                        // REPLAY once → compare to the non-graph reference (same kernels, same buffers).
                        gexec.Launch(capStream);
                        await capStream.SynchronizeAsync();
                        float[] replayData = ReadOut(capTensor);
                        maxAbsDiff = 0f;
                        for (int i = 0; i < outCount; i++)
                            maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(replayData[i] - refData[i]));

                        // TIME graph replays (pure cuGraphLaunch + GPU compute, ~zero host dispatch).
                        var gsw = System.Diagnostics.Stopwatch.StartNew();
                        for (int r = 0; r < R; r++) { gexec.Launch(capStream); await capStream.SynchronizeAsync(); }
                        gsw.Stop(); graphMs = gsw.Elapsed.TotalMilliseconds / R;
                    }

                    // TIME direct non-graph forwards (production ring path) at the same input — the apples-to-apples baseline.
                    FusedAttentionKernel.UseStableCaptureSlots = false;
                    var dsw = System.Diagnostics.Stopwatch.StartNew();
                    for (int r = 0; r < R; r++) { await session.RunAsync(inputs); await accelerator.SynchronizeAsync(); }
                    dsw.Stop(); directMs = dsw.Elapsed.TotalMilliseconds / R;
                }
                catch (Exception ex)
                {
                    var log = Graph.GraphExecutor.LastRunOpLog;
                    int start = Math.Max(0, log.Count - 120);
                    // No inner exception: UnitTestRunner surfaces THIS message (with the breaking-op trace).
                    throw new Exception($"[CAPTURE] failed: {ex.Message}\n"
                        + $"ops={log.Count} totalNodes~{session.NodeCount}\nlast {log.Count - start} ops:\n  "
                        + string.Join("\n  ", log.GetRange(start, log.Count - start)));
                }
                finally { FusedAttentionKernel.UseStableCaptureSlots = false; Graph.GraphExecutor.UseCaptureParamSlots = false; Graph.GraphExecutor.SuppressDrains = false; Graph.GraphExecutor.CaptureTraceFile = null; }

                if (maxAbsDiff > 1e-3f)
                    throw new Exception($"[CAPTURE] replay DIVERGED from non-graph forward: maxAbsDiff={maxAbsDiff:E3} (outCount={outCount})");
                var passSummary = $"[CAPTURE] PASSED. maxAbsDiff={maxAbsDiff:E3} directMs={directMs:F1} "
                    + $"graphMs={graphMs:F1} speedup={directMs / graphMs:F2}x (outCount={outCount})";
                Console.WriteLine(passSummary);
                // Surface the perf result (PMT discards subprocess stdout on a PASS).
                try { System.IO.File.WriteAllText(@"D:\users\tj\Projects\SpawnDev.ILGPU.ML\_mldump\capture-result.txt", passSummary); } catch { }
            }
            finally { inputBuf.Dispose(); try { capStream.Dispose(); } catch { } }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// PRODUCTIONIZED CUDA-graph capture via the reusable <see cref="CudaGraphCapture"/> API — the VIDEO /
    /// repeat-inference path. Capture the DAv3 forward ONCE at a fixed resolution (the "first frame"), then
    /// REPLAY it for every subsequent frame with a single cuGraphLaunch instead of re-running the ~2524-node
    /// loop. Validates the API is bit-identical to a non-graph forward and reports the replay speedup. CUDA-only;
    /// env-gated DA3_CAPTURE_PROBE=1 (shares the capture-probe gate).
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DA3Small_Pipeline_5D_CudaGraphReplay_Api() => await RunTest(async accelerator =>
    {
        if (Environment.GetEnvironmentVariable("DA3_CAPTURE_PROBE") != "1")
            throw new UnsupportedTestException("DA3 CUDA-graph replay API probe (WIP) — set DA3_CAPTURE_PROBE=1 to run");
        if (accelerator is not CudaAccelerator)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: CUDA graph capture is CUDA-only");
        if (!CudaStream.SupportsGraphCapture)
            throw new UnsupportedTestException("driver does not expose the CUDA graph API");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = true;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);

            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) { int v = (int)(x / (float)(W - 1) * 255f); rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v; }

            var preprocess = new ImagePreprocessKernel(accelerator);
            using var rgbaBuf = accelerator.Allocate1D(rgba);
            // STABLE input buffer — the capture reads THIS buffer; per "frame" we re-preprocess into it + replay.
            var inputBuf = accelerator.Allocate1D<float>(3 * W * H);
            try
            {
                preprocess.Forward(rgbaBuf.View, inputBuf.View, W, H, W, H);
                await accelerator.SynchronizeAsync();
                var inputTensor = new Tensor(inputBuf.View, new[] { 1, 1, 3, W, H }, session.InputNames[0]);
                var inputs = new Dictionary<string, Tensor> { [session.InputNames[0]] = inputTensor };
                float[] ReadOut(Tensor t) { var h = new float[t.ElementCount]; t.Data.CopyToCPU(h); return h; }

                // Non-graph reference (the correctness oracle).
                var refOut = await session.RunAsync(inputs);
                await accelerator.SynchronizeAsync();
                float[] refData = ReadOut(refOut[session.OutputNames[0]]);
                int outCount = refData.Length;

                // Capture ONCE (the "first frame").
                using var cap = await CudaGraphCapture.TryCaptureAsync(session, inputs);
                if (cap == null) throw new UnsupportedTestException("CudaGraphCapture.TryCaptureAsync returned null");

                // Replay (a "subsequent frame"): same input → bit-identical output.
                var replayOut = await cap.ReplayAsync(inputs);
                float maxAbsDiff = 0f;
                { float[] rd = ReadOut(replayOut[session.OutputNames[0]]); for (int i = 0; i < outCount; i++) maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(rd[i] - refData[i])); }
                if (maxAbsDiff > 1e-3f)
                    throw new Exception($"[REPLAY-API] replay DIVERGED from non-graph forward: maxAbsDiff={maxAbsDiff:E3} (outCount={outCount})");

                // Time N replays vs N direct forwards (the per-frame video win).
                const int R = 10;
                var gsw = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < R; r++) await cap.ReplayAsync(inputs);
                gsw.Stop(); double replayMs = gsw.Elapsed.TotalMilliseconds / R;
                var dsw = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < R; r++) { await session.RunAsync(inputs); await accelerator.SynchronizeAsync(); }
                dsw.Stop(); double directMs = dsw.Elapsed.TotalMilliseconds / R;

                var summary = $"[REPLAY-API] PASSED. maxAbsDiff={maxAbsDiff:E3} directMs={directMs:F1} "
                    + $"replayMs={replayMs:F1} speedup={directMs / replayMs:F2}x (outCount={outCount})";
                Console.WriteLine(summary);
                try { System.IO.File.WriteAllText(@"D:\users\tj\Projects\SpawnDev.ILGPU.ML\_mldump\capture-api-result.txt", summary); } catch { }
            }
            finally { inputBuf.Dispose(); }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// END-TO-END VIDEO path via <see cref="Pipelines.DepthEstimationPipeline"/> with EnableGraphCapture: the
    /// pipeline captures the forward on the first frame and REPLAYS it (single cuGraphLaunch) for every
    /// subsequent frame at the same resolution — preprocess → capture/replay → postprocess. Validates the
    /// pipeline-produced depth is bit-identical to the non-capture pipeline path and reports the per-frame
    /// speedup. CUDA-only; env-gated DA3_CAPTURE_PROBE=1.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task DA3Small_Pipeline_VideoCapture_Api() => await RunTest(async accelerator =>
    {
        if (Environment.GetEnvironmentVariable("DA3_CAPTURE_PROBE") != "1")
            throw new UnsupportedTestException("DA3 pipeline video-capture probe (WIP) — set DA3_CAPTURE_PROBE=1 to run");
        if (accelerator is not CudaAccelerator)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: CUDA graph capture is CUDA-only");
        if (!CudaStream.SupportsGraphCapture)
            throw new UnsupportedTestException("driver does not expose the CUDA graph API");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = true;
        Graph.GraphExecutor.ShapeInterpValidate = false;
        Graph.GraphExecutor.ShapeInterpElideDispatch = true;
        try
        {
            var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
            var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
                "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
            using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, 518, 518 } },
                externalData: extDataBytes);

            const int W = 518, H = 518;
            var rgba = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) { int v = (int)(x / (float)(W - 1) * 255f); rgba[y * W + x] = (255 << 24) | (v << 16) | (v << 8) | v; }

            using var pipeline = new Pipelines.DepthEstimationPipeline(session, accelerator);
            int outSize;
            float[] refDepth;

            // REFERENCE: non-capture pipeline frame.
            var refR = await pipeline.EstimateGpuRawAsync(rgba, W, H);
            outSize = refR.Width * refR.Height;
            using (refR.RawDepth) refDepth = await refR.RawDepth.CopyToHostAsync<float>(0, outSize);

            // CAPTURE path: frame 1 captures, frame 2 replays.
            pipeline.EnableGraphCapture = true;
            var f1 = await pipeline.EstimateGpuRawAsync(rgba, W, H);   // captures at this resolution
            f1.RawDepth.Dispose();
            var f2 = await pipeline.EstimateGpuRawAsync(rgba, W, H);   // replays
            float[] replayDepth;
            using (f2.RawDepth) replayDepth = await f2.RawDepth.CopyToHostAsync<float>(0, outSize);

            float maxAbsDiff = 0f;
            for (int i = 0; i < outSize; i++) maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(replayDepth[i] - refDepth[i]));
            if (maxAbsDiff > 1e-3f)
                throw new Exception($"[VIDEO-API] pipeline replay DIVERGED from non-capture pipeline: maxAbsDiff={maxAbsDiff:E3} (outSize={outSize})");

            // Time N replay frames (capture on) vs N non-capture frames.
            const int R = 8;
            var gsw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < R; r++) { var fr = await pipeline.EstimateGpuRawAsync(rgba, W, H); fr.RawDepth.Dispose(); }
            gsw.Stop(); double captureFrameMs = gsw.Elapsed.TotalMilliseconds / R;
            pipeline.EnableGraphCapture = false;
            var dsw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < R; r++) { var fr = await pipeline.EstimateGpuRawAsync(rgba, W, H); fr.RawDepth.Dispose(); }
            dsw.Stop(); double directFrameMs = dsw.Elapsed.TotalMilliseconds / R;

            var summary = $"[VIDEO-API] PASSED. maxAbsDiff={maxAbsDiff:E3} directFrameMs={directFrameMs:F1} "
                + $"captureFrameMs={captureFrameMs:F1} speedup={directFrameMs / captureFrameMs:F2}x (per full pipeline frame, outSize={outSize})";
            Console.WriteLine(summary);
            try { System.IO.File.WriteAllText(@"D:\users\tj\Projects\SpawnDev.ILGPU.ML\_mldump\capture-video-result.txt", summary); } catch { }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = false;
            Graph.GraphExecutor.ShapeInterpElideDispatch = false;
        }
    });

    /// <summary>
    /// Diagnostic: run only the first N nodes of DA3-Small with PerOpSync, capturing
    /// per-node Execute() wall-clock time. Surfaces which kernels are pathologically
    /// expensive to JIT-compile (the suspected dominant cost on Wasm/WebGPU first
    /// inference per Captain 2026-05-05: "kernel compiling is another bottleneck.
    /// unrolling and method inlining of large methods called a lot can substantially
    /// increase kernel compile time"). Bounds work via GraphExecutor.BreakAtNode
    /// so the test fits in a reasonable budget even on a slow backend.
    /// </summary>
    [TestMethod(Timeout = 120000, Category = "HeavyModel")]
    public async Task DA3Small_FirstNNodes_DiagnosticPerOpSync() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var onnxBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx");
        var extDataBytes = await InferenceSession.DownloadBytesChunkedAsync(http,
            "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data");
        long tDownload = sw.ElapsedMilliseconds;

        using var session = InferenceSession.CreateFromOnnx(accelerator, onnxBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["pixel_values"] = new[] { 1, 3, 224, 224 }
            },
            externalData: extDataBytes);
        long tCreate = sw.ElapsedMilliseconds - tDownload;
        int totalNodes = session.NodeCount;

        var inputData = new float[3 * 224 * 224];
        // Deterministic input - constant gray (zeros pre-normalization).
        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, 224, 224 });

        // Bound work to first 200 nodes. Default for this committed diagnostic.
        // Adjust BREAK_AT in working tree (don't commit) when bisecting deeper
        // into the model. Empirical points captured 2026-05-05:
        //   first 100: ~7.4s on Wasm (mostly shape ops + patch_embed)
        //   first 200: ~13s   (covers 1st transformer block; node 146 qkv MatMul = 463ms post-extraction, was 4611ms)
        //   first 800: ~82s   (rope blocks 4-5: 12+ Concat/Slice nodes at 700-2271ms each due to per-dispatch overhead, NOT compile time)
        const int BREAK_AT = 200;
        Graph.GraphExecutor.BreakAtNode = BREAK_AT;
        Graph.GraphExecutor.PerOpSync = true;
        Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
        Graph.GraphExecutor.LastRunOpLog.Clear();
        try
        {
            // Peek at next 5 op types past BREAK_AT for bisection (when test times out
            // mid-run, no per-node timings are emitted; this diagnostic at least tells
            // us which op type was about to run).
            var nextOps = new System.Text.StringBuilder();
            for (int peek = 0; peek < 5 && BREAK_AT + peek < totalNodes; peek++)
            {
                var (op, name) = session.GetNodeInfo(BREAK_AT + peek);
                nextOps.Append($"+{peek}=[{BREAK_AT + peek}]{op}({name}) ");
            }

            long tRunStart = sw.ElapsedMilliseconds;
            await session.RunAsync(new Dictionary<string, Tensor>
            {
                [session.InputNames[0]] = inputTensor
            });
            await accelerator.SynchronizeAsync();
            long tRun = sw.ElapsedMilliseconds - tRunStart;

            var timings = Graph.GraphExecutor.CapturedNodeTimingsMs;
            var ordered = timings.OrderBy(kv => kv.Key).ToList();
            var perNode = string.Join("|", ordered.Select(kv => $"{kv.Key}={kv.Value:F0}ms"));
            throw new Exception(
                $"PASSED first-{BREAK_AT}-nodes diagnostic. "
                + $"download={tDownload}ms create={tCreate}ms run={tRun}ms ({timings.Count}/{totalNodes} nodes); "
                + $"NEXT5: {nextOps}; "
                + $"per-node: {perNode}");
        }
        finally
        {
            Graph.GraphExecutor.BreakAtNode = null;
            Graph.GraphExecutor.PerOpSync = false;
            Graph.GraphExecutor.CapturedNodeTimingsMs = null;
        }
    });
}
