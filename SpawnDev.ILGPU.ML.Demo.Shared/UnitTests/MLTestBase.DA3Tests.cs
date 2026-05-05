using ILGPU;
using ILGPU.Runtime;
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

    [TestMethod(Timeout = 300000)]
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

    [TestMethod(Timeout = 300000)]
    public async Task DA3Small_Inference_ProducesDepth() => await RunTest(async accelerator =>
    {
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
        throw new Exception($"PASSED. timing: download={tDownload}ms create={tCreate}ms run={tRun}ms verify={tVerify}ms total={sw.ElapsedMilliseconds}ms; output absMax={absMax:F4} meanAbs={meanAbs:F4} NaN={nanCount}/{elems}");
    });

    [TestMethod(Timeout = 300000)]
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
    /// Diagnostic: run only the first N nodes of DA3-Small with PerOpSync, capturing
    /// per-node Execute() wall-clock time. Surfaces which kernels are pathologically
    /// expensive to JIT-compile (the suspected dominant cost on Wasm/WebGPU first
    /// inference per Captain 2026-05-05: "kernel compiling is another bottleneck.
    /// unrolling and method inlining of large methods called a lot can substantially
    /// increase kernel compile time"). Bounds work via GraphExecutor.BreakAtNode
    /// so the test fits in a reasonable budget even on a slow backend.
    /// </summary>
    [TestMethod(Timeout = 120000)]
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

        // Bound work to first 8 nodes. Compile cost dominates on first dispatch;
        // 8 nodes is enough to surface several distinct kernel types without burning
        // the full graph on a slow backend.
        const int BREAK_AT = 8;
        Graph.GraphExecutor.BreakAtNode = BREAK_AT;
        Graph.GraphExecutor.PerOpSync = true;
        Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
        Graph.GraphExecutor.LastRunOpLog.Clear();
        try
        {
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
