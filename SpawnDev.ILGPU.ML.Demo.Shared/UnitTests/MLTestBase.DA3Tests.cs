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

        // Generate test input: random normalized image [1, 3, 224, 224]
        var rng = new Random(42);
        int pixelCount = 3 * 224 * 224;
        var inputData = new float[pixelCount];
        for (int i = 0; i < pixelCount; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1);

        using var inputBuf = accelerator.Allocate1D(inputData);
        var inputTensor = new Tensor(inputBuf.View, new[] { 1, 3, 224, 224 });

        var outputs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            [session.InputNames[0]] = inputTensor
        });

        var output = outputs[session.OutputNames[0]];
        int elems = output.ElementCount;
        Console.WriteLine($"[DA3] Output shape: [{string.Join(",", output.Shape)}], elements: {elems}");

        if (elems < 100)
            throw new Exception($"DA3 output too small: {elems} elements (shape=[{string.Join(",", output.Shape)}])");

        // Read output to verify values are finite
        using var readBuf = accelerator.Allocate1D<float>(elems);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, elems), readBuf.View, elems, 1f);
        await accelerator.SynchronizeAsync();
        var actual = await readBuf.CopyToHostAsync<float>(0, elems);

        float absMax = 0, sum = 0;
        int nanCount = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (float.IsNaN(actual[i]) || float.IsInfinity(actual[i])) { nanCount++; continue; }
            absMax = MathF.Max(absMax, MathF.Abs(actual[i]));
            sum += actual[i];
        }
        float mean = sum / (actual.Length - nanCount + 1e-10f);

        Console.WriteLine($"[DA3] Depth output: absMax={absMax:F4}, mean={mean:F4}, NaN={nanCount}/{actual.Length}");

        if (nanCount > actual.Length / 10)
            throw new Exception($"DA3 output has {nanCount}/{actual.Length} NaN values");
        if (absMax == 0)
            throw new Exception("DA3 output is all zeros");

        Console.WriteLine($"[DA3] DA3-Small inference: PASS");
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
}
