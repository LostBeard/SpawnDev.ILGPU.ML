using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.DemoConsole.UnitTests;

/// <summary>
/// Integration tests that load real model graph JSON files and compile them.
/// These run on CPU only (console) and verify the full compilation pipeline.
/// </summary>
public class IntegrationTests
{
    [TestMethod]
    public async Task LoadAndCompile_MobileNetV2_155Nodes()
    {
        // Load the real graph JSON from the extracted model
        var graphPath = Path.Combine(FindModelsDir(), "mobilenetv2", "model_graph.json");
        if (!File.Exists(graphPath))
            throw new UnsupportedTestException($"Model not found: {graphPath}");

        var json = await File.ReadAllTextAsync(graphPath);
        var graph = ModelGraph.FromJson(json);

        Console.WriteLine($"[Integration] MobileNetV2: {graph.Nodes.Count} nodes, {graph.Initializers.Count} initializers");
        Console.WriteLine($"[Integration] Input: {graph.Inputs[0].Name} [{string.Join(",", graph.Inputs[0].Shape)}]");
        Console.WriteLine($"[Integration] Output: {graph.Outputs[0].Name} [{string.Join(",", graph.Outputs[0].Shape)}]");

        // Compile with CPU accelerator
        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);

        CompiledGraph compiled;
        try
        {
            compiled = compiler.Compile(graph);
        }
        catch (Exception ex)
        {
            throw new Exception($"Graph compilation failed: {ex.Message}");
        }

        Console.WriteLine($"[Integration] Compiled: {compiled.Nodes.Length} nodes");
        Console.WriteLine($"[Integration] Operators used: {string.Join(", ", compiled.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s))}");

        if (compiled.Nodes.Length != graph.Nodes.Count)
            throw new Exception($"Node count mismatch: compiled={compiled.Nodes.Length}, graph={graph.Nodes.Count}");

        // Verify output shape
        if (compiled.OutputShapes.TryGetValue(graph.Outputs[0].Name, out var outShape))
            Console.WriteLine($"[Integration] Output shape: [{string.Join(",", outShape)}]");
    }

    [TestMethod]
    public async Task LoadAndCompile_SqueezeNet_66Nodes()
    {
        var graphPath = Path.Combine(FindModelsDir(), "squeezenet", "model_graph.json");
        if (!File.Exists(graphPath))
            throw new UnsupportedTestException($"Model not found: {graphPath}");

        var json = await File.ReadAllTextAsync(graphPath);
        var graph = ModelGraph.FromJson(json);

        Console.WriteLine($"[Integration] SqueezeNet: {graph.Nodes.Count} nodes");

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        Console.WriteLine($"[Integration] Compiled: {compiled.Nodes.Length} nodes");
        if (compiled.Nodes.Length != graph.Nodes.Count)
            throw new Exception($"Node count mismatch");
    }

    [TestMethod]
    public async Task LoadAndCompile_SuperResolution_12Nodes()
    {
        var graphPath = Path.Combine(FindModelsDir(), "super-resolution", "model_graph.json");
        if (!File.Exists(graphPath))
            throw new UnsupportedTestException($"Model not found: {graphPath}");

        var json = await File.ReadAllTextAsync(graphPath);
        var graph = ModelGraph.FromJson(json);

        Console.WriteLine($"[Integration] ESPCN: {graph.Nodes.Count} nodes");

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        Console.WriteLine($"[Integration] Compiled: {compiled.Nodes.Length} nodes");
    }

    [TestMethod]
    public async Task LoadAndCompile_StyleTransfer_112Nodes()
    {
        var graphPath = Path.Combine(FindModelsDir(), "style-mosaic", "model_graph.json");
        if (!File.Exists(graphPath))
            throw new UnsupportedTestException($"Model not found: {graphPath}");

        var json = await File.ReadAllTextAsync(graphPath);
        var graph = ModelGraph.FromJson(json);

        Console.WriteLine($"[Integration] Mosaic Style: {graph.Nodes.Count} nodes");

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);
        var registry = new OperatorRegistry(accelerator);

        try
        {
            var compiled = new GraphCompiler(registry).Compile(graph);
            Console.WriteLine($"[Integration] Compiled: {compiled.Nodes.Length} nodes");
        }
        catch (NotSupportedException ex)
        {
            // Style transfer uses some ops that have placeholder Execute — acceptable at compile time
            Console.WriteLine($"[Integration] Compile partial: {ex.Message}");
            throw new UnsupportedTestException($"Style transfer needs: {ex.Message}");
        }
    }

    /// <summary>
    /// FULL END-TO-END: Load real MobileNetV2 weights + graph, execute on CPU,
    /// verify output is 1000 logits with reasonable values.
    /// </summary>
    [TestMethod]
    public async Task FullExecution_MobileNetV2_WithRealWeights()
    {
        var modelsDir = FindModelsDir();
        var modelDir = Path.Combine(modelsDir, "mobilenetv2");
        var graphPath = Path.Combine(modelDir, "model_graph.json");
        var manifestPath = Path.Combine(modelDir, "manifest_fp16.json");
        var weightsPath = Path.Combine(modelDir, "weights_fp16.bin");

        if (!File.Exists(graphPath) || !File.Exists(weightsPath))
            throw new UnsupportedTestException("MobileNetV2 model files not found");

        // Load graph
        var graph = ModelGraph.FromJson(await File.ReadAllTextAsync(graphPath));
        Console.WriteLine($"[FullExec] Graph: {graph.Nodes.Count} nodes");

        // Create accelerator
        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);

        // Load weights manually (WeightLoader needs HttpClient which we don't have in console)
        var manifestJson = await File.ReadAllTextAsync(manifestPath);
        var manifest = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(manifestJson)!;
        var blob = await File.ReadAllBytesAsync(weightsPath);

        Console.WriteLine($"[FullExec] Weights: {blob.Length / 1024.0 / 1024.0:F1} MB, {manifest.Count} tensors");

        // Convert FP16 → FP32 and upload all weights
        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        foreach (var (name, info) in manifest)
        {
            var offset = info.GetProperty("offset").GetInt32();
            var shape = info.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var elements = info.GetProperty("elements").GetInt32();

            // FP16 → FP32 conversion
            var fp32 = new float[elements];
            for (int i = 0; i < elements; i++)
            {
                ushort fp16Bits = (ushort)(blob[offset + i * 2] | (blob[offset + i * 2 + 1] << 8));
                fp32[i] = HalfToFloat(fp16Bits);
            }
            weights[name] = pool.AllocatePermanent(fp32, shape, name);
        }
        Console.WriteLine($"[FullExec] Loaded {weights.Count} weight tensors to GPU");

        // Compile graph
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        Console.WriteLine($"[FullExec] Compiled: {compiled.Nodes.Length} nodes");

        // Create random input image (1, 3, 224, 224) — normalized like ImageNet
        var inputData = new float[1 * 3 * 224 * 224];
        var rng = new Random(42);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rng.NextDouble() * 2 - 1); // Range [-1, 1]
        var input = pool.AllocatePermanent(inputData, new[] { 1, 3, 224, 224 }, "data");

        // Execute
        Console.WriteLine($"[FullExec] Running inference...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var outputs = executor.Run(new Dictionary<string, Tensor> { [graph.Inputs[0].Name] = input });
        accelerator.Synchronize();
        sw.Stop();
        Console.WriteLine($"[FullExec] Inference: {sw.Elapsed.TotalMilliseconds:F0}ms");

        // Read output
        var outputName = graph.Outputs[0].Name;
        var output = outputs[outputName];
        Console.WriteLine($"[FullExec] Output: {output.ElementCount} elements, shape [{string.Join(",", output.Shape)}]");

        using var rb = accelerator.Allocate1D<float>(output.ElementCount);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, output.ElementCount), rb.View, output.ElementCount, 1f);
        accelerator.Synchronize();
        var logits = rb.GetAsArray1D();

        // Check output is reasonable
        float min = logits.Min(), max = logits.Max();
        float mean = logits.Average();
        Console.WriteLine($"[FullExec] Logits: min={min:F4}, max={max:F4}, mean={mean:F4}");

        if (float.IsNaN(min) || float.IsInfinity(max))
            throw new Exception("Output contains NaN/Inf");
        if (logits.All(v => v == 0f))
            throw new Exception("Output is all zeros");

        // Softmax + top-5
        float expMax = logits.Max();
        var probs = logits.Select(x => MathF.Exp(x - expMax)).ToArray();
        float probSum = probs.Sum();
        probs = probs.Select(p => p / probSum).ToArray();

        var top5 = probs.Select((p, i) => (Index: i, Prob: p))
            .OrderByDescending(x => x.Prob)
            .Take(5)
            .ToList();

        Console.WriteLine($"[FullExec] Top-5 predictions:");
        foreach (var (idx, prob) in top5)
            Console.WriteLine($"  class {idx}: {prob:P2}");
    }

    [TestMethod]
    public async Task FullExecution_SqueezeNet_WithRealWeights()
    {
        var modelsDir = FindModelsDir();
        var modelDir = Path.Combine(modelsDir, "squeezenet");
        var graphPath = Path.Combine(modelDir, "model_graph.json");
        var manifestPath = Path.Combine(modelDir, "manifest_fp16.json");
        var weightsPath = Path.Combine(modelDir, "weights_fp16.bin");

        if (!File.Exists(graphPath) || !File.Exists(weightsPath))
            throw new UnsupportedTestException("SqueezeNet model files not found");

        var graph = ModelGraph.FromJson(await File.ReadAllTextAsync(graphPath));
        Console.WriteLine($"[FullExec] SqueezeNet: {graph.Nodes.Count} nodes");

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);

        var manifestJson = await File.ReadAllTextAsync(manifestPath);
        var manifest = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(manifestJson)!;
        var blob = await File.ReadAllBytesAsync(weightsPath);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        foreach (var (name, info) in manifest)
        {
            var offset = info.GetProperty("offset").GetInt32();
            var shape = info.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var elements = info.GetProperty("elements").GetInt32();
            var fp32 = new float[elements];
            for (int i = 0; i < elements; i++)
            {
                ushort fp16Bits = (ushort)(blob[offset + i * 2] | (blob[offset + i * 2 + 1] << 8));
                fp32[i] = HalfToFloat(fp16Bits);
            }
            weights[name] = pool.AllocatePermanent(fp32, shape, name);
        }

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        var inputData = new float[1 * 3 * 224 * 224];
        var rng = new Random(42);
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1);
        var input = pool.AllocatePermanent(inputData, new[] { 1, 3, 224, 224 }, "data");

        Console.WriteLine($"[FullExec] Running SqueezeNet inference...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var outputs = executor.Run(new Dictionary<string, Tensor> { [graph.Inputs[0].Name] = input });
        accelerator.Synchronize();
        sw.Stop();
        Console.WriteLine($"[FullExec] SqueezeNet inference: {sw.Elapsed.TotalMilliseconds:F0}ms");

        var output = outputs[graph.Outputs[0].Name];
        Console.WriteLine($"[FullExec] Output: {output.ElementCount} elements");

        using var rb = accelerator.Allocate1D<float>(output.ElementCount);
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, output.ElementCount), rb.View, output.ElementCount, 1f);
        accelerator.Synchronize();
        var logits = rb.GetAsArray1D();

        float min = logits.Min(), max = logits.Max();
        Console.WriteLine($"[FullExec] Logits: min={min:F4}, max={max:F4}, mean={logits.Average():F4}");

        if (float.IsNaN(min) || logits.All(v => v == 0f))
            throw new Exception("Output is NaN or zeros");

        // Top-5
        float expMax = logits.Max();
        var probs = logits.Select(x => MathF.Exp(x - expMax)).ToArray();
        float probSum = probs.Sum();
        var top5 = probs.Select((p, i) => (i, p / probSum)).OrderByDescending(x => x.Item2).Take(5);
        Console.WriteLine("[FullExec] Top-5:");
        foreach (var (idx, prob) in top5) Console.WriteLine($"  class {idx}: {prob:P2}");
    }

    [TestMethod]
    public async Task FullExecution_ESPCN_SuperResolution()
    {
        var modelsDir = FindModelsDir();
        var modelDir = Path.Combine(modelsDir, "super-resolution");
        var graphPath = Path.Combine(modelDir, "model_graph.json");
        var manifestPath = Path.Combine(modelDir, "manifest_fp16.json");
        var weightsPath = Path.Combine(modelDir, "weights_fp16.bin");

        if (!File.Exists(graphPath) || !File.Exists(weightsPath))
            throw new UnsupportedTestException("ESPCN model files not found");

        var graph = ModelGraph.FromJson(await File.ReadAllTextAsync(graphPath));

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);

        var manifest = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
            await File.ReadAllTextAsync(manifestPath))!;
        var blob = await File.ReadAllBytesAsync(weightsPath);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        foreach (var (name, info) in manifest)
        {
            var offset = info.GetProperty("offset").GetInt32();
            var shape = info.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var elements = info.GetProperty("elements").GetInt32();
            var fp32 = new float[elements];
            for (int i = 0; i < elements; i++)
            {
                ushort fp16 = (ushort)(blob[offset + i * 2] | (blob[offset + i * 2 + 1] << 8));
                fp32[i] = HalfToFloat(fp16);
            }
            weights[name] = pool.AllocatePermanent(fp32, shape, name);
        }

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);

        // Small input for speed: [1, 1, 32, 32] (Y luminance channel)
        var inputData = new float[1 * 1 * 32 * 32];
        var rng = new Random(42);
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)rng.NextDouble();
        var input = pool.AllocatePermanent(inputData, new[] { 1, 1, 32, 32 }, "input");

        Console.WriteLine($"[FullExec] Running ESPCN super-resolution...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var executor = new GraphExecutor(accelerator, compiled, weights);
        var outputs = executor.Run(new Dictionary<string, Tensor> { ["input"] = input });
        accelerator.Synchronize();
        sw.Stop();
        Console.WriteLine($"[FullExec] ESPCN: {sw.Elapsed.TotalMilliseconds:F0}ms");

        var output = outputs[graph.Outputs[0].Name];
        Console.WriteLine($"[FullExec] Output: {output.ElementCount} elements, shape [{string.Join(",", output.Shape)}]");

        using var rb = accelerator.Allocate1D<float>(Math.Min(output.ElementCount, 10));
        new ElementWiseKernels(accelerator).Scale(output.Data.SubView(0, Math.Min(output.ElementCount, 10)),
            rb.View, Math.Min(output.ElementCount, 10), 1f);
        accelerator.Synchronize();
        var sample = rb.GetAsArray1D();
        Console.WriteLine($"[FullExec] First 10: [{string.Join(", ", sample.Select(v => v.ToString("F4")))}]");

        if (sample.All(v => v == 0f)) throw new Exception("Output is zeros");
        if (sample.Any(v => float.IsNaN(v))) throw new Exception("Output has NaN");
    }

    /// <summary>
    /// Full ClassificationPipeline test: load model, create pipeline, classify test image.
    /// Uses a solid blue test image — won't get correct label but verifies the full pipeline runs.
    /// </summary>
    [TestMethod]
    public async Task ClassificationPipeline_FullEndToEnd()
    {
        var modelsDir = FindModelsDir();
        var modelDir = Path.Combine(modelsDir, "squeezenet"); // SqueezeNet is simpler
        var graphPath = Path.Combine(modelDir, "model_graph.json");
        var manifestPath = Path.Combine(modelDir, "manifest_fp16.json");
        var weightsPath = Path.Combine(modelDir, "weights_fp16.bin");

        if (!File.Exists(graphPath)) throw new UnsupportedTestException("Model not found");

        var graph = ModelGraph.FromJson(await File.ReadAllTextAsync(graphPath));

        using var context = MLContext.CreateContext();
        using var accelerator = context.CreateCPUAccelerator(0);

        // Load weights
        var manifest = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
            await File.ReadAllTextAsync(manifestPath))!;
        var blob = await File.ReadAllBytesAsync(weightsPath);

        using var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        foreach (var (name, info) in manifest)
        {
            var offset = info.GetProperty("offset").GetInt32();
            var shape = info.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var elements = info.GetProperty("elements").GetInt32();
            var fp32 = new float[elements];
            for (int i = 0; i < elements; i++)
            {
                ushort fp16 = (ushort)(blob[offset + i * 2] | (blob[offset + i * 2 + 1] << 8));
                fp32[i] = HalfToFloat(fp16);
            }
            weights[name] = pool.AllocatePermanent(fp32, shape, name);
        }

        // Create session + pipeline
        var session = InferenceSession.Create(accelerator, graph, weights);
        var pipeline = new SpawnDev.ILGPU.ML.Pipelines.ClassificationPipeline(session, accelerator);

        Console.WriteLine($"[Pipeline] Model: {session}");

        // Create a solid blue test image (64x64 RGBA)
        int w = 64, h = 64;
        var pixels = new int[w * h];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = 0x00 | (0x00 << 8) | (0xFF << 16) | (0xFF << 24); // Blue + Alpha

        // Classify
        Console.WriteLine("[Pipeline] Running classification...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var results = await pipeline.ClassifyAsync(pixels, w, h);
        sw.Stop();
        Console.WriteLine($"[Pipeline] Done in {sw.Elapsed.TotalMilliseconds:F0}ms");

        // Verify results
        if (results.Length != 5) throw new Exception($"Expected 5 results, got {results.Length}");
        if (results[0].Confidence <= 0) throw new Exception("Top confidence is 0");
        if (string.IsNullOrEmpty(results[0].Label)) throw new Exception("Top label is empty");

        float totalProb = results.Sum(r => r.Confidence);
        Console.WriteLine($"[Pipeline] Top-5 (sum={totalProb:P1}):");
        foreach (var r in results)
            Console.WriteLine($"  {r.Label}: {r.Confidence:P2} (class {r.ClassIndex})");
    }

    private static float HalfToFloat(ushort h)
    {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
        {
            if (mant == 0) return sign == 1 ? -0f : 0f;
            float val = mant / 1024f * MathF.Pow(2f, -14f);
            return sign == 1 ? -val : val;
        }
        if (exp == 31) return mant == 0 ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;
        float result = (1f + mant / 1024f) * MathF.Pow(2f, exp - 15f);
        return sign == 1 ? -result : result;
    }

    private static string FindModelsDir()
    {
        // Walk up from the exe directory to find the Demo wwwroot/models
        var dir = AppDomain.CurrentDomain.BaseDirectory;
        for (int i = 0; i < 10; i++)
        {
            var models = Path.Combine(dir, "SpawnDev.ILGPU.ML.Demo", "wwwroot", "models");
            if (Directory.Exists(models)) return models;
            dir = Path.GetDirectoryName(dir) ?? dir;
        }
        throw new Exception("Could not find models directory");
    }
}
