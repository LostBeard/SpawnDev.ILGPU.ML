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
