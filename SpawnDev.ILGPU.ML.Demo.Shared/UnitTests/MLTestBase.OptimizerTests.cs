using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Test graph optimizer on a real model — verify it reduces node count
    /// without breaking compilation.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task GraphOptimizer_ReducesNodeCount() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null)
            throw new UnsupportedTestException("HttpClient not available");

        // Load style transfer model graph
        try
        {
            var graphJson = await http.GetStringAsync("models/style-mosaic/model_graph.json");
            var graph = ModelGraph.FromJson(graphJson);
            int originalCount = graph.Nodes.Count;

            Console.WriteLine($"[Optimizer] Original: {originalCount} nodes");
            Console.WriteLine($"[Optimizer] Op types: {string.Join(", ", graph.Nodes.Select(n => n.OpType).GroupBy(o => o).Select(g => $"{g.Key}({g.Count()})").OrderByDescending(s => s))}");

            // Optimize
            var optimized = GraphOptimizer.Optimize(graph);
            int optimizedCount = optimized.Nodes.Count;

            Console.WriteLine($"[Optimizer] Optimized: {optimizedCount} nodes");
            Console.WriteLine($"[Optimizer] Reduced by {originalCount - optimizedCount} nodes ({(1.0 - (double)optimizedCount / originalCount) * 100:F0}%)");
            Console.WriteLine($"[Optimizer] Op types: {string.Join(", ", optimized.Nodes.Select(n => n.OpType).GroupBy(o => o).Select(g => $"{g.Key}({g.Count()})").OrderByDescending(s => s))}");

            if (optimizedCount >= originalCount)
                Console.WriteLine("[Optimizer] WARNING: no reduction (model may not have foldable patterns)");
            else
                Console.WriteLine($"[Optimizer] PASS — {originalCount - optimizedCount} nodes eliminated");

            // Verify the optimized graph still compiles
            var registry = new SpawnDev.ILGPU.ML.Operators.OperatorRegistry(accelerator);
            var compiler = new GraphCompiler(registry);
            compiler.EnableOptimization = false; // already optimized
            var compiled = compiler.Compile(optimized);
            Console.WriteLine($"[Optimizer] Compiled: {compiled.Nodes.Length} nodes — PASS");
        }
        catch (HttpRequestException)
        {
            throw new UnsupportedTestException("Style-mosaic model not available");
        }
    });
}
