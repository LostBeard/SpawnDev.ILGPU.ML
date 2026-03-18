using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML;

/// <summary>
/// High-level API for loading and running ONNX models on GPU.
///
/// Usage (from pre-extracted weights + graph JSON):
///   var session = await InferenceSession.CreateAsync(accelerator, httpClient, "models/my_model");
///   var output = session.Run("input", inputTensor);
///
/// The model directory should contain:
///   model_graph.json — Graph structure (nodes, inputs, outputs)
///   weights_fp16.bin — FP16 weight blob
///   manifest_fp16.json — Weight tensor manifest
/// </summary>
public class InferenceSession : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly OperatorRegistry _registry;
    private readonly GraphExecutor _executor;
    private readonly CompiledGraph _compiled;
    private readonly BufferPool _pool;
    private readonly Dictionary<string, Tensor> _weights;

    /// <summary>Model input names and shapes.</summary>
    public string[] InputNames => _compiled.InputNames;
    public Dictionary<string, int[]> InputShapes => _compiled.InputShapes;

    /// <summary>Model output names and shapes.</summary>
    public string[] OutputNames => _compiled.OutputNames;
    public Dictionary<string, int[]> OutputShapes => _compiled.OutputShapes;

    /// <summary>Number of supported ONNX operators.</summary>
    public int SupportedOpCount => _registry.SupportedOps.Count;

    private InferenceSession(Accelerator accelerator, OperatorRegistry registry,
        CompiledGraph compiled, GraphExecutor executor, BufferPool pool,
        Dictionary<string, Tensor> weights)
    {
        _accelerator = accelerator;
        _registry = registry;
        _compiled = compiled;
        _executor = executor;
        _pool = pool;
        _weights = weights;
    }

    /// <summary>
    /// Create an InferenceSession from pre-extracted model files.
    /// Loads graph JSON + weight manifest + weight blob from the given base path.
    /// </summary>
    public static async Task<InferenceSession> CreateAsync(
        Accelerator accelerator, HttpClient http, string basePath)
    {
        // Load graph
        var graphJson = await http.GetStringAsync($"{basePath}/model_graph.json");
        var modelGraph = ModelGraph.FromJson(graphJson);

        // Load weights via existing WeightLoader
        var weightLoader = new WeightLoader(accelerator, http);
        await weightLoader.LoadAsync(basePath);

        // Create operator registry
        var registry = new OperatorRegistry(accelerator);

        // Compile graph
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(modelGraph);

        // Create weight tensors from WeightLoader
        var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        foreach (var name in compiled.InitializerNames)
        {
            var view = weightLoader.TryGetView(name);
            if (view != null && weightLoader.Shapes.TryGetValue(name, out var shape))
            {
                weights[name] = new Tensor(view.Value, shape, name);
            }
        }

        // Create executor
        var executor = new GraphExecutor(accelerator, compiled, weights);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights);
    }

    /// <summary>
    /// Create an InferenceSession from a ModelGraph and pre-loaded weight tensors.
    /// For programmatic use without HTTP loading.
    /// </summary>
    public static InferenceSession Create(
        Accelerator accelerator, ModelGraph graph, Dictionary<string, Tensor> weights)
    {
        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new BufferPool(accelerator);
        var executor = new GraphExecutor(accelerator, compiled, weights);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights);
    }

    /// <summary>Run inference with named input tensors. Returns named output tensors.</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => _executor.Run(inputs);

    /// <summary>Run inference with a single input. Returns the first output tensor.</summary>
    public Tensor Run(string inputName, Tensor input)
    {
        var outputs = _executor.Run(new Dictionary<string, Tensor> { [inputName] = input });
        return outputs.Values.First();
    }

    public void Dispose()
    {
        _executor.Dispose();
        _pool.Dispose();
    }
}
