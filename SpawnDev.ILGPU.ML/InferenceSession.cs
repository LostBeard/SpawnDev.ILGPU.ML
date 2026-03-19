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

    /// <summary>Enable diagnostic logging to Console.</summary>
    public static bool VerboseLogging { get; set; }

    /// <summary>Model input names and shapes.</summary>
    public string[] InputNames => _compiled.InputNames;
    public Dictionary<string, int[]> InputShapes => _compiled.InputShapes;

    /// <summary>Model output names and shapes.</summary>
    public string[] OutputNames => _compiled.OutputNames;
    public Dictionary<string, int[]> OutputShapes => _compiled.OutputShapes;

    /// <summary>Number of supported ONNX operators.</summary>
    public int SupportedOpCount => _registry.SupportedOps.Count;

    /// <summary>Number of nodes in the compiled graph.</summary>
    public int NodeCount => _compiled.Nodes.Length;

    /// <summary>Number of weight tensors loaded.</summary>
    public int WeightCount => _weights.Count;

    /// <summary>Distinct operator types used in this model.</summary>
    public string[] OperatorTypes => _compiled.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToArray();

    /// <summary>Model name (from graph metadata).</summary>
    public string ModelName { get; private set; } = "";

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
    /// <param name="onProgress">Optional progress callback: (stage, percent) where stage is
    /// "graph", "weights", "compile", "ready" and percent is 0-100.</param>
    public static async Task<InferenceSession> CreateAsync(
        Accelerator accelerator, HttpClient http, string basePath,
        Action<string, int>? onProgress = null)
    {
        // Load graph
        onProgress?.Invoke("graph", 0);
        var graphJson = await http.GetStringAsync($"{basePath}/model_graph.json");
        var modelGraph = ModelGraph.FromJson(graphJson);
        onProgress?.Invoke("graph", 100);

        // Load weights
        onProgress?.Invoke("weights", 0);
        var weightLoader = new WeightLoader(accelerator, http);
        await weightLoader.LoadAsync(basePath);
        onProgress?.Invoke("weights", 100);

        // Extract small constant values for shape inference AND runtime operator use.
        // Use ONE shared read buffer to avoid allocating hundreds of tiny GPU buffers.
        modelGraph.ConstantData = new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        {
            // Find max small tensor size, allocate one shared readback buffer
            int maxSmallElems = 0;
            foreach (var (name, shape) in modelGraph.Initializers)
            {
                int elems = shape.Aggregate(1, (a, b) => a * b);
                if (elems > 0 && elems <= 64 && weightLoader.TryGetView(name) != null)
                    maxSmallElems = Math.Max(maxSmallElems, elems);
            }

            if (maxSmallElems > 0)
            {
                using var readBuf = accelerator.Allocate1D<float>(maxSmallElems);
                foreach (var (name, shape) in modelGraph.Initializers)
                {
                    int elems = shape.Aggregate(1, (a, b) => a * b);
                    if (elems > 0 && elems <= 64)
                    {
                        var view = weightLoader.TryGetView(name);
                        if (view != null)
                        {
                            readBuf.View.SubView(0, elems).CopyFrom(view.Value.SubView(0, elems));
                            await accelerator.SynchronizeAsync();
                            var hostBuf = await readBuf.CopyToHostAsync<float>(0, elems);
                            constantFloatValues[name] = hostBuf;
                            if (elems <= 16)
                                modelGraph.ConstantData[name] = hostBuf.Select(v => (int)v).ToArray();
                        }
                    }
                }
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(modelGraph);
        onProgress?.Invoke("compile", 100);

        // Create weight tensors from WeightLoader
        // Include both compiled initializer names AND all graph initializers
        // (Constant node outputs are stored as initializers by the extraction script)
        var pool = new BufferPool(accelerator);
        var weights = new Dictionary<string, Tensor>();
        var allInitNames = new HashSet<string>(compiled.InitializerNames);
        foreach (var name in modelGraph.Initializers.Keys)
            allInitNames.Add(name);

        int loadedCount = 0;
        foreach (var name in allInitNames)
        {
            var view = weightLoader.TryGetView(name);
            if (view != null && weightLoader.Shapes.TryGetValue(name, out var shape))
            {
                // Copy each weight into its OWN buffer. WeightLoader uses a single
                // shared buffer with SubViews, but WebGPU doesn't allow binding the
                // same GPUBuffer to multiple storage slots in one kernel dispatch.
                // Without separate buffers, Conv2D (weight + bias from same buffer)
                // produces silent zeros on WebGPU.
                int count = Tensors.TensorHelpers.ElementCount(shape);
                var ownBuf = accelerator.Allocate1D<float>(count);
                // Use Scale kernel (GPU→GPU copy) instead of CopyTo (sync, fails on WebGPU)
                registry.ElementWise.Scale(view.Value.SubView(0, count), ownBuf.View, count, 1f);
                weights[name] = new Tensor(ownBuf.View, shape, name);
                loadedCount++;
            }
        }

        // Log weight loading stats
        int missingCount = allInitNames.Count - loadedCount;
        if (missingCount > 0)
        {
            var missing = allInitNames.Where(n => !weights.ContainsKey(n)).Take(5);
            Console.WriteLine($"[InferenceSession] WARNING: {missingCount} initializers not found in weights. First few: {string.Join(", ", missing)}");
        }
        Console.WriteLine($"[InferenceSession] Loaded {loadedCount}/{allInitNames.Count} weights, {compiled.Nodes.Length} nodes compiled");

        // Diagnostic: check first weight tensor has non-zero values
        if (weights.Count > 0)
        {
            var firstWeight = weights.Values.First();
            Console.WriteLine($"[InferenceSession] First weight '{firstWeight.Name}': shape=[{string.Join(",", firstWeight.Shape)}], elements={firstWeight.ElementCount}");
        }

        // Create executor with pre-read constant values (avoids GPU→CPU readback during inference)
        var executor = new GraphExecutor(accelerator, compiled, weights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights)
        {
            ModelName = modelGraph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from a ModelGraph and pre-loaded weight tensors.
    /// For programmatic use without HTTP loading.
    /// </summary>
    public static InferenceSession Create(
        Accelerator accelerator, ModelGraph graph, Dictionary<string, Tensor> weights)
    {
        // Extract small constant values for shape inference (Reshape targets, etc.)
        // Only do sync CPU readback on desktop backends — browser backends require async.
        // Browser callers should use CreateAsync or CreateFromOnnxAsync instead.
        var constantFloatValues = new Dictionary<string, float[]>();
        bool canSyncCopy = accelerator.AcceleratorType != AcceleratorType.WebGPU
                        && accelerator.AcceleratorType != AcceleratorType.WebGL
                        && accelerator.AcceleratorType != AcceleratorType.Wasm;
        if (graph.ConstantData == null)
        {
            graph.ConstantData = new Dictionary<string, int[]>();
        }
        if (canSyncCopy)
        {
            foreach (var (name, shape) in graph.Initializers)
            {
                int elems = shape.Aggregate(1, (a, b) => a * b);
                if (elems > 0 && elems <= 64 && weights.TryGetValue(name, out var tensor))
                {
                    var hostBuf = new float[elems];
                    tensor.Data.SubView(0, elems).CopyToCPU(hostBuf);
                    accelerator.Synchronize();
                    constantFloatValues[name] = hostBuf;
                    if (elems <= 16)
                        graph.ConstantData[name] = hostBuf.Select(v => (int)v).ToArray();
                }
            }
        }

        var registry = new OperatorRegistry(accelerator);
        var compiler = new GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new BufferPool(accelerator);
        var executor = new GraphExecutor(accelerator, compiled, weights, constantFloatValues);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession directly from a .onnx file loaded via HTTP.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    public static async Task<InferenceSession> CreateFromOnnxAsync(
        Accelerator accelerator, HttpClient http, string onnxUrl,
        Action<string, int>? onProgress = null)
    {
        // Download .onnx file
        onProgress?.Invoke("download", 0);
        var onnxBytes = await http.GetByteArrayAsync(onnxUrl);
        onProgress?.Invoke("download", 100);

        return CreateFromOnnx(accelerator, onnxBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession directly from raw .onnx bytes.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    public static InferenceSession CreateFromOnnx(
        Accelerator accelerator, byte[] onnxBytes,
        Action<string, int>? onProgress = null)
    {
        // Parse ONNX protobuf
        onProgress?.Invoke("parse", 0);
        var (modelInfo, cpuWeights) = Onnx.OnnxLoader.LoadModel(onnxBytes);
        onProgress?.Invoke("parse", 100);

        // Convert OnnxModelInfo → ModelGraph
        var graph = ConvertToModelGraph(modelInfo);

        // Extract small constant values — data is already on CPU (from ONNX parser), no readback needed
        graph.ConstantData = new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                if (elems <= 16)
                    graph.ConstantData[name] = data.Select(v => (int)v).ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Upload weights to GPU
        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;
        foreach (var (name, data) in cpuWeights)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX: {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
    }

    /// <summary>
    /// Convert OnnxModelInfo (from native parser) to ModelGraph (used by GraphCompiler).
    /// </summary>
    public static ModelGraph ConvertToModelGraph(Onnx.OnnxModelInfo info)
    {
        var graph = new ModelGraph
        {
            Name = info.Name,
            Inputs = info.InputNames.Select(name => new GraphValueInfo
            {
                Name = name,
                Shape = info.ValueShapes.TryGetValue(name, out var s) ? s : Array.Empty<int>()
            }).ToList(),
            Outputs = info.OutputNames.Select(name => new GraphValueInfo
            {
                Name = name,
                Shape = info.ValueShapes.TryGetValue(name, out var s) ? s : Array.Empty<int>()
            }).ToList(),
            Initializers = new Dictionary<string, int[]>(),
        };

        // Register initializer shapes
        foreach (var initName in info.InitializerNames)
        {
            if (info.ValueShapes.TryGetValue(initName, out var shape))
                graph.Initializers[initName] = shape;
        }

        // Convert nodes
        foreach (var node in info.Nodes)
        {
            var graphNode = new GraphNode
            {
                OpType = node.OpType,
                Inputs = node.Inputs.ToList(),
                Outputs = node.Outputs.ToList(),
            };

            // Convert typed attributes to JsonElement-backed attributes
            // The GraphNode uses JsonElement for serialization compatibility,
            // but we have typed objects from the ONNX parser. Serialize and re-parse.
            if (node.Attributes.Count > 0)
            {
                var jsonDict = new Dictionary<string, System.Text.Json.JsonElement>();
                foreach (var (key, value) in node.Attributes)
                {
                    var json = System.Text.Json.JsonSerializer.Serialize(value);
                    jsonDict[key] = System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
                }
                graphNode.Attributes = jsonDict;
            }

            graph.Nodes.Add(graphNode);
        }

        return graph;
    }

    /// <summary>Run inference with named input tensors. Returns named output tensors.</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => _executor.Run(inputs);

    /// <summary>Async inference — required for browser backends (WebGPU/WebGL/Wasm)
    /// which deadlock on synchronous Synchronize(). Periodically flushes GPU commands.</summary>
    public Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
        => _executor.RunAsync(inputs);

    /// <summary>Run inference with a single input. Returns the first output tensor.</summary>
    public Tensor Run(string inputName, Tensor input)
    {
        var outputs = _executor.Run(new Dictionary<string, Tensor> { [inputName] = input });
        return outputs.Values.First();
    }

    /// <summary>Summary string for logging/display.</summary>
    public override string ToString()
    {
        var inShape = InputShapes.Count > 0 ? $"[{string.Join(",", InputShapes.Values.First())}]" : "?";
        var outShape = OutputShapes.Count > 0 ? $"[{string.Join(",", OutputShapes.Values.First())}]" : "?";
        return $"{ModelName}: {NodeCount} nodes, {WeightCount} weights, {string.Join("+", OperatorTypes)}, input={inShape} output={outShape}";
    }

    public void Dispose()
    {
        _executor.Dispose();
        _pool.Dispose();
    }
}
