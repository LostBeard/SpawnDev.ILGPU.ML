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
    private List<IDisposable>? _ownedBuffers; // Tracks buffers not managed by the pool (GGUF quantized bytes, etc.)

    // ── Dynamic-shape recompilation (shape specialization) ──
    // A model with dynamic input dims (e.g. a transformer decoder's sequence_length) is compiled
    // ONCE at its declared shape (dynamic dims → 1). Running it at a DIFFERENT shape (the prompt
    // length, then a growing decode sequence) needs every seq-dependent buffer re-sized. The
    // executor's per-node compiled shapes can't be patched correctly at runtime (the compiler
    // folds whole Shape→Gather→Concat→Reshape subgraphs with concrete dims; a runtime patch can't
    // reproduce that), so instead we RECOMPILE the graph at the actual input shape and run that —
    // identical to having pinned `inputShapes` to those dims. Compiled executors are cached per
    // input-shape signature (kernels are cached at the accelerator level, so recompile is pure CPU
    // shape inference; weights are shared, never re-uploaded). Null when recompilation isn't wired
    // for this load path (TFLite/GGUF/etc.) — those keep the single-executor behavior.
    private Graph.ModelGraph? _recompileGraph;             // graph structure (nodes/inputs/initializers)
    private Dictionary<string, int[]>? _recompileConstSeed;    // CLEAN ConstantData seed (pre-fold)
    private Dictionary<string, float[]>? _recompileFloatSeed;  // CLEAN FloatConstantData seed (pre-fold)
    private bool _recompileEnableOptimization = true;
    private readonly Dictionary<string, GraphExecutor> _shapeExecutors = new();
    private readonly List<string> _shapeExecutorLru = new();
    private const int MaxShapeExecutors = 3; // bound GPU memory: each holds its own intermediate pool

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

    /// <summary>Try to get a weight tensor by name (for diagnostics).</summary>
    public Tensor? TryGetWeight(string name)
        => _weights.TryGetValue(name, out var t) ? t : null;

    /// <summary>Get all weight tensor names (for diagnostics).</summary>
    public IEnumerable<string> GetWeightNames() => _weights.Keys;

    /// <summary>Distinct operator types used in this model.</summary>
    public string[] OperatorTypes => _compiled.Nodes.Select(n => n.OpType).Distinct().OrderBy(s => s).ToArray();

    /// <summary>Get the OpType + first output name for a node by index. Diagnostic only.</summary>
    public (string opType, string outputName) GetNodeInfo(int idx)
    {
        if (idx < 0 || idx >= _compiled.Nodes.Length) return ("?", "?");
        var n = _compiled.Nodes[idx];
        return (n.OpType, n.OutputNames.Length > 0 ? n.OutputNames[0] : "?");
    }

    /// <summary>Model name (from graph metadata).</summary>
    public string ModelName { get; private set; } = "";

    /// <summary>Access to the underlying GraphExecutor (for KV cache management).</summary>
    public GraphExecutor Executor => _executor;

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
    /// Wire up dynamic-shape recompilation for this session. Called by the ONNX load paths, passing
    /// the <see cref="Graph.ModelGraph"/> plus SNAPSHOTS of its ConstantData/FloatConstantData taken
    /// BEFORE the first compile. The seeds matter because with optimization OFF (the NLP path),
    /// <see cref="GraphCompiler.Compile"/> folds constants directly onto the graph it is given — so
    /// after the base compile the graph's ConstantData holds seq=1 folded values. Each recompile
    /// resets the graph to these clean seeds first, then pins the actual input shapes. When a later
    /// Run/RunAsync supplies an input shape that differs from the compile-time shape, the session
    /// recompiles at the actual shape and runs that executor instead.
    /// </summary>
    private void EnableShapeRecompilation(Graph.ModelGraph graph,
        Dictionary<string, int[]>? constSeed, Dictionary<string, float[]>? floatSeed,
        bool enableOptimization)
    {
        _recompileGraph = graph;
        _recompileConstSeed = constSeed;
        _recompileFloatSeed = floatSeed;
        _recompileEnableOptimization = enableOptimization;
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Pick (or build) the executor whose compiled graph matches these input shapes. Returns the
    /// base executor when the inputs match the compile-time shapes (the common case: every shape-
    /// pinned/static model), or recompilation isn't wired. Otherwise returns a per-shape executor,
    /// recompiling + caching on first use of each distinct input-shape signature.
    /// </summary>
    private GraphExecutor ResolveExecutor(Dictionary<string, Tensor> inputs)
    {
        if (_recompileGraph == null) return _executor;

        // Match against the shapes the base graph was compiled for. If every supplied input matches
        // (or isn't a graph input), the base executor is already specialized for this call.
        bool differs = false;
        foreach (var (name, tensor) in inputs)
        {
            if (_compiled.CompiledInputShapes.TryGetValue(name, out var compiledShape)
                && !ShapesEqual(tensor.Shape, compiledShape))
            {
                differs = true;
                break;
            }
        }
        if (!differs) return _executor;

        var sig = string.Join("|", inputs.OrderBy(kv => kv.Key)
            .Select(kv => $"{kv.Key}:{string.Join(",", kv.Value.Shape)}"));
        if (_shapeExecutors.TryGetValue(sig, out var cached))
        {
            _shapeExecutorLru.Remove(sig);
            _shapeExecutorLru.Add(sig);
            return cached;
        }

        var exec = RecompileForShapes(inputs);
        _shapeExecutors[sig] = exec;
        _shapeExecutorLru.Add(sig);
        // Bound GPU memory — each cached executor owns an intermediate buffer pool sized for its
        // sequence length. Evict the least-recently-used (the base executor is never in this cache).
        while (_shapeExecutorLru.Count > MaxShapeExecutors)
        {
            var evict = _shapeExecutorLru[0];
            _shapeExecutorLru.RemoveAt(0);
            if (_shapeExecutors.Remove(evict, out var old)) old.Dispose();
        }
        return exec;
    }

    /// <summary>
    /// Recompile <see cref="_recompileGraph"/> with the supplied input shapes pinned, and build an
    /// executor for it that SHARES this session's already-uploaded weights (no re-upload). The
    /// compiler clones the graph internally, so mutating the stored graph's input shapes here is
    /// safe and reused across recompiles.
    /// </summary>
    private GraphExecutor RecompileForShapes(Dictionary<string, Tensor> inputs)
    {
        var graph = _recompileGraph!;
        // Reset to the clean pre-fold constant seeds (the prior compile may have folded seq-specific
        // values onto the graph when optimization is off), then pin the actual input shapes.
        graph.ConstantData = _recompileConstSeed != null ? new Dictionary<string, int[]>(_recompileConstSeed) : new();
        graph.FloatConstantData = _recompileFloatSeed != null ? new Dictionary<string, float[]>(_recompileFloatSeed) : new();
        foreach (var inp in graph.Inputs)
            if (inputs.TryGetValue(inp.Name, out var t))
                inp.Shape = t.Shape.ToArray();

        var compiled = new GraphCompiler(_registry) { EnableOptimization = _recompileEnableOptimization }.Compile(graph);
        var exec = new GraphExecutor(_accelerator, compiled, _weights, _recompileFloatSeed, registry: _registry)
        {
            Format = _executor.Format,
        };
        if (VerboseLogging)
            Console.WriteLine($"[InferenceSession] Recompiled for shapes [{string.Join("; ", inputs.Select(kv => $"{kv.Key}:[{string.Join(",", kv.Value.Shape)}]"))}] — {compiled.Nodes.Length} nodes");
        return exec;
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
        await weightLoader.LoadAsync(basePath, onProgress);
        onProgress?.Invoke("weights", 100);

        // For Wasm/WebGL, pre-fetch master weight buffer to CPU once.
        // Avoids GPU→GPU SubView copies that cause Wasm OOB and WebGL peer-to-peer issues.
        bool useCpuStaging = accelerator.AcceleratorType == AcceleratorType.Wasm ||
                             accelerator.AcceleratorType == AcceleratorType.WebGL;
        float[]? cpuWeightsAll = null;
        if (useCpuStaging)
        {
            await accelerator.SynchronizeAsync();
            cpuWeightsAll = await weightLoader.CopyAllToHostAsync();
        }

        // Extract small constant values for shape inference AND runtime operator use.
        // Use ONE shared read buffer to avoid allocating hundreds of tiny GPU buffers.
        modelGraph.ConstantData ??= new Dictionary<string, int[]>();
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
                            if (useCpuStaging && cpuWeightsAll != null)
                            {
                                var slice = weightLoader.GetSlice(name);
                                if (slice != null)
                                {
                                    var hostBuf = new float[elems];
                                    Array.Copy(cpuWeightsAll, slice.Value.offset, hostBuf, 0, elems);
                                    constantFloatValues[name] = hostBuf;
                                    modelGraph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                                }
                            }
                            else
                            {
                                // GPU→GPU copy via CopyFrom — uses native CopyBufferToBuffer on WebGPU.
                                // CopyFrom (GPU→GPU) works on ALL backends. Only CopyTo (GPU→CPU) throws
                                // on browser backends. CopyFrom is preferred over Scale(×1) here because
                                // it's a native GPU command with no kernel compilation overhead.
                                readBuf.View.SubView(0, elems).CopyFrom(view.Value.SubView(0, elems));
                                await accelerator.SynchronizeAsync();
                                var hostBuf = await readBuf.CopyToHostAsync<float>(0, elems);
                                constantFloatValues[name] = hostBuf;
                                modelGraph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                            }
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

        // Reuse pre-fetched CPU weights (already fetched before constant extraction)

        int loadedCount = 0;
        foreach (var name in allInitNames)
        {
            var view = weightLoader.TryGetView(name);
            if (view != null && weightLoader.Shapes.TryGetValue(name, out var shape))
            {
                // Copy each weight into its OWN buffer. WeightLoader uses a single
                // shared buffer with SubViews, but WebGPU doesn't allow binding the
                // same GPUBuffer to multiple storage slots in one kernel dispatch.
                int count = Tensors.TensorHelpers.ElementCount(shape);
                var ownBuf = accelerator.Allocate1D<float>(count);
                if (useCpuStaging && cpuWeightsAll != null)
                {
                    // CPU staging: slice from the pre-fetched master buffer
                    var slice = weightLoader.GetSlice(name);
                    if (slice != null)
                    {
                        var weightSlice = new float[count];
                        Array.Copy(cpuWeightsAll, slice.Value.offset, weightSlice, 0, count);
                        ownBuf.CopyFromCPU(weightSlice);
                    }
                }
                else
                {
                    // GPU→GPU copy via Scale kernel
                    registry.ElementWise.Scale(view.Value.SubView(0, count), ownBuf.View, count, 1f);
                }
                weights[name] = new Tensor(ownBuf.View, shape, name);
                loadedCount++;
            }
        }

        // Log weight loading stats
        int missingCount = allInitNames.Count - loadedCount;
        if (VerboseLogging)
        {
            if (missingCount > 0)
            {
                var missing = allInitNames.Where(n => !weights.ContainsKey(n)).Take(5);
                if (VerboseLogging) Console.WriteLine($"[InferenceSession] WARNING: {missingCount} initializers not found in weights. First few: {string.Join(", ", missing)}");
            }
            if (VerboseLogging) Console.WriteLine($"[InferenceSession] Loaded {loadedCount}/{allInitNames.Count} weights, {compiled.Nodes.Length} nodes compiled");
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
            graph.ConstantData ??= new Dictionary<string, int[]>();
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
                    graph.ConstantData[name] = hostBuf.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
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
    /// Create an InferenceSession from any supported model file via HTTP.
    /// Auto-detects format from file extension (.onnx, .tflite) or magic bytes.
    /// </summary>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions.
    /// e.g. new Dictionary&lt;string, int[]&gt; { ["pixel_values"] = new[] { 1, 3, 518, 518 } }</param>
    public static async Task<InferenceSession> CreateFromFileAsync(
        Accelerator accelerator, HttpClient http, string modelUrl,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        onProgress?.Invoke("download", 0);
        var bytes = await DownloadBytesChunkedAsync(http, modelUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromFile(accelerator, bytes, onProgress, inputShapes, enableOptimization);
    }

    /// <summary>
    /// Create an InferenceSession from raw model bytes.
    /// Auto-detects format from magic bytes: ONNX (protobuf) or TFLite (FlatBuffers).
    /// </summary>
    public static InferenceSession CreateFromFile(
        Accelerator accelerator, byte[] modelBytes,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        var format = DetectModelFormat(modelBytes);
        return format switch
        {
            ModelFormat.ONNX => CreateFromOnnx(accelerator, modelBytes, onProgress, inputShapes, enableOptimization),
            ModelFormat.TFLite => CreateFromTFLite(accelerator, modelBytes, onProgress),
            ModelFormat.GGUF => CreateFromGGUF(accelerator, modelBytes, onProgress),
            ModelFormat.SafeTensors => CreateFromSafeTensors(accelerator, modelBytes, onProgress),
            ModelFormat.PyTorch => CreateFromPyTorch(accelerator, modelBytes, onProgress),
            ModelFormat.CoreML => CreateFromCoreML(accelerator, modelBytes, onProgress),
            ModelFormat.TFGraphDef => CreateFromTFGraphDef(accelerator, modelBytes, onProgress),
            _ => throw new NotSupportedException($"Unknown model format '{format}'. Supported: ONNX, TFLite, GGUF, SafeTensors, PyTorch, CoreML, TFGraphDef.")
        };
    }

    /// <summary>Detect model format from magic bytes.</summary>
    public static ModelFormat DetectModelFormat(byte[] data)
    {
        if (data.Length < 4) return ModelFormat.Unknown;

        // GGUF: bytes 0-3 = "GGUF"
        if (data[0] == 'G' && data[1] == 'G' && data[2] == 'U' && data[3] == 'F')
            return ModelFormat.GGUF;

        // TFLite: bytes 4-7 = "TFL3" (FlatBuffers file identifier)
        if (data.Length > 7 && data[4] == 'T' && data[5] == 'F' && data[6] == 'L' && data[7] == '3')
            return ModelFormat.TFLite;

        // ONNX: starts with protobuf varint field tag (typically 0x08 for field 1, varint type)
        // More reliable: check for the "onnx" or "pytorch" producer string within first 64 bytes.
        // Bound is data.Length - 3 (not - 4): the last valid 4-byte window [i..i+3] starts at
        // data.Length - 4, so the loop must allow i == data.Length - 4 (i.e. i < data.Length - 3).
        // The old "- 4" bound skipped that final window, so an "onnx" string sitting exactly at
        // offset 4 of a minimal 8-byte buffer was never scanned and the bytes fell through to the
        // 0x08 protobuf fallback (mis-detected as CoreML version 7).
        for (int i = 0; i < Math.Min(64, data.Length - 3); i++)
        {
            if (data[i] == 'o' && data[i + 1] == 'n' && data[i + 2] == 'n' && data[i + 3] == 'x')
                return ModelFormat.ONNX;
            if (data[i] == 'p' && data[i + 1] == 'y' && data[i + 2] == 't' && data[i + 3] == 'o')
                return ModelFormat.ONNX; // pytorch producer
        }

        // glTF binary: "glTF" magic (0x46546C67)
        if (data[0] == 0x67 && data[1] == 0x6C && data[2] == 0x54 && data[3] == 0x46)
            return ModelFormat.GLTF;

        // SPZ: gzip header (0x1F 0x8B) — decompress and check for SPZ magic
        if (data[0] == 0x1F && data[1] == 0x8B && Formats.SPZParser.IsValidSPZ(data))
            return ModelFormat.SPZ;

        // PLY: starts with "ply\n"
        if (data[0] == 'p' && data[1] == 'l' && data[2] == 'y' && (data[3] == '\n' || data[3] == '\r'))
            return ModelFormat.PLY;

        // OBJ: starts with "# " (comment) or "v " (vertex) or "o " (object)
        if ((data[0] == '#' && data[1] == ' ') || (data[0] == 'v' && data[1] == ' ') || (data[0] == 'o' && data[1] == ' '))
            return ModelFormat.OBJ;

        // PyTorch: ZIP archive (PK header)
        if (PyTorch.PyTorchLoader.IsPyTorchCheckpoint(data))
            return ModelFormat.PyTorch;

        // SafeTensors: starts with uint64 header size, then '{'
        if (SafeTensors.SafeTensorsParser.IsSafeTensors(data))
            return ModelFormat.SafeTensors;

        // Fallback: if first byte is a protobuf field tag (0x08, 0x0A, etc.)
        // Could be ONNX, TF GraphDef, or CoreML — differentiate by structure
        if (data[0] == 0x08 || data[0] == 0x0A)
        {
            // CoreML: field 1 (specificationVersion) is varint tag 0x08, version in range 1-10
            if (CoreML.CoreMLParser.IsCoreML(data))
                return ModelFormat.CoreML;

            // TFGraphDef starts with field 1 (node) = 0x0A, and the "onnx"/"pytorch" checks above
            // already returned ONNX if those strings were present. If we reach here with 0x0A and
            // TFGraphDefParser validates it, it's likely a frozen TF graph.
            if (data[0] == 0x0A && TensorFlow.TFGraphDefParser.IsGraphDef(data))
            {
                // Additional heuristic: try parsing the first node — if it has a valid TF op name, it's TFGraphDef
                try
                {
                    var testGraph = TensorFlow.TFGraphDefParser.Parse(data);
                    if (testGraph.Nodes.Count > 0 && testGraph.Nodes.Any(n =>
                        n.Op == "Placeholder" || n.Op == "Const" || n.Op == "Conv2D" ||
                        n.Op == "MatMul" || n.Op == "Relu" || n.Op == "BiasAdd" ||
                        n.Op == "FusedBatchNorm" || n.Op == "Add" || n.Op == "AddV2"))
                        return ModelFormat.TFGraphDef;
                }
                catch { /* Not a valid TFGraphDef — fall through to ONNX */ }
            }
            return ModelFormat.ONNX;
        }

        return ModelFormat.Unknown;
    }

    /// <summary>
    /// Create an InferenceSession from a HuggingFace Hub model with OPFS caching.
    /// Downloads the model on first call; subsequent calls load instantly from cache.
    /// <code>
    /// var hub = new ModelHub(js);
    /// var session = await InferenceSession.CreateFromHuggingFaceAsync(
    ///     accelerator, hub, "onnx-community/squeezenet1.1-7", "model.onnx");
    /// </code>
    /// </summary>
    /// <param name="accelerator">GPU accelerator to compile kernels on</param>
    /// <param name="hub">ModelHub instance (provides OPFS caching)</param>
    /// <param name="repoId">HuggingFace repository ID (e.g., "onnx-community/squeezenet1.1-7")</param>
    /// <param name="filename">File path within the repo (e.g., "model.onnx" or "onnx/model.onnx")</param>
    /// <param name="revision">Git revision (default: "main")</param>
    /// <param name="onProgress">Progress callback: (stage, percent)</param>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions</param>
    public static async Task<InferenceSession> CreateFromHuggingFaceAsync(
        Accelerator accelerator, Hub.ModelHub hub,
        string repoId, string filename, string revision = "main",
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null)
    {
        onProgress?.Invoke("download", 0);
        var bytes = await hub.LoadAsync(repoId, filename, revision);
        onProgress?.Invoke("download", 100);

        // Check if this ONNX model uses external data format
        byte[]? externalData = null;
        var format = DetectModelFormat(bytes);
        if (format == ModelFormat.ONNX)
        {
            // Quick parse to check for external data — lightweight since we only scan initializer headers
            var quickParse = Onnx.OnnxParser.Parse(bytes, zeroCopyThreshold: 1024 * 1024);
            if (Onnx.OnnxLoader.HasExternalData(quickParse))
            {
                // Download the external data file (typically model.onnx_data)
                var dataFilename = filename + "_data";
                // Check if any initializer specifies a different location
                var firstExtInit = quickParse.Graph.Initializers.FirstOrDefault(i => i.DataLocation == 1 && i.ExternalData != null);
                if (firstExtInit?.ExternalData?.TryGetValue("location", out var loc) == true && !string.IsNullOrEmpty(loc))
                {
                    // External data location is relative to the ONNX file
                    var dir = filename.Contains('/') ? filename[..filename.LastIndexOf('/')] + "/" : "";
                    dataFilename = dir + loc;
                }
                onProgress?.Invoke("download_data", 0);
                externalData = await hub.LoadAsync(repoId, dataFilename, revision);
                onProgress?.Invoke("download_data", 100);

                // Resolve external data in the already-parsed model
                Onnx.OnnxLoader.ResolveExternalData(quickParse, externalData);
                externalData = null; // Free memory — data is now in the tensors

                // Use the already-parsed model directly to avoid re-parsing
                return CreateFromOnnxParsed(accelerator, quickParse, onProgress, inputShapes);
            }
        }

        return CreateFromFile(accelerator, bytes, onProgress, inputShapes);
    }

    /// <summary>
    /// Create an InferenceSession directly from a .onnx file loaded via HTTP.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    public static async Task<InferenceSession> CreateFromOnnxAsync(
        Accelerator accelerator, HttpClient http, string onnxUrl,
        Action<string, int>? onProgress = null)
    {
        // Download .onnx file using chunked download to avoid browser WASM OOM
        onProgress?.Invoke("download", 0);
        var onnxBytes = await DownloadBytesChunkedAsync(http, onnxUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromOnnx(accelerator, onnxBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession directly from raw .onnx bytes.
    /// No Python extraction step needed — uses the native ONNX protobuf parser.
    /// </summary>
    /// <param name="inputShapes">Optional: override input shapes for models with dynamic dimensions.</param>
    /// <param name="externalData">Optional: raw bytes of the external data file (model.onnx_data)
    /// for models that store weights in a separate file.</param>
    public static InferenceSession CreateFromOnnx(
        Accelerator accelerator, byte[] onnxBytes,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true,
        byte[]? externalData = null)
    {
        // Single-parse architecture: parse ONNX protobuf ONCE with zero-copy for large tensors.
        // The parsed model is kept in memory — graph info extracted for compilation,
        // then tensors streamed to GPU from the same parsed result.
        // Avoids scanning 652MB of protobuf twice (was the GPT-2 bottleneck).
        onProgress?.Invoke("parse", 0);
        var parsedModel = Onnx.OnnxParser.Parse(onnxBytes, zeroCopyThreshold: 1024 * 1024);

        // Resolve external data if provided (models with separate weight files)
        if (externalData != null && Onnx.OnnxLoader.HasExternalData(parsedModel))
            Onnx.OnnxLoader.ResolveExternalData(parsedModel, externalData);

        var modelInfo = Onnx.OnnxLoader.ExtractModelInfoFromParsed(parsedModel);
        var cpuSmallWeights = new Dictionary<string, float[]>();
        foreach (var init in parsedModel.Graph.Initializers)
        {
            if (init.DataLocation == 1) continue;
            if (init.ElementCount <= 64)
                cpuSmallWeights[init.Name] = init.ToFloatArray();
        }
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr != null)
                {
                    if (valueAttr.T != null && valueAttr.T.ElementCount <= 64)
                        cpuSmallWeights[node.Outputs[0]] = valueAttr.T.ToFloatArray();
                    else if (valueAttr.T == null)
                    {
                        if (valueAttr.F != 0 || valueAttr.I != 0)
                            cpuSmallWeights[node.Outputs[0]] = new[] { valueAttr.F != 0 ? valueAttr.F : (float)valueAttr.I };
                    }
                }
            }
        }
        onProgress?.Invoke("parse", 100);

        // Apply input shape overrides (for models with dynamic dimensions)
        if (inputShapes != null)
        {
            foreach (var (name, shape) in inputShapes)
                modelInfo.ValueShapes[name] = shape;
        }

        // Convert OnnxModelInfo → ModelGraph
        ModelGraph graph;
        try { graph = ConvertToModelGraph(modelInfo); }
        catch (Exception ex) { throw new InvalidOperationException($"ConvertToModelGraph failed: {ex.GetType().Name}: {ex.Message}", ex); }

        // Extract small constant values — data is already on CPU (from pass 1), no readback needed
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuSmallWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                // Populate ConstantData and FloatConstantData for ALL small constants
                // (matching the 64-element cpuSmallWeights threshold). The old ≤16 limit
                // broke Upsample/Resize shape inference: scales computed via
                // Shape→Gather→Mul→Floor→Concat need FloatConstantData at every step.
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Pre-extract Pad node pads tensors so GraphExecutor never falls back to GPU readback at execute time.
        PreExtractPads(parsedModel, cpuSmallWeights, constantFloatValues, graph);

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        // Snapshot the CLEAN constant seeds before Compile folds seq-specific values onto the graph
        // (with optimization off, folding mutates `graph` directly). Used for dynamic-shape recompile.
        var _constSeed = graph.ConstantData != null ? new Dictionary<string, int[]>(graph.ConstantData) : null;
        var _floatSeed = graph.FloatConstantData != null ? new Dictionary<string, float[]>(graph.FloatConstantData) : null;
        var compiled = new GraphCompiler(registry) { EnableOptimization = enableOptimization }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Pass 2: Stream weights to GPU — one tensor at a time, minimal CPU peak.
        // Re-enumerate the ONNX weight stream. Each tensor is uploaded to GPU immediately,
        // then the CPU float[] goes out of scope and can be collected.
        // GC between passes: free Pass 1 intermediates (cpuSmallWeights, modelInfo, etc.)
        // to maximize headroom for large tensor raw data in Pass 2.
        // Critical for browser WASM where 652MB onnxBytes + 147MB tensor = OOM without GC.
        cpuSmallWeights = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;

        // Stream weights to GPU from the already-parsed model (no second parse).
        // Small tensors: standard float[] path. Large tensors (>1M elements): chunked upload
        // via AllocatePermanentChunked — 1MB chunks, never allocates full float[] (fixes GPT-2 OOM).
        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
                if (tensor.ElementCount == 0 && expectedElems > 0)
                    gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name);
                else
                    gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name);
                loaded++;
            }
        }
        // Create tensors for optimizer-folded constants that aren't in the weight dictionary.
        // The optimizer adds these as initializers but they have no weight data — fill from ConstantData/FloatConstantData.
        foreach (var name in compiled.InitializerNames)
        {
            if (gpuWeights.ContainsKey(name)) continue;
            if (constantFloatValues.TryGetValue(name, out var fData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fData, shape, name);
                loaded++;
            }
            else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var fcdData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fcdData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fcdData, shape, name);
                loaded++;
            }
            else if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var iData))
            {
                var fVals = iData.Select(v => (float)v).ToArray();
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fVals.Length };
                gpuWeights[name] = pool.AllocatePermanent(fVals, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX: {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        var session = new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
        // Enable dynamic-shape recompilation: a later Run at a different input shape (e.g. a growing
        // decode sequence) recompiles the graph at that shape rather than mis-sizing buffers.
        session.EnableShapeRecompilation(graph, _constSeed, _floatSeed, enableOptimization);
        return session;
    }

    /// <summary>
    /// Create an InferenceSession from an already-parsed ONNX model.
    /// Used when external data has been resolved in-place before compilation.
    /// </summary>
    private static InferenceSession CreateFromOnnxParsed(
        Accelerator accelerator, Onnx.OnnxModelProto parsedModel,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true)
    {
        onProgress?.Invoke("parse", 0);
        var modelInfo = Onnx.OnnxLoader.ExtractModelInfoFromParsed(parsedModel);
        var cpuSmallWeights = new Dictionary<string, float[]>();
        foreach (var init in parsedModel.Graph.Initializers)
        {
            if (init.DataLocation == 1) continue;
            if (init.ElementCount <= 64)
                cpuSmallWeights[init.Name] = init.ToFloatArray();
        }
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr != null)
                {
                    if (valueAttr.T != null && valueAttr.T.ElementCount <= 64)
                        cpuSmallWeights[node.Outputs[0]] = valueAttr.T.ToFloatArray();
                    else if (valueAttr.T == null)
                    {
                        // Scalar Constant: "value" attribute with F (float) or I (int) field
                        // instead of T (tensor). Common in PyTorch-exported ONNX models.
                        if (valueAttr.F != 0 || valueAttr.I != 0)
                            cpuSmallWeights[node.Outputs[0]] = new[] { valueAttr.F != 0 ? valueAttr.F : (float)valueAttr.I };
                    }
                }
            }
        }
        onProgress?.Invoke("parse", 100);

        if (inputShapes != null)
        {
            foreach (var (name, shape) in inputShapes)
                modelInfo.ValueShapes[name] = shape;
        }

        ModelGraph graph;
        try { graph = ConvertToModelGraph(modelInfo); }
        catch (Exception ex) { throw new InvalidOperationException($"ConvertToModelGraph failed: {ex.GetType().Name}: {ex.Message}", ex); }

        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuSmallWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                // Populate ConstantData and FloatConstantData for ALL small constants
                // (matching the 64-element cpuSmallWeights threshold). The old ≤16 limit
                // broke Upsample/Resize shape inference: scales computed via
                // Shape→Gather→Mul→Floor→Concat need FloatConstantData at every step.
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Pre-extract Pad node pads tensors so GraphExecutor never falls back to GPU readback at execute time.
        PreExtractPads(parsedModel, cpuSmallWeights, constantFloatValues, graph);

        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        // Snapshot the CLEAN constant seeds before Compile folds seq-specific values onto the graph
        // (with optimization off, folding mutates `graph` directly). Used for dynamic-shape recompile.
        var _constSeed = graph.ConstantData != null ? new Dictionary<string, int[]>(graph.ConstantData) : null;
        var _floatSeed = graph.FloatConstantData != null ? new Dictionary<string, float[]>(graph.FloatConstantData) : null;
        var compiled = new GraphCompiler(registry) { EnableOptimization = enableOptimization }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        cpuSmallWeights = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;

        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
                if (tensor.ElementCount == 0 && expectedElems > 0)
                    gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name);
                else
                    gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name);
                loaded++;
            }
        }
        foreach (var name in compiled.InitializerNames)
        {
            if (gpuWeights.ContainsKey(name)) continue;
            if (constantFloatValues.TryGetValue(name, out var fData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fData, shape, name);
                loaded++;
            }
            else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var fcdData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fcdData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fcdData, shape, name);
                loaded++;
            }
            else if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var iData))
            {
                var fVals = iData.Select(v => (float)v).ToArray();
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fVals.Length };
                gpuWeights[name] = pool.AllocatePermanent(fVals, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX (ext): {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        var session = new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
        // Enable dynamic-shape recompilation: a later Run at a different input shape (e.g. a growing
        // decode sequence) recompiles the graph at that shape rather than mis-sizing buffers.
        session.EnableShapeRecompilation(graph, _constSeed, _floatSeed, enableOptimization);
        return session;
    }

    /// <summary>
    /// Create an InferenceSession from a SEEKABLE ONNX stream WITHOUT holding the whole model in memory.
    /// The structure is parsed via <see cref="Onnx.OnnxParser.ParseFromStreamAsync"/> (weights recorded as
    /// stream offsets, never materialized), the graph is compiled, then each large weight is seeked to and
    /// uploaded to the GPU in 1 MB chunks — so a multi-GB model loads with a CPU peak of one chunk. The
    /// stream stays open for the duration of this call (the loader seeks back to each weight); the caller
    /// owns disposing it afterward. Foundation for loading a model directly from a <c>TorrentReadStream</c> /
    /// HTTP-Range / Blob source, and for sharded loading (a peer fetches only its shard's tensors).
    /// </summary>
    public static async Task<InferenceSession> CreateFromOnnxStreamAsync(
        Accelerator accelerator, Stream stream,
        Action<string, int>? onProgress = null,
        Dictionary<string, int[]>? inputShapes = null,
        bool enableOptimization = true,
        int streamThreshold = 1024 * 1024,
        CancellationToken ct = default)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        onProgress?.Invoke("parse", 0);
        var parsedModel = await Onnx.OnnxParser.ParseFromStreamAsync(stream, streamThreshold, ct).ConfigureAwait(false);

        var modelInfo = Onnx.OnnxLoader.ExtractModelInfoFromParsed(parsedModel);
        var cpuSmallWeights = new Dictionary<string, float[]>();
        foreach (var init in parsedModel.Graph.Initializers)
        {
            if (init.DataLocation == 1) continue;
            if (init.ElementCount <= 64)
                cpuSmallWeights[init.Name] = init.ToFloatArray();
        }
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Count > 0)
            {
                var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value");
                if (valueAttr != null)
                {
                    if (valueAttr.T != null && valueAttr.T.ElementCount <= 64)
                        cpuSmallWeights[node.Outputs[0]] = valueAttr.T.ToFloatArray();
                    else if (valueAttr.T == null)
                    {
                        if (valueAttr.F != 0 || valueAttr.I != 0)
                            cpuSmallWeights[node.Outputs[0]] = new[] { valueAttr.F != 0 ? valueAttr.F : (float)valueAttr.I };
                    }
                }
            }
        }
        onProgress?.Invoke("parse", 100);

        if (inputShapes != null)
            foreach (var (name, shape) in inputShapes)
                modelInfo.ValueShapes[name] = shape;

        ModelGraph graph;
        try { graph = ConvertToModelGraph(modelInfo); }
        catch (Exception ex) { throw new InvalidOperationException($"ConvertToModelGraph failed: {ex.GetType().Name}: {ex.Message}", ex); }

        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuSmallWeights.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        PreExtractPads(parsedModel, cpuSmallWeights, constantFloatValues, graph);

        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        // Snapshot the CLEAN constant seeds before Compile folds seq-specific values onto the graph
        // (with optimization off, folding mutates `graph` directly). Used for dynamic-shape recompile.
        var _constSeed = graph.ConstantData != null ? new Dictionary<string, int[]>(graph.ConstantData) : null;
        var _floatSeed = graph.FloatConstantData != null ? new Dictionary<string, float[]>(graph.FloatConstantData) : null;
        var compiled = new GraphCompiler(registry) { EnableOptimization = enableOptimization }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        cpuSmallWeights = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;

        // Stream weights to GPU: large tensors are seeked to + chunk-uploaded straight from the stream
        // (never materialized); small/inline tensors use the in-memory chunked/standard path.
        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (!graph.Initializers.TryGetValue(name, out var shape)) continue;
            int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
            if (tensor.RawDataStreamOffset >= 0)
                gpuWeights[name] = await pool.AllocatePermanentFromStreamAsync(
                    stream, tensor.RawDataStreamOffset, tensor.RawDataLength, tensor.DataType, shape, name, ct).ConfigureAwait(false);
            else if (tensor.ElementCount == 0 && expectedElems > 0)
                gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name);
            else
                gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name);
            loaded++;
        }
        foreach (var name in compiled.InitializerNames)
        {
            if (gpuWeights.ContainsKey(name)) continue;
            if (constantFloatValues.TryGetValue(name, out var fData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fData, shape, name);
                loaded++;
            }
            else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var fcdData))
            {
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fcdData.Length };
                gpuWeights[name] = pool.AllocatePermanent(fcdData, shape, name);
                loaded++;
            }
            else if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var iData))
            {
                var fVals = iData.Select(v => (float)v).ToArray();
                var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { fVals.Length };
                gpuWeights[name] = pool.AllocatePermanent(fVals, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] ONNX (stream): {modelInfo.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        onProgress?.Invoke("ready", 100);

        var session = new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = modelInfo.Name
        };
        // Enable dynamic-shape recompilation: a later Run at a different input shape (e.g. a growing
        // decode sequence) recompiles the graph at that shape rather than mis-sizing buffers.
        session.EnableShapeRecompilation(graph, _constSeed, _floatSeed, enableOptimization);
        return session;
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

        // Carry ONNX-declared initializer dtypes forward. The runtime needs these
        // to honour integer-vs-float Div semantics (TF tf.floordiv exports as
        // Cast(int)→Div(int,int) and ONNX Div on integer dtypes truncates).
        if (info.InitializerDataTypes != null && info.InitializerDataTypes.Count > 0)
            graph.InitializerDataTypes = new Dictionary<string, int>(info.InitializerDataTypes);

        // Register Constant node outputs as initializers so their weight data gets uploaded to GPU.
        // OnnxLoader.ExtractWeights() already extracted the tensor data into cpuWeightsAll,
        // but without registering them here, the weight upload loop skips them.
        foreach (var node in info.Nodes)
        {
            if (node.OpType == "Constant" && node.Outputs.Length > 0)
            {
                var outputName = node.Outputs[0];
                if (!graph.Initializers.ContainsKey(outputName))
                {
                    // Get shape from ValueShapes if available, otherwise from the weight data size
                    if (info.ValueShapes.TryGetValue(outputName, out var constShape))
                        graph.Initializers[outputName] = constShape;
                    else
                        graph.Initializers[outputName] = new[] { 1 }; // Fallback — will be overridden by actual data
                }
            }
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
            // ONNX models can contain NaN/Infinity in attributes (e.g., min/max bounds),
            // so AllowNamedFloatingPointLiterals is required.
            if (node.Attributes.Count > 0)
            {
                var nanSafeOptions = new System.Text.Json.JsonSerializerOptions
                {
                    NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
                };
                var jsonDict = new Dictionary<string, System.Text.Json.JsonElement>();
                foreach (var (key, value) in node.Attributes)
                {
                    var json = System.Text.Json.JsonSerializer.Serialize(value, nanSafeOptions);
                    jsonDict[key] = System.Text.Json.JsonDocument.Parse(json).RootElement.Clone();
                }
                graphNode.Attributes = jsonDict;
            }

            graph.Nodes.Add(graphNode);
        }

        return graph;
    }

    /// <summary>
    /// Create an InferenceSession from a .tflite file loaded via HTTP.
    /// No Python extraction step needed — uses the native TFLite FlatBuffers parser.
    /// </summary>
    public static async Task<InferenceSession> CreateFromTFLiteAsync(
        Accelerator accelerator, HttpClient http, string tfliteUrl,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("download", 0);
        var tfliteBytes = await DownloadBytesChunkedAsync(http, tfliteUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromTFLite(accelerator, tfliteBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession directly from raw .tflite bytes.
    /// Uses the native TFLite FlatBuffers parser — zero dependencies.
    /// </summary>
    public static InferenceSession CreateFromTFLite(
        Accelerator accelerator, byte[] tfliteBytes,
        Action<string, int>? onProgress = null)
    {
        // Parse TFLite FlatBuffers
        onProgress?.Invoke("parse", 0);
        var (graph, cpuWeightsAll) = TFLite.TFLiteLoader.LoadModel(tfliteBytes);
        onProgress?.Invoke("parse", 100);

        // Extract small constant values for shape inference
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuWeightsAll.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = true }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Upload weights to GPU
        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] TFLite: {graph.Name}, {compiled.Nodes.Length} nodes, {loaded} weights uploaded");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues);
        executor.Format = DataFormat.NHWC; // TFLite models use NHWC natively
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>
    /// Create an InferenceSession from a .gguf file loaded via HTTP.
    /// Parses GGUF metadata, constructs transformer graph, uploads weights.
    /// </summary>
    public static async Task<InferenceSession> CreateFromGGUFAsync(
        Accelerator accelerator, HttpClient http, string ggufUrl,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("download", 0);
        var ggufBytes = await DownloadBytesChunkedAsync(http, ggufUrl, onProgress);
        onProgress?.Invoke("download", 100);

        return CreateFromGGUF(accelerator, ggufBytes, onProgress);
    }

    /// <summary>
    /// Create an InferenceSession from raw .gguf bytes.
    /// Constructs the transformer graph from architecture metadata.
    /// Note: currently only supports F32/F16 weights. Quantized (Q4/Q8) requires dequantization kernels.
    /// </summary>
    public static InferenceSession CreateFromGGUF(
        Accelerator accelerator, byte[] ggufBytes,
        Action<string, int>? onProgress = null)
    {
        // Parse GGUF
        onProgress?.Invoke("parse", 0);
        var ggufModel = GGUF.GGUFParser.Parse(ggufBytes);
        onProgress?.Invoke("parse", 100);

        // Build transformer graph from architecture metadata
        onProgress?.Invoke("build_graph", 0);
        var (graph, cpuWeightsAll, quantizedWeightBytes) = GGUF.GGUFGraphBuilder.BuildGraph(ggufModel);
        onProgress?.Invoke("build_graph", 100);

        // Extract small constant values
        graph.ConstantData ??= new Dictionary<string, int[]>();
        var constantFloatValues = new Dictionary<string, float[]>();
        foreach (var (name, shape) in graph.Initializers)
        {
            int elems = shape.Aggregate(1, (a, b) => a * b);
            if (elems > 0 && elems <= 64 && cpuWeightsAll.TryGetValue(name, out var data))
            {
                constantFloatValues[name] = data;
                graph.ConstantData[name] = data.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
                graph.FloatConstantData ??= new Dictionary<string, float[]>();
                graph.FloatConstantData[name] = data.ToArray();
            }
        }

        // Compile graph
        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = true }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        // Upload weights to GPU
        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        var gpuQuantizedWeights = new Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>();
        var quantizedBuffers = new List<MemoryBuffer1D<byte, Stride1D.Dense>>(); // keep alive
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                // Check if this weight has quantized bytes — upload raw if so
                if (quantizedWeightBytes.TryGetValue(name, out var qBytes))
                {
                    var qBuf = accelerator.Allocate1D<byte>(qBytes.Length);
                    qBuf.View.CopyFromCPU(qBytes);
                    gpuQuantizedWeights[name] = qBuf.View;
                    quantizedBuffers.Add(qBuf);
                    // Still need a Tensor entry for shape tracking (empty data)
                    gpuWeights[name] = pool.Rent(shape, name);
                }
                else if (data.Length > 0)
                {
                    gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                }
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        int qCount = gpuQuantizedWeights.Count;
        if (VerboseLogging) Console.WriteLine($"[InferenceSession] GGUF: {ggufModel.Name} ({ggufModel.Architecture}), {compiled.Nodes.Length} nodes, {loaded} weights ({qCount} quantized), {ggufModel.BlockCount} layers");

        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues,
            quantizedWeights: gpuQuantizedWeights.Count > 0 ? gpuQuantizedWeights : null);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name,
            _ownedBuffers = quantizedBuffers.Count > 0
                ? quantizedBuffers.Cast<IDisposable>().ToList() : null
        };
    }

    // ═══════════════════════════════════════════════════════════
    //  SafeTensors
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Create from SafeTensors weights. Requires config.json for graph construction.
    /// If no config is provided, attempts to infer architecture from tensor names.
    /// </summary>
    public static InferenceSession CreateFromSafeTensors(
        Accelerator accelerator, byte[] safeTensorsBytes,
        Action<string, int>? onProgress = null,
        string? configJson = null)
    {
        onProgress?.Invoke("parse", 0);
        var stFile = SafeTensors.SafeTensorsParser.Parse(safeTensorsBytes);

        // Parse config if provided, or create default
        var config = configJson != null
            ? Hub.HFModelConfig.Parse(configJson)
            : InferConfigFromTensors(stFile);

        onProgress?.Invoke("graph", 0);
        var graph = SafeTensors.SafeTensorsGraphBuilder.BuildGraph(config, stFile);

        onProgress?.Invoke("compile", 0);
        var registry = new Operators.OperatorRegistry(accelerator);
        var compiler = new Graph.GraphCompiler(registry);
        var compiled = compiler.Compile(graph);

        // Upload weights
        onProgress?.Invoke("upload", 0);
        var pool = new Tensors.BufferPool(accelerator);
        var weights = new Dictionary<string, Tensors.Tensor>();
        var constantFloatValues = new Dictionary<string, float[]>();

        foreach (var tensor in stFile.Tensors)
        {
            if (!compiled.InitializerNames.Contains(tensor.Name) &&
                !graph.Initializers.ContainsKey(tensor.Name)) continue;

            var data = stFile.GetTensorFloat32(tensor);
            if (data != null && data.Length > 0)
            {
                weights[tensor.Name] = pool.AllocatePermanent(data, tensor.Shape.Select(s => (int)s).ToArray(), tensor.Name);
                if (data.Length <= 64)
                    constantFloatValues[tensor.Name] = data;
            }
        }

        // Add scale constants for attention
        for (int L = 0; L < config.NumHiddenLayers; L++)
        {
            string scaleName = $"L{L}_scale";
            if (graph.Initializers.ContainsKey(scaleName))
            {
                float scale = 1f / MathF.Sqrt(config.HiddenSize / config.NumAttentionHeads);
                weights[scaleName] = pool.AllocatePermanent(new[] { scale }, new[] { 1 }, scaleName);
                constantFloatValues[scaleName] = new[] { scale };
            }
        }

        var executor = new Graph.GraphExecutor(accelerator, compiled, weights, constantFloatValues, registry: registry);
        onProgress?.Invoke("ready", 100);

        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights)
        {
            ModelName = graph.Name
        };
    }

    /// <summary>Infer architecture config from tensor names when no config.json is available.</summary>
    private static Hub.HFModelConfig InferConfigFromTensors(SafeTensors.SafeTensorsFile stFile)
    {
        var config = new Hub.HFModelConfig();
        var names = stFile.Tensors.Select(t => t.Name).ToHashSet();

        // Detect model type from tensor naming pattern
        if (names.Any(n => n.StartsWith("model.layers.")))
            config.ModelType = "llama"; // LLaMA/Mistral naming
        else if (names.Any(n => n.StartsWith("transformer.h.")))
            config.ModelType = "gpt2";
        else if (names.Any(n => n.StartsWith("bert.")))
            config.ModelType = "bert";
        else
            config.ModelType = "llama"; // default

        // Count layers
        config.NumHiddenLayers = names
            .Where(n => n.Contains(".self_attn.q_proj.") || n.Contains(".attn.c_attn."))
            .Select(n => { var parts = n.Split('.'); for (int i = 0; i < parts.Length; i++) if (int.TryParse(parts[i], out var v)) return v; return -1; })
            .Where(v => v >= 0).DefaultIfEmpty(0).Max() + 1;

        // Infer hidden size from embedding weight
        var embedTensor = stFile.Tensors.FirstOrDefault(t =>
            t.Name == "model.embed_tokens.weight" || t.Name == "transformer.wte.weight");
        if (embedTensor != null && embedTensor.Shape.Length >= 2)
        {
            config.VocabSize = (int)embedTensor.Shape[0];
            config.HiddenSize = (int)embedTensor.Shape[1];
        }

        // Infer heads from Q projection shape
        var qTensor = stFile.Tensors.FirstOrDefault(t => t.Name.Contains("q_proj.weight"));
        if (qTensor != null && qTensor.Shape.Length >= 2)
            config.NumAttentionHeads = (int)(qTensor.Shape[0] / (config.HiddenSize / 128)); // rough estimate

        if (config.NumAttentionHeads <= 0)
            config.NumAttentionHeads = config.HiddenSize / 64; // fallback: 64-dim heads

        config.NumKeyValueHeads = config.NumAttentionHeads;
        config.IntermediateSize = config.HiddenSize * 4;
        config.ArchitectureFamily = "decoder";

        return config;
    }

    // ═══════════════════════════════════════════════════════════
    //  PyTorch
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Create from PyTorch checkpoint (.pt/.pth).
    /// Parses ZIP archive, extracts pickle metadata, loads tensor data.
    /// Requires config.json for graph construction (same as SafeTensors).
    /// </summary>
    public static InferenceSession CreateFromPyTorch(
        Accelerator accelerator, byte[] ptBytes,
        Action<string, int>? onProgress = null,
        string? configJson = null)
    {
        onProgress?.Invoke("parse", 0);
        var checkpoint = PyTorch.PyTorchLoader.Parse(ptBytes);

        // Try to read config.json from the ZIP if present
        if (configJson == null && checkpoint.ConfigJson != null)
            configJson = checkpoint.ConfigJson;

        // Parse pickle for tensor metadata
        var tensorMetas = new List<PyTorch.PickleReader.TensorMeta>();
        if (checkpoint.PickleData != null)
            tensorMetas = PyTorch.PickleReader.ReadTensors(checkpoint.PickleData);

        // Build a SafeTensorsFile-compatible wrapper for the graph builder
        var config = configJson != null
            ? Hub.HFModelConfig.Parse(configJson)
            : new Hub.HFModelConfig { ModelType = "llama", NumHiddenLayers = 1, HiddenSize = 768, NumAttentionHeads = 12 };

        // For now, PyTorch models need to be exported to SafeTensors or ONNX for full support.
        // This provides basic weight extraction and model loading.
        onProgress?.Invoke("compile", 0);
        var registry = new Operators.OperatorRegistry(accelerator);
        var graph = new Graph.ModelGraph { Name = $"PyTorch ({config.ModelType})" };
        graph.Inputs.Add(new Graph.GraphValueInfo { Name = "input_ids", Shape = new[] { 1, -1 } });
        graph.Outputs.Add(new Graph.GraphValueInfo { Name = "logits", Shape = new[] { 1, -1, config.VocabSize } });

        var compiler = new Graph.GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new Tensors.BufferPool(accelerator);
        var weights = new Dictionary<string, Tensors.Tensor>();

        // Load tensor data from ZIP data files
        foreach (var meta in tensorMetas)
        {
            if (checkpoint.DataFiles.TryGetValue(meta.StorageKey, out var rawData))
            {
                // Convert raw bytes to float32 based on dtype
                float[]? floats = meta.DType switch
                {
                    "torch.FloatStorage" or "float32" => ConvertBytesToFloat32(rawData, meta.Offset, meta.Shape),
                    "torch.HalfStorage" or "float16" => ConvertHalfToFloat32(rawData, meta.Offset, meta.Shape),
                    "torch.BFloat16Storage" or "bfloat16" => ConvertBFloat16ToFloat32(rawData, meta.Offset, meta.Shape),
                    _ => null
                };
                if (floats != null)
                {
                    var shape = meta.Shape.Select(s => (int)s).ToArray();
                    weights[meta.Name] = pool.AllocatePermanent(floats, shape, meta.Name);
                }
            }
        }

        var executor = new Graph.GraphExecutor(accelerator, compiled, weights, new Dictionary<string, float[]>(), registry: registry);
        onProgress?.Invoke("ready", 100);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights) { ModelName = graph.Name };
    }

    private static float[]? ConvertBytesToFloat32(byte[] data, long offset, long[] shape)
    {
        int count = (int)shape.Aggregate(1L, (a, b) => a * b);
        if (count <= 0) return null;
        var result = new float[count];
        Buffer.BlockCopy(data, (int)(offset * 4), result, 0, count * 4);
        return result;
    }

    private static float[]? ConvertHalfToFloat32(byte[] data, long offset, long[] shape)
    {
        int count = (int)shape.Aggregate(1L, (a, b) => a * b);
        if (count <= 0) return null;
        var result = new float[count];
        int byteOff = (int)(offset * 2);
        for (int i = 0; i < count; i++)
        {
            ushort h = (ushort)(data[byteOff + i * 2] | (data[byteOff + i * 2 + 1] << 8));
            result[i] = (float)BitConverter.Int16BitsToHalf((short)h);
        }
        return result;
    }

    private static float[]? ConvertBFloat16ToFloat32(byte[] data, long offset, long[] shape)
    {
        int count = (int)shape.Aggregate(1L, (a, b) => a * b);
        if (count <= 0) return null;
        var result = new float[count];
        int byteOff = (int)(offset * 2);
        for (int i = 0; i < count; i++)
        {
            ushort bf16 = (ushort)(data[byteOff + i * 2] | (data[byteOff + i * 2 + 1] << 8));
            result[i] = BitConverter.Int32BitsToSingle(bf16 << 16);
        }
        return result;
    }

    // ═══════════════════════════════════════════════════════════
    //  CoreML
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Create from CoreML model (.mlmodel).
    /// Parses protobuf structure and extracts neural network layers.
    /// </summary>
    public static InferenceSession CreateFromCoreML(
        Accelerator accelerator, byte[] coremlBytes,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("parse", 0);
        var model = CoreML.CoreMLParser.Parse(coremlBytes);

        onProgress?.Invoke("compile", 0);
        var registry = new Operators.OperatorRegistry(accelerator);

        // Build graph from CoreML layers — use actual model input/output names
        var graph = new Graph.ModelGraph { Name = $"CoreML (v{model.SpecVersion})" };

        // Use model's declared input names, or fall back to "input"
        string inputName = model.InputNames.Count > 0 ? model.InputNames[0] : "input";
        string outputName = model.OutputNames.Count > 0 ? model.OutputNames[0] : "output";

        // Infer input shape from first conv/linear layer's weight dimensions
        int[] inputShape = new[] { 1, 3, 224, 224 }; // default
        foreach (var layer in model.Layers)
        {
            if (layer.Weights != null && layer.LayerType is "convolution" or "innerProduct")
            {
                int wLen = layer.Weights.Length;
                // Conv weights are [outC, inC, kH, kW] — if sqrt is integer, likely a square kernel
                int sqLen = (int)Math.Sqrt(wLen);
                if (sqLen > 3) inputShape = new[] { 1, 3, 224, 224 };
                break;
            }
        }

        // Infer output shape from last layer's output count
        int[] outputShape = new[] { 1, 1000 }; // default for classifiers
        for (int i = model.Layers.Count - 1; i >= 0; i--)
        {
            if (model.Layers[i].Bias != null)
            {
                outputShape = new[] { 1, model.Layers[i].Bias.Length };
                break;
            }
        }

        graph.Inputs.Add(new Graph.GraphValueInfo { Name = inputName, Shape = inputShape });
        graph.Outputs.Add(new Graph.GraphValueInfo { Name = outputName, Shape = outputShape });

        // Map CoreML layers to ONNX operators
        string prev = inputName;
        for (int i = 0; i < model.Layers.Count; i++)
        {
            var layer = model.Layers[i];
            string outName = layer.Outputs.Count > 0 ? layer.Outputs[0] : $"layer_{i}_out";
            string inName = layer.Inputs.Count > 0 ? layer.Inputs[0] : prev;

            string? opType = layer.LayerType switch
            {
                "convolution" => "Conv",
                "innerProduct" => "MatMul",
                "batchnorm" => "BatchNormalization",
                "pooling" => "MaxPool",
                "softmax" => "Softmax",
                "activation" => "Relu",
                "add" => "Add",
                "multiply" => "Mul",
                "concat" => "Concat",
                "reshape" => "Reshape",
                "flatten" => "Flatten",
                "upsample" => "Resize",
                _ => "Identity"
            };

            graph.Nodes.Add(new Graph.GraphNode
            {
                OpType = opType,
                Inputs = new List<string> { inName },
                Outputs = new List<string> { outName }
            });
            prev = outName;
        }

        var compiler = new Graph.GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new Tensors.BufferPool(accelerator);
        var weights = new Dictionary<string, Tensors.Tensor>();

        // Upload extracted weights from CoreML layers
        onProgress?.Invoke("upload", 0);
        for (int i = 0; i < model.Layers.Count; i++)
        {
            var layer = model.Layers[i];
            if (layer.Weights != null)
            {
                string wName = $"{layer.Name}.weight";
                weights[wName] = pool.AllocatePermanent(layer.Weights, new[] { layer.Weights.Length }, wName);
                graph.Initializers[wName] = new[] { layer.Weights.Length };
            }
            if (layer.Bias != null)
            {
                string bName = $"{layer.Name}.bias";
                weights[bName] = pool.AllocatePermanent(layer.Bias, new[] { layer.Bias.Length }, bName);
                graph.Initializers[bName] = new[] { layer.Bias.Length };
            }
        }

        var executor = new Graph.GraphExecutor(accelerator, compiled, weights, new Dictionary<string, float[]>(), registry: registry);
        onProgress?.Invoke("ready", 100);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights) { ModelName = graph.Name };
    }

    // ═══════════════════════════════════════════════════════════
    //  TensorFlow GraphDef (.pb frozen graph)
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Create from a TensorFlow frozen graph (.pb).
    /// Parses the GraphDef protobuf, maps TF ops to ONNX equivalents,
    /// extracts Const weights, and builds an executable graph.
    /// </summary>
    public static InferenceSession CreateFromTFGraphDef(
        Accelerator accelerator, byte[] graphDefBytes,
        Action<string, int>? onProgress = null)
    {
        onProgress?.Invoke("parse", 0);
        var tfGraph = TensorFlow.TFGraphDefParser.Parse(graphDefBytes);

        onProgress?.Invoke("compile", 0);
        var registry = new Operators.OperatorRegistry(accelerator);
        var (graph, constants) = TensorFlow.TFGraphDefGraphBuilder.BuildGraph(tfGraph);

        var compiler = new Graph.GraphCompiler(registry);
        var compiled = compiler.Compile(graph);
        var pool = new Tensors.BufferPool(accelerator);
        var weights = new Dictionary<string, Tensors.Tensor>();

        // Upload constant tensors (weights/biases from Const nodes)
        onProgress?.Invoke("upload", 0);
        int uploaded = 0;
        foreach (var (name, data) in constants)
        {
            var shape = graph.Initializers.TryGetValue(name, out var s) ? s : new[] { data.Length };
            weights[name] = pool.AllocatePermanent(data, shape, name);
            uploaded++;
            onProgress?.Invoke("upload", (int)(100.0 * uploaded / constants.Count));
        }

        var executor = new Graph.GraphExecutor(accelerator, compiled, weights, new Dictionary<string, float[]>(), registry: registry);
        onProgress?.Invoke("ready", 100);
        return new InferenceSession(accelerator, registry, compiled, executor, pool, weights) { ModelName = graph.Name };
    }

    /// <summary>Run inference with named input tensors. Returns named output tensors.
    /// Recompiles for the actual input shape when it differs from the compile-time shape
    /// (dynamic-shape models such as autoregressive decoders).</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => ResolveExecutor(inputs).Run(inputs);

    /// <summary>Async inference — required for browser backends (WebGPU/WebGL/Wasm)
    /// which deadlock on synchronous Synchronize(). Periodically flushes GPU commands.
    /// Recompiles for the actual input shape when it differs from the compile-time shape.</summary>
    public Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
        => ResolveExecutor(inputs).RunAsync(inputs);

    /// <summary>
    /// Transformers.js-style async inference. Inputs are <see cref="Tensors.Tensor{T}"/>
    /// (non-owning views — caller manages the underlying buffers, an
    /// <see cref="Tensors.OwnedTensor{T}"/> converts implicitly). Outputs are
    /// <see cref="Tensors.OwnedTensor{T}"/> wrapped in an
    /// <see cref="Tensors.OwnedTensorMap{T}"/>: the caller fully owns the returned
    /// buffers and disposes them by wrapping the map in <c>using</c>. Internally each
    /// output is copied off the executor's pool-managed buffer to a fresh
    /// caller-owned buffer, so subsequent inference runs cannot mutate previously-
    /// returned tensors.
    /// </summary>
    public async Task<Tensors.OwnedTensorMap<float>> RunOwnedAsync(
        IDictionary<string, Tensors.Tensor<float>> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));

        // Convert generic Tensor<float> inputs back to the legacy non-generic Tensor
        // the executor accepts. Both wrap the same ArrayView<float, Stride1D.Dense>,
        // so this is metadata-only — no data movement.
        var legacyInputs = new Dictionary<string, Tensor>(inputs.Count);
        foreach (var kv in inputs)
            legacyInputs[kv.Key] = new Tensor(kv.Value.Data, kv.Value.Shape, kv.Value.Name);

        var executorOutputs = await _executor.RunAsync(legacyInputs);

        // Copy each output to a fresh caller-owned buffer. Tensors returned by the
        // executor view into pool-managed memory that may be reused by the next
        // RunAsync invocation — copying gives the caller buffers with independent
        // lifetimes. CopyFrom is a GPU-to-GPU copy on every backend, no host readback.
        var owned = new Dictionary<string, Tensors.OwnedTensor<float>>(executorOutputs.Count);
        foreach (var kv in executorOutputs)
        {
            var src = kv.Value;
            var buf = _accelerator.Allocate1D<float>(src.ElementCount);
            await buf.View.CopyFromAsync(src.Data);
            owned[kv.Key] = new Tensors.OwnedTensor<float>(buf, src.Shape, kv.Key);
        }
        await _accelerator.SynchronizeAsync();

        return new Tensors.OwnedTensorMap<float>(owned);
    }

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

    /// <summary>
    /// Download bytes using stream-based chunked reading when possible.
    /// For desktop backends: uses ResponseHeadersRead for streaming download with progress.
    /// For browser WASM: falls back to ReadAsByteArrayAsync which works with fetch API.
    /// Yields periodically to keep the UI thread responsive during long downloads.
    /// </summary>
    /// <summary>
    /// Download bytes using stream-based chunked reading with progress.
    /// Avoids OOM for large files in browser WASM. Yields periodically for UI responsiveness.
    /// </summary>
    public static async Task<byte[]> DownloadBytesChunkedAsync(HttpClient http, string url,
        Action<string, int>? onProgress = null)
    {
        try
        {
            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var contentLength = response.Content.Headers.ContentLength;
            using var stream = await response.Content.ReadAsStreamAsync();

            // If content length is known, read in 1MB chunks with progress
            if (contentLength.HasValue && contentLength.Value > 0)
            {
                var result = new byte[contentLength.Value];
                int totalRead = 0;
                int lastPercent = -1;
                int yieldCounter = 0;
                var buffer = new byte[1024 * 1024]; // 1MB chunks
                while (totalRead < result.Length)
                {
                    int read = await stream.ReadAsync(buffer, 0, Math.Min(buffer.Length, result.Length - totalRead));
                    if (read == 0) break;
                    Buffer.BlockCopy(buffer, 0, result, totalRead, read);
                    totalRead += read;

                    int pct = (int)(totalRead * 100L / result.Length);
                    if (pct != lastPercent)
                    {
                        lastPercent = pct;
                        onProgress?.Invoke("download", pct);
                    }

                    // Yield every 4MB to keep UI responsive
                    if (++yieldCounter % 4 == 0)
                        await Task.Yield();
                }

                // Verify we got data — browser WASM may return 0 bytes with streaming
                if (totalRead > 0)
                    return totalRead == result.Length ? result : result[..totalRead];
            }
            else
            {
                // Unknown length: stream into expanding MemoryStream
                using var ms = new MemoryStream();
                var unknownBuffer = new byte[1024 * 1024];
                int unknownYield = 0;
                int bytesRead;
                while ((bytesRead = await stream.ReadAsync(unknownBuffer, 0, unknownBuffer.Length)) > 0)
                {
                    ms.Write(unknownBuffer, 0, bytesRead);
                    if (++unknownYield % 4 == 0)
                        await Task.Yield();
                }
                if (ms.Length > 0)
                    return ms.ToArray();
            }
        }
        catch { /* Streaming download not supported — fall through to ReadAsByteArrayAsync */ }

        // Fallback: standard byte array download (works on all platforms including browser WASM)
        onProgress?.Invoke("download", 0);
        using var fallbackResponse = await http.GetAsync(url);
        fallbackResponse.EnsureSuccessStatusCode();
        var bytes = await fallbackResponse.Content.ReadAsByteArrayAsync();
        onProgress?.Invoke("download", 100);
        return bytes;
    }

    public void Dispose()
    {
        _executor.Dispose();
        // Dispose per-shape recompiled executors (each owns its own intermediate buffer pool).
        foreach (var exec in _shapeExecutors.Values)
            try { exec.Dispose(); } catch { }
        _shapeExecutors.Clear();
        _shapeExecutorLru.Clear();
        _pool.Dispose();
        _registry.Dispose();
        // Dispose buffers not tracked by the pool (GGUF quantized bytes, etc.)
        if (_ownedBuffers != null)
            foreach (var buf in _ownedBuffers)
                try { buf.Dispose(); } catch { }
    }

    // Pre-extract pads tensors (Pad opset >= 11) into runtime constants at session init.
    // Without this, GraphExecutor must GPU-readback each Pad node's pads tensor at execute time
    // (~50ms/Pad on Wasm) just to size the output buffer correctly. Pads are tiny (2*rank ints)
    // so safe to cache once at session creation. Closes StyleMosaic Wasm 2GiB cap by removing
    // per-execute readback allocation pressure. Geordi 2026-05-04 endorsed approach.
    private static void PreExtractPads(
        Onnx.OnnxModelProto parsedModel,
        Dictionary<string, float[]> cpuSmallWeights,
        Dictionary<string, float[]> constantFloatValues,
        Graph.ModelGraph graph)
    {
        foreach (var node in parsedModel.Graph.Nodes)
        {
            if (node.OpType != "Pad" || node.Inputs.Count < 2) continue;
            var padsName = node.Inputs[1];
            if (string.IsNullOrEmpty(padsName) || constantFloatValues.ContainsKey(padsName)) continue;

            float[]? padsData = null;
            // Layer 1: cpuSmallWeights covers <=64-elem initializers + Constant-node outputs already populated upstream.
            if (cpuSmallWeights.TryGetValue(padsName, out var fromSmall))
            {
                padsData = fromSmall;
            }
            else
            {
                // Layer 2: oversized initializer (defensive - pads are typically 2*rank ints, never >64 in practice).
                Onnx.OnnxTensorProto? init = null;
                foreach (var i in parsedModel.Graph.Initializers)
                {
                    if (i.Name == padsName) { init = i; break; }
                }
                if (init != null && init.DataLocation != 1)
                {
                    padsData = init.ToFloatArray();
                }
                else
                {
                    // Layer 3: Constant node output not captured upstream (defensive - Constant loop above already covers).
                    Onnx.OnnxNodeProto? constNode = null;
                    foreach (var n in parsedModel.Graph.Nodes)
                    {
                        if (n.OpType == "Constant" && n.Outputs.Count > 0 && n.Outputs[0] == padsName) { constNode = n; break; }
                    }
                    if (constNode != null)
                    {
                        var valueAttr = constNode.Attributes.FirstOrDefault(a => a.Name == "value");
                        if (valueAttr?.T != null)
                            padsData = valueAttr.T.ToFloatArray();
                    }
                }
            }

            if (padsData == null) continue;
            // Seed constantFloatValues only - that flows into runtimeConstants in GraphExecutor
            // and lets the runtime Pad shape resolver hit path 1. Do NOT mirror to
            // graph.ConstantData / graph.FloatConstantData; those feed the optimizer's
            // compile-time fold pass which is tested for tensors that are also in
            // graph.Initializers. Adding Constant-node-sourced entries there puts the
            // optimizer in a state it has not been validated against and tripped
            // StyleMosaic WebGPU regular-path execution at rc.24 (verified via bisect).
            constantFloatValues[padsName] = padsData;
        }
    }
}

/// <summary>Supported model file formats.</summary>
public enum ModelFormat
{
    Unknown,
    ONNX,
    TFLite,
    GGUF,
    SafeTensors,
    TFGraphDef,
    PyTorch,
    CoreML,
    SPZ,
    PLY,
    GLTF,
    OBJ,
}
