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
    /// <summary>Weight bytes uploaded straight from JS to the GPU (zero-copy, never entered the .NET heap) during
    /// this session's streaming load. 0 unless the model was loaded via a JS-backed stream on a browser backend.</summary>
    public long ZeroCopyWeightBytes => _pool?.ZeroCopyWeightBytes ?? 0;
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

    /// <summary>Diagnostic: log per-tensor weight-upload timing (only tensors &gt;50ms) + a stream-vs-.NET
    /// summary, to attribute a slow model load to torrent seek-back waits vs .NET materialization. Off by
    /// default (a healthy load is silent anyway). Enable from a consumer for a one-run load-perf trace.</summary>
    public static bool TraceWeightLoad { get; set; }

    /// <summary>DIAGNOSTIC: wall-clock ms of the most recent per-shape recompile (Compile + executor
    /// build). 0 when the last Run hit the base/cached executor (no recompile). Lets the decode loop
    /// attribute per-step cost to CPU recompile vs GPU forward.</summary>
    public double LastRecompileMs { get; private set; }

    private bool _cacheShapeReadbacks;
    /// <summary>Cache the mid-graph shape-derived runtime-constant readbacks (the ≤64-elem shape/scalar
    /// tensors read back per node) so repeated same-shape inference skips ~643 GPU round-trips/forward
    /// (~7.8s on DistilGPT-2 WebGPU). Safe ONLY for models whose small captured tensors are shape-
    /// derived not data-derived — true for transformer decoders. Set by autoregressive pipelines
    /// before the decode loop. Default off; general inference is unchanged. See
    /// <see cref="GraphExecutor.CacheShapeReadbacks"/>.</summary>
    public bool CacheShapeReadbacks
    {
        get => _cacheShapeReadbacks;
        set
        {
            _cacheShapeReadbacks = value;
            _executor.CacheShapeReadbacks = value;
            foreach (var e in _shapeExecutors.Values) e.CacheShapeReadbacks = value;
        }
    }

    /// <summary>DIAGNOSTIC: GPU intermediate-buffer count of the executor used by the most recent
    /// Run/RunAsync. A per-shape executor rebuilt each step starts fresh (re-allocating every
    /// intermediate buffer) — a high count that resets each step points the per-step cost at
    /// fresh-pool churn rather than CPU recompile or forward compute.</summary>
    public int LastExecutorBufferCount { get; private set; }

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

    /// <summary>Full compiled-node view (OpType + input/output tensor names) by index — for the tiled VAE
    /// decoder's weight walk (map conv/γ/β tensor names to nodes). The arrays are the live node's; do not mutate.</summary>
    public (string opType, string[] inputs, string[] outputs) GetNode(int idx)
    {
        var n = _compiled.Nodes[idx];
        return (n.OpType, n.InputNames, n.OutputNames);
    }

    /// <summary>Model name (from graph metadata).</summary>
    public string ModelName { get; private set; } = "";

    /// <summary>Access to the underlying GraphExecutor (for KV cache management).</summary>
    public GraphExecutor Executor => _executor;

    /// <summary>The accelerator this session runs on (used by the CUDA-graph capture path).</summary>
    public Accelerator Accelerator => _accelerator;

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
        LastRecompileMs = 0; // set by RecompileForShapes on a cache miss; stays 0 for base/cached hits
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
        var recompileSw = System.Diagnostics.Stopwatch.StartNew();
        var graph = _recompileGraph!;
        // Reset to the clean pre-fold constant seeds (the prior compile may have folded seq-specific
        // values onto the graph when optimization is off), then pin the actual input shapes.
        graph.ConstantData = _recompileConstSeed != null ? new Dictionary<string, int[]>(_recompileConstSeed) : new();
        graph.FloatConstantData = _recompileFloatSeed != null ? new Dictionary<string, float[]>(_recompileFloatSeed) : new();
        foreach (var inp in graph.Inputs)
            if (inputs.TryGetValue(inp.Name, out var t))
                inp.Shape = t.Shape.ToArray();

        var compiled = new GraphCompiler(_registry) { EnableOptimization = _recompileEnableOptimization }.Compile(graph);
        // The recompiled executor must carry the SAME quantized byte-view map as the base executor:
        // quantized weights are session-lifetime GPU buffers (uploaded once at load), and without the
        // map every quantized MatMul/Gather silently falls back to the F32 path against a ShapeOnly
        // tensor's EMPTY view — a CUDA illegal memory access at the first quantized node. This was
        // the gemma4 multi-token (seq>1) fault: seq=1 matched the base compile shape and never
        // recompiled, so only recompiled (seq>1) executors faulted. 2026-06-12.
        var exec = new GraphExecutor(_accelerator, compiled, _weights, _recompileFloatSeed,
            quantizedWeights: _executor.QuantizedWeights, registry: _registry)
        {
            Format = _executor.Format,
            CacheShapeReadbacks = _cacheShapeReadbacks,
            ActivationDtype = _executor.ActivationDtype, // carry the activation precision to the recompiled executor
        };
        recompileSw.Stop();
        LastRecompileMs = recompileSw.Elapsed.TotalMilliseconds;
        if (VerboseLogging)
            Console.WriteLine($"[InferenceSession] Recompiled for shapes [{string.Join("; ", inputs.Select(kv => $"{kv.Key}:[{string.Join(",", kv.Value.Shape)}]"))}] — {compiled.Nodes.Length} nodes in {LastRecompileMs:F0}ms");
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
        // PREFER the streaming path: hub.OpenStreamAsync hands back a BlobStream over the OPFS cache
        // entry, which is an IJSReadStream - so the graph structure is parsed from the stream and each
        // weight is seeked to and uploaded JS->GPU without the model ever landing on the .NET/WASM managed
        // heap. hub.LoadAsync below returns the whole file as a byte[]; for a 300 MB+ model that is the
        // "bulk bytes stay in JS" rule broken on every browser load, and it is what made loads OOM under
        // memory pressure.
        //
        // External-data models (weights in a sibling .onnx_data file) still take the byte[] path: resolving
        // those needs the parsed model plus a second file, which the block below already handles.
        //
        // ⚠️ Known inefficiency, stated rather than hidden: when the model has no external data this parses
        // the stream TWICE - once to answer "does it have external data", then again inside
        // CreateFromOnnxStreamAsync. Both reads come from the OPFS-cached blob so it is cheap, but the right
        // fix is a stream entry point that accepts an already-parsed model.
        var hubStream = await hub.OpenStreamAsync(repoId, filename, revision).ConfigureAwait(false);
        if (hubStream != null)
        {
            await using (hubStream)
            {
                var probe = await Onnx.OnnxParser.ParseFromStreamAsync(hubStream, 1024 * 1024).ConfigureAwait(false);
                if (!Onnx.OnnxLoader.HasExternalData(probe))
                {
                    hubStream.Position = 0;
                    onProgress?.Invoke("download", 100);
                    return await CreateFromOnnxStreamAsync(accelerator, hubStream, onProgress, inputShapes)
                        .ConfigureAwait(false);
                }
            }
        }

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
        Stream? externalDataStream = null,
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
        // Denominator for INCREMENTAL upload progress. This loop seeks+reads each weight from the stream,
        // so on a torrent/HTTP stream it IS the heavy multi-hundred-MB transfer. Without per-weight
        // progress the caller's bar sat frozen at the start of "upload" for the entire download, then
        // jumped to done — the "/text-gen stuck at 15%" report.
        int totalWeights = Math.Max(1, parsedModel.Graph.Initializers.Count);
        int lastUploadPct = -1;

        // ── f16 weight gating ──
        // An fp16 initializer is loaded as HALF (ILGPU.Half — half the GPU bytes) only if EVERY node that
        // consumes it does so as the WEIGHT operand (input index 1) of a HALF-CAPABLE op: MatMul, Gemm
        // (both transB layouts — native low-p + transposed-low-p kernels), or a standard NCHW group-1 Conv.
        // Anything else (shared weight, non-weight use, depthwise/grouped Conv, ConvTranspose — not wired for
        // half yet) keeps it fp32. The MatMul/Conv operator guards are
        // the runtime safety net: a mis-gated half weight throws clearly instead of corrupting. (Generic
        // kernels later will let the loader half ALL fp16 weights with no gating; until then, conservative.)
        var halfEligible = new HashSet<string>();
        var halfBlocked = new HashSet<string>();
        // Every initializer a compiled KERNEL actually consumes as an input. An initializer NOT in this set
        // is used only by CPU-side shape inference (a folded shape/scalar constant) or is unreferenced — it
        // needs NO GPU buffer, so its upload (a CopyFromCPU per tensor on WebGPU) is pure waste (Captain
        // 2026-07-05: "don't upload what the GPU never consumes"). We skip those below. CPU shape inference
        // reads its values from the SEPARATE cpuSmallWeights/ConstantData, so skipping the GPU upload is safe.
        var gpuConsumed = new HashSet<string>();
        // Diagnostic (VerboseLogging): for each blocked weight, the op-type(s) that force it off the native
        // low-p path — so we can see which op still needs a native low-p kernel to widen the gate.
        var blockingOps = new Dictionary<string, HashSet<string>>();
        foreach (var node in compiled.Nodes)
        {
            var ins = node.InputNames;
            for (int oi = 0; oi < ins.Length; oi++)
            {
                var inName = ins[oi];
                if (string.IsNullOrEmpty(inName)) continue;
                gpuConsumed.Add(inName);
                int convGroup = node.OpType == "Conv" && node.Attributes.TryGetValue("group", out var gv) && gv is long gl ? (int)gl : 1;
                bool okAsWeight = oi == 1 && (node.OpType == "MatMul" || node.OpType == "Gemm" || (node.OpType == "Conv" && convGroup == 1));
                if (okAsWeight) halfEligible.Add(inName);
                else
                {
                    halfBlocked.Add(inName);
                    if (!blockingOps.TryGetValue(inName, out var ops)) blockingOps[inName] = ops = new HashSet<string>();
                    ops.Add($"{node.OpType}[in{oi}]");
                }
            }
        }
        halfEligible.ExceptWith(halfBlocked); // used EXCLUSIVELY as a half-capable weight operand
        int halfLoaded = 0;
        // no-needless-conversion diagnostic: fp16 weights forced to f32 because a consumer has no native low-p path.
        int blockedFp16Count = 0; long blockedFp16Elems = 0;
        var blockedByOp = new Dictionary<string, (int count, long elems)>();

        // Stream weights to GPU: large tensors are seeked to + chunk-uploaded straight from the stream
        // (never materialized); small/inline tensors use the in-memory chunked/standard path.
        // WeightLoadTrace: attribute a slow load to torrent seek-back waits (stream branch) vs .NET
        // materialization (chunked branch). Only anomalies (>50ms) log; a healthy load is silent.
        long _prevOff = -1; double _streamMs = 0, _chunkedMs = 0; int _nStream = 0, _nChunked = 0;
        int skippedCpuOnly = 0; long _maxChunkedBytes = 0;
        // Fail-loud (Geordi's ILGPU 4.17.2-local.8 guard, my ILGPU-side placement call): during weight
        // upload, ANY host CopyFromCPU over 64KB throws — i.e. a bulk weight that regressed onto the .NET
        // path instead of streaming JS-side. Largest LEGIT materialized constant is ~308B (measured), so 64KB
        // never false-fires on a real weight; a regression (KB-MB) screams in PMT. ENFORCE ONLY when the
        // source is a browser zero-copy stream (IJSReadStream = the hub in production) — where zero-copy IS
        // available so a bulk CopyFromCPU is a genuine regression. A plain .NET stream (MemoryStream in
        // equivalence tests, or desktop) legitimately CopyFromCPUs (no zero-copy source), so don't guard it.
        bool _guardHostCopy = stream is SpawnDev.SpawnJS.Toolbox.IJSReadStream;
        if (_guardHostCopy) SpawnDev.ILGPU.BrowserBufferPolicy.StrictHostCopyMaxBytes = 65536;
        try
        {
        foreach (var (name, tensor) in Onnx.OnnxLoader.StreamTensorsFromParsed(parsedModel))
        {
            if (!graph.Initializers.TryGetValue(name, out var shape)) continue;
            // No compiled kernel consumes this initializer → it is a CPU-only shape/scalar constant (or
            // unreferenced). CPU shape inference already has its value (cpuSmallWeights); a GPU buffer +
            // upload would be pure waste (the many tiny CopyFromCPU on WebGPU). Skip it.
            if (!gpuConsumed.Contains(name)) { skippedCpuOnly++; continue; }
            int expectedElems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
            long _t0 = TraceWeightLoad ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            string _branch;
            // fp16-source (dtype 10) weight, consumed exclusively by a half-capable op as its weight: keep
            // it fp16 on the GPU (half the bytes). Only fp16 SOURCE — never downcast a fp32 weight to fp16
            // (that would lose precision the model expects). Streaming path only (large weights = the win).
            if (halfEligible.Contains(name) && tensor.DataType == 10 && tensor.RawDataStreamOffset >= 0)
            {
                _branch = "half-stream";
                var halfW = await pool.AllocateHalfWeightFromStreamAsync(
                    stream, tensor.RawDataStreamOffset, tensor.RawDataLength, tensor.DataType, shape, name, ct).ConfigureAwait(false);
                gpuWeights[name] = Tensor.FromHalf(halfW);
                halfLoaded++;
            }
            else if (tensor.RawDataStreamOffset >= 0)
            {
                _branch = "f32-stream";
                if (tensor.DataType == 10) // a BLOCKED fp16 weight: no native consumer -> downcast to f32 (the unpacking)
                {
                    blockedFp16Count++; blockedFp16Elems += expectedElems;
                    if (blockingOps.TryGetValue(name, out var bops))
                        foreach (var op in bops) { var c = blockedByOp.GetValueOrDefault(op); blockedByOp[op] = (c.count + 1, c.elems + expectedElems); }
                }
                gpuWeights[name] = await pool.AllocatePermanentFromStreamAsync(
                    stream, tensor.RawDataStreamOffset, tensor.RawDataLength, tensor.DataType, shape, name, ct).ConfigureAwait(false);
            }
            else if (tensor.ElementCount == 0 && expectedElems > 0)
            { _branch = "empty"; gpuWeights[name] = pool.AllocatePermanent(new float[expectedElems], shape, name); }
            else
            { _branch = "NET-chunked"; gpuWeights[name] = pool.AllocatePermanentChunked(tensor, shape, name); }
            loaded++;
            if (TraceWeightLoad)
            {
                double _ms = System.Diagnostics.Stopwatch.GetElapsedTime(_t0).TotalMilliseconds;
                long _off = tensor.RawDataStreamOffset;
                if (_branch == "NET-chunked" || _branch == "empty") { _chunkedMs += _ms; _nChunked++; _maxChunkedBytes = Math.Max(_maxChunkedBytes, (long)expectedElems * 4); } else { _streamMs += _ms; _nStream++; }
                if (_ms > 50)
                    Console.WriteLine($"[WL SLOW] {_branch,-11} off={_off,-11} dOff={(_prevOff >= 0 && _off >= 0 ? _off - _prevOff : 0),-11} bytes={tensor.RawDataLength,-9} dt={tensor.DataType} {_ms,7:F0}ms  {name}");
                if (_off >= 0) _prevOff = _off;
            }
            int uploadPct = (int)Math.Min(99, loaded * 100L / totalWeights);
            if (uploadPct != lastUploadPct) { lastUploadPct = uploadPct; onProgress?.Invoke("upload", uploadPct); }
        }
        }
        finally { if (_guardHostCopy) SpawnDev.ILGPU.BrowserBufferPolicy.StrictHostCopyMaxBytes = -1; }
        // Drain once + free the deferred fp16-upcast temp buffers (one drain for the whole load, not per weight).
        await pool.FlushPendingFp16ConvertsAsync().ConfigureAwait(false);
        if (TraceWeightLoad)
            Console.WriteLine($"[WL SUMMARY] stream(zero-copy)={_nStream} tensors {_streamMs:F0}ms | NET-chunked={_nChunked} tensors {_chunkedMs:F0}ms (maxMaterialized={_maxChunkedBytes}B) | skipped(CPU-only, no GPU upload)={skippedCpuOnly}  → time is in the larger bucket");
        if (VerboseLogging)
        {
            Console.WriteLine($"[InferenceSession] f16 weights: {halfLoaded} loaded as fp16 (half GPU bytes) of {halfEligible.Count} half-eligible; the rest fp32.");
            if (blockedFp16Count > 0)
            {
                double mb16 = blockedFp16Elems * 2 / 1048576.0, mb32 = blockedFp16Elems * 4 / 1048576.0;
                var top = string.Join(", ", blockedByOp.OrderByDescending(k => k.Value.elems)
                    .Take(8).Select(k => $"{k.Key} x{k.Value.count} ({k.Value.elems * 2 / 1048576.0:F1}MB)"));
                Console.WriteLine($"[InferenceSession] no-native-path: {blockedFp16Count} fp16 weights downcast to f32 " +
                    $"({mb16:F1}MB fp16 source -> {mb32:F1}MB f32 on GPU). Top blocking ops: {top}. " +
                    "Give these ops a native low-p path to keep their weights native.");
            }
        }

        // External-data weights (DataLocation==1 → a SEPARATE file, e.g. DAv3's model.onnx_data).
        // StreamTensorsFromParsed above skips these, so stream them here from the provided external-data stream
        // at their ONNX-recorded offsets — zero-copy JS→GPU (fp32 via CopyFromStreamAsync → CopyFromJS), never
        // the managed heap. Before this, external-data ONNX models (DAv3 + most onnx-community exports) could
        // ONLY load via the byte[] path (whole file into the WASM heap) — this is the JS-data-rule-compliant path.
        if (externalDataStream != null)
        {
            int extLoaded = 0;
            foreach (var init in parsedModel.Graph.Initializers)
            {
                if (init.DataLocation != 1 || init.ExternalData == null) continue;
                if (!graph.Initializers.TryGetValue(init.Name, out var shape)) continue;
                if (gpuWeights.ContainsKey(init.Name)) continue;
                int elems = shape.Length > 0 ? shape.Aggregate(1, (a, b) => a * b) : 1;
                long extOffset = init.ExternalData.TryGetValue("offset", out var eo) && long.TryParse(eo, out var eov) ? eov : 0;
                long extLen = init.ExternalData.TryGetValue("length", out var el) && long.TryParse(el, out var elv)
                    ? elv : (long)elems * (init.DataType == 10 ? 2 : 4);
                gpuWeights[init.Name] = await pool.AllocatePermanentFromStreamAsync(
                    externalDataStream, extOffset, (int)extLen, init.DataType, shape, init.Name, ct).ConfigureAwait(false);
                loaded++; extLoaded++;
            }
            if (VerboseLogging) Console.WriteLine($"[InferenceSession] external-data (stream): {extLoaded} weights streamed from model.onnx_data (zero-copy)");
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
        Action<string, int>? onProgress = null, bool acceptInputsEmbeds = false)
    {
        // Parse GGUF
        onProgress?.Invoke("parse", 0);
        var ggufModel = GGUF.GGUFParser.Parse(ggufBytes);
        onProgress?.Invoke("parse", 100);

        // Build transformer graph from architecture metadata
        onProgress?.Invoke("build_graph", 0);
        var (graph, cpuWeightsAll, quantizedWeightsTyped, transposeOnUpload, lowPWeightsTyped) = GGUF.GGUFGraphBuilder.BuildGraph(ggufModel, acceptInputsEmbeds);
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

        // Snapshot CLEAN (pre-fold) constants for dynamic-shape recompiles (see the streaming path).
        var constSeed = graph.ConstantData != null ? new Dictionary<string, int[]>(graph.ConstantData) : null;
        var floatSeed = graph.FloatConstantData != null ? new Dictionary<string, float[]>(graph.FloatConstantData) : null;

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
        var quantizedTypes = new Dictionary<string, GGUF.GGMLType>();
        var quantizedBuffers = new List<MemoryBuffer1D<byte, Stride1D.Dense>>(); // keep alive
        // Transpose-on-upload temps that must outlive this sync entry point on browser
        // backends (Synchronize() only flushes there) — owned by the session.
        var transposeTemps = new List<IDisposable>();
        // Native BF16/F16 linear-weight transposed buffers (FromLowP-backed) — owned by the session.
        var lowPBuffers = new List<IDisposable>();
        // Tied-embed aliases share one byte[]; dedupe by reference so the compressed
        // table is uploaded ONCE and serves both Gather and the LM-head MatMul.
        var uploadedQuant = new Dictionary<byte[], ArrayView1D<byte, Stride1D.Dense>>(
            ReferenceEqualityComparer.Instance);
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (graph.Initializers.TryGetValue(name, out var shape))
            {
                if (quantizedWeightsTyped.TryGetValue(name, out var qw))
                {
                    if (!uploadedQuant.TryGetValue(qw.Bytes, out var qView))
                    {
                        // Pad to a 4-byte multiple: the fused kernels read the bytes as
                        // packed int32 words (Cast<byte,int> truncates a ragged tail) and
                        // WebGPU requires 4-byte buffer sizes anyway.
                        int padded = (qw.Bytes.Length + 3) & ~3;
                        var qBuf = accelerator.Allocate1D<byte>(padded);
                        qBuf.View.SubView(0, qw.Bytes.Length).CopyFromCPU(qw.Bytes);
                        qView = qBuf.View;
                        quantizedBuffers.Add(qBuf);
                        uploadedQuant[qw.Bytes] = qView;
                    }
                    gpuQuantizedWeights[name] = qView;
                    quantizedTypes[name] = qw.Type;
                    // Shape-tracking entry only - the floats never exist. ShapeOnly
                    // carries no buffer (a full F32 rent here would be ~4GB for a
                    // gemma-class embedding) and fails loudly if any op reads .Data.
                    gpuWeights[name] = Tensor.ShapeOnly(shape, name);
                }
                else if (lowPWeightsTyped.TryGetValue(name, out var lp))
                {
                    // Native BF16/F16 linear weight: upload packed bytes, transpose in the element dtype to
                    // the declared [K, N], wrap as a FromLowP tensor (NO f32 upcast — half the VRAM). The
                    // src [N,K] byte temp is freed after the transpose (browser: outlive the sync entry).
                    int padded = (lp.Bytes.Length + 3) & ~3;
                    var srcBuf = accelerator.Allocate1D<byte>(padded);
                    srcBuf.View.SubView(0, lp.Bytes.Length).CopyFromCPU(lp.Bytes);
                    gpuWeights[name] = WrapLowPWeight(accelerator, registry.Transpose, srcBuf, lp, shape, name, lowPBuffers);
                    accelerator.Flush();
                    if (accelerator.AcceleratorType is AcceleratorType.Wasm
                        or AcceleratorType.WebGL or AcceleratorType.WebGPU)
                        transposeTemps.Add(srcBuf);
                    else
                        srcBuf.Dispose();
                }
                else if (data.Length > 0)
                {
                    if (transposeOnUpload.Contains(name))
                    {
                        // GGUF storage is [N rows][K contig] but the declared MatMul B is
                        // [K, N] row-major: one-time GPU transpose at load (never a CPU
                        // pass - unacceptable in interpreted Blazor WASM). shape = [K, N]
                        // declared; the raw floats are its reverse.
                        var final = pool.AllocatePermanent(shape, name);
                        var temp = accelerator.Allocate1D(data);
                        registry.Transpose.Transpose(temp.View, final.Data,
                            new[] { shape[1], shape[0] }, new[] { 1, 0 });
                        // Flush so the transpose is no longer pending in a command encoder.
                        // Flush() (sync submit) is the 4.12.0 contract here — NOT Synchronize(),
                        // which now throws on browser (submit+wait is desktop-only). The intent was
                        // always submit-only, as the note below describes.
                        accelerator.Flush();
                        // On the BROWSER backends this only FLUSHES — it cannot drain
                        // in-flight work on the single Blazor thread, so the transpose may still
                        // be queued and the temp buffer must OUTLIVE this sync entry point.
                        // Disposing it here made the deferred Wasm dispatch read a freed
                        // SharedArrayBuffer region → "RangeError: offset is out of bounds" at
                        // the first real SynchronizeAsync (2026-06-12). Desktop Synchronize()
                        // drains, so there the temp can be freed immediately.
                        if (accelerator.AcceleratorType is AcceleratorType.Wasm
                            or AcceleratorType.WebGL or AcceleratorType.WebGPU)
                            transposeTemps.Add(temp);
                        else
                            temp.Dispose();
                        gpuWeights[name] = final;
                    }
                    else
                    {
                        gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                    }
                }
                loaded++;
            }
        }
        onProgress?.Invoke("upload", 100);

        int qCount = gpuQuantizedWeights.Count;
        if (VerboseLogging) Console.WriteLine($"[InferenceSession] GGUF: {ggufModel.Name} ({ggufModel.Architecture}), {compiled.Nodes.Length} nodes, {loaded} weights ({qCount} quantized, {uploadedQuant.Count} buffers), {ggufModel.BlockCount} layers");

        registry.QuantizedWeightTypes = quantizedTypes.Count > 0 ? quantizedTypes : null;
        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues,
            quantizedWeights: gpuQuantizedWeights.Count > 0 ? gpuQuantizedWeights : null);
        onProgress?.Invoke("ready", 100);

        var ownedDisposables = quantizedBuffers.Cast<IDisposable>().Concat(transposeTemps).Concat(lowPBuffers).ToList();
        var session = new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name,
            _ownedBuffers = ownedDisposables.Count > 0 ? ownedDisposables : null
        };
        session.EnableShapeRecompilation(graph, constSeed, floatSeed, enableOptimization: true);
        return session;
    }

    /// <summary>
    /// Create an InferenceSession from a .gguf FILE PATH by STREAMING — the only way to load a model larger
    /// than ~2 GB (a single byte[] caps there; gemma4:12b is 7 GB). Opens an async, seekable FileStream and
    /// delegates to <see cref="CreateFromGGUFStreamAsync"/>.
    /// </summary>
    public static async Task<InferenceSession> CreateFromGGUFFileAsync(
        Accelerator accelerator, string ggufPath,
        Action<string, int>? onProgress = null, CancellationToken ct = default, bool acceptInputsEmbeds = false)
    {
        await using var fs = new FileStream(ggufPath, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 20, useAsync: true);
        return await CreateFromGGUFStreamAsync(accelerator, fs, onProgress, ct, acceptInputsEmbeds);
    }

    /// <summary>
    /// Create an InferenceSession from a SEEKABLE .gguf stream by STREAMING the weights to the GPU — never
    /// materializing the whole model as a byte[] (impossible past ~2 GB). The header is parsed async; small
    /// F32/F16 tensors (norms/scales) are read on demand; the bulk quantized weights (Q4_K/Q6_K, the 7 GB)
    /// stream tensor-by-tensor straight to GPU byte buffers via <see cref="BufferPool.AllocateQuantizedBytesFromStreamAsync"/>.
    /// The stream must outlive this call (it is read here) and be seekable. Mirrors <see cref="CreateFromGGUF"/>
    /// (byte[]); the only difference is the quantized upload path.
    /// </summary>
    public static async Task<InferenceSession> CreateFromGGUFStreamAsync(
        Accelerator accelerator, Stream stream,
        Action<string, int>? onProgress = null, CancellationToken ct = default, bool acceptInputsEmbeds = false)
    {
        if (!stream.CanSeek)
            throw new ArgumentException("CreateFromGGUFStreamAsync requires a seekable stream.", nameof(stream));

        onProgress?.Invoke("parse", 0);
        var ggufModel = await GGUF.GGUFParser.ParseHeaderAsync(stream, ct).ConfigureAwait(false);
        ggufModel.SourceStream = stream; // small F32/F16 tensors are read on demand during BuildGraph
        // Pre-read the small non-quantized tensors ASYNC so BuildGraph's synchronous dequant never does a
        // sync Stream.Read — which throws on an async-only browser stream (TorrentReadStream/OPFS). The big
        // quantized weights stream async to the GPU later (AllocateQuantizedBytesFromStreamAsync).
        await ggufModel.HydrateNonQuantizedAsync(ct).ConfigureAwait(false);
        onProgress?.Invoke("parse", 100);

        onProgress?.Invoke("build_graph", 0);
        var (graph, cpuWeightsAll, quantizedWeightsTyped, transposeOnUpload, lowPWeightsTyped) = GGUF.GGUFGraphBuilder.BuildGraph(ggufModel, acceptInputsEmbeds);
        onProgress?.Invoke("build_graph", 100);

        // Small constants (identical to CreateFromGGUF).
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

        // Snapshot the CLEAN (pre-fold) constants so dynamic-shape recompiles reseed correctly (decoders
        // run at a growing seq length — without recompilation the graph stays pinned to the seq=1 compile
        // shape and silently ignores all but the first token).
        var constSeed = graph.ConstantData != null ? new Dictionary<string, int[]>(graph.ConstantData) : null;
        var floatSeed = graph.FloatConstantData != null ? new Dictionary<string, float[]>(graph.FloatConstantData) : null;

        onProgress?.Invoke("compile", 0);
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry) { EnableOptimization = true }.Compile(graph);
        onProgress?.Invoke("compile", 100);

        onProgress?.Invoke("upload", 0);
        var pool = new BufferPool(accelerator);
        var gpuWeights = new Dictionary<string, Tensor>();
        var gpuQuantizedWeights = new Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>();
        var quantizedTypes = new Dictionary<string, GGUF.GGMLType>();
        var quantizedBuffers = new List<MemoryBuffer1D<byte, Stride1D.Dense>>();
        // Native BF16/F16 linear-weight transposed buffers (FromLowP-backed) — owned by the session.
        var lowPBuffers = new List<IDisposable>();
        // Dedup tied-embed aliases: stream-offset key (boxed long compares by value) for streamed tensors,
        // byte[] reference for any in-memory ones — one Dictionary<object,...> handles both.
        var uploadedQuant = new Dictionary<object, ArrayView1D<byte, Stride1D.Dense>>();
        int loaded = 0;
        foreach (var (name, data) in cpuWeightsAll)
        {
            if (!graph.Initializers.TryGetValue(name, out var shape)) continue;

            if (quantizedWeightsTyped.TryGetValue(name, out var qw))
            {
                object dedupKey = qw.StreamOffset >= 0 ? qw.StreamOffset : qw.Bytes;
                if (!uploadedQuant.TryGetValue(dedupKey, out var qView))
                {
                    MemoryBuffer1D<byte, Stride1D.Dense> qBuf;
                    if (qw.StreamOffset >= 0)
                        try { qBuf = await pool.AllocateQuantizedBytesFromStreamAsync(stream, qw.StreamOffset, qw.StreamByteSize, ct).ConfigureAwait(false); }
                        catch (InvalidDataException ex) { throw new InvalidDataException($"quantized tensor '{name}' (type={qw.Type}, shape=[{string.Join(",", shape)}]): {ex.Message}", ex); }
                    else
                    {
                        int padded = (qw.Bytes.Length + 3) & ~3;
                        qBuf = accelerator.Allocate1D<byte>(padded);
                        qBuf.View.SubView(0, qw.Bytes.Length).CopyFromCPU(qw.Bytes);
                    }
                    qView = qBuf.View;
                    quantizedBuffers.Add(qBuf);
                    uploadedQuant[dedupKey] = qView;
                }
                gpuQuantizedWeights[name] = qView;
                quantizedTypes[name] = qw.Type;
                gpuWeights[name] = Tensor.ShapeOnly(shape, name); // floats never exist (a Q6_K embed would be ~4 GB F32)
            }
            else if (lowPWeightsTyped.TryGetValue(name, out var lp))
            {
                // Native BF16/F16 linear weight: stream packed bytes straight to a GPU byte buffer, transpose
                // in the element dtype to the declared [K, N], wrap as a FromLowP tensor (NO f32 upcast = half
                // the VRAM + bandwidth). DRAIN before freeing the streamed byte temp (browser sync only flushes).
                MemoryBuffer1D<byte, Stride1D.Dense> srcBuf;
                try { srcBuf = await pool.AllocateQuantizedBytesFromStreamAsync(stream, lp.StreamOffset, lp.StreamByteSize, ct).ConfigureAwait(false); }
                catch (InvalidDataException ex) { throw new InvalidDataException($"low-precision tensor '{name}' (shape=[{string.Join(",", shape)}]): {ex.Message}", ex); }
                gpuWeights[name] = WrapLowPWeight(accelerator, registry.Transpose, srcBuf, lp, shape, name, lowPBuffers);
                await accelerator.SynchronizeAsync().ConfigureAwait(false);
                srcBuf.Dispose();
            }
            else if (data.Length > 0)
            {
                if (transposeOnUpload.Contains(name))
                {
                    var final = pool.AllocatePermanent(shape, name);
                    var temp = accelerator.Allocate1D(data);
                    registry.Transpose.Transpose(temp.View, final.Data, new[] { shape[1], shape[0] }, new[] { 1, 0 });
                    // DRAIN (not just flush) before disposing the temp: on browser backends a
                    // sync Synchronize() only flushes, so the transpose could still be in flight
                    // and a disposed temp becomes a freed-SharedArrayBuffer read (Wasm
                    // "RangeError: offset is out of bounds", 2026-06-12). This path is async —
                    // do it properly.
                    await accelerator.SynchronizeAsync().ConfigureAwait(false);
                    temp.Dispose();
                    gpuWeights[name] = final;
                }
                else
                {
                    gpuWeights[name] = pool.AllocatePermanent(data, shape, name);
                }
            }
            loaded++;
        }
        onProgress?.Invoke("upload", 100);

        if (VerboseLogging) Console.WriteLine($"[InferenceSession] GGUF(stream): {ggufModel.Name} ({ggufModel.Architecture}), {compiled.Nodes.Length} nodes, {loaded} weights ({gpuQuantizedWeights.Count} quantized, {uploadedQuant.Count} buffers), {ggufModel.BlockCount} layers");

        registry.QuantizedWeightTypes = quantizedTypes.Count > 0 ? quantizedTypes : null;
        var executor = new GraphExecutor(accelerator, compiled, gpuWeights, constantFloatValues,
            quantizedWeights: gpuQuantizedWeights.Count > 0 ? gpuQuantizedWeights : null);
        onProgress?.Invoke("ready", 100);

        var session = new InferenceSession(accelerator, registry, compiled, executor, pool, gpuWeights)
        {
            ModelName = graph.Name,
            _ownedBuffers = (quantizedBuffers.Count > 0 || lowPBuffers.Count > 0)
                ? quantizedBuffers.Cast<IDisposable>().Concat(lowPBuffers).ToList() : null
        };
        // Dynamic-shape recompilation: a Run at a growing decode length recompiles (CPU-only; GPU weights
        // are reused) rather than running the seq=1 compile shape and dropping all but the first token.
        session.EnableShapeRecompilation(graph, constSeed, floatSeed, enableOptimization: true);
        return session;
    }

    // ═══════════════════════════════════════════════════════════
    //  Native low-precision (BF16/F16) GGUF linear-weight upload
    // ═══════════════════════════════════════════════════════════
    // The GGUF loader keeps BF16/F16 linear weights NATIVE end-to-end: a packed-bytes GPU buffer is
    // reinterpreted as ArrayView<T> (zero-copy Cast), transposed in the element dtype from the on-disk
    // [N rows][K] storage to the declared MatMul-B [K, N], and wrapped as a FromLowP tensor (no f32
    // buffer). The MatMul/Gemm operators read it via MatMulLowPWeight<T> and decode at the MAC — half the
    // VRAM + upload bandwidth of the old GetTensorFloat32 upcast (no-needless-conversion).

    /// <summary>Reinterpret a packed [N,K] byte buffer as <typeparamref name="T"/>, transpose to the
    /// declared [K, N], and wrap as a FromLowP tensor. The permanent transposed output buffer is added to
    /// <paramref name="owned"/>; the caller owns <paramref name="byteBuf"/> (free it AFTER the transpose
    /// has drained).</summary>
    private static Tensor WrapLowPTransposed<T>(Accelerator acc, Kernels.TransposeKernel transpose,
        MemoryBuffer1D<byte, Stride1D.Dense> byteBuf, int[] shapeKN, Tensors.TensorDataType dtype,
        string name, List<IDisposable> owned) where T : unmanaged
    {
        var srcNK = byteBuf.View.Cast<byte, T>();                       // zero-copy reinterpret, [N*K] elements
        var outBuf = acc.Allocate1D<T>((long)shapeKN[0] * shapeKN[1]);  // declared [K, N]
        // input declared as [N, K] = [shapeKN[1], shapeKN[0]]; perm [1,0] -> [K, N].
        transpose.Transpose(srcNK, outBuf.View, new[] { shapeKN[1], shapeKN[0] }, new[] { 1, 0 });
        owned.Add(outBuf);
        return Tensors.Tensor.FromLowP(outBuf.View, dtype, shapeKN, name);
    }

    /// <summary>Dispatch <see cref="WrapLowPTransposed{T}"/> on the native element dtype.</summary>
    private static Tensor WrapLowPWeight(Accelerator acc, Kernels.TransposeKernel transpose,
        MemoryBuffer1D<byte, Stride1D.Dense> byteBuf, GGUF.GGUFLowPWeight lp, int[] shape, string name,
        List<IDisposable> owned) => lp.DType switch
        {
            Tensors.TensorDataType.BFloat16 => WrapLowPTransposed<global::ILGPU.BFloat16>(acc, transpose, byteBuf, shape, lp.DType, name, owned),
            Tensors.TensorDataType.Float16 => WrapLowPTransposed<global::ILGPU.Half>(acc, transpose, byteBuf, shape, lp.DType, name, owned),
            _ => throw new NotSupportedException($"GGUF native low-p weight '{name}': dtype {lp.DType} has no native upload path."),
        };

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
    {
        var exec = ResolveExecutor(inputs);
        var result = exec.Run(inputs);
        LastExecutorBufferCount = exec.AllocatedBufferCount;
        return result;
    }

    /// <summary>Async inference — required for browser backends (WebGPU/WebGL/Wasm): a synchronous
    /// Synchronize() only flushes (dispatches) the GPU queue and returns without awaiting (it does NOT
    /// deadlock), so you must SynchronizeAsync() to await GPU completion before a readback. Periodically
    /// awaits to drain GPU commands. Recompiles for the actual input shape when it differs from the
    /// compile-time shape.</summary>
    public async Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
    {
        var exec = ResolveExecutor(inputs);
        var result = await exec.RunAsync(inputs);
        LastExecutorBufferCount = exec.AllocatedBufferCount;
        return result;
    }

    // ── GGUF incremental decode (full-precision KV-cache) ──
    private Kernels.GGUFDecodeKVCache? _decodeCache;
    // LFM2 / short-conv mixer models also need a per-conv-layer conv-state cache (the KV cache only covers
    // attention). Auto-created in EnableGGUFDecode when the graph carries ShortConv nodes; null otherwise.
    private Kernels.ShortConvStateCache? _convStateCache;

    /// <summary>Tokens already cached (advances per <see cref="RunDecodeStepAsync"/>; 0 before prefill).</summary>
    public int DecodePastLen { get; private set; }

    /// <summary>The live conv-state cache (LFM2 / short-conv models), or null. Exposed for
    /// <see cref="WebGPUDecodeCapture"/>, which re-runs the decode graph at one cursor several times to
    /// discover its patch points and must snapshot/restore this SHIFT-REGISTER state around those probes -
    /// unlike the KV cache, re-running a cursor here is NOT idempotent.</summary>
    internal Kernels.ShortConvStateCache? ConvStateCache => _convStateCache;

    /// <summary>Enable incremental KV-cache decode with a caller-built full-precision cache (per-layer
    /// kvHeads/headDim matching the model's attention geometry). Turns the O(n^2) full-recompute decode
    /// into O(n): each step only computes the new token's K/V and attends it against the cached history.
    /// Resets the decode cursor. Gemma4 etc. via <see cref="GGUF.GGUFGraphBuilder"/> (FusedAttention nodes
    /// carry the "layer" tag the executor intercept needs).</summary>
    public void EnableGGUFDecode(Kernels.GGUFDecodeKVCache cache)
    {
        _decodeCache = cache ?? throw new ArgumentNullException(nameof(cache));
        DecodePastLen = 0;
        // Short-conv mixer models (LFM2): decode needs a conv-state cache so ShortConv layers see the prior
        // L-1 tokens' history (else every 1-token decode step zero-pads and diverges from full-recompute).
        // Auto-detect from the graph; inert (never created) for pure-attention models like gemma/qwen.
        bool hasShortConv = _recompileGraph?.Nodes.Any(n => n.OpType == "ShortConv") ?? false;
        if (hasShortConv)
            (_convStateCache ??= new Kernels.ShortConvStateCache(_accelerator, _registry.ShortConv)).Reset();
        else { _convStateCache?.Dispose(); _convStateCache = null; }
    }

    /// <summary>Reset the decode cursor to begin a fresh sequence (reuses the cache allocation).</summary>
    public void ResetGGUFDecode() { DecodePastLen = 0; _convStateCache?.Reset(); }

    /// <summary>Set the decode cursor to <paramref name="p"/> WITHOUT clearing the KV-cache contents — the
    /// prefix-cache reuse path. When tokens 0..p-1 already hold the bit-identical K/V from a previous request
    /// (same tokens at the same absolute positions), set the cursor to p and prefill only the new suffix
    /// p..end: the suffix tokens are written at p and attend the cached 0..p-1 history + themselves, exactly
    /// as a fresh full prefill would. RoPE uses the absolute position (kv_offset = DecodePastLen), so the
    /// cached prefix's positions match and the result is token-identical to a full re-prefill. Leaves the
    /// executor in the same valid mid-decode state as <see cref="ResetGGUFDecode"/>, just at position p.</summary>
    public void SetGGUFDecodePastLen(int p)
    {
        if (_decodeCache == null)
            throw new InvalidOperationException("Call EnableGGUFDecode(cache) before SetGGUFDecodePastLen.");
        if (p < 0) throw new ArgumentOutOfRangeException(nameof(p));
        DecodePastLen = p;
    }

    /// <summary>Detach the KV-cache: clears the session's reference and resets the cursor. Call before
    /// disposing the cache so the session never holds a dangling reference to freed GPU buffers.</summary>
    public void DisableGGUFDecode() { _decodeCache = null; DecodePastLen = 0; _convStateCache?.Dispose(); _convStateCache = null; }

    /// <summary>Run ONE decode/prefill step with the KV-cache active. The step's tokens are written at
    /// the current <see cref="DecodePastLen"/> and attended against all cached history; the cursor then
    /// advances by the input sequence length. Prefill = the first call with the full prompt (seq=N);
    /// subsequent calls feed exactly one new token (seq=1). The decode-mode flag is set on the resolved
    /// per-shape executor only for the duration of this run (cleared in finally) so normal runs are
    /// unaffected; the cache STATE persists on the session.</summary>
    public async Task<Dictionary<string, Tensor>> RunDecodeStepAsync(Dictionary<string, Tensor> inputs)
    {
        if (_decodeCache == null)
            throw new InvalidOperationException("Call EnableGGUFDecode(cache) before RunDecodeStepAsync.");
        var exec = ResolveExecutor(inputs);
        exec.DecodeKVCache = _decodeCache;
        exec.ConvStateCache = _convStateCache;   // LFM2 short-conv history (null for pure-attention models)
        exec.DecodePastLen = DecodePastLen;
        try
        {
            var result = await exec.RunAsync(inputs);
            LastExecutorBufferCount = exec.AllocatedBufferCount;
            // Advance the KV cursor by the step's sequence length. input_ids is [1, seq] (seq = last dim);
            // the multimodal inputs_embeds path is [1, seq, n_embd] (seq = the MIDDLE dim, not n_embd).
            int seq = inputs.TryGetValue("input_ids", out var t) ? t.Shape[^1]
                : inputs.TryGetValue("inputs_embeds", out var e) ? (e.Shape.Length >= 2 ? e.Shape[^2] : 1)
                : 1;
            DecodePastLen += seq;
            return result;
        }
        finally { exec.DecodeKVCache = null; exec.ConvStateCache = null; }
    }

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
    /// Download bytes by streaming the response in 1 MB chunks, reporting progress and yielding periodically so a
    /// long download does not freeze the UI. Streams on desktop AND in the browser.
    /// </summary>
    /// <remarks>
    /// ⚠️ Two stacked summaries used to sit here, and the second one claimed this "avoids OOM for large files in
    /// browser WASM". It did not. <c>HttpCompletionOption.ResponseHeadersRead</c> does nothing by itself in the
    /// browser, so the body was buffered whole into the managed heap - by DOUBLING - and a 329 MB model then
    /// OOM'd before reaching the chunk loop. The request now opts in to browser response streaming explicitly,
    /// which is what makes the claim true. See the comments in the body.
    /// <para>
    /// This still returns a <c>byte[]</c>, so ONE copy of the file does land on the managed heap. That is the
    /// method's contract and its callers'. Where a model can stay JS-side, prefer the streaming route
    /// (<c>HubModelStream</c> / <c>IJSReadStream</c>) over materialising it here.
    /// </para>
    /// </remarks>
    /// <param name="http">Client to download with.</param>
    /// <param name="url">Absolute URL of the file.</param>
    /// <param name="onProgress">Optional ("download", percent) callback.</param>
    /// <returns>The downloaded bytes.</returns>
    public static async Task<byte[]> DownloadBytesChunkedAsync(HttpClient http, string url,
        Action<string, int>? onProgress = null)
    {
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, url);

            // In the BROWSER, HttpCompletionOption.ResponseHeadersRead does NOTHING on its own: the fetch-based
            // handler buffers the whole body into the MANAGED heap unless the request explicitly opts in to
            // response streaming. Without this line ReadAsStreamAsync below lands in LoadIntoBufferAsync and a
            // 329 MB model is materialised whole - and grown by DOUBLING - before a single byte reaches the
            // chunk loop this method is named for. That is the OOM, and it is the standing "bulk bytes stay in
            // JS, never the .NET managed heap" rule broken on every browser model download.
            //
            // Set by the raw option key rather than Blazor's SetBrowserResponseStreamingEnabled extension, so
            // this library keeps taking no dependency on Microsoft.AspNetCore.Components.WebAssembly. The key is
            // exactly what that extension sets, verified against the shipped assembly. Off-browser handlers
            // ignore an unknown option, so this is a no-op on desktop.
            request.Options.Set(new HttpRequestOptionsKey<bool>("WebAssemblyEnableStreamingResponse"), true);

            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
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
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // Streaming download not supported — fall through to ReadAsByteArrayAsync.
            //
            // An OutOfMemoryException is deliberately NOT caught here. The fallback below buffers the ENTIRE
            // body a second time, so swallowing an OOM meant retrying the allocation that just failed - it
            // turned one 329 MB allocation into two and reported the failure from the retry, hiding where the
            // memory actually went. If we are out of memory, say so at the point it happened.
            _ = ex;
        }

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
        _convStateCache?.Dispose(); _convStateCache = null;   // session-owned conv-state buffers (LFM2 decode)
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
