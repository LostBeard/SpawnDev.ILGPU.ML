using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// Executes a compiled graph on GPU.
/// Manages tensor allocation, operator dispatch, and buffer lifecycle.
/// Automatically detects and manages KV cache with TurboQuant compression
/// for autoregressive transformer models (GPT-2, Whisper decoder, etc.).
/// </summary>
public class GraphExecutor : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly CompiledGraph _graph;
    private readonly BufferPool _pool;
    private readonly Dictionary<string, Tensor> _weights;
    private readonly Dictionary<string, float[]>? _constantValues;
    private readonly ElementWiseKernels _ew;

    // TurboQuant KV cache (auto-detected)
    private readonly KVCacheAnalyzer.KVCacheInfo? _kvCacheInfo;
    private QuantizedKVCache? _kvCache;
    private readonly Dictionary<string, int>? _presentKeyOutputToLayer;
    private readonly Dictionary<string, int>? _presentValueOutputToLayer;

    /// <summary>When true, logs each node execution to Console.</summary>
    public static bool VerboseLogging { get; set; }

    /// <summary>
    /// When non-null, captures first 10 values of each node's output for debugging.
    /// Performance cost: GPU sync + readback per node. Only use for diagnostics.
    /// </summary>
    public static Dictionary<string, float[]>? CapturedOutputs { get; set; }

    /// <summary>Whether TurboQuant KV cache compression is active for this model.</summary>
    public bool HasKVCache => _kvCache != null;

    /// <summary>Access to the quantized KV cache (null if model doesn't use KV cache).</summary>
    public QuantizedKVCache? KVCache => _kvCache;

    /// <summary>Quantized weight byte buffers on GPU (Q4_0, Q8_0, etc.)
    /// for fused dequantization during MatMul.</summary>
    private readonly Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? _quantizedWeights;

    public GraphExecutor(Accelerator accelerator, CompiledGraph graph,
        Dictionary<string, Tensor> weights, Dictionary<string, float[]>? constantValues = null,
        Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? quantizedWeights = null)
    {
        _accelerator = accelerator;
        _graph = graph;
        _pool = new BufferPool(accelerator);
        _weights = weights;
        _constantValues = constantValues;
        _quantizedWeights = quantizedWeights;
        _ew = new ElementWiseKernels(accelerator);

        // Auto-detect KV cache pattern
        var inputShapes = new Dictionary<string, int[]>();
        foreach (var node in graph.Nodes)
        {
            for (int i = 0; i < node.InputNames.Length; i++)
            {
                var name = node.InputNames[i];
                if (!string.IsNullOrEmpty(name) && weights.TryGetValue(name, out var wt))
                    inputShapes[name] = wt.Shape;
            }
        }
        _kvCacheInfo = KVCacheAnalyzer.Analyze(graph.InputNames, graph.OutputNames, inputShapes);

        if (_kvCacheInfo.ShouldQuantize)
        {
            try
            {
                _kvCache = new QuantizedKVCache(accelerator, _kvCacheInfo);

                // Build lookup maps for fast output interception
                _presentKeyOutputToLayer = new Dictionary<string, int>();
                _presentValueOutputToLayer = new Dictionary<string, int>();
                foreach (var layer in _kvCacheInfo.Layers)
                {
                    _presentKeyOutputToLayer[layer.PresentKeyOutput] = layer.LayerIndex;
                    _presentValueOutputToLayer[layer.PresentValueOutput] = layer.LayerIndex;
                }

                if (VerboseLogging)
                    Console.WriteLine($"[GraphExecutor] TurboQuant KV cache enabled: {_kvCacheInfo.NumLayers} layers, headDim={_kvCacheInfo.Layers[0].HeadDim}");
            }
            catch (Exception ex)
            {
                // KV cache allocation failed (e.g., insufficient GPU memory) — fall back to no cache
                if (VerboseLogging)
                    Console.WriteLine($"[GraphExecutor] TurboQuant KV cache disabled: {ex.Message}");
                _kvCache = null;
                _kvCacheInfo = null;
            }
        }
    }

    /// <summary>
    /// Run inference. Input tensors are provided by name.
    /// Returns output tensors by name.
    /// </summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
    {
        // Tensor registry: maps value names to tensors
        var tensors = new Dictionary<string, Tensor>();

        // Register inputs
        foreach (var (name, tensor) in inputs)
            tensors[name] = tensor;

        // Register weights/initializers
        foreach (var (name, tensor) in _weights)
            tensors[name] = tensor;

        // Reference counting: track how many more times each tensor is needed as input.
        // When a tensor's ref count reaches 0, return it to the pool to free GPU memory.
        var refCounts = new Dictionary<string, int>();
        var outputNameSet = new HashSet<string>(_graph.OutputNames);
        foreach (var node in _graph.Nodes)
        {
            foreach (var inputName in node.InputNames)
            {
                if (!string.IsNullOrEmpty(inputName))
                    refCounts[inputName] = refCounts.GetValueOrDefault(inputName, 0) + 1;
            }
        }
        // Mark graph outputs as "never release"
        foreach (var name in outputNameSet)
            refCounts[name] = int.MaxValue;
        // Mark weights as "never release"
        foreach (var name in _weights.Keys)
            refCounts[name] = int.MaxValue;
        // Mark external inputs as "never release"
        foreach (var name in inputs.Keys)
            refCounts[name] = int.MaxValue;

        // Runtime constant values: starts with initializer constants, grows as small
        // intermediate tensors (shape vectors, scalars) are captured back to CPU.
        // This enables operators like Slice, Reshape, Expand to resolve their parameters
        // from runtime-computed shape tensors (Shape→Gather→Concat chains in transformers).
        var runtimeConstants = _constantValues != null
            ? new Dictionary<string, float[]>(_constantValues)
            : new Dictionary<string, float[]>();

        // Remove stale compile-time constants for dynamically-computed node outputs.
        // These may have been computed with different input dimensions at compile time.
        // KEEP: Constant node outputs (fixed model values like indices, scales, axes).
        // CLEAR: Shape, Gather, Concat, Slice, etc. outputs that depend on input dims.
        var constantNodeOutputs = new HashSet<string>(
            _graph.Nodes.Where(n => n.OpType == "Constant").SelectMany(n => n.OutputNames));
        foreach (var node in _graph.Nodes)
        {
            if (node.OpType == "Constant") continue;
            foreach (var outName in node.OutputNames)
                if (!constantNodeOutputs.Contains(outName))
                    runtimeConstants.Remove(outName);
        }

        // Execute each node in topological order
        int nodeIdx = 0;
        foreach (var node in _graph.Nodes)
        {
            if (VerboseLogging)
            {
                var shapeInfo = string.Join(", ", node.OutputShapes.Select(s => $"[{string.Join(",", s)}]"));
                Console.WriteLine($"[GraphExecutor] Node {nodeIdx}/{_graph.Nodes.Length}: {node.OpType} [{string.Join(",", node.InputNames)}] -> [{string.Join(",", node.OutputNames)}] shapes={shapeInfo}");
                Console.Out.Flush();
            }
            // Constant nodes: output is already in weights (stored by extraction script)
            if (node.OpType == "Constant")
            {
                for (int i = 0; i < node.OutputNames.Length; i++)
                {
                    var outName = node.OutputNames[i];
                    if (tensors.ContainsKey(outName)) continue; // Already registered as weight
                    // Allocate empty tensor if not found (shouldn't happen with fixed extraction)
                    var shape = node.OutputShapes.Length > i ? node.OutputShapes[i] : new[] { 1 };
                    tensors[outName] = _pool.Rent(shape, outName);
                }
                continue;
            }

            // Gather input tensors
            var nodeInputs = new Tensor[node.InputNames.Length];
            for (int i = 0; i < node.InputNames.Length; i++)
            {
                var name = node.InputNames[i];
                if (string.IsNullOrEmpty(name)) continue; // Optional inputs
                if (!tensors.TryGetValue(name, out var tensor))
                    throw new InvalidOperationException($"Tensor '{name}' not found (needed by {node.OpType})");
                nodeInputs[i] = tensor;
            }

            // Use COMPILED shapes by default — they're correct for the compiled input dims.
            // Only override for operators with runtime-dependent shape tensors
            // (Reshape, Slice, Expand, Resize) resolved below.
            int[][] runtimeOutputShapes = node.OutputShapes;

            // Runtime Slice: resolve output shape from starts/ends/axes constants
            if (node.OpType == "Slice" && node.InputNames.Length >= 3)
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                float[]? starts = node.InputNames.Length > 1 ? (runtimeConstants.GetValueOrDefault(node.InputNames[1])) : null;
                float[]? ends = node.InputNames.Length > 2 ? (runtimeConstants.GetValueOrDefault(node.InputNames[2])) : null;
                float[]? axes = node.InputNames.Length > 3 && !string.IsNullOrEmpty(node.InputNames[3]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[3])) : null;
                float[]? steps = node.InputNames.Length > 4 && !string.IsNullOrEmpty(node.InputNames[4]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[4])) : null;
                if (starts != null && ends != null)
                {
                    var resolved = inShape.ToArray();
                    for (int si = 0; si < starts.Length; si++)
                    {
                        int ax = axes != null && si < axes.Length ? (int)axes[si] : si;
                        if (ax < 0) ax += resolved.Length;
                        if (ax >= 0 && ax < resolved.Length)
                        {
                            int s = (int)starts[si]; if (s < 0) s += resolved[ax];
                            int e = (int)ends[si]; if (e < 0) e += resolved[ax]; if (e > resolved[ax]) e = resolved[ax];
                            int st = steps != null && si < steps.Length ? (int)steps[si] : 1;
                            resolved[ax] = (e - s + st - 1) / st;
                        }
                    }
                    if (resolved.All(d => d > 0))
                        runtimeOutputShapes = new[] { resolved };
                }
            }

            // Runtime Reshape: DISABLED — compiled shapes are authoritative and Reshape
            // operator applies correct shape at execution time via Tensor.Shape setter.
            // Enabling this caused cascading buffer size mismatches in attention blocks.
            if (false && node.OpType == "Reshape" && node.InputNames.Length >= 2
                && runtimeConstants.TryGetValue(node.InputNames[1], out var reshapeTarget)
                && reshapeTarget.Length > 0)
            {
                int inputElems = nodeInputs[0]?.ElementCount ?? runtimeOutputShapes[0].Aggregate(1, (a, b) => a * b);
                var resolved = reshapeTarget.Select(v => (int)v).ToArray();
                // Handle 0 dims (copy from input) and -1 dims (infer)
                for (int j = 0; j < resolved.Length; j++)
                    if (resolved[j] == 0 && j < (nodeInputs[0]?.Shape.Length ?? 0)) resolved[j] = nodeInputs[0]!.Shape[j];
                int negIdx = Array.IndexOf(resolved, -1);
                if (negIdx >= 0)
                {
                    int known = 1;
                    for (int j = 0; j < resolved.Length; j++) if (j != negIdx && resolved[j] > 0) known *= resolved[j];
                    resolved[negIdx] = known > 0 ? inputElems / known : 1;
                }
                // Validate: all dims positive and total matches input elements
                bool valid = resolved.All(d => d > 0) &&
                    resolved.Aggregate(1L, (a, b) => a * b) == inputElems;
                if (valid)
                    runtimeOutputShapes = new[] { resolved };
                // else fall through to compiled shapes
            }

            // For Expand/Resize, also check runtime constants for dynamic targets
            if (node.OpType == "Expand" && node.InputNames.Length >= 2
                && runtimeConstants.TryGetValue(node.InputNames[1], out var expandTarget))
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                int outRank = Math.Max(inShape.Length, expandTarget.Length);
                var resolved = new int[outRank];
                for (int j = 0; j < outRank; j++)
                {
                    int inDim = j < outRank - inShape.Length ? 1 : inShape[j - (outRank - inShape.Length)];
                    int tgtDim = j < outRank - expandTarget.Length ? 1 : (int)expandTarget[j - (outRank - expandTarget.Length)];
                    resolved[j] = Math.Max(inDim, tgtDim);
                }
                runtimeOutputShapes = new[] { resolved };
            }
            if (node.OpType is "Resize" or "Upsample")
            {
                int sizesIdx = node.OpType == "Resize" ? 3 : -1;
                int scalesIdx = node.OpType == "Upsample" ? 1 : 2;
                if (sizesIdx >= 0 && node.InputNames.Length > sizesIdx
                    && !string.IsNullOrEmpty(node.InputNames[sizesIdx])
                    && runtimeConstants.TryGetValue(node.InputNames[sizesIdx], out var sizes)
                    && sizes.Length > 0)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    var resolved = new int[Math.Max(sizes.Length, inShape.Length)];
                    for (int j = 0; j < resolved.Length; j++)
                        resolved[j] = j < sizes.Length && (int)sizes[j] > 0 ? (int)sizes[j] : (j < inShape.Length ? inShape[j] : 1);
                    runtimeOutputShapes = new[] { resolved };
                }
                else if (node.InputNames.Length > scalesIdx
                    && !string.IsNullOrEmpty(node.InputNames[scalesIdx])
                    && runtimeConstants.TryGetValue(node.InputNames[scalesIdx], out var scales)
                    && scales.Length > 0)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    var resolved = new int[inShape.Length];
                    for (int j = 0; j < inShape.Length; j++)
                        resolved[j] = j < scales.Length ? (int)MathF.Floor(inShape[j] * scales[j]) : inShape[j];
                    runtimeOutputShapes = new[] { resolved };
                }
            }

            var nodeOutputs = new Tensor[node.OutputShapes.Length];
            for (int i = 0; i < node.OutputShapes.Length; i++)
            {
                var shape = i < runtimeOutputShapes.Length ? runtimeOutputShapes[i] : node.OutputShapes[i];
                // Replace zero/negative dimensions with 1 — zero-sized buffers are always
                // a compile-time inference error. The runtime operator will produce correct
                // data within the allocated buffer.
                for (int d = 0; d < shape.Length; d++)
                    if (shape[d] <= 0) shape[d] = 1;
                var name = i < node.OutputNames.Length ? node.OutputNames[i] : $"_anon_{i}";
                nodeOutputs[i] = _pool.Rent(shape, name);
            }

            // Execute operator
            var ctx = new OnnxOpContext
            {
                Inputs = nodeInputs,
                Outputs = nodeOutputs,
                Attributes = node.Attributes,
                Pool = _pool,
                InputNames = node.InputNames,
                ConstantValues = runtimeConstants,
                QuantizedWeights = _quantizedWeights,
            };
            var nodeSw = VerboseLogging ? System.Diagnostics.Stopwatch.StartNew() : null;
            node.Operator.Execute(ctx);
            if (VerboseLogging && nodeSw != null)
            {
                _accelerator.Synchronize();
                nodeSw.Stop();
                Console.WriteLine($"[GraphExecutor]   -> {node.OpType} took {nodeSw.Elapsed.TotalMilliseconds:F0}ms");
                Console.Out.Flush();
            }

            // Flush GPU command buffer periodically (64 nodes between syncs)
            if (nodeIdx > 0 && nodeIdx % 64 == 0)
                _accelerator.Synchronize();

            // Register outputs
            for (int i = 0; i < node.OutputNames.Length; i++)
                tensors[node.OutputNames[i]] = nodeOutputs[i];

            // Capture small intermediate outputs as runtime constants.
            // Shape tensors, scalars, and small 1D vectors (≤64 elements) are read back
            // to CPU so downstream operators (Slice, Reshape, Gather, Expand) can resolve
            // their parameters from runtime-computed values (e.g., Shape→Concat→Slice chains).
            for (int i = 0; i < nodeOutputs.Length; i++)
            {
                var outTensor = nodeOutputs[i];
                if (outTensor != null && outTensor.ElementCount > 0 && outTensor.ElementCount <= 2048)
                {
                    var outName = i < node.OutputNames.Length ? node.OutputNames[i] : null;
                    if (outName != null)
                    {
                        // Skip runtime constant capture in sync Run() — WebGPU/WebGL/Wasm
                        // don't support synchronous GPU→CPU copies. NLP models that need
                        // runtime constants (Shape→Slice chains) should use RunAsync().
                        // Desktop backends (CPU/CUDA/OpenCL) can use sync copies.
                        try
                        {
                            int elCount = outTensor.ElementCount;
                            using var tmpBuf = _accelerator.Allocate1D<float>(elCount);
                            tmpBuf.View.SubView(0, elCount).CopyFrom(outTensor.Data.SubView(0, elCount));
                            _accelerator.Synchronize();
                            runtimeConstants[outName] = tmpBuf.GetAsArray1D();
                        }
                        catch (NotSupportedException) { /* Browser/WASM backend — skip sync copy */ }
                    }
                }
            }

            // Release input tensors whose ref count reached 0
            foreach (var inputName in node.InputNames)
            {
                if (string.IsNullOrEmpty(inputName)) continue;
                if (refCounts.TryGetValue(inputName, out var rc) && rc < int.MaxValue)
                {
                    refCounts[inputName] = rc - 1;
                    if (rc - 1 <= 0 && tensors.TryGetValue(inputName, out var releaseTensor))
                    {
                        _pool.Return(releaseTensor);
                    }
                }
            }

            nodeIdx++;
        }

        // Flush all dispatches before readback
        _accelerator.Synchronize();

        // Collect requested outputs
        var results = new Dictionary<string, Tensor>();
        foreach (var name in _graph.OutputNames)
        {
            if (tensors.TryGetValue(name, out var tensor))
                results[name] = tensor;
        }
        return results;
    }

    /// <summary>
    /// Async version of Run. Required for browser backends (WebGPU/WebGL/Wasm)
    /// which deadlock on synchronous Synchronize(). Periodically awaits
    /// SynchronizeAsync() to flush GPU command buffers.
    /// </summary>
    public async Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
    {
        var tensors = new Dictionary<string, Tensor>();
        foreach (var (name, tensor) in inputs) tensors[name] = tensor;
        foreach (var (name, tensor) in _weights) tensors[name] = tensor;

        // Reference counting for buffer recycling (same as Run)
        var refCounts = new Dictionary<string, int>();
        var outputNameSet = new HashSet<string>(_graph.OutputNames);
        foreach (var node in _graph.Nodes)
            foreach (var inputName in node.InputNames)
                if (!string.IsNullOrEmpty(inputName))
                    refCounts[inputName] = refCounts.GetValueOrDefault(inputName, 0) + 1;
        foreach (var name in outputNameSet) refCounts[name] = int.MaxValue;
        foreach (var name in _weights.Keys) refCounts[name] = int.MaxValue;
        foreach (var name in inputs.Keys) refCounts[name] = int.MaxValue;

        // Runtime constant capture (same as Run — see comments there)
        var runtimeConstants = _constantValues != null
            ? new Dictionary<string, float[]>(_constantValues)
            : new Dictionary<string, float[]>();

        // Clear stale compile-time constants for non-Constant node outputs (same as Run).
        // PRESERVE Constant node outputs — they're fixed model values (indices, axes, etc.).
        var constantNodeOutputsAsync = new HashSet<string>(
            _graph.Nodes.Where(n => n.OpType == "Constant").SelectMany(n => n.OutputNames));
        foreach (var node in _graph.Nodes)
        {
            if (node.OpType == "Constant") continue;
            foreach (var outName in node.OutputNames)
                if (!constantNodeOutputsAsync.Contains(outName))
                    runtimeConstants.Remove(outName);
        }

        int nodeIdx = 0;
        var pendingReleases = new List<Tensor>();
        foreach (var node in _graph.Nodes)
        {
            if (VerboseLogging)
            {
                var shapeInfo = string.Join(", ", node.OutputShapes.Select(s => $"[{string.Join(",", s)}]"));
                Console.WriteLine($"[GraphExecutor] Node {nodeIdx}/{_graph.Nodes.Length}: {node.OpType} [{string.Join(",", node.InputNames)}] -> [{string.Join(",", node.OutputNames)}] shapes={shapeInfo}");
                Console.Out.Flush();
            }

            if (node.OpType == "Constant")
            {
                for (int i = 0; i < node.OutputNames.Length; i++)
                {
                    var outName = node.OutputNames[i];
                    if (tensors.ContainsKey(outName)) continue;
                    var shape = node.OutputShapes.Length > i ? node.OutputShapes[i] : new[] { 1 };
                    tensors[outName] = _pool.Rent(shape, outName);
                }
                continue;
            }

            var nodeInputs = new Tensor[node.InputNames.Length];
            for (int i = 0; i < node.InputNames.Length; i++)
            {
                var name = node.InputNames[i];
                if (string.IsNullOrEmpty(name)) continue;
                if (!tensors.TryGetValue(name, out var tensor))
                    throw new InvalidOperationException($"Tensor '{name}' not found (needed by {node.OpType})");
                nodeInputs[i] = tensor;
            }

            // Runtime shape cascade (same as sync Run — see comments there)
            var actualInputShapes = nodeInputs
                .Select(t => t?.Shape ?? Array.Empty<int>())
                .ToArray();

            // Use COMPILED shapes by default (same as sync Run path).
            // Full runtime re-inference caused cascading shape mismatches in attention blocks.
            int[][] runtimeOutputShapes = node.OutputShapes;

            // Runtime Slice (same as sync Run)
            if (node.OpType == "Slice" && node.InputNames.Length >= 3)
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                float[]? starts = node.InputNames.Length > 1 ? (runtimeConstants.GetValueOrDefault(node.InputNames[1])) : null;
                float[]? ends = node.InputNames.Length > 2 ? (runtimeConstants.GetValueOrDefault(node.InputNames[2])) : null;
                float[]? axes = node.InputNames.Length > 3 && !string.IsNullOrEmpty(node.InputNames[3]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[3])) : null;
                float[]? steps = node.InputNames.Length > 4 && !string.IsNullOrEmpty(node.InputNames[4]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[4])) : null;
                if (starts != null && ends != null)
                {
                    var resolved = inShape.ToArray();
                    for (int si = 0; si < starts.Length; si++)
                    {
                        int ax = axes != null && si < axes.Length ? (int)axes[si] : si;
                        if (ax < 0) ax += resolved.Length;
                        if (ax >= 0 && ax < resolved.Length)
                        {
                            int s = (int)starts[si]; if (s < 0) s += resolved[ax];
                            int e = (int)ends[si]; if (e < 0) e += resolved[ax]; if (e > resolved[ax]) e = resolved[ax];
                            int st = steps != null && si < steps.Length ? (int)steps[si] : 1;
                            resolved[ax] = (e - s + st - 1) / st;
                        }
                    }
                    if (resolved.All(d => d > 0))
                        runtimeOutputShapes = new[] { resolved };
                }
            }

            // Runtime Reshape (same as sync Run)
            if (node.OpType == "Reshape" && node.InputNames.Length >= 2
                && runtimeConstants.TryGetValue(node.InputNames[1], out var reshapeTargetAsync)
                && reshapeTargetAsync.Length > 0)
            {
                int inputElems = nodeInputs[0]?.ElementCount ?? runtimeOutputShapes[0].Aggregate(1, (a, b) => a * b);
                var resolved = reshapeTargetAsync.Select(v => (int)v).ToArray();
                for (int j = 0; j < resolved.Length; j++)
                    if (resolved[j] == 0 && j < (nodeInputs[0]?.Shape.Length ?? 0)) resolved[j] = nodeInputs[0]!.Shape[j];
                int negIdx = Array.IndexOf(resolved, -1);
                if (negIdx >= 0)
                {
                    int known = 1;
                    for (int j = 0; j < resolved.Length; j++) if (j != negIdx && resolved[j] > 0) known *= resolved[j];
                    resolved[negIdx] = known > 0 ? inputElems / known : 1;
                }
                bool valid = resolved.All(d => d > 0) &&
                    resolved.Aggregate(1L, (a, b) => a * b) == inputElems;
                if (valid)
                    runtimeOutputShapes = new[] { resolved };
                else if (nodeInputs[0] != null)
                {
                    // Reshape target doesn't match input elements — use input shape as
                    // safe fallback. Prevents both undersized (crash) and oversized
                    // (garbage data from uninitialized memory) buffer allocation.
                    long compiledElems = runtimeOutputShapes[0].Aggregate(1L, (a, b) => a * Math.Max(b, 1));
                    if (compiledElems != inputElems)
                        runtimeOutputShapes = new[] { nodeInputs[0].Shape };
                }
            }

            if (node.OpType == "Expand" && node.InputNames.Length >= 2
                && runtimeConstants.TryGetValue(node.InputNames[1], out var expandTarget))
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                int outRank = Math.Max(inShape.Length, expandTarget.Length);
                var resolved = new int[outRank];
                for (int j = 0; j < outRank; j++)
                {
                    int inDim = j < outRank - inShape.Length ? 1 : inShape[j - (outRank - inShape.Length)];
                    int tgtDim = j < outRank - expandTarget.Length ? 1 : (int)expandTarget[j - (outRank - expandTarget.Length)];
                    resolved[j] = Math.Max(inDim, tgtDim);
                }
                runtimeOutputShapes = new[] { resolved };
            }
            if (node.OpType is "Resize" or "Upsample")
            {
                int sizesIdx = node.OpType == "Resize" ? 3 : -1;
                int scalesIdx = node.OpType == "Upsample" ? 1 : 2;
                if (sizesIdx >= 0 && node.InputNames.Length > sizesIdx
                    && !string.IsNullOrEmpty(node.InputNames[sizesIdx])
                    && runtimeConstants.TryGetValue(node.InputNames[sizesIdx], out var sizes)
                    && sizes.Length > 0)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    var resolved = new int[Math.Max(sizes.Length, inShape.Length)];
                    for (int j = 0; j < resolved.Length; j++)
                        resolved[j] = j < sizes.Length && (int)sizes[j] > 0 ? (int)sizes[j] : (j < inShape.Length ? inShape[j] : 1);
                    runtimeOutputShapes = new[] { resolved };
                }
                else if (node.InputNames.Length > scalesIdx
                    && !string.IsNullOrEmpty(node.InputNames[scalesIdx])
                    && runtimeConstants.TryGetValue(node.InputNames[scalesIdx], out var scales)
                    && scales.Length > 0)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    var resolved = new int[inShape.Length];
                    for (int j = 0; j < inShape.Length; j++)
                        resolved[j] = j < scales.Length ? (int)MathF.Floor(inShape[j] * scales[j]) : inShape[j];
                    runtimeOutputShapes = new[] { resolved };
                }
            }

            var nodeOutputs = new Tensor[node.OutputShapes.Length];
            for (int i = 0; i < node.OutputShapes.Length; i++)
            {
                var shape = i < runtimeOutputShapes.Length ? runtimeOutputShapes[i] : node.OutputShapes[i];
                // Replace zero/negative dimensions with 1 — zero-sized buffers are always
                // a compile-time inference error. The runtime operator will produce correct
                // data within the allocated buffer.
                for (int d = 0; d < shape.Length; d++)
                    if (shape[d] <= 0) shape[d] = 1;
                var name = i < node.OutputNames.Length ? node.OutputNames[i] : $"_anon_{i}";
                nodeOutputs[i] = _pool.Rent(shape, name);
            }

            var ctx = new OnnxOpContext
            {
                Inputs = nodeInputs,
                Outputs = nodeOutputs,
                Attributes = node.Attributes,
                Pool = _pool,
                InputNames = node.InputNames,
                ConstantValues = runtimeConstants,
                QuantizedWeights = _quantizedWeights,
            };
            try
            {
                node.Operator.Execute(ctx);
            }
            catch (Exception ex)
            {
                var inputInfo = string.Join(", ", nodeInputs.Where(t => t != null).Select(t => $"[{string.Join(",", t.Shape)}]({t.ElementCount})"));
                var outputInfo = string.Join(", ", node.OutputShapes.Select(s => $"[{string.Join(",", s)}]"));
                var msg = $"Node {nodeIdx}/{_graph.Nodes.Length} '{node.OpType}' failed: {ex.Message} | " +
                    $"Inputs: [{string.Join(",", node.InputNames)}] shapes=({inputInfo}), " +
                    $"Outputs: [{string.Join(",", node.OutputNames)}] shapes=({outputInfo})";
                Console.WriteLine($"[GraphExecutor] ERROR: {msg}");
                throw new InvalidOperationException(msg, ex);
            }

            for (int i = 0; i < node.OutputNames.Length; i++)
                tensors[node.OutputNames[i]] = nodeOutputs[i];

            // Capture small intermediate outputs as runtime constants.
            // Only sync+readback for truly small shape tensors (≤64 elements) that downstream
            // operators need for parameter resolution (Slice starts/ends, Reshape dims, Expand shapes).
            // Was ≤2048 with double-sync per node — killed GPT-2 perf with hundreds of unnecessary syncs.
            for (int oi = 0; oi < nodeOutputs.Length; oi++)
            {
                var outTensor = nodeOutputs[oi];
                if (outTensor != null && outTensor.ElementCount > 0 && outTensor.ElementCount <= 64)
                {
                    var outName = oi < node.OutputNames.Length ? node.OutputNames[oi] : null;
                    if (outName != null)
                    {
                        try
                        {
                            int elCount = outTensor.ElementCount;
                            // WORKAROUND: GPU→GPU via Scale kernel + temp buffer because
                            // CopyToHostAsync<T>(offset, count) doesn't exist yet on ArrayView.
                            // Data is adding the overload to SpawnDev.ILGPU — when available,
                            // replace with: runtimeConstants[outName] = await outTensor.Data.CopyToHostAsync<float>(0, elCount);
                            using var tmpBuf = _accelerator.Allocate1D<float>(elCount);
                            _ew.Scale(outTensor.Data.SubView(0, elCount), tmpBuf.View, elCount, 1f);
                            await _accelerator.SynchronizeAsync();
                            runtimeConstants[outName] = await tmpBuf.CopyToHostAsync<float>(0, elCount);
                        }
                        catch (NotSupportedException) { /* Backend doesn't support async readback */ }
                    }
                }
            }

            // Capture intermediate values for debugging (when enabled)
            if (CapturedOutputs != null && nodeOutputs.Length > 0 && nodeOutputs[0] != null)
            {
                var captureOutput = nodeOutputs[0];
                // Capture enough values to get a meaningful absMax (at least one full
                // channel for Conv outputs). 1024 covers most shape tensors and small features.
                int captureCount = Math.Min(1024, captureOutput.ElementCount);
                if (captureCount > 0)
                {
                    try
                    {
                        await _accelerator.SynchronizeAsync();
                        using var capBuf = _accelerator.Allocate1D<float>(captureCount);
                        new ElementWiseKernels(_accelerator).Scale(
                            captureOutput.Data.SubView(0, captureCount), capBuf.View, captureCount, 1f);
                        await _accelerator.SynchronizeAsync();
                        var vals = await capBuf.CopyToHostAsync<float>(0, captureCount);
                        var key = $"{nodeIdx:D3}_{node.OpType}_{node.OutputNames[0]}";
                        CapturedOutputs[key] = vals;
                    }
                    catch { /* Don't crash on capture failure */ }
                }
            }

            // Defer buffer release to sync points to prevent reuse while GPU is in-flight
            foreach (var inputName in node.InputNames)
            {
                if (string.IsNullOrEmpty(inputName)) continue;
                if (refCounts.TryGetValue(inputName, out var rc) && rc < int.MaxValue)
                {
                    refCounts[inputName] = rc - 1;
                    if (rc - 1 <= 0 && tensors.TryGetValue(inputName, out var releaseTensor))
                        pendingReleases.Add(releaseTensor);
                }
            }

            nodeIdx++;

            // Flush GPU command buffer periodically to prevent massive single submissions.
            // 64 nodes between syncs balances latency vs throughput (was 16, too many syncs for GPT-2's 2620 nodes).
            if (nodeIdx % 64 == 0)
            {
                await _accelerator.SynchronizeAsync();
                // Now safe to return deferred buffers — GPU has finished reading them
                foreach (var t in pendingReleases)
                    _pool.Return(t);
                pendingReleases.Clear();
            }
        }

        // Final yield + sync
        await Task.Yield();
        await _accelerator.SynchronizeAsync();
        // Release any remaining deferred buffers
        foreach (var t in pendingReleases)
            _pool.Return(t);
        pendingReleases.Clear();

        var results = new Dictionary<string, Tensor>();
        foreach (var name in _graph.OutputNames)
        {
            if (tensors.TryGetValue(name, out var tensor))
                results[name] = tensor;
        }

        // TurboQuant KV cache: intercept present.N.key/value outputs and quantize
        if (_kvCache != null && _presentKeyOutputToLayer != null && _presentValueOutputToLayer != null)
        {
            foreach (var layer in _kvCacheInfo!.Layers)
            {
                if (results.TryGetValue(layer.PresentKeyOutput, out var presentKey) &&
                    results.TryGetValue(layer.PresentValueOutput, out var presentValue))
                {
                    // Extract the LAST token's K/V from the present output
                    // present.N.key shape: [batch, heads, seqLen, headDim]
                    int vecDim = _kvCache.NumLayers > 0 ? presentKey.Shape[^1] * presentKey.Shape[^3] : 0;
                    if (vecDim <= 0) continue;
                    int seqLen = presentKey.Shape.Length >= 3 ? presentKey.Shape[^2] : 1;
                    int lastTokenOffset = (seqLen - 1) * vecDim;

                    if (lastTokenOffset >= 0 && lastTokenOffset + vecDim <= presentKey.ElementCount)
                    {
                        _kvCache.Append(layer.LayerIndex,
                            presentKey.Data.SubView(lastTokenOffset, vecDim),
                            presentValue.Data.SubView(lastTokenOffset, vecDim));
                    }
                }
            }
            _kvCache.AdvanceToken();
        }

        return results;
    }

    /// <summary>
    /// Inject dequantized KV cache tensors into the input dictionary.
    /// Call this before RunAsync() for autoregressive generation steps 2+.
    /// If the model has no KV cache or the cache is empty, this is a no-op.
    /// </summary>
    public void InjectKVCacheInputs(Dictionary<string, Tensor> inputs)
    {
        if (_kvCache == null || !_kvCache.HasCache || _kvCacheInfo == null) return;

        foreach (var layer in _kvCacheInfo.Layers)
        {
            // Shape: [batch=1, heads, seqLen, headDim]
            var shape = layer.Shape != null ? (int[])layer.Shape.Clone() : new[] { 1, 12, _kvCache.CurrentSeqLen, 64 };
            if (shape.Length >= 3) shape[^2] = _kvCache.CurrentSeqLen; // Update seq dim

            inputs[layer.PastKeyInput] = _kvCache.GetDequantizedK(layer.LayerIndex, shape);
            inputs[layer.PastValueInput] = _kvCache.GetDequantizedV(layer.LayerIndex, shape);
        }

        // Set use_cache_branch if the model has it
        if (_kvCacheInfo.UseCacheBranchInput != null)
        {
            using var flagBuf = _accelerator.Allocate1D(new float[] { 1f });
            inputs[_kvCacheInfo.UseCacheBranchInput] = new Tensor(flagBuf.View, new[] { 1 });
        }
    }

    /// <summary>
    /// Reset the KV cache (e.g., when starting a new generation sequence).
    /// </summary>
    public void ResetKVCache()
    {
        if (_kvCache != null)
        {
            _kvCache.Dispose();
            _kvCache = _kvCacheInfo != null ? new QuantizedKVCache(_accelerator, _kvCacheInfo) : null;
        }
    }

    public void Dispose()
    {
        _pool.Dispose();
        _kvCache?.Dispose();
    }

    /// <summary>
    /// Convert JsonElement attributes to CLR types (long[], string, long) for InferOutputShapes.
    /// Attributes are stored as JsonElement from graph compilation but operators expect typed values.
    /// </summary>
    private static Dictionary<string, object> ConvertAttributes(Dictionary<string, object>? attrs)
    {
        if (attrs == null) return new Dictionary<string, object>();
        var result = new Dictionary<string, object>();
        foreach (var (key, val) in attrs)
        {
            if (val is System.Text.Json.JsonElement je)
            {
                try
                {
                    if (je.ValueKind == System.Text.Json.JsonValueKind.Array)
                        result[key] = je.EnumerateArray().Select(e => e.GetInt64()).ToArray();
                    else if (je.ValueKind == System.Text.Json.JsonValueKind.Number)
                        result[key] = je.GetInt64();
                    else if (je.ValueKind == System.Text.Json.JsonValueKind.String)
                        result[key] = je.GetString() ?? "";
                    else
                        result[key] = val;
                }
                catch { result[key] = val; }
            }
            else
                result[key] = val; // Already CLR type
        }
        return result;
    }
}
