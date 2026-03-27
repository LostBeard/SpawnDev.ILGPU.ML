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

    public GraphExecutor(Accelerator accelerator, CompiledGraph graph,
        Dictionary<string, Tensor> weights, Dictionary<string, float[]>? constantValues = null)
    {
        _accelerator = accelerator;
        _graph = graph;
        _pool = new BufferPool(accelerator);
        _weights = weights;
        _constantValues = constantValues;
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

            // Allocate output tensors
            var nodeOutputs = new Tensor[node.OutputShapes.Length];
            for (int i = 0; i < node.OutputShapes.Length; i++)
            {
                var shape = node.OutputShapes[i];

                // Runtime shape resolution for Expand with dynamic targets
                if (node.OpType == "Expand" && node.InputNames.Length >= 2
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var expandTarget))
                {
                    var inShape = nodeInputs[0]?.Shape ?? shape;
                    int outRank = Math.Max(inShape.Length, expandTarget.Length);
                    var resolved = new int[outRank];
                    for (int j = 0; j < outRank; j++)
                    {
                        int inDim = j < outRank - inShape.Length ? 1 : inShape[j - (outRank - inShape.Length)];
                        int tgtDim = j < outRank - expandTarget.Length ? 1 : (int)expandTarget[j - (outRank - expandTarget.Length)];
                        resolved[j] = Math.Max(inDim, tgtDim);
                    }
                    shape = resolved;
                }

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
                    if (outName != null && !runtimeConstants.ContainsKey(outName))
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

            var nodeOutputs = new Tensor[node.OutputShapes.Length];
            for (int i = 0; i < node.OutputShapes.Length; i++)
            {
                var shape = node.OutputShapes[i];

                // Runtime shape resolution for Expand/Reshape with dynamic targets.
                // The compiled shape may be wrong if the target shape tensor wasn't
                // resolvable at compile time (e.g., Expand with Where-computed shapes).
                // Use runtime constants to compute the real output shape.
                if (node.OpType == "Expand" && node.InputNames.Length >= 2
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var expandTarget))
                {
                    var inShape = nodeInputs[0]?.Shape ?? shape;
                    int outRank = Math.Max(inShape.Length, expandTarget.Length);
                    var resolved = new int[outRank];
                    for (int j = 0; j < outRank; j++)
                    {
                        int inDim = j < outRank - inShape.Length ? 1 : inShape[j - (outRank - inShape.Length)];
                        int tgtDim = j < outRank - expandTarget.Length ? 1 : (int)expandTarget[j - (outRank - expandTarget.Length)];
                        resolved[j] = Math.Max(inDim, tgtDim);
                    }
                    shape = resolved;
                }

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
                    if (outName != null && !runtimeConstants.ContainsKey(outName))
                    {
                        await _accelerator.SynchronizeAsync();
                        int elCount = outTensor.ElementCount;
                        using var tmpBuf = _accelerator.Allocate1D<float>(elCount);
                        tmpBuf.View.SubView(0, elCount).CopyFrom(outTensor.Data.SubView(0, elCount));
                        runtimeConstants[outName] = await tmpBuf.CopyToHostAsync<float>(0, elCount);
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
}
