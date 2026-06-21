using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>Storage precision for graph ACTIVATION intermediates (weights are handled separately).
/// F32 = the default all-fp32 path. F16 = store eligible large float feature-maps as fp16 (half the held
/// GPU bytes); operators still compute fp32 (convert at boundaries). Extensible to bf16/fp8 as Geordi's
/// low-precision types land — add an enum value + a switch case in RunAsync's convert/rent.</summary>
public enum ActivationPrecision { F32, F16 }

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

    // Per-RUN setup templates, precomputed ONCE from the fixed graph (_graph/_weights/_constantValues are all
    // readonly; recompile builds a NEW executor). RunAsync clones these instead of re-walking all ~1400 nodes
    // (refcount build) + a LINQ Constant scan + a node walk to strip stale constants on EVERY token — the
    // super-linear per-node CPU residual that forced multimodal prefill token-by-token. Built lazily.
    private Dictionary<string, int>? _baseRefCounts;       // node-input refcounts; graph OUTPUTS + WEIGHTS pinned to int.MaxValue
    private Dictionary<string, float[]>? _cleanConstants;   // _constantValues with non-Constant-node outputs already stripped

    private void EnsureRunTemplates()
    {
        if (_baseRefCounts != null) return;
        var rc = new Dictionary<string, int>();
        foreach (var node in _graph.Nodes)
            foreach (var inputName in node.InputNames)
                if (!string.IsNullOrEmpty(inputName))
                    rc[inputName] = rc.GetValueOrDefault(inputName, 0) + 1;
        foreach (var name in _graph.OutputNames) rc[name] = int.MaxValue;
        foreach (var name in _weights.Keys) rc[name] = int.MaxValue;

        var clean = _constantValues != null
            ? new Dictionary<string, float[]>(_constantValues)
            : new Dictionary<string, float[]>();
        var constantNodeOutputs = new HashSet<string>(
            _graph.Nodes.Where(n => n.OpType == "Constant").SelectMany(n => n.OutputNames));
        foreach (var node in _graph.Nodes)
        {
            if (node.OpType == "Constant") continue;
            foreach (var outName in node.OutputNames)
                if (!constantNodeOutputs.Contains(outName))
                    clean.Remove(outName);
        }

        _cleanConstants = clean;
        _baseRefCounts = rc; // set LAST so the null-check above is the completion signal
    }
    // Precision-aware (F16 pass-through) op kernels — owned here (not via the optional _registry, which is null
    // in registry-less executor uses like the controlled mixed-precision test). Stateless kernel caches.
    private readonly PrecisionAwareKernels _precisionAware;
    // Owned (not via the optional _registry, which is null for ONNX-model executors) — used for the in-place
    // InstanceNorm executor path.
    private readonly NormalizationKernels _normalization;
    private readonly Operators.OperatorRegistry? _registry;

    // Mixed-precision activations (opt-in). When != F32, RunAsync stores eligible large float feature-map
    // intermediates in low precision (half the held GPU bytes) and converts at fp32 op boundaries — operators
    // stay fp32 (zero per-op risk). F32 (default) is byte-identical to the all-fp32 path (the whole mechanism
    // is guarded off). dtype-parameterized: a new low-p type = one switch case + its convert + a RentX pool.
    // Plan: Plans/fp16-bf16-mixed-precision-activations-2026-06-16.md.
    /// <summary>Activation storage precision for graph intermediates (default F32 = unchanged).</summary>
    public ActivationPrecision ActivationDtype { get; set; } = ActivationPrecision.F32;
    // Optional allowlist of OpTypes permitted to take the F16 precision-aware pass-through. Null = all eligible
    // ops pass through (the default). Env `PA_OPS=Conv,Relu` restricts it (bisection / escape hatch); the
    // disallowed ops fall back to the fp32 convert-around-node path. Read once from env at construction.
    private static readonly HashSet<string>? _paOpsAllowlist =
        Environment.GetEnvironmentVariable("PA_OPS") is { Length: > 0 } s
            ? new HashSet<string>(s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            : null;
    private Kernels.PrecisionConvertKernels? _convert;

    // TurboQuant KV cache (auto-detected)
    private readonly KVCacheAnalyzer.KVCacheInfo? _kvCacheInfo;
    private QuantizedKVCache? _kvCache;
    private MemoryBuffer1D<float, Stride1D.Dense>? _kvCacheFlagBuf;
    private readonly Dictionary<string, int>? _presentKeyOutputToLayer;
    private readonly Dictionary<string, int>? _presentValueOutputToLayer;

    /// <summary>GGUF incremental-decode KV cache (full-precision, GGUF-native — distinct from the
    /// TurboQuant <see cref="_kvCache"/>). Session-owned and shared across the prefill executor (seq=N)
    /// and the decode executor (seq=1); set by the session before a decode-mode run. When non-null, the
    /// async node loop intercepts each FusedAttention node (tagged with a "layer" attr): it writes the
    /// step's K/V into the cache at <see cref="DecodePastLen"/> and feeds the kernel the full history.
    /// Null = the normal full-recompute forward (untouched).</summary>
    public GGUFDecodeKVCache? DecodeKVCache { get; set; }

    /// <summary>Number of tokens already cached BEFORE this run's tokens (the session advances it between
    /// runs: 0 for prefill, then prompt-length, then +1 per decoded token). All layers in a run share it.</summary>
    public int DecodePastLen { get; set; }

    /// <summary>When set, the mid-graph runtime-constant readbacks (the ≤64-elem shape/scalar tensors
    /// captured via SynchronizeAsync+CopyToHostAsync — measured at ~7.8s of a 1554-node DistilGPT-2
    /// forward) are cached and reused across calls. This executor is shape-specialized (one input
    /// shape for its whole life), so SHAPE-derived readbacks are constant across calls — but some
    /// captured ≤64-elem tensors are DATA-derived (e.g. input_ids itself, when seq≤64), which change
    /// per call. The cache AUTO-DETECTS which: it probes the first two runs (full readback both),
    /// caches ONLY the readbacks whose values were IDENTICAL across those two different-data runs,
    /// and re-reads the rest every call. Correct by construction. Opt-in via
    /// InferenceSession.CacheShapeReadbacks; default off so general inference is unchanged.</summary>
    public bool CacheShapeReadbacks { get; set; }
    private Dictionary<string, float[]>? _readbackProbe;     // first probe run's values (for comparison)
    private Dictionary<string, float[]>? _readbackStable;    // proven-stable values, reused once finalized
    private bool _readbackStableFinalized;                   // true once _readbackStable is built
    private List<Tensor>? _priorRunOutputs;                  // last run's graph-output tensors, recycled next run

    /// <summary>Fixed-shape decode (CacheShapeReadbacks) buffer cache for Shape op outputs: output name →
    /// (retained GPU buffer, the input dims it holds). ShapeOperator re-uploads the constant input dims via
    /// CopyFromCPU on EVERY step (measured ~1537ms across 141 Shape nodes per GPT-2 decode step on WebGPU —
    /// per-op writeBuffer latency, not data size). In a shape-specialized executor those dims never change,
    /// so the buffer is uploaded once (first step) and reused thereafter. Reusing the BUFFER (not just the
    /// value) is required because at least one downstream consumer reads the Shape output as a GPU TENSOR,
    /// so skipping the upload outright corrupts output — see
    /// feedback-shape-outputs-consumed-as-gpu-tensor-not-just-value. Retained buffers are never returned to
    /// the pool (so no other node's Rent can clobber them) and are freed when the pool is disposed.</summary>
    private Dictionary<string, (Tensor Buf, int[] Dims)>? _shapeBufferCache;

    /// <summary>When true, logs each node execution to Console.</summary>
    public static bool VerboseLogging { get; set; }

    /// <summary>Number of nodes RunAsync executes between GPU command-buffer flushes
    /// (<c>SynchronizeAsync</c>). Each flush is an async round-trip; on WebGPU/Blazor that latency
    /// dominates large-graph forward time (a 1554-node decoder at interval 64 = ~24 round-trips).
    /// Higher = fewer round-trips (faster) but more unsubmitted commands + deferred buffer releases
    /// held longer (higher peak GPU memory). Tunable so the autoregressive decode loop can trade
    /// memory for latency.</summary>
    public static int SyncIntervalNodes = 64;

    /// <summary>Memory bound on the async deferred-release window: when the bytes of buffers awaiting
    /// release (dead by refcount, but held until a drain so a still-queued kernel can't read freed memory)
    /// exceed this, force the sync+return EARLY — independent of <see cref="SyncIntervalNodes"/>. Bounds the
    /// peak GPU working set to ~(true live set + this cap) instead of "sum of up to N nodes' intermediates",
    /// which is what blew a 512² VAE decode to ~10 GB (227 large feature maps released only every 64 nodes).
    /// A small-tensor graph (LLM token decode) never hits the cap, so it keeps the cheap N-node cadence.
    /// 512 MiB default = generous headroom vs the latency of an extra drain. Exact: changes only WHEN buffers
    /// are recycled, never the math.</summary>
    public static long MaxPendingReleaseBytes = 512L * 1024 * 1024;

    /// <summary>
    /// DIAGNOSTIC: when set, RunAsync's main loop breaks after executing
    /// `BreakAtNode` nodes (1-indexed). Used to bisect which operator triggers
    /// WebGPU buffer-used-while-destroyed in Pipeline_Diffusion_DDPM.
    /// </summary>
    public static int? BreakAtNode { get; set; }

    /// <summary>
    /// DIAGNOSTIC: captures the OpType + node index of every operator run in
    /// the most recent RunAsync invocation. Cleared at the start of each call.
    /// </summary>
    public static List<string> LastRunOpLog { get; } = new();

    /// <summary>
    /// When non-null, captures first 10 values of each node's output for debugging.
    /// Performance cost: GPU sync + readback per node. Only use for diagnostics.
    /// </summary>
    public static Dictionary<string, float[]>? CapturedOutputs { get; set; }

    /// <summary>
    /// DIAGNOSTIC sibling to <see cref="CapturedOutputs"/>: when CapturedOutputs is
    /// non-null, this dictionary captures per-node metadata keyed identically -
    /// OpType, input names + shapes, output names + shapes, and any stringified
    /// attributes. Lets the test harness identify the EXACT op variant (axis,
    /// dtype combo, broadcast pattern) when the captured float[] sample shows
    /// the bug. Use the same key as CapturedOutputs to look up shape context.
    /// </summary>
    public static Dictionary<string, string>? CapturedNodeInfo { get; set; }

    /// <summary>DIAGNOSTIC: max elements captured per node into <see cref="CapturedOutputs"/>.
    /// Default 1024 (covers shape tensors + small features). Raise to capture full
    /// feature channels (e.g. a 48x48x17 heatmap = 39168) when hunting a spatial bug.</summary>
    public static int CaptureMaxElements = 1024;

    /// <summary>When non-null (and <see cref="CapturedOutputs"/> is set), restrict capture to ONLY the listed
    /// node output names, and capture those in FULL (ignoring <see cref="CaptureMaxElements"/>). Used to pull a
    /// single intermediate tensor out of a graph cheaply — e.g. the tiled VAE decoder captures just the mid-block
    /// output (combined with <see cref="BreakAtNode"/> so the rest of the graph never runs).</summary>
    public static HashSet<string>? CaptureOutputNames { get; set; }

    /// <summary>
    /// DIAGNOSTIC: when non-null, captures per-node Execute() wall-clock time in
    /// milliseconds keyed by the same node key as <see cref="CapturedOutputs"/>.
    /// Combined with <see cref="PerOpSync"/> the timing reflects each op's full
    /// dispatch + sync cost, which surfaces slow kernels (e.g. why is WebGPU
    /// SemanticSearch 5x slower than CUDA on the same model graph - is it a
    /// specific MatMul / LayerNorm taking the bulk of the time, or is the cost
    /// spread evenly?).
    ///
    /// Off by default; opt-in alongside CapturedOutputs / CapturedNodeInfo.
    /// </summary>
    public static Dictionary<string, double>? CapturedNodeTimingsMs { get; set; }

    /// <summary>
    /// DIAGNOSTIC: when true, calls await accelerator.SynchronizeAsync() after EVERY
    /// node's Execute. Without this, async-dispatch backends (Wasm worker pool, WebGPU
    /// command-encoder batches) only surface kernel traps at the next periodic sync
    /// (every 64 nodes by default), so an OOB or remainder-by-zero in node 17 reads as
    /// "node-64 sync caught Wasm Worker N error: memory access out of bounds" with no
    /// way to identify the actual failing op. Per-op sync makes the augmented Execute
    /// catch fire on the exact node that traps.
    ///
    /// Performance cost is significant - one full GPU flush per op, which serializes
    /// what would otherwise be a batched submission. Only enable for kernel-bisection
    /// debugging on small models.
    /// </summary>
    public static bool PerOpSync { get; set; }

    /// <summary>
    /// DIAGNOSTIC: incremented each time the Pad readback fallback fires at execute time.
    /// After session-init Pad pre-extraction (InferenceSession 2026-05-05), this should
    /// remain 0 for every well-formed ONNX model. Tests assert == 0 after a Run/RunAsync
    /// to verify pre-extraction covered every Pad node. Reset manually before a run.
    /// </summary>
    public static int PadReadbackFallbackFiredCount;

    /// <summary>
    /// DIAGNOSTIC: stringified info from the last fired Pad readback fallback (path + node + pads tensor name).
    /// Use with <see cref="PadReadbackFallbackFiredCount"/> to identify which Pad node missed pre-extraction.
    /// </summary>
    public static string? LastPadReadbackFallbackInfo;

    /// <summary>Data layout format (NCHW for ONNX, NHWC for TFLite).</summary>
    public DataFormat Format { get; set; } = DataFormat.NCHW;

    /// <summary>Whether TurboQuant KV cache compression is active for this model.</summary>
    public bool HasKVCache => _kvCache != null;

    /// <summary>Access to the quantized KV cache (null if model doesn't use KV cache).</summary>
    public QuantizedKVCache? KVCache => _kvCache;

    /// <summary>DIAGNOSTIC: number of GPU intermediate buffers this executor's pool has allocated.
    /// A freshly-recompiled executor starts at 0 and allocates one per distinct intermediate-buffer
    /// size on its first Run — lets the decode loop see whether per-step cost is fresh-pool churn.</summary>
    public int AllocatedBufferCount => _pool.AllocatedBufferCount;

    /// <summary>Quantized weight byte buffers on GPU (Q4_0, Q8_0, etc.)
    /// for fused dequantization during MatMul.</summary>
    private readonly Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? _quantizedWeights;

    /// <summary>The quantized weight byte-view map this executor was constructed with (null when the
    /// model has no quantized weights). A shape-recompiled executor MUST be constructed with the SAME
    /// map as the base executor — quantized weights are session-lifetime GPU buffers shared across all
    /// per-shape executors. InferenceSession.RecompileForShapes reads it from here; omitting it routes
    /// every quantized MatMul/Gather to the F32 path against a ShapeOnly tensor's empty view (a CUDA
    /// illegal memory access at the first quantized node — the gemma4 seq&gt;1 fault, 2026-06-12).</summary>
    internal Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? QuantizedWeights => _quantizedWeights;

    /// <summary>Names of tensors whose ONNX-declared dtype is integer
    /// (INT8/16/32/64, UINT8/16/32/64, BOOL). Built once at session-init by
    /// walking initializer dtypes + integer-producing op outputs + propagating
    /// through dtype-preserving / binary ops. See <see cref="BuildIntegerTensorNames"/>.</summary>
    private readonly HashSet<string> _integerTensorNames;

    /// <summary>Output tensor names whose ≤64-elem mid-graph readback is provably unnecessary and is
    /// SKIPPED (built once at construction by <see cref="BuildReadbackSkipSet"/>). These are
    /// feature/data tensors — e.g. LayerNorm ReduceMean/Add/Sqrt intermediates — that no downstream op
    /// consumes as a runtime-constant value, only as a GPU tensor. The eager ≤64 readback grabbed them
    /// anyway (they're small + token-dependent so the warm cache can't elide them), costing ~2.7s/step
    /// on WebGPU GPT-2 (53 readbacks) for values nothing reads. See <see cref="BuildReadbackSkipSet"/>
    /// for the safety rules (never skips anything a value-needing op consumes).</summary>
    private readonly HashSet<string> _readbackSkipNames;
    /// <summary>DIAGNOSTIC: count of names in <see cref="_readbackSkipNames"/> for the most recently
    /// constructed executor.</summary>
    public static int LastReadbackSkipCount;

    /// <summary>Number of integer-typed tensors identified by BuildIntegerTensorNames
    /// in the most recently constructed GraphExecutor. Diagnostic for verifying that
    /// dtype propagation is reaching Div / Mul / Mod chains it needs to.</summary>
    public static int LastIntegerTensorCount;
    /// <summary>Snapshot of the integer-tensor-name set built by the most recently
    /// constructed GraphExecutor. Diagnostic; cleared and overwritten per construction.</summary>
    public static List<string> LastIntegerTensorNames = new();
    /// <summary>Size of CompiledGraph.InitializerDataTypes at GraphExecutor construction
    /// time (or -1 if null). Diagnostic to verify dtype plumbing is reaching the executor.</summary>
    public static int LastInitializerDataTypesCount;
    /// <summary>Number of Div ops in the most recent RunAsync where all inputs were
    /// flagged integer and TruncateInPlace was applied. Useful for verifying that the
    /// MoveNet Cast(int)→Div(int,int) floordiv chain is actually being trunc'd.
    /// Reset to 0 at the start of every RunAsync.</summary>
    public static int LastRunIntegerDivCount;

    /// <summary>DIAGNOSTIC: number of mid-graph small-tensor (≤64-elem) runtime-constant readbacks
    /// (each a SynchronizeAsync + CopyToHostAsync GPU round-trip) in the most recent RunAsync. With
    /// const-folding (enableOptimization) these shape/scalar values are computed at compile time and
    /// this drops toward 0. Reset at the start of every RunAsync.</summary>
    public static int LastRunReadbackCount;
    /// <summary>DIAGNOSTIC: total wall-clock ms spent in those mid-graph readbacks. Reset per RunAsync.</summary>
    public static double LastRunReadbackMs;
    /// <summary>DIAGNOSTIC: "OpType:outName" of every mid-graph readback that actually fired in the most
    /// recent RunAsync (i.e. NOT skipped by the warm shape-readback cache). In warm fixed-shape decode this
    /// is the RESIDUAL set — the per-step token-dependent readbacks the cache can't elide. Reset per RunAsync.
    /// Used to decide whether those readbacks are even needed downstream (vs pure-data waste).</summary>
    public static List<string> LastRunReadbackNames = new();

    /// <summary>DIAGNOSTIC: total wall-clock ms of the entire most-recent RunAsync (executor-internal,
    /// excludes the session's recompile and the caller's logits readback). Combined with
    /// <see cref="LastRunReadbackMs"/> and <see cref="LastRunSyncDrainMs"/> this partitions per-token
    /// decode time into readback round-trips / periodic GPU sync-drains / (dispatch+CPU+alloc) residual.
    /// Reset per RunAsync.</summary>
    public static double LastRunTotalMs;
    /// <summary>DIAGNOSTIC: count of the periodic <see cref="SyncIntervalNodes"/> command-buffer drains
    /// plus the one final drain in the most recent RunAsync (each an <c>await SynchronizeAsync()</c> that
    /// forces GPU completion). These are NOT the shape readbacks — they carry no host copy. Reset per RunAsync.</summary>
    public static int LastRunSyncDrainCount;
    /// <summary>DIAGNOSTIC: total wall-clock ms spent in those periodic + final GPU sync-drains. Reset per RunAsync.</summary>
    public static double LastRunSyncDrainMs;

    public GraphExecutor(Accelerator accelerator, CompiledGraph graph,
        Dictionary<string, Tensor> weights, Dictionary<string, float[]>? constantValues = null,
        Dictionary<string, ArrayView1D<byte, Stride1D.Dense>>? quantizedWeights = null,
        Operators.OperatorRegistry? registry = null)
    {
        _accelerator = accelerator;
        _graph = graph;
        _pool = new BufferPool(accelerator);
        _weights = weights;
        _constantValues = constantValues;
        _quantizedWeights = quantizedWeights;
        _registry = registry;
        _ew = new ElementWiseKernels(accelerator);
        _precisionAware = new PrecisionAwareKernels(accelerator);
        _normalization = new NormalizationKernels(accelerator);
        LastInitializerDataTypesCount = graph.InitializerDataTypes?.Count ?? -1;
        _integerTensorNames = BuildIntegerTensorNames(graph);
        LastIntegerTensorCount = _integerTensorNames.Count;
        LastIntegerTensorNames = _integerTensorNames.ToList();
        _readbackSkipNames = BuildReadbackSkipSet(graph);
        LastReadbackSkipCount = _readbackSkipNames.Count;

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
            // (Reshape, Slice, Expand, Resize) resolved below. Dynamic input shapes (e.g. a
            // growing decode sequence) are handled by InferenceSession recompiling the graph at
            // the actual shape, so the executor always runs a graph compiled for THESE dims.
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
            // Runtime Pad: opset >= 11 has pads as input[1] tensor, not as attribute.
            // PadOperator.InferOutputShapes can't read the tensor value at compile time
            // and returns inputs[0] unchanged — output buffer would be sized to INPUT,
            // but the kernel writes to OUTPUT (input + pads) → OOB. Resolve here.
            // 2026-05-04 Data: surfaced StyleMosaic Wasm hang via PerOpSync diag at
            // disp=176 'Kernel_PadImpl' with items=108M into 102.7M-float V0 buffer.
            if (node.OpType == "Pad")
            {
                // Try runtime constant first
                int[]? padsResolved = null;
                if (node.InputNames.Length >= 2
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var padsTensorRC)
                    && padsTensorRC.Length > 0)
                {
                    padsResolved = padsTensorRC.Select(v => (int)v).ToArray();
                }
                // Fallback to attribute (opset < 11)
                else if (node.Attributes.TryGetValue("pads", out var padsAttrObj) && padsAttrObj is long[] padsAttr)
                {
                    padsResolved = padsAttr.Select(v => (int)v).ToArray();
                }

                if (padsResolved != null)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    int rank = inShape.Length;
                    if (padsResolved.Length == 2 * rank)
                    {
                        var resolved = new int[rank];
                        for (int j = 0; j < rank; j++)
                            resolved[j] = inShape[j] + padsResolved[j] + padsResolved[rank + j];
                        if (resolved.All(d => d > 0))
                            runtimeOutputShapes = new[] { resolved };
                    }
                }
                else if (node.InputNames.Length >= 2 && nodeInputs.Length > 1 && nodeInputs[1] != null)
                {
                    // DIAGNOSTIC: arriving here means session-init Pad pre-extraction missed this node.
                    System.Threading.Interlocked.Increment(ref PadReadbackFallbackFiredCount);
                    LastPadReadbackFallbackInfo = $"sync output={(node.OutputNames.Length > 0 ? node.OutputNames[0] : "?")} pads={node.InputNames[1]}";
                    if (VerboseLogging) Console.WriteLine($"[GraphExecutor] PAD READBACK FALLBACK FIRED ({LastPadReadbackFallbackInfo}) - session-init pre-extraction missed this node");

                    // Sync path: try sync GPU readback for pads tensor. May NotSupported
                    // on browser backends — that's fine, the async RunAsync() path is what
                    // those backends use anyway.
                    var padsTensor = nodeInputs[1]!;
                    if (padsTensor.ElementCount > 0 && padsTensor.ElementCount <= 32)
                    {
                        try
                        {
                            using var tmp = _accelerator.Allocate1D<float>(padsTensor.ElementCount);
                            _ew.Scale(padsTensor.Data.SubView(0, padsTensor.ElementCount), tmp.View, padsTensor.ElementCount, 1f);
                            _accelerator.Synchronize();
                            var padsHost = tmp.GetAsArray1D();
                            runtimeConstants[node.InputNames[1]] = padsHost;
                            var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                            int rank = inShape.Length;
                            if (padsHost.Length == 2 * rank)
                            {
                                var resolved = new int[rank];
                                for (int j = 0; j < rank; j++)
                                    resolved[j] = inShape[j] + (int)padsHost[j] + (int)padsHost[rank + j];
                                if (resolved.All(d => d > 0))
                                    runtimeOutputShapes = new[] { resolved };
                            }
                        }
                        catch { /* fall through to compiled shapes */ }
                    }
                }
            }

            // Runtime Unsqueeze: re-infer the output shape from the ACTUAL runtime input
            // shape (insert size-1 dims at axes). At compile time, an upstream dynamic op
            // (e.g. Range, whose placeholder shape is [1]) poisons Unsqueeze's compiled output
            // shape, so the buffer is mis-sized and element count collapses. Unsqueeze only
            // inserts size-1 dims, so re-inferring from the real input shape is always safe and
            // preserves element count. (Needed so the position-id range survives Range→Unsqueeze
            // →Reshape into the wpe Gather.)
            if (node.OpType == "Unsqueeze" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                long[]? axesArr = null;
                if (node.Attributes.TryGetValue("axes", out var axObj) && axObj is long[] al) axesArr = al;
                else if (node.InputNames.Length >= 2 && !string.IsNullOrEmpty(node.InputNames[1])
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var axC) && axC.Length > 0)
                    axesArr = axC.Select(v => (long)v).ToArray();
                if (axesArr != null)
                {
                    var inShape = nodeInputs[0]!.Shape;
                    int outRank = inShape.Length + axesArr.Length;
                    var norm = new HashSet<int>();
                    foreach (var a in axesArr) { int x = (int)a; if (x < 0) x += outRank; norm.Add(x); }
                    var resolved = new int[outRank];
                    int ii = 0;
                    for (int j = 0; j < outRank; j++)
                        resolved[j] = norm.Contains(j) ? 1 : (ii < inShape.Length ? inShape[ii++] : 1);
                    if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
                }
            }

            // Runtime Range: output length = ceil((limit - start) / delta), knowable only
            // from the scalar input VALUES, not their shapes. RangeOperator.InferOutputShapes
            // returns the [1] placeholder, so without this the output buffer is sized to ONE
            // element and Range writes nothing (its ElementCount>=count guard fails) — collapsing
            // a position-id range like [0,1,2,3,4] to [0]. That silently broke every wpe /
            // position-embedding Gather (only position 0 correct, the rest stale garbage).
            if (node.OpType == "Range" && node.InputNames.Length >= 3)
            {
                var startV = runtimeConstants.GetValueOrDefault(node.InputNames[0]);
                var limitV = runtimeConstants.GetValueOrDefault(node.InputNames[1]);
                var deltaV = runtimeConstants.GetValueOrDefault(node.InputNames[2]);
                if (startV != null && startV.Length > 0 && limitV != null && limitV.Length > 0
                    && deltaV != null && deltaV.Length > 0 && deltaV[0] != 0f)
                {
                    int count = Math.Max(0, (int)MathF.Ceiling((limitV[0] - startV[0]) / deltaV[0]));
                    if (count > 0) runtimeOutputShapes = new[] { new[] { count } };
                }
            }

            // Runtime ConstantOfShape: output shape = input shape-tensor VALUES (e.g. [77,77]). Placeholder
            // at compile time → fill buffer collapses → CLIP causal mask broke. Same class as Range.
            if (node.OpType == "ConstantOfShape" && node.InputNames.Length >= 1 && !string.IsNullOrEmpty(node.InputNames[0])
                && runtimeConstants.TryGetValue(node.InputNames[0], out var cosDims) && cosDims.Length > 0)
            {
                var resolved = cosDims.Select(v => (int)MathF.Round(v)).ToArray();
                if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
            }

            // Runtime Expand: output = broadcast(input shape, target-shape VALUES) (input[1] is runtime).
            if (node.OpType == "Expand" && node.InputNames.Length >= 2 && nodeInputs.Length > 0 && nodeInputs[0] != null
                && !string.IsNullOrEmpty(node.InputNames[1])
                && runtimeConstants.TryGetValue(node.InputNames[1], out var expDims) && expDims.Length > 0)
            {
                var tgt = expDims.Select(v => (int)MathF.Round(v)).ToArray();
                var inS = nodeInputs[0]!.Shape; int rank = Math.Max(tgt.Length, inS.Length);
                var resolved = new int[rank];
                for (int dd = 0; dd < rank; dd++)
                {
                    int tv = dd - (rank - tgt.Length) >= 0 ? tgt[dd - (rank - tgt.Length)] : 1;
                    int iv = dd - (rank - inS.Length) >= 0 ? inS[dd - (rank - inS.Length)] : 1;
                    resolved[dd] = Math.Max(tv, iv);
                }
                if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
            }

            // Runtime Shape: output = [input rank] at runtime (compile-time buffer can be too small).
            if (node.OpType == "Shape" && nodeInputs.Length > 0 && nodeInputs[0] != null)
                runtimeOutputShapes = new[] { new[] { nodeInputs[0]!.Shape.Length } };

            // Runtime broadcast re-inference for elementwise/select ops poisoned by an upstream
            // value-dependent placeholder — re-infer the output from the ACTUAL runtime input shapes.
            if ((node.OpType == "Where" || node.OpType == "Cast" || node.OpType == "Add" || node.OpType == "Sub"
                 || node.OpType == "Mul" || node.OpType == "Div" || node.OpType == "Equal" || node.OpType == "Less"
                 || node.OpType == "Greater" || node.OpType == "And" || node.OpType == "Or" || node.OpType == "Not"
                 || node.OpType == "Min" || node.OpType == "Max") && nodeInputs.Length > 0)
            {
                int wr = 0;
                foreach (var t in nodeInputs) if (t != null) wr = Math.Max(wr, t.Shape.Length);
                if (wr > 0)
                {
                    var resolved = new int[wr];
                    for (int dd = 0; dd < wr; dd++)
                    {
                        int mx = 1;
                        foreach (var t in nodeInputs) if (t != null) { int id = dd - (wr - t.Shape.Length); if (id >= 0) mx = Math.Max(mx, t.Shape[id]); }
                        resolved[dd] = mx;
                    }
                    if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
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
                Format = Format,
                InputNames = node.InputNames,
                ConstantValues = runtimeConstants,
                QuantizedWeights = _quantizedWeights,
                Registry = _registry,
                IntegerTensorNames = _integerTensorNames,
            };
            var nodeSw = VerboseLogging ? System.Diagnostics.Stopwatch.StartNew() : null;
            node.Operator.Execute(ctx);
            if (VerboseLogging && nodeSw != null)
            {
                _accelerator.Flush();   // submit (4.12.0: Synchronize() throws on browser)
                nodeSw.Stop();
                Console.WriteLine($"[GraphExecutor]   -> {node.OpType} took {nodeSw.Elapsed.TotalMilliseconds:F0}ms");
                Console.Out.Flush();
            }

            // Flush GPU command buffer periodically (64 nodes between flushes).
            // Flush() not Synchronize(): the intent is to submit the encoder, not block-wait;
            // and Synchronize() throws on browser in 4.12.0 (submit+wait is desktop-only).
            if (nodeIdx > 0 && nodeIdx % 64 == 0)
                _accelerator.Flush();

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
                            _ew.Scale(outTensor.Data.SubView(0, elCount), tmpBuf.View, elCount, 1f);
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

        // Flush all dispatches before readback. Flush() (submit) not Synchronize() (submit+wait,
        // which throws on browser in 4.12.0): Run() returns GPU tensor REFERENCES, and callers
        // read them via the async readback path (CopyToHostAsync) which drains on its own; a
        // desktop sync read also synchronizes implicitly, so no completion is lost here.
        _accelerator.Flush();

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
    /// Async version of Run. Required for browser backends (WebGPU/WebGL/Wasm): a synchronous
    /// Synchronize() only FLUSHES (dispatches/submits) the GPU queue and returns WITHOUT waiting —
    /// it does NOT deadlock — so results aren't ready for a synchronous readback. You must
    /// SynchronizeAsync() to AWAIT GPU completion. Periodically awaits SynchronizeAsync() to
    /// flush + drain GPU command buffers.
    /// </summary>
    public async Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
    {
        LastRunOpLog.Clear();
        LastRunIntegerDivCount = 0;
        LastRunReadbackCount = 0;
        LastRunReadbackMs = 0;
        LastRunReadbackNames.Clear();
        LastRunSyncDrainCount = 0;
        LastRunSyncDrainMs = 0;
        var _runSw = System.Diagnostics.Stopwatch.StartNew();
        var _drainSw = new System.Diagnostics.Stopwatch();
        var tensors = new Dictionary<string, Tensor>();
        foreach (var (name, tensor) in inputs) tensors[name] = tensor;
        foreach (var (name, tensor) in _weights) tensors[name] = tensor;

        // Reference counting for buffer recycling + the runtime-constant map. Clone the graph-fixed templates
        // (precomputed once by EnsureRunTemplates) and pin only this call's inputs — instead of re-walking all
        // ~1400 nodes for the refcounts + a LINQ Constant scan + a node walk to strip stale constants on EVERY
        // token. The templates already pin graph outputs + weights to int.MaxValue and pre-strip non-Constant
        // node outputs from the constants, so the result is identical to the per-run rebuild.
        EnsureRunTemplates();
        var refCounts = new Dictionary<string, int>(_baseRefCounts!);
        foreach (var name in inputs.Keys) refCounts[name] = int.MaxValue;

        var runtimeConstants = new Dictionary<string, float[]>(_cleanConstants!);

        // Decode-loop output recycling: this executor is reused every step (fixed-shape decode), but
        // the graph's OUTPUT buffers (logits + present.*) are never refcount-released — they're the
        // results. Without this they'd accumulate ~13 fresh buffers/step (logits alone ≈11MB) and OOM
        // long generations. Gated on CacheShapeReadbacks (the decode-loop signal, where the caller has
        // consumed the prior outputs before this call): return last run's output buffers to the pool
        // BEFORE renting, so this run's same-named, same-shape outputs reuse them. _namedBuffers still
        // maps the output names to the prior buffers here, so Return→bucket→Rent recycles cleanly.
        if (CacheShapeReadbacks && _priorRunOutputs != null)
        {
            foreach (var t in _priorRunOutputs) _pool.Return(t);
            _priorRunOutputs = null;
        }

        // Readback cache (auto-detecting). Once finalized, seed the proven-stable shape-derived values
        // so the per-node capture loop SKIPS their GPU round-trips. While probing (first two runs),
        // record this run's readbacks for cross-run comparison.
        bool warmReadback = CacheShapeReadbacks && _readbackStableFinalized && _readbackStable != null;
        if (warmReadback)
            foreach (var (k, v) in _readbackStable!) runtimeConstants[k] = v;
        Dictionary<string, float[]>? readbackThisRun =
            (CacheShapeReadbacks && !_readbackStableFinalized) ? new Dictionary<string, float[]>() : null;

        int nodeIdx = 0;
        var pendingReleases = new List<Tensor>();
        long pendingReleaseBytes = 0; // bytes of buffers in pendingReleases; triggers an early drain past the cap
        // Mixed-precision activations (ActivationDtype != F32): eligible float feature-map intermediates are
        // stored low-p here (NOT in `tensors`); consumers convert back to an fp32 temp at input-gather.
        // Empty + untouched when ActivationDtype == F32 (the whole path is guarded), so F32 is unchanged.
        var halfTensors = new Dictionary<string, HalfTensor>();
        var pendingHalfReleases = new List<HalfTensor>();

        // Decrement each consumed input's refcount and, when it hits zero, defer-release its buffer (low-p or
        // fp32) until the next drain (ordered, browser-safe). Shared by the normal fp32 path and the F16
        // precision-aware pass-through below — single source of truth for input lifetime.
        void ReleaseConsumedInputs(CompiledNode n)
        {
            foreach (var inputName in n.InputNames)
            {
                if (string.IsNullOrEmpty(inputName)) continue;
                if (refCounts.TryGetValue(inputName, out var rc) && rc < int.MaxValue)
                {
                    refCounts[inputName] = rc - 1;
                    if (rc - 1 <= 0)
                    {
                        // Release low-p OR fp32 storage, whichever holds this tensor (deferred to the drain).
                        if (halfTensors.TryGetValue(inputName, out var hrel))
                        {
                            halfTensors.Remove(inputName);
                            pendingHalfReleases.Add(hrel);
                            pendingReleaseBytes += (long)hrel.ElementCount * 2;
                        }
                        else if (tensors.TryGetValue(inputName, out var releaseTensor))
                        {
                            pendingReleases.Add(releaseTensor);
                            pendingReleaseBytes += (long)releaseTensor.ElementCount * sizeof(float);
                        }
                    }
                }
            }
        }

        // Periodic GPU command-buffer drain: flush + wait every SyncIntervalNodes nodes, or early when the
        // deferred-release backlog exceeds MaxPendingReleaseBytes, then return the deferred buffers. Shared by
        // both execution paths so peak GPU memory is bounded to ~(live set + cap) regardless of which path ran.
        async Task DrainPointAsync()
        {
            if (nodeIdx % SyncIntervalNodes == 0 || pendingReleaseBytes >= MaxPendingReleaseBytes)
            {
                _drainSw.Restart();
                try { await _accelerator.SynchronizeAsync(); }
                catch (Exception syncEx)
                {
                    var tailStart = Math.Max(0, LastRunOpLog.Count - 40);
                    var tailLen = LastRunOpLog.Count - tailStart;
                    var tail = string.Join(" | ", LastRunOpLog.GetRange(tailStart, tailLen));
                    throw new Exception(
                        $"[GE node-{nodeIdx} sync] {syncEx.Message} || last {tailLen} ops: {tail}");
                }
                _drainSw.Stop(); LastRunSyncDrainCount++; LastRunSyncDrainMs += _drainSw.Elapsed.TotalMilliseconds;
                // Now safe to return deferred buffers — GPU has finished reading them
                foreach (var t in pendingReleases)
                    _pool.Return(t);
                foreach (var h in pendingHalfReleases)
                    _pool.ReturnHalf(h);
                pendingReleases.Clear();
                pendingHalfReleases.Clear();
                pendingReleaseBytes = 0;
            }
        }

        // F16 precision-aware pass-through: when an op implements IPrecisionAwareOperator AND all its inputs are
        // resolvable (low-p or fp32) AND its output is half-eligible, run it reading low-p inputs and writing a
        // RentHalf output DIRECTLY — NO fp32 temp. Returns the half output on success (caller stores it), or null
        // to fall back to the fp32 convert-around-node path. This is what actually cuts the activation working
        // set (the convert-around-node path keeps an fp32 temp live next to the fp32 output, so it does not).
        HalfTensor? TryPrecisionAwarePassThrough(CompiledNode n, IPrecisionAwareOperator pao)
        {
            if (_paOpsAllowlist != null && !_paOpsAllowlist.Contains(n.OpType)) return null;
            if (n.OutputNames.Length != 1) return null;
            var outName = n.OutputNames[0];
            if (string.IsNullOrEmpty(outName)) return null;
            if (_graph.OutputNames.Contains(outName)) return null;                  // graph outputs stay fp32 for the caller
            if (_integerTensorNames.Contains(outName)) return null;
            if (runtimeConstants.ContainsKey(outName)) return null;
            if (refCounts.TryGetValue(outName, out var orc) && orc >= int.MaxValue) return null;  // retained (shape-cache)
            if (n.OutputShapes.Length < 1) return null;
            var rawShape = n.OutputShapes[0];
            var outShape = new int[rawShape.Length];
            for (int d = 0; d < rawShape.Length; d++) outShape[d] = rawShape[d] <= 0 ? 1 : rawShape[d];
            if (TensorHelpers.ElementCount(outShape) < 4096) return null;           // small tensors stay fp32 (parity with store floor)

            var mixed = new PrecisionAwareInput[n.InputNames.Length];
            for (int i = 0; i < n.InputNames.Length; i++)
            {
                var name = n.InputNames[i];
                if (string.IsNullOrEmpty(name)) return null;                        // these ops have no empty-optional inputs we pass through
                if (halfTensors.TryGetValue(name, out var h)) mixed[i] = new PrecisionAwareInput(h);
                else if (tensors.TryGetValue(name, out var t)) mixed[i] = new PrecisionAwareInput(t);
                else return null;                                                   // input not materialized → can't pass through
            }

            var halfOut = _pool.RentHalf(outShape, outName);
            var pctx = new OnnxOpContext
            {
                Inputs = Array.Empty<Tensor>(),
                Outputs = Array.Empty<Tensor>(),
                Attributes = n.Attributes,
                Pool = _pool,
                Format = Format,
                InputNames = n.InputNames,
                ConstantValues = runtimeConstants,
                IntegerTensorNames = _integerTensorNames,
                Registry = _registry,
            };
            bool ok;
            try { ok = pao.TryExecuteHalf(pctx, mixed, halfOut, _precisionAware); }
            catch { _pool.ReturnHalf(halfOut); throw; }   // a real kernel failure must surface, not silently fall back
            if (!ok) { _pool.ReturnHalf(halfOut); return null; }
            return halfOut;
        }

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

            // ── F16 precision-aware pass-through (approach i) ──
            // Before the fp32 gather, if this op can run read-low-p / write-low-p with no fp32 temp, do so. On
            // success the node is fully handled here (output stored low-p, inputs released, drain advanced) and
            // we skip the fp32 path. On a miss (null) we fall through to the convert-around-node fp32 path.
            if (ActivationDtype == ActivationPrecision.F16 && node.Operator is IPrecisionAwareOperator pao)
            {
                var halfOut = TryPrecisionAwarePassThrough(node, pao);
                if (halfOut != null)
                {
                    halfTensors[node.OutputNames[0]] = halfOut;
                    ReleaseConsumedInputs(node);
                    nodeIdx++;
                    LastRunOpLog.Add($"{nodeIdx:D4} {node.OpType}~f16");
                    await DrainPointAsync();
                    if (BreakAtNode.HasValue && nodeIdx >= BreakAtNode.Value) break;
                    continue;
                }
            }

            var nodeInputs = new Tensor[node.InputNames.Length];
            for (int i = 0; i < node.InputNames.Length; i++)
            {
                var name = node.InputNames[i];
                if (string.IsNullOrEmpty(name)) continue;
                // Mixed-precision: an input stored low-p → convert to an fp32 temp for the (fp32) operator.
                // The temp is deferred-released after the op (ordered, browser-safe) via pendingReleases.
                if (halfTensors.TryGetValue(name, out var htIn))
                {
                    EnsureConvert();
                    var tmp = _pool.Rent(htIn.Shape, $"__f32in_{nodeIdx}_{i}");
                    _convert!.HalfToFloat(htIn.Data, tmp.Data.SubView(0, htIn.ElementCount), htIn.ElementCount);
                    nodeInputs[i] = tmp;
                    pendingReleases.Add(tmp);
                    pendingReleaseBytes += (long)htIn.ElementCount * sizeof(float);
                    continue;
                }
                if (!tensors.TryGetValue(name, out var tensor))
                    throw new InvalidOperationException($"Tensor '{name}' not found (needed by {node.OpType})");
                nodeInputs[i] = tensor;
            }

            // Runtime shape cascade (same as sync Run — see comments there)
            var actualInputShapes = nodeInputs
                .Select(t => t?.Shape ?? Array.Empty<int>())
                .ToArray();

            // Use COMPILED shapes by default (same as sync Run path).
            // Full runtime re-inference caused cascading shape mismatches in attention blocks;
            // dynamic input shapes are instead handled by InferenceSession recompiling the graph
            // for the actual shape, so this executor always runs a graph compiled for THESE dims.
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
            // Runtime Pad: opset >= 11 has pads as input[1] tensor, not as attribute.
            // PadOperator.InferOutputShapes can't read the tensor value at compile time
            // and returns inputs[0] unchanged — output buffer would be sized to INPUT,
            // but the kernel writes to OUTPUT (input + pads) → OOB. Resolve here.
            // 2026-05-04 Data: surfaced StyleMosaic Wasm hang via PerOpSync diag at
            // disp=176 'Kernel_PadImpl' with items=108M into 102.7M-float V0 buffer.
            if (node.OpType == "Pad")
            {
                // Try runtime constant first
                int[]? padsResolved = null;
                if (node.InputNames.Length >= 2
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var padsTensorRC)
                    && padsTensorRC.Length > 0)
                {
                    padsResolved = padsTensorRC.Select(v => (int)v).ToArray();
                }
                // Fallback to attribute (opset < 11)
                else if (node.Attributes.TryGetValue("pads", out var padsAttrObj) && padsAttrObj is long[] padsAttr)
                {
                    padsResolved = padsAttr.Select(v => (int)v).ToArray();
                }

                if (padsResolved != null)
                {
                    var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                    int rank = inShape.Length;
                    if (padsResolved.Length == 2 * rank)
                    {
                        var resolved = new int[rank];
                        for (int j = 0; j < rank; j++)
                            resolved[j] = inShape[j] + padsResolved[j] + padsResolved[rank + j];
                        if (resolved.All(d => d > 0))
                            runtimeOutputShapes = new[] { resolved };
                    }
                }
                else if (node.InputNames.Length >= 2 && nodeInputs.Length > 1 && nodeInputs[1] != null)
                {
                    // DIAGNOSTIC: arriving here means session-init Pad pre-extraction missed this node.
                    System.Threading.Interlocked.Increment(ref PadReadbackFallbackFiredCount);
                    LastPadReadbackFallbackInfo = $"async output={(node.OutputNames.Length > 0 ? node.OutputNames[0] : "?")} pads={node.InputNames[1]}";
                    if (VerboseLogging) Console.WriteLine($"[GraphExecutor] PAD READBACK FALLBACK FIRED ({LastPadReadbackFallbackInfo}) - session-init pre-extraction missed this node");

                    // Pads tensor exists on GPU but wasn't pre-extracted as a runtime constant.
                    // Async readback (this is the async RunAsync path).
                    var padsTensor = nodeInputs[1]!;
                    if (padsTensor.ElementCount > 0 && padsTensor.ElementCount <= 32)
                    {
                        try
                        {
                            using var tmp = _accelerator.Allocate1D<float>(padsTensor.ElementCount);
                            _ew.Scale(padsTensor.Data.SubView(0, padsTensor.ElementCount), tmp.View, padsTensor.ElementCount, 1f);
                            await _accelerator.SynchronizeAsync();
                            var padsHost = await tmp.CopyToHostAsync<float>(0, padsTensor.ElementCount);
                            runtimeConstants[node.InputNames[1]] = padsHost;
                            var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                            int rank = inShape.Length;
                            if (padsHost.Length == 2 * rank)
                            {
                                var resolved = new int[rank];
                                for (int j = 0; j < rank; j++)
                                    resolved[j] = inShape[j] + (int)padsHost[j] + (int)padsHost[rank + j];
                                if (resolved.All(d => d > 0))
                                    runtimeOutputShapes = new[] { resolved };
                            }
                        }
                        catch { /* fall through to compiled shapes */ }
                    }
                }
            }

            // Runtime Unsqueeze: re-infer the output shape from the ACTUAL runtime input
            // shape (insert size-1 dims at axes). At compile time, an upstream dynamic op
            // (e.g. Range, whose placeholder shape is [1]) poisons Unsqueeze's compiled output
            // shape, so the buffer is mis-sized and element count collapses. Unsqueeze only
            // inserts size-1 dims, so re-inferring from the real input shape is always safe and
            // preserves element count. (Needed so the position-id range survives Range→Unsqueeze
            // →Reshape into the wpe Gather.)
            if (node.OpType == "Unsqueeze" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                long[]? axesArr = null;
                if (node.Attributes.TryGetValue("axes", out var axObj) && axObj is long[] al) axesArr = al;
                else if (node.InputNames.Length >= 2 && !string.IsNullOrEmpty(node.InputNames[1])
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var axC) && axC.Length > 0)
                    axesArr = axC.Select(v => (long)v).ToArray();
                if (axesArr != null)
                {
                    var inShape = nodeInputs[0]!.Shape;
                    int outRank = inShape.Length + axesArr.Length;
                    var norm = new HashSet<int>();
                    foreach (var a in axesArr) { int x = (int)a; if (x < 0) x += outRank; norm.Add(x); }
                    var resolved = new int[outRank];
                    int ii = 0;
                    for (int j = 0; j < outRank; j++)
                        resolved[j] = norm.Contains(j) ? 1 : (ii < inShape.Length ? inShape[ii++] : 1);
                    if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
                }
            }

            // Runtime Range: output length = ceil((limit - start) / delta), knowable only
            // from the scalar input VALUES, not their shapes. RangeOperator.InferOutputShapes
            // returns the [1] placeholder, so without this the output buffer is sized to ONE
            // element and Range writes nothing (its ElementCount>=count guard fails) — collapsing
            // a position-id range like [0,1,2,3,4] to [0]. That silently broke every wpe /
            // position-embedding Gather (only position 0 correct, the rest stale garbage).
            if (node.OpType == "Range" && node.InputNames.Length >= 3)
            {
                var startV = runtimeConstants.GetValueOrDefault(node.InputNames[0]);
                var limitV = runtimeConstants.GetValueOrDefault(node.InputNames[1]);
                var deltaV = runtimeConstants.GetValueOrDefault(node.InputNames[2]);
                if (startV != null && startV.Length > 0 && limitV != null && limitV.Length > 0
                    && deltaV != null && deltaV.Length > 0 && deltaV[0] != 0f)
                {
                    int count = Math.Max(0, (int)MathF.Ceiling((limitV[0] - startV[0]) / deltaV[0]));
                    if (count > 0) runtimeOutputShapes = new[] { new[] { count } };
                }
            }

            // Runtime ConstantOfShape: output shape = input shape-tensor VALUES (e.g. [77,77]). Placeholder
            // at compile time → fill buffer collapses → CLIP causal mask broke. Same class as Range.
            if (node.OpType == "ConstantOfShape" && node.InputNames.Length >= 1 && !string.IsNullOrEmpty(node.InputNames[0])
                && runtimeConstants.TryGetValue(node.InputNames[0], out var cosDims) && cosDims.Length > 0)
            {
                var resolved = cosDims.Select(v => (int)MathF.Round(v)).ToArray();
                if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
            }

            // Runtime Expand: output = broadcast(input shape, target-shape VALUES) (input[1] is runtime).
            if (node.OpType == "Expand" && node.InputNames.Length >= 2 && nodeInputs.Length > 0 && nodeInputs[0] != null
                && !string.IsNullOrEmpty(node.InputNames[1])
                && runtimeConstants.TryGetValue(node.InputNames[1], out var expDims) && expDims.Length > 0)
            {
                var tgt = expDims.Select(v => (int)MathF.Round(v)).ToArray();
                var inS = nodeInputs[0]!.Shape; int rank = Math.Max(tgt.Length, inS.Length);
                var resolved = new int[rank];
                for (int dd = 0; dd < rank; dd++)
                {
                    int tv = dd - (rank - tgt.Length) >= 0 ? tgt[dd - (rank - tgt.Length)] : 1;
                    int iv = dd - (rank - inS.Length) >= 0 ? inS[dd - (rank - inS.Length)] : 1;
                    resolved[dd] = Math.Max(tv, iv);
                }
                if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
            }

            // Runtime Shape: output = [input rank] at runtime (compile-time buffer can be too small).
            if (node.OpType == "Shape" && nodeInputs.Length > 0 && nodeInputs[0] != null)
                runtimeOutputShapes = new[] { new[] { nodeInputs[0]!.Shape.Length } };

            // Runtime broadcast re-inference for elementwise/select ops poisoned by an upstream
            // value-dependent placeholder — re-infer the output from the ACTUAL runtime input shapes.
            if ((node.OpType == "Where" || node.OpType == "Cast" || node.OpType == "Add" || node.OpType == "Sub"
                 || node.OpType == "Mul" || node.OpType == "Div" || node.OpType == "Equal" || node.OpType == "Less"
                 || node.OpType == "Greater" || node.OpType == "And" || node.OpType == "Or" || node.OpType == "Not"
                 || node.OpType == "Min" || node.OpType == "Max") && nodeInputs.Length > 0)
            {
                int wr = 0;
                foreach (var t in nodeInputs) if (t != null) wr = Math.Max(wr, t.Shape.Length);
                if (wr > 0)
                {
                    var resolved = new int[wr];
                    for (int dd = 0; dd < wr; dd++)
                    {
                        int mx = 1;
                        foreach (var t in nodeInputs) if (t != null) { int id = dd - (wr - t.Shape.Length); if (id >= 0) mx = Math.Max(mx, t.Shape[id]); }
                        resolved[dd] = mx;
                    }
                    if (resolved.All(d => d > 0)) runtimeOutputShapes = new[] { resolved };
                }
            }

            // Fixed-shape decode: reuse the Shape op's output BUFFER across steps instead of
            // re-uploading the (constant) input dims via CopyFromCPU every step. See
            // _shapeBufferCache and feedback-shape-outputs-consumed-as-gpu-tensor-not-just-value.
            // We reuse the BUFFER (not just publish the value) because a downstream consumer reads
            // the Shape output as a GPU tensor — skipping the upload outright corrupts output.
            bool shapeCacheHit = false;
            Tensor? cachedShapeBuf = null;
            if (CacheShapeReadbacks && node.OpType == "Shape" && node.OutputNames.Length == 1
                && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                _shapeBufferCache ??= new Dictionary<string, (Tensor Buf, int[] Dims)>();
                var sName = node.OutputNames[0];
                var curDims = nodeInputs[0]!.Shape;
                if (_shapeBufferCache.TryGetValue(sName, out var entry)
                    && entry.Dims.Length == curDims.Length
                    && entry.Dims.AsSpan().SequenceEqual(curDims))
                {
                    // Cache hit: reuse the retained buffer; skip this step's CopyFromCPU upload.
                    shapeCacheHit = true;
                    cachedShapeBuf = entry.Buf;
                    tensors[sName] = entry.Buf;
                    refCounts[sName] = int.MaxValue; // retained across runs; never returned to the pool
                    // Publish the dims to downstream VALUE consumers (Reshape/Slice/Concat targets),
                    // exactly as the post-execute readback-capture would, but with no GPU round-trip.
                    var dimVals = new float[curDims.Length];
                    for (int d = 0; d < curDims.Length; d++) dimVals[d] = curDims[d];
                    runtimeConstants[sName] = dimVals;
                    if (readbackThisRun != null) readbackThisRun[sName] = dimVals;
                }
            }

            // ── Zero-copy metadata-only ops (Reshape / Squeeze / Unsqueeze / Flatten) ──
            // These just reinterpret the same data at a new shape, but their Execute COPIES into a freshly-rented
            // buffer (the 256 MiB VAE GroupNorm reshape was a pure duplicate). When the data input is a
            // single-consumer pooled fp32 intermediate (the common case), HAND OFF its buffer to the output as a
            // view (Tensor over the same Data, new shape) + transfer the pool ownership — no Rent, no copy.
            // Single-consumer = provably safe (no aliasing): this op is the buffer's last reader. Falls through to
            // the copy path for shared / graph-IO / fp16 / shape-cached inputs (and tiny / shape-value ones).
            if ((node.OpType is "Reshape" or "Squeeze" or "Unsqueeze" or "Flatten") && !shapeCacheHit
                && node.OutputNames.Length == 1 && nodeInputs.Length >= 1 && nodeInputs[0] != null)
            {
                var src = nodeInputs[0];
                var srcName = node.InputNames[0];
                var outName = node.OutputNames[0];
                if (!string.IsNullOrEmpty(srcName) && !string.IsNullOrEmpty(outName) && !src.IsHalf
                    && src.ElementCount >= 4096                                  // only LARGE reshapes (the memory win);
                    && !runtimeConstants.ContainsKey(outName)                    // not a value a downstream reads on CPU
                    && tensors.TryGetValue(srcName, out var srcT) && ReferenceEquals(srcT, src)
                    && refCounts.TryGetValue(srcName, out var srcRc) && srcRc == 1
                    && !_graph.OutputNames.Contains(srcName))
                {
                    var rshape = (runtimeOutputShapes.Length > 0 ? runtimeOutputShapes[0] : node.OutputShapes[0]).ToArray();
                    for (int d = 0; d < rshape.Length; d++) if (rshape[d] <= 0) rshape[d] = 1;
                    if (TensorHelpers.ElementCount(rshape) == src.ElementCount && _pool.Rename(srcName, outName))
                    {
                        tensors[outName] = new Tensor(src.Data, rshape, outName);
                        tensors.Remove(srcName);
                        refCounts[srcName] = 0;   // buffer handed off to outName; never re-release srcName
                        // Release this Reshape's OTHER inputs (e.g. the shape tensor) — the data input was handed off.
                        for (int ii = 1; ii < node.InputNames.Length; ii++)
                        {
                            var inN = node.InputNames[ii];
                            if (string.IsNullOrEmpty(inN)) continue;
                            if (refCounts.TryGetValue(inN, out var rc) && rc < int.MaxValue)
                            {
                                refCounts[inN] = rc - 1;
                                if (rc - 1 <= 0 && tensors.TryGetValue(inN, out var rt))
                                { pendingReleases.Add(rt); pendingReleaseBytes += (long)rt.ElementCount * sizeof(float); }
                            }
                        }
                        nodeIdx++;
                        LastRunOpLog.Add($"{nodeIdx:D4} {node.OpType}~view");
                        await DrainPointAsync();
                        if (BreakAtNode.HasValue && nodeIdx >= BreakAtNode.Value) break;
                        continue;
                    }
                }
            }

            // ── In-place InstanceNormalization ──
            // InstanceNorm's pass-2 reads in[idx] then writes out[idx]; with a single read_write buffer (the
            // in-place kernel) it can write back over its input. When that input is a single-consumer pooled fp32
            // intermediate (refCount==1 → the norm is its last reader; a residual/shortcut share would make it >1
            // and fall through to the two-buffer copy), normalize IN PLACE and hand the buffer to the output —
            // dropping the separate output feature map (256 MiB in the SD VAE at 512²). fp32 scale/bias only.
            if (node.OpType == "InstanceNormalization" && !shapeCacheHit
                && node.OutputNames.Length == 1 && nodeInputs.Length >= 3
                && nodeInputs[0] != null && nodeInputs[1] != null && nodeInputs[2] != null)
            {
                var src = nodeInputs[0]; var srcName = node.InputNames[0]; var outName = node.OutputNames[0];
                if (!string.IsNullOrEmpty(srcName) && !string.IsNullOrEmpty(outName)
                    && !src.IsHalf && !nodeInputs[1].IsHalf && !nodeInputs[2].IsHalf && src.ElementCount >= 4096
                    && tensors.TryGetValue(srcName, out var st) && ReferenceEquals(st, src)
                    && refCounts.TryGetValue(srcName, out var rc) && rc == 1
                    && !_graph.OutputNames.Contains(srcName) && _pool.Rename(srcName, outName))
                {
                    var shape = src.Shape;
                    var (nN, nC, _, _) = shape.Length >= 4 ? LayoutHelper.GetDims(shape, Format)
                        : (shape[0], shape.Length > 1 ? shape[1] : 1, 1, 1);
                    int sp = src.ElementCount / (nN * nC);
                    _normalization.InstanceNormInPlace(src.Data, nodeInputs[1].Data, nodeInputs[2].Data, nN, nC, sp);
                    tensors[outName] = new Tensor(src.Data, shape, outName);
                    tensors.Remove(srcName);
                    refCounts[srcName] = 0;
                    // Release the scale/bias inputs (the data input was handed off in place).
                    for (int ii = 1; ii < node.InputNames.Length; ii++)
                    {
                        var inN = node.InputNames[ii];
                        if (string.IsNullOrEmpty(inN)) continue;
                        if (refCounts.TryGetValue(inN, out var brc) && brc < int.MaxValue)
                        {
                            refCounts[inN] = brc - 1;
                            if (brc - 1 <= 0 && tensors.TryGetValue(inN, out var rt))
                            { pendingReleases.Add(rt); pendingReleaseBytes += (long)rt.ElementCount * sizeof(float); }
                        }
                    }
                    nodeIdx++;
                    LastRunOpLog.Add($"{nodeIdx:D4} InstanceNormalization~inplace");
                    await DrainPointAsync();
                    if (BreakAtNode.HasValue && nodeIdx >= BreakAtNode.Value) break;
                    continue;
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
                // Shape-cache hit: bind the retained buffer instead of renting a fresh one.
                nodeOutputs[i] = shapeCacheHit && i == 0 ? cachedShapeBuf! : _pool.Rent(shape, name);
            }

            // ── GGUF incremental-decode KV-cache intercept (gated on DecodeKVCache; normal forward
            //    untouched when it's null) ──
            // The static graph rope's/attends at kv_offset=0 (full-recompute). In decode mode the step's
            // tokens live at absolute positions [DecodePastLen, DecodePastLen+seqQ). So:
            //  - RoPE nodes: override kv_offset = DecodePastLen, so Q/K rotate at their true position
            //    (position = kv_offset + row/rows_per_position). Uniform across all RoPE nodes in a step.
            //  - FusedAttention nodes (tagged "layer"): write this step's K/V (the post-transpose
            //    head-major [kvHeads,seqQ,hd] inputs) into the per-layer cache at DecodePastLen, then
            //    feed the kernel the FULL history [kvHeads, DecodePastLen+seqQ, hd] with kv_offset =
            //    DecodePastLen. Prefill (seq=N, pastLen=0) is numerically identical to the normal forward
            //    AND populates the cache; decode (seq=1) attends the new query against all cached tokens.
            Dictionary<string, object>? decodeAttrs = null;
            if (DecodeKVCache != null && node.Attributes != null)
            {
                if (node.OpType == "RoPE")
                {
                    decodeAttrs = new Dictionary<string, object>(node.Attributes) { ["kv_offset"] = (long)DecodePastLen };
                }
                else if (node.OpType == "FusedAttention"
                    && node.Attributes.TryGetValue("layer", out var _dLayerEl)
                    && nodeInputs.Length >= 3 && nodeInputs[1] != null && nodeInputs[2] != null)
                {
                    int dLayer = Convert.ToInt32(_dLayerEl);
                    int dHd = DecodeKVCache.HeadDim(dLayer), dKvH = DecodeKVCache.KvHeads(dLayer);
                    int dSeqQ = nodeInputs[1]!.ElementCount / (dKvH * dHd);
                    int dTotal = DecodePastLen + dSeqQ;
                    // Write this step's K/V into the per-layer store (CopyFromAsync orders the copy against the
                    // producing kernel on the Wasm worker pool — a sync CopyFrom of a node OUTPUT silently races
                    // there; the await completes it before the attention kernel below reads the store).
                    await DecodeKVCache.WriteAsync(dLayer, nodeInputs[1]!.Data, nodeInputs[2]!.Data, DecodePastLen, dSeqQ).ConfigureAwait(false);
                    // Feed FusedAttention the [kvHeads, maxSeq, hd] store DIRECTLY in its native type (bf16/f32),
                    // read maxSeq-strided — NO per-token repack + bf16→f32-widen of the whole history (that was
                    // O(history) memory-bandwidth per token). kv_seq_len = the LIVE length (the store is padded).
                    // EXCEPTION: WebGL's sub-word (bf16) kernel read of the large strided store mis-addresses
                    // (an ILGPU WebGL backend limitation — f32-strided + all other backends are byte-exact;
                    // surfaced to Geordi). WebGL+bf16 falls back to the repack (correct, just the old O(history)).
                    bool stridedOk = _accelerator.AcceleratorType != AcceleratorType.WebGL
                                     || DecodeKVCache.Precision == KVCachePrecision.F32;
                    if (stridedOk)
                    {
                        nodeInputs[1] = DecodeKVCache.StoreK(dLayer, node.InputNames[1]);
                        nodeInputs[2] = DecodeKVCache.StoreV(dLayer, node.InputNames[2]);
                        decodeAttrs = new Dictionary<string, object>(node.Attributes)
                        {
                            ["kv_offset"] = (long)DecodePastLen,
                            ["kv_seq_len"] = (long)dTotal,
                        };
                    }
                    else
                    {
                        nodeInputs[1] = new Tensor(await DecodeKVCache.PackedKAsync(dLayer, dTotal).ConfigureAwait(false), new[] { 1, dKvH, dTotal, dHd }, node.InputNames[1]);
                        nodeInputs[2] = new Tensor(await DecodeKVCache.PackedVAsync(dLayer, dTotal).ConfigureAwait(false), new[] { 1, dKvH, dTotal, dHd }, node.InputNames[2]);
                        decodeAttrs = new Dictionary<string, object>(node.Attributes) { ["kv_offset"] = (long)DecodePastLen };
                    }
                }
            }

            var ctx = new OnnxOpContext
            {
                Inputs = nodeInputs,
                Outputs = nodeOutputs,
                Attributes = decodeAttrs ?? node.Attributes,
                Pool = _pool,
                Format = Format,
                InputNames = node.InputNames,
                ConstantValues = runtimeConstants,
                QuantizedWeights = _quantizedWeights,
                Registry = _registry,
                IntegerTensorNames = _integerTensorNames,
            };
            // CapturedNodeTimingsMs (opt-in): wall-clock time per Execute + optional sync.
            // Captures via Stopwatch around node.Operator.Execute (and the PerOpSync sync
            // when enabled). Pairs with CapturedOutputs key for cross-lookup.
            var sw = CapturedNodeTimingsMs != null ? System.Diagnostics.Stopwatch.StartNew() : null;
            try
            {
                // Async dispatch: operators that need a runtime GPU->CPU readback (Loop/If/Scan
                // condition, Einsum dynamic inputs) override ExecuteAsync to await the
                // browser-safe readback; all others fall through to the synchronous Execute via
                // the default interface method. This is the browser-parity path.
                // shapeCacheHit: buffer already holds the correct dims from a prior step — skip.
                if (!shapeCacheHit) await node.Operator.ExecuteAsync(ctx);
                // PerOpSync: opt-in diagnostic flag (off by default). Forces a flush + wait
                // after every Execute so async-backend kernel traps (Wasm worker errors,
                // WebGPU command-encoder errors) surface AT the failing node instead of
                // being lumped into the next periodic 64-node sync. Significant perf cost;
                // only useful for kernel-bisection debugging.
                if (PerOpSync) await _accelerator.SynchronizeAsync();
                if (sw != null && CapturedNodeTimingsMs != null && node.OutputNames.Length > 0)
                {
                    sw.Stop();
                    // Match CapturedOutputs key format (nodeIdx is pre-increment here, same
                    // as where CapturedOutputs is written downstream).
                    CapturedNodeTimingsMs[$"{nodeIdx:D3}_{node.OpType}_{node.OutputNames[0]}"] = sw.Elapsed.TotalMilliseconds;
                }
            }
            catch (Exception ex)
            {
                var inputInfo = string.Join(", ", nodeInputs.Where(t => t != null).Select(t => $"[{string.Join(",", t.Shape)}]({t.ElementCount})"));
                var outputInfo = string.Join(", ", node.OutputShapes.Select(s => $"[{string.Join(",", s)}]"));
                var msg = $"Node {nodeIdx}/{_graph.Nodes.Length} '{node.OpType}' failed: {ex.Message} | " +
                    $"Inputs: [{string.Join(",", node.InputNames)}] shapes=({inputInfo}), " +
                    $"Outputs: [{string.Join(",", node.OutputNames)}] shapes=({outputInfo})";
                if (InferenceSession.VerboseLogging) Console.WriteLine($"[GraphExecutor] ERROR: {msg}");
                throw new InvalidOperationException(msg, ex);
            }

            for (int i = 0; i < node.OutputNames.Length; i++)
                tensors[node.OutputNames[i]] = nodeOutputs[i];

            // First step (or shape changed): retain this Shape output buffer so later steps reuse it.
            // refCounts=MaxValue keeps it out of the pool's return path for the rest of this run AND
            // marks it retained; it's freed when the pool is disposed (BufferPool tracks _allBuffers).
            if (!shapeCacheHit && CacheShapeReadbacks && node.OpType == "Shape" && node.OutputNames.Length == 1
                && nodeInputs.Length > 0 && nodeInputs[0] != null && nodeOutputs.Length > 0 && nodeOutputs[0] != null)
            {
                var sName = node.OutputNames[0];
                (_shapeBufferCache ??= new Dictionary<string, (Tensor Buf, int[] Dims)>())[sName] =
                    (nodeOutputs[0], (int[])nodeInputs[0]!.Shape.Clone());
                refCounts[sName] = int.MaxValue;
            }

            // Capture small intermediate outputs as runtime constants.
            // Only sync+readback for truly small shape tensors (≤64 elements) that downstream
            // operators need for parameter resolution (Slice starts/ends, Reshape dims, Expand shapes).
            // Was ≤2048 with double-sync per node — killed GPT-2 perf with hundreds of unnecessary syncs.
            // Shape-cache hit already published its value above (no GPU output to read) — skip.
            if (!shapeCacheHit)
            for (int oi = 0; oi < nodeOutputs.Length; oi++)
            {
                var outTensor = nodeOutputs[oi];
                if (outTensor != null && outTensor.ElementCount > 0 && outTensor.ElementCount <= 64)
                {
                    var outName = oi < node.OutputNames.Length ? node.OutputNames[oi] : null;
                    // Static skip set: this output is feature/data that no value-needing op consumes
                    // (e.g. LayerNorm ReduceMean/Add/Sqrt intermediates) — the eager ≤64 readback was
                    // pure waste (token-dependent, so the warm cache can't help). See BuildReadbackSkipSet.
                    if (outName != null && _readbackSkipNames.Contains(outName))
                        continue;
                    // Warm readback cache: value already seeded into runtimeConstants from the proven-
                    // stable set — skip the GPU round-trip entirely.
                    if (outName != null && warmReadback && _readbackStable!.ContainsKey(outName))
                        continue;
                    if (outName != null)
                    {
                        // Defensive null-check: outTensor.Data can be null on backends where
                        // BufferPool.Rent returned an empty tensor for a zero-or-degenerate
                        // shape (replaced to size-1 above), or where SubView is unsupported.
                        // Without this check, "outTensor.Data.SubView(...)" throws bare NRE
                        // and the capture-sync catch reports "Arg_NullReferenceException"
                        // with no clue which line trapped.
                        string captureStage = "init";
                        try
                        {
                            int elCount = outTensor.ElementCount;
                            captureStage = $"alloc[{elCount}]";
                            using var tmpBuf = _accelerator.Allocate1D<float>(elCount);
                            captureStage = $"data-subview[{elCount}]";
                            if (outTensor.Data.Length == 0)
                                throw new InvalidOperationException("outTensor.Data has zero length");
                            var srcView = outTensor.Data.SubView(0, elCount);
                            captureStage = $"scale[{elCount}]";
                            // GPU→GPU copy via Scale kernel (works on all backends),
                            // then async readback via CopyToHostAsync(offset, count).
                            _ew.Scale(srcView, tmpBuf.View, elCount, 1f);
                            captureStage = $"sync[{elCount}]";
                            var _rbSw = System.Diagnostics.Stopwatch.StartNew();
                            await _accelerator.SynchronizeAsync();
                            captureStage = $"copy-back[{elCount}]";
                            runtimeConstants[outName] = await tmpBuf.CopyToHostAsync<float>(0, elCount);
                            _rbSw.Stop();
                            LastRunReadbackCount++;
                            LastRunReadbackMs += _rbSw.Elapsed.TotalMilliseconds;
                            LastRunReadbackNames.Add($"{node.OpType}:{outName}");
                            // Probing: record this run's value for cross-run stability comparison.
                            if (readbackThisRun != null) readbackThisRun[outName] = runtimeConstants[outName];
                        }
                        catch (NotSupportedException) { /* Backend doesn't support async readback */ }
                        catch (NullReferenceException) {
                            // WORKAROUND: SpawnDev.ILGPU 4.9.4 WebGPU CopyToHostAsync NRE on tiny
                            // (1-element / 4-byte) staging buffers. Tracked via
                            // _DevComms/SpawnDev.ILGPU/data-to-geordi-webgpu-tiny-readback-nre-2026-05-03.md.
                            // Cached runtime-constants are an OPTIMIZATION for downstream parameter
                            // resolution (Slice starts/ends, Reshape dims) - swallowing a single
                            // readback just means the downstream op reads the value from the GPU
                            // view directly. Should NOT abort the whole graph run.
                            // Repro: WebGPUTests.Pipeline_YOLOv8 node 227 'Div' shape=[1].
                            // Remove this catch when upstream lands the fix in WebGPUMemoryBuffer.
                        }
                        catch (Exception captureEx)
                        {
                            // The flush during runtime-const capture is the first sync after a
                            // queued kernel might fire its trap (Wasm divide-by-zero, WebGPU
                            // device error). For non-NRE exceptions we still want the augmented
                            // throw because they're real graph-execution failures, not optimization
                            // hiccups. Inline-augment with op log; no wrapping because
                            // SpawnDev.UnitTesting.UnitTestRunner unwraps InnerException on
                            // report and would lose the augmentation.
                            var tailStart = Math.Max(0, LastRunOpLog.Count - 40);
                            var tailLen = LastRunOpLog.Count - tailStart;
                            var tail = string.Join(" | ", LastRunOpLog.GetRange(tailStart, tailLen));
                            var exType = captureEx.GetType().Name;
                            string dataLenStr;
                            try { dataLenStr = outTensor.Data.Length.ToString(); } catch { dataLenStr = "(unreadable)"; }
                            var shape = outTensor.Shape != null ? string.Join(",", outTensor.Shape) : "(null shape)";
                            throw new Exception(
                                $"[GE capture-sync node {nodeIdx + 1} '{node.OpType}' out '{outName}' shape=[{shape}] dataLen={dataLenStr} stage={captureStage}] "
                                + $"{exType}: {captureEx.Message} || last {tailLen} ops: {tail}");
                        }
                    }
                }
            }

            // Capture intermediate values for debugging (when enabled)
            if (!shapeCacheHit && CapturedOutputs != null && nodeOutputs.Length > 0 && nodeOutputs[0] != null)
            {
                var captureOutput = nodeOutputs[0];
                bool nameFiltered = CaptureOutputNames != null;
                if (nameFiltered && !CaptureOutputNames!.Contains(node.OutputNames[0]))
                    goto skipCapture;
                // Capture enough values to get a meaningful absMax (at least one full
                // channel for Conv outputs). 1024 covers most shape tensors and small features.
                // A name-filtered capture takes the FULL tensor (the caller wants this exact intermediate).
                int captureCount = nameFiltered ? captureOutput.ElementCount
                                                : Math.Min(CaptureMaxElements, captureOutput.ElementCount);
                if (captureCount > 0)
                {
                    try
                    {
                        await _accelerator.SynchronizeAsync();
                        using var capBuf = _accelerator.Allocate1D<float>(captureCount);
                        _ew.Scale(captureOutput.Data.SubView(0, captureCount), capBuf.View, captureCount, 1f);
                        await _accelerator.SynchronizeAsync();
                        var vals = await capBuf.CopyToHostAsync<float>(0, captureCount);
                        var key = $"{nodeIdx:D3}_{node.OpType}_{node.OutputNames[0]}";
                        CapturedOutputs[key] = vals;

                        // Sibling capture: per-node metadata keyed identically. Surfaces the
                        // op variant (input shapes, output shapes, attribute hints) that the
                        // captured float[] alone doesn't carry.
                        if (CapturedNodeInfo != null)
                        {
                            string FormatShapes(Tensor?[]? tensors) => tensors == null ? "null"
                                : string.Join(",", tensors.Select(t => t == null ? "(null)" : "[" + string.Join(",", t.Shape) + "]"));
                            string FormatOutShapes(int[][] shapes) => string.Join(",", shapes.Select(s => "[" + string.Join(",", s) + "]"));
                            string attrHint = "";
                            if (node.Attributes != null && node.Attributes.Count > 0)
                            {
                                attrHint = " | attrs: " + string.Join(",", node.Attributes.Take(8).Select(kv => kv.Key + "=" + (kv.Value?.ToString() ?? "null")));
                            }
                            CapturedNodeInfo[key] = $"in: [{string.Join(",", node.InputNames)}] shapes={FormatShapes(nodeInputs)} | out: [{string.Join(",", node.OutputNames)}] shapes={FormatOutShapes(node.OutputShapes)}{attrHint}";
                        }
                    }
                    catch { /* Don't crash on capture failure */ }
                }
                skipCapture: ;
            }

            // Mixed-precision: store eligible LARGE float feature-map outputs as low precision (half the held
            // bytes). The op ran fp32; convert its fp32 output to fp16 storage, move it to `halfTensors`, and
            // free the fp32 buffer. Runs AFTER the runtime-const/capture sections above (they read the fp32
            // output first). Guarded entirely off when ActivationDtype == F32 (→ byte-identical to the fp32
            // path). Excludes graph outputs (caller reads fp32), integer/shape/runtime-const + small tensors,
            // and retained (shape-cache) buffers. dtype seam: extend the convert/rent switch for bf16/fp8.
            if (ActivationDtype != ActivationPrecision.F32 && !shapeCacheHit)
            {
                EnsureConvert();
                for (int oi = 0; oi < node.OutputNames.Length; oi++)
                {
                    var outName = node.OutputNames[oi];
                    var outT = oi < nodeOutputs.Length ? nodeOutputs[oi] : null;
                    if (string.IsNullOrEmpty(outName) || outT == null || outT.ElementCount < 4096) continue;
                    if (_graph.OutputNames.Contains(outName) || _integerTensorNames.Contains(outName)
                        || runtimeConstants.ContainsKey(outName)) continue;
                    if (node.OpType is "Shape" or "ConstantOfShape" or "Cast" or "NonZero" or "Range") continue;
                    if (refCounts.TryGetValue(outName, out var orc) && orc >= int.MaxValue) continue; // retained
                    var half = _pool.RentHalf(outT.Shape, outName);              // ActivationDtype==F16 (only value today)
                    _convert!.FloatToHalf(outT.Data.SubView(0, outT.ElementCount), half.Data, outT.ElementCount);
                    halfTensors[outName] = half;
                    tensors.Remove(outName);                                     // consumers now read it from halfTensors
                    pendingReleases.Add(outT);                                   // free the fp32 buffer (deferred → ordered after the convert)
                    pendingReleaseBytes += (long)outT.ElementCount * sizeof(float);
                }
            }

            // Defer buffer release to sync points to prevent reuse while GPU is in-flight
            ReleaseConsumedInputs(node);

            nodeIdx++;
            LastRunOpLog.Add($"{nodeIdx:D4} {node.OpType}");

            // Flush GPU command buffer periodically (every SyncIntervalNodes, or early when the deferred-release
            // backlog exceeds MaxPendingReleaseBytes) and return the drained buffers. See DrainPointAsync.
            await DrainPointAsync();

            // DIAGNOSTIC: stop early at requested node count to bisect failures.
            if (BreakAtNode.HasValue && nodeIdx >= BreakAtNode.Value)
                break;
        }

        // Final yield + sync
        await Task.Yield();
        _drainSw.Restart();
        try { await _accelerator.SynchronizeAsync(); }
        catch (Exception syncEx)
        {
            var tailStart = Math.Max(0, LastRunOpLog.Count - 40);
            var tailLen = LastRunOpLog.Count - tailStart;
            var tail = string.Join(" | ", LastRunOpLog.GetRange(tailStart, tailLen));
            throw new Exception(
                $"[GE final sync, {LastRunOpLog.Count} ops total] {syncEx.Message} || last {tailLen} ops: {tail}");
        }
        _drainSw.Stop(); LastRunSyncDrainCount++; LastRunSyncDrainMs += _drainSw.Elapsed.TotalMilliseconds;
        // Release any remaining deferred buffers
        foreach (var t in pendingReleases)
            _pool.Return(t);
        foreach (var h in pendingHalfReleases)
            _pool.ReturnHalf(h);
        pendingReleases.Clear();
        pendingHalfReleases.Clear();

        var results = new Dictionary<string, Tensor>();
        foreach (var name in _graph.OutputNames)
        {
            if (tensors.TryGetValue(name, out var tensor))
                results[name] = tensor;
        }

        // Remember this run's output buffers so the next decode step can recycle them (see the
        // CacheShapeReadbacks-gated Return at RunAsync start). Only the distinct output buffers —
        // an output that aliases an input/weight is excluded (its buffer isn't pool-owned).
        if (CacheShapeReadbacks)
            _priorRunOutputs = results.Values
                .Where(t => t.Name != null && !inputs.ContainsKey(t.Name) && !_weights.ContainsKey(t.Name))
                .ToList();

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

        // Auto-detecting readback cache: finalize the proven-stable set from two probe runs.
        if (CacheShapeReadbacks && !_readbackStableFinalized && readbackThisRun != null)
        {
            if (_readbackProbe == null)
            {
                _readbackProbe = readbackThisRun; // first probe run — keep for comparison
            }
            else
            {
                // Second probe run (different input data): a readback is shape-derived (cache-safe)
                // iff its value is IDENTICAL across the two runs. Data-derived ones (input_ids when
                // seq≤64, etc.) differ and are excluded — they keep getting read back every call.
                var stable = new Dictionary<string, float[]>();
                foreach (var (k, v) in readbackThisRun)
                    if (_readbackProbe.TryGetValue(k, out var prev) && prev.Length == v.Length
                        && prev.AsSpan().SequenceEqual(v))
                        stable[k] = v;
                _readbackStable = stable;
                _readbackStableFinalized = true;
                _readbackProbe = null;
            }
        }

        _runSw.Stop(); LastRunTotalMs = _runSw.Elapsed.TotalMilliseconds;
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
        // NOTE: Do NOT use 'using' here — the buffer must survive until RunAsync reads it.
        // It is tracked by _kvCacheFlagBuf and disposed on next call or in Dispose().
        if (_kvCacheInfo.UseCacheBranchInput != null)
        {
            _kvCacheFlagBuf?.Dispose();
            _kvCacheFlagBuf = _accelerator.Allocate1D(new float[] { 1f });
            inputs[_kvCacheInfo.UseCacheBranchInput] = new Tensor(_kvCacheFlagBuf.View, new[] { 1 });
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
        _kvCacheFlagBuf?.Dispose();
        _ew.Dispose();
        _precisionAware.Dispose();
        _normalization.Dispose();
        _convert?.Dispose();
    }

    /// <summary>Lazily create the fp32↔low-precision convert kernels (used only in mixed-precision mode).</summary>
    private void EnsureConvert() => _convert ??= new Kernels.PrecisionConvertKernels(_accelerator);

    // ── Mid-graph readback skip set ──────────────────────────────────────────────
    // The async executor eagerly reads back EVERY ≤64-elem node output into runtimeConstants so that
    // downstream value-consuming ops (Slice starts/ends, Reshape dims, Range bounds, Gather indices, …)
    // can resolve their parameters on the CPU. But many small outputs are pure feature DATA that NO op
    // ever reads as a value — most notably LayerNorm's per-row ReduceMean / Add(eps) / Sqrt intermediates
    // ([1, seq, 1], so ≤64 whenever seq≤64). Those are token-dependent (the warm shape-readback cache
    // can't elide them) so they were read back EVERY decode step — measured 53 readbacks ≈ 2.7s/step on
    // WebGPU GPT-2, a ~32% decode cost for values nothing consumes. We pre-compute, from the static graph,
    // the outputs whose readback is provably unnecessary and skip them.

    /// <summary>Ops whose output is ALWAYS feature/data and is never consumed as a runtime-constant
    /// param value (reductions to a mean/var, activations, big matmuls). Conservative on purpose — only
    /// ops that cannot appear in a shape/index computation. Dual-use ops (Add/Sub/Mul/Div/Pow, ReduceSum/
    /// ReduceProd, Shape/Gather/Concat/Cast) are deliberately EXCLUDED.</summary>
    private static readonly HashSet<string> ReadbackFeatureOnlyProducers = new(StringComparer.Ordinal)
    {
        "ReduceMean", "ReduceMax", "ReduceMin", "ReduceL1", "ReduceL2", "ReduceSumSquare",
        "Sqrt", "Softmax", "LogSoftmax", "Gelu", "Erf", "Relu", "LeakyRelu", "PRelu",
        "Sigmoid", "Tanh", "HardSigmoid", "HardSwish", "SiLU", "Mish", "Softplus", "Elu", "Selu", "Celu",
        "MatMul", "Gemm", "Conv", "ConvTranspose", "LayerNormalization", "BatchNormalization",
        "InstanceNormalization", "GroupNormalization", "RMSNormalization",
    };

    /// <summary>Ops that read at least one input's runtime-constant VALUE and have NO correct GPU-only
    /// fallback for it — i.e. they GENUINELY need the readback (shape/index/param resolution). If an
    /// output feeds any of these, it is NEVER skipped. Over-inclusion here is safe (it only keeps a
    /// readback that may be unnecessary); omission is the dangerous direction, so the list is broad.</summary>
    private static readonly HashSet<string> ReadbackRequiresValueConsumers = new(StringComparer.Ordinal)
    {
        "Reshape", "Slice", "Expand", "Resize", "Upsample", "Pad", "Unsqueeze", "Squeeze", "Range",
        "Tile", "ConstantOfShape", "TopK", "NonZero", "Compress", "Unique", "OneHot", "Gather",
        "GatherElements", "GatherND", "ScatterElements", "ScatterND", "Scatter", "EyeLike", "Multinomial",
        "CumSum", "HannWindow", "HammingWindow", "BlackmanWindow", "Cast", "Mod", "Trilu", "Split",
        "ReduceSum", "ReduceProd", "Concat", "Shape", "Equal", "Where",
    };

    /// <summary>Build the set of node-output names whose ≤64-elem mid-graph readback can be safely skipped.
    /// An output is skipped iff (a) NO consumer is in <see cref="ReadbackRequiresValueConsumers"/>, AND
    /// (b) either its producer is in <see cref="ReadbackFeatureOnlyProducers"/> OR every consumer is itself
    /// a feature-only (value-never-reading) op. Both directions are conservative: the only outputs removed
    /// are ones provably read only as GPU tensors, so correctness is unchanged while the wasted per-step
    /// GPU round-trips disappear.</summary>
    private static HashSet<string> BuildReadbackSkipSet(CompiledGraph graph)
    {
        var skip = new HashSet<string>(StringComparer.Ordinal);
        // producer op-type per output, consumer op-types per tensor name
        var producerOp = new Dictionary<string, string>(StringComparer.Ordinal);
        var consumerOps = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            foreach (var o in node.OutputNames)
                if (!string.IsNullOrEmpty(o)) producerOp[o] = node.OpType;
            foreach (var inp in node.InputNames)
            {
                if (string.IsNullOrEmpty(inp)) continue;
                if (!consumerOps.TryGetValue(inp, out var list)) { list = new List<string>(); consumerOps[inp] = list; }
                list.Add(node.OpType);
            }
        }

        foreach (var node in graph.Nodes)
        {
            foreach (var o in node.OutputNames)
            {
                if (string.IsNullOrEmpty(o)) continue;
                var cons = consumerOps.GetValueOrDefault(o);
                // Any consumer that genuinely needs the value → must keep the readback.
                if (cons != null && cons.Any(c => ReadbackRequiresValueConsumers.Contains(c))) continue;
                bool featureProducer = ReadbackFeatureOnlyProducers.Contains(node.OpType);
                // Rule B: every consumer is itself a feature-only op (which never reads a value). Vacuously
                // true for a dead/graph-output tensor (nothing reads its value either).
                bool allConsumersFeatureOnly = cons == null || cons.All(c => ReadbackFeatureOnlyProducers.Contains(c));
                if (featureProducer || allConsumersFeatureOnly)
                    skip.Add(o);
            }
        }
        return skip;
    }

    /// <summary>
    /// Walk the compiled graph to identify every tensor whose ONNX-declared dtype
    /// is integer (INT8/16/32/64, UINT8/16/32/64, BOOL). Seeds from
    /// <see cref="CompiledGraph.InitializerDataTypes"/>, then sweeps node outputs:
    ///   * Cast: output dtype = `to` attribute
    ///   * ArgMax / ArgMin / Shape / Size / NonZero: outputs are always integer
    ///   * TopK: second output (indices) is integer
    ///   * Dtype-preserving ops (Reshape, Squeeze, Unsqueeze, Transpose, Concat,
    ///     Slice, Gather, GatherND, GatherElements, Identity, Tile, Expand, Pad,
    ///     Flatten, Where, Compress, ScatterND, ScatterElements): output dtype =
    ///     input[0] dtype
    ///   * Binary arithmetic (Add, Sub, Mul, Div, Pow, Mod, Min, Max, BitwiseAnd/Or/Xor):
    ///     output is integer iff all numeric inputs are integer
    /// Iterates to fixed point so propagation closes through chains. The result
    /// drives <see cref="OnnxOpContext.AllInputsAreInteger"/> at execute time,
    /// which DivOperator uses to apply ONNX-spec truncation toward zero.
    /// </summary>
    private static HashSet<string> BuildIntegerTensorNames(CompiledGraph graph)
    {
        var intNames = new HashSet<string>();

        static bool IsIntDataType(int dt) => dt switch
        {
            // OnnxDataType codes
            2 or 3 or 4 or 5 or 6 or 7 or 9 or 12 or 13 => true, // UINT8/INT8/UINT16/INT16/INT32/INT64/BOOL/UINT32/UINT64
            _ => false,
        };

        // Seed from initializer / Constant-node dtypes
        if (graph.InitializerDataTypes != null)
        {
            foreach (var (name, dt) in graph.InitializerDataTypes)
            {
                if (IsIntDataType(dt))
                    intNames.Add(name);
            }
        }

        // Sweep nodes to fixed point. Most propagation closes in a single pass
        // because topological order is preserved in CompiledGraph.Nodes; the
        // outer loop guards against pathological graphs where Cast/ArgMax output
        // feeds back into a propagator earlier in the array.
        bool changed = true;
        int guardIterations = 0;
        while (changed && guardIterations < 8)
        {
            changed = false;
            guardIterations++;

            foreach (var node in graph.Nodes)
            {
                int beforeCount = intNames.Count;

                switch (node.OpType)
                {
                    case "Cast":
                    {
                        // 'to' attribute is the target ONNX dtype code (long from JSON parse).
                        long to = 0;
                        if (node.Attributes.TryGetValue("to", out var toObj))
                        {
                            try { to = Convert.ToInt64(toObj); } catch { to = 0; }
                        }
                        bool toIsInt = IsIntDataType((int)to);
                        foreach (var outName in node.OutputNames)
                        {
                            if (string.IsNullOrEmpty(outName)) continue;
                            if (toIsInt) intNames.Add(outName);
                            // else: explicitly NOT integer — do not add (Cast to float kills int chain)
                        }
                        break;
                    }
                    case "ArgMax":
                    case "ArgMin":
                    case "Shape":
                    case "Size":
                    case "NonZero":
                        foreach (var outName in node.OutputNames)
                            if (!string.IsNullOrEmpty(outName))
                                intNames.Add(outName);
                        break;
                    case "TopK":
                        // Output 0 is values (same dtype as input[0]); output 1 is indices (int).
                        if (node.OutputNames.Length > 1 && !string.IsNullOrEmpty(node.OutputNames[1]))
                            intNames.Add(node.OutputNames[1]);
                        if (node.InputNames.Length > 0 && intNames.Contains(node.InputNames[0])
                            && node.OutputNames.Length > 0 && !string.IsNullOrEmpty(node.OutputNames[0]))
                            intNames.Add(node.OutputNames[0]);
                        break;
                    case "Equal":
                    case "Greater":
                    case "GreaterOrEqual":
                    case "Less":
                    case "LessOrEqual":
                    case "Not":
                    case "And":
                    case "Or":
                    case "Xor":
                    case "IsInf":
                    case "IsNaN":
                        // Boolean-producing ops
                        foreach (var outName in node.OutputNames)
                            if (!string.IsNullOrEmpty(outName))
                                intNames.Add(outName);
                        break;
                    case "Reshape":
                    case "Squeeze":
                    case "Unsqueeze":
                    case "Transpose":
                    case "Identity":
                    case "Tile":
                    case "Expand":
                    case "Pad":
                    case "Flatten":
                    case "Slice":
                    case "Gather":
                    case "GatherND":
                    case "GatherElements":
                    case "Compress":
                    case "ScatterND":
                    case "ScatterElements":
                    case "DepthToSpace":
                    case "SpaceToDepth":
                    case "ReverseSequence":
                    case "Concat":
                    case "Split":
                    case "Where":
                    {
                        // Output dtype = data-input dtype. For Where the data inputs are 1,2.
                        int dataInputIdx = node.OpType == "Where" ? 1 : 0;
                        if (node.OpType == "Concat" || node.OpType == "Split")
                            dataInputIdx = 0; // all inputs share dtype for Concat; output 0..N for Split
                        if (node.InputNames.Length > dataInputIdx
                            && !string.IsNullOrEmpty(node.InputNames[dataInputIdx])
                            && intNames.Contains(node.InputNames[dataInputIdx]))
                        {
                            foreach (var outName in node.OutputNames)
                                if (!string.IsNullOrEmpty(outName))
                                    intNames.Add(outName);
                        }
                        break;
                    }
                    case "Add":
                    case "Sub":
                    case "Mul":
                    case "Div":
                    case "Mod":
                    case "Pow":
                    case "Min":
                    case "Max":
                    case "BitwiseAnd":
                    case "BitwiseOr":
                    case "BitwiseXor":
                    case "BitShift":
                    {
                        // Output is integer iff all numeric inputs (skip optional empty names)
                        // are integer-typed.
                        bool allInt = true;
                        bool sawAny = false;
                        for (int i = 0; i < node.InputNames.Length; i++)
                        {
                            var nm = node.InputNames[i];
                            if (string.IsNullOrEmpty(nm)) continue;
                            sawAny = true;
                            if (!intNames.Contains(nm)) { allInt = false; break; }
                        }
                        if (sawAny && allInt)
                        {
                            foreach (var outName in node.OutputNames)
                                if (!string.IsNullOrEmpty(outName))
                                    intNames.Add(outName);
                        }
                        break;
                    }
                    case "Neg":
                    case "Abs":
                    case "Sign":
                    case "BitwiseNot":
                    {
                        // Unary dtype-preserving
                        if (node.InputNames.Length > 0
                            && !string.IsNullOrEmpty(node.InputNames[0])
                            && intNames.Contains(node.InputNames[0]))
                        {
                            foreach (var outName in node.OutputNames)
                                if (!string.IsNullOrEmpty(outName))
                                    intNames.Add(outName);
                        }
                        break;
                    }
                    // All other ops (Conv, MatMul, Gemm, BatchNorm, LayerNorm, Softmax,
                    // Sigmoid, Tanh, Relu, Gelu, Exp, Log, Sin, Cos, Sqrt, ...) produce
                    // float output regardless of input dtype - no entry here means they
                    // do not add to intNames, which is correct.
                }

                if (intNames.Count != beforeCount) changed = true;
            }
        }

        return intNames;
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
