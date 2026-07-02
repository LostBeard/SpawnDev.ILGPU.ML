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
    /// CUDA-GRAPH CAPTURE: when true, <see cref="RunAsync"/> skips ALL periodic + final
    /// <c>SynchronizeAsync</c> drains. A synchronize is illegal during
    /// <c>cuStreamBeginCapture</c> (it aborts the capture), so the decode-graph capture path
    /// sets this for the single forward it records into a <see cref="ILGPU.Runtime.Cuda.CudaGraph"/>.
    /// The forward must already be WARM (pools + readback cache primed) so the drain-free pass
    /// allocates nothing and reads nothing back. Always reset to false immediately after EndCapture.
    /// </summary>
    public static bool SuppressDrains;

    /// <summary>
    /// CUDA-GRAPH CAPTURE: bumped once at the start of every <see cref="RunAsync"/>. Per-node operators
    /// that hand out stable device "slots" for capture (e.g. <c>FusedAttentionKernel</c>'s params buffer)
    /// reset their per-forward slot counter when this value changes, so the k-th call of a given op gets
    /// the SAME device pointer every forward — the requirement for a captured node to read a stable buffer
    /// the host refreshes between replays. Harmless (a monotonic counter) when capture is not in use.
    /// </summary>
    public static long ForwardGeneration;

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

            // Runtime Slice: resolve output shape from compiler-resolved attrs (backend/elide-independent),
            // falling back to runtimeConstants. See the async-path copy for the root cause (Seven's Slice_4).
            if (node.OpType == "Slice" && node.InputNames.Length >= 3)
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                float[]? starts = ResolvedShapeAttr(node, "_resolved_starts") ?? (node.InputNames.Length > 1 ? (runtimeConstants.GetValueOrDefault(node.InputNames[1])) : null);
                float[]? ends = ResolvedShapeAttr(node, "_resolved_ends") ?? (node.InputNames.Length > 2 ? (runtimeConstants.GetValueOrDefault(node.InputNames[2])) : null);
                float[]? axes = ResolvedShapeAttr(node, "_resolved_axes") ?? (node.InputNames.Length > 3 && !string.IsNullOrEmpty(node.InputNames[3]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[3])) : null);
                float[]? steps = ResolvedShapeAttr(node, "_resolved_steps") ?? (node.InputNames.Length > 4 && !string.IsNullOrEmpty(node.InputNames[4]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[4])) : null);
                if (starts != null && ends != null)
                {
                    var resolved = inShape.ToArray();
                    for (int si = 0; si < starts.Length; si++)
                    {
                        int ax = axes != null && si < axes.Length ? (int)axes[si] : si;
                        if (ax < 0) ax += resolved.Length;
                        if (ax >= 0 && ax < resolved.Length)
                        {
                            int s = SatFloatToInt(starts[si]); if (s < 0) s += resolved[ax]; if (s > resolved[ax]) s = resolved[ax]; if (s < 0) s = 0;
                            int e = SatFloatToInt(ends[si]); if (e < 0) e += resolved[ax]; if (e > resolved[ax]) e = resolved[ax];
                            int st = steps != null && si < steps.Length ? SatFloatToInt(steps[si]) : 1;
                            // Empty slices ARE valid (e<=s → 0): DAv3's extrinsics builds an EMPTY [.,.,0,4] row
                            // (Slice [3:3] on a size-3 axis, ORT value_info [?,?,0,4]) that a later Concat treats
                            // as a no-op. Rejecting the 0-dim here collapsed the whole shape to the compile-time
                            // rank-2 [1,4] → the extrinsics Concat then crashed on a rank/inner mismatch.
                            resolved[ax] = Math.Max(0, (e - s + st - 1) / st);
                        }
                    }
                    if (resolved.All(d => d >= 0))
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

            // Runtime Squeeze: output = input shape with the listed size-1 axes removed (or ALL size-1 dims if
            // no axes). Compile-time collapses it — DAv3's DEPTH head does Reshape(target = Shape(Squeeze(...))),
            // and a collapsed Squeeze shape made the reshape target's W dim read 0 → depth came out [50176,1,1,1]
            // instead of [1,1,224,224] (VALUES were correct, only the final shape was wrong).
            if (node.OpType == "Squeeze" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var sqIn = nodeInputs[0]!.Shape;
                int[]? sqAxes = null;
                if (node.Attributes.TryGetValue("axes", out var sqAxObj) && sqAxObj is long[] sqal)
                    sqAxes = sqal.Select(x => (int)x).ToArray();
                else if (node.InputNames.Length >= 2 && !string.IsNullOrEmpty(node.InputNames[1])
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var sqAxV))
                    sqAxes = sqAxV.Select(x => (int)Math.Round(x)).ToArray();
                var sqOut = new List<int>();
                if (sqAxes != null && sqAxes.Length > 0)
                {
                    var sqSet = sqAxes.Select(a => a < 0 ? a + sqIn.Length : a).ToHashSet();
                    for (int d = 0; d < sqIn.Length; d++) if (!sqSet.Contains(d)) sqOut.Add(sqIn[d]);
                }
                else
                    for (int d = 0; d < sqIn.Length; d++) if (sqIn[d] != 1) sqOut.Add(sqIn[d]);
                if (sqOut.Count == 0) sqOut.Add(1);
                if (sqOut.All(x => x > 0)) runtimeOutputShapes = new[] { sqOut.ToArray() };
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

            // Runtime Concat: output shape = input0's shape with the concat axis replaced by the SUM of
            // all inputs' axis dims. Build-time inference leaves the axis dim unresolved when an upstream
            // produces a dynamic dim — DAv3's 2D-RoPE position grid (/backbone/Concat_12: [grid,1]+[grid,1]
            // axis=1 -> [grid,2]) collapses the output buffer to [1], and the fused Concat kernel then
            // overflows it (the "Concat fused launch extent exceeds output buffer" abort). Resolve from the
            // ACTUAL runtime input shapes (all resolved by now). Only the clean same-rank case — a rank
            // mismatch falls back to ConcatOperator's flat-concat path, so leave those on compiled shapes.
            if (node.OpType == "Concat" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var cBase = nodeInputs[0]!.Shape;
                int cAxis = 0;
                if (node.Attributes.TryGetValue("axis", out var cAxObj)) cAxis = Convert.ToInt32(cAxObj);
                if (cAxis < 0) cAxis += cBase.Length;
                bool cOk = cAxis >= 0 && cAxis < cBase.Length;
                if (cOk)
                    foreach (var t in nodeInputs)
                        if (t == null || t.Shape.Length != cBase.Length || cAxis >= t.Shape.Length) { cOk = false; break; }
                if (cOk)
                {
                    int cSum = 0;
                    foreach (var t in nodeInputs) cSum += t!.Shape[cAxis];
                    var cResolved = (int[])cBase.Clone();
                    cResolved[cAxis] = cSum;
                    if (cResolved.All(d => d > 0)) runtimeOutputShapes = new[] { cResolved };
                }
            }

            // Runtime Gather: output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:], computed
            // from the ACTUAL runtime input shapes. Compile-time inference bakes an upstream multi-view dynamic
            // dim into the output — DAv3's 2D-RoPE `Gather(Reshape_26=[1,257,2], axis=2, scalar)` was baked
            // [3,257] (the channel-3 leak) instead of [1,257]; that 3x-bloats the RoPE cos/sin and corrupts
            // attention from block 4 (~5%/block → ~11% depth error). Mirrors GatherOperator.InferOutputShapes.
            if (node.OpType == "Gather" && nodeInputs.Length >= 2 && nodeInputs[0] != null && nodeInputs[1] != null)
            {
                var gData = nodeInputs[0]!.Shape;
                var gIdx = nodeInputs[1]!.Shape;
                int gAxis = 0;
                if (node.Attributes.TryGetValue("axis", out var gAxObj)) gAxis = Convert.ToInt32(gAxObj);
                if (gAxis < 0) gAxis += gData.Length;
                if (gAxis >= 0 && gAxis < gData.Length)
                {
                    // A [1] index on multi-dim data collapses to a scalar (drops the axis) — matches the operator.
                    var gEff = (gData.Length > 1 && gIdx.Length == 1 && gIdx[0] == 1) ? Array.Empty<int>() : gIdx;
                    var gOut = new List<int>();
                    for (int i = 0; i < gAxis; i++) gOut.Add(gData[i]);
                    gOut.AddRange(gEff);
                    for (int i = gAxis + 1; i < gData.Length; i++) gOut.Add(gData[i]);
                    if (gOut.Count > 0 && gOut.All(d => d > 0)) runtimeOutputShapes = new[] { gOut.ToArray() };
                }
            }

            // Runtime Einsum: resolve output shape from the equation + ACTUAL runtime input dims. Compile-time
            // collapses a dynamic outer-product/contraction to [1] — DAv3's 2D-RoPE freqs
            // Einsum("i,j->ij", pos[17], invfreq[16]) should be [17,16]=272 but was baked [1], which collapses
            // the whole cos/sin chain and ZEROES the rotation (RoPE'd q loses magnitude → flat attention).
            if (node.OpType == "Einsum" && nodeInputs.Length > 0
                && node.Attributes.TryGetValue("equation", out var eqObj))
            {
                var eq = eqObj?.ToString()?.Replace(" ", "");
                if (!string.IsNullOrEmpty(eq) && eq.Contains("->"))
                {
                    var eqParts = eq.Split("->");
                    var inSpecs = eqParts[0].Split(',');
                    var outSpec = eqParts[1];
                    var dimOf = new Dictionary<char, int>();
                    bool eOk = inSpecs.Length == nodeInputs.Length && !outSpec.Contains('.');
                    for (int ii = 0; eOk && ii < inSpecs.Length; ii++)
                    {
                        var spec = inSpecs[ii]; var sh = nodeInputs[ii]?.Shape;
                        if (sh == null || spec.Contains('.') || spec.Length != sh.Length) { eOk = false; break; }
                        for (int d = 0; d < spec.Length; d++) dimOf[spec[d]] = sh[d];
                    }
                    if (eOk)
                    {
                        var eOut = new int[outSpec.Length];
                        for (int d = 0; d < outSpec.Length; d++)
                            eOut[d] = dimOf.TryGetValue(outSpec[d], out var dv) ? dv : 1;
                        if (eOut.Length > 0 && eOut.All(x => x > 0)) runtimeOutputShapes = new[] { eOut };
                    }
                }
            }

            // Runtime Transpose: output shape = input shape permuted by perm, from the ACTUAL runtime input.
            // Compile-time collapses dynamic dims — DAv3's RoPE position transpose input [16,16,48]=12288 got
            // output [48,1,2]=96, and the transpose kernel (launch extent = product(input)=12288) then WRITES
            // 12288 floats into the 96-float output buffer → a massive OOB write that corrupts a live buffer a
            // few blocks later (illegal memory access surfacing at block 9). Permute the runtime input shape.
            if (node.OpType == "Transpose" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var tIn = nodeInputs[0]!.Shape;
                int[] tPerm;
                if (node.Attributes.TryGetValue("perm", out var tPermObj) && tPermObj is long[] tpl)
                    tPerm = tpl.Select(x => (int)x).ToArray();
                else { tPerm = new int[tIn.Length]; for (int i = 0; i < tIn.Length; i++) tPerm[i] = tIn.Length - 1 - i; }
                if (tPerm.Length == tIn.Length && tPerm.All(pp => pp >= 0 && pp < tIn.Length))
                {
                    var tOut = new int[tIn.Length];
                    for (int i = 0; i < tIn.Length; i++) tOut[i] = tIn[tPerm[i]];
                    if (tOut.All(d => d > 0)) runtimeOutputShapes = new[] { tOut };
                }
            }

            // Runtime MatMul: batched matmul output = broadcast(a.batchDims, b.batchDims) + [M, N].
            // Compile-time inference can size the batch from the wrong operand — DAv3's multi-view attention
            // (a=[3,6,6,64] @ b=[1,6,64,6] should broadcast batch to [3,6] → [3,6,6,6]=648, but compile-time
            // used b's [1,6] → [1,6,6,6]=216 and the kernel overflows). Resolve from the ACTUAL runtime ranks.
            if (node.OpType == "MatMul" && nodeInputs.Length >= 2 && nodeInputs[0] != null && nodeInputs[1] != null)
            {
                var mA = nodeInputs[0]!.Shape;
                var mB = nodeInputs[1]!.Shape;
                if (mA.Length >= 2 && mB.Length >= 2)
                {
                    int mM = mA[mA.Length - 2];
                    int mN = mB[mB.Length - 1];
                    int aBatch = mA.Length - 2, bBatch = mB.Length - 2;
                    int batchRank = Math.Max(aBatch, bBatch);
                    var mOut = new int[batchRank + 2];
                    for (int d = 0; d < batchRank; d++)
                    {
                        int ad = d - (batchRank - aBatch); int av = ad >= 0 ? mA[ad] : 1;
                        int bd = d - (batchRank - bBatch); int bv = bd >= 0 ? mB[bd] : 1;
                        mOut[d] = Math.Max(av, bv); // ONNX numpy-style batch broadcast
                    }
                    mOut[batchRank] = mM;
                    mOut[batchRank + 1] = mN;
                    if (mOut.All(d => d > 0)) runtimeOutputShapes = new[] { mOut };
                }
            }

            // Runtime shape-preserving ops: a single-output op whose output has input[0]'s exact shape
            // (softmax + unary activations). When an upstream dynamic dim (DAv3's multi-view batch) leaves
            // the compiled output buffer sized for the wrong batch, resolve from the actual runtime input.
            if (nodeInputs.Length > 0 && nodeInputs[0] != null && (
                node.OpType is "Softmax" or "LogSoftmax" or "Relu" or "Gelu" or "Sigmoid" or "Tanh"
                or "Erf" or "Exp" or "Sqrt" or "Neg" or "Reciprocal" or "Softplus" or "Elu" or "LeakyRelu"
                or "Abs" or "Sin" or "Cos" or "Clip" or "Mish"))
                runtimeOutputShapes = new[] { (int[])nodeInputs[0]!.Shape.Clone() };

            // Runtime Reduce (ReduceMax/Min/Mean/Sum/...): output = input shape with the reduced axes removed
            // (keepdims=0) or set to 1 (keepdims=1). opset-18 passes axes as input[1]; when axes are absent
            // the ONNX default is REDUCE ALL (unless noop_with_empty_axes=1) — NOT last-dim. Compile-time
            // inference can't read the runtime axes input, so resolve here. DAv3 RoPE: ReduceMax over a
            // 3-D dynamic tensor must collapse to a scalar; the attr-only default left [3,257].
            if ((node.OpType is "ReduceMax" or "ReduceMin" or "ReduceMean" or "ReduceSum" or "ReduceProd"
                 or "ReduceL1" or "ReduceL2" or "ReduceSumSquare" or "ReduceLogSum" or "ReduceLogSumExp")
                && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var rIn = nodeInputs[0]!.Shape; int rRank = rIn.Length;
                int[] rAx;
                float[]? rAxV = node.InputNames.Length > 1 && !string.IsNullOrEmpty(node.InputNames[1])
                    ? runtimeConstants.GetValueOrDefault(node.InputNames[1]) : null;
                if (rAxV != null && rAxV.Length > 0)
                    rAx = rAxV.Select(a => (int)MathF.Round(a)).Select(a => a < 0 ? a + rRank : a).ToArray();
                else if (node.Attributes.TryGetValue("axes", out var rAxObj) && rAxObj is long[] rAl && rAl.Length > 0)
                    rAx = rAl.Select(a => (int)(a < 0 ? a + rRank : a)).ToArray();
                else
                {
                    bool rNoop = node.Attributes.TryGetValue("noop_with_empty_axes", out var rNop) && Convert.ToInt32(rNop) != 0;
                    rAx = rNoop ? Array.Empty<int>() : Enumerable.Range(0, rRank).ToArray(); // ONNX default: reduce ALL
                }
                bool rKeep = !node.Attributes.TryGetValue("keepdims", out var rKd) || Convert.ToInt32(rKd) != 0;
                var rOut = new List<int>();
                for (int i = 0; i < rRank; i++)
                {
                    if (Array.IndexOf(rAx, i) >= 0) { if (rKeep) rOut.Add(1); }
                    else rOut.Add(rIn[i]);
                }
                runtimeOutputShapes = new[] { rOut.Count > 0 ? rOut.ToArray() : new[] { 1 } }; // empty => scalar (1 elem)
            }

            // Runtime broadcast re-inference for elementwise/select ops poisoned by an upstream
            // value-dependent placeholder — re-infer the output from the ACTUAL runtime input shapes.
            if ((node.OpType == "Where" || node.OpType == "Cast" || node.OpType == "Add" || node.OpType == "Sub"
                 || node.OpType == "Mul" || node.OpType == "Div" || node.OpType == "Equal" || node.OpType == "Less"
                 || node.OpType == "Greater" || node.OpType == "And" || node.OpType == "Or" || node.OpType == "Not"
                 || node.OpType == "Min" || node.OpType == "Max" || node.OpType == "Pow") && nodeInputs.Length > 0)
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
    /// <summary>DIAGNOSTIC: shape-interp values that disagreed with the GPU readback on the last run (validation
    /// mode). Must be 0 - any mismatch is a bug in the CPU shape eval.</summary>
    public static int LastRunShapeInterpMismatches;
    /// <summary>DIAGNOSTIC: number of shape-op outputs the CPU interpreter resolved on the last run.</summary>
    public static int LastRunShapeInterpResolved;
    /// <summary>Dispatch-elide (all-backend): when true, shape ops the CPU interpreter resolves are NOT dispatched
    /// to the GPU at all (not just their readback skipped) — they run entirely on the CPU like ORT. This removes
    /// ~1200 nodes of per-node dispatch/sync/alloc orchestration (the ~1200ms CUDA residual Seven measured) on
    /// EVERY backend. GPU-tensor consumers of an elided output get it materialized on-demand from the CPU value.
    /// Requires ShapeSubgraphFoldEnabled. Default off until validated bit-exact on the DAv3 rig.</summary>
    public static bool ShapeInterpElideDispatch;
    /// <summary>DIAGNOSTIC: when non-null, the executor records "OpType in=[shapes]" for each FLOP-carrying node
    /// (MatMul/Conv/Einsum/FusedLinear/FusedAttention/Gemm/ConvTranspose) so we can histogram the actual kernel
    /// shapes (M,K,N / C,H,W,k) — the input Seven needs to pick per-shape tile configs for the SGEMM core.</summary>
    public static System.Collections.Generic.List<string>? CaptureKernelShapes;
    /// <summary>When true, the CPU shape interpreter STILL does the GPU readback and compares (populating
    /// <see cref="LastRunShapeInterpMismatches"/>) instead of skipping it - used to prove the eval is correct
    /// before trusting it. Default false (skip the readback = the actual perf win).</summary>
    public static bool ShapeInterpValidate;

    /// <summary>
    /// Runtime CPU shape interpreter. Computes a shape-subgraph op on the CPU from the ACTUAL runtime tensor
    /// shapes (metadata the CPU already has) + already-resolved shape values - so its tiny integer result never
    /// has to be read back from the GPU (the WebGPU per-op mapAsync round-trip that dominates DAv3). Returns true
    /// + the fp32 value array (runtimeConstants' representation) for a supported single-output shape op whose
    /// inputs are all CPU-available; false otherwise (the node then runs + reads back as usual). The heavy tensor
    /// math is untouched - only the shape bookkeeping moves to the CPU, where ORT does it too.
    /// </summary>
    /// <summary>
    /// Rank-CHANGING shape ops: their CPU value (a flat integer list in runtimeConstants) corresponds to a
    /// tensor whose true rank is NOT 1 — Unsqueeze/Squeeze add/drop axes, Reshape/Expand/ConstantOfShape build
    /// a multi-dim tensor. Dispatch-elide must NOT skip these: an elided output consumed as a GPU tensor is
    /// materialized on-demand as rank-1 [len] (see the input-gather materialization), which is only correct when
    /// the true output IS rank-1. A rank-1 materialization of a genuinely multi-dim tensor collapses a downstream
    /// rank-matched op (the DAv3 2D-RoPE Concat_12: [grid,1]+[grid,1] → [grid,2]). All the OTHER interpreter ops
    /// (Shape/Gather/Concat-axis0/Slice-axis0/Cast/Identity/arithmetic/compare) produce a rank-1 vector, so their
    /// rank-1 materialization is exact and they elide freely.
    /// </summary>
    private static bool IsRankChangingShapeOp(string opType) => opType is
        "Unsqueeze" or "Squeeze" or "Reshape" or "Expand" or "ConstantOfShape";

    // Cached per-graph set of shape-op outputs that dispatch-elide must NOT skip (path-c consumer gate).
    private HashSet<string>? _elideBlockedOutputs;

    /// <summary>
    /// Path-c dispatch-elide safety: an interpreter-resolved shape op may skip its GPU dispatch ONLY if its
    /// output is never read as a GPU tensor - every consumer must read it as a CPU shape-param (from
    /// runtimeConstants) OR be itself a pure CPU-resolvable shape op (reads all inputs as CPU values). If ANY
    /// consumer reads it as a real tensor (on-demand materialization, or a runtime shape-resolver reading
    /// nodeInputs[i].Shape), eliding it corrupts that consumer (the DAv3 Concat_19 / FusedAttention class).
    /// Blocking such an output just makes it dispatch (the proven elide-off path) - provably non-regressive.
    /// Computed once from the fixed graph + cached.
    /// </summary>
    private HashSet<string> ElideBlockedOutputs()
    {
        if (_elideBlockedOutputs != null) return _elideBlockedOutputs;

        // 1) shapeValue: the pure CPU shape subgraph (least-fixpoint). A name is a shape value if it's a
        //    compile-time constant, or the single output of an interpreter-resolvable op whose EVERY value
        //    input is itself a shape value. (Shape reads tensor metadata, so it's a shape value unconditionally.)
        var shapeValue = new HashSet<string>();
        if (_constantValues != null) foreach (var k in _constantValues.Keys) shapeValue.Add(k);
        foreach (var n in _graph.Nodes)
            if (n.OpType == "Constant")
                foreach (var o in n.OutputNames) if (!string.IsNullOrEmpty(o)) shapeValue.Add(o);

        bool changed = true;
        while (changed)
        {
            changed = false;
            foreach (var n in _graph.Nodes)
            {
                if (n.OutputNames.Length != 1) continue;
                var outName = n.OutputNames[0];
                if (string.IsNullOrEmpty(outName) || shapeValue.Contains(outName)) continue;
                if (!IsInterpreterResolvableOp(n)) continue;
                bool allVals = n.OpType == "Shape";   // Shape needs no value inputs
                if (!allVals)
                {
                    allVals = true;
                    foreach (var vi in n.InputNames)
                        if (!string.IsNullOrEmpty(vi) && !shapeValue.Contains(vi)) { allVals = false; break; }
                }
                if (allVals) { shapeValue.Add(outName); changed = true; }
            }
        }

        // 2) blocked: a graph output, OR consumed by some node in a NON-shape-param slot where that consumer is
        //    NOT itself a pure shape value (so it reads the input as a real GPU tensor).
        var blocked = new HashSet<string>(_graph.OutputNames.Where(o => !string.IsNullOrEmpty(o)));
        foreach (var c in _graph.Nodes)
        {
            // A consumer "reads all inputs via Vals" (tensor-consumes nothing) ONLY if it actually resolves on the
            // CPU AND is elided. A rank-CHANGING op is interpreter-resolvable (a shape value) BUT is excluded from
            // dispatch-elide (IsRankChangingShapeOp) - so it DISPATCHES and reads its data input as a real GPU
            // tensor (ONNX Reshape keep-dim reads nodeInputs[0].Shape). Treating it as consumerIsShape left its
            // data producer un-blocked -> elided -> materialized wrong -> the channel-3 leak (Reshape_26 dim0=3
            // -> block-4 RoPE Gather -> q/k rank-5). So a rank-changing consumer is a REAL tensor consumer.
            bool consumerIsShape = c.OutputNames.Length == 1 && !string.IsNullOrEmpty(c.OutputNames[0])
                && shapeValue.Contains(c.OutputNames[0]) && !IsRankChangingShapeOp(c.OpType);
            if (consumerIsShape) continue;   // resolves on the CPU AND elided -> reads every input via Vals
            for (int i = 0; i < c.InputNames.Length; i++)
            {
                var inName = c.InputNames[i];
                if (string.IsNullOrEmpty(inName)) continue;
                if (IsShapeParamSlot(c.OpType, i)) continue;   // read from runtimeConstants, never as a tensor
                blocked.Add(inName);
            }
        }
        _elideBlockedOutputs = blocked;
        return blocked;
    }

    // Ops the CPU interpreter (TryComputeShapeOnCpu) can resolve, with their statically-checkable op-conditions.
    private static bool IsInterpreterResolvableOp(CompiledNode n) => n.OpType switch
    {
        "Shape" => true,
        "Concat" or "Gather" => AttrAxisIsZero(n),   // interpreter only handles axis 0
        "Slice" or "Unsqueeze" or "Squeeze" or "Identity" or "Cast" or "Reshape"
            or "Mul" or "Add" or "Sub" or "Div" or "Where" or "Equal" or "Greater" or "Less"
            or "Floor" or "Ceil" or "Neg" or "Abs" or "Mod" or "Min" or "Max"
            or "ConstantOfShape" or "Expand" => true,
        _ => false,
    };

    private static bool AttrAxisIsZero(CompiledNode n)
    {
        if (n.Attributes != null && n.Attributes.TryGetValue("axis", out var a) && a != null)
        {
            try { return Convert.ToInt32(a) == 0; } catch { return false; }
        }
        return true;   // ONNX default axis is 0
    }

    // Input slots an op reads as a CPU shape-PARAM (from runtimeConstants) even when it dispatches on the GPU.
    // Conservative: anything NOT listed here is treated as a real-tensor slot (blocks elide of its producer).
    private static bool IsShapeParamSlot(string opType, int slot) => opType switch
    {
        "Reshape" or "Expand" or "Unsqueeze" or "Squeeze" or "Tile" or "Pad" => slot == 1,
        "ConstantOfShape" => slot == 0,
        "Slice" => slot >= 1 && slot <= 4,
        "Resize" => slot >= 1 && slot <= 3,
        "Range" => slot >= 0 && slot <= 2,
        _ => false,
    };

    // Compiler-resolved shape params (`_resolved_starts`/`_resolved_ends`/... on the node's attrs, stored by
    // GraphCompiler when it can resolve a Slice at compile time). Backend-independent, no GPU readback - the
    // reliable source for the runtime shape override, vs runtimeConstants which is NOT populated at cascade
    // time on the WebGPU async path OR when the producing shape op is dispatch-elided. Returns null if absent
    // (non-compiler-resolvable Slice) so the caller falls back to runtimeConstants.
    // Saturating float->int for slice params. The ONNX "to the end" sentinel is INT_MAX (2147483647), but a
    // float can't hold it exactly: (float)2147483647 rounds to 2147483648f, and a plain (int) cast of that
    // OVERFLOWS to INT_MIN - which then reads as a huge NEGATIVE start/end and collapses the slice to 0 (DAv3
    // blocks.4 rope [16:32] -> [.,.,.,0]). Saturate instead, exactly like SliceOperator path-2.
    private static int SatFloatToInt(float v) => v <= int.MinValue ? int.MinValue : v >= int.MaxValue ? int.MaxValue : (int)v;

    private static float[]? ResolvedShapeAttr(CompiledNode node, string key)
    {
        if (node.Attributes != null && node.Attributes.TryGetValue(key, out var o) && o != null)
        {
            switch (o)
            {
                case long[] la: { var r = new float[la.Length]; for (int i = 0; i < la.Length; i++) r[i] = la[i]; return r; }
                case int[] ia: { var r = new float[ia.Length]; for (int i = 0; i < ia.Length; i++) r[i] = ia[i]; return r; }
                case float[] fa: return fa;
            }
        }
        return null;
    }

    private bool TryComputeShapeOnCpu(CompiledNode node,
        Dictionary<string, Tensor> tensors,
        Dictionary<string, HalfTensor> halfTensors,
        Dictionary<string, float[]> runtimeConstants,
        out float[] result)
    {
        result = System.Array.Empty<float>();
        if (node.OutputNames.Length != 1 || string.IsNullOrEmpty(node.OutputNames[0])) return false;
        var ins = node.InputNames;

        float[]? Vals(string? name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            if (runtimeConstants.TryGetValue(name!, out var v)) return v;
            if (_constantValues != null && _constantValues.TryGetValue(name!, out var c)) return c;
            return null;
        }
        int[]? ShapeOf(string? name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            if (tensors.TryGetValue(name!, out var t)) return t.Shape;
            if (halfTensors.TryGetValue(name!, out var h)) return h.Shape;
            return null;
        }
        long AttrLong(string key, long dflt)
        {
            if (node.Attributes == null || !node.Attributes.TryGetValue(key, out var o) || o == null) return dflt;
            return o switch
            {
                long l => l,
                int i => i,
                long[] la when la.Length > 0 => la[0],
                int[] ia when ia.Length > 0 => ia[0],
                _ => dflt
            };
        }

        switch (node.OpType)
        {
            case "Shape":
            {
                var s = ShapeOf(ins.Length > 0 ? ins[0] : null);
                if (s == null) return false;
                int rank = s.Length;
                long start = AttrLong("start", 0), end = AttrLong("end", rank);
                if (start < 0) start += rank;
                if (end < 0) end += rank;
                start = System.Math.Clamp(start, 0, rank);
                end = System.Math.Clamp(end, 0, rank);
                if (end < start) end = start;
                var outv = new float[end - start];
                for (int i = 0; i < outv.Length; i++) outv[i] = s[start + i];
                result = outv; return true;
            }
            case "Gather":
            {
                var data = Vals(ins.Length > 0 ? ins[0] : null);
                var idx = Vals(ins.Length > 1 ? ins[1] : null);
                if (data == null || idx == null || AttrLong("axis", 0) != 0) return false;
                var outv = new float[idx.Length];
                for (int i = 0; i < idx.Length; i++)
                {
                    int ii = (int)idx[i]; if (ii < 0) ii += data.Length;
                    if (ii < 0 || ii >= data.Length) return false;
                    outv[i] = data[ii];
                }
                result = outv; return true;
            }
            case "Concat":
            {
                if (AttrLong("axis", 0) != 0) return false;
                var parts = new System.Collections.Generic.List<float>();
                foreach (var inp in ins) { var v = Vals(inp); if (v == null) return false; parts.AddRange(v); }
                result = parts.ToArray(); return true;
            }
            case "Unsqueeze":
            case "Squeeze":
            case "Identity":
            case "Cast":
            {
                // These change RANK/dtype but not the flat integer value list runtimeConstants stores.
                var v = Vals(ins.Length > 0 ? ins[0] : null);
                if (v == null) return false;
                result = v; return true;
            }
            case "Slice":
            {
                var data = Vals(ins.Length > 0 ? ins[0] : null);
                var starts = Vals(ins.Length > 1 ? ins[1] : null);
                var ends = Vals(ins.Length > 2 ? ins[2] : null);
                if (data == null || starts == null || ends == null || starts.Length < 1 || ends.Length < 1) return false;
                var axesV = Vals(ins.Length > 3 ? ins[3] : null);
                var stepsV = Vals(ins.Length > 4 ? ins[4] : null);
                if ((axesV != null && axesV.Length > 0 && (long)axesV[0] != 0)) return false;
                if (stepsV != null && stepsV.Length > 0 && (long)stepsV[0] != 1) return false;
                int st = (int)starts[0], en = (int)ends[0];
                if (st < 0) st += data.Length;
                if (en < 0) en += data.Length;
                st = System.Math.Clamp(st, 0, data.Length);
                en = System.Math.Clamp(en, 0, data.Length);
                if (en < st) en = st;
                var outv = new float[en - st];
                for (int i = 0; i < outv.Length; i++) outv[i] = data[st + i];
                result = outv; return true;
            }
            case "Mul": case "Add": case "Sub": case "Div":
            {
                var a = Vals(ins.Length > 0 ? ins[0] : null);
                var b = Vals(ins.Length > 1 ? ins[1] : null);
                if (a == null || b == null) return false;
                int len = System.Math.Max(a.Length, b.Length);
                if ((a.Length != len && a.Length != 1) || (b.Length != len && b.Length != 1)) return false;
                var outv = new float[len];
                for (int i = 0; i < len; i++)
                {
                    float av = a[a.Length == 1 ? 0 : i], bv = b[b.Length == 1 ? 0 : i];
                    outv[i] = node.OpType switch { "Mul" => av * bv, "Add" => av + bv, "Sub" => av - bv, "Div" => bv != 0 ? av / bv : 0, _ => av };
                }
                result = outv; return true;
            }
            case "Reshape":
            {
                // Reshaping a SHAPE VECTOR (in runtimeConstants) only changes rank, not the flat values. If the
                // input isn't a CPU shape value (it's GPU feature data), Vals returns null -> fall through to GPU.
                var v = Vals(ins.Length > 0 ? ins[0] : null);
                if (v == null) return false;
                result = v; return true;
            }
            case "Where":
            {
                var cond = Vals(ins.Length > 0 ? ins[0] : null);
                var x = Vals(ins.Length > 1 ? ins[1] : null);
                var y = Vals(ins.Length > 2 ? ins[2] : null);
                if (cond == null || x == null || y == null) return false;
                int len = System.Math.Max(cond.Length, System.Math.Max(x.Length, y.Length));
                if ((cond.Length != len && cond.Length != 1) || (x.Length != len && x.Length != 1) || (y.Length != len && y.Length != 1)) return false;
                var outv = new float[len];
                for (int i = 0; i < len; i++)
                    outv[i] = (cond[cond.Length == 1 ? 0 : i] != 0f) ? x[x.Length == 1 ? 0 : i] : y[y.Length == 1 ? 0 : i];
                result = outv; return true;
            }
            case "Equal": case "Greater": case "Less":
            {
                var a = Vals(ins.Length > 0 ? ins[0] : null);
                var b = Vals(ins.Length > 1 ? ins[1] : null);
                if (a == null || b == null) return false;
                int len = System.Math.Max(a.Length, b.Length);
                if ((a.Length != len && a.Length != 1) || (b.Length != len && b.Length != 1)) return false;
                var outv = new float[len];
                for (int i = 0; i < len; i++)
                {
                    float av = a[a.Length == 1 ? 0 : i], bv = b[b.Length == 1 ? 0 : i];
                    outv[i] = node.OpType switch { "Equal" => av == bv ? 1f : 0f, "Greater" => av > bv ? 1f : 0f, "Less" => av < bv ? 1f : 0f, _ => 0f };
                }
                result = outv; return true;
            }
            case "Floor": case "Ceil": case "Neg": case "Abs":
            {
                var v = Vals(ins.Length > 0 ? ins[0] : null);
                if (v == null) return false;
                var outv = new float[v.Length];
                for (int i = 0; i < v.Length; i++)
                    outv[i] = node.OpType switch { "Floor" => (float)System.Math.Floor(v[i]), "Ceil" => (float)System.Math.Ceiling(v[i]), "Neg" => -v[i], "Abs" => System.Math.Abs(v[i]), _ => v[i] };
                result = outv; return true;
            }
            case "Mod": case "Min": case "Max":
            {
                var a = Vals(ins.Length > 0 ? ins[0] : null);
                var b = Vals(ins.Length > 1 ? ins[1] : null);
                if (a == null || b == null) return false;
                int len = System.Math.Max(a.Length, b.Length);
                if ((a.Length != len && a.Length != 1) || (b.Length != len && b.Length != 1)) return false;
                var outv = new float[len];
                for (int i = 0; i < len; i++)
                {
                    float av = a[a.Length == 1 ? 0 : i], bv = b[b.Length == 1 ? 0 : i];
                    outv[i] = node.OpType switch { "Mod" => bv != 0 ? av - bv * (float)System.Math.Floor(av / bv) : 0, "Min" => System.Math.Min(av, bv), "Max" => System.Math.Max(av, bv), _ => av };
                }
                result = outv; return true;
            }
            case "ConstantOfShape":
            {
                var shp = Vals(ins.Length > 0 ? ins[0] : null);
                if (shp == null) return false;
                long len = 1; foreach (var d in shp) len *= (long)d;
                if (len < 0 || len > 4096) return false;
                // The fill value is a tensor-valued "value" attribute (default 0 per ONNX, but often 1). Parse it
                // across the representations the loader may produce; if we can't, DON'T guess - read it back.
                float val;
                object? vo = null;
                node.Attributes?.TryGetValue("value", out vo);
                switch (vo)
                {
                    case float[] vf when vf.Length > 0: val = vf[0]; break;
                    case int[] vi when vi.Length > 0: val = vi[0]; break;
                    case long[] vl when vl.Length > 0: val = vl[0]; break;
                    case float f: val = f; break;
                    case int ii: val = ii; break;
                    case long ll: val = ll; break;
                    case null: val = 0f; break; // no attribute → ONNX default 0
                    default: return false;      // unknown representation → fall through to GPU readback
                }
                var outv = new float[len];
                for (int i = 0; i < len; i++) outv[i] = val;
                result = outv; return true;
            }
            case "Expand":
            {
                var v = Vals(ins.Length > 0 ? ins[0] : null);
                var shp = Vals(ins.Length > 1 ? ins[1] : null);
                if (v == null || shp == null) return false;
                long len = 1; foreach (var d in shp) len *= (long)d;
                if (len < 0 || len > 4096) return false;
                if (v.Length == 1) { var outv = new float[len]; for (int i = 0; i < len; i++) outv[i] = v[0]; result = outv; return true; }
                if (v.Length == len) { result = v; return true; }
                return false;
            }
            // NOTE: Range/Pow/Reciprocal/Cos/Sin are NOT handled - they generate FLOAT data (RoPE/positional
            // frequencies), not integer shape vectors. They are float tensors, so the integer-tensor readback gate
            // skips their (unused) readback anyway; a CPU re-derivation would be float-rounding-unreliable.
            default: return false;
        }
    }

    public async Task<Dictionary<string, Tensor>> RunAsync(Dictionary<string, Tensor> inputs)
    {
        ForwardGeneration++;   // signals per-forward "stable capture slot" counters to reset (CUDA-graph capture)
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
        // Runtime CPU shape interpreter (gated on ShapeSubgraphFoldEnabled): shape-op outputs it resolves on the
        // CPU this run. Their value is put straight into runtimeConstants and their per-node GPU->CPU readback is
        // skipped - correct by construction because it reads the REAL runtime tensor shapes as tensors flow.
        var shapeInterpVals = new Dictionary<string, float[]>();
        // Gate the CPU shape interpreter to the browser GPU backends. On WebGPU/WebGL a per-op readback is a
        // ~345ms mapAsync round-trip, so computing shape ops on the CPU + skipping the readback is a massive win.
        // On CUDA/OpenCL/CPU a readback is a cheap synchronous memcpy, so the interpreter's CPU->GPU write cost is
        // a slight net loss there (and its CPU-backend path can fault) — so it stays off on native backends.
        // Interpreter runs on the browser GPU backends (readback-skip win) OR whenever dispatch-elide is on
        // (all-backend: eliding the node removes orchestration everywhere, not just the readback).
        bool shapeInterp = GraphCompiler.ShapeSubgraphFoldEnabled
            && (_accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL
                || ShapeInterpElideDispatch);
        LastRunShapeInterpMismatches = 0;
        LastRunShapeInterpResolved = 0;

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
                        // CUDA-graph capture: drains are suppressed, so the deferred-release path would never
                        // return buffers mid-forward → the whole working set stays live → pool miss → cuMemAlloc
                        // (ILLEGAL during capture, crashes). Instead return IMMEDIATELY: on the single capture
                        // stream, a later node's kernel that re-Rents this buffer is recorded after this input's
                        // last consumer, so stream ordering makes the reuse safe with no host drain. Keeps the
                        // captured forward's pool footprint bounded with ZERO allocation.
                        if (halfTensors.TryGetValue(inputName, out var hrel))
                        {
                            halfTensors.Remove(inputName);
                            if (SuppressDrains) _pool.ReturnHalf(hrel);
                            else { pendingHalfReleases.Add(hrel); pendingReleaseBytes += (long)hrel.ElementCount * 2; }
                        }
                        else if (tensors.TryGetValue(inputName, out var releaseTensor))
                        {
                            if (SuppressDrains) _pool.Return(releaseTensor);
                            else { pendingReleases.Add(releaseTensor); pendingReleaseBytes += (long)releaseTensor.ElementCount * sizeof(float); }
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
            // CUDA-graph capture records this forward; a synchronize would abort the capture. The
            // captured forward is warm, so skipping the drain leaks no buffers within the single pass.
            if (SuppressDrains) return;
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

            // Runtime CPU shape interpreter: resolve this shape op's value on the CPU from the REAL tensor shapes
            // now flowing through, and publish it into runtimeConstants so downstream shape params read it without
            // a GPU->CPU readback. The node STILL runs on the GPU below (so any tensor consumer is unaffected) -
            // we only eliminate the round-trip of its value. Correct because it mirrors the GPU op on real inputs.
            if (shapeInterp && TryComputeShapeOnCpu(node, tensors, halfTensors, runtimeConstants, out var cpuShapeVal))
            {
                runtimeConstants[node.OutputNames[0]] = cpuShapeVal;
                shapeInterpVals[node.OutputNames[0]] = cpuShapeVal;
                LastRunShapeInterpResolved++;
                // DISPATCH-ELIDE: a CPU-resolved shape op (value now in runtimeConstants) need not dispatch to the
                // GPU — that removes its per-node orchestration (dispatch/sync/alloc), the ~1200ms CUDA residual. A
                // GPU consumer of an elided output materializes it on-demand at input-gather below AS RANK-1 [len],
                // which is EXACT only when the true output is a rank-<=1 SHAPE VECTOR. So elide ONLY a genuine
                // shape vector: (a) not a rank-changing op (Unsqueeze/Squeeze/Reshape/Expand/ConstantOfShape), AND
                // (b) a SMALL value (<=64 — a real shape/dim list is tiny; its LENGTH is the described tensor's
                // RANK, <=8), AND (c) compile rank <=1. This rejects the interpreter OVER-REACHING on a feature
                // tensor it merely happened to be able to compute on the CPU — DAv3's /backbone/Add_2 is the
                // 2738-elem [1,1,1369,2] RoPE position grid; eliding it and materializing rank-1 collapsed its
                // rank-matched Concat_19 (axis=2) consumer to [3] -> buffer overflow. Non-elided ops still gained
                // the readback-skip (value already in runtimeConstants); only their GPU dispatch is retained.
                // Provably safe: a non-elided op falls back to the proven elide-off path (dispatch + real tensor).
                bool elideSafe = !IsRankChangingShapeOp(node.OpType)
                    && cpuShapeVal.Length <= 64
                    && node.OutputShapes.Length > 0 && node.OutputShapes[0].Length <= 1
                    && !ElideBlockedOutputs().Contains(node.OutputNames[0]);   // path-c: no GPU-tensor consumer
                if (ShapeInterpElideDispatch && elideSafe)
                {
                    ReleaseConsumedInputs(node);
                    nodeIdx++;
                    LastRunOpLog.Add($"{nodeIdx:D4} {node.OpType}~cpu-elided");
                    continue;
                }
            }

            // Kernel-shape histogram for Seven's SGEMM tile-config design: record the actual operand shapes of each
            // FLOP-carrying node. Input tensors are already in `tensors` (producers ran in topo order).
            if (CaptureKernelShapes != null && (node.OpType is "MatMul" or "Conv" or "ConvTranspose" or "Einsum"
                    or "FusedLinear" or "FusedAttention" or "Gemm" or "BatchedMatMul"))
            {
                var inShapes = string.Join(";", node.InputNames.Where(n => !string.IsNullOrEmpty(n))
                    .Select(n => tensors.TryGetValue(n, out var t) ? "[" + string.Join(",", t.Shape) + "]" : "?"));
                CaptureKernelShapes.Add($"{node.OpType} in={inShapes}");
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
                {
                    // A dispatch-elided shape op's output consumed here as a GPU tensor: materialize it on-demand
                    // from the CPU value the interpreter computed (the producing node was never dispatched). Rare —
                    // most elided outputs are consumed only as CPU shape params. Refcount for `name` is tracked
                    // normally, so this pooled tensor is released after its last consumer.
                    if (shapeInterp && runtimeConstants.TryGetValue(name, out var cval) && cval.Length > 0)
                    {
                        var mt = _pool.Rent(new[] { cval.Length }, name);
                        mt.Data.SubView(0, cval.Length).CopyFromCPU(cval);
                        tensors[name] = mt;
                        tensor = mt;
                    }
                    else
                        throw new InvalidOperationException($"Tensor '{name}' not found (needed by {node.OpType})");
                }
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

            // Runtime Slice (same as sync Run). Prefer compiler-resolved attrs over runtimeConstants (Seven's
            // root cause: on WebGPU async / under dispatch-elide the runtimeConstants values are not present at
            // cascade time, the override silently no-ops, and the compile-time OutputShapes garbage stands -
            // DAv3 blocks.4 rope Slice_4 got [1,6,1370,1] not [...,16] -> wrong RoPE -> 0.1616 vs 0.1365).
            if (node.OpType == "Slice" && node.InputNames.Length >= 3)
            {
                var inShape = nodeInputs[0]?.Shape ?? runtimeOutputShapes[0];
                float[]? starts = ResolvedShapeAttr(node, "_resolved_starts") ?? (node.InputNames.Length > 1 ? (runtimeConstants.GetValueOrDefault(node.InputNames[1])) : null);
                float[]? ends = ResolvedShapeAttr(node, "_resolved_ends") ?? (node.InputNames.Length > 2 ? (runtimeConstants.GetValueOrDefault(node.InputNames[2])) : null);
                float[]? axes = ResolvedShapeAttr(node, "_resolved_axes") ?? (node.InputNames.Length > 3 && !string.IsNullOrEmpty(node.InputNames[3]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[3])) : null);
                float[]? steps = ResolvedShapeAttr(node, "_resolved_steps") ?? (node.InputNames.Length > 4 && !string.IsNullOrEmpty(node.InputNames[4]) ? (runtimeConstants.GetValueOrDefault(node.InputNames[4])) : null);
                if (starts != null && ends != null)
                {
                    var resolved = inShape.ToArray();
                    for (int si = 0; si < starts.Length; si++)
                    {
                        int ax = axes != null && si < axes.Length ? (int)axes[si] : si;
                        if (ax < 0) ax += resolved.Length;
                        if (ax >= 0 && ax < resolved.Length)
                        {
                            int s = SatFloatToInt(starts[si]); if (s < 0) s += resolved[ax]; if (s > resolved[ax]) s = resolved[ax]; if (s < 0) s = 0;
                            int e = SatFloatToInt(ends[si]); if (e < 0) e += resolved[ax]; if (e > resolved[ax]) e = resolved[ax];
                            int st = steps != null && si < steps.Length ? SatFloatToInt(steps[si]) : 1;
                            // Empty slices ARE valid (e<=s → 0): DAv3's extrinsics builds an EMPTY [.,.,0,4] row
                            // (Slice [3:3] on a size-3 axis, ORT value_info [?,?,0,4]) that a later Concat treats
                            // as a no-op. Rejecting the 0-dim here collapsed the whole shape to the compile-time
                            // rank-2 [1,4] → the extrinsics Concat then crashed on a rank/inner mismatch.
                            resolved[ax] = Math.Max(0, (e - s + st - 1) / st);
                        }
                    }
                    if (resolved.All(d => d >= 0))
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

            // Runtime Squeeze: output = input shape with the listed size-1 axes removed (or ALL size-1 dims if
            // no axes). Compile-time collapses it — DAv3's DEPTH head does Reshape(target = Shape(Squeeze(...))),
            // and a collapsed Squeeze shape made the reshape target's W dim read 0 → depth came out [50176,1,1,1]
            // instead of [1,1,224,224] (VALUES were correct, only the final shape was wrong).
            if (node.OpType == "Squeeze" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var sqIn = nodeInputs[0]!.Shape;
                int[]? sqAxes = null;
                if (node.Attributes.TryGetValue("axes", out var sqAxObj) && sqAxObj is long[] sqal)
                    sqAxes = sqal.Select(x => (int)x).ToArray();
                else if (node.InputNames.Length >= 2 && !string.IsNullOrEmpty(node.InputNames[1])
                    && runtimeConstants.TryGetValue(node.InputNames[1], out var sqAxV))
                    sqAxes = sqAxV.Select(x => (int)Math.Round(x)).ToArray();
                var sqOut = new List<int>();
                if (sqAxes != null && sqAxes.Length > 0)
                {
                    var sqSet = sqAxes.Select(a => a < 0 ? a + sqIn.Length : a).ToHashSet();
                    for (int d = 0; d < sqIn.Length; d++) if (!sqSet.Contains(d)) sqOut.Add(sqIn[d]);
                }
                else
                    for (int d = 0; d < sqIn.Length; d++) if (sqIn[d] != 1) sqOut.Add(sqIn[d]);
                if (sqOut.Count == 0) sqOut.Add(1);
                if (sqOut.All(x => x > 0)) runtimeOutputShapes = new[] { sqOut.ToArray() };
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

            // Runtime Concat: output shape = input0's shape with the concat axis replaced by the SUM of
            // all inputs' axis dims. Build-time inference leaves the axis dim unresolved when an upstream
            // produces a dynamic dim — DAv3's 2D-RoPE position grid (/backbone/Concat_12: [grid,1]+[grid,1]
            // axis=1 -> [grid,2]) collapses the output buffer to [1], and the fused Concat kernel then
            // overflows it (the "Concat fused launch extent exceeds output buffer" abort). Resolve from the
            // ACTUAL runtime input shapes (all resolved by now). Only the clean same-rank case — a rank
            // mismatch falls back to ConcatOperator's flat-concat path, so leave those on compiled shapes.
            if (node.OpType == "Concat" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var cBase = nodeInputs[0]!.Shape;
                int cAxis = 0;
                if (node.Attributes.TryGetValue("axis", out var cAxObj)) cAxis = Convert.ToInt32(cAxObj);
                if (cAxis < 0) cAxis += cBase.Length;
                bool cOk = cAxis >= 0 && cAxis < cBase.Length;
                if (cOk)
                    foreach (var t in nodeInputs)
                        if (t == null || t.Shape.Length != cBase.Length || cAxis >= t.Shape.Length) { cOk = false; break; }
                if (cOk)
                {
                    int cSum = 0;
                    foreach (var t in nodeInputs) cSum += t!.Shape[cAxis];
                    var cResolved = (int[])cBase.Clone();
                    cResolved[cAxis] = cSum;
                    if (cResolved.All(d => d > 0)) runtimeOutputShapes = new[] { cResolved };
                }
            }

            // Runtime Gather: output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:], computed
            // from the ACTUAL runtime input shapes. Compile-time inference bakes an upstream multi-view dynamic
            // dim into the output — DAv3's 2D-RoPE `Gather(Reshape_26=[1,257,2], axis=2, scalar)` was baked
            // [3,257] (the channel-3 leak) instead of [1,257]; that 3x-bloats the RoPE cos/sin and corrupts
            // attention from block 4 (~5%/block → ~11% depth error). Mirrors GatherOperator.InferOutputShapes.
            if (node.OpType == "Gather" && nodeInputs.Length >= 2 && nodeInputs[0] != null && nodeInputs[1] != null)
            {
                var gData = nodeInputs[0]!.Shape;
                var gIdx = nodeInputs[1]!.Shape;
                int gAxis = 0;
                if (node.Attributes.TryGetValue("axis", out var gAxObj)) gAxis = Convert.ToInt32(gAxObj);
                if (gAxis < 0) gAxis += gData.Length;
                if (gAxis >= 0 && gAxis < gData.Length)
                {
                    // A [1] index on multi-dim data collapses to a scalar (drops the axis) — matches the operator.
                    var gEff = (gData.Length > 1 && gIdx.Length == 1 && gIdx[0] == 1) ? Array.Empty<int>() : gIdx;
                    var gOut = new List<int>();
                    for (int i = 0; i < gAxis; i++) gOut.Add(gData[i]);
                    gOut.AddRange(gEff);
                    for (int i = gAxis + 1; i < gData.Length; i++) gOut.Add(gData[i]);
                    if (gOut.Count > 0 && gOut.All(d => d > 0)) runtimeOutputShapes = new[] { gOut.ToArray() };
                }
            }

            // Runtime Einsum: resolve output shape from the equation + ACTUAL runtime input dims. Compile-time
            // collapses a dynamic outer-product/contraction to [1] — DAv3's 2D-RoPE freqs
            // Einsum("i,j->ij", pos[17], invfreq[16]) should be [17,16]=272 but was baked [1], which collapses
            // the whole cos/sin chain and ZEROES the rotation (RoPE'd q loses magnitude → flat attention).
            if (node.OpType == "Einsum" && nodeInputs.Length > 0
                && node.Attributes.TryGetValue("equation", out var eqObj))
            {
                var eq = eqObj?.ToString()?.Replace(" ", "");
                if (!string.IsNullOrEmpty(eq) && eq.Contains("->"))
                {
                    var eqParts = eq.Split("->");
                    var inSpecs = eqParts[0].Split(',');
                    var outSpec = eqParts[1];
                    var dimOf = new Dictionary<char, int>();
                    bool eOk = inSpecs.Length == nodeInputs.Length && !outSpec.Contains('.');
                    for (int ii = 0; eOk && ii < inSpecs.Length; ii++)
                    {
                        var spec = inSpecs[ii]; var sh = nodeInputs[ii]?.Shape;
                        if (sh == null || spec.Contains('.') || spec.Length != sh.Length) { eOk = false; break; }
                        for (int d = 0; d < spec.Length; d++) dimOf[spec[d]] = sh[d];
                    }
                    if (eOk)
                    {
                        var eOut = new int[outSpec.Length];
                        for (int d = 0; d < outSpec.Length; d++)
                            eOut[d] = dimOf.TryGetValue(outSpec[d], out var dv) ? dv : 1;
                        if (eOut.Length > 0 && eOut.All(x => x > 0)) runtimeOutputShapes = new[] { eOut };
                    }
                }
            }

            // Runtime Transpose: output shape = input shape permuted by perm, from the ACTUAL runtime input.
            // Compile-time collapses dynamic dims — DAv3's RoPE position transpose input [16,16,48]=12288 got
            // output [48,1,2]=96, and the transpose kernel (launch extent = product(input)=12288) then WRITES
            // 12288 floats into the 96-float output buffer → a massive OOB write that corrupts a live buffer a
            // few blocks later (illegal memory access surfacing at block 9). Permute the runtime input shape.
            if (node.OpType == "Transpose" && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var tIn = nodeInputs[0]!.Shape;
                int[] tPerm;
                if (node.Attributes.TryGetValue("perm", out var tPermObj) && tPermObj is long[] tpl)
                    tPerm = tpl.Select(x => (int)x).ToArray();
                else { tPerm = new int[tIn.Length]; for (int i = 0; i < tIn.Length; i++) tPerm[i] = tIn.Length - 1 - i; }
                if (tPerm.Length == tIn.Length && tPerm.All(pp => pp >= 0 && pp < tIn.Length))
                {
                    var tOut = new int[tIn.Length];
                    for (int i = 0; i < tIn.Length; i++) tOut[i] = tIn[tPerm[i]];
                    if (tOut.All(d => d > 0)) runtimeOutputShapes = new[] { tOut };
                }
            }

            // Runtime MatMul: batched matmul output = broadcast(a.batchDims, b.batchDims) + [M, N].
            // Compile-time inference can size the batch from the wrong operand — DAv3's multi-view attention
            // (a=[3,6,6,64] @ b=[1,6,64,6] should broadcast batch to [3,6] → [3,6,6,6]=648, but compile-time
            // used b's [1,6] → [1,6,6,6]=216 and the kernel overflows). Resolve from the ACTUAL runtime ranks.
            if (node.OpType == "MatMul" && nodeInputs.Length >= 2 && nodeInputs[0] != null && nodeInputs[1] != null)
            {
                var mA = nodeInputs[0]!.Shape;
                var mB = nodeInputs[1]!.Shape;
                if (mA.Length >= 2 && mB.Length >= 2)
                {
                    int mM = mA[mA.Length - 2];
                    int mN = mB[mB.Length - 1];
                    int aBatch = mA.Length - 2, bBatch = mB.Length - 2;
                    int batchRank = Math.Max(aBatch, bBatch);
                    var mOut = new int[batchRank + 2];
                    for (int d = 0; d < batchRank; d++)
                    {
                        int ad = d - (batchRank - aBatch); int av = ad >= 0 ? mA[ad] : 1;
                        int bd = d - (batchRank - bBatch); int bv = bd >= 0 ? mB[bd] : 1;
                        mOut[d] = Math.Max(av, bv); // ONNX numpy-style batch broadcast
                    }
                    mOut[batchRank] = mM;
                    mOut[batchRank + 1] = mN;
                    if (mOut.All(d => d > 0)) runtimeOutputShapes = new[] { mOut };
                }
            }

            // Runtime shape-preserving ops: a single-output op whose output has input[0]'s exact shape
            // (softmax + unary activations). When an upstream dynamic dim (DAv3's multi-view batch) leaves
            // the compiled output buffer sized for the wrong batch, resolve from the actual runtime input.
            if (nodeInputs.Length > 0 && nodeInputs[0] != null && (
                node.OpType is "Softmax" or "LogSoftmax" or "Relu" or "Gelu" or "Sigmoid" or "Tanh"
                or "Erf" or "Exp" or "Sqrt" or "Neg" or "Reciprocal" or "Softplus" or "Elu" or "LeakyRelu"
                or "Abs" or "Sin" or "Cos" or "Clip" or "Mish"))
                runtimeOutputShapes = new[] { (int[])nodeInputs[0]!.Shape.Clone() };

            // Runtime Reduce (ReduceMax/Min/Mean/Sum/...): output = input shape with the reduced axes removed
            // (keepdims=0) or set to 1 (keepdims=1). opset-18 passes axes as input[1]; when axes are absent
            // the ONNX default is REDUCE ALL (unless noop_with_empty_axes=1) — NOT last-dim. Compile-time
            // inference can't read the runtime axes input, so resolve here. DAv3 RoPE: ReduceMax over a
            // 3-D dynamic tensor must collapse to a scalar; the attr-only default left [3,257].
            if ((node.OpType is "ReduceMax" or "ReduceMin" or "ReduceMean" or "ReduceSum" or "ReduceProd"
                 or "ReduceL1" or "ReduceL2" or "ReduceSumSquare" or "ReduceLogSum" or "ReduceLogSumExp")
                && nodeInputs.Length > 0 && nodeInputs[0] != null)
            {
                var rIn = nodeInputs[0]!.Shape; int rRank = rIn.Length;
                int[] rAx;
                float[]? rAxV = node.InputNames.Length > 1 && !string.IsNullOrEmpty(node.InputNames[1])
                    ? runtimeConstants.GetValueOrDefault(node.InputNames[1]) : null;
                if (rAxV != null && rAxV.Length > 0)
                    rAx = rAxV.Select(a => (int)MathF.Round(a)).Select(a => a < 0 ? a + rRank : a).ToArray();
                else if (node.Attributes.TryGetValue("axes", out var rAxObj) && rAxObj is long[] rAl && rAl.Length > 0)
                    rAx = rAl.Select(a => (int)(a < 0 ? a + rRank : a)).ToArray();
                else
                {
                    bool rNoop = node.Attributes.TryGetValue("noop_with_empty_axes", out var rNop) && Convert.ToInt32(rNop) != 0;
                    rAx = rNoop ? Array.Empty<int>() : Enumerable.Range(0, rRank).ToArray(); // ONNX default: reduce ALL
                }
                bool rKeep = !node.Attributes.TryGetValue("keepdims", out var rKd) || Convert.ToInt32(rKd) != 0;
                var rOut = new List<int>();
                for (int i = 0; i < rRank; i++)
                {
                    if (Array.IndexOf(rAx, i) >= 0) { if (rKeep) rOut.Add(1); }
                    else rOut.Add(rIn[i]);
                }
                runtimeOutputShapes = new[] { rOut.Count > 0 ? rOut.ToArray() : new[] { 1 } }; // empty => scalar (1 elem)
            }

            // Runtime broadcast re-inference for elementwise/select ops poisoned by an upstream
            // value-dependent placeholder — re-infer the output from the ACTUAL runtime input shapes.
            if ((node.OpType == "Where" || node.OpType == "Cast" || node.OpType == "Add" || node.OpType == "Sub"
                 || node.OpType == "Mul" || node.OpType == "Div" || node.OpType == "Equal" || node.OpType == "Less"
                 || node.OpType == "Greater" || node.OpType == "And" || node.OpType == "Or" || node.OpType == "Not"
                 || node.OpType == "Min" || node.OpType == "Max" || node.OpType == "Pow") && nodeInputs.Length > 0)
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
                    && src.ElementCount >= 256                                   // skip only trivial/shape-ish reshapes;
                                                                                // 256 (was 4096) captures the GGUF DECODE
                                                                                // q/k/v + head-merge reshapes (seq=1 →
                                                                                // ~512-3584 elems, 112/step) → zero-copy
                                                                                // view, dropping 112 CopyFrom dispatches/
                                                                                // step (biggest on WebGPU). Single-consumer
                                                                                // gate below keeps it safe (the memory win);
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
                        // SEQ-MAJOR packed tensors [1, dTotal, dKvH, dHd] (step 3): the pack is contiguous seq-major;
                        // the contiguous Forward path reads it with seq_major_kv (carried in node.Attributes below).
                        nodeInputs[1] = new Tensor(await DecodeKVCache.PackedKAsync(dLayer, dTotal).ConfigureAwait(false), new[] { 1, dTotal, dKvH, dHd }, node.InputNames[1]);
                        nodeInputs[2] = new Tensor(await DecodeKVCache.PackedVAsync(dLayer, dTotal).ConfigureAwait(false), new[] { 1, dTotal, dKvH, dHd }, node.InputNames[2]);
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
                    // Runtime CPU shape interpreter resolved this output already (value in runtimeConstants) — skip
                    // the redundant GPU->CPU drain (DAv3-518's ~1400 of these are the WebGPU-dominant cost). In
                    // validate mode we DON'T skip: we read back + compare below to prove the CPU eval is correct.
                    if (outName != null && shapeInterpVals.ContainsKey(outName) && !ShapeInterpValidate)
                        continue;
                    // Skip the readback for pure-DATA producer ops whose small output is RoPE/positional FLOAT math
                    // (trig/reciprocal/pow/einsum, and float Range frequencies) — only consumed by GPU element-wise
                    // ops, never read as a CPU shape value. Their INPUTS are outputs of OTHER nodes and are read
                    // back normally, so Range still gets its scalars. Integer Range (indices) is NOT skipped.
                    if (shapeInterp && !ShapeInterpValidate && outName != null
                        && (node.OpType is "Cos" or "Sin" or "Reciprocal" or "Pow" or "Einsum" or "Sqrt" or "Erf"
                                or "Exp" or "Tanh" or "Sigmoid"
                            || (node.OpType == "Range" && !_integerTensorNames.Contains(outName))))
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
                            // Validation: the CPU shape interpreter claimed this value — confirm it matches the GPU
                            // truth. Any mismatch is a bug in TryComputeShapeOnCpu; log the first few loud.
                            if (ShapeInterpValidate && shapeInterpVals.TryGetValue(outName, out var cpuCheck))
                            {
                                var gpu = runtimeConstants[outName];
                                bool match = gpu.Length == cpuCheck.Length;
                                for (int c = 0; match && c < gpu.Length; c++)
                                    if (System.Math.Abs(gpu[c] - cpuCheck[c]) > 0.5f) match = false;
                                if (!match)
                                {
                                    LastRunShapeInterpMismatches++;
                                    if (LastRunShapeInterpMismatches <= 20)
                                        LastRunReadbackNames.Add($"SHAPEMISMATCH {node.OpType}:{outName} cpu=[{string.Join(",", cpuCheck)}] gpu=[{string.Join(",", gpu)}]");
                                }
                            }
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

        // Final yield + sync. Skip the yield during capture: Task.Yield ALWAYS reschedules onto another
        // thread-pool thread, and cuStreamEndCapture must run on the thread that began the capture. With the
        // yield (and the drains) suppressed, the whole captured forward runs synchronously on one thread.
        if (!SuppressDrains)
            await Task.Yield();
        // CUDA-graph capture: skip the final synchronize (illegal during capture) AND the buffer
        // returns. The captured forward is warm + single-pass, so leaving these pending is safe; the
        // caller resets SuppressDrains right after EndCapture and the normal drain path resumes.
        if (!SuppressDrains)
        {
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
        }

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
