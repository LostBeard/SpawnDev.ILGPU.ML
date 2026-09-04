using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// Compiles a ModelGraph into an executable CompiledGraph.
/// Steps: validate ops → topological sort → shape inference.
/// </summary>
public class GraphCompiler
{
    private readonly OperatorRegistry _registry;

    /// <summary>Diagnostic: number of shape-subgraph nodes folded out of the last Compile (no runtime
    /// dispatch/readback). ~1400 for DAv3-518. Read by perf tests to prove the fold engaged.</summary>
    public static int LastCompileFoldedNodeCount;

    /// <summary>Compile-time shape-subgraph fold (2026-07-01). Removes pure compile-time shape-math nodes
    /// (Shape/Gather/Concat/Unsqueeze/Slice/Cast/Mul/Add/Sub/Div on non-dynamic inputs) from the executed graph
    /// to kill ~1400 per-inference GPU shape readbacks on DAv3-518. **DEFAULT ON (Tuvok 2026-07-11)** — bit-exact
    /// on the DAv3 rig (DA3Small_Pipeline_5D_ElideDispatch, maxAbsDiff&lt;1e-3) + SD-Turbo WebGPU (ElideAB, 0.00%)
    /// after the CLIP Range INT64_MAX-sentinel fix AND the MoveNet integer-floordiv fix (TryComputeShapeOnCpu now
    /// truncates integer Div, matching the GPU DivOperator). Validated by the standard 6-backend PMT sweep.
    /// GPU-tensor-consumed folded values are handled by the elideSafe path-c gate (ElideBlockedOutputs).</summary>
    public static bool ShapeSubgraphFoldEnabled = true;

    public GraphCompiler(OperatorRegistry registry) => _registry = registry;

    /// <summary>Check if constant data is valid (no INT_MAX/INT_MIN sentinels from dynamic dims).</summary>
    private static bool IsValidConstant(int[] data) =>
        data.Length > 0 && !data.Any(v => v == int.MaxValue || v == int.MinValue);

    /// <summary>
    /// Compile a model graph for execution.
    /// Resolves operators, topologically sorts nodes, infers output shapes.
    /// </summary>
    /// <summary>Enable graph optimization (operator fusion) before compilation.</summary>
    public bool EnableOptimization { get; set; } = true;

    public CompiledGraph Compile(ModelGraph graph)
    {
      try
      {
        // Apply graph optimizations (operator fusion) before compilation
        if (EnableOptimization)
        {
            try { graph = GraphOptimizer.Optimize(graph); }
            catch (IndexOutOfRangeException optEx)
            {
                throw new InvalidOperationException(
                    $"[GraphCompiler] Optimizer crashed (IndexOutOfRange) on graph with {graph.Nodes.Count} nodes, " +
                    $"{graph.Initializers.Count} initializers, {graph.Inputs.Count} inputs. " +
                    $"Inputs: [{string.Join(", ", graph.Inputs.Select(i => $"{i.Name}:[{string.Join(",", i.Shape)}]"))}]",
                    optEx);
            }
        }

        // Initialize float constant data for precise compile-time arithmetic.
        // ConstantData uses int (fine for shapes/indices) but Upsample scale chains
        // need float precision (e.g., Mul(dim, 0.5) must give 0.5, not 0).
        graph.FloatConstantData ??= new Dictionary<string, float[]>();
        // Seed from existing FloatConstantData (populated by InferenceSession)
        // and from ConstantData (int→float promotion)
        if (graph.ConstantData != null)
        {
            foreach (var (name, intVals) in graph.ConstantData)
            {
                if (!graph.FloatConstantData.ContainsKey(name))
                    graph.FloatConstantData[name] = intVals.Select(v => (float)v).ToArray();
            }
        }

        // Validate all ops are supported
        foreach (var node in graph.Nodes)
        {
            if (!_registry.IsSupported(node.OpType))
                throw new NotSupportedException($"Unsupported ONNX operator: {node.OpType} (node outputs: {string.Join(",", node.Outputs)})");
        }

        // Topological sort
        List<GraphNode> sorted;
        try { sorted = TopologicalSort(graph.Nodes); }
        catch (Exception ex) { throw new InvalidOperationException($"TopologicalSort failed on {graph.Nodes.Count} nodes: {ex.Message}", ex); }

        // Shape inference: track known shapes from inputs, initializers, and outputs
        var knownShapes = new Dictionary<string, int[]>();
        var dynamicInputs = new HashSet<string>(); // Inputs with dynamic dims (d<=0 in ONNX)
        foreach (var input in graph.Inputs)
        {
            var shape = input.Shape.Select(d => d <= 0 ? 1 : d).ToArray();
            knownShapes[input.Name] = shape;
            if (input.Shape.Any(d => d <= 0))
                dynamicInputs.Add(input.Name);
        }
        foreach (var (name, shape) in graph.Initializers)
        {
            if (shape != null) knownShapes[name] = shape;
        }
        // Pre-register graph output shapes (overrides inferred shapes for Reshape etc.)
        // Stored RAW (dynamic dims stay <=0): a declared dynamic dim must NEVER clobber a
        // correctly-inferred runtime dim. The old code resolved -1 -> 1 here and overrode
        // unconditionally below, pinning e.g. a GGUF decoder's declared logits [1,-1,vocab]
        // to [1,1,vocab] even when inference had the true seq>1 - so the output buffer was
        // allocated one position big and the LM-head MatMul (which sizes M from its RUNTIME
        // input) silently wrote seq*vocab floats past it (2026-06-12).
        var graphOutputShapes = new Dictionary<string, int[]>();
        foreach (var output in graph.Outputs)
        {
            if (output.Shape.Length > 0)
                graphOutputShapes[output.Name] = output.Shape;
        }

        // Constant names known to hold INTEGER (int64) values, seeded from Shape/Size - which ONNX
        // defines as int64 - and carried through the value-preserving folds. Used to pick integer
        // arithmetic (truncating Div) over float arithmetic when folding.
        var integerConstNames = new HashSet<string>();

        // Optional shape tracing, see the SHAPE_TRACE line below.
        // Hoisted: this is consulted per Gather NODE, and an env lookup per node at compile time is a
        // syscall the compiler does not need.
        bool _traceScalars = Environment.GetEnvironmentVariable("ML_TRACE_SCALARS") == "1";
        var _traceShapes = Environment.GetEnvironmentVariable("ML_TRACE_SHAPES");
        if (string.IsNullOrEmpty(_traceShapes)) _traceShapes = null;
        if (_traceShapes != null) Console.WriteLine($"[SHAPE_TRACE] enabled, matching '{_traceShapes}', {sorted.Count} nodes");

        // Compile each node
        var compiledNodes = new List<CompiledNode>();
        int nodeCompileIdx = 0;
        int foldedNodeCount = 0;
        // Compile-time-constant node outputs (name -> fp32) whose per-node <=64-elem readback the executor can
        // skip: it seeds runtimeConstants from these instead of doing the GPU->CPU drain. Nodes still execute.
        var foldedConstants = new Dictionary<string, float[]>();
        foreach (var node in sorted)
        {
          try
          {
            IOnnxOperator op;
            try { op = _registry.Resolve(node.OpType); }
            catch (Exception ex) { throw new InvalidOperationException($"Node {nodeCompileIdx} '{node.OpType}': operator not registered — {ex.Message}"); }
            Dictionary<string, object> attrs;
            try { attrs = node.GetTypedAttributes(); }
            catch (Exception ex) { throw new InvalidOperationException($"Node {nodeCompileIdx} '{node.OpType}': attribute parse failed — {ex.Message}"); }

            // Gather input shapes (empty string = optional ONNX input, use empty shape)
            var inputShapes = node.Inputs
                .Select(name => string.IsNullOrEmpty(name) ? Array.Empty<int>()
                    : knownShapes.TryGetValue(name, out var s) ? s
                    : throw new InvalidOperationException($"Unknown shape for '{name}' (needed by {node.OpType})"))
                .ToArray();

            // Split: inject split sizes from constant input[1] (opset 13+) or node output count
            if (node.OpType == "Split")
            {
                if (!attrs.ContainsKey("split") && node.Inputs.Count >= 2
                    && !string.IsNullOrEmpty(node.Inputs[1])
                    && graph.ConstantData != null
                    && graph.ConstantData.TryGetValue(node.Inputs[1], out var splitVals))
                {
                    attrs["split"] = splitVals.Select(v => (long)v).ToArray();
                }
                if (!attrs.ContainsKey("num_outputs"))
                    attrs["num_outputs"] = (long)node.Outputs.Count;
            }

            // ConstantOfShape's output SHAPE is its input tensor's VALUES, not that tensor's shape - a
            // 1-element input holding [2] means "produce 2 elements". InferOutputShapes is only handed
            // shapes, so it cannot know that; hand it the values when they are foldable. Without this the
            // output collapses to one element per input ENTRY, and every shape computed downstream of it
            // is short - which is how ZipVoice's encoder ended up building a 2-entry pad list for a
            // rank-2 tensor that needs 4, and crashing inside the Pad kernel.
            if (node.OpType == "ConstantOfShape"
                && node.Inputs.Count >= 1 && !string.IsNullOrEmpty(node.Inputs[0])
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var cosDimVals)
                && cosDimVals.Length > 0)
            {
                var cosDims = new long[cosDimVals.Length];
                for (int i = 0; i < cosDimVals.Length; i++) cosDims[i] = (long)cosDimVals[i];
                attrs["_resolved_shape"] = cosDims;
            }

            // Unsqueeze/Squeeze moved `axes` from an attribute to an INPUT in opset 13, and shape inference
            // is handed only shapes - so it silently returned the input rank unchanged and the inserted
            // axis disappeared. Supplying the folded axes under the attribute name the operator already
            // reads restores the pre-13 behaviour exactly. Left unfixed, an index tensor meant to broadcast
            // to [412,103] stays [412], and the mismatch only surfaces in an Add several nodes later.
            // The Reduce family moved `axes` the same way (ReduceSum at opset 13, the rest at 18), and the
            // consequence is worse than a missing dimension: inference falls back to reducing the LAST axis,
            // so summing [303,4,2,512] over axis 2 produced [303,4,2] instead of [303,4,512] - a plausible
            // shape carrying the wrong data. Execute already resolves the input form; only inference was blind.
            if ((node.OpType == "Unsqueeze" || node.OpType == "Squeeze"
                    || node.OpType is "ReduceSum" or "ReduceMean" or "ReduceMax" or "ReduceMin"
                        or "ReduceProd" or "ReduceL1" or "ReduceL2" or "ReduceLogSum"
                        or "ReduceLogSumExp" or "ReduceSumSquare")
                && !attrs.ContainsKey("axes")
                && node.Inputs.Count >= 2 && !string.IsNullOrEmpty(node.Inputs[1])
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var axesVals)
                && axesVals.Length > 0)
            {
                var axesLongs = new long[axesVals.Length];
                for (int d = 0; d < axesVals.Length; d++) axesLongs[d] = axesVals[d];
                attrs["axes"] = axesLongs;
            }

            // Infer output shapes
            int[][] outputShapes;
            try
            {
                outputShapes = op.InferOutputShapes(inputShapes, attrs);
            }
            catch (Exception shapeEx)
            {
                var shapeMsg = $"[GraphCompiler] Shape inference failed at node {nodeCompileIdx} '{node.OpType}' " +
                    $"inputs=[{string.Join("; ", inputShapes.Select(s => $"[{string.Join(",", s)}]"))}] " +
                    $"inputNames=[{string.Join(",", node.Inputs)}] outputs=[{string.Join(",", node.Outputs)}]: {shapeEx.Message}";
                if (InferenceSession.VerboseLogging) Console.WriteLine(shapeMsg);
                // Log for debugging but allow fallback (many models work despite imperfect shapes)
                // Fallback: try known output shape (from Initializers), then first input shape
                if (node.Outputs.Count > 0 && knownShapes.TryGetValue(node.Outputs[0], out var fallbackShape))
                    outputShapes = new[] { fallbackShape };
                else if (inputShapes.Length > 0 && inputShapes[0].Length > 0)
                    outputShapes = new[] { inputShapes[0] };
                else
                    outputShapes = new[] { new[] { 1 } };
            }

            // ── ONNX Gather rank rule for a RANK-0 index ──
            // out rank = data.rank - 1 + indices.rank, so a rank-0 index DROPS the gathered axis. The
            // operator cannot see this itself: every scalar is stored as rank-1 [1], so it receives [1] for
            // both a true scalar and a 1-element vector and has to keep the axis to stay safe for the
            // vector case. ScalarTensorNames carries the distinction the storage format loses.
            //
            // ⚠️ Only the compile-time SHAPE changes; the buffer is still one element. Without this,
            // Gather(Shape[R], scalarIdx) reported rank 1, the following Unsqueeze produced [1,1] instead
            // of [1], and the Concat assembling a dim list got [[1,1];[1]] - see ScalarTensorNames for the
            // ZipVoice measurement.
            if (node.OpType == "Gather" && node.Inputs.Count >= 2 && _traceScalars)
                Console.WriteLine($"[scalars] Gather '{node.Outputs[0]}' idx='{node.Inputs[1]}' "
                    + $"isScalar={graph.ScalarTensorNames?.Contains(node.Inputs[1])} "
                    + $"dataKnown={knownShapes.ContainsKey(node.Inputs[0])} "
                    + $"dataShape=[{(knownShapes.TryGetValue(node.Inputs[0], out var _dbg) ? string.Join(",", _dbg) : "?")}]");

            if (node.OpType == "Gather" && node.Inputs.Count >= 2
                && graph.ScalarTensorNames != null && graph.ScalarTensorNames.Contains(node.Inputs[1])
                && knownShapes.TryGetValue(node.Inputs[0], out var gDataShape) && gDataShape.Length > 0)
            {
                int gAxis = attrs.TryGetValue("axis", out var gAxObj) ? Convert.ToInt32(gAxObj) : 0;
                if (gAxis < 0) gAxis += gDataShape.Length;
                if (gAxis >= 0 && gAxis < gDataShape.Length)
                {
                    var dropped = new List<int>();
                    for (int gd = 0; gd < gDataShape.Length; gd++)
                        if (gd != gAxis) dropped.Add(gDataShape[gd]);
                    outputShapes = new[] { dropped.ToArray() };   // rank-0 when data was rank-1
                }
            }

            // Compile-time evaluation of Shape nodes: output = input's known shape as a 1D tensor.
            // Skip folding for graph inputs with dynamic dims — their compile-time shapes
            // (with d<=0 replaced by 1) produce wrong values for downstream Reshape/Resize.
            // The runtime GraphExecutor resolves these correctly from actual tensor shapes.
            if (node.OpType == "Shape" && node.Inputs.Count >= 1
                && knownShapes.TryGetValue(node.Inputs[0], out var shapeInputShape)
                && !dynamicInputs.Contains(node.Inputs[0]))
            {
                var shapeValues = shapeInputShape;
                outputShapes = new[] { new[] { shapeValues.Length } };
                // Store computed values so downstream Reshape/Gather can use them
                graph.ConstantData ??= new Dictionary<string, int[]>();
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = shapeValues;
                    graph.FloatConstantData![node.Outputs[0]] = shapeValues.Select(v => (float)v).ToArray();
                    integerConstNames.Add(node.Outputs[0]);   // Shape is int64 by definition
                }
            }

            // Size, like Shape, is int64 by definition.
            if (node.OpType == "Size" && node.Outputs.Count > 0) integerConstNames.Add(node.Outputs[0]);

            // Integer-ness rides along value-preserving folds; a Cast to a floating type ends it.
            if (node.Outputs.Count > 0 && node.Inputs.Count > 0 && !string.IsNullOrEmpty(node.Inputs[0])
                && integerConstNames.Contains(node.Inputs[0]))
            {
                bool carries = node.OpType switch
                {
                    "Gather" or "Concat" or "Unsqueeze" or "Squeeze" or "Reshape" or "Transpose"
                        or "Slice" or "Identity" or "Tile" or "Expand" or "Where"
                        or "Min" or "Max" or "Mod" or "Floor" or "Ceil" or "Neg" or "Abs" => true,
                    // 1 = float, 10 = float16, 11 = double: casting to one of those makes it float.
                    "Cast" => !(attrs.TryGetValue("to", out var castTo) && castTo is long castToType
                                && (castToType == 1 || castToType == 10 || castToType == 11)),
                    _ => false,
                };
                if (carries) integerConstNames.Add(node.Outputs[0]);
            }

            // Compile-time evaluation of Gather on known constant data
            if (node.OpType == "Gather" && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var gatherSrc)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var gatherIdxData)
                && gatherIdxData.Length == 1
                && IsValidConstant(gatherSrc) && IsValidConstant(gatherIdxData))
            {
                int gIdx = gatherIdxData[0];
                if (gIdx < 0) gIdx += gatherSrc.Length;
                if (gIdx >= 0 && gIdx < gatherSrc.Length)
                {
                    // ⚠️ "scalar as 1D" is the STORAGE shape, and it used to be the inferred shape too -
                    // which silently discarded the ONNX rank rule on the constant-folded path. A rank-0
                    // index makes this Gather rank-0 (data.rank - 1 + 0), and the fold has to say so or the
                    // following Unsqueeze adds an axis to a rank it should not have had. The uncorrected
                    // block above is reached only when the fold does not apply, so both paths need it.
                    outputShapes = new[] {
                        graph.ScalarTensorNames != null && graph.ScalarTensorNames.Contains(node.Inputs[1])
                            && knownShapes.TryGetValue(node.Inputs[0], out var gfData) && gfData.Length == 1
                            ? Array.Empty<int>()
                            : new[] { 1 } };
                    if (node.Outputs.Count > 0)
                    {
                        graph.ConstantData[node.Outputs[0]] = new[] { gatherSrc[gIdx] };
                        // Float: use float source if available (preserves fractional values)
                        if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fSrc))
                            graph.FloatConstantData[node.Outputs[0]] = new[] { fSrc[gIdx] };
                        else
                            graph.FloatConstantData[node.Outputs[0]] = new[] { (float)gatherSrc[gIdx] };
                    }
                }
            }

            // Compile-time Concat evaluation on known constants
            if (node.OpType == "Concat" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && node.Inputs.All(inp => !string.IsNullOrEmpty(inp) && graph.ConstantData.ContainsKey(inp)
                    && IsValidConstant(graph.ConstantData[inp])))
            {
                var concatVals = node.Inputs.SelectMany(inp => graph.ConstantData[inp]).ToArray();
                outputShapes = new[] { new[] { concatVals.Length } };
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = concatVals;
                    // Float: concat float arrays if all available
                    if (graph.FloatConstantData != null && node.Inputs.All(inp => graph.FloatConstantData.ContainsKey(inp)))
                        graph.FloatConstantData[node.Outputs[0]] = node.Inputs.SelectMany(inp => graph.FloatConstantData[inp]).ToArray();
                }
            }

            // Unsqueeze on known constants
            if (node.OpType == "Unsqueeze" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var unsqData))
            {
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = unsqData;
                    if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fUnsq))
                        graph.FloatConstantData[node.Outputs[0]] = fUnsq;
                }
            }

            // Range on known bounds. Its length is a function of the VALUES, so inference (which only sees
            // shapes) has to answer [1] and everything sized from it - Expand, Tile, and the GatherElements
            // that consumes them as indices - collapses to a single element. The executor resolves Range
            // itself at runtime, but that fixes only Range's own buffer, not the ops shaped from it.
            if (node.OpType == "Range" && node.Inputs.Count >= 3
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var rangeStartVals) && rangeStartVals.Length > 0
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var rangeLimitVals) && rangeLimitVals.Length > 0
                && graph.ConstantData.TryGetValue(node.Inputs[2], out var rangeDeltaVals) && rangeDeltaVals.Length > 0
                && rangeDeltaVals[0] != 0)
            {
                int rangeStart = rangeStartVals[0], rangeLimit = rangeLimitVals[0], rangeDelta = rangeDeltaVals[0];
                long rangeCount = (long)Math.Ceiling((rangeLimit - rangeStart) / (double)rangeDelta);
                if (rangeCount < 0) rangeCount = 0;

                // Bounded: this is shape arithmetic, not a place to materialise a large tensor.
                if (rangeCount <= 65536)
                {
                    var rangeVals = new int[rangeCount];
                    for (int d = 0; d < rangeVals.Length; d++) rangeVals[d] = rangeStart + d * rangeDelta;
                    outputShapes = new[] { new[] { (int)rangeCount } };
                    if (node.Outputs.Count > 0)
                    {
                        graph.ConstantData[node.Outputs[0]] = rangeVals;
                        if (graph.FloatConstantData != null)
                            graph.FloatConstantData[node.Outputs[0]] = rangeVals.Select(v => (float)v).ToArray();
                        integerConstNames.Add(node.Outputs[0]);
                    }
                }
            }

            // ConstantOfShape on a known shape: a buffer of that many entries, all the value attribute.
            // Its SHAPE being right is not enough - the chain that consumes it needs the VALUES, and
            // without them the very next Concat has an unknown input and the whole fold stops there.
            if (node.OpType == "ConstantOfShape" && node.Inputs.Count >= 1 && !string.IsNullOrEmpty(node.Inputs[0])
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var cosShapeVals)
                && cosShapeVals.Length > 0)
            {
                long cosCount = 1;
                foreach (var d in cosShapeVals) cosCount *= d;

                // Bounded: this is for shape arithmetic, not for materialising a large tensor at compile time.
                if (cosCount > 0 && cosCount <= 4096)
                {
                    int cosFill = 0;
                    bool cosKnown = true;
                    if (attrs.TryGetValue("value", out var cosValueObj))
                    {
                        switch (cosValueObj)
                        {
                            case long[] cosLongs when cosLongs.Length > 0: cosFill = (int)cosLongs[0]; break;
                            case int[] cosInts when cosInts.Length > 0: cosFill = cosInts[0]; break;
                            case float[] cosFloats when cosFloats.Length > 0: cosFill = (int)cosFloats[0]; break;
                            case long cosLong: cosFill = (int)cosLong; break;
                            case int cosInt: cosFill = cosInt; break;
                            case double cosDouble: cosFill = (int)cosDouble; break;
                            case null: cosFill = 0; break;   // ONNX default
                            default: cosKnown = false; break;   // unknown representation - do not guess
                        }
                    }

                    if (cosKnown && node.Outputs.Count > 0)
                    {
                        var cosOut = new int[cosCount];
                        for (int d = 0; d < cosOut.Length; d++) cosOut[d] = cosFill;
                        graph.ConstantData[node.Outputs[0]] = cosOut;
                    }
                }
            }

            // Reshape on known constants: only the RANK changes, so the flat value list passes through.
            // Without this the pad-amount chain torch's F.pad export builds (concat -> reshape into pairs
            // -> reverse -> transpose -> flatten) stops being foldable at its first Reshape, and the Pad
            // consuming it cannot know its own output size at compile time - which leaves every tensor
            // after the Pad sized as if no padding had happened.
            if (node.OpType == "Reshape" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var reshapeConstData))
            {
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = reshapeConstData;
                    if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fReshape))
                        graph.FloatConstantData[node.Outputs[0]] = fReshape;
                }
            }

            // Transpose on known constants. Unlike Reshape this REORDERS values, so it needs the input's
            // shape and the permutation, not just the flat list.
            if (node.OpType == "Transpose" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var transposeConstData)
                && inputShapes.Length > 0 && inputShapes[0].Length > 1)
            {
                var tShape = inputShapes[0];
                long tTotal = 1;
                foreach (var d in tShape) tTotal *= d;
                long[]? tPerm = attrs.TryGetValue("perm", out var tPermObj) && tPermObj is long[] tPermArr ? tPermArr : null;

                if (tTotal == transposeConstData.Length)
                {
                    var tAxes = new int[tShape.Length];
                    bool tValid = true;
                    for (int d = 0; d < tShape.Length && tValid; d++)
                    {
                        // ONNX default permutation is the reverse of the axes.
                        int axis = tPerm != null && tPerm.Length == tShape.Length ? (int)tPerm[d] : tShape.Length - 1 - d;
                        if (axis < 0) axis += tShape.Length;
                        if (axis < 0 || axis >= tShape.Length) { tValid = false; break; }
                        tAxes[d] = axis;
                    }

                    if (tValid)
                    {
                        var tOutShape = new int[tShape.Length];
                        for (int d = 0; d < tShape.Length; d++) tOutShape[d] = tShape[tAxes[d]];

                        var tStrides = new int[tShape.Length];
                        tStrides[tShape.Length - 1] = 1;
                        for (int d = tShape.Length - 2; d >= 0; d--) tStrides[d] = tStrides[d + 1] * tShape[d + 1];

                        var tResult = new int[transposeConstData.Length];
                        var tIndex = new int[tShape.Length];
                        for (int flat = 0; flat < tResult.Length; flat++)
                        {
                            int src = 0;
                            for (int d = 0; d < tShape.Length; d++) src += tIndex[d] * tStrides[tAxes[d]];
                            tResult[flat] = transposeConstData[src];
                            for (int d = tShape.Length - 1; d >= 0; d--)
                            {
                                if (++tIndex[d] < tOutShape[d]) break;
                                tIndex[d] = 0;
                            }
                        }

                        if (node.Outputs.Count > 0) graph.ConstantData[node.Outputs[0]] = tResult;
                    }
                }
            }

            // Pad's amounts are an INPUT for opset >= 11, and its shape inference only sees shapes - so it
            // returned the input shape unchanged and the padding vanished from every shape downstream.
            // Hand it the folded amounts when they are known.
            if (node.OpType == "Pad" && node.Inputs.Count >= 2 && !string.IsNullOrEmpty(node.Inputs[1])
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var padAmountVals)
                && padAmountVals.Length > 0)
            {
                var padAmounts = new long[padAmountVals.Length];
                for (int d = 0; d < padAmountVals.Length; d++) padAmounts[d] = padAmountVals[d];
                attrs["_resolved_pads"] = padAmounts;
            }

            // Compile-time Slice on known constants: Slice(data, starts, ends[, axes, steps])
            // Handles both opset >= 11 (starts/ends as tensor inputs) and opset < 11 (as attributes)
            if (node.OpType == "Slice" && graph.ConstantData != null)
            {
                // DEBUG: Log Slice resolution for attention scaling diagnosis
                if (node.Inputs.Count >= 3)
                {
                    bool in0 = graph.ConstantData.ContainsKey(node.Inputs[0]);
                    bool in1 = node.Inputs.Count > 1 && graph.ConstantData.ContainsKey(node.Inputs[1]);
                    bool in2 = node.Inputs.Count > 2 && graph.ConstantData.ContainsKey(node.Inputs[2]);
                    if (InferenceSession.VerboseLogging) Console.WriteLine($"[GraphCompiler] Slice: in0={node.Inputs[0]}(const={in0}) in1={(node.Inputs.Count > 1 ? node.Inputs[1] : "?")}(const={in1}) in2={(node.Inputs.Count > 2 ? node.Inputs[2] : "?")}(const={in2})");
                }
                // Try opset >= 11: starts/ends from inputs[1], inputs[2]
                if (node.Inputs.Count >= 3
                    && graph.ConstantData.TryGetValue(node.Inputs[0], out var sliceData)
                    && graph.ConstantData.TryGetValue(node.Inputs[1], out var sliceStarts)
                    && graph.ConstantData.TryGetValue(node.Inputs[2], out var sliceEnds)
                    && sliceData.Length > 0 && sliceStarts.Length > 0 && sliceEnds.Length > 0)
                {
                    // Axis and step both matter. This used to assume axis 0 of a FLAT list walked forwards,
                    // which mis-slices a 2-D constant (axis 0 selects ROWS, not elements) and cannot express
                    // the reversal torch emits when it builds a pad list. Only the single-axis-0 case is
                    // folded; anything else is left to execute normally.
                    var sliceAxes = node.Inputs.Count > 3 && graph.ConstantData.TryGetValue(node.Inputs[3], out var sliceAxesVals) ? sliceAxesVals : null;
                    var sliceSteps = node.Inputs.Count > 4 && graph.ConstantData.TryGetValue(node.Inputs[4], out var sliceStepVals) ? sliceStepVals : null;
                    int sliceAxis = sliceAxes != null && sliceAxes.Length > 0 ? sliceAxes[0] : 0;
                    long sliceStep = sliceSteps != null && sliceSteps.Length > 0 ? sliceSteps[0] : 1;

                    var dataShape = inputShapes.Length > 0 && inputShapes[0].Length > 0
                        ? inputShapes[0]
                        : new[] { sliceData.Length };
                    if (sliceAxis < 0) sliceAxis += dataShape.Length;

                    long dataTotal = 1;
                    foreach (var d in dataShape) dataTotal *= d;

                    int foldRowSize = 1;
                    for (int d = 1; d < dataShape.Length; d++) foldRowSize *= dataShape[d];

                    // A shape we cannot reconcile with the value count means one of them is already wrong;
                    // folding on it would turn a detectable problem into a plausible wrong answer.
                    if (sliceStarts.Length == 1 && sliceStep != 0 && sliceAxis == 0
                        && dataTotal == sliceData.Length && foldRowSize > 0
                        && sliceData.Length % foldRowSize == 0 && sliceData.Length / foldRowSize > 0)
                    {
                        int dim0 = sliceData.Length / foldRowSize;
                        long start = sliceStarts[0];
                        long end = sliceEnds[0];
                        if (start < 0) start += dim0;
                        if (end < 0) end += dim0;

                        var rows = new List<int>();
                        if (sliceStep > 0)
                        {
                            start = Math.Clamp(start, 0, dim0);
                            end = Math.Clamp(end, 0, dim0);
                            for (long i = start; i < end; i += sliceStep) rows.Add((int)i);
                        }
                        else
                        {
                            // A negative step may legitimately end one BEFORE index 0 - that is what the
                            // INT64_MIN sentinel means - so the end floor is -1, not 0.
                            start = Math.Clamp(start, 0, dim0 - 1);
                            end = Math.Clamp(end, -1, dim0 - 1);
                            for (long i = start; i > end; i += sliceStep) rows.Add((int)i);
                        }

                        var sliced = new int[rows.Count * foldRowSize];
                        for (int r = 0; r < rows.Count; r++)
                            Array.Copy(sliceData, rows[r] * foldRowSize, sliced, r * foldRowSize, foldRowSize);

                        var slicedShape = (int[])dataShape.Clone();
                        slicedShape[0] = rows.Count;
                        outputShapes = new[] { slicedShape };
                        if (node.Outputs.Count > 0)
                        {
                            graph.ConstantData[node.Outputs[0]] = sliced;
                            if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fSlice)
                                && fSlice.Length == sliceData.Length)
                            {
                                var fOut = new float[sliced.Length];
                                for (int r = 0; r < rows.Count; r++)
                                    Array.Copy(fSlice, rows[r] * foldRowSize, fOut, r * foldRowSize, foldRowSize);
                                graph.FloatConstantData[node.Outputs[0]] = fOut;
                            }
                        }
                    }
                }
                // Shape-only inference (opset >= 11): data is NOT constant but starts/ends ARE
                else if (node.Inputs.Count >= 3
                    && !graph.ConstantData.ContainsKey(node.Inputs[0]) // data is runtime
                    && graph.ConstantData.TryGetValue(node.Inputs[1], out var shapeStarts)
                    && graph.ConstantData.TryGetValue(node.Inputs[2], out var shapeEnds))
                {
                    var shapeAxes = node.Inputs.Count > 3 && graph.ConstantData.TryGetValue(node.Inputs[3], out var sa)
                        ? sa : Enumerable.Range(0, shapeStarts.Length).ToArray();
                    var shapeSteps = node.Inputs.Count > 4 && graph.ConstantData.TryGetValue(node.Inputs[4], out var ss)
                        ? ss : Enumerable.Repeat(1, shapeStarts.Length).ToArray();
                    var outShape = (int[])inputShapes[0].Clone();
                    bool sliceValid = true;
                    for (int si = 0; si < shapeAxes.Length && sliceValid; si++)
                    {
                        int ax = shapeAxes[si] < 0 ? shapeAxes[si] + outShape.Length : shapeAxes[si];
                        if (ax < 0 || ax >= outShape.Length) { sliceValid = false; break; }
                        // Long arithmetic throughout: the "to the beginning" sentinel is INT64_MIN, and
                        // adding the dimension to it in int would overflow into a positive number.
                        long s = shapeStarts[si]; long e = shapeEnds[si];
                        long step = shapeSteps[si];
                        if (step == 0) step = 1;
                        long dim = outShape[ax];
                        if (s < 0) s += dim;
                        if (e < 0) e += dim;

                        long sliceLength;
                        if (step > 0)
                        {
                            s = Math.Clamp(s, 0, dim);
                            e = Math.Clamp(e, 0, dim);
                            sliceLength = e > s ? (e - s + step - 1) / step : 0;
                        }
                        else
                        {
                            // A negative step walks BACKWARDS, and its bounds are not the forward ones: the
                            // start may sit on the last element, and the end may legitimately fall one before
                            // index 0 - which is exactly what the INT64_MIN sentinel means in the reversal
                            // that torch emits when it builds a pad list. Taking Math.Abs of the step and
                            // clamping the end up to 0, as this did, turns "reverse the whole axis" into an
                            // EMPTY slice, and every shape downstream collapses with it.
                            s = Math.Clamp(s, 0, dim - 1);
                            e = Math.Clamp(e, -1, dim - 1);
                            sliceLength = s > e ? (s - e - 1) / (-step) + 1 : 0;
                        }
                        outShape[ax] = (int)Math.Max(0, sliceLength);
                    }
                    if (sliceValid)
                        outputShapes = new[] { outShape };

                    // Store resolved slice params in the typed attrs dict so the executor
                    // can read them at runtime via GetInts("_resolved_starts") etc.
                    attrs["_resolved_starts"] = shapeStarts.Select(v => (long)v).ToArray();
                    attrs["_resolved_ends"] = shapeEnds.Select(v => (long)v).ToArray();
                    attrs["_resolved_axes"] = shapeAxes.Select(v => (long)v).ToArray();
                    attrs["_resolved_steps"] = shapeSteps.Select(v => (long)v).ToArray();
                }
                // Try opset < 11: starts/ends from attributes
                else if (node.Inputs.Count >= 1
                    && graph.ConstantData.TryGetValue(node.Inputs[0], out var sliceDataAttr)
                    && attrs.TryGetValue("starts", out var startsObj) && startsObj is long[] startsArr
                    && attrs.TryGetValue("ends", out var endsObj) && endsObj is long[] endsArr)
                {
                    int start = (int)startsArr[0];
                    int end = (int)endsArr[0];
                    if (start < 0) start += sliceDataAttr.Length;
                    if (end < 0) end += sliceDataAttr.Length;
                    end = Math.Min(end, sliceDataAttr.Length);
                    if (start >= 0 && end > start)
                    {
                        var sliced = sliceDataAttr.Skip(start).Take(end - start).ToArray();
                        outputShapes = new[] { new[] { sliced.Length } };
                        if (node.Outputs.Count > 0)
                        {
                            graph.ConstantData[node.Outputs[0]] = sliced;
                            if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fSlice))
                                graph.FloatConstantData[node.Outputs[0]] = fSlice.Skip(start).Take(end - start).ToArray();
                        }
                    }
                }
            }

            // Compile-time scalar/element-wise arithmetic on known constants.
            // Uses FloatConstantData for precise arithmetic (0.5 * dim must not truncate to 0).
            // Falls back to int ConstantData if float not available.
            // ONNX arithmetic follows the operand DTYPE, and Div on an integer dtype truncates rather
            // than producing a fraction. Shape and Size are int64 by definition, and ONNX requires both
            // operands of a binary op to share a dtype - so if either side is Shape-derived, this is
            // integer arithmetic and the float path must not be used. Otherwise a chain like
            // (432 + 2) / 3 keeps 144.667 instead of 144, and the *next* multiply lands on 289 instead
            // of 288 - which moves a slice boundary and mismatches two tensors by one element, far from
            // where the mistake was made. The float path stays for genuinely float shape math (0.5 * dim
            // must not truncate to 0), which is never Shape-derived.
            bool arithIsInteger = node.Inputs.Count >= 2
                && ((!string.IsNullOrEmpty(node.Inputs[0]) && integerConstNames.Contains(node.Inputs[0]))
                    || (!string.IsNullOrEmpty(node.Inputs[1]) && integerConstNames.Contains(node.Inputs[1])));

            if (node.OpType is "Mul" or "Add" or "Sub" or "Div"
                && !arithIsInteger
                && node.Inputs.Count >= 2 && graph.FloatConstantData != null
                && graph.FloatConstantData.TryGetValue(node.Inputs[0], out var fArithA)
                && graph.FloatConstantData.TryGetValue(node.Inputs[1], out var fArithB))
            {
                int len = Math.Max(fArithA.Length, fArithB.Length);
                var fResult = new float[len];
                var iResult = new int[len];
                for (int j = 0; j < len; j++)
                {
                    float a = fArithA[j % fArithA.Length];
                    float b = fArithB[j % fArithB.Length];
                    fResult[j] = node.OpType switch
                    {
                        "Mul" => a * b,
                        "Add" => a + b,
                        "Sub" => a - b,
                        "Div" => b != 0 ? a / b : 0,
                        _ => a
                    };
                    iResult[j] = (int)fResult[j];
                }
                outputShapes = new[] { fArithA.Length >= fArithB.Length ? inputShapes[0] : inputShapes[1] };
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData![node.Outputs[0]] = iResult;
                    graph.FloatConstantData[node.Outputs[0]] = fResult;
                }
            }
            else if (node.OpType is "Mul" or "Add" or "Sub" or "Div"
                && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var arithA)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var arithB))
            {
                // Int-only fallback (no float data available)
                int len = Math.Max(arithA.Length, arithB.Length);
                var result = new int[len];
                for (int j = 0; j < len; j++)
                {
                    int a = arithA[j % arithA.Length];
                    int b = arithB[j % arithB.Length];
                    result[j] = node.OpType switch
                    {
                        "Mul" => a * b,
                        "Add" => a + b,
                        "Sub" => a - b,
                        "Div" => b != 0 ? a / b : 0,
                        _ => a
                    };
                }
                outputShapes = new[] { arithA.Length >= arithB.Length ? inputShapes[0] : inputShapes[1] };
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = result;
                    // Keep the float copy in step with the integer one. Leaving a stale (or fractional)
                    // float behind is what let the truncation be undone by the very next operation.
                    if (graph.FloatConstantData != null)
                        graph.FloatConstantData[node.Outputs[0]] = result.Select(v => (float)v).ToArray();
                    if (arithIsInteger) integerConstNames.Add(node.Outputs[0]);
                }
            }

            // Compile-time Cast on known constants
            if (node.OpType == "Cast" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var castData))
            {
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = castData;
                    // Float: Cast may truncate (e.g., float→int64), apply Floor for int casts
                    if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fCast))
                        graph.FloatConstantData[node.Outputs[0]] = fCast; // preserve float through cast
                }
            }

            // Compile-time Where(condition, X, Y) on known constants
            // Common in Resize size computation: Where(Equal(dim, 0), origDim, newDim)
            if (node.OpType == "Where" && node.Inputs.Count >= 3
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var whereCond)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var whereX)
                && graph.ConstantData.TryGetValue(node.Inputs[2], out var whereY))
            {
                int len = Math.Max(whereCond.Length, Math.Max(whereX.Length, whereY.Length));
                var result = new int[len];
                for (int j = 0; j < len; j++)
                {
                    bool cond = whereCond[j % whereCond.Length] != 0;
                    result[j] = cond ? whereX[j % whereX.Length] : whereY[j % whereY.Length];
                }
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = result;
                    graph.FloatConstantData![node.Outputs[0]] = result.Select(v => (float)v).ToArray();
                }
            }

            // Compile-time Equal on known constants (produces boolean 0/1)
            if (node.OpType == "Equal" && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var eqA)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var eqB))
            {
                int len = Math.Max(eqA.Length, eqB.Length);
                var result = new int[len];
                for (int j = 0; j < len; j++)
                    result[j] = eqA[j % eqA.Length] == eqB[j % eqB.Length] ? 1 : 0;
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = result;
                    graph.FloatConstantData![node.Outputs[0]] = result.Select(v => (float)v).ToArray();
                }
            }

            // Compile-time ORDERING comparisons on known constants (boolean 0/1), the family `Equal` above
            // was missing.
            //
            // ⚠️ WHY IT MATTERS OUT OF ALL PROPORTION TO ITS SIZE. Every `If` in ZipVoice's decoder and text
            // encoder is guarded by exactly this shape: `GreaterOrEqual(<initializer>, <Shape-derived>)` -
            // "is the cached positional table long enough?". `Shape`, `Gather`, `Mul` and `Sub` all fold at
            // compile time already, so the whole subtree collapses to two constants and stops one node short.
            // MEASURED 2026-09-03 on CUDA: `ONE decoder step: 25 readbacks | Shapex18, GreaterOrEqualx5,
            // Expandx2` - five GPU round trips per Euler step to evaluate a comparison whose operands are both
            // known before the model runs, on the stage that is 82% of a synthesis in the browser.
            //
            // ⚠️ FLOAT operands, not the int mirror. ConstantData is int, and these conditions compare
            // sequence arithmetic where truncation would silently change which branch is taken; FloatConstantData
            // holds the same values without that risk. Fall back to the int copy only when no float one exists.
            if (node.OpType is "Greater" or "GreaterOrEqual" or "Less" or "LessOrEqual"
                && node.Inputs.Count >= 2 && graph.ConstantData != null)
            {
                float[]? cmpA = graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fa) ? fa
                    : graph.ConstantData.TryGetValue(node.Inputs[0], out var ia) ? ia.Select(v => (float)v).ToArray() : null;
                float[]? cmpB = graph.FloatConstantData!.TryGetValue(node.Inputs[1], out var fb) ? fb
                    : graph.ConstantData.TryGetValue(node.Inputs[1], out var ib) ? ib.Select(v => (float)v).ToArray() : null;
                if (cmpA != null && cmpB != null && cmpA.Length > 0 && cmpB.Length > 0 && node.Outputs.Count > 0)
                {
                    int len = Math.Max(cmpA.Length, cmpB.Length);
                    var result = new int[len];
                    for (int j = 0; j < len; j++)
                    {
                        float a = cmpA[j % cmpA.Length], b = cmpB[j % cmpB.Length];
                        result[j] = node.OpType switch
                        {
                            "Greater" => a > b,
                            "GreaterOrEqual" => a >= b,
                            "Less" => a < b,
                            _ => a <= b,
                        } ? 1 : 0;
                    }
                    graph.ConstantData[node.Outputs[0]] = result;
                    graph.FloatConstantData![node.Outputs[0]] = result.Select(v => (float)v).ToArray();
                }
            }

            // Compile-time Floor/Ceil on known constants
            if (node.OpType is "Floor" or "Ceil" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var floorData))
            {
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = floorData;
                    // Float: apply actual floor/ceil
                    if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fFloor))
                    {
                        var fResult = node.OpType == "Floor"
                            ? fFloor.Select(v => MathF.Floor(v)).ToArray()
                            : fFloor.Select(v => MathF.Ceiling(v)).ToArray();
                        graph.FloatConstantData[node.Outputs[0]] = fResult;
                        graph.ConstantData[node.Outputs[0]] = fResult.Select(v => (int)v).ToArray();
                    }
                }
            }

            // Compile-time Squeeze on known constants
            if (node.OpType == "Squeeze" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var sqData))
            {
                if (node.Outputs.Count > 0)
                    graph.ConstantData[node.Outputs[0]] = sqData;
            }

            // Special-case: Reshape needs the actual shape tensor values.
            if (node.OpType == "Reshape" && node.Inputs.Count >= 2)
            {
                var shapeTensorName = node.Inputs[1];
                if (graph.ConstantData != null && graph.ConstantData.TryGetValue(shapeTensorName, out var targetDims))
                {
                    int inputElems = inputShapes[0].Aggregate(1, (a, b) => a * b);
                    var outShape = targetDims.ToArray();
                    // Handle 0 dims (copy from input shape)
                    for (int j = 0; j < outShape.Length; j++)
                        if (outShape[j] == 0 && j < inputShapes[0].Length) outShape[j] = inputShapes[0][j];
                    int negIdx = Array.IndexOf(outShape, -1);
                    if (negIdx >= 0)
                    {
                        int knownProduct = 1;
                        for (int j = 0; j < outShape.Length; j++)
                            if (j != negIdx) knownProduct *= outShape[j];
                        outShape[negIdx] = knownProduct > 0 ? inputElems / knownProduct : 1;
                    }
                    outputShapes = new[] { outShape };
                }
                else
                {
                    // Shape tensor not resolved — use rank from shape tensor's known shape, put elements in dim 0
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    if (InferenceSession.VerboseLogging)
                        Console.WriteLine($"[SHAPE_WARN] Reshape '{outName}': shape tensor '{shapeTensorName}' not in ConstantData — using fallback");
                    if (knownShapes.TryGetValue(shapeTensorName, out var shapeTensorShape)
                        && shapeTensorShape.Length == 1)
                    {
                        int outRank = shapeTensorShape[0];
                        var outShape = new int[outRank];
                        int inputElems = inputShapes[0].Aggregate(1, (a, b) => a * b);
                        outShape[0] = inputElems;
                        for (int j = 1; j < outRank; j++) outShape[j] = 1;
                        outputShapes = new[] { outShape };
                    }
                }
            }

            // Special-case: Expand needs the shape tensor to compute broadcast output shape.
            // Second input is a 1D tensor of target dimensions. Output = numpy-broadcast(input, target).
            if (node.OpType == "Expand" && node.Inputs.Count >= 2)
            {
                var shapeTensorName = node.Inputs[1];
                if (graph.ConstantData != null && graph.ConstantData.TryGetValue(shapeTensorName, out var targetDims))
                {
                    // Numpy-style broadcast: pad shorter shape with leading 1s, then max per dim
                    var inShape = inputShapes[0];
                    int outRank = Math.Max(inShape.Length, targetDims.Length);
                    var outShape = new int[outRank];
                    for (int j = 0; j < outRank; j++)
                    {
                        int inDim = j < outRank - inShape.Length ? 1 : inShape[j - (outRank - inShape.Length)];
                        int tgtDim = j < outRank - targetDims.Length ? 1 : targetDims[j - (outRank - targetDims.Length)];
                        // Broadcast, not max. A size-1 dimension takes the OTHER side's size, and that
                        // includes ZERO: expanding [1,1,512] to a target of [0,1,512] must give an EMPTY
                        // tensor. Taking the maximum turns it into a one-element one, which then survives a
                        // Concat as a phantom extra frame and breaks the next Reshape's element count.
                        outShape[j] = inDim == 1 ? tgtDim : tgtDim == 1 ? inDim : Math.Max(inDim, tgtDim);
                    }
                    outputShapes = new[] { outShape };
                }
                else
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    if (InferenceSession.VerboseLogging) Console.WriteLine($"[SHAPE_WARN] Expand '{outName}': shape tensor '{shapeTensorName}' not in ConstantData — using fallback");
                }
            }

            // Special-case: Tile needs the repeats tensor to compute its output shape - out[i] = in[i]*repeats[i].
            // Without this the node reported its INPUT shape as its output, so the executor allocated an
            // output the size of the input and TileOperator's own "inCount == outCount -> just copy" fast path
            // fired: the tile silently never happened. That is not a cosmetic shape error - whisper-tiny's
            // decoder builds its causal mask through a Tile, so the mask arrived as a single zero, every
            // position could see every other, and the model emitted end-of-text as its first token. Resolved
            // here (and re-published as `_resolved_repeats`) rather than in the operator because the buffer is
            // ALLOCATED from this shape; by the time Execute runs, an undersized output is already fixed.
            if (node.OpType == "Tile" && node.Inputs.Count >= 2 && graph.ConstantData != null)
            {
                var repeatsName = node.Inputs[1];
                if (graph.ConstantData.TryGetValue(repeatsName, out var repeats))
                {
                    var inShape = inputShapes[0];
                    // ONNX: repeats is 1-D with one entry per input dimension.
                    var outShape = (int[])inShape.Clone();
                    for (int j = 0; j < outShape.Length; j++)
                    {
                        int r = j < repeats.Length ? repeats[j] : 1;
                        outShape[j] = inShape[j] * Math.Max(0, r);
                    }
                    outputShapes = new[] { outShape };
                    attrs["_resolved_repeats"] = repeats.Select(v => (long)v).ToArray();
                }
                else if (InferenceSession.VerboseLogging)
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    Console.WriteLine($"[SHAPE_WARN] Tile '{outName}': repeats '{repeatsName}' not in ConstantData — output shape unresolved");
                }
            }

            // Special-case: Upsample/Resize need scales or sizes to compute output shape.
            // Scales tensor is the second input for Upsample, or third/fourth for Resize.
            if (node.OpType is "Upsample" or "Resize" && graph.ConstantData != null)
            {
                // Try scales from input[1] (Upsample) or input[2] (Resize)
                int scalesIdx = node.OpType == "Upsample" ? 1 : 2;
                // Resize also has optional sizes at input[3]
                int sizesIdx = 3;

                bool resolved = false;

                // Try sizes first (Resize input[3]) — absolute output dimensions
                if (!resolved && node.OpType == "Resize" && node.Inputs.Count > sizesIdx
                    && !string.IsNullOrEmpty(node.Inputs[sizesIdx])
                    && graph.ConstantData.TryGetValue(node.Inputs[sizesIdx], out var sizesData)
                    && sizesData.Length == inputShapes[0].Length)
                {
                    var outShape = sizesData.ToArray();
                    // Replace 0s with input dims
                    for (int j = 0; j < outShape.Length; j++)
                        if (outShape[j] <= 0) outShape[j] = inputShapes[0][j];
                    outputShapes = new[] { outShape };
                    resolved = true;
                }

                // Try scales — multiply input dimensions by scale factors.
                // MUST use FloatConstantData: scale factors like [1.0, 1.0, 2.0, 2.0] truncate
                // to [1, 1, 2, 2] in int ConstantData (OK), but the computation chain that
                // PRODUCES them goes through Mul(dim, 0.5) where 0.5→0 in int kills the chain.
                if (!resolved && node.Inputs.Count > scalesIdx
                    && !string.IsNullOrEmpty(node.Inputs[scalesIdx])
                    && graph.FloatConstantData != null
                    && graph.FloatConstantData.TryGetValue(node.Inputs[scalesIdx], out var fScalesData)
                    && fScalesData.Length == inputShapes[0].Length)
                {
                    var outShape = new int[inputShapes[0].Length];
                    for (int j = 0; j < outShape.Length; j++)
                        outShape[j] = (int)MathF.Floor(inputShapes[0][j] * fScalesData[j]);
                    outputShapes = new[] { outShape };
                    resolved = true;
                }

                // Fallback: try int scales
                if (!resolved && node.Inputs.Count > scalesIdx
                    && !string.IsNullOrEmpty(node.Inputs[scalesIdx])
                    && graph.ConstantData!.TryGetValue(node.Inputs[scalesIdx], out var scalesData)
                    && scalesData.Length == inputShapes[0].Length)
                {
                    var outShape = new int[inputShapes[0].Length];
                    for (int j = 0; j < outShape.Length; j++)
                        outShape[j] = inputShapes[0][j] * scalesData[j];
                    outputShapes = new[] { outShape };
                    resolved = true;
                }

                // Log resolution result (verbose only)
                if (InferenceSession.VerboseLogging)
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    var resolvedShape = resolved ? $"[{string.Join(",", outputShapes[0])}]" : "FALLBACK";
                    Console.WriteLine($"[GraphCompiler] {node.OpType} '{outName}': resolved={resolved} shape={resolvedShape} input=[{string.Join(",", inputShapes[0])}]");
                }
                // Fallback for Resize: try to resolve sizes from the Shape of a known
                // tensor in the sizes chain. DepthAnything V2 computes Resize sizes via
                // Shape→Slice→Concat→Where which is hard to propagate fully. But if ANY
                // node in the sizes chain is a Shape of a tensor with known dims, we can
                // infer the target spatial dimensions.
                if (!resolved && node.OpType == "Resize" && node.Inputs.Count > sizesIdx
                    && !string.IsNullOrEmpty(node.Inputs[sizesIdx]))
                {
                    // Trace the sizes input back to find a Shape node with known dims
                    var visited = new HashSet<string>();
                    var queue = new Queue<string>();
                    queue.Enqueue(node.Inputs[sizesIdx]);
                    while (queue.Count > 0 && visited.Count < 20 && !resolved)
                    {
                        var traceName = queue.Dequeue();
                        if (string.IsNullOrEmpty(traceName) || !visited.Add(traceName)) continue;
                        // Check if this value is in ConstantData
                        if (graph.ConstantData.TryGetValue(traceName, out var traceData)
                            && traceData.Length == inputShapes[0].Length
                            && IsValidConstant(traceData))
                        {
                            var outShape = traceData.ToArray();
                            for (int j = 0; j < outShape.Length; j++)
                                if (outShape[j] <= 0) outShape[j] = inputShapes[0][j];
                            outputShapes = new[] { outShape };
                            resolved = true;
                            break;
                        }
                        // Trace through producer node
                        var producer = sorted.FirstOrDefault(nd => nd.Outputs.Contains(traceName));
                        if (producer != null)
                            foreach (var inp in producer.Inputs.Where(i => !string.IsNullOrEmpty(i)))
                                queue.Enqueue(inp);
                    }
                }

                if (!resolved)
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    if (InferenceSession.VerboseLogging) Console.WriteLine($"[SHAPE_WARN] {node.OpType} '{outName}': scales/sizes not in ConstantData — using input shape as fallback");
                }
            }

            // Compile-time Pad shape resolution: if pads tensor is a known constant,
            // compute output shape = input + pads. Handles TFLite PAD and ONNX opset >= 11.
            if (node.OpType == "Pad" && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && !string.IsNullOrEmpty(node.Inputs[1])
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var padConst)
                && padConst.Length == inputShapes[0].Length * 2)
            {
                var padded = (int[])inputShapes[0].Clone();
                int rank = padded.Length;
                for (int d = 0; d < rank; d++)
                    padded[d] += padConst[d] + padConst[rank + d];
                if (padded.All(d => d > 0))
                    outputShapes = new[] { padded };
            }

            // If the operator returned fewer shapes than outputs (e.g., Split returning
            // equal splits without knowing exact output count), extend to match.
            if (outputShapes.Length < node.Outputs.Count && outputShapes.Length > 0)
            {
                var extended = new int[node.Outputs.Count][];
                for (int i = 0; i < node.Outputs.Count; i++)
                    extended[i] = i < outputShapes.Length ? outputShapes[i] : (int[])outputShapes[^1].Clone();
                outputShapes = extended;
            }

            // Register output shapes. Priority: graph output override > initializer shape > inferred shape
            for (int i = 0; i < node.Outputs.Count && i < outputShapes.Length; i++)
            {
                var outName = node.Outputs[i];
                // Don't overwrite a known Initializer shape with a weaker inference (e.g., Constant returns [1])
                if (knownShapes.TryGetValue(outName, out var existingShape)
                    && existingShape.Length > 1 && outputShapes[i].Length <= 1)
                    outputShapes[i] = existingShape;
                if (graphOutputShapes.TryGetValue(outName, out var declaredOutShape))
                {
                    // Declared graph-output shape vs inferred: a fully-STATIC declaration wins
                    // (the original rescue for ops whose inference is weak, e.g. Reshape). A
                    // declaration with dynamic dims (<=0) only contributes its static dims;
                    // every dynamic dim keeps the INFERRED value (resolving -1 -> 1 here pinned
                    // dynamic decoders to seq=1 and undersized their output buffers).
                    if (declaredOutShape.All(d => d > 0))
                        outputShapes[i] = declaredOutShape;
                    else if (declaredOutShape.Length == outputShapes[i].Length)
                        outputShapes[i] = declaredOutShape
                            .Zip(outputShapes[i], (dec, inf) => dec > 0 ? dec : inf).ToArray();
                    // else: dynamic declaration with a rank mismatch - trust the inference.
                }
                // A Constant node's shape comes from its own value, but attributes reach the operator as
                // plain numbers with the tensor's dims stripped, so inference can only answer [1]. When the
                // folded value is longer than that, its LENGTH is the shape - these constants are 1-D. Left
                // wrong, a 2-entry pad list is called one element and every shape derived from it collapses,
                // which is what broke ZipVoice's encoder several nodes later inside Pad.
                if (node.OpType == "Constant" && outputShapes[i].Length <= 1)
                {
                    int foldedLength = 0;
                    if (graph.ConstantData != null && graph.ConstantData.TryGetValue(outName, out var foldedInts))
                        foldedLength = foldedInts.Length;
                    else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(outName, out var foldedFloats))
                        foldedLength = foldedFloats.Length;
                    if (foldedLength > 1) outputShapes[i] = new[] { foldedLength };
                }

                knownShapes[outName] = outputShapes[i];

                // Trace how one tensor's compiled shape was arrived at. Set ML_TRACE_SHAPES to a
                // substring of the tensor names you care about. A collapsed shape here is invisible at
                // runtime - the graph just crashes somewhere downstream with a size nothing explains -
                // so being able to watch the shape as it is decided is the difference between reading
                // the answer and guessing at it.
                if (_traceShapes != null && outName.Contains(_traceShapes, StringComparison.Ordinal))
                {
                    // Show the folded VALUE too when it is small. Shape arithmetic goes wrong through the
                    // values (an off-by-one in one folded scalar moves every slice boundary after it), and a
                    // shape alone cannot show that.
                    var folded = "";
                    if (graph.ConstantData != null && graph.ConstantData.TryGetValue(outName, out var traceInts) && traceInts.Length <= 8)
                        folded = $" = [{string.Join(",", traceInts)}]";
                    else if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(outName, out var traceFloats) && traceFloats.Length <= 8)
                        folded = $" = [{string.Join(",", traceFloats)}]f";

                    Console.WriteLine($"[SHAPE_TRACE] {node.OpType,-16} {outName,-46} -> [{string.Join(",", outputShapes[i])}]{folded}"
                        + $"  from [{string.Join("; ", inputShapes.Select(sh => "[" + string.Join(",", sh) + "]"))}]"
                        + $"  inputs=[{string.Join(",", node.Inputs)}]");
                }
            }

            // COMPILE-TIME READBACK ELIMINATION (2026-07-01): the inline evaluators above (Shape/Gather/Concat/
            // Unsqueeze/Slice/Cast/Mul/Add/Sub/Div on non-dynamic inputs) proved this node's output is a small
            // compile-time constant. The node still RUNS (so every tensor + shape is produced exactly as before -
            // no removal, no shape-resolution or missing-tensor hazards), but the executor's per-node <=64-elem
            // capture readback of it (a full GPU->CPU pipeline drain, ~1400 of them on DAv3-518 = the dominant
            // cost) is REDUNDANT: we already know the value. Record it so the executor seeds runtimeConstants from
            // it and SKIPS the readback. Net: same correctness, ~1400 drains removed. (Node REMOVAL was tried and
            // corrupts shape resolution - a folded Concat input regressed to the wrong rank; readback-skip keeps
            // the graph intact.) The readback only fires for <=64-elem outputs, so match that bound.
            bool isReadbackConstOp = ShapeSubgraphFoldEnabled && (node.OpType is "Shape" or "Gather" or "Concat"
                or "Unsqueeze" or "Squeeze" or "Slice" or "Cast" or "Mul" or "Add" or "Sub" or "Div");
            // IsValidConstant rejects INT_MAX/INT_MIN sentinels the compiler stores for dims it could NOT resolve
            // at compile time (dynamic / unknown). Seeding those is garbage - only skip the readback for outputs
            // that are fully, correctly resolved. This is the reliability gate: the readback stays for anything
            // uncertain (so the executor's runtime value wins), the readback-skip applies only to proven constants.
            if (isReadbackConstOp && node.Outputs.Count > 0 && graph.ConstantData != null
                && node.Outputs.All(o => !string.IsNullOrEmpty(o)
                    && graph.ConstantData.TryGetValue(o, out var cv) && cv.Length > 0 && cv.Length <= 64
                    && IsValidConstant(cv)))
            {
                foreach (var o in node.Outputs)
                {
                    var intVals = graph.ConstantData[o];
                    float[] floatVals;
                    if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(o, out var fv)
                        && fv.Length == intVals.Length)
                        floatVals = fv;
                    else
                    {
                        floatVals = new float[intVals.Length];
                        for (int k = 0; k < intVals.Length; k++) floatVals[k] = intVals[k];
                    }
                    foldedConstants[o] = floatVals;
                }
                foldedNodeCount++;
                // NOTE: no `continue` - the node is still compiled + executed below.
            }

            compiledNodes.Add(new CompiledNode
            {
                OpType = node.OpType,
                Operator = op,
                InputNames = node.Inputs.ToArray(),
                OutputNames = node.Outputs.ToArray(),
                Attributes = attrs,
                OutputShapes = outputShapes,
            });
          }
          catch (Exception nodeEx)
          {
            var outNames = string.Join(",", node.Outputs);
            var inNames = string.Join(",", node.Inputs.Take(3));
            var inShapes = string.Join("; ", node.Inputs.Take(3).Select(n =>
                knownShapes.TryGetValue(n ?? "", out var s) ? $"[{string.Join(",", s)}]" : "?"));
            // Include preceding 10 nodes for context + constant data values
            var prevNodes = new System.Text.StringBuilder();
            for (int p = Math.Max(0, compiledNodes.Count - 10); p < compiledNodes.Count; p++)
            {
                var cn = compiledNodes[p];
                var constInfo = "";
                if (cn.OpType is "Reshape" or "Concat" or "Gather" or "Shape" or "Unsqueeze")
                {
                    foreach (var inp in cn.InputNames)
                    {
                        if (graph.ConstantData != null && graph.ConstantData.TryGetValue(inp, out var cv))
                            constInfo += $" {inp}=const[{string.Join(",", cv.Take(5))}]";
                    }
                }
                prevNodes.Append($"\n  #{p} {cn.OpType} in=[{string.Join(",", cn.InputNames.Take(4))}] " +
                    $"out=[{string.Join(",", cn.OutputNames)}] " +
                    $"shapes=[{string.Join("; ", cn.OutputShapes.Select(s => $"[{string.Join(",", s)}]"))}]{constInfo}");
            }
            throw new IndexOutOfRangeException(
                $"Node {nodeCompileIdx}/{sorted.Count} '{node.OpType}' crashed. " +
                $"Inputs=[{inNames}] shapes=({inShapes}) Outputs=[{outNames}]" +
                $"\nPreceding nodes:{prevNodes}", nodeEx);
          }
            nodeCompileIdx++;
        }

        // Log compile-time evaluation stats
        LastCompileFoldedNodeCount = foldedNodeCount;
        if (InferenceSession.VerboseLogging && graph.ConstantData != null && graph.ConstantData.Count > 0)
            Console.WriteLine($"[GraphCompiler] Compile-time constants: {graph.ConstantData.Count} tensors evaluated; "
                + $"{foldedNodeCount} shape-subgraph nodes folded out (no runtime dispatch/readback)");

        return new CompiledGraph
        {
            Nodes = compiledNodes.ToArray(),
            InputNames = graph.Inputs.Select(i => i.Name).ToArray(),
            OutputNames = graph.Outputs.Select(o => o.Name).ToArray(),
            InputShapes = graph.Inputs.ToDictionary(i => i.Name, i => i.Shape),
            // Resolved input shapes actually used for compile-time shape inference
            // (dynamic dims d<=0 replaced with 1, or pinned via inputShapes override).
            // The executor compares runtime input tensor shapes against these to decide
            // whether the model is being run at a DIFFERENT shape than it was compiled for
            // (e.g. autoregressive generation growing the sequence dim) and, if so,
            // re-infers per-node output buffer sizes from the actual runtime input shapes.
            CompiledInputShapes = graph.Inputs.ToDictionary(
                i => i.Name,
                i => knownShapes.TryGetValue(i.Name, out var s) ? s : i.Shape.Select(d => d <= 0 ? 1 : d).ToArray()),
            OutputShapes = graph.Outputs.ToDictionary(o => o.Name, o => knownShapes.TryGetValue(o.Name, out var s) ? s : Array.Empty<int>()),
            InitializerNames = graph.Initializers.Keys.ToHashSet(),
            InitializerDataTypes = graph.InitializerDataTypes,
            ScalarTensorNames = graph.ScalarTensorNames,
            FoldedShapeConstants = foldedConstants,
        };
      }
      catch (Exception compileEx)
      {
        throw new InvalidOperationException(
            $"[GraphCompiler] Compile crashed: {compileEx.GetType().Name}: {compileEx.Message} " +
            $"(graph: {graph.Nodes.Count} nodes, {graph.Initializers.Count} initializers, " +
            $"optimization={EnableOptimization})", compileEx);
      }
    }

    /// <summary>Topological sort using Kahn's algorithm.</summary>
    private static List<GraphNode> TopologicalSort(List<GraphNode> nodes)
    {
        // Build dependency graph: map each tensor name to the node that produces it.
        // If multiple nodes produce the same name (e.g., If branches flattened), keep the LAST
        // producer — ONNX guarantees the last writer is authoritative for downstream consumers.
        var produced = new Dictionary<string, GraphNode>();
        foreach (var node in nodes)
            foreach (var output in node.Outputs)
                if (!string.IsNullOrEmpty(output))
                    produced[output] = node;

        // Build adjacency: for each node, find which other nodes it depends on
        var deps = new Dictionary<GraphNode, HashSet<GraphNode>>();
        foreach (var node in nodes) deps[node] = new HashSet<GraphNode>();
        foreach (var node in nodes)
            foreach (var input in node.Inputs)
                if (!string.IsNullOrEmpty(input) && produced.TryGetValue(input, out var producer) && producer != node)
                    deps[node].Add(producer);

        var inDegree = nodes.ToDictionary(n => n, n => deps[n].Count);
        // Reverse map: for each node, which nodes depend on it?
        var consumers = new Dictionary<GraphNode, List<GraphNode>>();
        foreach (var node in nodes) consumers[node] = new List<GraphNode>();
        foreach (var node in nodes)
            foreach (var dep in deps[node])
                consumers[dep].Add(node);

        var queue = new Queue<GraphNode>(nodes.Where(n => inDegree[n] == 0));
        var sorted = new List<GraphNode>();
        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            sorted.Add(node);
            // Notify all consumers that this dependency is satisfied (once per consumer)
            foreach (var consumer in consumers[node])
            {
                inDegree[consumer]--;
                if (inDegree[consumer] == 0)
                    queue.Enqueue(consumer);
            }
        }

        if (sorted.Count != nodes.Count)
        {
            // Diagnostic: find stuck nodes and their unresolved dependencies
            var stuckNodes = nodes.Where(n => !sorted.Contains(n)).Take(5);
            var diag = string.Join("; ", stuckNodes.Select(n =>
            {
                var unresolved = deps[n].Where(d => !sorted.Contains(d))
                    .Select(d => $"{d.OpType}({string.Join(",", d.Outputs.Take(1))})")
                    .Take(3);
                return $"{n.OpType}({string.Join(",", n.Outputs.Take(1))}) waits on [{string.Join(",", unresolved)}]";
            }));
            throw new InvalidOperationException(
                $"Graph has cycles: sorted {sorted.Count}/{nodes.Count} nodes. Stuck: {diag}");
        }

        return sorted;
    }
}

/// <summary>A compiled graph ready for execution.</summary>
public class CompiledGraph
{
    public required CompiledNode[] Nodes { get; init; }
    public required string[] InputNames { get; init; }
    public required string[] OutputNames { get; init; }
    public required Dictionary<string, int[]> InputShapes { get; init; }
    public required Dictionary<string, int[]> OutputShapes { get; init; }
    /// <summary>Input shapes as resolved at compile time for shape inference: dynamic dims
    /// (ONNX d&lt;=0) replaced with 1, or pinned to a caller-supplied override. The executor
    /// compares actual runtime input tensor shapes against these to detect a shape change
    /// (e.g. a growing sequence_length during autoregressive decode) and re-infer per-node
    /// output buffer sizes from the real input shapes. Defaults to empty for graphs built
    /// without the compiler (none currently — kept non-required for forward compatibility).</summary>
    public Dictionary<string, int[]> CompiledInputShapes { get; init; } = new();
    public required HashSet<string> InitializerNames { get; init; }
    /// <summary>Maps initializer (and Constant-node output) name to its ONNX-declared
    /// data type code (see <see cref="Onnx.OnnxDataType"/>). Consumed by GraphExecutor
    /// to seed integer-tensor dataflow propagation. Null when the source model didn't
    /// supply dtype information (e.g., TFLite / CoreML paths).</summary>
    public Dictionary<string, int>? InitializerDataTypes { get; init; }

    /// <summary>ONNX rank-0 tensor names - see <see cref="ModelGraph.ScalarTensorNames"/>.
    /// The RUNTIME shape overrides need this too: they recompute Gather's output from live tensor
    /// shapes, where a scalar is indistinguishable from a [1] vector for exactly the same reason.</summary>
    public HashSet<string>? ScalarTensorNames { get; init; }
    /// <summary>Compile-time-constant node outputs (&lt;=64 elem, name -&gt; fp32) proven by the compiler
    /// (Shape/Gather/Concat/Unsqueeze/Slice/Cast/Mul/Add/Sub/Div on non-dynamic inputs). The producing nodes
    /// STILL execute (graph unchanged) - the executor seeds runtimeConstants from these and SKIPS their per-node
    /// &lt;=64-elem capture readback (the GPU-&gt;CPU drain that dominates DAv3-518). Null/empty when the fold is off.</summary>
    public Dictionary<string, float[]>? FoldedShapeConstants { get; init; }
}

/// <summary>A single compiled operation.</summary>
public class CompiledNode
{
    public required string OpType { get; init; }
    public required IOnnxOperator Operator { get; init; }
    public required string[] InputNames { get; init; }
    public required string[] OutputNames { get; init; }
    public required Dictionary<string, object> Attributes { get; init; }
    public required int[][] OutputShapes { get; init; }
}
