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
        foreach (var input in graph.Inputs)
        {
            var shape = input.Shape.Select(d => d <= 0 ? 1 : d).ToArray();
            knownShapes[input.Name] = shape;
        }
        foreach (var (name, shape) in graph.Initializers)
        {
            if (shape != null) knownShapes[name] = shape;
        }
        // Pre-register graph output shapes (overrides inferred shapes for Reshape etc.)
        var graphOutputShapes = new Dictionary<string, int[]>();
        foreach (var output in graph.Outputs)
        {
            if (output.Shape.Length > 0)
            {
                var shape = output.Shape.Select(d => d <= 0 ? 1 : d).ToArray();
                graphOutputShapes[output.Name] = shape;
            }
        }

        // Compile each node
        var compiledNodes = new List<CompiledNode>();
        int nodeCompileIdx = 0;
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
                Console.WriteLine(shapeMsg);
                // Log for debugging but allow fallback (many models work despite imperfect shapes)
                // Fallback: try known output shape (from Initializers), then first input shape
                if (node.Outputs.Count > 0 && knownShapes.TryGetValue(node.Outputs[0], out var fallbackShape))
                    outputShapes = new[] { fallbackShape };
                else if (inputShapes.Length > 0 && inputShapes[0].Length > 0)
                    outputShapes = new[] { inputShapes[0] };
                else
                    outputShapes = new[] { new[] { 1 } };
            }

            // Compile-time evaluation of Shape nodes: output = input's known shape as a 1D tensor
            if (node.OpType == "Shape" && node.Inputs.Count >= 1
                && knownShapes.TryGetValue(node.Inputs[0], out var shapeInputShape))
            {
                var shapeValues = shapeInputShape;
                outputShapes = new[] { new[] { shapeValues.Length } };
                // Store computed values so downstream Reshape/Gather can use them
                graph.ConstantData ??= new Dictionary<string, int[]>();
                if (node.Outputs.Count > 0)
                {
                    graph.ConstantData[node.Outputs[0]] = shapeValues;
                    graph.FloatConstantData![node.Outputs[0]] = shapeValues.Select(v => (float)v).ToArray();
                }
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
                    outputShapes = new[] { new[] { 1 } }; // scalar as 1D
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
                    Console.WriteLine($"[GraphCompiler] Slice: in0={node.Inputs[0]}(const={in0}) in1={(node.Inputs.Count > 1 ? node.Inputs[1] : "?")}(const={in1}) in2={(node.Inputs.Count > 2 ? node.Inputs[2] : "?")}(const={in2})");
                }
                // Try opset >= 11: starts/ends from inputs[1], inputs[2]
                if (node.Inputs.Count >= 3
                    && graph.ConstantData.TryGetValue(node.Inputs[0], out var sliceData)
                    && graph.ConstantData.TryGetValue(node.Inputs[1], out var sliceStarts)
                    && graph.ConstantData.TryGetValue(node.Inputs[2], out var sliceEnds)
                    && sliceData.Length > 0 && sliceStarts.Length > 0 && sliceEnds.Length > 0)
                {
                    int start = sliceStarts[0];
                    int end = sliceEnds[0];
                    if (start < 0) start += sliceData.Length;
                    if (end < 0) end += sliceData.Length;
                    end = Math.Min(end, sliceData.Length);
                    if (start >= 0 && end > start)
                    {
                        var sliced = sliceData.Skip(start).Take(end - start).ToArray();
                        outputShapes = new[] { new[] { sliced.Length } };
                        if (node.Outputs.Count > 0)
                        {
                            graph.ConstantData[node.Outputs[0]] = sliced;
                            if (graph.FloatConstantData!.TryGetValue(node.Inputs[0], out var fSlice))
                                graph.FloatConstantData[node.Outputs[0]] = fSlice.Skip(start).Take(end - start).ToArray();
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
                        int s = shapeStarts[si]; int e = shapeEnds[si]; int step = Math.Abs(shapeSteps[si]);
                        if (step == 0) step = 1;
                        if (s < 0) s += outShape[ax];
                        if (e < 0) e += outShape[ax];
                        s = Math.Clamp(s, 0, outShape[ax]);
                        e = Math.Clamp(e, 0, outShape[ax]);
                        outShape[ax] = Math.Max(0, (e - s + step - 1) / step);
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
            if (node.OpType is "Mul" or "Add" or "Sub" or "Div"
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
                    graph.ConstantData[node.Outputs[0]] = result;
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
                        outShape[j] = Math.Max(inDim, tgtDim);
                    }
                    outputShapes = new[] { outShape };
                }
                else
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    Console.WriteLine($"[SHAPE_WARN] Expand '{outName}': shape tensor '{shapeTensorName}' not in ConstantData — using fallback");
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
                if (!resolved)
                {
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
                    Console.WriteLine($"[SHAPE_WARN] {node.OpType} '{outName}': scales/sizes not in ConstantData — using input shape as fallback");
                }
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
                if (graphOutputShapes.TryGetValue(outName, out var knownOutShape))
                    outputShapes[i] = knownOutShape; // Use known graph output shape
                knownShapes[outName] = outputShapes[i];
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
        if (graph.ConstantData != null && graph.ConstantData.Count > 0)
            Console.WriteLine($"[GraphCompiler] Compile-time constants: {graph.ConstantData.Count} tensors evaluated");

        return new CompiledGraph
        {
            Nodes = compiledNodes.ToArray(),
            InputNames = graph.Inputs.Select(i => i.Name).ToArray(),
            OutputNames = graph.Outputs.Select(o => o.Name).ToArray(),
            InputShapes = graph.Inputs.ToDictionary(i => i.Name, i => i.Shape),
            OutputShapes = graph.Outputs.ToDictionary(o => o.Name, o => knownShapes.TryGetValue(o.Name, out var s) ? s : Array.Empty<int>()),
            InitializerNames = graph.Initializers.Keys.ToHashSet(),
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
        // Build dependency graph
        var produced = new Dictionary<string, GraphNode>();
        foreach (var node in nodes)
            foreach (var output in node.Outputs)
                produced[output] = node;

        var inDegree = nodes.ToDictionary(n => n, _ => 0);
        foreach (var node in nodes)
            foreach (var input in node.Inputs)
                if (produced.TryGetValue(input, out var producer) && producer != node)
                    inDegree[node]++;

        var queue = new Queue<GraphNode>(nodes.Where(n => inDegree[n] == 0));
        var sorted = new List<GraphNode>();
        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            sorted.Add(node);
            foreach (var output in node.Outputs)
            {
                foreach (var consumer in nodes)
                {
                    if (consumer.Inputs.Contains(output) && consumer != node)
                    {
                        inDegree[consumer]--;
                        if (inDegree[consumer] == 0)
                            queue.Enqueue(consumer);
                    }
                }
            }
        }

        if (sorted.Count != nodes.Count)
            throw new InvalidOperationException($"Graph has cycles: sorted {sorted.Count}/{nodes.Count} nodes");

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
    public required HashSet<string> InitializerNames { get; init; }
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
