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

    /// <summary>
    /// Compile a model graph for execution.
    /// Resolves operators, topologically sorts nodes, infers output shapes.
    /// </summary>
    /// <summary>Enable graph optimization (operator fusion) before compilation.</summary>
    public bool EnableOptimization { get; set; } = true;

    public CompiledGraph Compile(ModelGraph graph)
    {
        // Apply graph optimizations (operator fusion) before compilation
        if (EnableOptimization)
            graph = GraphOptimizer.Optimize(graph);

        // Validate all ops are supported
        foreach (var node in graph.Nodes)
        {
            if (!_registry.IsSupported(node.OpType))
                throw new NotSupportedException($"Unsupported ONNX operator: {node.OpType} (node outputs: {string.Join(",", node.Outputs)})");
        }

        // Topological sort
        var sorted = TopologicalSort(graph.Nodes);

        // Shape inference: track known shapes from inputs, initializers, and outputs
        var knownShapes = new Dictionary<string, int[]>();
        foreach (var input in graph.Inputs)
        {
            // Replace dynamic dimensions (-1) with 1 (default batch size)
            var shape = input.Shape.Select(d => d <= 0 ? 1 : d).ToArray();
            knownShapes[input.Name] = shape;
        }
        foreach (var (name, shape) in graph.Initializers)
            knownShapes[name] = shape;
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
        foreach (var node in sorted)
        {
            var op = _registry.Resolve(node.OpType);
            var attrs = node.GetTypedAttributes();

            // Gather input shapes (empty string = optional ONNX input, use empty shape)
            var inputShapes = node.Inputs
                .Select(name => string.IsNullOrEmpty(name) ? Array.Empty<int>()
                    : knownShapes.TryGetValue(name, out var s) ? s
                    : throw new InvalidOperationException($"Unknown shape for '{name}' (needed by {node.OpType})"))
                .ToArray();

            // Infer output shapes
            int[][] outputShapes;
            try
            {
                outputShapes = op.InferOutputShapes(inputShapes, attrs);
            }
            catch
            {
                // Fallback: output shape = first input shape (common for element-wise)
                outputShapes = new[] { inputShapes[0] };
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
                    graph.ConstantData[node.Outputs[0]] = shapeValues;
            }

            // Compile-time evaluation of Gather on known constant data
            if (node.OpType == "Gather" && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var gatherSrc)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var gatherIdxData)
                && gatherIdxData.Length == 1)
            {
                int gIdx = gatherIdxData[0];
                if (gIdx < 0) gIdx += gatherSrc.Length;
                if (gIdx >= 0 && gIdx < gatherSrc.Length)
                {
                    outputShapes = new[] { new[] { 1 } }; // scalar as 1D
                    if (node.Outputs.Count > 0)
                        graph.ConstantData[node.Outputs[0]] = new[] { gatherSrc[gIdx] };
                }
            }

            // Compile-time Concat evaluation on known constants
            if (node.OpType == "Concat" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && node.Inputs.All(inp => !string.IsNullOrEmpty(inp) && graph.ConstantData.ContainsKey(inp)))
            {
                var concatVals = node.Inputs.SelectMany(inp => graph.ConstantData[inp]).ToArray();
                outputShapes = new[] { new[] { concatVals.Length } };
                if (node.Outputs.Count > 0)
                    graph.ConstantData[node.Outputs[0]] = concatVals;
            }

            // Unsqueeze on known constants
            if (node.OpType == "Unsqueeze" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var unsqData))
            {
                if (node.Outputs.Count > 0)
                    graph.ConstantData[node.Outputs[0]] = unsqData;
            }

            // Compile-time Slice on known constants: Slice(data, starts, ends[, axes, steps])
            if (node.OpType == "Slice" && node.Inputs.Count >= 3
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var sliceData)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var sliceStarts)
                && graph.ConstantData.TryGetValue(node.Inputs[2], out var sliceEnds))
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
                        graph.ConstantData[node.Outputs[0]] = sliced;
                }
            }

            // Compile-time scalar/element-wise arithmetic on known constants
            if (node.OpType is "Mul" or "Add" or "Sub" or "Div"
                && node.Inputs.Count >= 2
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var arithA)
                && graph.ConstantData.TryGetValue(node.Inputs[1], out var arithB))
            {
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

            // Compile-time Cast on known constants (int values stay the same)
            if (node.OpType == "Cast" && node.Inputs.Count >= 1
                && graph.ConstantData != null
                && graph.ConstantData.TryGetValue(node.Inputs[0], out var castData))
            {
                if (node.Outputs.Count > 0)
                    graph.ConstantData[node.Outputs[0]] = castData;
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
                    // Shape tensor not resolved — log warning for debugging
                    var outName = node.Outputs.Count > 0 ? node.Outputs[0] : "?";
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

            // Register output shapes (override with graph output shapes if known)
            for (int i = 0; i < node.Outputs.Count && i < outputShapes.Length; i++)
            {
                var outName = node.Outputs[i];
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
