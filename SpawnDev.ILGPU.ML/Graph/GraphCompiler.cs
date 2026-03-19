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
    public CompiledGraph Compile(ModelGraph graph)
    {
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

            // Special-case: Reshape needs the actual shape tensor values.
            // If the second input is a known initializer with shape [N], the output
            // has N dimensions. The Transpose perm length must match this rank.
            // We can infer the output shape from the initializer's known data
            // by reading it from the graph's initializer constants.
            if (node.OpType == "Reshape" && node.Inputs.Count >= 2)
            {
                var shapeTensorName = node.Inputs[1];
                if (knownShapes.TryGetValue(shapeTensorName, out var shapeTensorShape)
                    && shapeTensorShape.Length == 1)
                {
                    int outRank = shapeTensorShape[0];
                    // We don't have the actual values here, but we know the output rank.
                    // If the graph provides initializer values via the ConstantData dict, use them.
                    if (graph.ConstantData != null && graph.ConstantData.TryGetValue(shapeTensorName, out var targetDims))
                    {
                        // targetDims contains the actual reshape target values (may have -1)
                        int inputElems = inputShapes[0].Aggregate(1, (a, b) => a * b);
                        var outShape = targetDims.ToArray();
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
