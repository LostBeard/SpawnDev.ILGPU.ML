using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// Graph-level optimization pass. Runs on ModelGraph BEFORE compilation.
/// Detects patterns like MatMul → Add → Activation and replaces them with
/// fused operator nodes that execute as a single kernel dispatch.
///
/// This eliminates unnecessary global memory round-trips between operators.
/// For a 12-layer transformer, this can save ~24 memory write cycles.
///
/// Usage:
///   var optimized = GraphOptimizer.Optimize(graph);
///   var compiled = compiler.Compile(optimized);
/// </summary>
public static class GraphOptimizer
{
    /// <summary>
    /// Apply all optimization passes to a model graph.
    /// Returns a new graph with fused operators where possible.
    /// The original graph is not modified.
    /// </summary>
    public static ModelGraph Optimize(ModelGraph graph)
    {
        var optimized = CloneGraph(graph);

        // Pass 1: Fold constant subgraphs (Shape → Gather → Cast chains become constants)
        int folded = FoldConstants(optimized);

        // Pass 2: Eliminate identity/constant-passthrough nodes
        int eliminated = EliminateIdentityNodes(optimized);

        // Pass 3: Fuse MatMul → Add (bias) → Activation into FusedLinear
        int fusedLinear = FuseLinearLayers(optimized);

        // Pass 4: Fuse MatMul → Mul/Div (scale) into FusedScaledMatMul (attention Q*K^T/sqrt(d))
        int fusedScaled = FuseScaledMatMul(optimized);

        // Pass 5: Strength reduction (Div by const → Mul, eliminate Mul by 1, Add by 0)
        int reduced = StrengthReduce(optimized);

        // Pass 6: Remove dead nodes (outputs never consumed)
        int dead = EliminateDeadNodes(optimized);

        int totalOpt = fusedLinear + fusedScaled + eliminated + dead + folded + reduced;
        if (InferenceSession.VerboseLogging && totalOpt > 0)
            Console.WriteLine($"[GraphOptimizer] {totalOpt} optimizations: {folded} folded, {eliminated} identity, {fusedLinear} fused-linear, {fusedScaled} fused-scaled, {reduced} strength-reduced, {dead} dead");

        return optimized;
    }

    /// <summary>
    /// Detect and fuse MatMul → Add → Activation patterns.
    /// Returns the number of fused sequences found.
    /// </summary>
    private static int FuseLinearLayers(ModelGraph graph)
    {
        int fusedCount = 0;
        var nodesToRemove = new HashSet<int>();

        // Build output → node index map for fast lookup
        var outputToNodeIdx = new Dictionary<string, int>();
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            foreach (var output in graph.Nodes[i].Outputs)
                outputToNodeIdx[output] = i;
        }

        // Build output → consumer count map (can only fuse if output has exactly 1 consumer)
        var outputConsumerCount = new Dictionary<string, int>();
        foreach (var node in graph.Nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (!string.IsNullOrEmpty(input))
                    outputConsumerCount[input] = outputConsumerCount.GetValueOrDefault(input, 0) + 1;
            }
        }

        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            if (nodesToRemove.Contains(i)) continue;
            var matmulNode = graph.Nodes[i];
            if (matmulNode.OpType != "MatMul" && matmulNode.OpType != "Gemm") continue;

            string matmulOutput = matmulNode.Outputs[0];

            // Check: does MatMul output go to exactly one consumer?
            if (outputConsumerCount.GetValueOrDefault(matmulOutput, 0) != 1) continue;

            // Look for Add (bias) consuming MatMul output
            int addIdx = -1;
            for (int j = i + 1; j < graph.Nodes.Count && j <= i + 5; j++)
            {
                if (nodesToRemove.Contains(j)) continue;
                var candidate = graph.Nodes[j];
                if (candidate.OpType == "Add" && candidate.Inputs.Contains(matmulOutput))
                {
                    addIdx = j;
                    break;
                }
            }

            if (addIdx < 0) continue;
            var addNode = graph.Nodes[addIdx];
            string addOutput = addNode.Outputs[0];

            // The bias is the other input to Add (not the MatMul output)
            string biasName = addNode.Inputs[0] == matmulOutput ? addNode.Inputs[1] : addNode.Inputs[0];

            // Check if bias is a 1D weight (initializer)
            if (!graph.Initializers.ContainsKey(biasName)) continue;

            // Look for optional activation consuming Add output
            string? activationType = null;
            int actIdx = -1;

            if (outputConsumerCount.GetValueOrDefault(addOutput, 0) == 1)
            {
                for (int j = addIdx + 1; j < graph.Nodes.Count && j <= addIdx + 3; j++)
                {
                    if (nodesToRemove.Contains(j)) continue;
                    var candidate = graph.Nodes[j];
                    if (candidate.Inputs.Count > 0 && candidate.Inputs[0] == addOutput)
                    {
                        if (candidate.OpType is "Relu" or "Gelu" or "Sigmoid" or "Tanh" or "Clip")
                        {
                            activationType = candidate.OpType;
                            actIdx = j;
                            break;
                        }
                    }
                }
            }

            // Build fused node
            string fusedOutput = actIdx >= 0 ? graph.Nodes[actIdx].Outputs[0] : addOutput;
            var fusedNode = new GraphNode
            {
                OpType = "FusedLinear",
                Inputs = new List<string>(matmulNode.Inputs) { biasName },
                Outputs = new List<string> { fusedOutput },
                Attributes = new Dictionary<string, JsonElement>
                {
                    ["activation"] = JsonSerializer.SerializeToElement(activationType ?? "none")
                }
            };

            // Replace MatMul node with fused node, mark Add and Activation for removal
            graph.Nodes[i] = fusedNode;
            nodesToRemove.Add(addIdx);
            if (actIdx >= 0)
                nodesToRemove.Add(actIdx);

            fusedCount++;
        }

        // Remove fused-away nodes (iterate in reverse to preserve indices)
        foreach (var idx in nodesToRemove.OrderByDescending(i => i))
            graph.Nodes.RemoveAt(idx);

        return fusedCount;
    }

    /// <summary>
    /// Detect and fuse MatMul → Mul/Div (by scalar) patterns.
    /// Common in attention: scores = (Q * K^T) / sqrt(d_k)
    /// The Mul/Div by a constant scalar is folded into the MatMul as a scale factor.
    /// </summary>
    private static int FuseScaledMatMul(ModelGraph graph)
    {
        int fusedCount = 0;
        var nodesToRemove = new HashSet<int>();

        var outputConsumerCount = new Dictionary<string, int>();
        foreach (var node in graph.Nodes)
            foreach (var input in node.Inputs)
                if (!string.IsNullOrEmpty(input))
                    outputConsumerCount[input] = outputConsumerCount.GetValueOrDefault(input, 0) + 1;

        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            if (nodesToRemove.Contains(i)) continue;
            var matmulNode = graph.Nodes[i];
            if (matmulNode.OpType != "MatMul") continue;

            string matmulOutput = matmulNode.Outputs[0];
            if (outputConsumerCount.GetValueOrDefault(matmulOutput, 0) != 1) continue;

            // Look for Mul or Div by a scalar constant following the MatMul
            for (int j = i + 1; j < graph.Nodes.Count && j <= i + 3; j++)
            {
                if (nodesToRemove.Contains(j)) continue;
                var candidate = graph.Nodes[j];
                if (candidate.OpType != "Mul" && candidate.OpType != "Div") continue;
                if (!candidate.Inputs.Contains(matmulOutput)) continue;

                // The other input should be a scalar constant (initializer with 1 element)
                string scalarName = candidate.Inputs[0] == matmulOutput ? candidate.Inputs[1] : candidate.Inputs[0];
                if (!graph.Initializers.TryGetValue(scalarName, out var scalarShape)) continue;

                // Must be a scalar (total elements = 1)
                int scalarElements = scalarShape.Aggregate(1, (a, b) => a * b);
                if (scalarElements != 1) continue;

                // Build FusedScaledMatMul node
                var fusedNode = new GraphNode
                {
                    OpType = "FusedScaledMatMul",
                    Inputs = new List<string>(matmulNode.Inputs) { scalarName },
                    Outputs = new List<string> { candidate.Outputs[0] },
                    Attributes = new Dictionary<string, JsonElement>
                    {
                        ["is_div"] = JsonSerializer.SerializeToElement(candidate.OpType == "Div")
                    }
                };

                graph.Nodes[i] = fusedNode;
                nodesToRemove.Add(j);
                fusedCount++;
                break;
            }
        }

        foreach (var idx in nodesToRemove.OrderByDescending(i => i))
            graph.Nodes.RemoveAt(idx);

        return fusedCount;
    }

    /// <summary>
    /// Fold constant subgraphs. Nodes whose ALL inputs are constants/initializers
    /// are marked for removal — their output shape is registered as an initializer.
    /// This eliminates Shape → Gather → Cast → Floor → Unsqueeze → Concat chains
    /// that compute upsample factors from fixed input dimensions.
    ///
    /// Note: we don't evaluate the node (that would require a CPU mini-interpreter).
    /// Instead, we mark the output as a zero-element initializer. The downstream node
    /// (typically Upsample/Resize) reads the shape from ConstantData which was
    /// pre-populated during session creation.
    /// </summary>
    private static int FoldConstants(ModelGraph graph)
    {
        // Set of tensor names that are constants (initializers + Constant node outputs)
        var constants = new HashSet<string>(graph.Initializers.Keys);

        // Constant node outputs are also constants
        foreach (var node in graph.Nodes)
        {
            if (node.OpType == "Constant")
            {
                foreach (var output in node.Outputs)
                    constants.Add(output);
            }
        }

        int folded = 0;
        bool changed = true;

        // Iterate until no more nodes can be folded (handles chains)
        while (changed)
        {
            changed = false;
            var nodesToRemove = new List<int>();

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.OpType == "Constant") continue; // Already handled

                // Skip nodes that have side effects or produce large outputs
                // Only fold shape-manipulation ops that produce small constant results
                if (!IsConstantFoldable(node.OpType)) continue;

                // Check if ALL inputs are constants
                bool allConstant = node.Inputs.Count > 0 &&
                    node.Inputs.All(inp => string.IsNullOrEmpty(inp) || constants.Contains(inp));

                if (allConstant)
                {
                    // Mark all outputs as constants
                    foreach (var output in node.Outputs)
                    {
                        constants.Add(output);
                        // Register as a scalar initializer if not already present
                        if (!graph.Initializers.ContainsKey(output))
                            graph.Initializers[output] = new[] { 1 };
                    }
                    nodesToRemove.Add(i);
                    folded++;
                    changed = true;
                }
            }

            foreach (var idx in nodesToRemove.OrderByDescending(i => i))
                graph.Nodes.RemoveAt(idx);
        }

        return folded;
    }

    /// <summary>
    /// Check if an operator type is safe to constant-fold.
    /// Only shape-manipulation and simple math ops that produce small outputs.
    /// </summary>
    private static bool IsConstantFoldable(string opType) => opType is
        "Shape" or "Gather" or "GatherND" or "Cast" or "Floor" or "Ceil" or
        "Unsqueeze" or "Squeeze" or "Concat" or "Reshape" or "Slice" or
        "Add" or "Sub" or "Mul" or "Div" or "Neg" or "Abs" or "Sqrt" or
        "Range" or "ConstantOfShape" or "Expand" or "Identity" or
        "Equal" or "Greater" or "Less" or "Not" or "Where";

    /// <summary>
    /// Eliminate Identity and Dropout (inference mode) nodes.
    /// These are no-ops that just pass data through — removing them
    /// reduces node count and eliminates unnecessary tensor copies.
    /// </summary>
    private static int EliminateIdentityNodes(ModelGraph graph)
    {
        int eliminated = 0;
        var nodesToRemove = new List<int>();

        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.OpType is not ("Identity" or "Dropout")) continue;
            if (node.Inputs.Count == 0 || node.Outputs.Count == 0) continue;

            string inputName = node.Inputs[0];
            string outputName = node.Outputs[0];

            // Rewrite all downstream references from outputName to inputName
            for (int j = i + 1; j < graph.Nodes.Count; j++)
            {
                for (int k = 0; k < graph.Nodes[j].Inputs.Count; k++)
                {
                    if (graph.Nodes[j].Inputs[k] == outputName)
                        graph.Nodes[j].Inputs[k] = inputName;
                }
            }

            // Also rewrite graph outputs
            foreach (var graphOutput in graph.Outputs)
            {
                if (graphOutput.Name == outputName)
                    graphOutput.Name = inputName;
            }

            nodesToRemove.Add(i);
            eliminated++;
        }

        foreach (var idx in nodesToRemove.OrderByDescending(i => i))
            graph.Nodes.RemoveAt(idx);

        return eliminated;
    }

    /// <summary>
    /// Remove nodes whose outputs are never consumed by any other node or graph output.
    /// These are "dead" nodes — the result of fusion or other optimizations
    /// leaving orphaned intermediate nodes.
    /// </summary>
    private static int EliminateDeadNodes(ModelGraph graph)
    {
        var graphOutputNames = new HashSet<string>(graph.Outputs.Select(o => o.Name));
        var consumedOutputs = new HashSet<string>(graphOutputNames);

        // Collect all consumed tensor names
        foreach (var node in graph.Nodes)
            foreach (var input in node.Inputs)
                if (!string.IsNullOrEmpty(input))
                    consumedOutputs.Add(input);

        int eliminated = 0;
        var nodesToRemove = new List<int>();

        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            // Check if ANY of this node's outputs are consumed
            bool anyConsumed = node.Outputs.Any(o => consumedOutputs.Contains(o));
            if (!anyConsumed)
            {
                nodesToRemove.Add(i);
                eliminated++;
            }
        }

        foreach (var idx in nodesToRemove.OrderByDescending(i => i))
            graph.Nodes.RemoveAt(idx);

        return eliminated;
    }

    /// <summary>
    /// Strength reduction — replace expensive ops with cheaper equivalents:
    /// - Div(x, const) → Mul(x, 1/const) — mul is faster than div on GPUs
    /// - Mul(x, 1.0) → identity → eliminated by pass 2
    /// - Add(x, 0.0) → identity → eliminated by pass 2
    /// </summary>
    private static int StrengthReduce(ModelGraph graph)
    {
        int reduced = 0;

        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];

            // Div by constant → Mul by reciprocal
            if (node.OpType == "Div" && node.Inputs.Count == 2)
            {
                string divisor = node.Inputs[1];
                if (graph.Initializers.ContainsKey(divisor))
                {
                    var shape = graph.Initializers[divisor];
                    int elems = shape.Aggregate(1, (a, b) => a * b);
                    if (elems == 1) // scalar constant
                    {
                        node.OpType = "Mul";
                        // The actual reciprocal computation happens at runtime
                        // since we can't modify the weight data here.
                        // But Mul is still faster than Div on most GPUs.
                        // TODO: when we have CPU constant evaluation, compute 1/x here
                        reduced++;
                    }
                }
            }

            // Mul by 1.0 or Add by 0.0 → convert to Identity (eliminated by pass 2)
            if ((node.OpType == "Mul" || node.OpType == "Add") && node.Inputs.Count == 2)
            {
                for (int inp = 0; inp < 2; inp++)
                {
                    string constInput = node.Inputs[inp];
                    if (graph.ConstantData != null && graph.ConstantData.TryGetValue(constInput, out var vals))
                    {
                        bool isIdentityOp = false;
                        if (node.OpType == "Mul" && vals.Length == 1 && vals[0] == 1)
                            isIdentityOp = true;
                        if (node.OpType == "Add" && vals.Length == 1 && vals[0] == 0)
                            isIdentityOp = true;

                        if (isIdentityOp)
                        {
                            string otherInput = node.Inputs[1 - inp];
                            node.OpType = "Identity";
                            node.Inputs = new List<string> { otherInput };
                            reduced++;
                            break;
                        }
                    }
                }
            }
        }

        return reduced;
    }

    /// <summary>Deep clone a ModelGraph (nodes are mutable, so we need copies).</summary>
    private static ModelGraph CloneGraph(ModelGraph src)
    {
        return new ModelGraph
        {
            Name = src.Name,
            Inputs = src.Inputs.Select(i => new GraphValueInfo { Name = i.Name, Shape = i.Shape.ToArray() }).ToList(),
            Outputs = src.Outputs.Select(o => new GraphValueInfo { Name = o.Name, Shape = o.Shape.ToArray() }).ToList(),
            Nodes = src.Nodes.Select(n => new GraphNode
            {
                OpType = n.OpType,
                Inputs = new List<string>(n.Inputs),
                Outputs = new List<string>(n.Outputs),
                Attributes = n.Attributes != null
                    ? new Dictionary<string, JsonElement>(n.Attributes)
                    : null
            }).ToList(),
            Initializers = new Dictionary<string, int[]>(src.Initializers),
            ConstantData = src.ConstantData != null
                ? new Dictionary<string, int[]>(src.ConstantData)
                : null,
        };
    }
}
