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

        // Pass 6: Re-run identity elimination (strength reduction may create new Identity nodes)
        eliminated += EliminateIdentityNodes(optimized);

        // Pass 7: Remove dead nodes (outputs never consumed)
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

            // Don't fuse Gemm nodes with transB=1 or non-default alpha/beta —
            // FusedLinearKernel assumes [K,N] weight layout, but transB=1 stores as [N,K]
            if (matmulNode.OpType == "Gemm" && matmulNode.Attributes != null)
            {
                bool hasTransB = matmulNode.Attributes.TryGetValue("transB", out var transB)
                    && transB.ValueKind == System.Text.Json.JsonValueKind.Number && transB.GetInt32() != 0;
                bool hasAlpha = matmulNode.Attributes.TryGetValue("alpha", out var alpha)
                    && alpha.ValueKind == System.Text.Json.JsonValueKind.Number && alpha.GetSingle() != 1.0f;
                bool hasBeta = matmulNode.Attributes.TryGetValue("beta", out var beta)
                    && beta.ValueKind == System.Text.Json.JsonValueKind.Number && beta.GetSingle() != 1.0f;
                if (hasTransB || hasAlpha || hasBeta) continue;
            }

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
        // Track known shapes for Shape node evaluation
        var knownShapes = new Dictionary<string, int[]>();
        foreach (var (name, shape) in graph.Initializers)
            knownShapes[name] = shape;
        foreach (var input in graph.Inputs)
            if (input.Shape != null && input.Shape.Length > 0)
                knownShapes[input.Name] = input.Shape;
        // Track known constant values (for Shape→Gather→Concat evaluation)
        graph.ConstantData ??= new Dictionary<string, int[]>();

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
                if (node.OpType == "Constant") continue;
                if (!IsConstantFoldable(node.OpType)) continue;

                bool allConstant = node.Inputs.Count > 0 &&
                    node.Inputs.All(inp => string.IsNullOrEmpty(inp) || constants.Contains(inp));

                if (allConstant)
                {
                    // Try to evaluate Shape nodes to produce actual constant values
                    if (node.OpType == "Shape" && node.Inputs.Count >= 1
                        && knownShapes.TryGetValue(node.Inputs[0], out var inputShape))
                    {
                        var outputName = node.Outputs[0];
                        var shapeValues = inputShape;
                        graph.ConstantData[outputName] = shapeValues;
                        graph.Initializers[outputName] = new[] { shapeValues.Length };
                        knownShapes[outputName] = new[] { shapeValues.Length };
                        constants.Add(outputName);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                        continue;
                    }

                    // Try to evaluate Gather(axis=0) on known constant data
                    if (node.OpType == "Gather" && node.Inputs.Count >= 2
                        && graph.ConstantData.TryGetValue(node.Inputs[0], out var gatherData)
                        && graph.ConstantData.TryGetValue(node.Inputs[1], out var gatherIdx)
                        && gatherIdx.Length == 1)
                    {
                        var outputName = node.Outputs[0];
                        int idx = gatherIdx[0];
                        if (idx < 0) idx += gatherData.Length;
                        if (idx >= 0 && idx < gatherData.Length)
                        {
                            graph.ConstantData[outputName] = new[] { gatherData[idx] };
                            graph.Initializers[outputName] = new[] { 1 };
                            knownShapes[outputName] = new[] { 1 };
                            constants.Add(outputName);
                            nodesToRemove.Add(i);
                            folded++;
                            changed = true;
                            continue;
                        }
                    }

                    // Try to evaluate Concat on known constant data
                    if (node.OpType == "Concat" && node.Inputs.Count >= 1
                        && node.Inputs.All(inp => graph.ConstantData.ContainsKey(inp)))
                    {
                        var outputName = node.Outputs[0];
                        var concatValues = node.Inputs.SelectMany(inp => graph.ConstantData[inp]).ToArray();
                        graph.ConstantData[outputName] = concatValues;
                        graph.Initializers[outputName] = new[] { concatValues.Length };
                        knownShapes[outputName] = new[] { concatValues.Length };
                        constants.Add(outputName);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                        continue;
                    }

                    // Generic folding: only fold small shape-computation nodes.
                    // Don't fold nodes that consume large constant tensors (anchor grids,
                    // stride multipliers) — we can't evaluate them at compile time, and
                    // registering outputs as shape [1] with no data produces zeros at runtime.
                    bool hasLargeInput = false;
                    foreach (var inp in node.Inputs)
                    {
                        if (string.IsNullOrEmpty(inp)) continue;
                        if (graph.Initializers.TryGetValue(inp, out var inpShape))
                        {
                            int inpSize = inpShape.Aggregate(1, (a, b) => a * b);
                            if (inpSize > 64) { hasLargeInput = true; break; }
                        }
                    }
                    if (hasLargeInput) continue; // Don't fold — large tensor inputs can't be evaluated

                    foreach (var output in node.Outputs)
                    {
                        constants.Add(output);
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

            // NOTE: Div→Mul strength reduction disabled.
            // The optimizer creates a copy of the graph, so new initializers added here
            // don't reach the InferenceSession's weight upload path. Div executes correctly
            // at runtime on all GPU backends. When CPU constant evaluation is implemented
            // in the optimizer, this can be re-enabled properly.

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
            FloatConstantData = src.FloatConstantData != null
                ? new Dictionary<string, float[]>(src.FloatConstantData)
                : null,
        };
    }
}
