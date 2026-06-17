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

        // Normalize scalar initializer shapes: ONNX allows shape [] for scalars, but
        // weight loading and buffer allocation require at least [1].
        // NOTE: This does NOT affect Gather output shape inference — Gather's InferOutputShapes
        // handles single-element [1] indices as scalar (squeezes the extra dimension).
        foreach (var key in optimized.Initializers.Keys.ToList())
        {
            if (optimized.Initializers[key].Length == 0)
                optimized.Initializers[key] = new[] { 1 };
        }
        if (optimized.ConstantData != null)
        {
            foreach (var key in optimized.ConstantData.Keys.ToList())
            {
                if (optimized.ConstantData[key].Length == 0)
                    optimized.ConstantData[key] = new[] { 0 };
            }
        }
        if (optimized.FloatConstantData != null)
        {
            foreach (var key in optimized.FloatConstantData.Keys.ToList())
            {
                if (optimized.FloatConstantData[key].Length == 0)
                    optimized.FloatConstantData[key] = new[] { 0f };
            }
        }

        // Pass 1: Fold constant subgraphs (Shape → Gather → Cast chains become constants)
        int folded = FoldConstants(optimized);

        // Pass 2: Eliminate identity/constant-passthrough nodes
        int eliminated = EliminateIdentityNodes(optimized);

        // Pass 3: Fuse MatMul → Add (bias) → Activation into FusedLinear
        int fusedLinear = FuseLinearLayers(optimized);

        // Pass 3b: Fuse a full decomposed self-attention subgraph (Q·Kᵀ → scale → [+zero-bias] → Softmax →
        // [Cast] → probs·V) into ONE flash-style FusedAttention node — never materializes the [B·H, S, S]
        // scores. Runs BEFORE FuseScaledMatMul so the Q·Kᵀ MatMul is still raw. This is the SD-UNet memory
        // win (the [8,4096,4096] scores at down_block.0 were the pipeline peak).
        int fusedAttn = FuseAttention(optimized);

        // Pass 4: Fuse MatMul → Mul/Div (scale) into FusedScaledMatMul (attention Q*K^T/sqrt(d))
        int fusedScaled = FuseScaledMatMul(optimized);

        // Pass 5: Strength reduction (Div by const → Mul, eliminate Mul by 1, Add by 0)
        int reduced = StrengthReduce(optimized);

        // Pass 6: Re-run identity elimination (strength reduction may create new Identity nodes)
        eliminated += EliminateIdentityNodes(optimized);

        // Pass 7: Remove dead nodes (outputs never consumed)
        int dead = EliminateDeadNodes(optimized);

        int totalOpt = fusedLinear + fusedScaled + fusedAttn + eliminated + dead + folded + reduced;
        if (InferenceSession.VerboseLogging && totalOpt > 0)
            Console.WriteLine($"[GraphOptimizer] {totalOpt} optimizations: {folded} folded, {eliminated} identity, {fusedLinear} fused-linear, {fusedScaled} fused-scaled, {fusedAttn} fused-attention, {reduced} strength-reduced, {dead} dead");

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

            // Do NOT fuse a WEIGHT matmul (B = a graph initializer). FusedScaledMatMul is an F32
            // activation×activation op meant for attention Q·Kᵀ·scale; a quantized weight B (e.g. the
            // gemma tied LM head, Q6_K) would be read as F32 — ~5x past the compressed buffer → illegal
            // memory access. Weight matmuls take the dequant/weight path; the scale stays a separate node.
            if (matmulNode.Inputs.Count >= 2 && graph.Initializers.ContainsKey(matmulNode.Inputs[1])) continue;

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
    /// Fuse a full decomposed self-attention subgraph into ONE <c>FusedAttention</c> node (flash-style online
    /// softmax — the [B·H, seq, seq] scores are never materialized). Matches the standard diffusers/torch ONNX
    /// export shape, anchored at the <c>Softmax</c>:
    ///   K → Transpose ─┐
    ///   Q ────────────→ MatMul(Q·Kᵀ) → Mul(×scale) → [Add(+zero-bias)] → Softmax → [Cast] → MatMul(probs·V) → out
    /// becomes <c>FusedAttention(Q, K, V) → out</c> with attrs causal=0, window=0, scale (read from the scale
    /// constant). Q/K/V are the [B·H, seq, head_dim] tensors feeding the MatMuls (their producing reshapes stay).
    /// The zero-bias branch (ConstantOfShape → Mul × 0 → Add) becomes dead and is removed by dead-node
    /// elimination. The operator derives n_heads/head_dim from the rank-3 Q at runtime, so this is shape-agnostic
    /// (works for every attention block/resolution). Guards: single-consumer on every fused tensor, both QKᵀ
    /// inputs must be activations (not a weight MatMul), K must be transposed. Falls through (no fusion) on any
    /// mismatch — correctness is never at risk, only the memory/perf win.
    /// </summary>
    private static int FuseAttention(ModelGraph graph)
    {
        int fusedCount = 0;
        var nodes = graph.Nodes;

        var producer = new Dictionary<string, int>();
        for (int i = 0; i < nodes.Count; i++)
            foreach (var o in nodes[i].Outputs)
                if (!string.IsNullOrEmpty(o)) producer[o] = i;
        var consumerCount = new Dictionary<string, int>();
        foreach (var n in nodes)
            foreach (var inp in n.Inputs)
                if (!string.IsNullOrEmpty(inp)) consumerCount[inp] = consumerCount.GetValueOrDefault(inp, 0) + 1;

        bool IsActivation(string name) => !string.IsNullOrEmpty(name) && !graph.Initializers.ContainsKey(name);
        int Prod(string name) => producer.TryGetValue(name, out var idx) ? idx : -1;
        int SingleConsumer(string name)
        {
            for (int i = 0; i < nodes.Count; i++) if (nodes[i].Inputs.Contains(name)) return i;
            return -1;
        }
        float? ScalarConst(string name)
        {
            if (graph.FloatConstantData != null && graph.FloatConstantData.TryGetValue(name, out var f) && f.Length >= 1) return f[0];
            if (graph.ConstantData != null && graph.ConstantData.TryGetValue(name, out var d) && d.Length >= 1) return d[0];
            return null;
        }
        // Walk back through Mul/Div/Cast from `name`; true if it reaches a MatMul of two activations (the scores
        // branch) rather than a ConstantOfShape (the zero-bias branch).
        bool ReachesScoresMatMul(string name, int depth)
        {
            if (depth > 4) return false;
            int p = Prod(name);
            if (p < 0) return false;
            var n = nodes[p];
            if (n.OpType == "MatMul" && n.Inputs.Count == 2 && IsActivation(n.Inputs[0]) && IsActivation(n.Inputs[1])) return true;
            if (n.OpType is "Mul" or "Div" or "Cast")
                foreach (var inp in n.Inputs)
                    if (IsActivation(inp) && ReachesScoresMatMul(inp, depth + 1)) return true;
            return false;
        }

        var remove = new HashSet<int>();

        for (int si = 0; si < nodes.Count; si++)
        {
            if (remove.Contains(si)) continue;
            var softmax = nodes[si];
            if (softmax.OpType != "Softmax" || softmax.Inputs.Count < 1 || softmax.Outputs.Count < 1) continue;

            var between = new List<int>();
            string scoresName = softmax.Inputs[0];
            if (consumerCount.GetValueOrDefault(scoresName, 0) != 1) continue;   // scores feed only softmax

            // Optional additive bias: Add(scaledScores, zeroBias). Keep the scores branch, drop the Add.
            int pAdd = Prod(scoresName);
            if (pAdd >= 0 && nodes[pAdd].OpType == "Add" && nodes[pAdd].Inputs.Count == 2)
            {
                var addN = nodes[pAdd];
                string br = ReachesScoresMatMul(addN.Inputs[0], 0) ? addN.Inputs[0]
                          : ReachesScoresMatMul(addN.Inputs[1], 0) ? addN.Inputs[1] : null!;
                if (br == null) continue;
                if (consumerCount.GetValueOrDefault(br, 0) != 1) continue;
                between.Add(pAdd);
                scoresName = br;
            }

            // Scale: Mul/Div by a scalar const on the QKᵀ output. (Absent → kernel default 1/sqrt(head_dim).)
            float scale = 0f;
            string matmulOut = scoresName;
            int pScale = Prod(scoresName);
            if (pScale >= 0 && (nodes[pScale].OpType == "Mul" || nodes[pScale].OpType == "Div") && nodes[pScale].Inputs.Count == 2)
            {
                var sN = nodes[pScale];
                float? sc = ScalarConst(sN.Inputs[1]); string actIn = sN.Inputs[0];
                if (sc == null) { sc = ScalarConst(sN.Inputs[0]); actIn = sN.Inputs[1]; }
                if (sc != null && IsActivation(actIn))
                {
                    if (consumerCount.GetValueOrDefault(actIn, 0) != 1) continue;
                    scale = sN.OpType == "Div" ? 1f / sc.Value : sc.Value;
                    matmulOut = actIn;
                    between.Add(pScale);
                }
            }

            // Scores MatMul (Q · Kᵀ), both activations.
            int pMM = Prod(matmulOut);
            if (pMM < 0 || nodes[pMM].OpType != "MatMul" || nodes[pMM].Inputs.Count != 2) continue;
            var mm = nodes[pMM];
            string qName = mm.Inputs[0], kTName = mm.Inputs[1];
            if (!IsActivation(qName) || !IsActivation(kTName)) continue;
            if (consumerCount.GetValueOrDefault(kTName, 0) != 1) continue;
            between.Add(pMM);

            // K transpose.
            int pKT = Prod(kTName);
            if (pKT < 0 || nodes[pKT].OpType != "Transpose" || nodes[pKT].Inputs.Count < 1) continue;
            string kName = nodes[pKT].Inputs[0];
            between.Add(pKT);

            // Forward: softmax → [Cast] → MatMul(probs, V).
            string probs = softmax.Outputs[0];
            if (consumerCount.GetValueOrDefault(probs, 0) != 1) continue;
            int cIdx = SingleConsumer(probs);
            if (cIdx < 0) continue;
            if (nodes[cIdx].OpType == "Cast")
            {
                between.Add(cIdx);
                probs = nodes[cIdx].Outputs[0];
                if (consumerCount.GetValueOrDefault(probs, 0) != 1) continue;
                cIdx = SingleConsumer(probs);
                if (cIdx < 0) continue;
            }
            var av = nodes[cIdx];
            if (av.OpType != "MatMul" || av.Inputs.Count != 2) continue;
            string vName = av.Inputs[0] == probs ? av.Inputs[1] : av.Inputs[0];
            if (!IsActivation(vName)) continue;
            string attnOut = av.Outputs[0];

            // Replace the probs·V MatMul (produces the kept output) with FusedAttention; remove the rest.
            nodes[cIdx] = new GraphNode
            {
                OpType = "FusedAttention",
                Inputs = new List<string> { qName, kName, vName },
                Outputs = new List<string> { attnOut },
                Attributes = new Dictionary<string, JsonElement>
                {
                    ["causal"] = JsonSerializer.SerializeToElement(0),
                    ["window"] = JsonSerializer.SerializeToElement(0),
                    ["scale"] = JsonSerializer.SerializeToElement(scale),
                }
            };
            remove.Add(si);
            foreach (var b in between) remove.Add(b);
            fusedCount++;
        }

        foreach (var idx in remove.OrderByDescending(i => i)) nodes.RemoveAt(idx);
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
                        graph.FloatConstantData ??= new Dictionary<string, float[]>();
                        graph.FloatConstantData[outputName] = shapeValues.Select(v => (float)v).ToArray();
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
                            graph.FloatConstantData ??= new Dictionary<string, float[]>();
                            graph.FloatConstantData[outputName] = new[] { (float)gatherData[idx] };
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
                        graph.FloatConstantData ??= new Dictionary<string, float[]>();
                        graph.FloatConstantData[outputName] = concatValues.Select(v => (float)v).ToArray();
                        graph.Initializers[outputName] = new[] { concatValues.Length };
                        knownShapes[outputName] = new[] { concatValues.Length };
                        constants.Add(outputName);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                        continue;
                    }

                    // Try to evaluate Cast on known constant data (identity for shape tensors)
                    if (node.OpType == "Cast" && node.Inputs.Count >= 1
                        && graph.ConstantData.TryGetValue(node.Inputs[0], out var castData))
                    {
                        var outputName = node.Outputs[0];
                        // Cast preserves values for shape tensors (int→float or float→int is identity for small ints)
                        graph.ConstantData[outputName] = castData.ToArray();
                        graph.FloatConstantData ??= new Dictionary<string, float[]>();
                        graph.FloatConstantData[outputName] = castData.Select(v => (float)v).ToArray();
                        graph.Initializers[outputName] = new[] { castData.Length };
                        knownShapes[outputName] = new[] { castData.Length };
                        constants.Add(outputName);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                        continue;
                    }

                    // Try to evaluate Sqrt on known constant data
                    if (node.OpType == "Sqrt" && node.Inputs.Count >= 1
                        && graph.ConstantData.TryGetValue(node.Inputs[0], out var sqrtData))
                    {
                        var outputName = node.Outputs[0];
                        var result = sqrtData.Select(v => (int)MathF.Sqrt(v)).ToArray();
                        graph.ConstantData[outputName] = result;
                        graph.FloatConstantData ??= new Dictionary<string, float[]>();
                        graph.FloatConstantData[outputName] = sqrtData.Select(v => MathF.Sqrt(v)).ToArray();
                        graph.Initializers[outputName] = new[] { result.Length };
                        knownShapes[outputName] = new[] { result.Length };
                        constants.Add(outputName);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                        continue;
                    }

                    // Try to evaluate Slice on known constant data
                    if (node.OpType == "Slice" && node.Inputs.Count >= 3
                        && graph.ConstantData.TryGetValue(node.Inputs[0], out var sliceData)
                        && graph.ConstantData.TryGetValue(node.Inputs[1], out var sliceStarts)
                        && graph.ConstantData.TryGetValue(node.Inputs[2], out var sliceEnds))
                    {
                        var outputName = node.Outputs[0];
                        int[] axes = node.Inputs.Count > 3 && graph.ConstantData.TryGetValue(node.Inputs[3], out var sa)
                            ? sa : Enumerable.Range(0, sliceStarts.Length).ToArray();
                        int[] steps = node.Inputs.Count > 4 && graph.ConstantData.TryGetValue(node.Inputs[4], out var ss)
                            ? ss : Enumerable.Repeat(1, sliceStarts.Length).ToArray();

                        // Compute sliced result (1D case — typical for shape tensors)
                        if (axes.Length == 1)
                        {
                            int ax = axes[0] < 0 ? axes[0] + 1 : axes[0]; // 1D → axis always 0
                            int s = sliceStarts[0]; int e = sliceEnds[0]; int st = Math.Max(1, Math.Abs(steps[0]));
                            if (s < 0) s += sliceData.Length;
                            if (e < 0) e += sliceData.Length;
                            // Clamp ends to INT_MAX → data length
                            if (e > sliceData.Length) e = sliceData.Length;
                            s = Math.Clamp(s, 0, sliceData.Length);
                            e = Math.Clamp(e, 0, sliceData.Length);
                            var sliced = new List<int>();
                            for (int si = s; si < e; si += st)
                                sliced.Add(sliceData[si]);
                            graph.ConstantData[outputName] = sliced.ToArray();
                            graph.FloatConstantData ??= new Dictionary<string, float[]>();
                            graph.FloatConstantData[outputName] = sliced.Select(v => (float)v).ToArray();
                            graph.Initializers[outputName] = new[] { sliced.Count };
                            knownShapes[outputName] = new[] { sliced.Count };
                            constants.Add(outputName);
                            nodesToRemove.Add(i);
                            folded++;
                            changed = true;
                            continue;
                        }
                    }

                    // Generic folding: ONLY for identity-like ops that pass through data unchanged.
                    // Arithmetic ops (Add, Sub, Mul, Div, Sqrt, etc.) MUST NOT blindly propagate
                    // input data — they transform values. Wrong constants cascade through shape
                    // inference chains and cause DivideByZero (e.g., DepthAnything Softmax).
                    bool isIdentityLike = node.OpType is "Unsqueeze" or "Squeeze" or "Reshape"
                        or "Identity" or "Flatten" or "Dropout";
                    if (isIdentityLike && node.Inputs.Count >= 1
                        && graph.ConstantData.TryGetValue(node.Inputs[0], out var genData))
                    {
                        bool hasLargeInput = genData.Length > 64;
                        if (hasLargeInput) continue; // Don't fold large tensors

                        var outputName = node.Outputs.Count > 0 ? node.Outputs[0] : null;
                        if (outputName != null)
                        {
                            // Propagate constant data. Preserve existing initializer shapes
                            // (don't override scalar [] with [1] — breaks Gather index dimensions).
                            graph.ConstantData[outputName] = genData.ToArray();
                            graph.FloatConstantData ??= new Dictionary<string, float[]>();
                            graph.FloatConstantData[outputName] = genData.Select(v => (float)v).ToArray();
                            if (!graph.Initializers.ContainsKey(outputName))
                                graph.Initializers[outputName] = new[] { genData.Length };
                            // Don't override known shapes — preserve scalar [] vs [1] distinction
                            if (!knownShapes.ContainsKey(outputName))
                                knownShapes[outputName] = new[] { genData.Length };
                        }
                        foreach (var output in node.Outputs)
                            constants.Add(output);
                        nodesToRemove.Add(i);
                        folded++;
                        changed = true;
                    }
                    // If no constant data available, don't fold — leave for runtime execution
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
        // Shape-manipulation ops that produce small int tensors — safe to fold.
        // Ops with explicit handlers (Shape, Gather, Concat, Slice) compute results correctly.
        // Generic folding handles the rest (Cast, Floor, Ceil, etc. — identity for shape tensors).
        "Shape" or "Gather" or "GatherND" or "Cast" or "Floor" or "Ceil" or
        "Unsqueeze" or "Squeeze" or "Concat" or "Reshape" or "Slice" or
        "Add" or "Sub" or "Mul" or "Div" or "Neg" or "Abs" or "Sqrt" or
        "Identity";
        // NOT foldable: Range, ConstantOfShape, Expand, Equal, Greater, Less, Not, Where
        // These produce large runtime tensors (attention masks, fill tensors, boolean masks)
        // that generic folding can't evaluate — it registers them as shape [1], destroying
        // downstream shape inference. NLP models (DistilBERT, GPT-2) crash without this fix.

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
        int totalEliminated = 0;

        // Iterate to a fixpoint: removing a dead node can make its (now-unconsumed) producer dead too. A single
        // pass only catches one layer — e.g. fusing attention leaves Add dead; pass 1 removes Add, which leaves
        // Mul dead; pass 2 removes Mul, which leaves the [B·H,S,S] ConstantOfShape zero-bias dead; pass 3 removes
        // THAT (a 512 MiB buffer per attention block). Looping collects the whole transitive dead chain.
        bool changed = true;
        while (changed)
        {
            changed = false;
            var consumedOutputs = new HashSet<string>(graphOutputNames);
            foreach (var node in graph.Nodes)
                foreach (var input in node.Inputs)
                    if (!string.IsNullOrEmpty(input))
                        consumedOutputs.Add(input);

            var nodesToRemove = new List<int>();
            for (int i = 0; i < graph.Nodes.Count; i++)
                if (!graph.Nodes[i].Outputs.Any(o => consumedOutputs.Contains(o)))
                    nodesToRemove.Add(i);

            foreach (var idx in nodesToRemove.OrderByDescending(i => i))
                graph.Nodes.RemoveAt(idx);
            if (nodesToRemove.Count > 0) { changed = true; totalEliminated += nodesToRemove.Count; }
        }

        return totalEliminated;
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
            // Without this copy the GraphExecutor's BuildIntegerTensorNames sees
            // a null InitializerDataTypes after optimization, which silently breaks
            // ONNX integer-Div trunc semantics (MoveNet keypoint X-coord regression).
            InitializerDataTypes = src.InitializerDataTypes != null
                ? new Dictionary<string, int>(src.InitializerDataTypes)
                : null,
        };
    }
}
