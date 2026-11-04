using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.Training;

/// <summary>A LoRA adapter on one projection, as added to a graph by <see cref="GraphAutograd.AddLora"/>.</summary>
/// <param name="PeftName">The PEFT module path, e.g. <c>base_model.model.model.layers.0.self_attn.q_proj</c>.</param>
/// <param name="AName">Graph initializer holding A TRANSPOSED: [in, rank] (PEFT stores lora_A as [rank, in]).</param>
/// <param name="BName">Graph initializer holding B TRANSPOSED: [rank, out] (PEFT stores lora_B as [out, rank]).</param>
public sealed record LoraAdapter(string PeftName, string AName, string BName, int In, int Out, int Rank);

/// <summary>What <see cref="GraphAutograd.AddBackward"/> added: the loss output, the labels input, and each parameter's gradient output.</summary>
public sealed record TrainingGraphInfo(string LossOutput, string LabelsInput, IReadOnlyDictionary<string, string> GradientOutputs);

/// <summary>
/// Reverse-mode automatic differentiation as a GRAPH REWRITE (Plans/training-autograd-lora-2026-11-04.md, M1.2): the backward pass is
/// appended to the forward <see cref="ModelGraph"/> as ordinary nodes, so it runs on the same executor, pool and kernels as inference,
/// on every backend. Rules cover the op set GGUFGraphBuilder emits for llama-family decoders; an op on the gradient path without a rule
/// throws rather than being silently skipped.
/// </summary>
public static class GraphAutograd
{
    // GGUF tensor suffix -> PEFT module path under base_model.model.model.layers.{L}.
    private static readonly Dictionary<string, string> PeftModule = new()
    {
        ["attn_q"] = "self_attn.q_proj", ["attn_k"] = "self_attn.k_proj", ["attn_v"] = "self_attn.v_proj", ["attn_output"] = "self_attn.o_proj",
        ["ffn_gate"] = "mlp.gate_proj", ["ffn_up"] = "mlp.up_proj", ["ffn_down"] = "mlp.down_proj",
    };

    /// <summary>
    /// Add LoRA adapters beside the MatMuls whose weight is <c>blk.{L}.{target}.weight</c> for each target in <paramref name="targets"/>
    /// (GGUF names: attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down): <c>y = x W + (alpha / rank) * (x A^T) B^T</c>, PEFT's
    /// semantics. A^T / B^T become f32 initializers in <paramref name="weights"/>: from <paramref name="initial"/> when it holds the PEFT tensors
    /// (<c>{peft}.lora_A.weight</c> [rank, in], <c>{peft}.lora_B.weight</c> [out, rank]), otherwise PEFT's defaults (A Kaiming-uniform with
    /// bound 1/sqrt(in), B zero) from <paramref name="seed"/>.
    /// </summary>
    public static List<LoraAdapter> AddLora(ModelGraph graph, Dictionary<string, float[]> weights, IReadOnlyCollection<string> targets,
        int rank, float alpha, IReadOnlyDictionary<string, float[]>? initial = null, int seed = 0)
    {
        var rng = new Random(seed);
        var adapters = new List<LoraAdapter>();
        string scaleName = $"lora.scale.{rank}.{alpha}";
        float scale = alpha / rank;
        var originals = graph.Nodes.ToList();
        foreach (var node in originals)
        {
            if (node.OpType != "MatMul" || node.Inputs.Count != 2) continue;
            var w = node.Inputs[1];
            var m = System.Text.RegularExpressions.Regex.Match(w, @"^blk\.(\d+)\.([a-z_]+)\.weight$");
            if (!m.Success || !targets.Contains(m.Groups[2].Value) || !PeftModule.TryGetValue(m.Groups[2].Value, out var module)) continue;
            if (!graph.Initializers.TryGetValue(w, out var wShape) || wShape.Length != 2)
                throw new InvalidOperationException($"LoRA target {w}: not a 2-D initializer");
            int fin = wShape[0], fout = wShape[1];   // the graph's MatMul weight is [in, out]
            string peft = $"base_model.model.model.layers.{m.Groups[1].Value}.{module}";
            string aName = $"lora:{peft}.A_T", bName = $"lora:{peft}.B_T";
            var aT = new float[fin * rank];
            var bT = new float[rank * fout];
            if (initial != null && initial.TryGetValue($"{peft}.lora_A.weight", out var a0) && initial.TryGetValue($"{peft}.lora_B.weight", out var b0))
            {
                if (a0.Length != rank * fin || b0.Length != fout * rank)
                    throw new ArgumentException($"{peft}: initial adapter sizes do not match rank {rank}, in {fin}, out {fout}");
                for (int r = 0; r < rank; r++) for (int i = 0; i < fin; i++) aT[i * rank + r] = a0[r * fin + i];
                for (int o = 0; o < fout; o++) for (int r = 0; r < rank; r++) bT[r * fout + o] = b0[o * rank + r];
            }
            else
            {
                double bound = 1.0 / Math.Sqrt(fin);
                for (int i = 0; i < aT.Length; i++) aT[i] = (float)((rng.NextDouble() * 2 - 1) * bound);
            }
            weights[aName] = aT; graph.Initializers[aName] = new[] { fin, rank };
            weights[bName] = bT; graph.Initializers[bName] = new[] { rank, fout };
            if (!weights.ContainsKey(scaleName)) { weights[scaleName] = new[] { scale }; graph.Initializers[scaleName] = new[] { 1 }; }

            string y = node.Outputs[0];
            string yBase = y + ".lora_base";
            node.Outputs[0] = yBase;
            int at = graph.Nodes.IndexOf(node) + 1;
            graph.Nodes.InsertRange(at, new[]
            {
                Node("MatMul", new[] { node.Inputs[0], aName }, new[] { y + ".lora_xa" }),
                Node("MatMul", new[] { y + ".lora_xa", bName }, new[] { y + ".lora_xab" }),
                Node("Mul", new[] { y + ".lora_xab", scaleName }, new[] { y + ".lora_delta" }),
                Node("Add", new[] { yBase, y + ".lora_delta" }, new[] { y }),
            });
            adapters.Add(new LoraAdapter(peft, aName, bName, fin, fout, rank));
        }
        if (adapters.Count == 0) throw new ArgumentException($"no MatMul matched the LoRA targets [{string.Join(", ", targets)}]");
        return adapters;
    }

    /// <summary>
    /// Append a language-model loss and the backward pass: <c>loss = mean softmax-CE(logits, labels)</c> over the positions whose label
    /// is not <paramref name="ignoreIndex"/> (labels are a new graph input <c>labels</c> [1, seq], already shifted by the caller so label t
    /// is the token that follows position t), and one gradient output per name in <paramref name="parameters"/> (graph initializers).
    /// </summary>
    /// <param name="checkpointLayers">
    /// Activation checkpointing per decoder layer: only each layer's INPUT is kept from the forward; just before a layer's backward its forward
    /// is re-emitted (outputs renamed <c>.rc</c>) and the layer's rules read those. Memory is then ~one layer's activations plus the layer inputs
    /// instead of every activation of every layer (llama3.1:8b at 2,048 tokens: ~20 GB -> ~1 GB of saved tensors) for one extra forward of
    /// compute. Layers are the GGUF builder's (<c>blk.L_*</c> / <c>block_L_out</c>); nodes outside a layer (embedding, final norm, LM head) are kept.
    /// Relies on the compiler keeping the given node order (GraphCompiler's stable topological sort).
    /// </param>
    public static TrainingGraphInfo AddBackward(ModelGraph graph, string logits, int vocab, IReadOnlyCollection<string> parameters, int ignoreIndex = -100,
        bool checkpointLayers = false)
    {
        var b = new Builder(graph);
        var forward = graph.Nodes.ToList();
        // layer of each forward node (-1 = outside every layer)
        var layerOf = new Dictionary<GraphNode, int>(ReferenceEqualityComparer.Instance);
        var layerRx = new System.Text.RegularExpressions.Regex(@"^(?:blk\.(\d+)_|block_(\d+)_out$)");
        foreach (var n in forward)
        {
            int L = -1;
            foreach (var o in n.Outputs)
            {
                var m = layerRx.Match(o);
                if (m.Success) { L = int.Parse(m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value); break; }
            }
            layerOf[n] = L;
        }
        int recomputed = -1;
        void Recompute(int L)
        {
            // re-emit layer L's forward, outputs renamed; inputs produced inside the layer read the renamed copies
            var inLayer = forward.Where(n => layerOf[n] == L).ToList();
            var outs = new HashSet<string>(inLayer.SelectMany(n => n.Outputs));
            string R(string v) => outs.Contains(v) ? v + ".rc" : v;
            foreach (var n in inLayer)
                graph.Nodes.Add(new GraphNode { OpType = n.OpType, Inputs = n.Inputs.Select(R).ToList(), Outputs = n.Outputs.Select(o => o + ".rc").ToList(),
                    Attributes = n.Attributes, RawAttributes = n.RawAttributes });
            b.Saved = R;
            recomputed = L;
        }

        // ── loss ──
        const string labels = "labels";
        graph.Inputs.Add(new GraphValueInfo { Name = labels, Shape = new[] { 1, -1 } });
        b.Add("Reshape", new[] { logits }, new[] { "train.logits2d" }, ("shape", new long[] { -1, vocab }));
        b.Add("Reshape", new[] { labels }, new[] { "train.labels1d" }, ("shape", new long[] { -1 }));
        b.Add("SoftmaxCrossEntropyLoss", new[] { "train.logits2d", "train.labels1d" }, new[] { "loss" },
            ("reduction", "mean"), ("ignore_index", (long)ignoreIndex));
        b.Add("SoftmaxCrossEntropyLossGrad", new[] { "train.logits2d", "train.labels1d" }, new[] { "train.dlogits2d" },
            ("reduction", "mean"), ("ignore_index", (long)ignoreIndex));
        graph.Outputs.Add(new GraphValueInfo { Name = "loss", Shape = new[] { 1 } });
        // d logits = reshape(d logits2d) back to logits' shape
        b.Contribute(logits, b.ReshapeLike("train.dlogits2d", logits));

        // ── which values carry a gradient: everything downstream of a parameter ──
        var requires = new HashSet<string>(parameters);
        foreach (var n in forward)
            if (n.Inputs.Any(requires.Contains)) foreach (var o in n.Outputs) requires.Add(o);
        if (!requires.Contains(logits)) throw new ArgumentException("the logits do not depend on any of the parameters");

        // ── reverse sweep ──
        for (int i = forward.Count - 1; i >= 0; i--)
        {
            var n = forward[i];
            if (!n.Outputs.Any(o => b.HasGradient(o))) continue;
            bool Need(int k) => k < n.Inputs.Count && requires.Contains(n.Inputs[k]);
            if (!Enumerable.Range(0, n.Inputs.Count).Any(Need)) continue;
            if (checkpointLayers)
            {
                int L = layerOf[n];
                if (L < 0) b.Saved = v => v;
                else if (L != recomputed) Recompute(L);
            }
            var S = b.Saved;   // a forward activation as the backward reads it (the recomputed copy inside a checkpointed layer)
            switch (n.OpType)
            {
                case "Add":
                {
                    var dy = b.Gradient(n.Outputs[0]);
                    if (Need(0)) b.Contribute(n.Inputs[0], dy);
                    if (Need(1)) b.Contribute(n.Inputs[1], dy);
                    break;
                }
                case "Mul":
                {
                    // y = a * c with c a frozen constant (the LoRA scale); both sides carrying a gradient is not needed yet.
                    var dy = b.Gradient(n.Outputs[0]);
                    if (Need(0) && Need(1)) throw new NotSupportedException($"autograd: Mul with two differentiable inputs ({n.Outputs[0]})");
                    int k = Need(0) ? 0 : 1;
                    b.Contribute(n.Inputs[k], b.Emit("Mul", new[] { dy, n.Inputs[1 - k] }, n.Outputs[0] + ".grad_mul"));
                    break;
                }
                case "MatMul":
                {
                    var dy = b.Gradient(n.Outputs[0]);
                    string x = n.Inputs[0], w = n.Inputs[1];
                    if (!graph.Initializers.TryGetValue(w, out var ws) || ws.Length != 2)
                        throw new NotSupportedException($"autograd: MatMul whose second input is not a 2-D initializer ({n.Outputs[0]})");
                    int K = ws[0], N = ws[1];
                    if (Need(1))
                    {
                        var dy2 = b.Emit("Reshape", new[] { dy }, dy + ".2d", ("shape", new long[] { -1, N }));
                        var x2 = b.Emit("Reshape", new[] { S(x) }, x + ".2d_for_" + w, ("shape", new long[] { -1, K }));
                        b.Contribute(w, b.Emit("Gemm", new[] { x2, dy2 }, "grad_part." + w + "." + n.Outputs[0], ("transA", 1L)));
                    }
                    // dx = dy W^T through the weight in whatever form it was loaded (GGUF-quantized, low precision, f32).
                    if (Need(0))
                        b.Contribute(x, b.Emit("MatMulGradInput", new[] { dy, w }, n.Outputs[0] + ".grad_x"));
                    break;
                }
                case "Reshape":
                    b.Contribute(n.Inputs[0], b.ReshapeLike(b.Gradient(n.Outputs[0]), S(n.Inputs[0])));
                    break;
                case "RMSNormalization":
                    if (Need(1)) throw new NotSupportedException("autograd: a trainable RMSNorm weight is not supported yet");
                    b.Contribute(n.Inputs[0], b.Emit("RMSNormGrad", new[] { S(n.Inputs[0]), n.Inputs[1], b.Gradient(n.Outputs[0]) },
                        n.Outputs[0] + ".grad_x", n.Attributes));
                    break;
                case "AddRMSNorm":
                {
                    if (Need(2)) throw new NotSupportedException("autograd: a trainable RMSNorm weight is not supported yet");
                    // res = a + b; norm = rmsnorm(res) * w. d res = d res_out + RMSNormGrad(res, w, d norm); d a = d b = d res.
                    string res = n.Outputs[0], norm = n.Outputs[1];
                    var parts = new List<string>();
                    if (b.HasGradient(res)) parts.Add(b.Gradient(res));
                    if (b.HasGradient(norm))
                        parts.Add(b.Emit("RMSNormGrad", new[] { S(res), n.Inputs[2], b.Gradient(norm) }, norm + ".grad_res", n.Attributes));
                    var dres = b.Sum(parts, res + ".grad_total");
                    if (Need(0)) b.Contribute(n.Inputs[0], dres);
                    if (Need(1)) b.Contribute(n.Inputs[1], dres);
                    break;
                }
                case "SwiGLU":
                {
                    var outs = b.EmitMulti("SwiGLUGrad", new[] { S(n.Inputs[0]), S(n.Inputs[1]), b.Gradient(n.Outputs[0]) },
                        new[] { n.Outputs[0] + ".grad_g", n.Outputs[0] + ".grad_u" }, null);
                    if (Need(0)) b.Contribute(n.Inputs[0], outs[0]);
                    if (Need(1)) b.Contribute(n.Inputs[1], outs[1]);
                    break;
                }
                case "RoPE":
                {
                    var ins = new List<string> { b.Gradient(n.Outputs[0]) };
                    if (n.Inputs.Count > 1 && !string.IsNullOrEmpty(n.Inputs[1])) ins.Add(n.Inputs[1]);
                    b.Contribute(n.Inputs[0], b.Emit("RoPEGrad", ins.ToArray(), n.Outputs[0] + ".grad_x", n.Attributes));
                    break;
                }
                case "FusedAttention":
                {
                    if (n.Inputs.Count > 3 && !string.IsNullOrEmpty(n.Inputs[3])) throw new NotSupportedException("autograd: attention sinks");
                    var outs = b.EmitMulti("FusedAttentionGrad",
                        new[] { S(n.Inputs[0]), S(n.Inputs[1]), S(n.Inputs[2]), S(n.Outputs[0]), b.Gradient(n.Outputs[0]) },
                        new[] { n.Outputs[0] + ".grad_q", n.Outputs[0] + ".grad_k", n.Outputs[0] + ".grad_v" }, n.Attributes);
                    for (int k = 0; k < 3; k++) if (Need(k)) b.Contribute(n.Inputs[k], outs[k]);
                    break;
                }
                case "Gather":
                    if (Need(0)) throw new NotSupportedException("autograd: a trainable embedding table is not supported yet");
                    break;
                default:
                    throw new NotSupportedException($"autograd: no gradient rule for {n.OpType} ({n.Outputs.FirstOrDefault()})");
            }
        }

        // ── parameter gradients as graph outputs ──
        var grads = new Dictionary<string, string>();
        foreach (var p in parameters)
        {
            if (!b.HasGradient(p)) throw new ArgumentException($"parameter {p} does not reach the loss");
            string name = "grad:" + p;
            b.Add("Identity", new[] { b.Gradient(p) }, new[] { name });
            graph.Outputs.Add(new GraphValueInfo { Name = name, Shape = graph.Initializers[p] });
            grads[p] = name;
        }
        return new TrainingGraphInfo("loss", labels, grads);
    }

    private static GraphNode Node(string op, string[] inputs, string[] outputs, Dictionary<string, JsonElement>? attrs = null)
        => new() { OpType = op, Inputs = inputs.ToList(), Outputs = outputs.ToList(), Attributes = attrs };

    /// <summary>Accumulates gradient contributions per value and emits the nodes; each value's gradient is summed once, on first use.</summary>
    private sealed class Builder(ModelGraph graph)
    {
        private readonly Dictionary<string, List<string>> _parts = new();
        private readonly Dictionary<string, string> _summed = new();
        private readonly Dictionary<string, string> _shapes = new();
        private int _n;

        /// <summary>Maps a forward activation to the name the backward reads (identity, or the recomputed copy in a checkpointed layer).</summary>
        public Func<string, string> Saved { get; set; } = v => v;

        public bool HasGradient(string v) => _parts.ContainsKey(v);

        public void Contribute(string v, string grad)
        {
            if (_summed.ContainsKey(v)) throw new InvalidOperationException($"autograd: gradient of {v} was used before all its contributions arrived");
            if (!_parts.TryGetValue(v, out var l)) _parts[v] = l = new List<string>();
            l.Add(grad);
        }

        public string Gradient(string v)
        {
            if (_summed.TryGetValue(v, out var s)) return s;
            return _summed[v] = Sum(_parts[v], "grad." + v);
        }

        public string Sum(List<string> parts, string name)
        {
            if (parts.Count == 0) throw new InvalidOperationException($"autograd: no gradient for {name}");
            string acc = parts[0];
            for (int i = 1; i < parts.Count; i++) acc = Emit("Add", new[] { acc, parts[i] }, name + (i == parts.Count - 1 ? "" : "." + i));
            return acc;
        }

        public string ReshapeLike(string v, string like)
        {
            if (!_shapes.TryGetValue(like, out var shape)) _shapes[like] = shape = Emit("Shape", new[] { like }, "shape_of." + like);
            return Emit("Reshape", new[] { v, shape }, v + ".as_" + like);
        }

        public string Emit(string op, string[] inputs, string output, params (string Key, object Value)[] attrs)
            => Emit(op, inputs, output, attrs.Length == 0 ? null : attrs.ToDictionary(a => a.Key, a => MLJson.ToElement(a.Value)));

        public string Emit(string op, string[] inputs, string output, Dictionary<string, JsonElement>? attrs)
        {
            output = Unique(output);
            graph.Nodes.Add(Node(op, inputs, new[] { output }, attrs));
            return output;
        }

        public string[] EmitMulti(string op, string[] inputs, string[] outputs, Dictionary<string, JsonElement>? attrs)
        {
            outputs = outputs.Select(Unique).ToArray();
            graph.Nodes.Add(Node(op, inputs, outputs, attrs));
            return outputs;
        }

        public void Add(string op, string[] inputs, string[] outputs, params (string Key, object Value)[] attrs)
            => graph.Nodes.Add(Node(op, inputs, outputs, attrs.Length == 0 ? null : attrs.ToDictionary(a => a.Key, a => MLJson.ToElement(a.Value))));

        private string Unique(string name) => _parts.ContainsKey(name) || graph.Nodes.Any(n => n.Outputs.Contains(name)) ? $"{name}.{++_n}" : name;
    }
}
