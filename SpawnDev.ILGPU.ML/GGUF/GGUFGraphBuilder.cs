using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.GGUF;

/// <summary>
/// Constructs a ModelGraph from GGUF model metadata and tensor names.
/// GGUF files contain weights + architecture metadata but NO computation graph.
/// This builder creates the full decoder-only transformer graph including:
/// - Token embedding lookup
/// - Per-layer: Norm → Q/K/V projection → Multi-head attention → Residual → Norm → FFN → Residual
/// - Final norm → LM head projection
///
/// Architecture-specific variations:
/// - Norm: RMSNorm (llama/mistral/gemma/qwen) vs LayerNorm (phi/gpt2)
/// - Activation: SiLU (llama/mistral) vs GELU (phi/gpt2)
/// - GQA: nKVHeads &lt; nHeads for grouped-query attention (llama3, mistral)
///
/// Note: RoPE (rotary position embeddings) and causal masking are applied at inference
/// time by the GraphExecutor's attention path, not baked into the static graph.
/// The graph represents the data flow; positional encoding is a runtime concern.
/// </summary>
public static class GGUFGraphBuilder
{
    public static (ModelGraph Graph, Dictionary<string, float[]> Weights, Dictionary<string, byte[]> QuantizedWeightBytes) BuildGraph(GGUFModel model)
    {
        var arch = model.Architecture.ToLowerInvariant();
        var graph = new ModelGraph { Name = $"{model.Name} ({arch})" };
        var weights = new Dictionary<string, float[]>();
        var quantizedBytes = new Dictionary<string, byte[]>();

        // Extract architecture hyperparameters
        int vocabSize = (int)model.VocabSize;
        int embedDim = (int)model.EmbeddingLength;
        int nLayers = (int)model.BlockCount;
        int nHeads = (int)model.AttentionHeadCount;
        int nKVHeads = (int)model.AttentionHeadCountKV;
        if (nKVHeads == 0) nKVHeads = nHeads;
        int headDim = embedDim / nHeads;
        int ffnDim = (int)model.GetMetadataInt($"{model.Architecture}.feed_forward_length",
            embedDim * 4);

        // Architecture-specific settings
        bool useRMSNorm = arch is "llama" or "mistral" or "gemma" or "gemma2" or "qwen" or "qwen2";
        bool useSiLU = arch is not "phi" and not "phi3" and not "gpt2" and not "falcon" and not "bloom" and not "mpt";

        // Graph input: token IDs [1, seq_len]
        graph.Inputs.Add(new GraphValueInfo { Name = "input_ids", Shape = new[] { 1, -1 } });
        graph.Outputs.Add(new GraphValueInfo { Name = "logits", Shape = new[] { 1, -1, vocabSize } });

        string prevOutput = "input_ids";

        // ═══════════════════════════════════════════════════════════
        //  1. Token embedding lookup
        // ═══════════════════════════════════════════════════════════
        var embedWeight = FindTensor(model, "token_embd.weight");
        if (embedWeight != null)
        {
            ExtractWeight(model, embedWeight, weights, quantizedBytes);
            graph.Initializers[embedWeight.Name] = embedWeight.Shape;
            AddNode(graph, "Gather", new[] { embedWeight.Name, prevOutput }, new[] { "embed_out" });
            prevOutput = "embed_out";
        }

        // ═══════════════════════════════════════════════════════════
        //  2. Transformer blocks
        // ═══════════════════════════════════════════════════════════
        for (int layer = 0; layer < nLayers; layer++)
        {
            string pfx = $"blk.{layer}";
            string layerIn = prevOutput;

            // ── Attention norm ──
            string normOut = $"{pfx}_attn_norm";
            AddNorm(graph, model, weights, $"{pfx}.attn_norm", layerIn, normOut, embedDim, useRMSNorm, quantizedBytes);

            // ── Q, K, V projections ──
            string qOut = $"{pfx}_q", kOut = $"{pfx}_k", vOut = $"{pfx}_v";
            AddLinear(graph, model, weights, $"{pfx}.attn_q", normOut, qOut, quantizedBytes);
            AddLinear(graph, model, weights, $"{pfx}.attn_k", normOut, kOut, quantizedBytes);
            AddLinear(graph, model, weights, $"{pfx}.attn_v", normOut, vOut, quantizedBytes);

            // ── Multi-head reshape: [batch, seq, embed] → [batch, nHeads, seq, headDim] ──
            string qReshaped = $"{pfx}_q_mh", kReshaped = $"{pfx}_k_mh", vReshaped = $"{pfx}_v_mh";
            // Reshape Q: [1, seq, nHeads*headDim] → [1, seq, nHeads, headDim] → transpose to [1, nHeads, seq, headDim]
            AddNode(graph, "Reshape", new[] { qOut }, new[] { $"{pfx}_q_4d" },
                Attrs("shape", new long[] { 1, -1, nHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"{pfx}_q_4d" }, new[] { qReshaped },
                Attrs("perm", new long[] { 0, 2, 1, 3 }));

            // Reshape K: [1, seq, nKVHeads*headDim] → [1, nKVHeads, seq, headDim]
            AddNode(graph, "Reshape", new[] { kOut }, new[] { $"{pfx}_k_4d" },
                Attrs("shape", new long[] { 1, -1, nKVHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"{pfx}_k_4d" }, new[] { kReshaped },
                Attrs("perm", new long[] { 0, 2, 1, 3 }));

            // Reshape V: same as K
            AddNode(graph, "Reshape", new[] { vOut }, new[] { $"{pfx}_v_4d" },
                Attrs("shape", new long[] { 1, -1, nKVHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"{pfx}_v_4d" }, new[] { vReshaped },
                Attrs("perm", new long[] { 0, 2, 1, 3 }));

            // ── Attention: Q @ K^T / sqrt(headDim) → softmax → @ V ──
            // Transpose K for matmul: [1, nKVHeads, seq, headDim] → [1, nKVHeads, headDim, seq]
            string kTransposed = $"{pfx}_k_t";
            AddNode(graph, "Transpose", new[] { kReshaped }, new[] { kTransposed },
                Attrs("perm", new long[] { 0, 1, 3, 2 }));

            // Q @ K^T → [1, nHeads, seq, seq]
            string qkOut = $"{pfx}_qk";
            AddNode(graph, "MatMul", new[] { qReshaped, kTransposed }, new[] { qkOut });

            // Scale by 1/sqrt(headDim)
            string qkScaled = $"{pfx}_qk_scaled";
            float scale = 1f / MathF.Sqrt(headDim);
            string scaleName = $"{pfx}_scale";
            weights[scaleName] = new[] { scale };
            graph.Initializers[scaleName] = new[] { 1 };
            AddNode(graph, "Mul", new[] { qkOut, scaleName }, new[] { qkScaled });

            // Softmax over last axis (seq dimension)
            string attnWeights = $"{pfx}_attn_weights";
            AddNode(graph, "Softmax", new[] { qkScaled }, new[] { attnWeights },
                Attrs("axis", -1L));

            // Attention @ V → [1, nHeads, seq, headDim]
            string attnValues = $"{pfx}_attn_val";
            AddNode(graph, "MatMul", new[] { attnWeights, vReshaped }, new[] { attnValues });

            // ── Merge heads: [1, nHeads, seq, headDim] → [1, seq, embed] ──
            string attnTransposed = $"{pfx}_attn_t";
            AddNode(graph, "Transpose", new[] { attnValues }, new[] { attnTransposed },
                Attrs("perm", new long[] { 0, 2, 1, 3 }));
            string attnMerged = $"{pfx}_attn_merged";
            AddNode(graph, "Reshape", new[] { attnTransposed }, new[] { attnMerged },
                Attrs("shape", new long[] { 1, -1, embedDim }));

            // ── Output projection ──
            string attnOut = $"{pfx}_attn_out";
            AddLinear(graph, model, weights, $"{pfx}.attn_output", attnMerged, attnOut, quantizedBytes);

            // ── Residual 1 ──
            string residual1 = $"{pfx}_res1";
            AddNode(graph, "Add", new[] { layerIn, attnOut }, new[] { residual1 });

            // ── FFN norm ──
            string ffnNormOut = $"{pfx}_ffn_norm";
            AddNorm(graph, model, weights, $"{pfx}.ffn_norm", residual1, ffnNormOut, embedDim, useRMSNorm, quantizedBytes);

            // ── FFN: gate + up → activation → down ──
            string gateOut = $"{pfx}_gate", upOut = $"{pfx}_up";
            AddLinear(graph, model, weights, $"{pfx}.ffn_gate", ffnNormOut, gateOut, quantizedBytes);
            AddLinear(graph, model, weights, $"{pfx}.ffn_up", ffnNormOut, upOut, quantizedBytes);

            string activated;
            if (useSiLU)
            {
                // SiLU(x) = x * sigmoid(x)
                string sigOut = $"{pfx}_gate_sig";
                AddNode(graph, "Sigmoid", new[] { gateOut }, new[] { sigOut });
                string siluOut = $"{pfx}_gate_silu";
                AddNode(graph, "Mul", new[] { gateOut, sigOut }, new[] { siluOut });
                activated = $"{pfx}_ffn_act";
                AddNode(graph, "Mul", new[] { siluOut, upOut }, new[] { activated });
            }
            else
            {
                // GELU
                string geluOut = $"{pfx}_gate_gelu";
                AddNode(graph, "Gelu", new[] { gateOut }, new[] { geluOut });
                activated = $"{pfx}_ffn_act";
                AddNode(graph, "Mul", new[] { geluOut, upOut }, new[] { activated });
            }

            string ffnOut = $"{pfx}_ffn_out";
            AddLinear(graph, model, weights, $"{pfx}.ffn_down", activated, ffnOut, quantizedBytes);

            // ── Residual 2 ──
            string layerOut = $"block_{layer}_out";
            AddNode(graph, "Add", new[] { residual1, ffnOut }, new[] { layerOut });
            prevOutput = layerOut;
        }

        // ═══════════════════════════════════════════════════════════
        //  3. Final norm
        // ═══════════════════════════════════════════════════════════
        string finalNormOut = "final_norm_out";
        AddNorm(graph, model, weights, "output_norm", prevOutput, finalNormOut, embedDim, useRMSNorm, quantizedBytes);

        // ═══════════════════════════════════════════════════════════
        //  4. LM head (output projection)
        // ═══════════════════════════════════════════════════════════
        var outputWeight = FindTensor(model, "output.weight");
        if (outputWeight != null)
        {
            ExtractWeight(model, outputWeight, weights, quantizedBytes);
            graph.Initializers[outputWeight.Name] = outputWeight.Shape;
            AddNode(graph, "MatMul", new[] { finalNormOut, outputWeight.Name }, new[] { "logits" });
        }
        else
        {
            // Tied embeddings: output.weight = token_embd.weight
            // Need transpose: embedding is [vocab, embed], LM head needs [embed, vocab]
            if (embedWeight != null)
            {
                string transposed = "output_weight_transposed";
                AddNode(graph, "Transpose", new[] { embedWeight.Name }, new[] { transposed },
                    Attrs("perm", new long[] { 1, 0 }));
                AddNode(graph, "MatMul", new[] { finalNormOut, transposed }, new[] { "logits" });
            }
        }

        return (graph, weights, quantizedBytes);
    }

    // ── Helpers ──

    private static GGUFTensorInfo? FindTensor(GGUFModel model, string name)
        => model.Tensors.FirstOrDefault(t => t.Name == name);

    private static void ExtractWeight(GGUFModel model, GGUFTensorInfo tensor,
        Dictionary<string, float[]> weights, Dictionary<string, byte[]>? quantizedBytes = null)
    {
        if (quantizedBytes != null && GGUFModel.IsQuantized(tensor.Type))
        {
            var rawBytes = model.GetTensorRawBytes(tensor);
            if (rawBytes != null)
            {
                quantizedBytes[tensor.Name] = rawBytes;
                weights[tensor.Name] = Array.Empty<float>();
                return;
            }
        }
        var data = model.GetTensorFloat32(tensor);
        if (data != null) weights[tensor.Name] = data;
    }

    private static void AddNode(ModelGraph graph, string opType, string[] inputs, string[] outputs,
        Dictionary<string, JsonElement>? attributes = null)
    {
        graph.Nodes.Add(new GraphNode
        {
            OpType = opType,
            Inputs = inputs.ToList(),
            Outputs = outputs.ToList(),
            Attributes = attributes
        });
    }

    private static Dictionary<string, JsonElement> Attrs(string key, object value)
        => new() { [key] = JsonSerializer.SerializeToElement(value) };

    private static void AddNorm(ModelGraph graph, GGUFModel model, Dictionary<string, float[]> weights,
        string tensorPrefix, string input, string output, int dim, bool useRMSNorm,
        Dictionary<string, byte[]>? quantizedBytes = null)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            ExtractWeight(model, weightTensor, weights, quantizedBytes);
            graph.Initializers[weightTensor.Name] = weightTensor.Shape;
        }

        if (useRMSNorm)
        {
            AddNode(graph, "LayerNormalization",
                weightTensor != null ? new[] { input, weightTensor.Name } : new[] { input },
                new[] { output });
        }
        else
        {
            var biasTensor = FindTensor(model, $"{tensorPrefix}.bias");
            if (biasTensor != null)
            {
                ExtractWeight(model, biasTensor, weights);
                graph.Initializers[biasTensor.Name] = biasTensor.Shape;
            }
            AddNode(graph, "LayerNormalization",
                new[] { input, weightTensor?.Name ?? "", biasTensor?.Name ?? "" }.Where(s => s.Length > 0).ToArray(),
                new[] { output });
        }
    }

    private static void AddLinear(ModelGraph graph, GGUFModel model, Dictionary<string, float[]> weights,
        string tensorPrefix, string input, string output,
        Dictionary<string, byte[]>? quantizedBytes = null)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            ExtractWeight(model, weightTensor, weights, quantizedBytes);
            graph.Initializers[weightTensor.Name] = weightTensor.Shape;
        }

        var biasTensor = FindTensor(model, $"{tensorPrefix}.bias");
        string matmulOut = biasTensor != null ? $"{output}_pre_bias" : output;

        if (weightTensor != null)
            AddNode(graph, "MatMul", new[] { input, weightTensor.Name }, new[] { matmulOut });

        if (biasTensor != null)
        {
            ExtractWeight(model, biasTensor, weights);
            graph.Initializers[biasTensor.Name] = biasTensor.Shape;
            AddNode(graph, "Add", new[] { matmulOut, biasTensor.Name }, new[] { output });
        }
    }
}
