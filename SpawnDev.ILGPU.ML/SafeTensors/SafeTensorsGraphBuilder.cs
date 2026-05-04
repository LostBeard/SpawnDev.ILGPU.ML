using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Hub;

namespace SpawnDev.ILGPU.ML.SafeTensors;

/// <summary>
/// Constructs a ModelGraph from SafeTensors weight names + HuggingFace config.json.
/// SafeTensors is weights-only — the computation graph is inferred from the architecture config.
/// This builder creates the same decoder-only transformer graph as GGUFGraphBuilder but uses
/// HuggingFace tensor naming conventions (model.layers.N.self_attn.q_proj.weight etc.)
/// </summary>
public static class SafeTensorsGraphBuilder
{
    public static ModelGraph BuildGraph(HFModelConfig config, SafeTensorsFile model)
    {
        // Route to architecture-specific builder
        return config.ArchitectureFamily switch
        {
            "encoder" => BuildEncoderGraph(config, model),
            "vision" => BuildVisionGraph(config, model),
            _ => BuildDecoderGraph(config, model)
        };
    }

    /// <summary>Build encoder transformer (BERT, DistilBERT, RoBERTa).</summary>
    private static ModelGraph BuildEncoderGraph(HFModelConfig config, SafeTensorsFile model)
    {
        var graph = new ModelGraph { Name = $"SafeTensors Encoder ({config.ModelType})" };
        int H = config.HiddenSize;
        int nLayers = config.NumHiddenLayers;

        graph.Inputs.Add(new GraphValueInfo { Name = "input_ids", Shape = new[] { 1, -1 } });
        graph.Inputs.Add(new GraphValueInfo { Name = "attention_mask", Shape = new[] { 1, -1 } });
        graph.Outputs.Add(new GraphValueInfo { Name = "last_hidden_state", Shape = new[] { 1, -1, H } });

        string prev = "input_ids";

        // Token embedding
        string embedName = config.GetEmbeddingName();
        if (model.HasTensor(embedName))
        {
            graph.Initializers[embedName] = new[] { config.VocabSize, H };
            AddNode(graph, "Gather", new[] { embedName, prev }, new[] { "embed_out" });
            prev = "embed_out";
        }

        // Encoder layers (simplified — same attention pattern as decoder but bidirectional)
        for (int L = 0; L < nLayers; L++)
        {
            string pfx = config.GetLayerPrefix(L);
            string layerIn = prev;

            // Self-attention (bidirectional — no causal mask needed)
            string normOut = $"enc_{L}_norm";
            string normW = $"{pfx}.attention.output.LayerNorm.weight";
            if (!model.HasTensor(normW)) normW = $"{pfx}.layer_norm.weight";
            if (model.HasTensor(normW)) graph.Initializers[normW] = new[] { H };
            AddNode(graph, "LayerNormalization", model.HasTensor(normW) ? new[] { layerIn, normW } : new[] { layerIn }, new[] { normOut });

            // Q/K/V → attention → output (simplified as Identity for now)
            string attnOut = $"enc_{L}_attn";
            AddNode(graph, "Identity", new[] { normOut }, new[] { attnOut });

            string res1 = $"enc_{L}_res1";
            AddNode(graph, "Add", new[] { layerIn, attnOut }, new[] { res1 });

            // FFN
            string ffnNormOut = $"enc_{L}_ffn_norm";
            AddNode(graph, "LayerNormalization", new[] { res1 }, new[] { ffnNormOut });
            string ffnOut = $"enc_{L}_ffn";
            AddNode(graph, "Identity", new[] { ffnNormOut }, new[] { ffnOut });

            prev = $"enc_{L}_out";
            AddNode(graph, "Add", new[] { res1, ffnOut }, new[] { prev });
        }

        // Final output
        AddNode(graph, "Identity", new[] { prev }, new[] { "last_hidden_state" });
        return graph;
    }

    /// <summary>Build vision transformer (ViT, DeiT, BEiT).</summary>
    private static ModelGraph BuildVisionGraph(HFModelConfig config, SafeTensorsFile model)
    {
        var graph = new ModelGraph { Name = $"SafeTensors Vision ({config.ModelType})" };
        int H = config.HiddenSize;

        graph.Inputs.Add(new GraphValueInfo { Name = "pixel_values", Shape = new[] { 1, 3, 224, 224 } });
        graph.Outputs.Add(new GraphValueInfo { Name = "logits", Shape = new[] { 1, 1000 } });

        // Patch embedding + transformer layers (simplified)
        string prev = "pixel_values";
        AddNode(graph, "Reshape", new[] { prev }, new[] { "patches" },
            Attr("shape", new long[] { 1, -1, H }));
        prev = "patches";

        for (int L = 0; L < config.NumHiddenLayers; L++)
        {
            string layerIn = prev;
            string normOut = $"vit_{L}_norm";
            AddNode(graph, "LayerNormalization", new[] { layerIn }, new[] { normOut });
            prev = $"vit_{L}_out";
            AddNode(graph, "Add", new[] { layerIn, normOut }, new[] { prev });
        }

        AddNode(graph, "ReduceMean", new[] { prev }, new[] { "pooled" },
            Attr("axes", new long[] { 1 }));
        AddNode(graph, "Identity", new[] { "pooled" }, new[] { "logits" });
        return graph;
    }

    /// <summary>Build decoder-only transformer (LLaMA, GPT-2, Mistral, Phi).</summary>
    private static ModelGraph BuildDecoderGraph(HFModelConfig config, SafeTensorsFile model)
    {
        var graph = new ModelGraph { Name = $"SafeTensors ({config.ModelType})" };

        int H = config.HiddenSize;
        int nLayers = config.NumHiddenLayers;
        int nHeads = config.NumAttentionHeads;
        int nKVHeads = config.NumKeyValueHeads;
        int headDim = H / nHeads;
        int vocabSize = config.VocabSize;
        bool useRMSNorm = config.UsesRMSNorm;

        // Graph I/O
        graph.Inputs.Add(new GraphValueInfo { Name = "input_ids", Shape = new[] { 1, -1 } });
        graph.Outputs.Add(new GraphValueInfo { Name = "logits", Shape = new[] { 1, -1, vocabSize } });

        string prev = "input_ids";

        // 1. Token embedding
        string embedName = config.GetEmbeddingName();
        if (model.HasTensor(embedName))
        {
            graph.Initializers[embedName] = new[] { vocabSize, H };
            AddNode(graph, "Gather", new[] { embedName, prev }, new[] { "embed_out" });
            prev = "embed_out";
        }

        // 2. Transformer blocks
        for (int L = 0; L < nLayers; L++)
        {
            string pfx = config.GetLayerPrefix(L);
            string layerIn = prev;

            // Attention norm
            string normOut = $"layer_{L}_attn_norm";
            string normW = $"{pfx}.input_layernorm.weight";
            if (!model.HasTensor(normW)) normW = $"{pfx}.self_attn_layer_norm.weight"; // distilbert style
            if (model.HasTensor(normW)) graph.Initializers[normW] = new[] { H };
            AddNode(graph, "LayerNormalization", model.HasTensor(normW) ? new[] { layerIn, normW } : new[] { layerIn }, new[] { normOut });

            // Q/K/V projections
            string qW = $"{pfx}.self_attn.q_proj.weight";
            string kW = $"{pfx}.self_attn.k_proj.weight";
            string vW = $"{pfx}.self_attn.v_proj.weight";
            if (!model.HasTensor(qW)) { qW = $"{pfx}.attn.c_attn.weight"; } // GPT-2 fused QKV

            string qOut = $"L{L}_q", kOut = $"L{L}_k", vOut = $"L{L}_v";
            AddLinear(graph, model, qW, normOut, qOut);
            AddLinear(graph, model, kW, normOut, kOut);
            AddLinear(graph, model, vW, normOut, vOut);

            // Multi-head reshape + attention (same pattern as GGUF)
            AddNode(graph, "Reshape", new[] { qOut }, new[] { $"L{L}_q4d" }, Attr("shape", new long[] { 1, -1, nHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"L{L}_q4d" }, new[] { $"L{L}_q_mh" }, Attr("perm", new long[] { 0, 2, 1, 3 }));
            AddNode(graph, "Reshape", new[] { kOut }, new[] { $"L{L}_k4d" }, Attr("shape", new long[] { 1, -1, nKVHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"L{L}_k4d" }, new[] { $"L{L}_k_mh" }, Attr("perm", new long[] { 0, 2, 1, 3 }));
            AddNode(graph, "Reshape", new[] { vOut }, new[] { $"L{L}_v4d" }, Attr("shape", new long[] { 1, -1, nKVHeads, headDim }));
            AddNode(graph, "Transpose", new[] { $"L{L}_v4d" }, new[] { $"L{L}_v_mh" }, Attr("perm", new long[] { 0, 2, 1, 3 }));

            // Q @ K^T / sqrt(d) → softmax → @ V
            AddNode(graph, "Transpose", new[] { $"L{L}_k_mh" }, new[] { $"L{L}_k_t" }, Attr("perm", new long[] { 0, 1, 3, 2 }));
            AddNode(graph, "MatMul", new[] { $"L{L}_q_mh", $"L{L}_k_t" }, new[] { $"L{L}_qk" });
            // Scale constant
            string scaleName = $"L{L}_scale";
            graph.Initializers[scaleName] = new[] { 1 };
            AddNode(graph, "Mul", new[] { $"L{L}_qk", scaleName }, new[] { $"L{L}_qk_s" });
            AddNode(graph, "Softmax", new[] { $"L{L}_qk_s" }, new[] { $"L{L}_attn_w" }, Attr("axis", -1L));
            AddNode(graph, "MatMul", new[] { $"L{L}_attn_w", $"L{L}_v_mh" }, new[] { $"L{L}_attn_v" });

            // Merge heads
            AddNode(graph, "Transpose", new[] { $"L{L}_attn_v" }, new[] { $"L{L}_attn_t" }, Attr("perm", new long[] { 0, 2, 1, 3 }));
            AddNode(graph, "Reshape", new[] { $"L{L}_attn_t" }, new[] { $"L{L}_attn_m" }, Attr("shape", new long[] { 1, -1, H }));

            // Output projection
            string oW = $"{pfx}.self_attn.o_proj.weight";
            if (!model.HasTensor(oW)) oW = $"{pfx}.attn.c_proj.weight";
            string attnOut = $"L{L}_attn_out";
            AddLinear(graph, model, oW, $"L{L}_attn_m", attnOut);

            // Residual 1
            string res1 = $"L{L}_res1";
            AddNode(graph, "Add", new[] { layerIn, attnOut }, new[] { res1 });

            // FFN norm
            string ffnNormOut = $"L{L}_ffn_norm";
            string ffnNormW = $"{pfx}.post_attention_layernorm.weight";
            if (!model.HasTensor(ffnNormW)) ffnNormW = $"{pfx}.ln_2.weight";
            if (model.HasTensor(ffnNormW)) graph.Initializers[ffnNormW] = new[] { H };
            AddNode(graph, "LayerNormalization", model.HasTensor(ffnNormW) ? new[] { res1, ffnNormW } : new[] { res1 }, new[] { ffnNormOut });

            // FFN
            string gateW = $"{pfx}.mlp.gate_proj.weight";
            string upW = $"{pfx}.mlp.up_proj.weight";
            string downW = $"{pfx}.mlp.down_proj.weight";
            // GPT-2 style
            if (!model.HasTensor(gateW))
            {
                gateW = $"{pfx}.mlp.c_fc.weight";
                downW = $"{pfx}.mlp.c_proj.weight";
            }

            string gateOut = $"L{L}_gate", upOut = $"L{L}_up";
            AddLinear(graph, model, gateW, ffnNormOut, gateOut);

            string activated;
            if (model.HasTensor(upW))
            {
                // Gated FFN (LLaMA style): SiLU(gate) * up → down
                AddLinear(graph, model, upW, ffnNormOut, upOut);
                AddNode(graph, "Sigmoid", new[] { gateOut }, new[] { $"L{L}_sig" });
                AddNode(graph, "Mul", new[] { gateOut, $"L{L}_sig" }, new[] { $"L{L}_silu" });
                activated = $"L{L}_ffn_act";
                AddNode(graph, "Mul", new[] { $"L{L}_silu", upOut }, new[] { activated });
            }
            else
            {
                // Standard FFN (GPT-2 style): GELU(gate) → down
                activated = $"L{L}_gelu";
                AddNode(graph, "Gelu", new[] { gateOut }, new[] { activated });
            }

            string ffnOut = $"L{L}_ffn_out";
            AddLinear(graph, model, downW, activated, ffnOut);

            // Residual 2
            prev = $"block_{L}_out";
            AddNode(graph, "Add", new[] { res1, ffnOut }, new[] { prev });
        }

        // 3. Final norm
        string fnW = config.GetFinalNormName();
        if (model.HasTensor(fnW)) graph.Initializers[fnW] = new[] { H };
        AddNode(graph, "LayerNormalization", model.HasTensor(fnW) ? new[] { prev, fnW } : new[] { prev }, new[] { "final_norm" });

        // 4. LM head
        string lmW = config.GetLMHeadName();
        if (model.HasTensor(lmW))
        {
            graph.Initializers[lmW] = new[] { vocabSize, H };
            AddNode(graph, "MatMul", new[] { "final_norm", lmW }, new[] { "logits" });
        }
        else if (config.TieWordEmbeddings && model.HasTensor(embedName))
        {
            AddNode(graph, "Transpose", new[] { embedName }, new[] { "lm_head_t" }, Attr("perm", new long[] { 1, 0 }));
            AddNode(graph, "MatMul", new[] { "final_norm", "lm_head_t" }, new[] { "logits" });
        }

        return graph;
    }

    private static void AddNode(ModelGraph g, string op, string[] ins, string[] outs,
        Dictionary<string, JsonElement>? attrs = null)
    {
        g.Nodes.Add(new GraphNode { OpType = op, Inputs = ins.ToList(), Outputs = outs.ToList(), Attributes = attrs });
    }

    private static Dictionary<string, JsonElement> Attr(string k, object v)
        => new() { [k] = JsonSerializer.SerializeToElement(v) };

    private static void AddLinear(ModelGraph g, SafeTensorsFile m, string wName, string input, string output)
    {
        if (m.HasTensor(wName))
        {
            g.Initializers[wName] = Array.Empty<int>(); // Shape filled at load time
            string bName = wName.Replace(".weight", ".bias");
            if (m.HasTensor(bName))
            {
                g.Initializers[bName] = Array.Empty<int>();
                string matOut = $"{output}_pre_bias";
                AddNode(g, "MatMul", new[] { input, wName }, new[] { matOut });
                AddNode(g, "Add", new[] { matOut, bName }, new[] { output });
            }
            else
            {
                AddNode(g, "MatMul", new[] { input, wName }, new[] { output });
            }
        }
    }
}
