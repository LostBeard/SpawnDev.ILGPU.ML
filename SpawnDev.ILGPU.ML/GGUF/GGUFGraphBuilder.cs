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
/// <summary>Raw GGUF block bytes + the GGML quantization type that decodes them.
/// The type MUST travel with the bytes: every GGML layout decodes differently, and
/// decoding one as another produces silent garbage (the K-quant landmine, 2026-06-11).</summary>
public sealed record GGUFQuantizedWeight(byte[] Bytes, GGMLType Type);

public static class GGUFGraphBuilder
{
    public static (ModelGraph Graph, Dictionary<string, float[]> Weights,
        Dictionary<string, GGUFQuantizedWeight> QuantizedWeights,
        HashSet<string> TransposeOnUpload) BuildGraph(GGUFModel model)
    {
        var arch = model.Architecture.ToLowerInvariant();
        var graph = new ModelGraph { Name = $"{model.Name} ({arch})" };
        var weights = new Dictionary<string, float[]>();
        var quantizedBytes = new Dictionary<string, GGUFQuantizedWeight>();
        var transposeOnUpload = new HashSet<string>();

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

        // Architecture-specific settings. The whole gemma family (gemma, gemma2, gemma3, gemma4) uses
        // RMSNorm + a GELU-gated (GeGLU) MLP — NOT SiLU/SwiGLU like llama. Match by prefix so each new
        // gemma generation is recognized without a hardcoded version list (gemma4 fell to LayerNorm+SiLU,
        // both wrong, before this).
        bool isGemma = arch.StartsWith("gemma", StringComparison.Ordinal);
        bool useRMSNorm = isGemma || arch is "llama" or "mistral" or "qwen" or "qwen2";
        bool useSiLU = !isGemma
            && arch is not "phi" and not "phi3" and not "gpt2" and not "falcon" and not "bloom" and not "mpt";

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
            // GATHER-TABLE role: quantized tables stay compressed (FusedDequantGather
            // decodes looked-up rows in-register). Declared shape is the PHYSICAL
            // row-major order [vocab, n_embd] - GGUF ne is fastest-dim-first
            // [n_embd, vocab], so reverse. Gather(axis 0) then reads token rows as
            // stored; no transpose, no F32 expansion.
            ExtractWeight(model, embedWeight, weights, quantizedBytes);
            graph.Initializers[embedWeight.Name] = embedWeight.Shape.Reverse().ToArray();
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
            AddNorm(graph, model, weights, $"{pfx}.attn_norm", layerIn, normOut, embedDim, useRMSNorm, isGemma);

            // ── Q, K, V projections ──
            string qOut = $"{pfx}_q", kOut = $"{pfx}_k", vOut = $"{pfx}_v";
            AddLinear(graph, model, weights, $"{pfx}.attn_q", normOut, qOut, quantizedBytes, transposeOnUpload);
            AddLinear(graph, model, weights, $"{pfx}.attn_k", normOut, kOut, quantizedBytes, transposeOnUpload);
            AddLinear(graph, model, weights, $"{pfx}.attn_v", normOut, vOut, quantizedBytes, transposeOnUpload);

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
            AddLinear(graph, model, weights, $"{pfx}.attn_output", attnMerged, attnOut, quantizedBytes, transposeOnUpload);

            // ── Post-attention norm (gemma 2/3/4 norm-sandwich: normalize the sublayer OUTPUT before the
            //    residual add). Presence-based — llama/mistral/etc. have no such tensor and are unaffected. ──
            string attnResInput = attnOut;
            if (FindTensor(model, $"{pfx}.post_attention_norm.weight") != null)
            {
                string postAttnOut = $"{pfx}_post_attn_norm";
                AddNorm(graph, model, weights, $"{pfx}.post_attention_norm", attnOut, postAttnOut, embedDim, useRMSNorm, isGemma);
                attnResInput = postAttnOut;
            }

            // ── Residual 1 ──
            string residual1 = $"{pfx}_res1";
            AddNode(graph, "Add", new[] { layerIn, attnResInput }, new[] { residual1 });

            // ── FFN norm ──
            string ffnNormOut = $"{pfx}_ffn_norm";
            AddNorm(graph, model, weights, $"{pfx}.ffn_norm", residual1, ffnNormOut, embedDim, useRMSNorm, isGemma);

            // ── FFN: gate + up → activation → down ──
            string gateOut = $"{pfx}_gate", upOut = $"{pfx}_up";
            AddLinear(graph, model, weights, $"{pfx}.ffn_gate", ffnNormOut, gateOut, quantizedBytes, transposeOnUpload);
            AddLinear(graph, model, weights, $"{pfx}.ffn_up", ffnNormOut, upOut, quantizedBytes, transposeOnUpload);

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
            AddLinear(graph, model, weights, $"{pfx}.ffn_down", activated, ffnOut, quantizedBytes, transposeOnUpload);

            // ── Post-FFN norm (gemma norm-sandwich), presence-based. ──
            string ffnResInput = ffnOut;
            if (FindTensor(model, $"{pfx}.post_ffw_norm.weight") != null)
            {
                string postFfwOut = $"{pfx}_post_ffw_norm";
                AddNorm(graph, model, weights, $"{pfx}.post_ffw_norm", ffnOut, postFfwOut, embedDim, useRMSNorm, isGemma);
                ffnResInput = postFfwOut;
            }

            // ── Residual 2 ──
            string layerOut = $"block_{layer}_out";
            AddNode(graph, "Add", new[] { residual1, ffnResInput }, new[] { layerOut });
            prevOutput = layerOut;
        }

        // ═══════════════════════════════════════════════════════════
        //  3. Final norm
        // ═══════════════════════════════════════════════════════════
        string finalNormOut = "final_norm_out";
        AddNorm(graph, model, weights, "output_norm", prevOutput, finalNormOut, embedDim, useRMSNorm, isGemma);

        // ═══════════════════════════════════════════════════════════
        //  4. LM head (output projection)
        // ═══════════════════════════════════════════════════════════
        // gemma2/gemma4 soft-cap the final logits: logits = cap * tanh(logits / cap). When present
        // (cap > 0) the LM head writes a pre-cap tensor and the cap is applied below to produce "logits".
        float logitCap = model.GetMetadataFloat($"{model.Architecture}.final_logit_softcapping", 0f);
        string lmHead = logitCap > 0f ? "logits_presoftcap" : "logits";
        var outputWeight = FindTensor(model, "output.weight");
        if (outputWeight != null)
        {
            ExtractWeight(model, outputWeight, weights, quantizedBytes, transposeOnUpload, isLinearB: true);
            graph.Initializers[outputWeight.Name] = outputWeight.Shape;
            AddNode(graph, "MatMul", new[] { finalNormOut, outputWeight.Name }, new[] { lmHead });
        }
        else if (embedWeight != null)
        {
            // Tied embeddings: the LM head is embed^T. The fused-MatMul orientation
            // contract (B declared [K, N], storage [N rows][K contig]) makes the RAW
            // embedding storage [vocab][n_embd] directly usable as B = [n_embd, vocab]:
            // register the SAME bytes under an alias declared in GGUF ne order. For a
            // quantized table that is zero-copy (the loader dedupes the upload, one
            // compressed buffer serves Gather AND the head). For an F32 table the alias
            // gets a one-time GPU transpose at upload. Either way: no runtime Transpose
            // node, no per-forward work.
            string headName = embedWeight.Name + "#lm_head";
            graph.Initializers[headName] = embedWeight.Shape; // ne order = [n_embd, vocab] = [K, N]
            if (quantizedBytes.TryGetValue(embedWeight.Name, out var qw))
            {
                quantizedBytes[headName] = qw; // same byte[] reference -> single GPU upload
            }
            else if (weights.TryGetValue(embedWeight.Name, out var fw))
            {
                weights[headName] = fw; // same float[] reference
                transposeOnUpload.Add(headName); // storage [vocab][n_embd] -> declared [n_embd][vocab]
            }
            AddNode(graph, "MatMul", new[] { finalNormOut, headName }, new[] { lmHead });
        }

        // ── Final logit soft-cap (gemma2/gemma4): logits = cap * tanh(logits / cap) ──
        if (logitCap > 0f)
        {
            const string capName = "logit_softcap";
            weights[capName] = new[] { logitCap };
            graph.Initializers[capName] = new[] { 1 };
            AddNode(graph, "Div", new[] { lmHead, capName }, new[] { "logits_div_cap" });
            AddNode(graph, "Tanh", new[] { "logits_div_cap" }, new[] { "logits_tanh" });
            AddNode(graph, "Mul", new[] { "logits_tanh", capName }, new[] { "logits" });
        }

        return (graph, weights, quantizedBytes, transposeOnUpload);
    }

    // ── Helpers ──

    private static GGUFTensorInfo? FindTensor(GGUFModel model, string name)
        => model.Tensors.FirstOrDefault(t => t.Name == name);

    /// <summary>Per-layer attention configuration for archs with interleaved sliding-window/global
    /// attention (gemma3/gemma4). Each field is the value the RoPE + FusedAttention node attributes use.</summary>
    public readonly record struct LayerAttnConfig(
        bool IsGlobal, int Window, float RopeBase, int RotaryDim, int NKVHeads, int HeadDim);

    /// <summary>
    /// Resolve a layer's attention config from the GGUF metadata. `sliding_window_pattern[L]`==true means a
    /// windowed (SWA) layer; ==false (or no pattern) means full/global attention. SWA layers use the *_swa
    /// metadata (window, freq_base_swa, key_length_swa, dimension_count_swa); global layers use the full
    /// values. `head_count_kv` is per-layer (an array) for gemma4 (8 sliding / 1 global) with a scalar fallback.
    /// </summary>
    public static LayerAttnConfig GetLayerAttnConfig(GGUFModel model, int layer, int nHeads, int defaultNKV, int defaultHeadDim)
    {
        string a = model.Architecture;
        bool[]? BoolArr(string k) => model.Metadata.TryGetValue(k, out var v)
            ? (v as bool[]) ?? (v is Array arr ? arr.Cast<object>().Select(Convert.ToBoolean).ToArray() : null) : null;
        int[]? IntArr(string k) => model.Metadata.TryGetValue(k, out var v)
            ? (v as int[]) ?? (v is Array arr ? arr.Cast<object>().Select(Convert.ToInt32).ToArray() : null) : null;

        var pattern = BoolArr($"{a}.attention.sliding_window_pattern");
        bool isGlobal = pattern == null || layer >= pattern.Length || !pattern[layer];

        var kv = IntArr($"{a}.attention.head_count_kv");
        int nkv = kv != null && layer < kv.Length ? kv[layer]
                : (int)model.GetMetadataInt($"{a}.attention.head_count_kv", defaultNKV);
        if (nkv <= 0) nkv = defaultNKV;

        int window = isGlobal ? 0 : (int)model.GetMetadataInt($"{a}.attention.sliding_window", 0);

        float baseFull = model.GetMetadataFloat($"{a}.rope.freq_base", 10000f);
        float ropeBase = isGlobal ? baseFull : model.GetMetadataFloat($"{a}.rope.freq_base_swa", baseFull);

        int dimFull = (int)model.GetMetadataInt($"{a}.rope.dimension_count", defaultHeadDim);
        int rotaryDim = isGlobal ? dimFull : (int)model.GetMetadataInt($"{a}.rope.dimension_count_swa", dimFull);

        int klFull = (int)model.GetMetadataInt($"{a}.attention.key_length", defaultHeadDim);
        int headDim = isGlobal ? klFull : (int)model.GetMetadataInt($"{a}.attention.key_length_swa", klFull);
        if (headDim <= 0) headDim = defaultHeadDim;
        if (rotaryDim <= 0) rotaryDim = headDim;

        return new LayerAttnConfig(isGlobal, window, ropeBase, rotaryDim, nkv, headDim);
    }

    /// <summary>
    /// Extract one tensor for GPU loading. ROLE-AWARE routing - the consumer determines
    /// what a correct representation is:
    /// - Quantized + fused-supported (Q4_0/Q8_0/Q4_K/Q6_K) + a consumer with a fused
    ///   kernel (MatMul B, Gather table): RAW BYTES + GGMLType. The tensor stays
    ///   compressed in GPU memory; the fused kernels decode blocks in-register.
    /// - Quantized otherwise: THROW. There is deliberately NO CPU-dequant fallback -
    ///   a multi-hundred-MB CPU pass is unacceptable in interpreted Blazor WASM, and a
    ///   silent F32 expansion blows VRAM. Honest failure until a kernel exists.
    /// - F32/F16: dequantized floats. Linear-B tensors are marked for a one-time GPU
    ///   transpose at upload: GGUF storage is [N rows][K contig] (ne = [K, N]) but the
    ///   declared ONNX MatMul B is [K, N] row-major. (The fused quantized path needs no
    ///   transpose - its kernels read the [N][K] storage directly; that IS the contract.)
    /// </summary>
    private static void ExtractWeight(GGUFModel model, GGUFTensorInfo tensor,
        Dictionary<string, float[]> weights,
        Dictionary<string, GGUFQuantizedWeight>? quantizedBytes = null,
        HashSet<string>? transposeOnUpload = null,
        bool isLinearB = false)
    {
        if (GGUFModel.IsQuantized(tensor.Type))
        {
            if (quantizedBytes != null && Kernels.FusedDequantMatMul.Supports(tensor.Type))
            {
                var rawBytes = model.GetTensorRawBytes(tensor)
                    ?? throw new InvalidDataException(
                        $"GGUF tensor '{tensor.Name}' ({tensor.Type}): raw data out of bounds.");
                quantizedBytes[tensor.Name] = new GGUFQuantizedWeight(rawBytes, tensor.Type);
                weights[tensor.Name] = Array.Empty<float>(); // presence marker; loader creates a ShapeOnly tensor
                return;
            }
            throw new NotSupportedException(
                $"GGUF tensor '{tensor.Name}' uses {tensor.Type}" +
                (quantizedBytes == null
                    ? " in a role with no fused dequant kernel (norm/bias)."
                    : ", which has no fused GPU dequant kernel yet (supported: Q4_0, Q8_0, Q4_K, Q6_K).") +
                " CPU dequantization is deliberately not performed (heavy CPU passes are " +
                "unacceptable in the browser). Re-quantize the model or request a kernel for this type.");
        }

        var data = model.GetTensorFloat32(tensor)
            ?? throw new NotSupportedException(
                $"GGUF tensor '{tensor.Name}' has unsupported type {tensor.Type}.");
        if (isLinearB && tensor.Dimensions.Length == 2)
            transposeOnUpload?.Add(tensor.Name);
        weights[tensor.Name] = data;
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
        string tensorPrefix, string input, string output, int dim, bool useRMSNorm, bool addOneToNormWeight = false)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            // Norm weights are 1-D vectors with no fused dequant consumer: F32/F16 only
            // (a quantized norm weight throws in ExtractWeight - honest failure).
            ExtractWeight(model, weightTensor, weights);
            // gemma RMSNorm convention: output = x_normed * (1 + weight). gemma stores the weights centered
            // at 0, so fold the +1 in at load — mathematically identical and the generic norm op needs no
            // change. (Applies to every gemma norm: attn/ffn, the post-norm sandwich, and the final norm.)
            if (addOneToNormWeight && weights.TryGetValue(weightTensor.Name, out var nw))
                for (int i = 0; i < nw.Length; i++) nw[i] += 1f;
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
        Dictionary<string, GGUFQuantizedWeight>? quantizedBytes = null,
        HashSet<string>? transposeOnUpload = null)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            // MatMul-B role: declared [K, N] (= GGUF ne order). Quantized stays raw
            // (fused kernel reads the [N][K] storage = the transpose, by contract);
            // F32/F16 gets a one-time GPU transpose at upload.
            ExtractWeight(model, weightTensor, weights, quantizedBytes, transposeOnUpload, isLinearB: true);
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
