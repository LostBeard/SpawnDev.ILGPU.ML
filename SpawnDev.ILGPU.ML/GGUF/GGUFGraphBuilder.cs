using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

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
/// <param name="Bytes">The raw quantized bytes (in-memory load). Empty when streaming — the bytes are
/// uploaded directly from the file via <paramref name="StreamOffset"/>/<paramref name="StreamByteSize"/>.</param>
/// <param name="StreamOffset">Absolute byte offset of this tensor in the streaming source, or -1 for the
/// in-memory path.</param>
public sealed record GGUFQuantizedWeight(byte[] Bytes, GGMLType Type, long StreamOffset = -1, int StreamByteSize = 0);

/// <summary>Raw NATIVE low-precision (non-block-quantized) weight bytes + the element dtype that reads
/// them. For a BF16/F16 linear weight we keep the on-disk 2-byte elements as-is and decode in-register at
/// the MAC (via <c>MatMulLowPWeight&lt;T&gt;</c>) instead of upcasting to f32 at load - the same
/// no-needless-conversion discipline as the quantized channel, but these decode through the typed
/// <c>ArrayView&lt;BFloat16&gt;</c>/<c>&lt;Half&gt;</c> path, not a fused block-dequant kernel.</summary>
/// <param name="Bytes">Raw element bytes (in-memory load); empty when streaming (see <paramref name="StreamOffset"/>).</param>
/// <param name="DType">Native element dtype (<see cref="TensorDataType.BFloat16"/> or <see cref="TensorDataType.Float16"/>).</param>
/// <param name="StreamOffset">Absolute byte offset in the streaming source, or -1 for the in-memory path.</param>
/// <param name="Transpose">True if this is a linear-B weight stored [N rows][K] that must be transposed to
/// [K, N] at upload (the declared MatMul B orientation), done natively in the element dtype.</param>
public sealed record GGUFLowPWeight(byte[] Bytes, TensorDataType DType, long StreamOffset, int StreamByteSize, bool Transpose);

public static class GGUFGraphBuilder
{
    /// <summary>
    /// When true, the LM head computes logits for ONLY the last sequence position at prefill (a Slice on the
    /// final hidden state before output_norm), turning the output projection from an M=seq GEMM into an M=1
    /// GEMV. For autoregressive generation only the last token's logits are sampled, so this is a pure waste
    /// elimination that SCALES with prompt length (the logits node is qwen's single biggest prefill node, and
    /// grows linearly with context). Decode (seq=1) is unaffected (Slice of the last of 1 row is a no-op).
    /// Opt-in (env <c>GGUF_LAST_POS=1</c>) until the full sweep promotes it; the generation consumers
    /// (<c>GgufGenerator</c>, Example 04) already read only the last position, so output is token-identical.
    /// Do NOT enable for a graph whose ALL-position logits are needed (e.g. perplexity/eval over the prompt).
    /// </summary>
    public static bool EnableLastPositionLogits =
        Environment.GetEnvironmentVariable("GGUF_LAST_POS") == "1";

    /// <param name="acceptInputsEmbeds">When true, the graph takes a pre-computed <c>inputs_embeds</c>
    /// [1, seq, n_embd] tensor and SKIPS the token Gather + gemma sqrt(n_embd) scale. The caller supplies the
    /// full embedding sequence (text rows gathered+scaled host-side, multimodal rows = RAW projected
    /// embeddings — gemma4 splices media embeddings unscaled). The default false path is the unchanged
    /// input_ids text path.</param>
    public static (ModelGraph Graph, Dictionary<string, float[]> Weights,
        Dictionary<string, GGUFQuantizedWeight> QuantizedWeights,
        HashSet<string> TransposeOnUpload,
        Dictionary<string, GGUFLowPWeight> LowPWeights) BuildGraph(GGUFModel model, bool acceptInputsEmbeds = false)
    {
        var arch = model.Architecture.ToLowerInvariant();
        var graph = new ModelGraph { Name = $"{model.Name} ({arch})" };
        var weights = new Dictionary<string, float[]>();
        var quantizedBytes = new Dictionary<string, GGUFQuantizedWeight>();
        var transposeOnUpload = new HashSet<string>();
        // Native low-precision (BF16/F16) linear weights — kept packed, decoded in-register at the MAC
        // instead of upcast to f32 at load. Travels alongside transposeOnUpload through AddLinear.
        var lowPBytes = new Dictionary<string, GGUFLowPWeight>();

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
        // gpt-oss (gptoss) uses RMSNorm (attention.layer_norm_rms_epsilon; llama.cpp openai-moe = LLM_NORM_RMS).
        bool useRMSNorm = isGemma || arch is "llama" or "mistral" or "qwen" or "qwen2" or "gptoss";
        bool useSiLU = !isGemma
            && arch is not "phi" and not "phi3" and not "gpt2" and not "falcon" and not "bloom" and not "mpt";

        graph.Outputs.Add(new GraphValueInfo { Name = "logits", Shape = new[] { 1, -1, vocabSize } });

        // ═══════════════════════════════════════════════════════════
        //  1. Token embedding lookup (or pre-computed inputs_embeds)
        // ═══════════════════════════════════════════════════════════
        // token_embd is extracted REGARDLESS of the entry mode — it doubles as the tied LM head below.
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
        }

        string prevOutput;
        if (acceptInputsEmbeds)
        {
            // Multimodal entry: the caller supplies the full [1, seq, n_embd] embedding sequence (text rows
            // gathered+scaled host-side; media rows = RAW projected embeddings). Skip Gather + gemma scale.
            graph.Inputs.Add(new GraphValueInfo { Name = "inputs_embeds", Shape = new[] { 1, -1, embedDim } });
            prevOutput = "inputs_embeds";
        }
        else
        {
            // Text entry: token IDs [1, seq_len] → Gather → (gemma) sqrt(n_embd) scale.
            graph.Inputs.Add(new GraphValueInfo { Name = "input_ids", Shape = new[] { 1, -1 } });
            prevOutput = "input_ids";
            if (embedWeight != null)
            {
                AddNode(graph, "Gather", new[] { embedWeight.Name, prevOutput }, new[] { "embed_out" });
                prevOutput = "embed_out";

                // gemma scales the token embeddings by sqrt(n_embd) right after the lookup
                // (llama.cpp `ggml_scale(inpL, sqrtf(n_embd))`; HF `inputs_embeds * hidden_size**0.5`).
                // There is no metadata key for it - it is a hardcoded gemma constant. RMSNorm is
                // scale-invariant, so the attention/FFN paths don't notice the scale; but the RESIDUAL
                // stream carries the token-identity signal, and without the ~62x boost that signal sits
                // at <1% of the (RMS-normed) sublayer outputs - so every position collapses to the same
                // argmax. Gemma-only; other archs add the embedding to the residual unscaled.
                // (Multimodal media embeddings are spliced RAW host-side, so they intentionally bypass this.)
                if (isGemma)
                {
                    const string embScaleName = "embed_scale";
                    weights[embScaleName] = new[] { MathF.Sqrt(embedDim) };
                    graph.Initializers[embScaleName] = new[] { 1 };
                    AddNode(graph, "Mul", new[] { "embed_out", embScaleName }, new[] { "embed_scaled" });
                    prevOutput = "embed_scaled";
                }
            }
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
            AddNorm(graph, model, weights, $"{pfx}.attn_norm", layerIn, normOut, embedDim, useRMSNorm);

            // ── Per-layer attention geometry ──
            // head_dim VARIES per layer (gemma4: 256 sliding / 512 global) — NEVER embedDim/nHeads.
            // GetLayerAttnConfig resolves window / rope-base / rotary-dim / KV-heads / head-dim from the
            // GGUF metadata; for llama/mistral/etc it collapses to one global config (head_dim=embedDim/nHeads,
            // base from rope.freq_base, no window). nHeads (query heads) is constant across layers.
            var cfg = GetLayerAttnConfig(model, layer, nHeads, nKVHeads, headDim);
            int hd = cfg.HeadDim;

            // ── Q, K, V projections (raw) ──
            string qOut = $"{pfx}_q", kOut = $"{pfx}_k", vOut = $"{pfx}_v";
            AddLinear(graph, model, weights, $"{pfx}.attn_q", normOut, qOut, quantizedBytes, transposeOnUpload, lowPBytes);
            AddLinear(graph, model, weights, $"{pfx}.attn_k", normOut, kOut, quantizedBytes, transposeOnUpload, lowPBytes);
            // gemma4 global layers carry no attn_v: V reuses the RAW K projection
            // (llama.cpp `Vcur = wv ? wv·x : Kcur`). All other archs always have attn_v.
            bool hasV = FindTensor(model, $"{pfx}.attn_v.weight") != null;
            if (hasV)
                AddLinear(graph, model, weights, $"{pfx}.attn_v", normOut, vOut, quantizedBytes, transposeOnUpload, lowPBytes);
            string vSrc = hasV ? vOut : kOut;

            // gemma4-style attention is signalled by the QK-norm tensors. When present we ALSO apply the
            // weightless V RMS-norm and (global layers) the rope_freqs NTK factors — all gemma4 behaviors.
            // Absent (llama/mistral/gemma2/...) = standard attention: no QK-norm, no V-norm.
            bool gemmaAttn = FindTensor(model, $"{pfx}.attn_q_norm.weight") != null;

            // rope_freqs (NTK / proportional rope) — global (full-attention) gemma4 layers only. Shared
            // model-level tensor; extract once, reference per global layer.
            string? freqFactors = null;
            if (gemmaAttn && cfg.IsGlobal)
            {
                var rf = FindTensor(model, "rope_freqs.weight");
                if (rf != null)
                {
                    if (!weights.ContainsKey(rf.Name)) { ExtractWeight(model, rf, weights); graph.Initializers[rf.Name] = rf.Shape; }
                    freqFactors = rf.Name;
                }
            }

            // Q/K/V: reshape → (QK-norm/weightless-norm) → (RoPE) — ALL drop their PRE-attention transpose now
            // (steps 2+3): FusedAttention reads Q/K/V seq-major (seq_major_q/seq_major_kv below), and the decode
            // KV-cache store is seq-major. RoPE still runs on the pre-transpose [1,seq,heads,hd] layout.
            string qReshaped = EmitAttnHead(graph, model, weights, pfx, "q", qOut, nHeads, hd, cfg,
                gemmaAttn ? $"{pfx}.attn_q_norm" : null, freqFactors, doRope: true, weightlessNorm: false,
                skipTranspose: true);
            string kReshaped = EmitAttnHead(graph, model, weights, pfx, "k", kOut, cfg.NKVHeads, hd, cfg,
                gemmaAttn ? $"{pfx}.attn_k_norm" : null, freqFactors, doRope: true, weightlessNorm: false,
                skipTranspose: true);
            string vReshaped = EmitAttnHead(graph, model, weights, pfx, "v", vSrc, cfg.NKVHeads, hd, cfg,
                qkNormTensor: null, freqFactors: null, doRope: false, weightlessNorm: gemmaAttn,
                skipTranspose: true);

            // ── Fused masked attention: softmax(QKᵀ·scale [+ causal/SWA mask]) · V in one dispatch ──
            // (GQA: n_kv_heads < n_heads; window 0 = global, else sliding.)
            // SCALE: gemma4 uses f_attention_scale = 1.0 (NOT 1/sqrt(head_dim)) - verbatim from llama.cpp
            // src/models/gemma4.cpp `load_arch_hparams`: `hparams.f_attention_scale = 1.0f; // Gemma4 uses
            // self.scaling = 1.0 (no pre-attn scaling)`. The QK-norm (attn_q_norm/attn_k_norm RMS) already
            // normalizes Q/K, so the usual 1/sqrt(d) is folded away. Omitting it (letting the op default to
            // 1/sqrt(head_dim) ≈ 0.044 for the 512-d global heads) made scores ~22x too small → softmax went
            // near-uniform → attention averaged all positions → the residual stream collapsed (cross-position
            // cosine → 1.0 by layer ~23) and content prediction degenerated to whitespace. Pass 1.0 explicitly
            // on the gemma4 QK-norm path; other archs keep the 1/sqrt(head_dim) default.
            string attnValues = $"{pfx}_attn_val";
            var faAttrs = new Dictionary<string, JsonElement>
            {
                ["n_heads"] = JsonSerializer.SerializeToElement((long)nHeads),
                ["n_kv_heads"] = JsonSerializer.SerializeToElement((long)cfg.NKVHeads),
                ["head_dim"] = JsonSerializer.SerializeToElement((long)hd),
                ["causal"] = JsonSerializer.SerializeToElement(1L),
                ["window"] = JsonSerializer.SerializeToElement((long)cfg.Window),
                ["kv_offset"] = JsonSerializer.SerializeToElement(0L),
                // layer index: lets the incremental-decode KV-cache (GGUFDecodeKVCache) associate this
                // FusedAttention node with its per-layer K/V buffer. Unused in the default full-recompute
                // forward; read only in decode mode. See Plans/gemma4-kvcache-decode-plan-2026-06-12.md.
                ["layer"] = JsonSerializer.SerializeToElement((long)layer),
                // seq_major_out: FusedAttention writes its output directly in seq-major [1,seq,heads,hd] layout
                // (kernel p[11]) so we can DROP the post-attention Transpose[0,2,1,3] below — the merged Reshape
                // then consumes the attention output directly. Universal: the group kernels scatter to the
                // seq-major base, the per-element kernels (WebGL) enumerate idx in seq-major order (own-slot
                // write, no scatter). Eliminates ~28 transpose dispatches+copies/decode-step. (Tuvok 2026-06-23.)
                ["seq_major_out"] = JsonSerializer.SerializeToElement(1L),
                // seq_major_q: Q is fed seq-major (its pre-attention transpose was dropped, step 2) so
                // FusedAttention reads Q with the seq-major base (p[12]). K/V stay heads-major (step 3 pending).
                ["seq_major_q"] = JsonSerializer.SerializeToElement(1L),
                // seq_major_kv: K/V pre-attention transposes dropped (step 3); the decode KV-cache store is seq-major.
                // FusedAttention reads K/V seq-major (p[13]: kvHead offset hd, per-token stride kvHeads*hd).
                ["seq_major_kv"] = JsonSerializer.SerializeToElement(1L),
            };
            if (gemmaAttn)
                faAttrs["scale"] = JsonSerializer.SerializeToElement(1.0f);
            // Attention sinks (gpt-oss): a per-head learned logit ([n_head]) added to the softmax
            // denominator (0 value contribution). Presence-based 4th input; absent elsewhere.
            var sinksTensor = FindTensor(model, $"{pfx}.attn_sinks");
            string[] faInputs;
            if (sinksTensor != null)
            {
                ExtractWeight(model, sinksTensor, weights);
                graph.Initializers[sinksTensor.Name] = sinksTensor.Shape;
                faInputs = new[] { qReshaped, kReshaped, vReshaped, sinksTensor.Name };
            }
            else faInputs = new[] { qReshaped, kReshaped, vReshaped };
            AddNode(graph, "FusedAttention", faInputs, new[] { attnValues }, faAttrs);

            // ── Merge heads: [1, seq, nHeads, hd] → [1, seq, nHeads*hd] ──
            // FusedAttention already wrote seq-major (seq_major_out above), so the old Transpose[0,2,1,3]
            // ([1,heads,seq,hd]→[1,seq,heads,hd]) is GONE — the Reshape (pure relabel, native CopyFrom) consumes
            // the attention output directly. (nHeads*hd, NOT embedDim: gemma4 attn_output input is 16*256=4096
            // sliding / 16*512=8192 global.)
            string attnMerged = $"{pfx}_attn_merged";
            AddNode(graph, "Reshape", new[] { attnValues }, new[] { attnMerged },
                Attrs("shape", new long[] { 1, -1, (long)(nHeads * hd) }));

            // ── Output projection ── (gpt-oss names it attn_out; llama/gemma/etc. use attn_output)
            string attnOut = $"{pfx}_attn_out";
            string attnOutPrefix = FindTensor(model, $"{pfx}.attn_output.weight") != null ? $"{pfx}.attn_output" : $"{pfx}.attn_out";
            AddLinear(graph, model, weights, attnOutPrefix, attnMerged, attnOut, quantizedBytes, transposeOnUpload, lowPBytes);

            // ── Post-attention norm (gemma 2/3/4 norm-sandwich: normalize the sublayer OUTPUT before the
            //    residual add). Presence-based — llama/mistral/etc. have no such tensor and are unaffected. ──
            string attnResInput = attnOut;
            if (FindTensor(model, $"{pfx}.post_attention_norm.weight") != null)
            {
                string postAttnOut = $"{pfx}_post_attn_norm";
                AddNorm(graph, model, weights, $"{pfx}.post_attention_norm", attnOut, postAttnOut, embedDim, useRMSNorm);
                attnResInput = postAttnOut;
            }

            // ── Residual 1 + FFN norm ──
            // Fuse the residual Add into the following RMSNorm (AddRMSNorm: 2 nodes → 1, residualOut + normedOut)
            // when it's a plain weighted RMSNorm (qwen/llama; gemma's (1+w) fold is baked into the GGUF weights so
            // RMSNorm is uniform). LayerNorm / no-weight archs keep the separate Add + AddNorm.
            string residual1 = $"{pfx}_res1";
            string ffnNormOut = $"{pfx}_ffn_norm";
            var ffnNormW = useRMSNorm ? FindTensor(model, $"{pfx}.ffn_norm.weight") : null;
            if (ffnNormW != null)
            {
                ExtractWeight(model, ffnNormW, weights);
                graph.Initializers[ffnNormW.Name] = ffnNormW.Shape;
                float rmsEps = model.GetMetadataFloat($"{model.Architecture}.attention.layer_norm_rms_epsilon", 1e-6f);
                AddNode(graph, "AddRMSNorm", new[] { layerIn, attnResInput, ffnNormW.Name },
                    new[] { residual1, ffnNormOut }, Attrs("epsilon", rmsEps));
            }
            else
            {
                AddNode(graph, "Add", new[] { layerIn, attnResInput }, new[] { residual1 });
                AddNorm(graph, model, weights, $"{pfx}.ffn_norm", residual1, ffnNormOut, embedDim, useRMSNorm);
            }

            // ── FFN: dense gate/up → activation → down, OR a Mixture-of-Experts block when the model carries
            //    a router (ffn_gate_inp, e.g. gpt-oss). ──
            string ffnOut = $"{pfx}_ffn_out";
            if (FindTensor(model, $"{pfx}.ffn_gate_inp.weight") != null)
            {
                AddMoEFFN(graph, model, weights, pfx, ffnNormOut, ffnOut, quantizedBytes, transposeOnUpload);
            }
            else
            {
                // ── Dense FFN: gate + up → activation → down ──
                string gateOut = $"{pfx}_gate", upOut = $"{pfx}_up";
                AddLinear(graph, model, weights, $"{pfx}.ffn_gate", ffnNormOut, gateOut, quantizedBytes, transposeOnUpload, lowPBytes);
                AddLinear(graph, model, weights, $"{pfx}.ffn_up", ffnNormOut, upOut, quantizedBytes, transposeOnUpload, lowPBytes);

                string activated;
                if (useSiLU)
                {
                    // Fused SwiGLU: (gate · sigmoid(gate)) · up in ONE kernel — was Sigmoid + Mul + Mul (3
                    // dispatches/layer → 1; 56 fewer dispatches/decode-step, biggest on WebGPU). Bit-identical.
                    activated = $"{pfx}_ffn_act";
                    AddNode(graph, "SwiGLU", new[] { gateOut, upOut }, new[] { activated });
                }
                else
                {
                    // GELU
                    string geluOut = $"{pfx}_gate_gelu";
                    AddNode(graph, "Gelu", new[] { gateOut }, new[] { geluOut });
                    activated = $"{pfx}_ffn_act";
                    AddNode(graph, "Mul", new[] { geluOut, upOut }, new[] { activated });
                }

                AddLinear(graph, model, weights, $"{pfx}.ffn_down", activated, ffnOut, quantizedBytes, transposeOnUpload, lowPBytes);
            }

            // ── Post-FFN norm (gemma norm-sandwich), presence-based. ──
            string ffnResInput = ffnOut;
            if (FindTensor(model, $"{pfx}.post_ffw_norm.weight") != null)
            {
                string postFfwOut = $"{pfx}_post_ffw_norm";
                AddNorm(graph, model, weights, $"{pfx}.post_ffw_norm", ffnOut, postFfwOut, embedDim, useRMSNorm);
                ffnResInput = postFfwOut;
            }

            // ── Residual 2 ──
            string layerOut = $"block_{layer}_out";
            AddNode(graph, "Add", new[] { residual1, ffnResInput }, new[] { layerOut });

            // ── Per-layer output scalar (gemma4 layer_output_scale) ──
            // llama.cpp: `if (out_scale) cur *= out_scale` — a per-layer [1] scalar multiply on the WHOLE
            // block output, AFTER residual-2 (not an attention/logit scale). Presence-based; absent elsewhere.
            var outScale = FindTensor(model, $"{pfx}.layer_output_scale.weight");
            if (outScale != null)
            {
                ExtractWeight(model, outScale, weights);
                graph.Initializers[outScale.Name] = outScale.Shape; // [1]
                string scaledOut = $"{pfx}_out_scaled";
                AddNode(graph, "Mul", new[] { layerOut, outScale.Name }, new[] { scaledOut });
                prevOutput = scaledOut;
            }
            else
            {
                prevOutput = layerOut;
            }
        }

        // ═══════════════════════════════════════════════════════════
        //  3. Final norm
        // ═══════════════════════════════════════════════════════════
        // Last-position-only logits (opt-in): slice the final hidden state [1, seq, n_embd] to its LAST seq
        // position [1, 1, n_embd] BEFORE output_norm, so output_norm + the LM head run at M=1 (a GEMV) instead
        // of M=seq. Only the last token's logits are sampled in generation, so this is waste elimination that
        // scales with prompt length. Slice on axis 1 (seq) with start=-1 → the last row, resolved against the
        // concrete seq dim each shape-recompile (so it is [1,1,n_embd] at any prefill length and a no-op at
        // seq=1 decode). RMSNorm is per-row, so slice-then-norm == norm-then-slice for the kept row.
        string headInput = prevOutput;
        if (EnableLastPositionLogits)
        {
            AddNode(graph, "Slice", new[] { prevOutput }, new[] { "last_token_hidden" },
                new Dictionary<string, JsonElement>
                {
                    ["starts"] = JsonSerializer.SerializeToElement(new long[] { -1 }),
                    ["ends"] = JsonSerializer.SerializeToElement(new long[] { int.MaxValue }),
                    ["axes"] = JsonSerializer.SerializeToElement(new long[] { 1 }),
                    ["steps"] = JsonSerializer.SerializeToElement(new long[] { 1 }),
                });
            headInput = "last_token_hidden";
        }

        string finalNormOut = "final_norm_out";
        AddNorm(graph, model, weights, "output_norm", headInput, finalNormOut, embedDim, useRMSNorm);

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
            ExtractWeight(model, outputWeight, weights, quantizedBytes, transposeOnUpload, isLinearB: true, lowPBytes: lowPBytes);
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
                quantizedBytes[headName] = qw; // same byte[]/stream-offset -> single GPU upload (deduped)
                weights[headName] = Array.Empty<float>(); // presence marker so the upload loop (iterates `weights`) processes the alias
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

        return (graph, weights, quantizedBytes, transposeOnUpload, lowPBytes);
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

        // Resolve the REAL head_dim first (key_length), then default the rope dimension to IT — not to
        // embedDim/nHeads. gpt-oss heads don't tile the embedding (64 heads x 64 != 2880), so the caller's
        // defaultHeadDim = embedDim/nHeads = 45 is wrong; key_length = 64 is the truth. With no explicit
        // rope.dimension_count, rotary = full head_dim (llama.cpp openai-moe n_rot = key_length).
        int klFull = (int)model.GetMetadataInt($"{a}.attention.key_length", defaultHeadDim);
        int headDim = isGlobal ? klFull : (int)model.GetMetadataInt($"{a}.attention.key_length_swa", klFull);
        if (headDim <= 0) headDim = defaultHeadDim;

        int dimFull = (int)model.GetMetadataInt($"{a}.rope.dimension_count", headDim);
        int rotaryDim = isGlobal ? dimFull : (int)model.GetMetadataInt($"{a}.rope.dimension_count_swa", dimFull);
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
        bool isLinearB = false,
        Dictionary<string, GGUFLowPWeight>? lowPBytes = null)
    {
        if (GGUFModel.IsQuantized(tensor.Type))
        {
            if (quantizedBytes != null && Kernels.FusedDequantMatMul.Supports(tensor.Type))
            {
                if (model.SourceStream != null)
                {
                    // Streaming load: record the file offset/size — the bytes stay on disk until the upload
                    // loop streams them straight to a GPU byte buffer (never materialized; a 7 GB model has no
                    // single byte[]). The embedding (Q6_K, ~787 MB) fits int; assert so a future giant tensor
                    // fails loudly rather than silently truncating.
                    long absOffset = model.GetTensorDataOffset(tensor);
                    long byteSize = GGMLTypes.TypeSize(tensor.Type, model.GetTensorElementCount(tensor));
                    if (byteSize > int.MaxValue)
                        throw new NotSupportedException(
                            $"GGUF tensor '{tensor.Name}' is {byteSize} bytes (> 2 GB) — single-tensor streaming upload not supported.");
                    quantizedBytes[tensor.Name] = new GGUFQuantizedWeight(Array.Empty<byte>(), tensor.Type, absOffset, (int)byteSize);
                }
                else
                {
                    var rawBytes = model.GetTensorRawBytes(tensor)
                        ?? throw new InvalidDataException(
                            $"GGUF tensor '{tensor.Name}' ({tensor.Type}): raw data out of bounds.");
                    quantizedBytes[tensor.Name] = new GGUFQuantizedWeight(rawBytes, tensor.Type);
                }
                weights[tensor.Name] = Array.Empty<float>(); // presence marker; loader creates a ShapeOnly tensor
                return;
            }
            throw new NotSupportedException(
                $"GGUF tensor '{tensor.Name}' uses {tensor.Type}" +
                (quantizedBytes == null
                    ? " in a role with no fused dequant kernel (norm/bias)."
                    : ", which has no fused GPU dequant kernel yet (supported: Q4_0, Q8_0, Q4_K, Q6_K, MXFP4).") +
                " CPU dequantization is deliberately not performed (heavy CPU passes are " +
                "unacceptable in the browser). Re-quantize the model or request a kernel for this type.");
        }

        // Native low-precision LINEAR weight: keep BF16/F16 elements packed and decode in-register at the
        // MAC (MatMulLowPWeight<T>) instead of upcasting to f32 at load - halves the weight's VRAM +
        // upload bandwidth (no-needless-conversion, the same discipline as the quantized channel). Only
        // 2-D linear-B weights take this path (the MatMul B path has a native low-p kernel); norms, biases
        // and the embedding/Gather table stay f32 (tiny, or no native kernel). Transpose is done natively
        // in the element dtype at upload (GGUF [N][K] storage -> declared MatMul B [K, N]).
        if (lowPBytes != null && isLinearB && tensor.Dimensions.Length == 2
            && NativeLowPDType(tensor.Type) is TensorDataType lowPType)
        {
            long elements = model.GetTensorElementCount(tensor);
            if (model.SourceStream != null)
            {
                long absOffset = model.GetTensorDataOffset(tensor);
                long byteSize = GGMLTypes.TypeSize(tensor.Type, elements);
                if (byteSize > int.MaxValue)
                    throw new NotSupportedException(
                        $"GGUF tensor '{tensor.Name}' is {byteSize} bytes (> 2 GB) — single-tensor streaming upload not supported.");
                lowPBytes[tensor.Name] = new GGUFLowPWeight(Array.Empty<byte>(), lowPType, absOffset, (int)byteSize, Transpose: true);
            }
            else
            {
                var rawBytes = model.GetTensorRawBytes(tensor)
                    ?? throw new InvalidDataException(
                        $"GGUF tensor '{tensor.Name}' ({tensor.Type}): raw data out of bounds.");
                lowPBytes[tensor.Name] = new GGUFLowPWeight(rawBytes, lowPType, -1, 0, Transpose: true);
            }
            weights[tensor.Name] = Array.Empty<float>(); // presence marker; loader builds a FromLowP tensor (no f32 buffer)
            return;
        }

        var data = model.GetTensorFloat32(tensor)
            ?? throw new NotSupportedException(
                $"GGUF tensor '{tensor.Name}' has unsupported type {tensor.Type}.");
        if (isLinearB && tensor.Dimensions.Length == 2)
            transposeOnUpload?.Add(tensor.Name);
        weights[tensor.Name] = data;
    }

    /// <summary>The non-block-quantized element types we can keep NATIVE (decode in-register at the MAC)
    /// instead of upcasting to f32 at load — maps the GGML element type to the consumer
    /// <see cref="TensorDataType"/>. Returns null for any type that should stay f32 (e.g. F32 itself) or
    /// goes through the block-quant channel.</summary>
    private static TensorDataType? NativeLowPDType(GGMLType t) => t switch
    {
        GGMLType.BF16 => TensorDataType.BFloat16,
        GGMLType.F16 => TensorDataType.Float16,
        _ => null,
    };

    /// <summary>
    /// Emit one attention head-stream as gemma4/llama.cpp wires it (verbatim-matched to
    /// src/models/gemma4.cpp): projection output → Reshape [1, seq, heads, hd] →
    /// (optional QK-norm = weighted RMS over hd) → (optional weightless V RMS-norm over hd) →
    /// (optional RoPE on the PRE-transpose layout, rows_per_position = heads) →
    /// Transpose [0,2,1,3] → [1, heads, seq, hd] (the flat layout FusedAttention expects).
    /// Q/K take a QK-norm + RoPE; V takes the weightless norm and NO RoPE.
    /// </summary>
    private static string EmitAttnHead(ModelGraph graph, GGUFModel model, Dictionary<string, float[]> weights,
        string pfx, string tag, string projOut, int heads, int hd, LayerAttnConfig cfg,
        string? qkNormTensor, string? freqFactors, bool doRope, bool weightlessNorm, bool skipTranspose = false)
    {
        string r4d = $"{pfx}_{tag}_4d";
        AddNode(graph, "Reshape", new[] { projOut }, new[] { r4d },
            Attrs("shape", new long[] { 1, -1, heads, hd }));
        string cur = r4d;

        // QK-norm: weighted RMS over head_dim, BEFORE RoPE (gemma: (1+weight) fold). Presence-gated.
        if (qkNormTensor != null && FindTensor(model, $"{qkNormTensor}.weight") != null)
        {
            string normed = $"{pfx}_{tag}_qknorm";
            AddNorm(graph, model, weights, qkNormTensor, cur, normed, hd, useRMSNorm: true);
            cur = normed;
        }

        // Weightless V RMS-norm (gemma4: `Vcur = ggml_rms_norm(Vcur, eps)`, no learned scale).
        if (weightlessNorm)
        {
            float rmsEps = model.GetMetadataFloat($"{model.Architecture}.attention.layer_norm_rms_epsilon", 1e-6f);
            string vn = $"{pfx}_{tag}_vnorm";
            AddNode(graph, "RMSNormalization", new[] { cur }, new[] { vn }, Attrs("epsilon", rmsEps));
            cur = vn;
        }

        // RoPE on the pre-transpose [1, seq, heads, hd] layout (rows_per_position = heads). Global gemma4
        // layers also pass freq_factors (rope_freqs, NTK); sliding/other archs pass none.
        if (doRope)
        {
            string roped = $"{pfx}_{tag}_roped";
            var ropeAttrs = new Dictionary<string, JsonElement>
            {
                ["rope_base"] = JsonSerializer.SerializeToElement(cfg.RopeBase),
                ["rotary_dim"] = JsonSerializer.SerializeToElement((long)cfg.RotaryDim),
                ["rows_per_position"] = JsonSerializer.SerializeToElement((long)heads),
                ["kv_offset"] = JsonSerializer.SerializeToElement(0L),
            };
            var ins = freqFactors != null ? new[] { cur, freqFactors } : new[] { cur };
            AddNode(graph, "RoPE", ins, new[] { roped }, ropeAttrs);
            cur = roped;
        }

        // skipTranspose: the FusedAttention consumer reads this input SEQ-major (p[12] for Q), so the
        // [1,seq,heads,hd]→[1,heads,seq,hd] Transpose is dropped — return the pre-transpose (seq-major) tensor.
        if (skipTranspose)
            return cur;

        string t = $"{pfx}_{tag}_t";
        AddNode(graph, "Transpose", new[] { cur }, new[] { t }, Attrs("perm", new long[] { 0, 2, 1, 3 }));
        return t;
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
        string tensorPrefix, string input, string output, int dim, bool useRMSNorm)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            // Norm weights are 1-D vectors with no fused dequant consumer: F32/F16 only
            // (a quantized norm weight throws in ExtractWeight - honest failure).
            // gemma's `output = x_normed * (1 + weight)` convention is ALREADY BAKED INTO THE GGUF:
            // llama.cpp's convert (conversion/gemma.py, Gemma3/Gemma4 `modify_tensors`) does
            // `data_torch = data_torch + 1` for every `*norm.weight` at conversion, and the graph then
            // uses the stored weight RAW. So the GGUF weights are NOT centered at 0 (verified: gemma4
            // attn_norm ranges -143..193, output_norm up to 604) - they are the final gains. Folding a
            // second +1 here double-counted; for the small-valued qk/post norms (k_norm = 0.1221) that
            // ~9x-inflated the gain and collapsed attention. Use the stored weights verbatim.
            ExtractWeight(model, weightTensor, weights);
            graph.Initializers[weightTensor.Name] = weightTensor.Shape;
        }

        if (useRMSNorm)
        {
            // TRUE RMSNorm (no mean-centering). Emitting "LayerNormalization" here was a latent bug:
            // it routed to the mean-centered LayerNorm kernel (wrong) and read an absent bias. Use the
            // model's RMS epsilon (gemma = 1e-6); the RMSNormalization op defaults to 1e-6 too.
            float rmsEps = model.GetMetadataFloat($"{model.Architecture}.attention.layer_norm_rms_epsilon", 1e-6f);
            AddNode(graph, "RMSNormalization",
                weightTensor != null ? new[] { input, weightTensor.Name } : new[] { input },
                new[] { output },
                Attrs("epsilon", rmsEps));
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
        HashSet<string>? transposeOnUpload = null,
        Dictionary<string, GGUFLowPWeight>? lowPBytes = null)
    {
        var weightTensor = FindTensor(model, $"{tensorPrefix}.weight");
        if (weightTensor != null)
        {
            // MatMul-B role: declared [K, N] (= GGUF ne order). Quantized stays raw
            // (fused kernel reads the [N][K] storage = the transpose, by contract);
            // BF16/F16 stays NATIVE (decode in-register, transposed at upload in the element dtype);
            // F32 gets a one-time GPU transpose at upload.
            ExtractWeight(model, weightTensor, weights, quantizedBytes, transposeOnUpload, isLinearB: true, lowPBytes: lowPBytes);
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

    /// <summary>
    /// Emit a Mixture-of-Experts FFN block (gpt-oss / OpenAI-MoE) as one fused "MoE" node. Extracts the
    /// router (ffn_gate_inp +bias, F32) and the per-expert gate/up/down weights (+biases) — the expert
    /// weights are typically MXFP4 (stay raw, decode in-register via FusedDequantMatMul; the MoEOperator
    /// slices per-expert). Input order matches MoEOperator: [x, gate_inp, gate_inp.b, gate_exps, gate_exps.b,
    /// up_exps, up_exps.b, down_exps, down_exps.b]. n_ff is derived from the gate_exps shape (ggml
    /// [n_embd, n_ff, n_expert]); n_expert / n_expert_used from arch metadata. gpt-oss = SwiGLU-OAI
    /// (alpha 1.702, limit 7) — the MoEOperator defaults match, so no override needed.
    /// </summary>
    private static void AddMoEFFN(ModelGraph graph, GGUFModel model, Dictionary<string, float[]> weights,
        string pfx, string input, string output,
        Dictionary<string, GGUFQuantizedWeight>? quantizedBytes, HashSet<string>? transposeOnUpload)
    {
        string Reg(string suffix)
        {
            var t = FindTensor(model, $"{pfx}.{suffix}")
                ?? throw new InvalidDataException($"MoE layer '{pfx}': required tensor '{pfx}.{suffix}' missing.");
            // exps are quantized (MXFP4) -> raw bytes via quantizedBytes; router + biases are F32. No transpose:
            // the MoEOperator reads the ggml [N][K] / [.,n_expert] layout natively (FusedDequant + in-op transpose).
            ExtractWeight(model, t, weights, quantizedBytes, transposeOnUpload);
            graph.Initializers[t.Name] = t.Shape;
            return t.Name;
        }

        var gateExps = FindTensor(model, $"{pfx}.ffn_gate_exps.weight")
            ?? throw new InvalidDataException($"MoE layer '{pfx}': ffn_gate_exps.weight missing.");
        // ggml expert weight shape [n_embd, n_ff, n_expert]; n_ff = Dimensions[1].
        int nFf = gateExps.Dimensions.Length >= 2 ? (int)gateExps.Dimensions[1] : 0;
        string a = model.Architecture;
        int nExpert = (int)model.GetMetadataInt($"{a}.expert_count", gateExps.Dimensions.Length >= 3 ? (int)gateExps.Dimensions[2] : 0);
        int nExpertUsed = (int)model.GetMetadataInt($"{a}.expert_used_count", 0);
        if (nFf <= 0 || nExpert <= 0 || nExpertUsed <= 0)
            throw new InvalidDataException($"MoE layer '{pfx}': bad shape/hparams n_ff={nFf} n_expert={nExpert} n_expert_used={nExpertUsed}.");

        var inputs = new[]
        {
            input,
            Reg("ffn_gate_inp.weight"), Reg("ffn_gate_inp.bias"),
            Reg("ffn_gate_exps.weight"), Reg("ffn_gate_exps.bias"),
            Reg("ffn_up_exps.weight"), Reg("ffn_up_exps.bias"),
            Reg("ffn_down_exps.weight"), Reg("ffn_down_exps.bias"),
        };
        var attrs = new Dictionary<string, JsonElement>
        {
            ["n_expert"] = JsonSerializer.SerializeToElement((long)nExpert),
            ["n_expert_used"] = JsonSerializer.SerializeToElement((long)nExpertUsed),
            ["n_ff"] = JsonSerializer.SerializeToElement((long)nFf),
        };
        AddNode(graph, "MoE", inputs, new[] { output }, attrs);
    }
}
