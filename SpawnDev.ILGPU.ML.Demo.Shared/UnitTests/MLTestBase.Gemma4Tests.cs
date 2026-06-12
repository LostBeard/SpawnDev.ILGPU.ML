using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// gemma4 bring-up — graph-builder STRUCTURE tests. The runtime attention pieces (dual-base RoPE,
/// sliding-window/global mask, QK-norm) are verified separately once Seven's kernels land; these lock
/// the architecture-recognition + graph-shape contracts that don't need a GPU.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Gemma4_GraphBuilder_UsesRMSNormAndGeGLU() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4Model(withPostNorms: false));

        // The gemma family uses a GeGLU MLP (a Gelu node), NOT SiLU (no Sigmoid). The same isGemma flag
        // also selects RMSNorm, so the activation choice transitively confirms gemma4 was recognized
        // rather than falling to the llama-style SiLU / LayerNorm path it used before.
        if (!graph.Nodes.Any(n => n.OpType == "Gelu"))
            throw new Exception("gemma4 must build a GeGLU MLP (Gelu node) — arch not recognized as gemma.");
        if (graph.Nodes.Any(n => n.OpType == "Sigmoid"))
            throw new Exception("gemma4 must NOT use SiLU (Sigmoid present) — falsely treated as llama-style.");

        Console.WriteLine("[Gemma4] arch recognized -> RMSNorm + GeGLU graph");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_GraphBuilder_PostNormSandwich() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4Model(withPostNorms: true));

        // gemma 2/3/4 norm-sandwich: each sublayer's OUTPUT is normed BEFORE the residual add —
        //   residual = x + post_attention_norm(attn);   out = residual + post_ffw_norm(ffn).
        if (!graph.Nodes.Any(n => n.OpType == "Add" && n.Inputs.Contains("blk.0_post_attn_norm")))
            throw new Exception("post_attention_norm output must feed the attention residual Add (norm-sandwich).");
        if (!graph.Nodes.Any(n => n.OpType == "Add" && n.Inputs.Contains("blk.0_post_ffw_norm")))
            throw new Exception("post_ffw_norm output must feed the FFN residual Add (norm-sandwich).");
        // The raw attention output must NOT bypass post_attention_norm straight into the residual.
        if (graph.Nodes.Any(n => n.OpType == "Add" && n.Inputs.Contains("blk.0_attn_out")))
            throw new Exception("attention output bypassed post_attention_norm into the residual.");

        Console.WriteLine("[Gemma4] 4-norm sandwich wired (post-attention + post-ffn norms before residuals)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_GraphBuilder_LogitSoftCap() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4Model(withPostNorms: false, logitSoftCap: 30f));

        // gemma4 final logit soft-cap: logits = cap * tanh(logits / cap). The LM-head MatMul writes a
        // pre-cap tensor; a Tanh + Mul produce the final "logits".
        if (!graph.Nodes.Any(n => n.OpType == "MatMul" && n.Outputs.Contains("logits_presoftcap")))
            throw new Exception("LM head must write a pre-cap tensor when final_logit_softcapping is set.");
        if (!graph.Nodes.Any(n => n.OpType == "Tanh"))
            throw new Exception("logit soft-cap missing its Tanh node.");
        if (!graph.Nodes.Any(n => n.OpType == "Mul" && n.Outputs.Contains("logits")))
            throw new Exception("final 'logits' must be produced by the soft-cap Mul (cap * tanh).");

        // Without a cap, "logits" comes straight from the MatMul (no Tanh).
        var (plain, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4Model(withPostNorms: false));
        if (plain.Nodes.Any(n => n.OpType == "Tanh"))
            throw new Exception("no soft-cap metadata -> no Tanh expected.");
        if (!plain.Nodes.Any(n => n.OpType == "MatMul" && n.Outputs.Contains("logits")))
            throw new Exception("without a cap, the LM-head MatMul should output 'logits' directly.");

        Console.WriteLine("[Gemma4] final logit soft-cap wired (cap * tanh(logits / cap))");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_GraphBuilder_RMSNormPlusOne() => await RunTest(async accelerator =>
    {
        var (_, weights, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4Model(withPostNorms: true));

        // gemma RMSNorm convention output = x_normed * (1 + weight); the builder folds the +1 into the
        // weight at load. Every norm (attn/ffn + the post-norm sandwich + the final norm) gets it.
        foreach (var name in new[]
        {
            "blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight", "blk.0.post_ffw_norm.weight", "output_norm.weight",
        })
        {
            var w = weights[name];
            if (Math.Abs(w[0] - 1.05f) > 1e-5f)  // raw 0.05 + 1
                throw new Exception($"{name}: gemma RMSNorm weight should be raw+1 (0.05+1=1.05), got {w[0]}.");
        }

        Console.WriteLine("[Gemma4] RMSNorm (1+weight) offset folded at load for all norms");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_PerLayerAttnConfig_SwaGlobalInterleave() => await RunTest(async accelerator =>
    {
        // Only metadata is needed — GetLayerAttnConfig reads config, not tensors.
        var model = new GGUFModel
        {
            Metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "gemma4",
                ["gemma4.attention.sliding_window_pattern"] = new[] { true, true, true, true, true, false }, // 5:1
                ["gemma4.attention.head_count_kv"] = new[] { 8, 8, 8, 8, 8, 1 },
                ["gemma4.attention.sliding_window"] = 1024L,
                ["gemma4.rope.freq_base"] = 1000000f,
                ["gemma4.rope.freq_base_swa"] = 10000f,
                ["gemma4.attention.key_length"] = 512L,
                ["gemma4.attention.key_length_swa"] = 256L,
                ["gemma4.rope.dimension_count"] = 512L,
                ["gemma4.rope.dimension_count_swa"] = 256L,
            },
        };

        var swa = GGUFGraphBuilder.GetLayerAttnConfig(model, layer: 0, nHeads: 16, defaultNKV: 8, defaultHeadDim: 256);
        if (swa.IsGlobal || swa.Window != 1024 || Math.Abs(swa.RopeBase - 10000f) > 1f
            || swa.NKVHeads != 8 || swa.HeadDim != 256 || swa.RotaryDim != 256)
            throw new Exception($"sliding layer 0 config wrong: {swa}");

        var glb = GGUFGraphBuilder.GetLayerAttnConfig(model, layer: 5, nHeads: 16, defaultNKV: 8, defaultHeadDim: 256);
        if (!glb.IsGlobal || glb.Window != 0 || Math.Abs(glb.RopeBase - 1000000f) > 1f
            || glb.NKVHeads != 1 || glb.HeadDim != 512 || glb.RotaryDim != 512)
            throw new Exception($"global layer 5 config wrong: {glb}");

        Console.WriteLine("[Gemma4] per-layer SWA/global attn config (window/base/KV-heads/head-dim) correct");
        await Task.CompletedTask;
    });

    // ── Attention-emission structure (RoPE + QK-norm + FusedAttention + V-reuse + layer_output_scale) ──
    // These lock the gemma4 per-layer attention wiring that BuildGraph emits (verbatim-matched to
    // llama.cpp src/models/gemma4.cpp). Runtime numerics are covered by the operator execution tests
    // (AttnOp*, RMSNorm*) + the eventual E2E; these assert the GRAPH SHAPE without a GPU.

    [TestMethod]
    public async Task Gemma4_Attn_EmitsRoPEAndFusedAttention_NotExplicitSoftmax() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4AttnModel());

        // The fused path replaces the old explicit Q@Kᵀ / scale-Mul / Softmax / @V block.
        if (!graph.Nodes.Any(n => n.OpType == "FusedAttention"))
            throw new Exception("gemma4 attention must emit a FusedAttention node.");
        if (!graph.Nodes.Any(n => n.OpType == "RoPE"))
            throw new Exception("gemma4 attention must emit RoPE nodes.");
        if (graph.Nodes.Any(n => n.OpType == "Softmax"))
            throw new Exception("the explicit-softmax attention block must be gone (FusedAttention owns the softmax).");

        // Every norm must be TRUE RMSNorm, never the mean-centered LayerNormalization (the floor bug).
        if (graph.Nodes.Any(n => n.OpType == "LayerNormalization"))
            throw new Exception("gemma4 must use RMSNormalization for all norms, not mean-centered LayerNormalization.");
        if (!graph.Nodes.Any(n => n.OpType == "RMSNormalization"))
            throw new Exception("gemma4 norms must be RMSNormalization.");

        Console.WriteLine("[Gemma4] attention emits RoPE + FusedAttention + RMSNormalization (no explicit softmax / LayerNorm)");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_Attn_QKNorm_BeforeRoPE() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4AttnModel());

        // QK-norm: RMSNormalization on Q and K consuming attn_q_norm / attn_k_norm, feeding the RoPE node.
        var qNorm = graph.Nodes.FirstOrDefault(n => n.OpType == "RMSNormalization"
            && n.Inputs.Contains("blk.0.attn_q_norm.weight"));
        if (qNorm == null) throw new Exception("layer 0 Q must be RMSNorm'd by attn_q_norm (QK-norm).");
        var kNorm = graph.Nodes.FirstOrDefault(n => n.OpType == "RMSNormalization"
            && n.Inputs.Contains("blk.0.attn_k_norm.weight"));
        if (kNorm == null) throw new Exception("layer 0 K must be RMSNorm'd by attn_k_norm (QK-norm).");

        // The QK-norm output must feed a RoPE node (norm BEFORE rope).
        if (!graph.Nodes.Any(n => n.OpType == "RoPE" && n.Inputs.Contains(qNorm.Outputs[0])))
            throw new Exception("QK-norm(Q) output must feed RoPE (QK-norm precedes RoPE).");

        Console.WriteLine("[Gemma4] QK-norm (RMSNorm over head_dim) precedes RoPE on Q and K");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_Attn_GlobalLayer_VReuseAndFreqFactors() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4AttnModel());

        // Layer 0 (sliding) HAS its own attn_v projection; layer 1 (global) does NOT — V reuses the raw K
        // projection (llama.cpp `Vcur = wv ? wv·x : Kcur`).
        if (!graph.Nodes.Any(n => n.OpType == "MatMul" && n.Inputs.Contains("blk.0.attn_v.weight")))
            throw new Exception("sliding layer 0 must project its own V (attn_v MatMul).");
        if (graph.Nodes.Any(n => n.Inputs.Contains("blk.1.attn_v.weight")))
            throw new Exception("global layer 1 has no attn_v — V must reuse the K projection, not project.");

        // Global-layer RoPE carries freq_factors (rope_freqs.weight) as its 2nd input; sliding-layer RoPE does not.
        var globalRope = graph.Nodes.Where(n => n.OpType == "RoPE" && n.Outputs[0].StartsWith("blk.1_")).ToList();
        if (globalRope.Count == 0 || !globalRope.All(n => n.Inputs.Count == 2 && n.Inputs[1] == "rope_freqs.weight"))
            throw new Exception("global layer 1 RoPE must pass rope_freqs.weight (NTK freq_factors) as its 2nd input.");
        var slidingRope = graph.Nodes.Where(n => n.OpType == "RoPE" && n.Outputs[0].StartsWith("blk.0_")).ToList();
        if (slidingRope.Count == 0 || slidingRope.Any(n => n.Inputs.Count != 1))
            throw new Exception("sliding layer 0 RoPE must NOT pass freq_factors (1 input only).");

        Console.WriteLine("[Gemma4] global layer reuses K-as-V (no attn_v) + RoPE freq_factors; sliding layer does neither");
        await Task.CompletedTask;
    });

    [TestMethod]
    public async Task Gemma4_Attn_LayerOutputScale_AfterResidual2() => await RunTest(async accelerator =>
    {
        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(MakeGemma4AttnModel());

        // layer_output_scale: a per-layer scalar Mul on the WHOLE block output, after the FFN residual add.
        var scaleMul = graph.Nodes.FirstOrDefault(n => n.OpType == "Mul"
            && n.Inputs.Contains("blk.0.layer_output_scale.weight"));
        if (scaleMul == null)
            throw new Exception("layer_output_scale must multiply the block output (Mul with the [1] scalar).");
        // Its input must be the post-FFN residual (block_0_out), and its output is what the next layer consumes.
        if (!scaleMul.Inputs.Contains("block_0_out"))
            throw new Exception("layer_output_scale must apply AFTER residual-2 (block output), not mid-block.");

        Console.WriteLine("[Gemma4] layer_output_scale applies after residual-2 (per-layer block-output scalar)");
        await Task.CompletedTask;
    });

    /// <summary>2-layer gemma4 with FULL attention structure: layer 0 SLIDING (has attn_v, 2 KV heads,
    /// head_dim 2, base 1e4), layer 1 GLOBAL (no attn_v, 1 KV head, head_dim 4, base 1e6 + rope_freqs).
    /// Both carry QK-norm (attn_q_norm/attn_k_norm) and layer_output_scale. Dims are tiny but consistent;
    /// BuildGraph only emits nodes so shape validity isn't executed — the wiring is what's asserted.</summary>
    private static GGUFModel MakeGemma4AttnModel()
    {
        const int embd = 8, ffn = 16, vocab = 4, nH = 2;
        const int hdS = 2, hdG = 4, kvS = 2, kvG = 1; // sliding/global head_dim + KV heads
        var raw = new List<byte>();
        var tensors = new List<GGUFTensorInfo>();
        void Add(string name, long[] ne)
        {
            long elems = 1; foreach (var d in ne) elems *= d;
            tensors.Add(new GGUFTensorInfo { Name = name, Dimensions = ne, Type = GGMLType.F32, DataOffset = (ulong)raw.Count });
            for (long i = 0; i < elems; i++) raw.AddRange(BitConverter.GetBytes(0.05f));
        }
        void Layer(int l, bool global)
        {
            int hd = global ? hdG : hdS, kv = global ? kvG : kvS;
            string p = $"blk.{l}";
            Add($"{p}.attn_norm.weight", new long[] { embd });
            Add($"{p}.attn_q.weight", new long[] { embd, nH * hd });
            Add($"{p}.attn_k.weight", new long[] { embd, kv * hd });
            if (!global) Add($"{p}.attn_v.weight", new long[] { embd, kv * hd }); // global has NO attn_v
            Add($"{p}.attn_q_norm.weight", new long[] { hd });
            Add($"{p}.attn_k_norm.weight", new long[] { hd });
            Add($"{p}.attn_output.weight", new long[] { nH * hd, embd });
            Add($"{p}.post_attention_norm.weight", new long[] { embd });
            Add($"{p}.ffn_norm.weight", new long[] { embd });
            Add($"{p}.ffn_gate.weight", new long[] { embd, ffn });
            Add($"{p}.ffn_up.weight", new long[] { embd, ffn });
            Add($"{p}.ffn_down.weight", new long[] { ffn, embd });
            Add($"{p}.post_ffw_norm.weight", new long[] { embd });
            Add($"{p}.layer_output_scale.weight", new long[] { 1 });
        }
        Add("token_embd.weight", new long[] { embd, vocab });
        Layer(0, global: false);
        Layer(1, global: true);
        Add("rope_freqs.weight", new long[] { hdG / 2 });   // NTK factors for the global layer
        Add("output_norm.weight", new long[] { embd });

        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "gemma4",
            ["gemma4.embedding_length"] = (long)embd,
            ["gemma4.block_count"] = 2L,
            ["gemma4.attention.head_count"] = (long)nH,
            ["gemma4.attention.head_count_kv"] = new[] { kvS, kvG },
            ["gemma4.attention.sliding_window_pattern"] = new[] { true, false }, // L0 sliding, L1 global
            ["gemma4.attention.sliding_window"] = 8L,
            ["gemma4.attention.key_length"] = (long)hdG,
            ["gemma4.attention.key_length_swa"] = (long)hdS,
            ["gemma4.rope.dimension_count"] = (long)hdG,
            ["gemma4.rope.dimension_count_swa"] = (long)hdS,
            ["gemma4.rope.freq_base"] = 1000000f,
            ["gemma4.rope.freq_base_swa"] = 10000f,
            ["gemma4.attention.layer_norm_rms_epsilon"] = 1e-6f,
            ["gemma4.vocab_size"] = (long)vocab,
            ["gemma4.feed_forward_length"] = (long)ffn,
            ["gemma4.context_length"] = 64L,
        };
        return new GGUFModel
        {
            RawData = raw.ToArray(),
            DataStartOffset = 0,
            Tensors = tensors.ToArray(),
            Metadata = meta,
        };
    }

    /// <summary>Minimal F32 gemma4 model. BuildGraph only constructs nodes + extracts weights (no
    /// execution), so tiny consistent dims suffice. <paramref name="withPostNorms"/> adds the
    /// post_attention_norm / post_ffw_norm tensors that trigger the norm-sandwich.</summary>
    private static GGUFModel MakeGemma4Model(bool withPostNorms, float logitSoftCap = 0f)
    {
        const int embd = 8, ffn = 16, vocab = 4;
        var raw = new List<byte>();
        var tensors = new List<GGUFTensorInfo>();
        void Add(string name, long[] ne)
        {
            long elems = 1; foreach (var d in ne) elems *= d;
            tensors.Add(new GGUFTensorInfo { Name = name, Dimensions = ne, Type = GGMLType.F32, DataOffset = (ulong)raw.Count });
            for (long i = 0; i < elems; i++) raw.AddRange(BitConverter.GetBytes(0.05f));
        }
        Add("token_embd.weight", new long[] { embd, vocab });    // ne fastest-first: storage [vocab][embd]
        Add("blk.0.attn_norm.weight", new long[] { embd });
        Add("blk.0.attn_q.weight", new long[] { embd, embd });
        Add("blk.0.attn_k.weight", new long[] { embd, embd });
        Add("blk.0.attn_v.weight", new long[] { embd, embd });
        Add("blk.0.attn_output.weight", new long[] { embd, embd });
        if (withPostNorms) Add("blk.0.post_attention_norm.weight", new long[] { embd });
        Add("blk.0.ffn_norm.weight", new long[] { embd });
        Add("blk.0.ffn_gate.weight", new long[] { embd, ffn });
        Add("blk.0.ffn_up.weight", new long[] { embd, ffn });
        Add("blk.0.ffn_down.weight", new long[] { ffn, embd });
        if (withPostNorms) Add("blk.0.post_ffw_norm.weight", new long[] { embd });
        Add("output_norm.weight", new long[] { embd });           // NO output.weight -> tied head

        var meta = new Dictionary<string, object>
        {
            ["general.architecture"] = "gemma4",
            ["gemma4.embedding_length"] = (long)embd,
            ["gemma4.block_count"] = 1L,
            ["gemma4.attention.head_count"] = 2L,
            ["gemma4.attention.head_count_kv"] = 2L,
            ["gemma4.vocab_size"] = (long)vocab,
            ["gemma4.feed_forward_length"] = (long)ffn,
            ["gemma4.context_length"] = 64L,
        };
        if (logitSoftCap > 0f) meta["gemma4.final_logit_softcapping"] = logitSoftCap;

        return new GGUFModel
        {
            RawData = raw.ToArray(),
            DataStartOffset = 0,
            Tensors = tensors.ToArray(),
            Metadata = meta,
        };
    }
}
