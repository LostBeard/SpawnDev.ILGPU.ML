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
