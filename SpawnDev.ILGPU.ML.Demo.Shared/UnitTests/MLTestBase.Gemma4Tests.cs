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
        // Minimal F32 gemma4 model — BuildGraph only constructs nodes + extracts weights (no execution),
        // so tiny consistent dims are enough to exercise the arch-recognition path.
        const int embd = 8, ffn = 16, vocab = 4;
        var raw = new List<byte>();
        var tensors = new List<GGUFTensorInfo>();
        void Add(string name, long[] ne)
        {
            long elems = 1; foreach (var d in ne) elems *= d;
            tensors.Add(new GGUFTensorInfo { Name = name, Dimensions = ne, Type = GGMLType.F32, DataOffset = (ulong)raw.Count });
            for (long i = 0; i < elems; i++) raw.AddRange(BitConverter.GetBytes(0.05f));
        }
        Add("token_embd.weight", new long[] { embd, vocab });   // ne fastest-first: storage [vocab][embd]
        Add("blk.0.attn_norm.weight", new long[] { embd });
        Add("blk.0.attn_q.weight", new long[] { embd, embd });
        Add("blk.0.attn_k.weight", new long[] { embd, embd });
        Add("blk.0.attn_v.weight", new long[] { embd, embd });
        Add("blk.0.attn_output.weight", new long[] { embd, embd });
        Add("blk.0.ffn_norm.weight", new long[] { embd });
        Add("blk.0.ffn_gate.weight", new long[] { embd, ffn });
        Add("blk.0.ffn_up.weight", new long[] { embd, ffn });
        Add("blk.0.ffn_down.weight", new long[] { ffn, embd });
        Add("output_norm.weight", new long[] { embd });          // NO output.weight -> tied head

        var model = new GGUFModel
        {
            RawData = raw.ToArray(),
            DataStartOffset = 0,
            Tensors = tensors.ToArray(),
            Metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "gemma4",
                ["gemma4.embedding_length"] = (long)embd,
                ["gemma4.block_count"] = 1L,
                ["gemma4.attention.head_count"] = 2L,
                ["gemma4.attention.head_count_kv"] = 2L,
                ["gemma4.vocab_size"] = (long)vocab,
                ["gemma4.feed_forward_length"] = (long)ffn,
                ["gemma4.context_length"] = 64L,
            },
        };

        var (graph, _, _, _) = GGUFGraphBuilder.BuildGraph(model);

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
}
