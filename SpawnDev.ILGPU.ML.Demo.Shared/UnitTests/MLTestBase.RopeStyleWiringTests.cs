using System.Linq;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Regression guard for the smollm2:360m (arch=llama) degenerate-repetition bug. The GGUF graph
/// builder used to apply NeoX / split-half RoPE to EVERY architecture, but the LLaMA lineage stores
/// its q/k weights PERMUTED for NORM / consecutive-pair rotation (llama.cpp
/// <c>llama_model_rope_type</c>; <c>convert_hf_to_gguf.py</c> bakes the permutation in). Applying NeoX
/// to a NORM-permuted model scrambles every q/k channel → garbage logits → the "TheThe answerThe
/// answer the…" repetition loop, while qwen2/gemma (true NeoX) stayed correct. The fix wires the RoPE
/// pairing style per arch (<see cref="GGUFGraphBuilder.UsesNormRope"/>). The RoPEKernel math for both
/// styles is covered separately by the RoPE oracle tests; this locks the BUILDER's per-arch selection.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task RopeStyle_BuilderWiresNormForLlama_NeoXForQwen() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask; // pure graph-construction test; no GPU work needed.

        // Taxonomy mirrors llama.cpp's NORM vs NEOX groups.
        foreach (var a in new[] { "llama", "llama4", "mistral", "minicpm", "granite", "olmo2", "cohere" })
            if (!GGUFGraphBuilder.UsesNormRope(a)) throw new Exception($"arch '{a}' must use NORM (consecutive-pair) RoPE");
        foreach (var a in new[] { "qwen", "qwen2", "gemma", "gemma2", "gemma3", "gptoss", "falcon", "phi3" })
            if (GGUFGraphBuilder.UsesNormRope(a)) throw new Exception($"arch '{a}' must use NeoX (split-half) RoPE");

        // End-to-end production wiring: BuildGraph must stamp the right 'interleaved' on every RoPE node.
        AssertGraphRopeInterleaved("llama", expected: 1);  // NORM lineage → interleaved=1
        AssertGraphRopeInterleaved("qwen2", expected: 0);  // NeoX → interleaved=0
    });

    /// <summary>Build the real GGUF graph for a tiny model of the given arch and assert every RoPE node
    /// carries the expected 'interleaved' pairing flag (1 = NORM/consecutive, 0 = NeoX/split-half).</summary>
    private static void AssertGraphRopeInterleaved(string arch, long expected)
    {
        var bytes = BuildTinyQuantizedLlamaGGUF(embd: 256, vocab: 32, ffn: 320, ctx: 64, new Random(11), arch);
        var model = GGUFParser.Parse(bytes);
        var (graph, _, _, _, _) = GGUFGraphBuilder.BuildGraph(model);

        var ropeNodes = graph.Nodes.Where(n => n.OpType == "RoPE").ToList();
        if (ropeNodes.Count == 0) throw new Exception($"arch '{arch}': graph has no RoPE nodes to check");
        foreach (var n in ropeNodes)
        {
            if (n.Attributes == null || !n.Attributes.TryGetValue("interleaved", out var el))
                throw new Exception($"arch '{arch}': RoPE node '{string.Join(",", n.Outputs)}' has no 'interleaved' attr");
            long got = el.GetInt64();
            if (got != expected)
                throw new Exception($"arch '{arch}': RoPE interleaved={got}, expected {expected} " +
                    "(NORM lineage must be 1, NeoX must be 0)");
        }
        Console.WriteLine($"[RopeStyle] arch={arch}: {ropeNodes.Count} RoPE node(s) all interleaved={expected}");
    }
}
