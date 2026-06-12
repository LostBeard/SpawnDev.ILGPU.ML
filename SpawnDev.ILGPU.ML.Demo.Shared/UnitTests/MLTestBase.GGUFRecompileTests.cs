using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GGUF dynamic-shape RECOMPILE regression suite. THE BUG THIS LOCKS OUT (2026-06-12, the
/// gemma4 multi-token blocker): InferenceSession.RecompileForShapes constructed the per-shape
/// GraphExecutor WITHOUT the session's quantized byte-view map, so every quantized MatMul /
/// Gather in a recompiled executor silently fell back to the F32 path against a ShapeOnly
/// tensor's EMPTY view — a CUDA illegal memory access at the FIRST quantized node. seq=1
/// matched the base compile shape and never recompiled, so only seq&gt;1 faulted, and no
/// GGUF forward had ever run multi-token before.
///
/// The test runs a complete tiny quantized model (Q4_K/Q6_K + F32 norms, tied head) through
/// the REAL production path: GGUF bytes → CreateFromGGUF → RunAsync at the base shape, then
/// RunAsync at a LONGER sequence (forces RecompileForShapes). Causality makes position 0 of
/// the longer run computable from position 0 alone, so its logits must match the seq=1 run —
/// a recompiled executor reading anything but the same quantized weights cannot pass.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task GGUFSession_Recompile_KeepsQuantizedWeights_MultiToken() => await RunTest(async accelerator =>
    {
        const int embd = 256, vocab = 32, ffn = 320, ctx = 64;
        var bytes = BuildTinyQuantizedLlamaGGUF(embd, vocab, ffn, ctx, new Random(2026));

        using var session = InferenceSession.CreateFromGGUF(accelerator, bytes);

        // ── Base-shape run (seq=1) ──
        var ids1 = new float[] { 3 };
        using var in1 = accelerator.Allocate1D(ids1);
        var out1 = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["input_ids"] = new Tensor(in1.View, new[] { 1, 1 }, "input_ids"),
        });
        var logits1T = out1.TryGetValue("logits", out var l1) ? l1 : out1.Values.First();
        if (logits1T.Shape[^1] != vocab)
            throw new Exception($"seq=1 logits last dim {logits1T.Shape[^1]}, want vocab={vocab}");
        // Copy off the executor's pool-managed buffer BEFORE the next RunAsync reuses it
        // (CopyFrom = native GPU→GPU on every backend), then read back.
        using var read1 = accelerator.Allocate1D<float>(vocab);
        await read1.View.CopyFromAsync(logits1T.Data.SubView(0, vocab));
        await accelerator.SynchronizeAsync();
        var logits1 = await read1.CopyToHostAsync<float>(0, vocab);
        if (logits1.Any(v => float.IsNaN(v) || float.IsInfinity(v)))
            throw new Exception("seq=1 logits contain non-finite values");

        // ── Multi-token run (seq=3) — MUST go through RecompileForShapes ──
        var ids3 = new float[] { 3, 9, 21 };
        using var in3 = accelerator.Allocate1D(ids3);
        var out3 = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["input_ids"] = new Tensor(in3.View, new[] { 1, 3 }, "input_ids"),
        });
        if (session.LastRecompileMs <= 0)
            throw new Exception(
                "seq=3 did NOT trigger a shape recompile — this test exists to exercise the " +
                "recompiled-executor path and is no longer testing anything. If recompilation " +
                "was intentionally removed/changed, update this test to force the new path.");
        var logits3T = out3.TryGetValue("logits", out var l3) ? l3 : out3.Values.First();
        int seq3 = logits3T.ElementCount / vocab;
        if (seq3 != 3)
            throw new Exception($"seq=3 logits cover {seq3} positions, want 3 (shape [{string.Join(",", logits3T.Shape)}])");
        using var read3 = accelerator.Allocate1D<float>(3 * vocab);
        await read3.View.CopyFromAsync(logits3T.Data.SubView(0, 3 * vocab));
        await accelerator.SynchronizeAsync();
        var logits3 = await read3.CopyToHostAsync<float>(0, 3 * vocab);
        if (logits3.Any(v => float.IsNaN(v) || float.IsInfinity(v)))
            throw new Exception("seq=3 logits contain non-finite values");

        // ── Causal cross-check: position 0 of the recompiled run == the seq=1 run ──
        // Same token, same weights, position 0 attends only to itself; any quantized weight
        // the recompiled executor lost or mis-bound shows up as a wild divergence here.
        for (int v = 0; v < vocab; v++)
        {
            float tol = MathF.Max(1e-4f, MathF.Abs(logits1[v]) * 1e-4f);
            if (MathF.Abs(logits3[v] - logits1[v]) > tol)
                throw new Exception(
                    $"Recompiled-executor logits diverge at vocab {v}: seq=1 gave {logits1[v]}, " +
                    $"seq=3 position 0 gave {logits3[v]} (tol {tol}). The per-shape executor is " +
                    "not computing with the same quantized weights as the base executor.");
        }

        Console.WriteLine($"[GGUFRecompile] seq=1 → seq=3 recompile keeps quantized weights; " +
            $"position-0 logits match (recompile {session.LastRecompileMs:F0}ms)");
    });

    /// <summary>A complete, runnable tiny llama-arch GGUF binary with QUANTIZED linears
    /// (Q4_K) + quantized tied-head embedding (Q6_K) + F32 norms — the same tensor set the
    /// graph-builder contract test uses, serialized to real GGUF v3 bytes. K=embd=256 = one
    /// K-quant super-block per row keeps every backend fast while still exercising the full
    /// fused-dequant path end-to-end.</summary>
    private static byte[] BuildTinyQuantizedLlamaGGUF(int embd, int vocab, int ffn, int ctx, Random rng)
    {
        // ── Tensor payloads (GGUF storage order: [N rows][K contiguous], ne fastest-first) ──
        byte[] F32Bytes(int count, float center = 0f, float spread = 0.5f)
        {
            var b = new byte[count * 4];
            for (int i = 0; i < count; i++)
                BitConverter.GetBytes(center + (float)(rng.NextDouble() - 0.5) * spread).CopyTo(b, i * 4);
            return b;
        }
        byte[] QRows(GGMLType t, int k, int n)
        {
            int rb = RowBytes(t, k);
            var all = new byte[n * rb];
            for (int r = 0; r < n; r++)
                Buffer.BlockCopy(MakeBlocks(t, k, rng), 0, all, r * rb, rb);
            return all;
        }

        var tensors = new List<(string Name, long[] Ne, GGMLType Type, byte[] Data)>
        {
            ("token_embd.weight", new long[] { embd, vocab }, GGMLType.Q6_K, QRows(GGMLType.Q6_K, embd, vocab)),
            ("blk.0.attn_norm.weight", new long[] { embd }, GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.0.attn_q.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd)),
            ("blk.0.attn_k.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd)),
            ("blk.0.attn_v.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd)),
            ("blk.0.attn_output.weight", new long[] { embd, embd }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, embd)),
            ("blk.0.ffn_norm.weight", new long[] { embd }, GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            ("blk.0.ffn_gate.weight", new long[] { embd, ffn }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, ffn)),
            ("blk.0.ffn_up.weight", new long[] { embd, ffn }, GGMLType.Q4_K, QRows(GGMLType.Q4_K, embd, ffn)),
            ("blk.0.ffn_down.weight", new long[] { ffn, embd }, GGMLType.F32, F32Bytes(ffn * embd, spread: 0.1f)),
            ("output_norm.weight", new long[] { embd }, GGMLType.F32, F32Bytes(embd, center: 1f, spread: 0.2f)),
            // NO output.weight → tied-embedding LM head (quantized MatMul via the alias).
        };

        var metadata = new (string Key, object Value)[]
        {
            ("general.architecture", "llama"),
            ("general.name", "tiny-recompile-test"),
            ("llama.embedding_length", (uint)embd),
            ("llama.block_count", 1u),
            ("llama.attention.head_count", 4u),
            ("llama.attention.head_count_kv", 4u),
            ("llama.vocab_size", (uint)vocab),
            ("llama.feed_forward_length", (uint)ffn),
            ("llama.context_length", (uint)ctx),
        };

        // ── Serialize: GGUF v3, default 32-byte data alignment ──
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((byte)'G'); bw.Write((byte)'G'); bw.Write((byte)'U'); bw.Write((byte)'F');
        bw.Write((uint)3);                    // version
        bw.Write((ulong)tensors.Count);       // tensor count
        bw.Write((ulong)metadata.Length);     // metadata KV count

        void WriteStr(string s)
        {
            var utf8 = System.Text.Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)utf8.Length);
            bw.Write(utf8);
        }
        foreach (var (key, value) in metadata)
        {
            WriteStr(key);
            switch (value)
            {
                case string s: bw.Write((uint)8); WriteStr(s); break;   // GGUFValueType.String
                case uint u: bw.Write((uint)4); bw.Write(u); break;     // GGUFValueType.UInt32
                default: throw new ArgumentException($"unsupported test metadata type {value.GetType()}");
            }
        }

        // Tensor infos with 32-aligned offsets relative to the data-section start.
        ulong offset = 0;
        var offsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = offset;
            offset = (ulong)(offset + (ulong)tensors[i].Data.Length + 31) & ~31UL;
        }
        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, ne, type, _) = tensors[i];
            WriteStr(name);
            bw.Write((uint)ne.Length);
            foreach (var d in ne) bw.Write((ulong)d);
            bw.Write((uint)type);
            bw.Write(offsets[i]);
        }

        // Data section starts at the next 32-byte boundary; each tensor padded to 32.
        while (ms.Position % 32 != 0) bw.Write((byte)0);
        for (int i = 0; i < tensors.Count; i++)
        {
            bw.Write(tensors[i].Data);
            while (ms.Position % 32 != 0) bw.Write((byte)0);
        }
        bw.Flush();
        return ms.ToArray();
    }
}
