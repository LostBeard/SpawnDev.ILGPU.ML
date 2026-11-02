using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The GGUF token-type enum (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte) and the GGUF's own special ids. Until 2026-11-02
    /// the tokenizer read type 2 as control and 1 as unknown, so on real GGUFs no control token was found: BOS / EOS fell back to ids 1 / 2 ('"' and
    /// '#' in llama 3's and qwen's vocabularies, so GgufGenerator stopped at a '#'; &lt;eos&gt; / &lt;bos&gt; swapped in gemma's), control tokens decoded
    /// as text, and <see cref="SentencePieceTokenizer.Decode"/> returned a byte-level vocab's GPT-2 alphabet ("ĠHello") and byte-fallback runs as one
    /// char per byte. Vocabularies shaped like llama3.1:8b's (byte-level, control specials, a user-defined piece) and gemma's (SentencePiece, specials
    /// at ids 0-3, byte fallback), built as GGUF metadata and loaded through <see cref="SentencePieceTokenizer.FromGGUF"/>.
    /// </summary>
    [TestMethod]
    public async Task SentencePiece_GgufTokenTypes_SpecialIdsAndDecode() => await RunTest(async accelerator =>
    {
        // llama 3 shape: ids 0-2 are ordinary pieces, the specials are control (3), <think> is user-defined (4); eos in the metadata is <|eot_id|>
        var l3Tokens = new[] { "!", "\"", "#", "ĠHello", "Ġworld", "Ċ", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<think>", "Ã©" };
        var l3Types = new[] { 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 1 };
        GGUFModel L3(bool ids)
        {
            var m = new GGUFModel();
            m.Metadata["tokenizer.ggml.model"] = "gpt2";
            m.Metadata["tokenizer.ggml.tokens"] = l3Tokens;
            m.Metadata["tokenizer.ggml.token_type"] = l3Types;
            if (ids) { m.Metadata["tokenizer.ggml.bos_token_id"] = 6u; m.Metadata["tokenizer.ggml.eos_token_id"] = 8u; }
            return m;
        }
        var notes = new List<string>();
        var l3 = SentencePieceTokenizer.FromGGUF(L3(true))!;
        if (l3.BosId != 6 || l3.EosId != 8) throw new Exception($"llama 3 with metadata ids: BOS {l3.BosId} EOS {l3.EosId}, want 6 / 8 (the GGUF's own ids)");
        var l3Names = SentencePieceTokenizer.FromGGUF(L3(false))!;
        if (l3Names.BosId != 6 || l3Names.EosId != 7) throw new Exception($"llama 3 without metadata ids: BOS {l3Names.BosId} EOS {l3Names.EosId}, want 6 / 7 (the control tokens by name, not ids 1 / 2 = '\"' / '#')");
        string got = l3.Decode(new[] { 6, 3, 4, 5, 9, 10, 8 });
        const string want = " Hello world\n<think>é";
        if (got != want) throw new Exception($"llama 3 Decode: '{got.Replace("\n", "\\n")}', want '{want.Replace("\n", "\\n")}' (control tokens silent, user-defined as text, GPT-2 bytes mapped back, UTF-8 joined)");
        notes.Add($"llama 3: BOS/EOS {l3.BosId}/{l3.EosId}, by name {l3Names.BosId}/{l3Names.EosId}, decode ok");

        // gemma shape: <pad> <eos> <bos> <unk> at 0-3, all control; byte fallback (type 6) for é
        var gTokens = new List<string> { "<pad>", "<eos>", "<bos>", "<unk>", "▁Hi", "<start_of_turn>" };
        var gTypes = new List<int> { 3, 3, 3, 3, 1, 3 };
        for (int b = 0; b < 256; b++) { gTokens.Add($"<0x{b:X2}>"); gTypes.Add(6); }
        var gm = new GGUFModel();
        gm.Metadata["tokenizer.ggml.model"] = "llama";
        gm.Metadata["tokenizer.ggml.tokens"] = gTokens.ToArray();
        gm.Metadata["tokenizer.ggml.scores"] = new float[gTokens.Count];
        gm.Metadata["tokenizer.ggml.token_type"] = gTypes.ToArray();
        var g = SentencePieceTokenizer.FromGGUF(gm)!;
        if (g.BosId != 2 || g.EosId != 1) throw new Exception($"gemma: BOS {g.BosId} EOS {g.EosId}, want 2 / 1 (<bos> / <eos> by name)");
        g.TryGetId("<0xC3>", out int c3); g.TryGetId("<0xA9>", out int a9);
        string gGot = g.Decode(new[] { 2, 5, 4, c3, a9, 1 });
        if (gGot != "Hié") throw new Exception($"gemma Decode: '{gGot}', want 'Hié' (control silent, byte fallback joined as UTF-8, the word-start space dropped)");
        notes.Add($"gemma: BOS/EOS {g.BosId}/{g.EosId}, decode ok");
        await Task.CompletedTask;
        return string.Join(" | ", notes);
    });
}
