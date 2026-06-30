using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Correctness + wiring guard for the WordPiece tokenizer (BERT / DistilBERT / ELECTRA family).
/// Until now TokenizerLoader always built a BPE tokenizer, so every BERT-style model produced
/// meaningless tokens (NLPPipelines literally commented "a WordPiece tokenizer should be used").
/// These cases are the canonical HuggingFace reference vectors from
/// <c>transformers/tests/models/bert/test_tokenization_bert.py</c> — the BasicTokenizer lowercase /
/// accent-strip / punctuation-split behavior and the WordpieceTokenizer greedy longest-match split —
/// plus a TokenizerLoader path test that proves model.type=="WordPiece" + BertNormalizer parsing wires
/// the real production tokenizer (not BPE).
/// </summary>
public abstract partial class MLTestBase
{
    // HF reference vocab from test_tokenization_bert.py::test_wordpiece_tokenizer, extended with the
    // BasicTokenizer-test tokens so the full lowercase/punctuation/accent pipeline is verifiable too.
    private static readonly string[] WordPieceVocab =
    {
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
        "want", "##want", "##ed", "wa", "un", "runn", "##ing",
        "hello", "how", "are", "you", "!", "?",
    };

    private static Dictionary<string, int> BuildWordPieceVocab()
    {
        var v = new Dictionary<string, int>();
        for (int i = 0; i < WordPieceVocab.Length; i++) v[WordPieceVocab[i]] = i;
        return v;
    }

    private static string[] TokensOf(WordPieceTokenizer t, Dictionary<string, int> vocab, string text)
    {
        var rev = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        return t.Encode(text).Select(id => rev[id]).ToArray();
    }

    private static void AssertTokens(string[] got, string[] expected, string label)
    {
        if (!got.SequenceEqual(expected))
            throw new Exception($"WordPiece {label}: got [{string.Join(" ", got)}], expected [{string.Join(" ", expected)}]");
    }

    [TestMethod]
    public async Task WordPiece_MatchesHuggingFaceReferenceVectors() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask; // pure CPU tokenization; no GPU work.

        var vocab = BuildWordPieceVocab();
        var t = new WordPieceTokenizer(vocab, doLowerCase: true);

        // ── WordpieceTokenizer greedy longest-match (HF test_wordpiece_tokenizer) ──
        AssertTokens(TokensOf(t, vocab, ""), Array.Empty<string>(), "empty");
        AssertTokens(TokensOf(t, vocab, "unwanted running"),
            new[] { "un", "##want", "##ed", "runn", "##ing" }, "unwanted running");
        // A word with a char absent from the vocab maps the WHOLE word to [UNK] (not a partial split).
        AssertTokens(TokensOf(t, vocab, "unwantedX running"),
            new[] { "[UNK]", "runn", "##ing" }, "unwantedX running");

        // ── BasicTokenizer lowercase + whitespace-collapse + punctuation-split (HF test_basic_tokenizer_lower) ──
        AssertTokens(TokensOf(t, vocab, "HeLLo!how  \n Are yoU?  "),
            new[] { "hello", "!", "how", "are", "you", "?" }, "mixed-case + punctuation");

        // ── Accent stripping (uncased models): "Héllo" → "hello" ──
        AssertTokens(TokensOf(t, vocab, "Héllo"), new[] { "hello" }, "accent strip");

        // ── do_lower_case=false keeps case (so "Hello" is no longer the lowercase vocab entry) ──
        var cased = new WordPieceTokenizer(vocab, doLowerCase: false);
        AssertTokens(TokensOf(cased, vocab, "hello"), new[] { "hello" }, "cased-but-already-lower");
        AssertTokens(TokensOf(cased, vocab, "HELLO"), new[] { "[UNK]" }, "cased uppercase → UNK");

        // ── Decode re-attaches ## continuations ──
        var decoded = t.Decode(t.Encode("unwanted"));
        if (decoded != "unwanted")
            throw new Exception($"WordPiece decode: got '{decoded}', expected 'unwanted'");

        Console.WriteLine("[WordPiece] HF reference vectors (greedy split, lowercase, accent strip, decode) all match.");
    });

    [TestMethod]
    public async Task WordPiece_TokenizerLoaderWiresWordPieceFromModelType() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask;

        // Minimal tokenizer.json with model.type == "WordPiece" + a BertNormalizer — the real format the
        // demos load. Proves the loader detects the type and builds a WordPieceTokenizer (NOT BPE).
        var vocab = BuildWordPieceVocab();
        var vocabJson = string.Join(",", vocab.Select(kv => $"\"{EscapeJson(kv.Key)}\":{kv.Value}"));
        var json = $$"""
        {
          "normalizer": { "type": "BertNormalizer", "lowercase": true, "strip_accents": null, "handle_chinese_chars": true },
          "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": { {{vocabJson}} }
          }
        }
        """;

        var loaded = TokenizerLoader.FromTokenizerJson(json);

        if (loaded.Tokenizer is not WordPieceTokenizer)
            throw new Exception($"TokenizerLoader built a {loaded.Tokenizer.GetType().Name}, expected WordPieceTokenizer (model.type=='WordPiece')");
        if (loaded.VocabSize != WordPieceVocab.Length)
            throw new Exception($"VocabSize {loaded.VocabSize}, expected {WordPieceVocab.Length}");

        // Through the production Encode path the same reference vector must hold.
        var rev = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        var toks = loaded.Encode("unwanted running").Select(id => rev[id]).ToArray();
        AssertTokens(toks, new[] { "un", "##want", "##ed", "runn", "##ing" }, "loader path");

        // EncodeForModel wraps with [CLS]/[SEP] when the loader resolved them (added_tokens path); here we
        // only assert the inner ids stay correct and special-token ids resolved from the vocab.
        Console.WriteLine($"[WordPiece] TokenizerLoader → WordPieceTokenizer, vocab={loaded.VocabSize}, reference vector matches.");
    });

    private static string EscapeJson(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
