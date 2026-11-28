using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="Preprocessing.UnigramTokenizer.FromTokenizerJson"/> reads the older tokenizer.json format with no <c>model.type</c> (google-t5/t5-base's: just
    /// <c>unk_id</c> and a [piece, score] vocab; Lore 2026-11-28: it threw KeyNotFoundException) the same as the typed one, and still refuses a BPE-typed model and an
    /// untyped one whose vocab is an object. A tiny in-memory vocab: host only, no download.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task UnigramTokenizer_ReadsTokenizerJsonWithoutModelType() => await RunTest(accelerator =>
    {
        const string vocab = "[[\"<pad>\",0.0],[\"</s>\",0.0],[\"<unk>\",0.0],[\"▁\",-2.0],[\"▁hello\",-3.0],[\"▁world\",-3.5],[\"h\",-6.0],[\"e\",-6.0],[\"l\",-6.0],[\"o\",-6.0],[\"w\",-6.0],[\"r\",-6.0],[\"d\",-6.0]]";
        const string post = "\"post_processor\":{\"type\":\"TemplateProcessing\",\"special_tokens\":{\"</s>\":{\"id\":\"</s>\",\"ids\":[1],\"tokens\":[\"</s>\"]}}}";
        static Preprocessing.UnigramTokenizer Read(string s) => Preprocessing.UnigramTokenizer.FromTokenizerJson(new MemoryStream(System.Text.Encoding.UTF8.GetBytes(s)));
        var typed = Read("{\"model\":{\"type\":\"Unigram\",\"unk_id\":2,\"vocab\":" + vocab + "}," + post + "}");
        var untyped = Read("{\"model\":{\"unk_id\":2,\"vocab\":" + vocab + "}," + post + "}");
        var a = typed.Encode("hello world"); var b = untyped.Encode("hello world");
        Ur_Expect(a.SequenceEqual(b), $"the untyped file encodes differently: [{string.Join(", ", b)}] vs [{string.Join(", ", a)}]");
        Ur_Expect(a.SequenceEqual(new[] { 4, 5, 1 }), $"expected [4, 5, 1] (hello, world, EOS), got [{string.Join(", ", a)}]");
        bool Refused(string s) { try { Read(s); return false; } catch (NotSupportedException) { return true; } }
        Ur_Expect(Refused("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0}}}"), "a BPE-typed model was accepted");
        Ur_Expect(Refused("{\"model\":{\"unk_id\":0,\"vocab\":{\"a\":0,\"b\":1}}}"), "an untyped model with an object vocab was accepted");
        string line = $"typed and untyped both [{string.Join(", ", a)}]; BPE-typed and object-vocab files refused";
        Console.WriteLine("[Unigram] " + line);
        return Task.FromResult(line);
    });
}
