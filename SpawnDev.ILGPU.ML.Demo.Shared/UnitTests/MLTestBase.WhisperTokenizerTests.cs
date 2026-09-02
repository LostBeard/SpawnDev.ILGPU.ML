using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Guards the Whisper special-token contract, which failed SILENTLY for every English-only checkpoint.
/// </summary>
/// <remarks>
/// The bug: <see cref="BPETokenizer.LoadFromTokenizerJson"/> ignored the tokenizer's <c>added_tokens</c>
/// block, where every Whisper special token actually lives. Callers therefore had to hard-code the ids,
/// and <c>SpeechRecognitionPipeline</c> hard-coded the MULTILINGUAL set. The English-only (.en)
/// checkpoints carry a byte-level BPE vocabulary one entry smaller, so all their special ids shift down
/// by one - priming an .en decoder with the multilingual ids feeds it tokens it was never trained to
/// emit, and it answers with end-of-text immediately. The transcript comes back as "" with no exception,
/// which is indistinguishable from silent audio, so nothing anywhere reported a fault.
///
/// The tokenizer JSON is built here rather than read from a model directory because the model files are
/// gitignored for size; the ids and structure below are those of the real Whisper tokenizers.
/// </remarks>
public abstract partial class MLTestBase
{
    // Real layouts. Multilingual: endoftext 50257, startoftranscript 50258. English-only: one lower.
    private const string MultilingualSpecials =
        "{\"content\":\"<|endoftext|>\",\"id\":50257}," +
        "{\"content\":\"<|startoftranscript|>\",\"id\":50258}," +
        "{\"content\":\"<|en|>\",\"id\":50259}," +
        "{\"content\":\"<|transcribe|>\",\"id\":50359}," +
        "{\"content\":\"<|notimestamps|>\",\"id\":50363}";

    private const string EnglishOnlySpecials =
        "{\"content\":\"<|endoftext|>\",\"id\":50256}," +
        "{\"content\":\"<|startoftranscript|>\",\"id\":50257}," +
        "{\"content\":\"<|en|>\",\"id\":50258}," +
        "{\"content\":\"<|transcribe|>\",\"id\":50358}," +
        "{\"content\":\"<|notimestamps|>\",\"id\":50362}";

    private static string WhisperTokenizerJson(string addedTokens) =>
        "{\"added_tokens\":[" + addedTokens + "]," +
        "\"model\":{\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"he\":4,\"ll\":5,\"hell\":6,\"hello\":7}," +
        "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\"]}}";

    [TestMethod]
    public async Task WhisperTokenizer_ResolvesMultilingualSpecialTokens() => await RunPureTest(() =>
    {
        var tok = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(MultilingualSpecials));

        AssertTokenId(tok, "<|endoftext|>", 50257);
        AssertTokenId(tok, "<|startoftranscript|>", 50258);
        AssertTokenId(tok, "<|en|>", 50259);
        AssertTokenId(tok, "<|transcribe|>", 50359);
        AssertTokenId(tok, "<|notimestamps|>", 50363);
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task WhisperTokenizer_ResolvesEnglishOnlySpecialTokens() => await RunPureTest(() =>
    {
        var tok = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(EnglishOnlySpecials));

        // Every id one lower than the multilingual set. That shift IS the defect: hard-coding the
        // multilingual numbers made these models decode to an empty string with no error.
        AssertTokenId(tok, "<|endoftext|>", 50256);
        AssertTokenId(tok, "<|startoftranscript|>", 50257);
        AssertTokenId(tok, "<|transcribe|>", 50358);
        AssertTokenId(tok, "<|notimestamps|>", 50362);
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task WhisperTokenizer_EndOfTextIdentifiesTheModelFamily() => await RunPureTest(() =>
    {
        // SpeechRecognitionPipeline picks its decoder prompt from this one signal, so it gets its own
        // test: English-only checkpoints were trained WITHOUT the language and task tokens, and priming
        // them with the four-token multilingual prompt is what produced the empty transcripts.
        var multilingual = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(MultilingualSpecials));
        var englishOnly = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(EnglishOnlySpecials));

        if (!multilingual.TryGetTokenId("<|endoftext|>", out var multiEot))
            throw new Exception("multilingual tokenizer did not resolve <|endoftext|>");
        if (!englishOnly.TryGetTokenId("<|endoftext|>", out var enEot))
            throw new Exception("English-only tokenizer did not resolve <|endoftext|>");

        if (enEot != 50256)
            throw new Exception($"English-only end-of-text should be 50256, got {enEot} - "
                              + "family detection reads this id");
        if (multiEot == 50256)
            throw new Exception("a multilingual model must NOT be detected as English-only");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task WhisperTokenizer_SpecialTokensDoNotDisturbOrdinaryText() => await RunPureTest(() =>
    {
        // Special tokens are BPE-unreachable - no merge rule produces them - so adding them to the vocab
        // must not change how ordinary text encodes. Without this guard, the fix could quietly corrupt
        // every other consumer of this tokenizer (GPT-2, CLIP).
        var withSpecials = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(MultilingualSpecials));
        var withoutSpecials = BPETokenizer.LoadFromTokenizerJson(WhisperTokenizerJson(""));

        var a = withSpecials.Encode("hello");
        var b = withoutSpecials.Encode("hello");

        if (a.Length == 0) throw new Exception("the fixture vocabulary encoded 'hello' to nothing");
        if (a.Length != b.Length)
            throw new Exception($"special tokens changed the token COUNT of ordinary text: "
                              + $"{b.Length} without, {a.Length} with");
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                throw new Exception($"special tokens changed ordinary text at token {i}: {b[i]} -> {a[i]}");
        return Task.CompletedTask;
    });

    private static void AssertTokenId(BPETokenizer tok, string token, int expected)
    {
        if (!tok.TryGetTokenId(token, out var id))
            throw new Exception($"{token} was not resolvable - added_tokens is being ignored, "
                              + "which is the original defect this test exists for");
        if (id != expected) throw new Exception($"{token} should be {expected}, got {id}");
    }
}
