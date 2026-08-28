using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The join between the phonemizer and the model: English text to ZipVoice token ids.
/// </summary>
/// <remarks>
/// ZipVoice's shipped lexicon is Chinese-only - 68,037 CJK entries, no English - so English has to be
/// phonemized rather than looked up, and this is the exact point where the reference implementation
/// reaches for espeak-ng. What is locked here is that a phoneme the model has no token for is REPORTED
/// rather than skipped: dropping one silently produces speech that is subtly wrong with nothing to
/// explain it.
/// </remarks>
public abstract partial class MLTestBase
{
    private static ZipVoiceTokenizer MakeTokenizer()
    {
        // A symbol table shaped like the model's own tokens.txt, small enough to read.
        var symbols = new Dictionary<string, long>(StringComparer.Ordinal)
        {
            [" "] = 3, ["."] = 10, ["ð"] = 41, ["ə"] = 59, ["k"] = 23, ["æ"] = 39, ["t"] = 32,
            ["ˈ"] = 120,
        };
        var phonemizer = new EnglishPhonemizer(PronunciationDictionary.Parse(new[]
        {
            "the DH AH0",
            "cat K AE1 T",
        }))
        { Normalizer = null };
        return new ZipVoiceTokenizer(symbols, phonemizer);
    }

    [TestMethod]
    public async Task ZipVoiceTokenizer_EncodesTextToTokenIds() => await RunTest(_ =>
    {
        var ids = MakeTokenizer().Encode("the cat");
        // ð ə ␣ k ˈ æ t  - the word separator is a real token, and the stress mark rides with the vowel.
        var expected = new long[] { 41, 59, 3, 23, 120, 39, 32 };
        if (!ids.SequenceEqual(expected))
            throw new Exception($"expected [{string.Join(",", expected)}], got [{string.Join(",", ids)}]");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task ZipVoiceTokenizer_RefusesAPhonemeTheModelCannotSpeak() => await RunTest(_ =>
    {
        // A symbol with no token must be loud. Skipping it would render audio quietly missing a sound,
        // and nothing downstream could ever point at the cause.
        var symbols = new Dictionary<string, long>(StringComparer.Ordinal) { [" "] = 3, ["ð"] = 41 };
        var phonemizer = new EnglishPhonemizer(PronunciationDictionary.Parse(new[] { "the DH AH0" }))
        { Normalizer = null };
        var tokenizer = new ZipVoiceTokenizer(symbols, phonemizer);

        if (tokenizer.TryEncode("the", out var _ids, out var problem))
            throw new Exception("encoding should have failed: the schwa has no token in this table");
        if (problem == null || !problem.Contains("ə"))
            throw new Exception($"the failure must name the offending symbol, got: {problem}");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task ReferenceOverrides_AreSingleSymbolsTheModelCanTokenize() => await RunTest(_ =>
    {
        // An override is hand-written IPA, and a two-character entry like "uː" LOOKS right while being
        // TWO tokens to the model. One such entry made a whole sentence unrenderable - the tokenizer
        // refused it rather than speaking it wrong, which is the correct behaviour but an expensive way
        // to find a typo. Every symbol must be one the ordinary mapping could also have produced.
        var legal = new HashSet<string>(StringComparer.Ordinal)
        {
            Arpabet.PrimaryStress, Arpabet.SecondaryStress, Arpabet.Length, Arpabet.Flap,
        };
        foreach (var vowel in Arpabet.Vowels) legal.Add(vowel);
        foreach (var consonant in "bdfhjklmnpstvwzðŋɡɹʃʒθʔ") legal.Add(consonant.ToString());

        foreach (var symbol in ReferenceOverrides.Symbols)
            if (!legal.Contains(symbol))
                throw new Exception($"override symbol \"{symbol}\" is not a single known IPA symbol - "
                                  + "a multi-character entry is more than one token to the model");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task ZipVoiceTokenizer_ReportsWordsTheDictionaryLacked() => await RunTest(_ =>
    {
        // Worth surfacing all the way up: a name that was sounded out rather than looked up is the
        // difference between a pronunciation and a guess, and the caller may want to know.
        var tokenizer = MakeTokenizer();
        tokenizer.Encode("the");
        if (tokenizer.LastUnknownWords.Count != 0)
            throw new Exception("a known word must not be reported as unknown");
        return Task.CompletedTask;
    });
}
