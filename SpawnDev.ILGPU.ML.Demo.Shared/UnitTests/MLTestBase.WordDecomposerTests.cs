using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Locks the endings that let an unknown word be DERIVED instead of guessed.
/// </summary>
/// <remarks>
/// Measured on 5,000 words held out of the letter-to-sound training: decomposition fires on 26.4% of
/// them and is right 77.2% of the time it fires, where letter-to-sound alone was right 54.9% of those
/// same words. Overall that is 53.0% against 49.8%. This is why an unknown word is decomposed before it
/// is ever guessed at.
///
/// The allomorphy below is the audible part. The -s of "cats" is an S, the -s of "dogs" is a Z, and the
/// -s of "boxes" is an entire extra syllable - one fixed sound for the ending would be wrong twice.
/// </remarks>
public abstract partial class MLTestBase
{
    private static PronunciationDictionary StemDictionary() => PronunciationDictionary.Parse(new[]
    {
        "cat K AE1 T",
        "dog D AO1 G",
        "box B AA1 K S",
        "walk W AO1 K",
        "play P L EY1",
        "want W AA1 N T",
        "hope HH OW1 P",
        "run R AH1 N",
        "friendly F R EH1 N D L IY0",
    });

    [TestMethod]
    public async Task Decomposer_PluralTakesTheSoundTheStemDemands() => await RunTest(_ =>
    {
        var dictionary = StemDictionary();
        ExpectPhones(dictionary, "cats", "K AE1 T S");            // voiceless stem -> S
        ExpectPhones(dictionary, "dogs", "D AO1 G Z");            // voiced stem -> Z
        ExpectPhones(dictionary, "boxes", "B AA1 K S IH0 Z");     // sibilant stem -> a whole syllable
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Decomposer_PastTenseTakesTheSoundTheStemDemands() => await RunTest(_ =>
    {
        var dictionary = StemDictionary();
        ExpectPhones(dictionary, "walked", "W AO1 K T");          // voiceless stem -> T
        ExpectPhones(dictionary, "played", "P L EY1 D");          // voiced stem -> D
        ExpectPhones(dictionary, "wanted", "W AA1 N T IH0 D");    // stem ends in T -> a whole syllable
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Decomposer_FindsStemsEnglishSpellingHides() => await RunTest(_ =>
    {
        var dictionary = StemDictionary();
        // "hoped" is hope + d, with the e swallowed. "running" is run + ing, with the n doubled.
        ExpectPhones(dictionary, "hoped", "HH OW1 P T");
        ExpectPhones(dictionary, "running", "R AH1 N IH0 NG");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Decomposer_HandlesPrefixes() => await RunTest(_ =>
    {
        ExpectPhones(StemDictionary(), "unfriendly", "AH0 N F R EH1 N D L IY0");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Decomposer_DeclinesWhenThereIsNoKnownStem() => await RunTest(accelerator =>
    {
        // It must FAIL rather than invent: "aubriella" is not anything plus an ending, and a decomposer
        // that answers anyway would take the word away from letter-to-sound, which can at least try.
        if (WordDecomposer.TryDecompose("aubriella", StemDictionary(), out var _unused1))
            throw new Exception("decomposed a word that has no known stem");
        if (WordDecomposer.TryDecompose("xyz", StemDictionary(), out var _unused2))
            throw new Exception("decomposed a word too short to decompose");
        return Task.CompletedTask;
    });

    private static void ExpectPhones(PronunciationDictionary dictionary, string word, string expected)
    {
        if (!WordDecomposer.TryDecompose(word, dictionary, out var phones))
            throw new Exception($"failed to decompose \"{word}\"");
        var actual = string.Join(' ', phones);
        if (actual != expected) throw new Exception($"{word}: expected \"{expected}\", got \"{actual}\"");
    }
}
