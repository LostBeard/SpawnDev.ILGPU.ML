using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// "I want to reCORD the RECord" - same spelling, different stress, different word.
/// </summary>
/// <remarks>
/// About 1% of words in ordinary text, but every miss is a STRESS error, and stress on the wrong syllable
/// measured at 34.3% word error through ZipVoice against roughly nothing for any segmental slip.
///
/// The important detail, and the reason the first implementation was wrong: this CHOOSES between the
/// dictionary's own entries rather than moving a stress mark. Those are not the same thing. The noun
/// "record" is REK-erd and the verb is ri-KORD - different vowels, not just a different beat - so
/// re-stressing the verb entry produced "RUH-kord", a word nobody says.
/// </remarks>
public abstract partial class MLTestBase
{
    private static EnglishPhonemizer HomographPhonemizer() => new(PronunciationDictionary.Parse(new[]
    {
        // Exactly as CMUdict writes them: the alternate is a second line, "word(2)".
        "record R AH0 K AO1 R D",
        "record(2) R EH1 K ER0 D",
        "the DH AH0",
        "to T UW1",
        "i AY1",
        "want W AA1 N T",
        "played P L EY1 D",
    }))
    { Normalizer = null };

    [TestMethod]
    public async Task Homograph_ReadsTheNounAndTheVerbDifferently() => await RunTest(_ =>
    {
        var p = HomographPhonemizer();

        // "to" before it means a verb: ri-KORD, stress on the second syllable.
        var verb = p.ToIpa("to record");
        if (!verb.EndsWith("ɹəkˈɔːɹd")) throw new Exception($"expected the verb reading, got \"{verb}\"");

        // "the" before it means a noun: REK-erd, stress on the first, and a DIFFERENT vowel.
        var noun = p.ToIpa("the record");
        if (!noun.EndsWith("ɹˈɛkɚd")) throw new Exception($"expected the noun reading, got \"{noun}\"");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Homograph_ChoosesAnEntryRatherThanMovingStress() => await RunTest(_ =>
    {
        // The noun reading must be the dictionary's OTHER entry, vowels and all. Re-stressing the first
        // entry would give "ɹˈʌkɔːɹd" - right beat, wrong word - which is what the first attempt did.
        var noun = HomographPhonemizer().ToIpa("the record");
        if (noun.Contains("ʌk")) throw new Exception($"stress was moved instead of choosing an entry: \"{noun}\"");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Homograph_LeavesOrdinaryWordsAlone() => await RunTest(_ =>
    {
        // A word that is not a stress-shifting homograph must come back exactly as the dictionary has it,
        // whatever precedes it.
        var p = HomographPhonemizer();
        var a = p.ToIpa("i played");
        var b = p.ToIpa("the played");
        if (!a.EndsWith("plˈeɪd") || !b.EndsWith("plˈeɪd"))
            throw new Exception($"an ordinary word changed with context: \"{a}\" against \"{b}\"");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Homograph_FallsBackWhenTheDictionaryOffersNoChoice() => await RunTest(_ =>
    {
        // Only one pronunciation on file means there is nothing to choose between, and inventing one
        // would be worse than the ambiguity.
        var p = new EnglishPhonemizer(PronunciationDictionary.Parse(new[]
        {
            "record R AH0 K AO1 R D",
            "the DH AH0",
        }))
        { Normalizer = null };

        var ipa = p.ToIpa("the record");
        if (!ipa.EndsWith("ɹəkˈɔːɹd")) throw new Exception($"expected the only entry on file, got \"{ipa}\"");
        return Task.CompletedTask;
    });
}
