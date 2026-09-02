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
        "wind W AY1 N D",
        "wind(2) W IH1 N D",
        "use Y UW1 S",
        "use(2) Y UW1 Z",
        "must M AH1 S T",
        "clock K L AA1 K",
    }))
    { Normalizer = null };

    [TestMethod]
    public async Task Homograph_ReadsTheNounAndTheVerbDifferently() => await RunPureTest(() =>
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
    public async Task Homograph_ChoosesAnEntryRatherThanMovingStress() => await RunPureTest(() =>
    {
        // The noun reading must be the dictionary's OTHER entry, vowels and all. Re-stressing the first
        // entry would give "ɹˈʌkɔːɹd" - right beat, wrong word - which is what the first attempt did.
        var noun = HomographPhonemizer().ToIpa("the record");
        if (noun.Contains("ʌk")) throw new Exception($"stress was moved instead of choosing an entry: \"{noun}\"");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Homograph_ReadsVowelHomographsToo() => await RunPureTest(() =>
    {
        // Not every homograph is a stress shift. "The WIND blows" and "WIND the clock" differ by a
        // VOWEL, and the dictionary's FIRST entry is the rarer verb - so left alone, "the way the wind
        // blows" rendered as "the way the WINED blows", and the transcriber heard exactly that.
        var p = HomographPhonemizer();
        var noun = p.ToIpa("the wind");
        if (!noun.EndsWith("wˈɪnd")) throw new Exception($"expected the noun reading, got \"{noun}\"");

        var verb = p.ToIpa("must wind the clock");
        if (!verb.Contains("wˈaɪnd")) throw new Exception($"expected the verb reading, got \"{verb}\"");

        // Same mechanism, a consonant this time: the USE of it against to USE it.
        if (!p.ToIpa("the use").EndsWith("jˈuːs")) throw new Exception("noun 'use' should end voiceless");
        if (!p.ToIpa("to use").EndsWith("jˈuːz")) throw new Exception("verb 'use' should end voiced");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Homograph_LeavesOrdinaryWordsAlone() => await RunPureTest(() =>
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
    public async Task Homograph_FallsBackWhenTheDictionaryOffersNoChoice() => await RunPureTest(() =>
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
