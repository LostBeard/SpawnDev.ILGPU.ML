using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Locks the phonemizer rules that a measurement proved matter.
/// </summary>
/// <remarks>
/// Every expectation below was READ OFF real reference output for that word, captured through
/// sherpa-onnx, not reasoned from a textbook. The rules they cover were prioritised by measuring what
/// ZipVoice actually punishes over 432 renders: stress errors cost more than mispronouncing a word
/// outright, while segmental detail cost nothing detectable. See Plans/mit-phonemizer-2026-08-27.md.
///
/// The dictionary is built inline rather than loaded, so these tests need no downloaded data and stay
/// honest about which entry produced which result.
/// </remarks>
public abstract partial class MLTestBase
{
    private static EnglishPhonemizer MakePhonemizer() => new(PronunciationDictionary.Parse(new[]
    {
        "better B EH1 T ER0",
        "ladder L AE1 D ER0",
        "water W AO1 T ER0",
        "roses R OW1 Z IH0 Z",
        "understand AH2 N D ER0 S T AE1 N D",
        "the DH AH0",
        "in IH0 N",
        "my M AY1",
        "mother M AH1 DH ER0",
        "rose R OW1 Z",
        "waited W EY1 T IH0 D",
        "wait W EY1 T",
        "hundred HH AH1 N D R AH0 D",
    }));

    [TestMethod]
    public async Task Phonemizer_PutsStressBeforeTheVowelAndTapsT() => await RunTest(_ =>
    {
        // Reference: better is b-STRESS-E-tap-schwar. Two rules at once - the stress mark sits before the
        // VOWEL rather than before the syllable, and an intervocalic T becomes a tap.
        Expect(MakePhonemizer().ToIpa("better"), "bˈɛɾɚ");
        Expect(MakePhonemizer().ToIpa("water"), "wˈɔːɾɚ");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_TapsTButNeverD() => await RunTest(_ =>
    {
        // The reference frontend taps T and leaves D alone: ladder keeps a plain d. Flapping D as well
        // was a real defect here, caught by the accuracy probe against captured reference output.
        Expect(MakePhonemizer().ToIpa("ladder"), "lˈædɚ");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_DestressesFunctionWords() => await RunTest(_ =>
    {
        // THE most important rule in the phonemizer. A dictionary stores citation forms, so function
        // words arrive carrying a stress they lose in running speech. Measured cost of getting this
        // wrong: 18.2% word error - worse than mispronouncing a whole word, which cost 13.5%.
        var p = MakePhonemizer();
        Expect(p.ToIpa("the"), "ðə");

        p.DestressFunctionWords = false;
        if (!p.ToIpa("the").Contains(Arpabet.PrimaryStress))
            throw new Exception("with destressing off, 'the' should show the defect this rule prevents");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_GivesAStresslessContentWordItsStress() => await RunTest(_ =>
    {
        // The dictionary stores "in" as IH0 N, with no stressed vowel at all. Emitted that way it leaves
        // a hole where a beat belongs; the reference frontend marks it. Any word that is not a weak form
        // must carry a stress somewhere.
        Expect(MakePhonemizer().ToIpa("in"), "ˈɪn");
        Expect(MakePhonemizer().ToIpa("my"), "mˈaɪ");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_KeepsSecondaryStress() => await RunTest(_ =>
    {
        // understand is SECONDARY-uh-n-d-schwar-s-t-PRIMARY-ae-n-d. Dropping the secondary mark barely
        // touches the words but moves the audio, which is the difference between intelligible and
        // natural - and natural is the product.
        Expect(MakePhonemizer().ToIpa("understand"), "ˌʌndɚstˈænd");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_WritesPunctuationTheWayTheModelWasTrainedOn() => await RunTest(_ =>
    {
        // The reference frontend writes "roses ,understand" - a space BEFORE the pause mark, which then
        // leads the clause it opens. Getting this backwards desynchronised every word-level comparison
        // against the reference until it was fixed.
        var ipa = MakePhonemizer().ToIpa("roses, understand");
        Expect(ipa, "ɹˈoʊzᵻz ,ˌʌndɚstˈænd");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_ReportsWordsItDoesNotKnow() => await RunTest(_ =>
    {
        // An unknown word is the one failure a caller must hear about: it is the difference between
        // speaking a name and silently skipping it. "aubriella" is genuinely absent from CMUdict, which
        // is exactly why the out-of-vocabulary path has to exist.
        var p = MakePhonemizer();
        p.ToIpa("my aubriella");
        if (p.LastUnknownWords.Count != 1 || p.LastUnknownWords[0] != "aubriella")
            throw new Exception("an unknown word must be reported, got: "
                              + string.Join(", ", p.LastUnknownWords));

        p.ToIpa("my mother");
        if (p.LastUnknownWords.Count != 0)
            throw new Exception("the unknown list must reset per call");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_ReducesPluralAndPastEndingsButOnlyRealOnes() => await RunTest(_ =>
    {
        // "roses" and "waited" take the reduced ending vowel. "hundred" looks identical to the phone
        // rule - unstressed vowel, then D, spelled -ed - and must NOT take it, because there is no word
        // "hundr" for it to be the past tense of. That is why the rule checks the STEM in the dictionary
        // rather than trusting the spelling.
        var p = MakePhonemizer();
        Expect(p.ToIpa("roses"), "ɹˈoʊzᵻz");
        Expect(p.ToIpa("waited"), "wˈeɪɾᵻd");
        Expect(p.ToIpa("hundred"), "hˈʌndɹəd");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_LoadsItsOwnDataInTheBrowser() => await RunTest(_ =>
    {
        // The library claims to be browser-capable, and its data is embedded gzipped in the assembly.
        // That path - manifest resource plus GZipStream - had only ever run on the desktop. This test
        // runs on the Wasm, WebGL and WebGPU lanes too, which is the only way that claim is worth making.
        var phonemizer = EmbeddedData.CreatePhonemizer();

        var ipa = phonemizer.ToIpa("She waited for 2 more minutes.");
        if (!ipa.Contains('ˈ')) throw new Exception($"no stress in \"{ipa}\" - the data did not load");
        if (!ipa.Contains("ɾ")) throw new Exception($"no tap in \"{ipa}\" - the rules did not run");
        if (!ipa.Contains("tˈuː")) throw new Exception($"\"2\" was not read as a word: \"{ipa}\"");

        // A name the dictionary does not have must still come out, which needs the embedded
        // letter-to-sound rules and not just the dictionary.
        var name = phonemizer.ToIpa("Aubriella");
        if (name.Length == 0) throw new Exception("the embedded letter-to-sound model did not load");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_DefineBeatsGuessingAndSurvivesASentence() => await RunTest(_ =>
    {
        // A name you KNOW should never be guessed at. Letter-to-sound is right about half the time,
        // and the words an application says most are exactly the ones CMUdict lacks.
        var phonemizer = MakePhonemizer();
        phonemizer.LetterToSound = null;   // so a miss is visibly a miss rather than a guess

        var before = phonemizer.ToIpa("Aubriella");
        if (before.Length != 0) throw new Exception($"expected nothing for an unknown word, got \"{before}\"");

        phonemizer.Define("Aubriella", "AO2 B R IY0 EH1 L AH0");

        // Stress lands on the ELL, which is the entire point - every "-ella" name in CMUdict is
        // stressed there and the guesser stresses the "au" instead.
        var after = phonemizer.ToIpa("Aubriella");
        if (!after.Contains("ˈɛ")) throw new Exception($"primary stress is not on the ELL: \"{after}\"");
        if (after.StartsWith('ˈ')) throw new Exception($"primary stress is still on the first syllable: \"{after}\"");

        // It has to hold mid-sentence too, not just alone.
        var sentence = phonemizer.ToIpa("my Aubriella");
        if (!sentence.Contains("ˈɛ")) throw new Exception($"the definition was lost in a sentence: \"{sentence}\"");

        // Defining REPLACES rather than adds. If the old pronunciations survived, homograph
        // resolution could still pick one of them, and the definition would not be authoritative.
        if (!phonemizer.Dictionary.TryLookupAll("Aubriella", out var all) || all.Count != 1)
            throw new Exception($"expected exactly one pronunciation after Define, got {all.Count}");

        // A word already in the dictionary can be corrected the same way.
        phonemizer.Define("water", "W AO1 T ER0 Z");
        if (!phonemizer.ToIpa("water").EndsWith('z')) throw new Exception("Define did not override an existing entry");

        // ...and removing it puts the word back to being unknown.
        phonemizer.Dictionary.Remove("Aubriella");
        if (phonemizer.ToIpa("Aubriella").Length != 0) throw new Exception("Remove did not forget the word");

        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_DefineRejectsPhonesItCannotSpeak() => await RunTest(_ =>
    {
        // A bad phone must fail HERE. Left unchecked it travels into the symbol stream and comes out
        // as a missing or wrong sound, which is far harder to trace back to the typo that caused it.
        var phonemizer = MakePhonemizer();

        Rejects(() => phonemizer.Define("x", "AO2 B XX9 R"), "an unknown phone");
        Rejects(() => phonemizer.Define("x", ""), "no phones at all");
        Rejects(() => phonemizer.Define("", "AO2"), "a blank word");

        // A vowel without its stress digit is the subtle one: it would otherwise be accepted and
        // quietly rendered unstressed, when stress is usually the whole reason for defining a word.
        Rejects(() => phonemizer.Define("x", "AO B R IY0"), "a vowel missing its stress digit");

        // Consonants legitimately carry no digit, so this must be ACCEPTED.
        phonemizer.Define("x", "B R IY0 Z");

        return Task.CompletedTask;

        static void Rejects(Action define, string what)
        {
            try { define(); }
            catch (ArgumentException) { return; }
            throw new Exception($"Define accepted {what}");
        }
    });

    private static void Expect(string actual, string expected)
    {
        if (actual != expected) throw new Exception($"expected \"{expected}\", got \"{actual}\"");
    }
}
