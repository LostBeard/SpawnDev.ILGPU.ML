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

    private static void Expect(string actual, string expected)
    {
        if (actual != expected) throw new Exception($"expected \"{expected}\", got \"{actual}\"");
    }
}
