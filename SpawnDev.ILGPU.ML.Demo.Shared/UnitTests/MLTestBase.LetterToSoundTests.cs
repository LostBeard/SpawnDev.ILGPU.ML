using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Guards the contract that lets an unknown word still be spoken.
/// </summary>
/// <remarks>
/// The model itself is trained and measured separately (see <c>tools/lts-train</c>, which reports its
/// held-out accuracy into the model file's header). What is locked here is the OUTPUT CONTRACT, because
/// breaking it fails silently: when the runtime skipped the stress model, every vowel came back without
/// a stress digit, every one of those was then discarded downstream as an unrecognised phone, and
/// "Aubriella" was pronounced "bɹl" - the consonants alone, with no error anywhere.
///
/// The model is written inline so these tests need no trained data and say exactly which rule produced
/// which result.
/// </remarks>
public abstract partial class MLTestBase
{
    // Sound rules for the letter alone, then stress rules prefixed with *.
    private const string TinyLtsModel = """
        # a tiny hand-written model, for testing the contract rather than the accuracy
        [c]	K
        [a]	AE
        [t]	T
        [e]	-
        *[a]	1
        """;

    [TestMethod]
    public async Task LetterToSound_EveryVowelCarriesAStressDigit() => await RunPureTest(() =>
    {
        // A vowel without a digit is not valid ARPAbet, and downstream it is dropped as unknown. This is
        // the exact defect that turned a name into its consonants.
        var lts = LetterToSound.Parse(TinyLtsModel.Split('\n'));
        var phones = lts.Predict("cat");

        foreach (var phone in phones)
        {
            bool isVowel = phone.StartsWith("AE") || phone.StartsWith("AH") || phone.StartsWith("IH");
            if (isVowel && !char.IsDigit(phone[^1]))
                throw new Exception($"vowel {phone} came back with no stress digit");
        }
        if (!phones.Contains("K") || !phones.Contains("T"))
            throw new Exception("consonants went missing: " + string.Join(" ", phones));
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task LetterToSound_SilentLettersProduceNothing() => await RunPureTest(() =>
    {
        // "-" means the letter is silent, which is how the model spells the e in "cate".
        var lts = LetterToSound.Parse(TinyLtsModel.Split('\n'));
        var withSilent = lts.Predict("cate");
        var without = lts.Predict("cat");
        if (!withSilent.SequenceEqual(without))
            throw new Exception($"a silent letter changed the result: {string.Join(" ", withSilent)}");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task LetterToSound_MarksExactlyOnePrimaryStress() => await RunPureTest(() =>
    {
        // English words have one primary stress. The per-letter model can mark several - "Tuvok" came
        // back with two - and stress is what the downstream model punishes hardest, so emitting two is
        // worse than picking the wrong one. Adding this constraint raised held-out accuracy from 42.6%
        // from 42.6% to 43.7%. Both figures predate the +/-5 model; the constraint itself is unchanged.
        var lts = LetterToSound.Parse(TinyLtsModel.Split('\n'));
        var phones = lts.Predict("catcat");           // two a's, both marked stressed by the rule above

        int primaries = phones.Count(p => p.EndsWith('1'));
        if (primaries != 1)
            throw new Exception($"expected exactly one primary stress, got {primaries}: {string.Join(" ", phones)}");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task LetterToSound_AlwaysStressesSomething() => await RunPureTest(() =>
    {
        // With no stress rule at all, the fallback has to stress the first vowel - a word with no
        // stressed syllable leaves a hole where a beat belongs.
        var lts = LetterToSound.Parse(new[] { "[c]\tK", "[a]\tAE", "[t]\tT" });
        var phones = lts.Predict("cat");
        if (!phones.Any(p => p.EndsWith('1')))
            throw new Exception("no syllable was stressed: " + string.Join(" ", phones));
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task LetterToSound_NeverReturnsAWordWithNoVowelAtAll() => await RunPureTest(() =>
    {
        // The failure this guards is the one that turned "Aubriella" into "bɹl" - a word spelled with
        // vowels coming back as bare consonants. It is unambiguous, because no English word is all
        // consonants, and it is catastrophic rather than merely wrong: the word becomes unsayable.
        //
        // Here every vowel rule says SILENT, which is what a real model does when it has never seen a
        // spelling. The ! rules are the last resort, used only when the result would otherwise have no
        // vowel - a narrow trigger, chosen because three wider ones each measured WORSE (see
        // tools/lts-train --analyze).
        var lts = LetterToSound.Parse(new[]
        {
            "[b]\tB", "[l]\tL", "[r]\tR",
            "[a]\t-", "[e]\t-",          // every vowel silent, so the word would come back as consonants
            "![a]\tAH", "![e]\tEH",      // ...unless the last resort speaks up
        });

        var phones = lts.Predict("bralea");
        if (!phones.Any(p => p.StartsWith("AH") || p.StartsWith("EH")))
            throw new Exception("a word spelled with vowels came back with none: " + string.Join(" ", phones));
        if (!phones.Any(p => p.EndsWith('1')))
            throw new Exception("the rescued word still has no stressed syllable: " + string.Join(" ", phones));
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task LetterToSound_LastResortDoesNotFireOnAWordThatAlreadySpeaks() => await RunPureTest(() =>
    {
        // The narrowness IS the feature. Letting the fallback fire whenever a letter came back silent
        // cost 19.5 points on held-out words, because a silent letter is usually CORRECT - English is
        // full of them. "cate" must still lose its final e even though a ! rule exists for it.
        var lts = LetterToSound.Parse(new[] { "[c]\tK", "[a]\tAE", "[t]\tT", "[e]\t-", "![e]\tEH" });

        var withSilent = lts.Predict("cate");
        var without = lts.Predict("cat");
        if (!withSilent.SequenceEqual(without))
            throw new Exception($"the last resort fired on a word that already spoke: {string.Join(" ", withSilent)}");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Phonemizer_SoundsOutWordsTheDictionaryLacks() => await RunPureTest(() =>
    {
        // The whole point: an unknown word still gets spoken, and is still REPORTED as unknown, because
        // "the dictionary did not have this" is the difference between a pronunciation and a guess.
        var dictionary = PronunciationDictionary.Parse(new[] { "the DH AH0" });
        var p = new EnglishPhonemizer(dictionary)
        {
            LetterToSound = LetterToSound.Parse(TinyLtsModel.Split('\n')),
            Normalizer = null,
        };

        var ipa = p.ToIpa("cat");
        if (ipa.Length == 0) throw new Exception("an unknown word produced nothing at all");
        if (!ipa.Contains('ˈ')) throw new Exception($"a sounded-out word must carry stress, got \"{ipa}\"");
        if (p.LastUnknownWords.Count != 1 || p.LastUnknownWords[0] != "cat")
            throw new Exception("a sounded-out word must still be reported as absent from the dictionary");
        return Task.CompletedTask;
    });
}
