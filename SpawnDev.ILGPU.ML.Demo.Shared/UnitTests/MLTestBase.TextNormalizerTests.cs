using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Locks the text normalization a listener notices first.
/// </summary>
/// <remarks>
/// Normalization matters more than its size suggests: CMUdict covers 99.5% of the thousand most common
/// English words, and what it misses at that level is abbreviations and units rather than exotic
/// vocabulary. Those want expanding, not guessing. And "$1.50" read as "dollar one point five zero" is
/// wrong in a way no amount of phonetic accuracy repairs.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Normalizer_ReadsYearsAsYears() => await RunTest(_ =>
    {
        // "one thousand nine hundred ninety-nine" is correct English for 1999 and sounds nothing like a
        // year. Four digits in this range are overwhelmingly dates.
        var n = new EnglishTextNormalizer();
        ExpectText(n.Normalize("in 1999"), "in nineteen ninety-nine");
        ExpectText(n.Normalize("in 1905"), "in nineteen oh five");
        ExpectText(n.Normalize("in 1900"), "in nineteen hundred");
        ExpectText(n.Normalize("in 2000"), "in two thousand");
        ExpectText(n.Normalize("in 2005"), "in two thousand five");
        ExpectText(n.Normalize("in 2026"), "in twenty twenty-six");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Normalizer_ReadsPlainNumbers() => await RunTest(_ =>
    {
        var n = new EnglishTextNormalizer();
        ExpectText(n.Normalize("0 apples"), "zero apples");
        ExpectText(n.Normalize("15 apples"), "fifteen apples");
        ExpectText(n.Normalize("123 apples"), "one hundred twenty-three apples");
        // Commas must come out of a number first, or it reads as two numbers with a pause between them.
        ExpectText(n.Normalize("12,345 apples"), "twelve thousand, three hundred forty-five apples");
        // And any four-digit value in the 1001-2999 range reads year-style even when it is not a year -
        // which is both what the reference frontend does and what an English speaker says: "fifteen
        // hundred apples", not "one thousand five hundred apples".
        ExpectText(n.Normalize("1,500 apples"), "fifteen hundred apples");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Normalizer_ReadsMoneyAsMoney() => await RunTest(_ =>
    {
        // The order of the rules is what makes this work: currency has to be handled before the plain
        // decimal rule, or $1.50 becomes "one point five zero dollars".
        var n = new EnglishTextNormalizer();
        ExpectText(n.Normalize("it costs $1.50"), "it costs one dollar, fifty cents");
        ExpectText(n.Normalize("it costs $1"), "it costs one dollar");
        ExpectText(n.Normalize("it costs $20"), "it costs twenty dollars");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Normalizer_ReadsOrdinalsFractionsAndPercent() => await RunTest(_ =>
    {
        var n = new EnglishTextNormalizer();
        ExpectText(n.Normalize("the 1st of May"), "the first of May");
        ExpectText(n.Normalize("the 23rd time"), "the twenty-third time");
        ExpectText(n.Normalize("1/2 of it"), "one half of it");
        ExpectText(n.Normalize("3/4 done"), "three quarters done");
        ExpectText(n.Normalize("5% better"), "five percent better");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Normalizer_ExpandsAbbreviations() => await RunTest(_ =>
    {
        // Whole-word and case-insensitive, longest first so "drs" is not eaten by "dr".
        var n = new EnglishTextNormalizer();
        ExpectText(n.Normalize("Dr. Tanner"), "doctor. Tanner");
        ExpectText(n.Normalize("Mrs. Tanner"), "missus. Tanner");
        ExpectText(n.Normalize("etc."), "et cetera.");
        // A word that merely CONTAINS an abbreviation must survive intact.
        ExpectText(n.Normalize("street drama"), "street drama");
        return Task.CompletedTask;
    });

    [TestMethod]
    public async Task Normalizer_NumbersAreOrdinalAware() => await RunTest(_ =>
    {
        // Only the final word takes the ordinal form, and the irregulars are irregular.
        ExpectText(NumberToWords.Ordinal(1), "first");
        ExpectText(NumberToWords.Ordinal(2), "second");
        ExpectText(NumberToWords.Ordinal(5), "fifth");
        ExpectText(NumberToWords.Ordinal(12), "twelfth");
        ExpectText(NumberToWords.Ordinal(20), "twentieth");
        ExpectText(NumberToWords.Ordinal(21), "twenty-first");
        ExpectText(NumberToWords.Ordinal(123), "one hundred twenty-third");
        return Task.CompletedTask;
    });

    private static void ExpectText(string actual, string expected)
    {
        if (actual != expected) throw new Exception($"expected \"{expected}\", got \"{actual}\"");
    }
}
