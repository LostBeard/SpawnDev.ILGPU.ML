using System.Text;
using System.Text.RegularExpressions;

namespace SpawnDev.Phonemizer;

/// <summary>
/// Turns written English into the spoken English a pronunciation dictionary can actually look up.
/// </summary>
/// <remarks>
/// <para>
/// This carries more weight than it looks like it should. Measured against a frequency list, CMUdict
/// covers 99.5% of the thousand most common English words - and what it misses is almost entirely
/// abbreviations, acronyms, state codes and units. Those want EXPANDING or SPELLING OUT, not guessing
/// phonetically, which makes normalization a bigger share of the real gap than letter-to-sound is.
/// </para>
/// <para>
/// It is also the part a listener notices first. "$1.50" read as "dollar one point five zero" is wrong in
/// a way no amount of phonetic accuracy repairs.
/// </para>
/// <para>
/// The rules and their ORDER follow ZipVoice's own normalizer (Apache-2.0), which is what the model was
/// trained on, reimplemented here rather than copied. One deliberate difference: the reference expands
/// "Mrs" to "misess", which is a typo and is not a word in any dictionary; this expands it to "missus",
/// which is real English and is in CMUdict.
/// </para>
/// </remarks>
public sealed class EnglishTextNormalizer
{
    // Whole-word, case-insensitive. Ordered longest-first where one is a prefix of another, so "drs" is
    // not eaten by "dr".
    private static readonly (Regex Pattern, string Replacement)[] Abbreviations =
    [
        (Word("mrs"), "missus"), (Word("drs"), "doctors"), (Word("mr"), "mister"), (Word("dr"), "doctor"),
        (Word("st"), "saint"), (Word("co"), "company"), (Word("jr"), "junior"), (Word("maj"), "major"),
        (Word("gen"), "general"), (Word("rev"), "reverend"), (Word("lt"), "lieutenant"),
        (Word("hon"), "honorable"), (Word("sgt"), "sergeant"), (Word("capt"), "captain"),
        (Word("esq"), "esquire"), (Word("ltd"), "limited"), (Word("col"), "colonel"),
        (Word("ft"), "fort"), (Word("etc"), "et cetera"), (Word("btw"), "by the way"),
    ];

    private static readonly Regex CommaNumber = new(@"([0-9][0-9,]+[0-9])", RegexOptions.Compiled);
    private static readonly Regex Pounds = new(@"£([0-9,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Dollars = new(@"\$([0-9.,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Fraction = new(@"([0-9]+)/([0-9]+)", RegexOptions.Compiled);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Percent = new(@"([0-9.,]*[0-9]+%)", RegexOptions.Compiled);
    private static readonly Regex Ordinal = new(@"[0-9]+(st|nd|rd|th)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex PlainNumber = new(@"[0-9]+", RegexOptions.Compiled);
    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.Compiled);

    private static Regex Word(string word) => new($@"\b{word}\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Normalize a stretch of text for speech.</summary>
    public string Normalize(string text)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;

        text = MapPunctuation(text);
        text = ExpandAbbreviations(text);
        text = NormalizeNumbers(text);
        return Whitespace.Replace(text, " ").Trim();
    }

    /// <summary>Fold typographic and CJK punctuation onto the ASCII marks the model was trained on.</summary>
    private static string MapPunctuation(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            sb.Append(c switch
            {
                '，' or '、' => ",",
                '。' => ".",
                '！' => "!",
                '？' => "?",
                '；' => ";",
                '：' => ":",
                '‘' or '’' => "'",
                '“' or '”' => "\"",
                '–' or '—' => " ",     // a dash is a pause, not a sound
                '⋯' => "…",
                _ => c.ToString(),
            });
        }
        return sb.Replace("...", "…").ToString();
    }

    private static string ExpandAbbreviations(string text)
    {
        foreach (var (pattern, replacement) in Abbreviations)
            text = pattern.Replace(text, replacement);
        return text;
    }

    /// <summary>
    /// Expand every numeric form, in an order that matters.
    /// </summary>
    /// <remarks>
    /// Commas come out of numbers first, or "1,500" would be read as two numbers with a pause between
    /// them. Currency and fractions run before the plain decimal rule, because "$1.50" is "one dollar
    /// fifty cents" rather than "one point five zero dollars". Plain digits are expanded last, so every
    /// more specific rule gets its chance first.
    /// </remarks>
    private static string NormalizeNumbers(string text)
    {
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Pounds.Replace(text, m => $" {m.Groups[1].Value} pounds ");
        text = Dollars.Replace(text, ExpandDollars);
        text = Fraction.Replace(text, ExpandFraction);
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = Percent.Replace(text, m => m.Groups[1].Value.Replace("%", " percent "));
        text = Ordinal.Replace(text, m => long.TryParse(m.Value[..^2], out var n) ? $" {NumberToWords.Ordinal(n)} " : m.Value);
        text = PlainNumber.Replace(text, ExpandNumber);
        return text;
    }

    private static string ExpandDollars(Match match)
    {
        var parts = match.Groups[1].Value.Split('.');
        if (parts.Length > 2) return $" {match.Groups[1].Value} dollars ";       // not an amount we understand

        long dollars = parts[0].Length > 0 && long.TryParse(parts[0], out var d) ? d : 0;
        long cents = parts.Length > 1 && parts[1].Length > 0 && long.TryParse(parts[1].PadRight(2, '0')[..2], out var c) ? c : 0;

        var dollarWord = dollars == 1 ? "dollar" : "dollars";
        var centWord = cents == 1 ? "cent" : "cents";
        if (dollars > 0 && cents > 0) return $" {dollars} {dollarWord}, {cents} {centWord} ";
        if (dollars > 0) return $" {dollars} {dollarWord} ";
        if (cents > 0) return $" {cents} {centWord} ";
        return " zero dollars ";
    }

    private static string ExpandFraction(Match match)
    {
        if (!long.TryParse(match.Groups[1].Value, out var numerator)) return match.Value;
        if (!long.TryParse(match.Groups[2].Value, out var denominator) || denominator == 0) return match.Value;

        if (numerator == 1 && denominator == 2) return " one half ";
        if (numerator == 1 && denominator == 4) return " one quarter ";
        if (denominator == 2) return $" {NumberToWords.Cardinal(numerator)} halves ";
        if (denominator == 4) return $" {NumberToWords.Cardinal(numerator)} quarters ";
        return $" {NumberToWords.Cardinal(numerator)} {NumberToWords.Ordinal(denominator)}s ";
    }

    /// <summary>Expand a bare run of digits, reading four-digit values in the 1001-2999 range as years.</summary>
    private static string ExpandNumber(Match match)
    {
        if (!long.TryParse(match.Value, out var value)) return match.Value;
        if (value is > 1000 and < 3000) return $" {NumberToWords.Year((int)value)} ";
        return $" {NumberToWords.Cardinal(value)} ";
    }
}
