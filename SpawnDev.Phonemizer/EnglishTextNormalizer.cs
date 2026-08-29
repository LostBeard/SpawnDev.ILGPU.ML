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

    /// <summary>
    /// A number written with thousands separators, e.g. "1,234".
    /// </summary>
    /// <remarks>
    /// Requires a comma to actually be present, which is the whole point: the grouping says this is a
    /// QUANTITY. Nobody writes a year that way. The previous pattern also matched bare digit runs, so it
    /// simply stripped commas and let the year heuristic below see "1234" - and "I have 1,234 of them"
    /// was read as "twelve thirty-four". The information needed to get it right was present in the input
    /// and being discarded.
    /// </remarks>
    private static readonly Regex CommaNumber =
        new(@"\b([0-9]{1,3}(?:,[0-9]{3})+)\b", RegexOptions.Compiled);

    /// <summary>A clock time, "3:30" or "9:05".</summary>
    private static readonly Regex ClockTime =
        new(@"\b([0-9]{1,2}):([0-5][0-9])\b", RegexOptions.Compiled);

    /// <summary>An ampersand standing in for "and".</summary>
    private static readonly Regex Ampersand = new(@"\s*&\s*", RegexOptions.Compiled);

    /// <summary>A hash used as "number", as in "#1".</summary>
    private static readonly Regex NumberSign = new(@"#\s*(?=[0-9])", RegexOptions.Compiled);

    /// <summary>
    /// "St." meaning Street rather than Saint - it follows the name of the road.
    /// </summary>
    /// <remarks>
    /// The abbreviation table maps every "st" to "saint", which turns "123 Main St." into "Main saint".
    /// A saint's name FOLLOWS the abbreviation ("St. Louis") while a street's name PRECEDES it, so what
    /// comes before decides it: a number or a capitalised word means street.
    /// </remarks>
    private static readonly Regex StreetSuffix =
        new(@"\b(?<=(?:[0-9]|\p{Lu}\p{Ll}{1,20})\s)[Ss]t\b", RegexOptions.Compiled);
    private static readonly Regex Pounds = new(@"£([0-9,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Dollars = new(@"\$([0-9.,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Fraction = new(@"([0-9]+)/([0-9]+)", RegexOptions.Compiled);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex Percent = new(@"([0-9.,]*[0-9]+%)", RegexOptions.Compiled);
    private static readonly Regex Ordinal = new(@"[0-9]+(st|nd|rd|th)\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);
    private static readonly Regex PlainNumber = new(@"[0-9]+", RegexOptions.Compiled);
    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.Compiled);

    private static Regex Word(string word) => new($@"\b{word}\b", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // A title's full stop is an abbreviation mark, NOT a pause. Left in, "Dr. Tanner" phonemizes as
    // "doctor", a sentence break, then "Tanner" - and the model renders that break as an audible stop in
    // the middle of a person's name. Only stripped when something follows, so a sentence that genuinely
    // ends on an abbreviation keeps its stop.
    private static readonly Regex TitleStop =
        new(@"\b(mr|mrs|ms|dr|drs|st|jr|sr|col|sgt|capt|lt|gen|maj|rev|hon|esq|prof)\.(?=\s)",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Normalize a stretch of text for speech.</summary>
    public string Normalize(string text)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;

        text = MapPunctuation(text);
        text = TitleStop.Replace(text, "$1");     // before expansion, while the abbreviation is still short
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
        // Symbols that stand for words. Left alone they reach the phonemizer as punctuation and are
        // spoken as a pause or not at all - "Mr. & Mrs." loses the "and" entirely.
        text = Ampersand.Replace(text, " and ");
        text = NumberSign.Replace(text, "number ");

        // Before the table below, which would otherwise turn every "st" into "saint".
        text = StreetSuffix.Replace(text, "street");

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
        // Times before anything else touches the digits, or the colon survives into the output as a
        // stray symbol and "3:30" is spoken as "three, thirty".
        text = ClockTime.Replace(text, ExpandTime);

        // Grouped numbers are expanded HERE rather than merely de-comma'd, so the year heuristic never
        // sees them - the comma is what says "quantity, not year".
        text = CommaNumber.Replace(text, m =>
        {
            var digits = m.Groups[1].Value.Replace(",", "");
            if (!long.TryParse(digits, out var grouped)) return digits;

            // Grouping says QUANTITY, but English still reads a round quantity year-style: "fifteen
            // hundred apples", not "one thousand five hundred apples". So year-style survives only for
            // a whole number of hundreds - which is where it sounds natural - and everything else is
            // counted out. "1,234 of them" was being read as "twelve thirty-four"; nobody counts that
            // way, and equally nobody says "one thousand five hundred" for 1,500.
            return grouped is > 1000 and < 3000 && grouped % 100 == 0
                ? $" {NumberToWords.Year((int)grouped)} "
                : $" {NumberToWords.Cardinal(grouped)} ";
        });
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

    /// <summary>
    /// Read a clock time the way it is said: "three thirty", "nine oh five", "two o'clock".
    /// </summary>
    private static string ExpandTime(Match match)
    {
        if (!int.TryParse(match.Groups[1].Value, out var hour)) return match.Value;
        if (!int.TryParse(match.Groups[2].Value, out var minute)) return match.Value;
        if (hour > 23) return match.Value;

        var spokenHour = NumberToWords.Cardinal(hour);
        return minute switch
        {
            0 => $" {spokenHour} o'clock ",
            // A leading zero is spoken, not skipped: 9:05 is "nine oh five", never "nine five".
            < 10 => $" {spokenHour} oh {NumberToWords.Cardinal(minute)} ",
            _ => $" {spokenHour} {NumberToWords.Cardinal(minute)} ",
        };
    }

    /// <summary>Expand a bare run of digits, reading four-digit values in the 1001-2999 range as years.</summary>
    private static string ExpandNumber(Match match)
    {
        if (!long.TryParse(match.Value, out var value)) return match.Value;
        if (value is > 1000 and < 3000) return $" {NumberToWords.Year((int)value)} ";
        return $" {NumberToWords.Cardinal(value)} ";
    }
}
