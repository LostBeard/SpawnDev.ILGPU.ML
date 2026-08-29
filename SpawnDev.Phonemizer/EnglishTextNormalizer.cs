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
        (Word("st"), "saint"), (Word("jr"), "junior"), (Word("maj"), "major"),
        (Word("lt"), "lieutenant"),
        (Word("sgt"), "sergeant"), (Word("capt"), "captain"),
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

    /// <summary>
    /// Unit abbreviations, as (singular, plural), expanded only when a number precedes them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number is what makes this safe. "5 kg" is a weight; a bare "kg" in running text might be
    /// anything, and "in", "m" and "s" are ordinary English words - so single letters are deliberately
    /// absent from this table. That is the whole ambiguity people warn about, and the fix is not to
    /// guess but to require the evidence.
    /// </para>
    /// <para>
    /// "ft" is here for a second reason: the abbreviation table below maps it to "fort", so "6 ft tall"
    /// was read as "six fort tall". Same fault as "St." meaning saint.
    /// </para>
    /// <para>
    /// Spellings are American because the dictionary is: every word here was checked against it, and
    /// only "gigahertz" was missing - so it is not offered. Expanding to a word the dictionary lacks
    /// only moves the problem to letter-to-sound.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<string, (string One, string Many)> Units =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["km"] = ("kilometer", "kilometers"), ["kg"] = ("kilogram", "kilograms"),
            ["cm"] = ("centimeter", "centimeters"), ["mm"] = ("millimeter", "millimeters"),
            ["ml"] = ("milliliter", "milliliters"), ["mg"] = ("milligram", "milligrams"),
            ["kb"] = ("kilobyte", "kilobytes"), ["mb"] = ("megabyte", "megabytes"),
            ["gb"] = ("gigabyte", "gigabytes"), ["tb"] = ("terabyte", "terabytes"),
            ["mph"] = ("mile per hour", "miles per hour"),
            ["kph"] = ("kilometer per hour", "kilometers per hour"),
            ["lb"] = ("pound", "pounds"), ["lbs"] = ("pound", "pounds"),
            ["oz"] = ("ounce", "ounces"), ["ft"] = ("foot", "feet"),
            ["hr"] = ("hour", "hours"), ["hrs"] = ("hour", "hours"),
            ["min"] = ("minute", "minutes"), ["mins"] = ("minute", "minutes"),
            ["sec"] = ("second", "seconds"), ["secs"] = ("second", "seconds"),
        };

    /// <summary>A number followed by a unit abbreviation, "5km" or "10 kg".</summary>
    private static readonly Regex NumberWithUnit =
        new(@"\b([0-9]+(?:\.[0-9]+)?)\s*(km|kg|cm|mm|ml|mg|kb|mb|gb|tb|mph|kph|lbs|lb|oz|ft|hrs|hr|mins|min|secs|sec)\b",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>A clock time, "3:30" or "9:05".</summary>
    private static readonly Regex ClockTime =
        new(@"\b([0-9]{1,2}):([0-5][0-9])\b", RegexOptions.Compiled);

    /// <summary>An ampersand standing in for "and".</summary>
    private static readonly Regex Ampersand = new(@"\s*&\s*", RegexOptions.Compiled);

    /// <summary>
    /// Arithmetic written between numbers, "5 + 3 = 8".
    /// </summary>
    /// <remarks>
    /// Only BETWEEN numbers. A bare "+" or "-" in prose is punctuation or a dash, and reading every one
    /// of them as a word would be worse than leaving them silent - which is why the minus below is
    /// restricted to a value that clearly is one.
    /// </remarks>
    private static readonly Regex Arithmetic =
        new(@"(?<=[0-9]\s?)([+*=])(?=\s?[0-9])", RegexOptions.Compiled);

    /// <summary>A negative value: a minus attached to a number, at a boundary rather than inside a range.</summary>
    private static readonly Regex NegativeNumber =
        new(@"(?<![0-9A-Za-z])-([0-9]+(?:\.[0-9]+)?)", RegexOptions.Compiled);

    /// <summary>
    /// A hyphen joining a word to a number, as in "COVID-19" or "3-D".
    /// </summary>
    /// <remarks>
    /// The hyphen survives into the output as punctuation, so "COVID-19" is spoken with a pause in the
    /// middle of a single word. Replacing it with a space lets both halves be read normally.
    /// </remarks>
    private static readonly Regex WordNumberHyphen =
        new(@"(?<=\p{L})-(?=[0-9])|(?<=[0-9])-(?=\p{L})", RegexOptions.Compiled);

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

    /// <summary>An abbreviation that is only an abbreviation when it carries its full stop.</summary>
    private static Regex WordWithStop(string word)
        => new($@"\b{word}\.", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Abbreviations that are ALSO ordinary English words or prefixes, so they need their period.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These were in the table below, matched on a word boundary alone, and it misfired: "co-op" was read
    /// as "company op", because a hyphen is a word boundary. "rev the engine" becomes "reverend the
    /// engine" by the same route, and "gen" and "hon" are a generation and a term of endearment as often
    /// as they are a general and an honorable.
    /// </para>
    /// <para>
    /// ⚠️ They must be expanded BEFORE <see cref="TitleStop"/>, which strips the period off exactly this
    /// set of words - so by the time the main table runs, the evidence that distinguishes them is gone.
    /// </para>
    /// </remarks>
    private static readonly (Regex Pattern, string Replacement)[] StopRequiredAbbreviations =
    [
        (WordWithStop("co"), "company"),
        (WordWithStop("rev"), "reverend"),
        (WordWithStop("gen"), "general"),
        (WordWithStop("hon"), "honorable"),
    ];

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
        // BEFORE TitleStop, which strips the very period these depend on.
        foreach (var (pattern, replacement) in StopRequiredAbbreviations)
            text = pattern.Replace(text, replacement);
        text = TitleStop.Replace(text, "$1");     // before expansion, while the abbreviation is still short
        // Before the abbreviation table, which would otherwise read "6 ft" as "six fort".
        text = NumberWithUnit.Replace(text, ExpandUnit);
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
        text = WordNumberHyphen.Replace(text, " ");
        text = Arithmetic.Replace(text, m => m.Value switch
        {
            "+" => " plus ",
            "*" => " times ",
            _ => " equals ",
        });
        text = NegativeNumber.Replace(text, " minus $1");

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
    /// Spell out a unit abbreviation, agreeing in number with the value in front of it.
    /// </summary>
    /// <remarks>
    /// Only "1" takes the singular. "1.5 km" is plural in English ("one point five kilometers"), and so
    /// is "0 kg" - the singular is the special case, not the default.
    /// </remarks>
    private static string ExpandUnit(Match match)
    {
        if (!Units.TryGetValue(match.Groups[2].Value, out var unit)) return match.Value;
        var value = match.Groups[1].Value;
        return $"{value} {(value == "1" ? unit.One : unit.Many)}";
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
