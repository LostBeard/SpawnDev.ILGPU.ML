using System.Text;

namespace SpawnDev.Phonemizer;

/// <summary>
/// Spells numbers the way an English speaker reads them aloud.
/// </summary>
/// <remarks>
/// A phonemizer cannot look up "1999" in a pronunciation dictionary, so text normalization has to turn it
/// into words first - and how it does that is audible: "nineteen ninety-nine" and "one thousand nine
/// hundred ninety-nine" are both correct English for the same digits, but only one of them sounds like a
/// year. This follows the conventions the reference frontend uses, which is what the model was trained on.
/// </remarks>
public static class NumberToWords
{
    private static readonly string[] Units =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] Tens =
    [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    // Long scale words in ascending order, each a thousand times the last.
    private static readonly string[] Scales = ["", " thousand", " million", " billion", " trillion"];

    /// <summary>Ordinal forms that are irregular, keyed by the cardinal word they replace.</summary>
    private static readonly Dictionary<string, string> IrregularOrdinals = new(StringComparer.Ordinal)
    {
        ["one"] = "first", ["two"] = "second", ["three"] = "third", ["five"] = "fifth",
        ["eight"] = "eighth", ["nine"] = "ninth", ["twelve"] = "twelfth",
    };

    /// <summary>Spell a whole number: 123 becomes "one hundred twenty-three".</summary>
    /// <remarks>
    /// No "and" before the tens - "one hundred twenty-three", not "one hundred and twenty-three" - which
    /// is the American convention the reference frontend follows. Scale groups are separated by a comma
    /// because that comma is a real pause when the number is spoken.
    /// </remarks>
    public static string Cardinal(long value)
    {
        if (value == 0) return Units[0];
        if (value < 0) return "minus " + Cardinal(-value);

        // Split into groups of three digits, least significant first.
        var groups = new List<int>();
        for (long remaining = value; remaining > 0; remaining /= 1000) groups.Add((int)(remaining % 1000));
        if (groups.Count > Scales.Length) return value.ToString();   // beyond trillions, read as digits

        var parts = new List<string>();
        for (int i = groups.Count - 1; i >= 0; i--)
        {
            if (groups[i] == 0) continue;
            parts.Add(UnderThousand(groups[i]) + Scales[i]);
        }
        return string.Join(", ", parts);
    }

    /// <summary>Spell an ordinal: 23 becomes "twenty-third".</summary>
    public static string Ordinal(long value)
    {
        var words = Cardinal(value);

        // Only the FINAL word changes: "one hundred twenty-three" becomes "one hundred twenty-third".
        int split = words.LastIndexOfAny([' ', '-']);
        var head = split < 0 ? "" : words[..(split + 1)];
        var tail = split < 0 ? words : words[(split + 1)..];

        if (IrregularOrdinals.TryGetValue(tail, out var irregular)) return head + irregular;
        if (tail.EndsWith('y')) return head + tail[..^1] + "ieth";     // twenty -> twentieth
        return head + tail + "th";
    }

    /// <summary>
    /// Spell a number the way a YEAR is read: 1999 becomes "nineteen ninety-nine".
    /// </summary>
    /// <remarks>
    /// Applies to 1001-2999, which is the range where a bare four-digit number is overwhelmingly a year.
    /// The turn of a century is read as "nineteen hundred", and the two-thousands are read in full ("two
    /// thousand five") rather than as pairs, because nobody says "twenty oh five".
    /// </remarks>
    public static string Year(int value)
    {
        if (value <= 1000 || value >= 3000) return Cardinal(value);
        if (value == 2000) return "two thousand";
        if (value is > 2000 and < 2010) return "two thousand " + Cardinal(value % 100);
        if (value % 100 == 0) return Cardinal(value / 100) + " hundred";

        int high = value / 100, low = value % 100;
        return low < 10
            ? $"{Cardinal(high)} oh {Cardinal(low)}"      // nineteen oh five
            : $"{Cardinal(high)} {Cardinal(low)}";        // nineteen ninety-nine
    }

    private static string UnderThousand(int value)
    {
        var sb = new StringBuilder();
        if (value >= 100)
        {
            sb.Append(Units[value / 100]).Append(" hundred");
            value %= 100;
            if (value > 0) sb.Append(' ');
        }
        if (value == 0) return sb.ToString();
        if (value < 20) { sb.Append(Units[value]); return sb.ToString(); }

        sb.Append(Tens[value / 10]);
        if (value % 10 > 0) sb.Append('-').Append(Units[value % 10]);
        return sb.ToString();
    }
}
