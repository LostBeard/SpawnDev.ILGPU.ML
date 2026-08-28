namespace SpawnDev.Phonemizer;

/// <summary>
/// Pronounces an unknown word by finding the known word inside it.
/// </summary>
/// <remarks>
/// <para>
/// "Unfriendliest" is not in the dictionary. "Friend" is. Guessing the whole word letter by letter throws
/// away that fact, and letter-to-sound is only right about 44% of the time - so anything that can be
/// DERIVED instead of guessed should be. 23% of CMUdict's own entries are a known stem plus a common
/// ending, which is the size of the class this covers.
/// </para>
/// <para>
/// The endings carry real English allomorphy rather than one fixed sound each: the -s of "cats" is an S,
/// the -s of "dogs" is a Z, and the -s of "boxes" is a whole extra syllable. Getting that wrong is
/// audible, so the rules follow the final sound of the stem, which is exactly what a speaker does.
/// </para>
/// </remarks>
public static class WordDecomposer
{
    /// <summary>Phones for the suffix, chosen from the sound the stem ends on.</summary>
    private delegate string[] SuffixPhones(string lastPhone);

    // Sounds that make a following -s or -ed take an extra syllable.
    private static readonly HashSet<string> Sibilants = new(StringComparer.Ordinal)
        { "S", "Z", "SH", "ZH", "CH", "JH" };

    // Voiceless consonants: they make -s an S and -ed a T.
    private static readonly HashSet<string> Voiceless = new(StringComparer.Ordinal)
        { "P", "T", "K", "F", "TH", "S", "SH", "CH", "HH" };

    private static readonly (string Ending, SuffixPhones Phones)[] Suffixes =
    [
        // Plural and third person. cats = S, dogs = Z, boxes = a syllable.
        ("s",    last => Sibilants.Contains(last) ? ["IH0", "Z"] : Voiceless.Contains(last) ? ["S"] : ["Z"]),
        ("es",   last => Sibilants.Contains(last) ? ["IH0", "Z"] : Voiceless.Contains(last) ? ["S"] : ["Z"]),
        // Past tense. walked = T, played = D, wanted = a syllable.
        ("ed",   last => last is "T" or "D" ? ["IH0", "D"] : Voiceless.Contains(last) ? ["T"] : ["D"]),
        ("d",    last => last is "T" or "D" ? ["IH0", "D"] : Voiceless.Contains(last) ? ["T"] : ["D"]),
        ("ing",  _ => ["IH0", "NG"]),
        ("ings", _ => ["IH0", "NG", "Z"]),
        ("ly",   _ => ["L", "IY0"]),
        ("er",   _ => ["ER0"]),
        ("ers",  _ => ["ER0", "Z"]),
        ("est",  _ => ["IH0", "S", "T"]),
        ("ness", _ => ["N", "AH0", "S"]),
        ("less", _ => ["L", "AH0", "S"]),
        ("ment", _ => ["M", "AH0", "N", "T"]),
        ("ful",  _ => ["F", "AH0", "L"]),
    ];

    // Prefixes whose pronunciation does not depend on what follows.
    private static readonly (string Prefix, string[] Phones)[] Prefixes =
    [
        ("un",    ["AH0", "N"]),
        ("re",    ["R", "IY0"]),
        ("dis",   ["D", "IH0", "S"]),
        ("mis",   ["M", "IH0", "S"]),
        ("non",   ["N", "AA1", "N"]),
        ("pre",   ["P", "R", "IY0"]),
        ("over",  ["OW1", "V", "ER0"]),
        ("under", ["AH1", "N", "D", "ER0"]),
    ];

    /// <summary>
    /// Try to pronounce <paramref name="word"/> by decomposing it into known parts.
    /// </summary>
    /// <returns>true when a known stem was found and phones were produced.</returns>
    public static bool TryDecompose(string word, PronunciationDictionary dictionary, out string[] phones)
    {
        phones = [];
        if (string.IsNullOrEmpty(word) || word.Length < 4) return false;
        var lower = word.ToLowerInvariant();

        foreach (var (ending, suffixPhones) in Suffixes)
        {
            if (!lower.EndsWith(ending, StringComparison.Ordinal)) continue;
            var body = lower[..^ending.Length];
            if (body.Length < 2) continue;

            // Three ways English spells a stem before an ending: as is ("walk" + ed), with an e that the
            // ending swallowed ("hope" + d), and with a doubled final consonant ("run" + ning).
            foreach (var stem in new[] { body, body + "e", Undouble(body) })
            {
                if (stem.Length < 2 || !dictionary.TryLookup(stem, out var stemPhones)) continue;
                if (stemPhones.Length == 0) continue;

                var last = Strip(stemPhones[^1]);
                phones = [.. stemPhones, .. suffixPhones(last)];
                return true;
            }
        }

        foreach (var (prefix, prefixPhones) in Prefixes)
        {
            if (!lower.StartsWith(prefix, StringComparison.Ordinal)) continue;
            var stem = lower[prefix.Length..];
            if (stem.Length < 3 || !dictionary.TryLookup(stem, out var stemPhones)) continue;

            phones = [.. prefixPhones, .. stemPhones];
            return true;
        }

        return false;
    }

    /// <summary>"running" minus "ing" is "runn"; the stem is "run".</summary>
    private static string Undouble(string body)
        => body.Length >= 2 && body[^1] == body[^2] && !"aeiou".Contains(body[^1]) ? body[..^1] : body;

    private static string Strip(string phone)
        => phone.Length > 0 && char.IsDigit(phone[^1]) ? phone[..^1] : phone;
}
