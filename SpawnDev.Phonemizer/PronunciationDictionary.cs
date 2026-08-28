namespace SpawnDev.Phonemizer;

/// <summary>
/// A word-to-ARPAbet lookup, in the format CMUdict ships.
/// </summary>
/// <remarks>
/// <para>
/// CMUdict is BSD-2-Clause: free to use and redistribute, requiring only that its copyright notice and
/// disclaimer travel with it. That is the same obligation MIT places on users of this library, which is
/// why it is the backbone here instead of espeak-ng's GPL-3 dictionary.
/// </para>
/// <para>
/// Coverage, measured 2026-08-27 against a frequency list: 99.5% of the thousand most common English
/// words, 94.3% of the top ten thousand - and what it misses at that level is almost entirely
/// abbreviations and acronyms (rss, faq, apr, ny, gmt), which want expanding or spelling out rather than
/// guessing. Real out-of-vocabulary words are mostly proper names: "todd", "nikki", "tanner", "picard"
/// are all present; "aubriella" is not.
/// </para>
/// </remarks>
public sealed class PronunciationDictionary
{
    private readonly Dictionary<string, List<string[]>> _entries;

    private PronunciationDictionary(Dictionary<string, List<string[]>> entries) => _entries = entries;

    /// <summary>Number of headwords loaded.</summary>
    public int Count => _entries.Count;

    /// <summary>Parse the CMUdict text format: one entry per line, "word  P1 P2 P3".</summary>
    /// <remarks>
    /// Alternate pronunciations are written "word(2)" and are KEPT, in the order the file lists them.
    /// <see cref="TryLookup"/> returns the first, which is the right default; choosing between them needs
    /// sentence context, and <see cref="Homographs"/> does that for the words where the choice changes
    /// which syllable is stressed - "the RECord" against "to reCORD".
    /// </remarks>
    public static PronunciationDictionary Parse(IEnumerable<string> lines)
    {
        var entries = new Dictionary<string, List<string[]>>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in lines)
        {
            var body = line;
            int comment = body.IndexOf('#');
            if (comment >= 0) body = body[..comment];
            body = body.Trim();
            if (body.Length == 0) continue;

            var parts = body.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;

            // "word(2)" is another pronunciation of "word", not another word.
            var headword = parts[0];
            int paren = headword.IndexOf('(');
            if (paren > 0) headword = headword[..paren];

            if (!entries.TryGetValue(headword, out var all)) entries[headword] = all = new List<string[]>();
            all.Add(parts[1..]);
        }
        return new PronunciationDictionary(entries);
    }

    /// <summary>Load from a CMUdict file on disk.</summary>
    public static PronunciationDictionary Load(string path) => Parse(File.ReadLines(path));

    /// <summary>Look up a word's first ARPAbet pronunciation. Case-insensitive.</summary>
    public bool TryLookup(string word, out string[] phones)
    {
        if (_entries.TryGetValue(word, out var all) && all.Count > 0) { phones = all[0]; return true; }
        phones = [];
        return false;
    }

    /// <summary>Look up EVERY pronunciation the dictionary lists, in file order.</summary>
    /// <remarks>
    /// The alternates are what make a homograph solvable: "record" is stored twice, once stressed on the
    /// first syllable and once on the second, with different vowels in each - so re-stressing one of them
    /// is not the same as choosing the other, and would produce a word nobody says.
    /// </remarks>
    public bool TryLookupAll(string word, out IReadOnlyList<string[]> pronunciations)
    {
        if (_entries.TryGetValue(word, out var all)) { pronunciations = all; return true; }
        pronunciations = [];
        return false;
    }
}
