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
    private readonly Dictionary<string, string[]> _entries;

    private PronunciationDictionary(Dictionary<string, string[]> entries) => _entries = entries;

    /// <summary>Number of headwords loaded.</summary>
    public int Count => _entries.Count;

    /// <summary>Parse the CMUdict text format: one entry per line, "word  P1 P2 P3".</summary>
    /// <remarks>
    /// Alternate pronunciations are written "word(2)" and are SKIPPED here rather than silently preferred
    /// or merged. Choosing between them needs sentence context - "read" and "record" are different words
    /// depending on their part of speech - and that is a separate problem which pretending to solve here
    /// would only hide.
    /// </remarks>
    public static PronunciationDictionary Parse(IEnumerable<string> lines)
    {
        var entries = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in lines)
        {
            var body = line;
            int comment = body.IndexOf('#');
            if (comment >= 0) body = body[..comment];
            body = body.Trim();
            if (body.Length == 0) continue;

            var parts = body.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            if (parts[0].EndsWith(')')) continue;                 // word(2), word(3), ...
            entries.TryAdd(parts[0], parts[1..]);
        }
        return new PronunciationDictionary(entries);
    }

    /// <summary>Load from a CMUdict file on disk.</summary>
    public static PronunciationDictionary Load(string path) => Parse(File.ReadLines(path));

    /// <summary>Look up a word's ARPAbet phones. Case-insensitive.</summary>
    public bool TryLookup(string word, out string[] phones) => _entries.TryGetValue(word, out phones!);
}
