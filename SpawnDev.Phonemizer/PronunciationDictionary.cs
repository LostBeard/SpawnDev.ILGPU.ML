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

    /// <summary>
    /// Teaches the dictionary how one word is said, overriding whatever it had.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gap this fills: a name you KNOW should never be guessed at. Letter-to-sound is right about
    /// half the time on words outside the dictionary, and the words a given application says most often
    /// are exactly the ones CMUdict lacks - character names, brands, jargon, and people. "aubriella" is
    /// the standing example in this library's own notes: it is absent from CMUdict, every "-ella" name
    /// that IS present stresses the "ell", and the guesser stresses the "au" instead. No rule fixes that,
    /// and it does not need one - somebody knows how the word is said.
    /// </para>
    /// <para>
    /// This REPLACES every pronunciation held for the word rather than adding another, which is what
    /// makes it authoritative: <see cref="Homographs"/> only chooses between alternates when two or more
    /// exist, so leaving the originals in place would let context resolution quietly pick one of them
    /// instead. Definitions also sit ahead of decomposition and letter-to-sound in
    /// <see cref="EnglishPhonemizer.ToSymbols"/>, so a defined word is never guessed.
    /// </para>
    /// <para>
    /// ⚠️ Define words at setup, before phonemizing. The dictionary is a plain map with no locking, so
    /// mutating it while another thread is looking words up is not safe.
    /// </para>
    /// </remarks>
    /// <param name="word">The word, matched case-insensitively.</param>
    /// <param name="arpabet">
    /// ARPAbet phones separated by spaces, exactly as CMUdict writes them - vowels carry a stress digit,
    /// consonants do not: <c>"AO2 B R IY0 EH1 L AH0"</c>.
    /// </param>
    /// <exception cref="ArgumentException">The word is blank, or a phone is not ARPAbet.</exception>
    public void Define(string word, string arpabet)
        => Define(word, (arpabet ?? "").Split(' ', StringSplitOptions.RemoveEmptyEntries));

    /// <summary>Teaches the dictionary how one word is said, from phones already split.</summary>
    /// <inheritdoc cref="Define(string, string)"/>
    public void Define(string word, IReadOnlyList<string> phones)
    {
        if (string.IsNullOrWhiteSpace(word)) throw new ArgumentException("A word is required.", nameof(word));
        if (phones is null || phones.Count == 0)
            throw new ArgumentException($"No phones given for '{word}'.", nameof(phones));

        // Validated rather than trusted. An unrecognised phone would otherwise travel silently into the
        // symbol stream and come out as a missing or wrong sound, which is far harder to trace back to a
        // typo here than an exception naming it.
        foreach (var phone in phones)
        {
            var (bare, stress) = Arpabet.SplitStress(phone);

            // Asked of the same tables that do the conversion, so a phone that validates here
            // cannot fail to map later. (Arpabet.Vowels is the IPA-side set, not this one.)
            if (Arpabet.TryConsonant(bare, out _)) continue;

            var isVowel = Arpabet.Vowel(bare, stress < 0 ? 1 : stress).Length > 0;
            if (isVowel && stress >= 0) continue;

            throw new ArgumentException(
                isVowel
                    ? $"vowel '{phone}' needs a stress digit (in the pronunciation of '{word}'): "
                    + $"{bare}0 unstressed, {bare}1 primary, {bare}2 secondary. Which syllable carries "
                    + "the stress is usually the whole reason for defining a word."
                    : $"'{phone}' is not an ARPAbet phone (in the pronunciation of '{word}'). "
                    + "Vowels carry a stress digit (AA0 AA1 AA2), consonants do not (B, CH, ZH).",
                nameof(phones));
        }

        _entries[word] = [[.. phones]];
    }

    /// <summary>Forgets a word entirely, so it falls back to being sounded out.</summary>
    /// <returns>True if the word had been held.</returns>
    public bool Remove(string word) => _entries.Remove(word);
}
