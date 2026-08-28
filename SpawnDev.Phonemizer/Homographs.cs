namespace SpawnDev.Phonemizer;

/// <summary>
/// Words spelled alike whose STRESS moves with their part of speech.
/// </summary>
/// <remarks>
/// <para>
/// "I want to RE-cord a REC-ord." Both are "record", and a dictionary can only offer one answer. English
/// has a productive pattern here: a two-syllable noun takes stress on the first syllable, the matching
/// verb on the second.
/// </para>
/// <para>
/// This is a small effect - about 1% of words in ordinary text - but each instance is a STRESS error, and
/// stress is the class the downstream model punishes hardest (34.3% word error when it lands on the wrong
/// syllable, against ~0% for any segmental slip). So it is worth the few rules it takes.
/// </para>
/// <para>
/// The part of speech is guessed from the PREVIOUS word only. That is deliberately shallow: a real tagger
/// is a model in its own right, and the cheap cue - "the" before it means noun, "to" before it means verb
/// - covers the common cases without pretending to more than it knows. When there is no cue, the noun
/// reading wins, because these words appear as nouns more often.
/// </para>
/// </remarks>
public static class Homographs
{
    /// <summary>Two-syllable words whose noun and verb readings differ only in which syllable is stressed.</summary>
    public static readonly IReadOnlySet<string> StressShifting = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "address", "combine", "compound", "compress", "conduct", "conflict", "conscript", "console",
        "consort", "content", "contest", "contract", "contrast", "converse", "convert", "convict",
        "decrease", "defect", "desert", "detail", "dictate", "digest", "discard", "discharge",
        "discount", "escort", "essay", "excise", "exploit", "export", "extract", "ferment", "impact",
        "implant", "import", "impress", "imprint", "incline", "increase", "indent", "insert", "insult",
        "intern", "invalid", "invite", "object", "perfect", "permit", "pervert", "present", "produce",
        "progress", "project", "protest", "rebel", "recall", "record", "refill", "refund", "refuse",
        "reject", "relay", "remake", "reprint", "rerun", "research", "reset", "subject", "survey",
        "suspect", "transfer", "transplant", "transport", "upgrade", "upset",
    };

    // A determiner, possessive or preposition before the word makes it a noun: "the record", "his conduct".
    private static readonly IReadOnlySet<string> NounCues = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our",
        "their", "some", "any", "no", "every", "each", "another", "of", "in", "on", "for", "with",
        "about", "from", "by", "at", "into", "onto", "one", "two", "three", "first", "last", "new",
        "old", "good", "bad", "great", "such",
    };

    // "to", a modal, or a subject pronoun before the word makes it a verb: "to record", "they object".
    private static readonly IReadOnlySet<string> VerbCues = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "to", "will", "would", "can", "could", "shall", "should", "must", "may", "might", "i", "we",
        "you", "they", "he", "she", "who", "please", "let", "lets", "dont", "not", "and", "or", "then",
        "also", "helps", "help", "helped", "cannot",
    };

    /// <summary>
    /// Homographs whose readings differ by a VOWEL rather than by stress, with the phone that tells
    /// them apart: the default reading first, then the one a verb cue selects.
    /// </summary>
    /// <remarks>
    /// These are the classic English traps - "the WIND blows" against "WIND the clock" - and the
    /// dictionary lists both with no way to choose. Worse, its FIRST entry is often the rarer reading:
    /// "wind" is stored as the verb, so untouched, "no doubt about the way the wind blows" came out as
    /// "the way the WINED blows" and the transcriber heard exactly that.
    ///
    /// Where the two readings are separated by meaning rather than by part of speech - "bass" the fish
    /// against "bass" the register, "tear" the eye against "tear" the rip - no cue this shallow can
    /// help, so only the DEFAULT is set and the verb column is left null. Guessing there would be worse
    /// than the ambiguity.
    /// </remarks>
    private static readonly Dictionary<string, (string Default, string? Verb)> VowelReadings =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["wind"] = ("IH", "AY"),      // the wind blows / wind the clock
            ["live"] = ("IH", "IH"),      // to live, live wires are rarer in ordinary text
            ["close"] = ("S", "Z"),       // close by / to close it
            ["use"] = ("S", "Z"),         // the use of it / to use it
            ["lead"] = ("IY", "IY"),      // to lead; the metal is the rarer reading
            ["read"] = ("IY", "IY"),      // present tense reads more often than past
            ["minute"] = ("IH", null),    // the time, overwhelmingly
            ["bass"] = ("EY", null),      // the register, in ordinary text
            ["wound"] = ("UW", null),     // the injury
        };

    /// <summary>
    /// Pick the pronunciation the sentence calls for, or return <paramref name="fallback"/> unchanged.
    /// </summary>
    /// <remarks>
    /// It CHOOSES between the dictionary's own entries rather than moving a stress mark. Those are not
    /// the same thing: the noun "record" is REK-erd and the verb is ri-KORD - different vowels, not just
    /// a different beat. Re-stressing the verb entry produced "RUH-kord", a word nobody says.
    /// </remarks>
    public static string[] Choose(string word, string? previousWord,
                                  PronunciationDictionary dictionary, string[] fallback)
    {
        if (!dictionary.TryLookupAll(word, out var all) || all.Count < 2) return fallback;

        // Vowel homographs first: they change the WORD, not just which syllable carries the beat.
        if (VowelReadings.TryGetValue(word, out var reading))
        {
            bool isVerb = previousWord != null && VerbCues.Contains(previousWord);
            var wantedPhone = isVerb ? reading.Verb ?? reading.Default : reading.Default;
            foreach (var candidate in all)
                if (candidate.Any(phone => Bare(phone) == wantedPhone)) return candidate;
            return fallback;
        }

        int wanted = StressedSyllable(word, previousWord);
        if (wanted < 0) return fallback;

        foreach (var candidate in all)
            if (PrimarySyllable(candidate) == wanted) return candidate;
        return fallback;
    }

    private static string Bare(string phone)
        => phone.Length > 0 && char.IsDigit(phone[^1]) ? phone[..^1] : phone;

    /// <summary>Which vowel carries the primary stress, counting from the start, or -1 if none does.</summary>
    private static int PrimarySyllable(string[] phones)
    {
        int syllable = 0;
        foreach (var phone in phones)
        {
            if (phone.Length == 0 || !char.IsDigit(phone[^1])) continue;
            if (phone[^1] == '1') return syllable;
            syllable++;
        }
        return -1;
    }

    /// <summary>Which syllable this word's stress belongs on, given the word before it.</summary>
    /// <returns>
    /// 0 for the first syllable (the noun reading), 1 for the second (the verb reading), or -1 when this
    /// is not a stress-shifting word and the dictionary should simply be believed.
    /// </returns>
    public static int StressedSyllable(string word, string? previousWord)
    {
        if (!StressShifting.Contains(word)) return -1;
        if (previousWord != null && VerbCues.Contains(previousWord)) return 1;
        if (previousWord != null && NounCues.Contains(previousWord)) return 0;
        return 0;   // no cue: the noun reading is the commoner one for this list
    }
}
