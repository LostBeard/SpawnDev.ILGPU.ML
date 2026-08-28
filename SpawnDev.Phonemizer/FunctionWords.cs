namespace SpawnDev.Phonemizer;

/// <summary>
/// The words English speakers do NOT stress in running speech.
/// </summary>
/// <remarks>
/// <para>
/// This list is the single most important rule in the phonemizer, and it exists because of a measurement.
/// A pronunciation dictionary stores CITATION forms - each word as it would be said alone - so "for",
/// "at" and "in" each arrive carrying a primary stress. Nobody speaks that way: in a sentence those words
/// lean on their neighbours and lose their stress entirely.
/// </para>
/// <para>
/// Getting this wrong is not cosmetic. Measured against ZipVoice over 432 renders, adding stress to
/// function words cost 18.2% word error - MORE than deliberately mispronouncing a whole word, which cost
/// 13.5%. It was also 19.4% of every real difference between the dictionary and the reference frontend.
/// Full method and numbers in Plans/mit-phonemizer-2026-08-27.md.
/// </para>
/// <para>
/// Words are listed only where the unstressed form is the ordinary one. Anything that is normally
/// stressed, or that flips between a stressed and unstressed reading by meaning, is left out - "can" as a
/// container against "can" as an auxiliary is a homograph problem, not a destressing one, and pretending
/// otherwise here would hide it.
/// </para>
/// </remarks>
public static class FunctionWords
{
    /// <summary>Words that lose their stress inside a phrase.</summary>
    public static readonly IReadOnlySet<string> Unstressed = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        // Articles and determiners
        "a", "an", "the",

        // The monosyllabic prepositions and conjunctions that have true WEAK FORMS - a reduced vowel in
        // running speech. Deliberately NOT here: in, on, up, out, off, and every polysyllabic preposition
        // (about, after, over, under, between). Those keep their stress, and stripping it is the mirror
        // image of the defect this list exists to prevent. Also absent, on evidence rather than theory:
        // "that", "my" and "would" are marked STRESSED by the reference frontend in ordinary sentences,
        // so destressing them would be inventing a defect.
        "and", "as", "at", "for", "from", "nor", "of", "or", "to", "with",

        // Pronouns and possessives
        "he", "her", "hers", "him", "his", "it", "its", "me", "our", "ours", "she",
        "their", "theirs", "them", "they", "us", "we", "you", "your", "yours",

        // Auxiliaries and copulas. These reduce as helpers ("he WAS going") but carry stress as main
        // verbs and in short answers ("yes he WAS") - a distinction that needs sentence context, which
        // belongs with homograph resolution rather than in a flat list.
        "am", "are", "be", "been", "can", "did", "do", "does", "had", "has",
        "have", "is", "must", "shall", "should", "was", "were", "will",
    };

    /// <summary>
    /// Weak words that keep a LIGHT beat rather than losing stress entirely.
    /// </summary>
    /// <remarks>
    /// Evidence, not theory: the reference frontend writes "some" as sˌʌm, "but" as bˌʌt and "such" as
    /// sˌʌtʃ - a secondary mark, not nothing. Stripping it outright cost real intelligibility; in one
    /// end-to-end render "some flower seeds" came back transcribed as "and flower seeds".
    /// "could" is absent from BOTH lists because the reference gives it full primary stress.
    /// </remarks>
    public static readonly IReadOnlySet<string> Secondary = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "some", "but", "such", "than", "that",
    };

    /// <summary>True when this word keeps a light beat instead of losing its stress.</summary>
    public static bool TakesSecondaryStress(string word) => Secondary.Contains(Letters(word));

    /// <summary>True when this word is normally spoken without stress.</summary>
    /// <remarks>
    /// Apostrophes are stripped before lookup so "don't" and "dont" both match, and a possessive or
    /// contracted form written either way behaves the same.
    /// </remarks>
    public static bool IsUnstressed(string word) => Unstressed.Contains(Letters(word));

    private static string Letters(string word)
    {
        if (string.IsNullOrEmpty(word)) return "";
        Span<char> buffer = word.Length <= 32 ? stackalloc char[word.Length] : new char[word.Length];
        int n = 0;
        foreach (var c in word)
            if (char.IsLetter(c)) buffer[n++] = char.ToLowerInvariant(c);
        return new string(buffer[..n]);
    }
}
