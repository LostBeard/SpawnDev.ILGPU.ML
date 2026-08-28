namespace SpawnDev.Phonemizer;

/// <summary>
/// A handful of very common words where the dictionary and the trained models simply disagree.
/// </summary>
/// <remarks>
/// <para>
/// CMUdict and the frontend these TTS models were trained on are two different descriptions of American
/// English, and on a few high-frequency words they differ outright - not by a rule that could be derived,
/// just lexically. "on" is stored with the open-a vowel and spoken with the open-o; "was" is stored with
/// the open-a and spoken with the wedge.
/// </para>
/// <para>
/// These are worth overriding precisely BECAUSE they are frequent: "on", "was", "and" and "a" appear in
/// almost every sentence, so a handful of entries closes a disproportionate share of the remaining
/// difference. Each was READ OFF captured reference output, not reasoned about.
/// </para>
/// <para>
/// The list is deliberately tiny and stays that way. It is a lexicon of exceptions, and a growing one
/// would mean a rule is missing somewhere - the right response to that is to find the rule.
/// </para>
/// </remarks>
public static class ReferenceOverrides
{
    /// <summary>Words whose IPA is taken verbatim rather than derived, with the reference's own symbols.</summary>
    public static readonly IReadOnlyDictionary<string, string[]> Words =
        new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            // Reference: "ˈɔn" - the open-o, no length mark. CMUdict stores AA1, giving "ɑː".
            ["on"] = ["ˈ", "ɔ", "n"],
            ["onto"] = ["ˈ", "ɔ", "n", "t", "uː"],

            // Reference: "wʌz" - unstressed, with the wedge. CMUdict stores AA1, giving "wˈɑːz".
            ["was"] = ["w", "ʌ", "z"],

            // Reference: "ænd" - unstressed but with the full vowel. CMUdict stores AH0, giving "ənd".
            ["and"] = ["æ", "n", "d"],

            // Reference: a bare "a" for the article, which is its own symbol in these models.
            ["a"] = ["a"],
        };

    /// <summary>
    /// One difference we deliberately do NOT copy.
    /// </summary>
    /// <remarks>
    /// The reference frontend reads a sentence-initial capital "A" as the LETTER NAME - "AY cat sat"
    /// rather than "uh cat sat". That is 16 of the remaining symbol differences and it would be trivial
    /// to match, but matching it would make the speech worse for a listener, and a listener is the
    /// target. Agreement with the reference is the proxy; sounding right is the goal, and where the two
    /// point in different directions the goal wins.
    /// </remarks>
    public const string DeliberateDivergence = "sentence-initial capital A is read as an article, not a letter";

    /// <summary>Look up an override, if this word has one.</summary>
    public static bool TryGet(string word, out string[] symbols) => Words.TryGetValue(word, out symbols!);
}
