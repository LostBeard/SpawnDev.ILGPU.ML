namespace SpawnDev.Phonemizer;

/// <summary>
/// Turns English text into the IPA phoneme symbols a neural TTS model expects.
/// </summary>
/// <remarks>
/// <para>
/// MIT-licensed, dependency-free, and browser-capable, which is the entire point: ZipVoice, Piper and
/// Kokoro all phonemize through espeak-ng, and that GPL-3 dependency is why no permissively licensed
/// English TTS frontend exists for .NET.
/// </para>
/// <para>
/// The rules below are ordered by MEASURED importance, not by intuition. Over 432 renders through
/// ZipVoice, stress errors cost more than mispronouncing a word outright, while every purely segmental
/// difference cost nothing detectable. So this spends its effort on stress and gives the fine phonetic
/// detail only what is cheap. The numbers are in Plans/mit-phonemizer-2026-08-27.md.
/// </para>
/// </remarks>
public sealed class EnglishPhonemizer
{
    private readonly PronunciationDictionary _dictionary;

    /// <summary>Create a phonemizer over a pronunciation dictionary.</summary>
    public EnglishPhonemizer(PronunciationDictionary dictionary)
        => _dictionary = dictionary ?? throw new ArgumentNullException(nameof(dictionary));

    /// <summary>
    /// Apply the alveolar tap that American English uses for T and D between vowels.
    /// </summary>
    /// <remarks>
    /// Measured as costing nothing either way - the model is indifferent to it - so this is on by default
    /// only because it matches the reference frontend more closely, and can be turned off freely.
    /// </remarks>
    public bool Flapping { get; set; } = true;

    /// <summary>
    /// Strip stress from words that carry it only in their citation form.
    /// </summary>
    /// <remarks>
    /// Leave this on. Turning it off reproduces the single worst realistic frontend defect: 18.2% word
    /// error, worse than mispronouncing a whole word. It exists as a switch only so the measurement that
    /// established that can be repeated.
    /// </remarks>
    public bool DestressFunctionWords { get; set; } = true;

    /// <summary>
    /// Turns written forms into spoken ones before lookup - numbers, money, abbreviations.
    /// </summary>
    /// <remarks>
    /// Set to null to phonemize text that has already been normalized. Leaving it on is almost always
    /// right: a dictionary cannot look up "1999", and what the reader hears if you skip this is either
    /// nothing at all or a string of digits read one by one.
    /// </remarks>
    public EnglishTextNormalizer? Normalizer { get; set; } = new();

    /// <summary>
    /// Read a stress-shifting homograph from its context: "the record" against "to record".
    /// </summary>
    /// <remarks>
    /// About 1% of words in ordinary text, but every miss is a STRESS error, and stress on the wrong
    /// syllable was measured at 34.3% word error against roughly nothing for any segmental slip.
    /// </remarks>
    public bool ResolveHomographs { get; set; } = true;

    /// <summary>
    /// Build an unknown word out of a known stem and its ending, before resorting to guesswork.
    /// </summary>
    /// <remarks>
    /// On by default and rarely worth turning off: it is right whenever it fires, where letter-to-sound
    /// is right about 44% of the time.
    /// </remarks>
    public bool Decompose { get; set; } = true;

    /// <summary>
    /// Pronounces words the dictionary does not contain, from their spelling.
    /// </summary>
    /// <remarks>
    /// Optional, and the phonemizer is honest either way: without it an unknown word is reported through
    /// <see cref="LastUnknownWords"/> and SKIPPED rather than guessed. With it, the word gets spoken -
    /// which for a name is the whole point, since a voice that silently drops "Aubriella" is not finished.
    /// </remarks>
    public LetterToSound? LetterToSound { get; set; }

    /// <summary>Words the dictionary did not contain, in encounter order, from the last call.</summary>
    /// <remarks>
    /// Surfaced rather than swallowed: an unknown word is the one failure a caller genuinely needs to
    /// know about, since it is the difference between speaking a name and skipping it.
    /// </remarks>
    public IReadOnlyList<string> LastUnknownWords => _unknown;
    private readonly List<string> _unknown = new();

    /// <summary>Phonemize a stretch of text into IPA symbols, one entry per symbol.</summary>
    public IReadOnlyList<string> ToSymbols(string text)
    {
        _unknown.Clear();
        var output = new List<string>();
        if (string.IsNullOrWhiteSpace(text)) return output;

        if (Normalizer != null) text = Normalizer.Normalize(text);

        string? previousWord = null;
        foreach (var token in Tokenize(text))
        {
            if (token.IsWord)
            {
                // A space separates words, but NOT a word from the punctuation that just preceded it:
                // the reference frontend writes "roses ,understand", with the pause mark leading the
                // clause it opens rather than trailing the one it closes.
                if (output.Count > 0 && output[^1] != " " && !IsPunctuation(output[^1])) output.Add(" ");
                if (_dictionary.TryLookup(token.Text, out var phones))
                {
                    output.AddRange(Word(phones, token.Text, previousWord));
                    previousWord = token.Text;
                }
                else
                {
                    // Recorded even when it can be sounded out, because "the dictionary did not have this"
                    // is worth knowing: it is the difference between a pronunciation and a guess.
                    _unknown.Add(token.Text);

                    // Decompose BEFORE guessing. "unfriendliest" is not in the dictionary but "friend"
                    // is, and letter-to-sound is right less than half the time - so anything that can be
                    // derived from a known word should never be guessed at.
                    if (Decompose && WordDecomposer.TryDecompose(token.Text, _dictionary, out var derived))
                    {
                        output.AddRange(Word(derived, token.Text, previousWord));
                    }
                    else
                    {
                        var guessed = LetterToSound?.Predict(token.Text);
                        if (guessed is { Length: > 0 }) output.AddRange(Word(guessed, token.Text, previousWord));
                    }
                    previousWord = token.Text;
                }
            }
            else
            {
                // Punctuation carries prosody - a comma is a pause the model was trained on - and it is
                // written with a space BEFORE it, attached to what follows.
                if (output.Count > 0 && output[^1] != " ") output.Add(" ");
                output.Add(token.Text);
                previousWord = null;      // a pause ends the phrase whose cue we were carrying
            }
        }

        while (output.Count > 0 && output[^1] == " ") output.RemoveAt(output.Count - 1);
        return output;
    }

    /// <summary>Phonemize into a single IPA string.</summary>
    public string ToIpa(string text) => string.Concat(ToSymbols(text));

    /// <summary>Map one word's ARPAbet phones to IPA symbols, applying the rules.</summary>
    /// <param name="previousWord">
    /// The word before this one, or null at the start of a phrase. Used only to read a homograph:
    /// "the record" is a noun, "to record" is a verb, and they are stressed on different syllables.
    /// </param>
    private List<string> Word(string[] phones, string spelling, string? previousWord = null)
    {
        bool destress = DestressFunctionWords && FunctionWords.IsUnstressed(spelling);
        var symbols = new List<string>(phones.Length + 4);
        int suffixVowel = SuffixVowelIndex(phones, spelling);

        // A stress-shifting homograph picks a DIFFERENT dictionary entry, chosen from context, before
        // mapping - so every later rule sees the reading the sentence actually calls for.
        if (ResolveHomographs && !destress)
            phones = Homographs.Choose(spelling, previousWord, _dictionary, phones);

        for (int index = 0; index < phones.Length; index++)
        {
            var raw = phones[index];
            var (phone, stress) = Arpabet.SplitStress(raw);
            if (index == suffixVowel)
            {
                // The reduced vowel of a plural or past-tense ending: roses, waited, boxes, collected.
                // The reference frontend writes a distinct centralised vowel here rather than a plain
                // small-capital I. Measured as costing nothing, so this is about matching the training
                // distribution rather than about intelligibility.
                symbols.Add("ᵻ");
                continue;
            }
            if (stress >= 0)
            {
                // The mark goes immediately before the vowel, not before the syllable onset: "better" is
                // b-STRESS-E, not STRESS-b-E. Verified against captured reference output.
                if (!destress)
                {
                    if (stress == 1) symbols.Add(Arpabet.PrimaryStress);
                    else if (stress == 2) symbols.Add(Arpabet.SecondaryStress);
                }
                // Destressing removes the MARK, not the vowel. Reducing the quality as well turned "of"
                // into "uhv" and "and" into "uhnd", where the reference keeps the fuller vowel and simply
                // does not mark it - which was 17% of all remaining differences over 120 sentences.
                symbols.AddRange(ContextualVowel(phone, stress, phones, index));
            }
            else if (Arpabet.TryConsonant(phone, out var consonant))
            {
                symbols.AddRange(consonant);
            }
        }

        // One PRIMARY stress per word. The dictionary marks both syllables of "nineteen" and "seventeen"
        // as primary; a speaker leans on the first and gives the second a lighter beat, which is what the
        // reference frontend writes (seventeen is SECONDARY-less first, then a secondary mark on -teen).
        // Two primaries in one word is not English, and stress is what the model punishes hardest.
        bool seenPrimary = false;
        for (int i = 0; i < symbols.Count; i++)
        {
            if (symbols[i] != Arpabet.PrimaryStress) continue;
            if (!seenPrimary) { seenPrimary = true; continue; }
            symbols[i] = Arpabet.SecondaryStress;
        }

        // A word that is not a weak form must carry a stress somewhere. The dictionary stores some
        // monosyllables with no stressed vowel at all - "in" is IH0 N - and emitting them unstressed
        // leaves the sentence with a hole where a beat should be. Stress is the axis this model is most
        // sensitive to, so the safe default is to give such a word the stress it would carry when spoken.
        if (!destress && symbols.Count > 0 && !symbols.Any(IsStressMark))
        {
            int firstVowel = symbols.FindIndex(Arpabet.Vowels.Contains);
            if (firstVowel >= 0) symbols.Insert(firstVowel, Arpabet.PrimaryStress);
        }

        if (Flapping) Flap(symbols);
        return symbols;
    }

    /// <summary>
    /// Pick a vowel's realisation from what surrounds it, where American English demands it.
    /// </summary>
    /// <remarks>
    /// Two contexts, both ordinary English phonology rather than quirks of one implementation:
    /// <list type="bullet">
    /// <item>The LOT-CLOTH split. Before a voiceless fricative the back vowel is short and rounded:
    /// "cost" is kost and "cloth" is kloth, not "caast" and "clawth". The dictionary writes the two
    /// classes inconsistently - "cost" is AA while "cloth" is AO - and both come out the same way.</item>
    /// </list>
    /// TRIED AND REVERTED: closing a stressed AO to o before R. It is right for "boards" and "port" and
    /// wrong for "quart", "or" and "for", which keep the opener vowel under stress too - twenty new
    /// differences against four fixed. The rule that looks clean is not always the rule that measures.
    /// </remarks>
    private static string[] ContextualVowel(string phone, int stress, string[] phones, int index)
    {
        if (phone is not ("AA" or "AO")) return Arpabet.Vowel(phone, stress);

        var next = index + 1 < phones.Length ? Arpabet.SplitStress(phones[index + 1]).Phone : "";
        if (next is "S" or "TH" or "F" or "SH") return ["ɔ"];                  // cost, cloth, off
        return Arpabet.Vowel(phone, stress);
    }

    /// <summary>
    /// Index of the vowel belonging to a plural or past-tense ending, or -1 when there is none.
    /// </summary>
    /// <remarks>
    /// This has to be MORPHOLOGICAL, not just phonetic. "roses", "waited" and "boxes" all end in an
    /// unstressed vowel followed by Z or D and all take the reduced vowel - but so does "hundred", which
    /// does not, because there is no word "hundr" for it to be a past tense of. The stem is therefore
    /// looked up in the dictionary: no stem, no suffix, no rule.
    /// </remarks>
    private int SuffixVowelIndex(string[] phones, string spelling)
    {
        if (phones.Length < 3) return -1;

        var last = Arpabet.SplitStress(phones[^1]).Phone;
        if (last != "Z" && last != "D") return -1;

        var (vowel, stress) = Arpabet.SplitStress(phones[^2]);
        if (stress != 0 || (vowel != "IH" && vowel != "AH")) return -1;

        var word = spelling.ToLowerInvariant();
        if (word.Length < 4) return -1;
        var final = word[^1];
        if (final != 's' && final != 'd') return -1;

        // "roses" -> "rose", "boxes" -> "box", "waited" -> "wait", "collected" -> "collect".
        bool isSuffix = _dictionary.TryLookup(word[..^1], out _)
                     || (word.Length > 4 && _dictionary.TryLookup(word[..^2], out _));
        return isSuffix ? phones.Length - 2 : -1;
    }

    /// <summary>
    /// Turn a T into a tap when it sits between vowels and the following vowel is unstressed.
    /// </summary>
    /// <remarks>
    /// This is what makes "water" and "better" sound American rather than clipped. Applied WITHIN a word
    /// only: across a word boundary it depends on how tightly the speaker runs the words together, which
    /// a dictionary cannot know.
    /// </remarks>
    private static void Flap(List<string> symbols)
    {
        for (int i = 1; i < symbols.Count - 1; i++)
        {
            // T only. The reference frontend taps T but leaves D alone - "ladder", "middle" and
            // "garden" all keep a plain d - and flapping D too was measurably wrong here.
            if (symbols[i] != "t") continue;

            // What precedes must be a vowel, possibly through its length mark or an r.
            var previous = symbols[i - 1];
            if (!Arpabet.Vowels.Contains(previous) && previous != Arpabet.Length && previous != "ɹ") continue;

            // What follows must be a vowel, and it must be unstressed - a tap never carries stress.
            int j = i + 1;
            if (j >= symbols.Count) continue;
            if (symbols[j] == Arpabet.PrimaryStress || symbols[j] == Arpabet.SecondaryStress) continue;
            if (!Arpabet.Vowels.Contains(symbols[j])) continue;

            symbols[i] = Arpabet.Flap;
        }
    }

    private static bool IsStressMark(string symbol)
        => symbol == Arpabet.PrimaryStress || symbol == Arpabet.SecondaryStress;

    private static bool IsPunctuation(string symbol) => symbol.Length == 1 && ".,;:!?".IndexOf(symbol[0]) >= 0;

    private readonly record struct Token(string Text, bool IsWord);

    /// <summary>Split text into words and the punctuation between them.</summary>
    private static IEnumerable<Token> Tokenize(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            if (char.IsLetter(c) || c == '\'')
            {
                int start = i;
                while (i < text.Length && (char.IsLetter(text[i]) || text[i] == '\'')) i++;
                yield return new Token(text[start..i], true);
            }
            else if (".,;:!?".IndexOf(c) >= 0)
            {
                yield return new Token(c.ToString(), false);
                i++;
            }
            else
            {
                i++;   // whitespace and anything else the normalizer has not turned into words yet
            }
        }
    }
}
