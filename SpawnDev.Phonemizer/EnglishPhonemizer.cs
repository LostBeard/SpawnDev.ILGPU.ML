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

        foreach (var token in Tokenize(text))
        {
            if (token.IsWord)
            {
                // A space separates words, but NOT a word from the punctuation that just preceded it:
                // the reference frontend writes "roses ,understand", with the pause mark leading the
                // clause it opens rather than trailing the one it closes.
                if (output.Count > 0 && output[^1] != " " && !IsPunctuation(output[^1])) output.Add(" ");
                if (_dictionary.TryLookup(token.Text, out var phones))
                    output.AddRange(Word(phones, token.Text));
                else
                    _unknown.Add(token.Text);
            }
            else
            {
                // Punctuation carries prosody - a comma is a pause the model was trained on - and it is
                // written with a space BEFORE it, attached to what follows.
                if (output.Count > 0 && output[^1] != " ") output.Add(" ");
                output.Add(token.Text);
            }
        }

        while (output.Count > 0 && output[^1] == " ") output.RemoveAt(output.Count - 1);
        return output;
    }

    /// <summary>Phonemize into a single IPA string.</summary>
    public string ToIpa(string text) => string.Concat(ToSymbols(text));

    /// <summary>Map one word's ARPAbet phones to IPA symbols, applying the rules.</summary>
    private List<string> Word(string[] phones, string spelling)
    {
        bool destress = DestressFunctionWords && FunctionWords.IsUnstressed(spelling);
        var symbols = new List<string>(phones.Length + 4);

        foreach (var raw in phones)
        {
            var (phone, stress) = Arpabet.SplitStress(raw);
            if (stress >= 0)
            {
                // The mark goes immediately before the vowel, not before the syllable onset: "better" is
                // b-STRESS-E, not STRESS-b-E. Verified against captured reference output.
                if (!destress)
                {
                    if (stress == 1) symbols.Add(Arpabet.PrimaryStress);
                    else if (stress == 2) symbols.Add(Arpabet.SecondaryStress);
                }
                symbols.AddRange(Arpabet.Vowel(phone, destress ? 0 : stress));
            }
            else if (Arpabet.TryConsonant(phone, out var consonant))
            {
                symbols.AddRange(consonant);
            }
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
