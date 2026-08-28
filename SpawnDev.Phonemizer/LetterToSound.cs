using System.Text;

namespace SpawnDev.Phonemizer;

/// <summary>
/// Pronounces words no dictionary contains, from their spelling.
/// </summary>
/// <remarks>
/// <para>
/// This is what lets the voice say a name. CMUdict covered every word in 120 test sentences and is full
/// of surnames - but it does not contain "Aubriella", and a text-to-speech that silently skips a child's
/// name is not finished.
/// </para>
/// <para>
/// The rules are LEARNED from CMUdict rather than hand-written, and then measured on words held out of
/// training - see <c>tools/lts-train</c>, which reports that number in the model file's own header. A
/// hand-written ruleset encodes one person's recollection of English spelling and can never honestly
/// claim an accuracy figure.
/// </para>
/// <para>
/// Lookup walks left to right, taking the widest context that matches: two letters either side, then one,
/// then the letter alone. That backoff is why "gh" can be silent in "night" and voiced in "ghost" without
/// anyone writing a rule for either.
/// </para>
/// </remarks>
public sealed class LetterToSound
{
    private readonly Dictionary<string, string> _sounds;
    private readonly Dictionary<string, string> _stress;

    private LetterToSound(Dictionary<string, string> sounds, Dictionary<string, string> stress)
    {
        _sounds = sounds;
        _stress = stress;
    }

    /// <summary>Number of context rules loaded, sound and stress together.</summary>
    public int Count => _sounds.Count + _stress.Count;

    /// <summary>Parse a model produced by <c>tools/lts-train</c>.</summary>
    public static LetterToSound Parse(IEnumerable<string> lines)
    {
        var sounds = new Dictionary<string, string>(StringComparer.Ordinal);
        var stress = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var line in lines)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            var emission = line[(tab + 1)..].Trim();

            // Stress rules are prefixed with * so a single file carries both models.
            if (line[0] == '*') stress[line[1..tab]] = emission;
            else sounds[line[..tab]] = emission == "-" ? "" : emission;
        }
        return new LetterToSound(sounds, stress);
    }

    /// <summary>Load a model from disk.</summary>
    public static LetterToSound Load(string path) => Parse(File.ReadLines(path));

    /// <summary>Predict ARPAbet phones, stress digits included, for a word's spelling.</summary>
    /// <remarks>
    /// Stress is predicted rather than added afterwards, because it is the single thing the downstream
    /// model is most sensitive to and because the training data carries it. A word that comes back with
    /// no stressed vowel at all is given one by <see cref="EnglishPhonemizer"/>, the same as a dictionary
    /// word that arrives unstressed.
    /// </remarks>
    public string[] Predict(string word)
    {
        if (string.IsNullOrEmpty(word)) return [];
        var letters = word.ToLowerInvariant();
        var output = new List<string>(letters.Length + 2);

        for (int i = 0; i < letters.Length; i++)
        {
            if (letters[i] is < 'a' or > 'z') continue;
            var emission = Lookup(_sounds, letters, i, 2) ?? Lookup(_sounds, letters, i, 1)
                        ?? Lookup(_sounds, letters, i, 0);
            if (string.IsNullOrEmpty(emission)) continue;

            // Stress comes from its own model over the same contexts. Every vowel MUST come back carrying
            // a digit: a vowel without one is not valid ARPAbet, and downstream it would be discarded as
            // an unknown phone - which is how an entire name once came out as its consonants alone.
            var phones = emission.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var digits = Lookup(_stress, letters, i, 2) ?? Lookup(_stress, letters, i, 1)
                      ?? Lookup(_stress, letters, i, 0) ?? "";
            for (int k = 0; k < phones.Length; k++)
                output.Add(IsVowel(phones[k])
                    ? phones[k] + (k < digits.Length && char.IsDigit(digits[k]) ? digits[k] : '0')
                    : phones[k]);
        }

        // Exactly ONE primary stress per word. The per-letter model can mark several, which is not
        // English - "Tuvok" came back with two - and stress is what the downstream model punishes
        // hardest. The first is kept, because English favours earlier stress.
        bool seenPrimary = false;
        for (int i = 0; i < output.Count; i++)
        {
            if (!output[i].EndsWith('1')) continue;
            if (!seenPrimary) { seenPrimary = true; continue; }
            output[i] = output[i][..^1] + "0";
        }

        // Every English word has a stressed syllable. If the model marked none, stress the first, which is
        // where English puts it more often than anywhere else.
        if (output.Any(IsVowel) && !output.Any(x => x.EndsWith('1')))
        {
            int first = output.FindIndex(IsVowel);
            output[first] = output[first][..^1] + "1";
        }
        return output.ToArray();
    }

    /// <summary>A vowel phone, in ARPAbet, is one of these fifteen.</summary>
    private static bool IsVowel(string phone)
    {
        var b = phone.Length > 0 && char.IsDigit(phone[^1]) ? phone[..^1] : phone;
        return b is "AA" or "AE" or "AH" or "AO" or "AW" or "AY" or "EH" or "ER"
                 or "EY" or "IH" or "IY" or "OW" or "OY" or "UH" or "UW";
    }

    private static string? Lookup(Dictionary<string, string> table, string word, int index, int window)
        => table.TryGetValue(ContextKey(word, index, window), out var emission) ? emission : null;

    /// <summary>Build the context key. Must match the trainer's exactly, or nothing ever matches.</summary>
    internal static string ContextKey(string word, int index, int window)
    {
        var sb = new StringBuilder(window * 2 + 3);
        for (int k = index - window; k <= index + window; k++)
        {
            if (k == index) sb.Append('[');
            sb.Append(k < 0 ? '^' : k >= word.Length ? '$' : word[k]);
            if (k == index) sb.Append(']');
        }
        return sb.ToString();
    }
}
