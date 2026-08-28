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
/// Lookup walks left to right, taking the widest context that matches, then narrowing a letter at a time
/// until it reaches the letter alone. That backoff is why "gh" can be silent in "night" and voiced in
/// "ghost" without anyone writing a rule for either. Context width is worth more than evidence per
/// context: measured over held-out words, the thinnest three-letter rule is right 83.3% of the time while
/// the best-supported one-letter rule manages 77.6%, which is why nothing is dropped for being rare.
/// </para>
/// <para>
/// How WIDE that context goes is read out of the model file rather than fixed here. A context states its
/// own width by its shape - <c>[a]</c> is the letter alone, <c>c[a]t</c> is one letter either side - so a
/// retrained model can carry more context than this code was written against and still load. The widths
/// are simply what the trainer found worth keeping.
/// </para>
/// </remarks>
public sealed class LetterToSound
{
    private readonly Dictionary<string, string> _sounds;
    private readonly Dictionary<string, string> _stress;
    private readonly int _widest;

    private LetterToSound(Dictionary<string, string> sounds, Dictionary<string, string> stress, int widest)
    {
        _sounds = sounds;
        _stress = stress;
        _widest = widest;
    }

    /// <summary>Number of context rules loaded, sound and stress together.</summary>
    public int Count => _sounds.Count + _stress.Count;

    /// <summary>Parse a model produced by <c>tools/lts-train</c>.</summary>
    public static LetterToSound Parse(IEnumerable<string> lines)
    {
        var sounds = new Dictionary<string, string>(StringComparer.Ordinal);
        var stress = new Dictionary<string, string>(StringComparer.Ordinal);
        int widest = 0;
        foreach (var line in lines)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            var emission = line[(tab + 1)..].Trim();

            // Stress rules are prefixed with * so a single file carries both models.
            var context = line[0] == '*' ? line[1..tab] : line[..tab];
            if (line[0] == '*') stress[context] = emission;
            else sounds[context] = emission == "-" ? "" : emission;

            // A context is the letter in brackets plus the same number of letters either side, so its
            // length states its width. Reading it from the data is what lets a wider model load into
            // code that predates it.
            widest = Math.Max(widest, (context.Length - 3) / 2);
        }
        return new LetterToSound(sounds, stress, widest);
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
            var emission = Widest(_sounds, letters, i);
            if (string.IsNullOrEmpty(emission)) continue;

            // Stress comes from its own model over the same contexts. Every vowel MUST come back carrying
            // a digit: a vowel without one is not valid ARPAbet, and downstream it would be discarded as
            // an unknown phone - which is how an entire name once came out as its consonants alone.
            var phones = emission.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var digits = Widest(_stress, letters, i) ?? "";
            for (int k = 0; k < phones.Length; k++)
                output.Add(IsVowel(phones[k])
                    ? phones[k] + (k < digits.Length && char.IsDigit(digits[k]) ? digits[k] : '0')
                    : phones[k]);
        }

        // Exactly ONE primary stress per word. The per-letter model can mark several, which is not
        // English - "Tuvok" came back with two - and stress is what the downstream model punishes
        // hardest, so emitting two is worse than picking the wrong one. The FIRST is kept.
        //
        // Keeping the first looks like a shortcut and is not: the alternative was tried and MEASURED.
        // Keeping the one whose context rule was most specific - on the reasoning that position is not
        // evidence - loses on held-out words. Of the 193 where the two policies disagree, earliest is
        // right 24 times and most-specific 13. English favours early stress by enough that it beats a
        // better-looking argument, so the shortcut stays. `tools/lts-train --analyze` reprints that
        // comparison, and it is the thing to re-run before changing this line.
        //
        // ⚠️ It has a known cost, and it is not a small one: "aubriella" comes back stressed on the
        // "au". Every "-ella" name in the dictionary stresses the ELL (gabriella, isabella, daniella,
        // ariella) and this model gets "briella" right on its own - but for "aubriella" the word-initial
        // rule marks a primary first and wins on position. That is a real defect in the name this
        // component exists to say; the fix is pronunciation by analogy, not a different tie-break.
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

    /// <summary>The widest context rule that matches at this letter, narrowing until one does.</summary>
    private string? Widest(Dictionary<string, string> table, string word, int index)
    {
        for (int window = _widest; window >= 0; window--)
            if (table.TryGetValue(ContextKey(word, index, window), out var emission)) return emission;
        return null;
    }

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
