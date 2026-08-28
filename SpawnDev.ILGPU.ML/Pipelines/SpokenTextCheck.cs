namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Compares what a synthesiser was ASKED to say with what a recogniser HEARD.
/// </summary>
/// <remarks>
/// <para>
/// The comparison skips freely at the head and tail of the transcript. That is not leniency, it is
/// necessary: ZipVoice regenerates its reference clip's own speech ahead of the line it was asked to
/// speak, so every transcript opens with a few words of the voice being cloned. Charging for those would
/// put a floor under every score and bury the thing being measured. Everything INSIDE the sentence -
/// every substitution, deletion and insertion - is charged in full.
/// </para>
/// <para>
/// It hears wrong WORDS only. It cannot hear an accent, an odd rhythm, a stutter or an audible breath -
/// all four have been observed in renders this scored at zero. Treat a clean result as "the words
/// survived", never as "it sounds right".
/// </para>
/// </remarks>
public static class SpokenTextCheck
{
    /// <summary>Word error rate of <paramref name="heard"/> against <paramref name="expected"/>, 0 to 1.</summary>
    public static double WordErrorRate(string expected, string heard)
    {
        var truth = Words(expected);
        var hypothesis = Words(heard);
        if (truth.Length == 0) return hypothesis.Length == 0 ? 0 : 1;

        // Row zero is all zeros, so starting anywhere in the transcript is free; taking the minimum of
        // the last row makes ending anywhere free too.
        var previous = new int[hypothesis.Length + 1];
        var current = new int[hypothesis.Length + 1];
        for (int i = 1; i <= truth.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= hypothesis.Length; j++)
                current[j] = Math.Min(Math.Min(previous[j] + 1, current[j - 1] + 1),
                                      previous[j - 1] + (truth[i - 1] == hypothesis[j - 1] ? 0 : 1));
            (previous, current) = (current, previous);
        }
        return previous.Min() / (double)truth.Length;
    }

    private static string[] Words(string text)
    {
        if (string.IsNullOrEmpty(text)) return [];
        var sb = new System.Text.StringBuilder(text.Length);
        foreach (var c in text.ToLowerInvariant())
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}
