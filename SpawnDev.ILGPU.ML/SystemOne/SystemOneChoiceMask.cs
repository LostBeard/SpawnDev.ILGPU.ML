namespace SpawnDev.ILGPU.ML.SystemOne;

/// <summary>
/// Softmax probability helpers for System One choice heads — masking, renormalize, argmax.
/// </summary>
public static class SystemOneChoiceMask
{
    /// <summary>
    /// Zero disallowed options, renormalize remaining mass, pick the new argmax.
    /// If every option is disallowed, returns the original answer unchanged.
    /// </summary>
    /// <param name="answer">Softmax answer from the head.</param>
    /// <param name="allowed">Length must equal the number of criteria keys; true = keep.</param>
    /// <param name="optionKeys">Option keys in the same order as the head's output logits.</param>
    public static ChoiceAnswer Apply(ChoiceAnswer answer, ReadOnlySpan<bool> allowed, IReadOnlyList<string> optionKeys)
    {
        if (allowed.Length != optionKeys.Count)
            throw new ArgumentException($"Mask length {allowed.Length} != option count {optionKeys.Count}.");

        int n = optionKeys.Count;
        var masked = new float[n];
        float sum = 0f;
        int allowedCount = 0;
        for (int i = 0; i < n; i++)
        {
            float p = answer.Probabilities.TryGetValue(optionKeys[i], out var v) ? v : 0f;
            if (allowed[i])
            {
                masked[i] = p;
                sum += p;
                allowedCount++;
            }
        }

        if (allowedCount == 0)
            return answer;

        if (sum <= 1e-12f)
        {
            // Model put ~0 on all legal moves — uniform over allowed.
            float u = 1f / allowedCount;
            for (int i = 0; i < n; i++)
                masked[i] = allowed[i] ? u : 0f;
            sum = 1f;
        }

        int best = -1;
        float bestP = -1f;
        var map = new Dictionary<string, float>(n);
        for (int i = 0; i < n; i++)
        {
            float p = allowed[i] ? masked[i] / sum : 0f;
            map[optionKeys[i]] = p;
            if (allowed[i] && p > bestP)
            {
                bestP = p;
                best = i;
            }
        }

        return new ChoiceAnswer(optionKeys[best], bestP, map);
    }

    /// <summary>Build a bool mask from a predicate over option indices.</summary>
    public static bool[] FromPredicate(int count, Func<int, bool> isAllowed)
    {
        var m = new bool[count];
        for (int i = 0; i < count; i++)
            m[i] = isAllowed(i);
        return m;
    }
}
