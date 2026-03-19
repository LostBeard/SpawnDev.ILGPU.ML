namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Token sampling strategies for autoregressive text generation.
/// Used by text-generation, translation, summarization, and image captioning pipelines.
/// All methods operate on CPU logits arrays (small, not worth GPU dispatch).
/// </summary>
public static class TextGenerationSampler
{
    /// <summary>
    /// Greedy sampling: always pick the token with highest probability.
    /// Deterministic, fast, but can produce repetitive output.
    /// </summary>
    public static int Greedy(float[] logits)
    {
        int bestIdx = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal)
            {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /// <summary>
    /// Top-K sampling: sample from the K most likely tokens.
    /// </summary>
    /// <param name="logits">Raw logits (pre-softmax)</param>
    /// <param name="k">Number of top tokens to consider</param>
    /// <param name="temperature">Temperature for softmax (lower = more deterministic)</param>
    /// <param name="rng">Random number generator</param>
    public static int TopK(float[] logits, int k, float temperature = 1.0f, Random? rng = null)
    {
        rng ??= Random.Shared;
        k = Math.Min(k, logits.Length);

        // Find top-K indices
        var indices = TopKIndices(logits, k);

        // Apply temperature and softmax over top-K only
        var probs = new float[k];
        float maxLogit = float.MinValue;
        for (int i = 0; i < k; i++)
            if (logits[indices[i]] > maxLogit) maxLogit = logits[indices[i]];

        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            probs[i] = MathF.Exp((logits[indices[i]] - maxLogit) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < k; i++) probs[i] /= sum;

        // Sample
        return indices[SampleFromDistribution(probs, rng)];
    }

    /// <summary>
    /// Top-P (nucleus) sampling: sample from the smallest set of tokens
    /// whose cumulative probability exceeds P.
    /// </summary>
    /// <param name="logits">Raw logits (pre-softmax)</param>
    /// <param name="p">Cumulative probability threshold (e.g., 0.9)</param>
    /// <param name="temperature">Temperature for softmax</param>
    /// <param name="rng">Random number generator</param>
    public static int TopP(float[] logits, float p = 0.9f, float temperature = 1.0f, Random? rng = null)
    {
        rng ??= Random.Shared;

        // Softmax with temperature
        float maxLogit = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxLogit) maxLogit = logits[i];

        var probs = new float[logits.Length];
        float sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp((logits[i] - maxLogit) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.Length; i++) probs[i] /= sum;

        // Sort by probability (descending)
        var sortedIndices = Enumerable.Range(0, probs.Length)
            .OrderByDescending(i => probs[i])
            .ToArray();

        // Find nucleus (smallest set where cumulative prob >= p)
        float cumulative = 0;
        int nucleusSize = 0;
        for (int i = 0; i < sortedIndices.Length; i++)
        {
            cumulative += probs[sortedIndices[i]];
            nucleusSize = i + 1;
            if (cumulative >= p) break;
        }

        // Renormalize within nucleus
        var nucleusProbs = new float[nucleusSize];
        float nucleusSum = 0;
        for (int i = 0; i < nucleusSize; i++)
        {
            nucleusProbs[i] = probs[sortedIndices[i]];
            nucleusSum += nucleusProbs[i];
        }
        for (int i = 0; i < nucleusSize; i++) nucleusProbs[i] /= nucleusSum;

        // Sample from nucleus
        int sampledIdx = SampleFromDistribution(nucleusProbs, rng);
        return sortedIndices[sampledIdx];
    }

    /// <summary>
    /// Apply repetition penalty to logits for tokens that have already appeared.
    /// Reduces the probability of repeating tokens.
    /// </summary>
    /// <param name="logits">Logits to modify (in-place)</param>
    /// <param name="previousTokens">Tokens that have appeared so far</param>
    /// <param name="penalty">Penalty factor (>1.0 reduces repetition, 1.0 = no effect)</param>
    public static void ApplyRepetitionPenalty(float[] logits, int[] previousTokens, float penalty = 1.2f)
    {
        if (penalty == 1.0f) return;
        foreach (int token in previousTokens)
        {
            if (token >= 0 && token < logits.Length)
            {
                if (logits[token] > 0)
                    logits[token] /= penalty;
                else
                    logits[token] *= penalty;
            }
        }
    }

    /// <summary>
    /// Apply temperature to logits (in-place). Lower temperature = more deterministic.
    /// </summary>
    public static void ApplyTemperature(float[] logits, float temperature)
    {
        if (temperature == 1.0f) return;
        float invTemp = 1.0f / temperature;
        for (int i = 0; i < logits.Length; i++)
            logits[i] *= invTemp;
    }

    // ── Helpers ──

    private static int[] TopKIndices(float[] values, int k)
    {
        // Simple O(n*k) selection
        var indices = new int[k];
        var used = new bool[values.Length];

        for (int ki = 0; ki < k; ki++)
        {
            float best = float.NegativeInfinity;
            int bestIdx = 0;
            for (int i = 0; i < values.Length; i++)
            {
                if (!used[i] && values[i] > best)
                {
                    best = values[i];
                    bestIdx = i;
                }
            }
            indices[ki] = bestIdx;
            used[bestIdx] = true;
        }

        return indices;
    }

    private static int SampleFromDistribution(float[] probs, Random rng)
    {
        float r = (float)rng.NextDouble();
        float cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.Length - 1;
    }
}

/// <summary>
/// Configuration for text generation.
/// </summary>
public class GenerationConfig
{
    /// <summary>Maximum number of new tokens to generate.</summary>
    public int MaxNewTokens { get; set; } = 128;

    /// <summary>Sampling strategy: "greedy", "top_k", "top_p".</summary>
    public string Strategy { get; set; } = "greedy";

    /// <summary>Temperature for sampling (lower = more deterministic).</summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>K value for top-K sampling.</summary>
    public int TopK { get; set; } = 50;

    /// <summary>P value for top-P (nucleus) sampling.</summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>Repetition penalty (>1.0 reduces repetition).</summary>
    public float RepetitionPenalty { get; set; } = 1.0f;

    /// <summary>End-of-sequence token ID. Generation stops when this is produced.</summary>
    public int EosTokenId { get; set; } = -1;

    /// <summary>Pad token ID.</summary>
    public int PadTokenId { get; set; } = 0;

    /// <summary>Whether to return the input tokens as part of the output.</summary>
    public bool ReturnInputTokens { get; set; } = true;
}
