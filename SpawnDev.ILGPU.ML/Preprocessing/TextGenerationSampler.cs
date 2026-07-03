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

        // CANDIDATE PRUNING (2026-07-03, same as TopP): the top-K by logit all sit within a window of
        // the max unless K is enormous - prune to logit >= max - 20*T first, then argsort the
        // (usually tiny) candidate set instead of the full ~152K vocab (the full Array.Sort was
        // ~100-250ms/token in non-AOT WASM). If pruning leaves fewer than K candidates, the missing
        // ones carried < e^-20 relative probability each - sampling among the survivors is the same
        // distribution to ~1e-4 mass.
        float pruneMax = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > pruneMax) pruneMax = logits[i];
        float pruneCutoff = pruneMax - 20f * temperature;
        var candIdx = new List<int>(Math.Max(256, k));
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] >= pruneCutoff) candIdx.Add(i);
        var candLogits = new float[candIdx.Count];
        for (int i = 0; i < candIdx.Count; i++) candLogits[i] = logits[candIdx[i]];
        k = Math.Min(k, candIdx.Count);
        var candOrder = ArgsortDescending(candLogits);
        var indices = new int[candOrder.Length];
        for (int i = 0; i < candOrder.Length; i++) indices[i] = candIdx[candOrder[i]];

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

        // CANDIDATE PRUNING (2026-07-03): the nucleus never contains a token whose scaled logit sits
        // more than LogitWindow below the max - e^-20 ≈ 2e-9, so even 150K such tokens contribute
        // < 3e-4 total probability mass, far below any p cutoff. Pruning first means the softmax
        // EXP and the argsort run over ~10s-100s of candidates instead of the FULL vocabulary
        // (qwen2.5 = 151,936). This is the difference between ~1ms and ~100-250ms PER TOKEN in
        // non-AOT Blazor WASM (152K MathF.Exp + a 152K Array.Sort each step made the /ai-chat page
        // sampler-bound). The nucleus itself is mathematically unchanged (same candidates, same
        // order, same renormalization) up to the negligible pruned mass.
        const float LogitWindow = 20f;
        float maxLogit = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxLogit) maxLogit = logits[i];
        float cutoff = maxLogit - LogitWindow * temperature;   // (l - max)/T >= -window

        var candIdx = new List<int>(256);
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] >= cutoff) candIdx.Add(i);

        int nCand = candIdx.Count;
        var probs = new float[nCand];
        float sum = 0;
        for (int i = 0; i < nCand; i++)
        {
            probs[i] = MathF.Exp((logits[candIdx[i]] - maxLogit) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < nCand; i++) probs[i] /= sum;

        // Sort candidates by probability (descending). MUST be Array.Sort, not LINQ (interpreted-
        // WASM delegate cost - see ArgsortDescending). Now over the pruned candidate set only.
        var sortedIndices = ArgsortDescending(probs);

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
        return candIdx[sortedIndices[sampledIdx]];
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

    /// <summary>
    /// Indices of <paramref name="values"/> ordered by value DESCENDING. Uses Array.Sort on primitive
    /// arrays (negated keys → ascending sort == descending by value) rather than LINQ OrderByDescending.
    /// This is a hard requirement, not a micro-opt: token-generation runs in INTERPRETED Blazor WASM
    /// (RunAOTCompilation=false), where LINQ's iterator + per-element keySelector delegate + boxed
    /// comparer over the full ~50k-token vocabulary cost ~tens of seconds PER call and hung multi-token
    /// top-p/top-k sampling. Array.Sort compares the float keys intrinsically (no interpreted delegate),
    /// which is orders of magnitude faster under the interpreter. Ties order arbitrarily (introsort is
    /// not stable), which is fine for sampling and still deterministic for identical input.
    /// </summary>
    private static int[] ArgsortDescending(float[] values)
    {
        int n = values.Length;
        var keys = new float[n];
        var idx = new int[n];
        for (int i = 0; i < n; i++) { keys[i] = -values[i]; idx[i] = i; }
        Array.Sort(keys, idx);
        return idx;
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
    /// <summary>Maximum number of new tokens to generate. NULL (the default) means "defer to the
    /// pipeline's own MaxNewTokens" - so passing a GenerationConfig for sampling does NOT silently
    /// override an explicitly-set TextGenerationPipeline.MaxNewTokens. Set a value here only to make
    /// the config itself the source of truth for the token count.</summary>
    public int? MaxNewTokens { get; set; } = null;

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

    /// <summary>Optional RNG seed for reproducible sampling. Null → non-deterministic (Random.Shared).
    /// Set for unit tests that must assert identical output across runs.</summary>
    public int? Seed { get; set; }

    /// <summary>End-of-sequence token ID. Generation stops when this is produced.</summary>
    public int EosTokenId { get; set; } = -1;

    /// <summary>Pad token ID.</summary>
    public int PadTokenId { get; set; } = 0;

    /// <summary>Whether to return the input tokens as part of the output.</summary>
    public bool ReturnInputTokens { get; set; } = true;
}
