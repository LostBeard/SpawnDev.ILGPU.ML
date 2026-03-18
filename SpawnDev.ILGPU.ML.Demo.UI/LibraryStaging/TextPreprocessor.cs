namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Text preprocessing utilities for NLP models.
/// Handles common text normalization, padding, and attention mask creation.
/// </summary>
public static class TextPreprocessor
{
    /// <summary>
    /// Normalize text for model input: lowercase, strip extra whitespace, basic cleanup.
    /// </summary>
    public static string NormalizeText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";

        // Normalize whitespace
        text = System.Text.RegularExpressions.Regex.Replace(text.Trim(), @"\s+", " ");

        return text;
    }

    /// <summary>
    /// Pad token IDs to a fixed length. Truncates if too long.
    /// </summary>
    /// <param name="tokenIds">Input token IDs</param>
    /// <param name="maxLength">Target length</param>
    /// <param name="padToken">Padding token ID (usually 0)</param>
    /// <returns>Padded/truncated token IDs</returns>
    public static int[] PadOrTruncate(int[] tokenIds, int maxLength, int padToken = 0)
    {
        if (tokenIds.Length == maxLength) return tokenIds;
        if (tokenIds.Length > maxLength) return tokenIds[..maxLength];

        var padded = new int[maxLength];
        Array.Copy(tokenIds, padded, tokenIds.Length);
        for (int i = tokenIds.Length; i < maxLength; i++)
            padded[i] = padToken;
        return padded;
    }

    /// <summary>
    /// Create an attention mask for padded token IDs.
    /// 1 for real tokens, 0 for padding.
    /// </summary>
    public static int[] CreateAttentionMask(int[] tokenIds, int padToken = 0)
    {
        var mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            mask[i] = tokenIds[i] != padToken ? 1 : 0;
        }
        return mask;
    }

    /// <summary>
    /// Create token type IDs for two-segment input (e.g., BERT sentence pairs).
    /// Segment A tokens get 0, segment B tokens get 1.
    /// </summary>
    public static int[] CreateTokenTypeIds(int segmentALength, int segmentBLength, int maxLength)
    {
        var typeIds = new int[maxLength];
        for (int i = segmentALength; i < segmentALength + segmentBLength && i < maxLength; i++)
        {
            typeIds[i] = 1;
        }
        return typeIds;
    }

    /// <summary>
    /// Convert token IDs to a float tensor (some models expect float input).
    /// </summary>
    public static float[] TokenIdsToFloat(int[] tokenIds)
    {
        var floats = new float[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            floats[i] = tokenIds[i];
        return floats;
    }

    /// <summary>
    /// Create position IDs [0, 1, 2, ..., length-1] for models that need explicit position input.
    /// </summary>
    public static int[] CreatePositionIds(int length)
    {
        var ids = new int[length];
        for (int i = 0; i < length; i++)
            ids[i] = i;
        return ids;
    }

    /// <summary>
    /// Simple whitespace tokenizer (for models that don't use BPE).
    /// Splits on whitespace and punctuation.
    /// </summary>
    public static string[] SimpleTokenize(string text)
    {
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();

        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }
            else if (char.IsPunctuation(c))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
                tokens.Add(c.ToString());
            }
            else
            {
                current.Append(c);
            }
        }

        if (current.Length > 0)
            tokens.Add(current.ToString());

        return tokens.ToArray();
    }

    /// <summary>
    /// Compute cosine similarity between two embedding vectors.
    /// Used for semantic search and zero-shot classification.
    /// </summary>
    public static float CosineSimilarity(float[] a, float[] b)
    {
        int length = Math.Min(a.Length, b.Length);
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denom > 1e-8f ? dot / denom : 0;
    }

    /// <summary>
    /// Softmax over an array of float values (e.g., CLIP logits).
    /// </summary>
    public static float[] Softmax(float[] logits)
    {
        float max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0;
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp(logits[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.Length; i++)
            result[i] /= sum;

        return result;
    }

    /// <summary>
    /// Top-K selection from a scored array. Returns (index, score) pairs sorted by score.
    /// </summary>
    public static (int Index, float Score)[] TopK(float[] scores, int k = 5)
    {
        return Enumerable.Range(0, scores.Length)
            .OrderByDescending(i => scores[i])
            .Take(k)
            .Select(i => (i, scores[i]))
            .ToArray();
    }
}
