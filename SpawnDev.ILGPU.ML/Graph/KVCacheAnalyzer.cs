namespace SpawnDev.ILGPU.ML.Graph;

/// <summary>
/// Detects KV cache accumulation patterns in autoregressive transformer models.
/// Used by GraphExecutor to transparently inject TurboQuant compression.
///
/// Pattern 1 (Explicit): Model has past_key_values.N.key/value inputs and
/// present.N.key/value outputs. The model tells us exactly where the KV cache is.
///
/// Pattern 2 (Implicit): Model recomputes all K,V each step. Detected by finding
/// Q/K/V projection triplets feeding attention MatMuls. (Future — v4.1.0)
/// </summary>
public class KVCacheAnalyzer
{
    /// <summary>
    /// A detected KV cache point — one per transformer layer.
    /// Maps past_key_values inputs to present outputs.
    /// </summary>
    public class KVCachePoint
    {
        /// <summary>Layer index (0-based).</summary>
        public int LayerIndex { get; init; }

        /// <summary>Model input name for cached keys (e.g., "past_key_values.0.key").</summary>
        public string PastKeyInput { get; init; } = "";

        /// <summary>Model input name for cached values (e.g., "past_key_values.0.value").</summary>
        public string PastValueInput { get; init; } = "";

        /// <summary>Model output name for new keys (e.g., "present.0.key").</summary>
        public string PresentKeyOutput { get; init; } = "";

        /// <summary>Model output name for new values (e.g., "present.0.value").</summary>
        public string PresentValueOutput { get; init; } = "";

        /// <summary>Shape of K/V tensors: [batch, heads, seq, head_dim].</summary>
        public int[]? Shape { get; init; }

        /// <summary>Head dimension (last dim of shape). Used for TurboQuant codebook selection.</summary>
        public int HeadDim => Shape != null && Shape.Length >= 1 ? Shape[^1] : 64;
    }

    /// <summary>
    /// Analysis result for a model's KV cache structure.
    /// </summary>
    public class KVCacheInfo
    {
        /// <summary>Whether the model uses explicit KV cache (past_key_values inputs).</summary>
        public bool HasExplicitKVCache { get; init; }

        /// <summary>Per-layer KV cache points.</summary>
        public KVCachePoint[] Layers { get; init; } = Array.Empty<KVCachePoint>();

        /// <summary>Number of transformer layers with KV cache.</summary>
        public int NumLayers => Layers.Length;

        /// <summary>Name of the use_cache_branch input (if present).</summary>
        public string? UseCacheBranchInput { get; init; }

        /// <summary>Whether TurboQuant should be applied to this model's KV cache.</summary>
        public bool ShouldQuantize => HasExplicitKVCache && NumLayers > 0;
    }

    /// <summary>
    /// Analyze a compiled graph for KV cache patterns.
    /// Returns info about detected cache points for TurboQuant injection.
    /// </summary>
    public static KVCacheInfo Analyze(string[] inputNames, string[] outputNames,
        Dictionary<string, int[]>? inputShapes = null)
    {
        // Pattern 1: Explicit past_key_values inputs
        var pastKeyInputs = new Dictionary<int, string>();
        var pastValueInputs = new Dictionary<int, string>();
        var presentKeyOutputs = new Dictionary<int, string>();
        var presentValueOutputs = new Dictionary<int, string>();
        string? useCacheBranch = null;

        // Scan model inputs for past_key_values.N.key / past_key_values.N.value
        foreach (var name in inputNames)
        {
            if (name == "use_cache_branch")
            {
                useCacheBranch = name;
                continue;
            }

            // Match patterns: "past_key_values.N.key", "past_key_values.N.value",
            // "past_key_values.N.decoder.key". Cross-attention (".encoder.") is NOT an autoregressive
            // cache - it is constant for the whole generation and has no present.* counterpart - so it is
            // skipped rather than allowed to overwrite the self-attention entry for the same layer.
            if (TryParseKVInput(name, out int layer, out bool isKey, out bool isCross))
            {
                if (isCross) continue;
                if (isKey)
                    pastKeyInputs[layer] = name;
                else
                    pastValueInputs[layer] = name;
            }
        }

        // Scan model outputs for present.N.key / present.N.value
        foreach (var name in outputNames)
        {
            if (TryParseKVOutput(name, out int layer, out bool isKey, out bool isCross))
            {
                if (isCross) continue;
                if (isKey)
                    presentKeyOutputs[layer] = name;
                else
                    presentValueOutputs[layer] = name;
            }
        }

        // Build layer list from matched pairs
        var layers = new List<KVCachePoint>();
        var allLayers = new HashSet<int>(pastKeyInputs.Keys);
        allLayers.UnionWith(pastValueInputs.Keys);
        allLayers.UnionWith(presentKeyOutputs.Keys);
        allLayers.UnionWith(presentValueOutputs.Keys);

        foreach (var layer in allLayers.OrderBy(l => l))
        {
            if (pastKeyInputs.ContainsKey(layer) && pastValueInputs.ContainsKey(layer) &&
                presentKeyOutputs.ContainsKey(layer) && presentValueOutputs.ContainsKey(layer))
            {
                int[]? shape = null;
                if (inputShapes != null)
                    inputShapes.TryGetValue(pastKeyInputs[layer], out shape);

                layers.Add(new KVCachePoint
                {
                    LayerIndex = layer,
                    PastKeyInput = pastKeyInputs[layer],
                    PastValueInput = pastValueInputs[layer],
                    PresentKeyOutput = presentKeyOutputs[layer],
                    PresentValueOutput = presentValueOutputs[layer],
                    Shape = shape,
                });
            }
        }

        return new KVCacheInfo
        {
            HasExplicitKVCache = layers.Count > 0,
            Layers = layers.ToArray(),
            UseCacheBranchInput = useCacheBranch,
        };
    }

    /// <summary>
    /// Parse a <c>past_key_values.N[.decoder|.encoder].key|value</c> input name.
    /// </summary>
    /// <remarks>
    /// ⚠️ Encoder-decoder models (Whisper, T5, BART) qualify the name with the ATTENTION BLOCK:
    /// <c>past_key_values.0.decoder.key</c> is the autoregressive self-attention cache, while
    /// <c>past_key_values.0.encoder.key</c> is CROSS-attention over the encoder output - computed once and
    /// constant for the whole generation, with no matching <c>present.*</c> output at all.
    /// <para>
    /// MEASURED 2026-08-30 on <c>onnx-community/whisper-tiny decoder_with_past_model.onnx</c> (17 inputs /
    /// 9 outputs): both forms used to parse to the SAME (layer, isKey) slot, so the encoder entry - which
    /// comes second in the input list - silently OVERWROTE the decoder one. The analyzer then paired
    /// ENCODER past against DECODER present and reported a healthy 4-layer cache, so the executor would
    /// have appended decoder keys into a cache addressed by the static cross-attention inputs. It failed
    /// loudly nowhere. Cross-attention entries are now identified and excluded.
    /// </para>
    /// </remarks>
    /// <param name="name">Input name.</param>
    /// <param name="layer">Parsed layer index.</param>
    /// <param name="isKey">True for a key entry, false for a value entry.</param>
    /// <param name="isCrossAttention">True when the name is an <c>.encoder.</c>-qualified entry.</param>
    /// <returns>True when the name is a past-KV entry.</returns>
    private static bool TryParseKVInput(string name, out int layer, out bool isKey, out bool isCrossAttention)
    {
        layer = -1;
        isKey = false;
        isCrossAttention = false;

        // "past_key_values.0.key" → layer=0, isKey=true
        // "past_key_values.0.decoder.key" → layer=0, isKey=true, self-attention
        // "past_key_values.0.encoder.key" → layer=0, isKey=true, CROSS-attention (excluded by the caller)
        if (!name.StartsWith("past_key_values.") && !name.StartsWith("past_")) return false;
        var parts = name.Split('.');
        if (parts.Length < 3) return false;
        if (!TryParseKeyOrValue(parts[^1], out isKey)) return false;
        if (!TryParseLayerAndQualifier(parts, out layer, out isCrossAttention)) return false;
        return true;
    }

    /// <summary>
    /// Parse a <c>present.N[.decoder|.encoder].key|value</c> output name. See
    /// <see cref="TryParseKVInput"/> for why the qualifier matters.
    /// </summary>
    /// <param name="name">Output name.</param>
    /// <param name="layer">Parsed layer index.</param>
    /// <param name="isKey">True for a key entry, false for a value entry.</param>
    /// <param name="isCrossAttention">True when the name is an <c>.encoder.</c>-qualified entry.</param>
    /// <returns>True when the name is a present-KV entry.</returns>
    private static bool TryParseKVOutput(string name, out int layer, out bool isKey, out bool isCrossAttention)
    {
        layer = -1;
        isKey = false;
        isCrossAttention = false;

        if (!name.StartsWith("present.")) return false;
        var parts = name.Split('.');
        if (parts.Length < 3) return false;
        if (!TryParseKeyOrValue(parts[^1], out isKey)) return false;
        if (!TryParseLayerAndQualifier(parts, out layer, out isCrossAttention)) return false;
        return true;
    }

    /// <summary>Recognise the trailing <c>key</c>/<c>k</c>/<c>value</c>/<c>v</c> segment.</summary>
    private static bool TryParseKeyOrValue(string last, out bool isKey)
    {
        switch (last.ToLowerInvariant())
        {
            case "key":
            case "k": isKey = true; return true;
            case "value":
            case "v": isKey = false; return true;
            default: isKey = false; return false;
        }
    }

    /// <summary>
    /// Pull the layer index out of a split KV name, and say whether the segment before key/value marks it
    /// as cross-attention. A NUMERIC segment there means there is no qualifier at all
    /// (<c>present.0.key</c>); otherwise that segment is the block name (<c>decoder</c> / <c>encoder</c>)
    /// and the layer index is the one right after the prefix.
    /// </summary>
    private static bool TryParseLayerAndQualifier(string[] parts, out int layer, out bool isCrossAttention)
    {
        isCrossAttention = false;
        if (int.TryParse(parts[^2], out layer)) return true;      // unqualified: ...N.key

        isCrossAttention = parts[^2].Equals("encoder", StringComparison.OrdinalIgnoreCase);
        return int.TryParse(parts[1], out layer);                 // qualified: ...N.<block>.key
    }
}
