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

            // Match patterns: "past_key_values.N.key", "past_key_values.N.value"
            if (TryParseKVInput(name, out int layer, out bool isKey))
            {
                if (isKey)
                    pastKeyInputs[layer] = name;
                else
                    pastValueInputs[layer] = name;
            }
        }

        // Scan model outputs for present.N.key / present.N.value
        foreach (var name in outputNames)
        {
            if (TryParseKVOutput(name, out int layer, out bool isKey))
            {
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
    /// Parse "past_key_values.N.key" or "past_key_values.N.value" input names.
    /// Also handles variants: "past_key_values.N.key", "pkv.N.key", etc.
    /// </summary>
    private static bool TryParseKVInput(string name, out int layer, out bool isKey)
    {
        layer = -1;
        isKey = false;

        // "past_key_values.0.key" → layer=0, isKey=true
        // "past_key_values.0.value" → layer=0, isKey=false
        if (name.StartsWith("past_key_values.") || name.StartsWith("past_"))
        {
            var parts = name.Split('.');
            if (parts.Length >= 3)
            {
                if (int.TryParse(parts[^2], out layer) || int.TryParse(parts[1], out layer))
                {
                    var lastPart = parts[^1].ToLowerInvariant();
                    if (lastPart == "key" || lastPart == "k") { isKey = true; return true; }
                    if (lastPart == "value" || lastPart == "v") { isKey = false; return true; }
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Parse "present.N.key" or "present.N.value" output names.
    /// </summary>
    private static bool TryParseKVOutput(string name, out int layer, out bool isKey)
    {
        layer = -1;
        isKey = false;

        // "present.0.key" → layer=0, isKey=true
        if (name.StartsWith("present."))
        {
            var parts = name.Split('.');
            if (parts.Length >= 3)
            {
                if (int.TryParse(parts[1], out layer))
                {
                    var lastPart = parts[^1].ToLowerInvariant();
                    if (lastPart == "key" || lastPart == "k") { isKey = true; return true; }
                    if (lastPart == "value" || lastPart == "v") { isKey = false; return true; }
                }
            }
        }

        return false;
    }
}
