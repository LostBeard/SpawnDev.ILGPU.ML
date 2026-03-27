namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Byte-Pair Encoding (BPE) tokenizer for text models.
/// Compatible with GPT-2/CLIP/Whisper tokenization.
/// Loads vocabulary and merge rules from standard files.
/// </summary>
public class BPETokenizer
{
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<string, string> _cache = new();

    /// <summary>Size of the vocabulary.</summary>
    public int VocabSize => _encoder.Count;

    /// <summary>
    /// Create a BPE tokenizer from vocab and merges data.
    /// </summary>
    /// <param name="vocab">Token-to-id mapping (JSON format: {"token": id, ...})</param>
    /// <param name="merges">BPE merge rules (one per line, space-separated pairs)</param>
    public BPETokenizer(Dictionary<string, int> vocab, string[] merges)
    {
        _encoder = vocab;
        _decoder = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

        _bpeRanks = new Dictionary<(string, string), int>();
        for (int i = 0; i < merges.Length; i++)
        {
            var parts = merges[i].Split(' ', 2);
            if (parts.Length == 2)
            {
                _bpeRanks[(parts[0], parts[1])] = i;
            }
        }
    }

    /// <summary>
    /// Load tokenizer from vocab JSON string and merges text.
    /// Standard format used by HuggingFace tokenizers.
    /// </summary>
    public static BPETokenizer Load(string vocabJson, string mergesText)
    {
        // Simple JSON parser for {"token": id} format
        var vocab = ParseVocabJson(vocabJson);

        // Parse merges (skip header line if present)
        var lines = mergesText.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var merges = lines
            .Where(l => !l.StartsWith("#version"))
            .ToArray();

        return new BPETokenizer(vocab, merges);
    }

    /// <summary>
    /// Load tokenizer from the unified HuggingFace tokenizer.json format.
    /// This single file contains both vocabulary and merge rules.
    /// All HuggingFace models use this format.
    /// </summary>
    public static BPETokenizer LoadFromTokenizerJson(string tokenizerJson)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(tokenizerJson);
        var model = doc.RootElement.GetProperty("model");

        // Extract vocab: {"token": id, ...}
        var vocab = new Dictionary<string, int>();
        foreach (var prop in model.GetProperty("vocab").EnumerateObject())
            vocab[prop.Name] = prop.Value.GetInt32();

        // Extract merges: can be ["pair1 pair2", ...] (strings) or [["a","b"], ...] (arrays)
        var mergesArray = model.GetProperty("merges").EnumerateArray().ToArray();
        string[] merges;
        if (mergesArray.Length > 0 && mergesArray[0].ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            // Array format: [["Ġ","t"], ["Ġ","a"], ...] → "Ġ t", "Ġ a"
            merges = mergesArray.Select(e =>
            {
                var parts = e.EnumerateArray().ToArray();
                return $"{parts[0].GetString()} {parts[1].GetString()}";
            }).ToArray();
        }
        else
        {
            // String format: ["Ġ t", "Ġ a", ...]
            merges = mergesArray.Select(e => e.GetString()!).ToArray();
        }

        return new BPETokenizer(vocab, merges);
    }

    /// <summary>
    /// Encode text to token IDs.
    /// </summary>
    public int[] Encode(string text)
    {
        var tokens = new List<int>();

        // Pre-tokenize: split on whitespace and punctuation boundaries
        var words = PreTokenize(text);

        foreach (var word in words)
        {
            // Convert to byte-level representation
            var byteWord = string.Concat(word.Select(c => ByteToUnicode((byte)c)));

            // Apply BPE
            var bpeResult = ApplyBPE(byteWord);

            // Look up token IDs
            foreach (var token in bpeResult.Split(' '))
            {
                if (_encoder.TryGetValue(token, out int id))
                {
                    tokens.Add(id);
                }
            }
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decode token IDs back to text.
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        var tokens = tokenIds
            .Where(id => _decoder.ContainsKey(id))
            .Select(id => _decoder[id]);

        var text = string.Concat(tokens);

        // Convert byte-level unicode back to bytes
        var bytes = new List<byte>();
        foreach (char c in text)
        {
            if (UnicodeToByte.TryGetValue(c, out byte b))
                bytes.Add(b);
        }

        return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>
    /// Encode text with special tokens for CLIP (start/end of text).
    /// </summary>
    public int[] EncodeCLIP(string text, int maxLength = 77)
    {
        var tokens = Encode(text.ToLowerInvariant());
        var result = new List<int> { 49406 }; // <|startoftext|>
        result.AddRange(tokens.Take(maxLength - 2));
        result.Add(49407); // <|endoftext|>

        // Pad to maxLength
        while (result.Count < maxLength)
            result.Add(0);

        return result.Take(maxLength).ToArray();
    }

    private string ApplyBPE(string token)
    {
        if (_cache.TryGetValue(token, out var cached))
            return cached;

        var word = token.Select(c => c.ToString()).ToList();

        while (word.Count >= 2)
        {
            // Find the pair with lowest BPE rank
            (string, string)? bestPair = null;
            int bestRank = int.MaxValue;

            for (int i = 0; i < word.Count - 1; i++)
            {
                var pair = (word[i], word[i + 1]);
                if (_bpeRanks.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestPair = pair;
                }
            }

            if (bestPair == null) break;

            // Merge the best pair
            var merged = bestPair.Value.Item1 + bestPair.Value.Item2;
            var newWord = new List<string>();
            int j = 0;
            while (j < word.Count)
            {
                if (j < word.Count - 1 && word[j] == bestPair.Value.Item1 && word[j + 1] == bestPair.Value.Item2)
                {
                    newWord.Add(merged);
                    j += 2;
                }
                else
                {
                    newWord.Add(word[j]);
                    j++;
                }
            }
            word = newWord;
        }

        var result = string.Join(" ", word);
        _cache[token] = result;
        return result;
    }

    /// <summary>
    /// Simple pre-tokenization: split on whitespace, keeping whitespace as part of following token.
    /// GPT-2 style: spaces become "Ġ" (byte 0x20 → unicode Ġ).
    /// </summary>
    private static List<string> PreTokenize(string text)
    {
        // GPT-2 style: spaces attach to the FOLLOWING word, not the preceding one.
        // "The cat sat" → ["The", " cat", " sat"]
        // Punctuation stays with the preceding word or becomes its own token.
        var words = new List<string>();
        var current = new List<char>();

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (char.IsWhiteSpace(c))
            {
                // Flush current word
                if (current.Count > 0)
                {
                    words.Add(new string(current.ToArray()));
                    current.Clear();
                }
                // Space attaches to the next word
                current.Add(c);
            }
            else if (char.IsPunctuation(c))
            {
                if (current.Count > 0)
                {
                    words.Add(new string(current.ToArray()));
                    current.Clear();
                }
                words.Add(c.ToString());
            }
            else
            {
                current.Add(c);
            }
        }

        if (current.Count > 0)
            words.Add(new string(current.ToArray()));

        return words;
    }

    // Byte-to-unicode mapping (GPT-2 style)
    private static readonly Dictionary<byte, char> _byteToUnicode = BuildByteToUnicode();
    private static readonly Dictionary<char, byte> UnicodeToByte =
        _byteToUnicode.ToDictionary(kv => kv.Value, kv => kv.Key);

    private static char ByteToUnicode(byte b) =>
        _byteToUnicode.TryGetValue(b, out char c) ? c : (char)(b + 256);

    private static Dictionary<byte, char> BuildByteToUnicode()
    {
        var map = new Dictionary<byte, char>();
        int n = 0;

        // Printable ASCII ranges
        for (int i = (int)'!'; i <= (int)'~'; i++) map[(byte)i] = (char)i;
        for (int i = 0xA1; i <= 0xAC; i++) map[(byte)i] = (char)i;
        for (int i = 0xAE; i <= 0xFF; i++) map[(byte)i] = (char)i;

        // Map remaining bytes to unicode characters starting at 256
        n = 0;
        for (int i = 0; i < 256; i++)
        {
            if (!map.ContainsKey((byte)i))
            {
                map[(byte)i] = (char)(256 + n);
                n++;
            }
        }

        return map;
    }

    private static Dictionary<string, int> ParseVocabJson(string json)
    {
        var vocab = new Dictionary<string, int>();
        // Simple parser for {"token": id, ...} format
        json = json.Trim();
        if (json.StartsWith("{")) json = json[1..];
        if (json.EndsWith("}")) json = json[..^1];

        int pos = 0;
        while (pos < json.Length)
        {
            // Skip whitespace and commas
            while (pos < json.Length && (json[pos] == ',' || json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t'))
                pos++;

            if (pos >= json.Length) break;

            // Parse key (quoted string)
            if (json[pos] != '"') { pos++; continue; }
            pos++; // skip opening quote
            int keyStart = pos;
            while (pos < json.Length && json[pos] != '"')
            {
                if (json[pos] == '\\') pos++; // skip escaped char
                pos++;
            }
            string key = json[keyStart..pos];
            pos++; // skip closing quote

            // Skip colon
            while (pos < json.Length && (json[pos] == ':' || json[pos] == ' '))
                pos++;

            // Parse value (integer)
            int valStart = pos;
            while (pos < json.Length && (char.IsDigit(json[pos]) || json[pos] == '-'))
                pos++;

            if (int.TryParse(json[valStart..pos], out int val))
            {
                // Unescape common sequences
                key = key.Replace("\\n", "\n").Replace("\\r", "\r")
                         .Replace("\\t", "\t").Replace("\\\"", "\"")
                         .Replace("\\\\", "\\");
                vocab[key] = val;
            }
        }

        return vocab;
    }
}
