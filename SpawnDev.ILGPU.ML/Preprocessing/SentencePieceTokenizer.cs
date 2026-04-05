namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// SentencePiece BPE tokenizer for LLaMA/Mistral/Gemma models.
/// Implements the Unigram/BPE algorithm used by SentencePiece:
/// - Tokens are UTF-8 byte sequences with U+2581 (▁) as word-start marker
/// - Merge priority determined by score (lower score = higher priority for Unigram,
///   higher score = higher priority for BPE — we use score-based greedy matching)
/// - Byte fallback: unknown characters encoded as &lt;0xHH&gt; byte tokens
///
/// Compatible with GGUF metadata format:
///   tokenizer.ggml.model = "llama"
///   tokenizer.ggml.tokens = string[] (token strings)
///   tokenizer.ggml.scores = float[] (token scores/log-probabilities)
///   tokenizer.ggml.token_type = int[] (0=normal, 1=unknown, 2=control, 3=user, 4=unused, 5=byte)
/// </summary>
public class SentencePieceTokenizer : ITokenizer
{
    private readonly string[] _vocab;
    private readonly float[] _scores;
    private readonly int[] _tokenTypes;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly int _bosId;
    private readonly int _eosId;
    private readonly int _unkId;

    /// <summary>Vocabulary size.</summary>
    public int VocabSize => _vocab.Length;

    /// <summary>BOS token ID.</summary>
    public int BosId => _bosId;

    /// <summary>EOS token ID.</summary>
    public int EosId => _eosId;

    /// <summary>
    /// Create a SentencePiece tokenizer from GGUF metadata arrays.
    /// </summary>
    public SentencePieceTokenizer(string[] tokens, float[] scores, int[]? tokenTypes = null)
    {
        _vocab = tokens;
        _scores = scores;
        _tokenTypes = tokenTypes ?? new int[tokens.Length];
        _tokenToId = new Dictionary<string, int>(tokens.Length);
        for (int i = 0; i < tokens.Length; i++)
            _tokenToId[tokens[i]] = i;

        // Find special token IDs by type or content
        _bosId = -1; _eosId = -1; _unkId = -1;
        for (int i = 0; i < _tokenTypes.Length; i++)
        {
            if (_tokenTypes[i] == 2) // control token
            {
                var t = tokens[i];
                if (t == "<s>" || t == "<|begin_of_text|>") _bosId = i;
                else if (t == "</s>" || t == "<|end_of_text|>") _eosId = i;
            }
            if (_tokenTypes[i] == 1) _unkId = i; // unknown
        }
        // Fallback: common positions
        if (_bosId < 0 && _tokenToId.TryGetValue("<s>", out int bid)) _bosId = bid;
        if (_eosId < 0 && _tokenToId.TryGetValue("</s>", out int eid)) _eosId = eid;
        if (_unkId < 0 && _tokenToId.TryGetValue("<unk>", out int uid)) _unkId = uid;
        if (_bosId < 0) _bosId = 1; // llama default
        if (_eosId < 0) _eosId = 2; // llama default
    }

    /// <summary>
    /// Encode text to token IDs using greedy longest-match with score-based BPE.
    /// </summary>
    public int[] Encode(string text)
    {
        var result = new List<int>();

        // SentencePiece treats the input as a single string with ▁ replacing spaces
        // The leading space is significant: "Hello world" → "▁Hello▁world"
        string normalized = "\u2581" + text.Replace(" ", "\u2581");

        // Greedy forward tokenization with longest match
        int pos = 0;
        while (pos < normalized.Length)
        {
            int bestLen = 0;
            int bestId = _unkId;
            float bestScore = float.NegativeInfinity;

            // Try all possible lengths from current position, find best (longest, then highest score)
            for (int len = 1; len <= normalized.Length - pos && len <= 64; len++)
            {
                string candidate = normalized.Substring(pos, len);
                if (_tokenToId.TryGetValue(candidate, out int id))
                {
                    float score = id < _scores.Length ? _scores[id] : 0f;
                    // Prefer longer matches; for same length, prefer higher score
                    if (len > bestLen || (len == bestLen && score > bestScore))
                    {
                        bestLen = len;
                        bestId = id;
                        bestScore = score;
                    }
                }
            }

            if (bestLen > 0)
            {
                result.Add(bestId);
                pos += bestLen;
            }
            else
            {
                // Byte fallback: encode as <0xHH> tokens
                byte[] bytes = System.Text.Encoding.UTF8.GetBytes(normalized.Substring(pos, 1));
                foreach (byte b in bytes)
                {
                    string byteToken = $"<0x{b:X2}>";
                    if (_tokenToId.TryGetValue(byteToken, out int byteId))
                        result.Add(byteId);
                    else if (_unkId >= 0)
                        result.Add(_unkId);
                }
                pos++;
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Decode token IDs back to text.
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        var sb = new System.Text.StringBuilder();
        foreach (int id in tokenIds)
        {
            if (id < 0 || id >= _vocab.Length) continue;
            int tokenType = id < _tokenTypes.Length ? _tokenTypes[id] : 0;
            if (tokenType == 2) continue; // skip control tokens (BOS, EOS)

            string token = _vocab[id];
            // Handle byte tokens: <0xHH>
            if (token.StartsWith("<0x") && token.EndsWith(">") && token.Length == 6)
            {
                if (byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte b))
                    sb.Append((char)b);
            }
            else
            {
                sb.Append(token);
            }
        }
        // Replace ▁ with space and trim leading space
        string result = sb.ToString().Replace('\u2581', ' ');
        if (result.StartsWith(' ')) result = result[1..];
        return result;
    }

    /// <summary>
    /// Create from GGUF model metadata.
    /// </summary>
    public static SentencePieceTokenizer? FromGGUF(GGUF.GGUFModel model)
    {
        var tokens = model.GetMetadataStringArray("tokenizer.ggml.tokens");
        if (tokens == null || tokens.Length == 0) return null;

        var scores = model.GetMetadataFloatArray("tokenizer.ggml.scores") ?? new float[tokens.Length];

        // Token types stored as object[] of ints in GGUF metadata
        int[]? tokenTypes = null;
        if (model.Metadata.TryGetValue("tokenizer.ggml.token_type", out var ttObj))
        {
            if (ttObj is int[] iarr) tokenTypes = iarr;
            else if (ttObj is object[] oarr) tokenTypes = oarr.Select(o => Convert.ToInt32(o)).ToArray();
        }

        return new SentencePieceTokenizer(tokens, scores, tokenTypes);
    }

    /// <summary>
    /// Create a LoadedTokenizer wrapper for use with InferenceSession.
    /// </summary>
    public LoadedTokenizer ToLoadedTokenizer()
    {
        var specialTokens = new Dictionary<string, int>();
        if (_bosId >= 0) specialTokens["<s>"] = _bosId;
        if (_eosId >= 0) specialTokens["</s>"] = _eosId;
        if (_unkId >= 0) specialTokens["<unk>"] = _unkId;

        return new LoadedTokenizer
        {
            Tokenizer = this,
            VocabSize = VocabSize,
            PadTokenId = 0,
            EosTokenId = _eosId,
            BosTokenId = _bosId,
            SpecialTokens = specialTokens,
        };
    }
}
