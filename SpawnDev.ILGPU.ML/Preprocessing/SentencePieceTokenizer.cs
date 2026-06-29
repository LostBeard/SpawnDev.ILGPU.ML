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
    private readonly bool _byteLevelBpe;
    // Byte-level BPE merge ranks ("A B" → rank). Lower rank = higher priority. Null when the GGUF carries no
    // merges (then the byte-level path falls back to score-greedy longest-match). Real GPT-2/qwen/llama3/smollm
    // tokenization is rank-ordered BPE, NOT greedy — greedy diverges on multi-token words (e.g. "assistant"),
    // which malforms the prompt for sensitive small models. See the byte-level Encode path.
    private readonly Dictionary<(string, string), int>? _bpeRanks;

    /// <summary>True for byte-level BPE vocabs (GGUF <c>tokenizer.ggml.model == "gpt2"</c>: qwen2/3, llama3,
    /// most modern models). Such vocabs encode raw bytes as printable unicode via GPT-2's bytes↔unicode map
    /// (space→Ġ, newline→Ċ, …) rather than SentencePiece's ▁ + &lt;0xHH&gt; scheme — decoding must reverse it.</summary>
    public bool ByteLevelBpe => _byteLevelBpe;

    // GPT-2 byte-level maps (the standard bytes_to_unicode), built once. Forward: raw byte → vocab-string char
    // (for ENCODE); reverse: char → raw byte (for DECODE).
    private static readonly char[] Gpt2ByteToChar = new char[256];
    private static readonly Dictionary<char, byte> Gpt2CharToByte = BuildGpt2Tables();
    private static Dictionary<char, byte> BuildGpt2Tables()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);        // 33..126
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);      // 161..172
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);      // 174..255
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
        var map = new Dictionary<char, byte>(256);
        for (int i = 0; i < bs.Count; i++) { map[(char)cs[i]] = (byte)bs[i]; Gpt2ByteToChar[bs[i]] = (char)cs[i]; }
        return map;
    }

    /// <summary>Vocabulary size.</summary>
    public int VocabSize => _vocab.Length;

    /// <summary>BOS token ID.</summary>
    public int BosId => _bosId;

    /// <summary>EOS token ID.</summary>
    public int EosId => _eosId;

    /// <summary>Look up the exact vocab id of a token string (e.g. a control token like
    /// "&lt;|turn&gt;"). Returns false if the string is not a single vocab entry. Used by chat
    /// templates that must emit control tokens as SINGLE ids rather than rely on greedy
    /// sub-word matching of their literal text.</summary>
    public bool TryGetId(string token, out int id) => _tokenToId.TryGetValue(token, out id);

    /// <summary>
    /// Create a SentencePiece tokenizer from GGUF metadata arrays.
    /// </summary>
    public SentencePieceTokenizer(string[] tokens, float[] scores, int[]? tokenTypes = null, bool byteLevelBpe = false,
        string[]? merges = null)
    {
        _vocab = tokens;
        _scores = scores;
        _byteLevelBpe = byteLevelBpe;
        // Build the BPE merge-rank table for the byte-level path. Each merge is "A B" (the two symbol strings to
        // join), listed in priority order — index = rank.
        if (byteLevelBpe && merges is { Length: > 0 })
        {
            _bpeRanks = new Dictionary<(string, string), int>(merges.Length);
            for (int i = 0; i < merges.Length; i++)
            {
                int sp = merges[i].IndexOf(' ');
                if (sp <= 0 || sp >= merges[i].Length - 1) continue;
                var pair = (merges[i].Substring(0, sp), merges[i].Substring(sp + 1));
                _bpeRanks.TryAdd(pair, i);
            }
        }
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
        string normalized;
        if (_byteLevelBpe)
        {
            // Byte-level BPE: map each UTF-8 byte of the input to its GPT-2 char, then greedy-match the vocab
            // (stored in GPT-2-char form: space=\u0120, newline=\u010a, \u2026). No leading \u2581, no <0xHH> fallback.
            var raw = System.Text.Encoding.UTF8.GetBytes(text);
            var sb = new System.Text.StringBuilder(raw.Length);
            foreach (var rb in raw) sb.Append(Gpt2ByteToChar[rb]);
            normalized = sb.ToString();
            // Proper rank-ordered BPE (what GPT-2/qwen/llama3/smollm actually use) when merges are present.
            // Greedy longest-match (below) only coincides with BPE for some vocabs and diverges on multi-token
            // words \u2014 fixing that is the whole point of loading the merges.
            if (_bpeRanks != null) return EncodeByteLevelBpe(normalized);
        }
        else
        {
            normalized = "\u2581" + text.Replace(" ", "\u2581");
        }

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
    /// Byte-level BPE encode (GPT-2 / qwen / llama3 / smollm). Operates on the already byte-mapped string:
    /// start from single-character symbols, then repeatedly merge the adjacent pair with the LOWEST merge rank
    /// (highest training priority) until no ranked pair remains — the canonical GPT-2 <c>bpe()</c> algorithm.
    /// This matches how the model was trained, unlike score-greedy longest-match which diverges on multi-token
    /// words (e.g. "assistant" → "assis"+"tan"+"t") and malforms the prompt for sensitive models. Special/control
    /// tokens are emitted by the chat-template layer as ids (not through here), so they need no special handling.
    /// </summary>
    private int[] EncodeByteLevelBpe(string normalized)
    {
        if (normalized.Length == 0) return Array.Empty<int>();
        var word = new List<string>(normalized.Length);
        foreach (var ch in normalized) word.Add(ch.ToString());

        while (word.Count > 1)
        {
            // Lowest-rank adjacent pair across the current symbol list.
            int bestRank = int.MaxValue;
            string? bf = null, bs = null;
            for (int i = 0; i < word.Count - 1; i++)
                if (_bpeRanks!.TryGetValue((word[i], word[i + 1]), out int r) && r < bestRank)
                { bestRank = r; bf = word[i]; bs = word[i + 1]; }
            if (bf == null) break;

            // Merge every non-overlapping occurrence of that pair, then re-scan (canonical GPT-2 bpe).
            var merged = new List<string>(word.Count);
            int k = 0;
            while (k < word.Count)
            {
                if (k < word.Count - 1 && word[k] == bf && word[k + 1] == bs) { merged.Add(bf + bs); k += 2; }
                else { merged.Add(word[k]); k++; }
            }
            word = merged;
        }

        var result = new List<int>(word.Count);
        foreach (var sym in word)
        {
            if (_tokenToId.TryGetValue(sym, out int id)) result.Add(id);
            else foreach (var ch in sym) // unmergeable leftover → emit its single byte-char token(s)
                {
                    if (_tokenToId.TryGetValue(ch.ToString(), out int cid)) result.Add(cid);
                    else if (_unkId >= 0) result.Add(_unkId);
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
    /// The raw bytes a single token contributes to the decoded output stream \u2014 the building block for
    /// incremental/streaming detokenization. Control tokens (type 2) and out-of-range ids contribute
    /// nothing. Byte-fallback tokens (&lt;0xHH&gt;) contribute their single raw byte. Normal pieces
    /// contribute the UTF-8 bytes of the piece with the SentencePiece word-start marker U+2581 (\u2581)
    /// mapped to a space. Concatenating this over a token run and decoding the result as UTF-8 is the
    /// correct, multi-byte-safe detokenization (unlike <see cref="Decode"/>'s per-byte char append).
    /// </summary>
    public byte[] TokenToBytes(int id)
    {
        if (id < 0 || id >= _vocab.Length) return Array.Empty<byte>();
        int tokenType = id < _tokenTypes.Length ? _tokenTypes[id] : 0;
        if (tokenType == 2) return Array.Empty<byte>(); // control token (BOS/EOS/turn markers)

        string token = _vocab[id];

        if (_byteLevelBpe)
        {
            // Byte-level BPE: each char of the vocab string maps back to ONE raw byte via the GPT-2 table
            // (\u0120\u21920x20, \u010a\u21920x0A, \u2026). Accumulating these and decoding as UTF-8 (the streaming decoder) yields
            // the correct text, incl. multi-byte glyphs encoded as a run of mapped bytes.
            var bytes = new byte[token.Length];
            int k = 0;
            foreach (char c in token)
                if (Gpt2CharToByte.TryGetValue(c, out var bb)) bytes[k++] = bb;
            return k == bytes.Length ? bytes : bytes[..k];
        }

        // Byte-fallback token <0xHH> \u2192 the single raw byte (same detection as Decode).
        if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith(">")
            && byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte b))
            return new[] { b };

        // Normal SentencePiece piece: UTF-8 bytes with \u2581 \u2192 space.
        return System.Text.Encoding.UTF8.GetBytes(token.Replace('\u2581', ' '));
    }

    /// <summary>
    /// Create a stateful, UTF-8-safe streaming detokenizer. Feed generated token ids one at a time via
    /// <see cref="SentencePieceStreamingDecoder.Push"/>; it returns the incremental text delta, holding
    /// back an incomplete trailing multi-byte UTF-8 sequence until the bytes completing it arrive.
    /// </summary>
    public SentencePieceStreamingDecoder CreateStreamingDecoder() => new SentencePieceStreamingDecoder(this);

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

        bool byteLevelBpe = model.GetMetadataString("tokenizer.ggml.model") == "gpt2";
        // Byte-level BPE vocabs carry the rank-ordered merge list — required to tokenize the way the model was
        // trained (greedy longest-match diverges on multi-token words and breaks sensitive models).
        var merges = byteLevelBpe ? model.GetMetadataStringArray("tokenizer.ggml.merges") : null;
        return new SentencePieceTokenizer(tokens, scores, tokenTypes, byteLevelBpe, merges);
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
