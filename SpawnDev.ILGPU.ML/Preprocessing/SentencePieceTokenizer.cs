namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// SentencePiece BPE tokenizer for LLaMA/Mistral/Gemma models (and byte-level BPE for GPT-2-style vocabs).
/// SentencePiece BPE (sentencepiece's bpe_model): spaces become U+2581 (▁), the text is split into characters (user-defined pieces kept whole and never
/// merged), then the adjacent pair whose merged piece has the HIGHEST score (a normal or user-defined piece; the leftmost on ties) is merged, repeatedly,
/// until no pair forms a piece; a symbol left without a piece becomes byte pieces &lt;0xHH&gt; (byte fallback) or the unknown id.
/// MEASURED 2026-11-17 (SpawnGen, LTX-2's Gemma 3 prompt): the greedy longest match this replaced split "sways" as "▁sway" + "s" where SentencePiece
/// gives "▁s" + "ways"; this algorithm gives all 116 of ComfyUI's ids. Unigram models (T5, umT5) are <see cref="UnigramTokenizer"/>.
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
    private int _bosId;
    private int _eosId;
    private int _unkId;

    // token types, the GGUF / SentencePiece enum (llama.cpp's llama_token_type; sentencepiece_model.proto's Type)
    /// <summary>GGUF <c>tokenizer.ggml.token_type</c>: an ordinary piece.</summary>
    public const int TypeNormal = 1;
    /// <summary>GGUF token type: the unknown token.</summary>
    public const int TypeUnknown = 2;
    /// <summary>GGUF token type: a control token (BOS, EOS, turn markers): no text when decoded.</summary>
    public const int TypeControl = 3;
    /// <summary>GGUF token type: a user-defined special piece (qwen's &lt;think&gt;, gemma4's &lt;|tool_call&gt;): decoded as its text.</summary>
    public const int TypeUserDefined = 4;
    /// <summary>GGUF token type: an unused slot.</summary>
    public const int TypeUnused = 5;
    /// <summary>GGUF token type: a byte-fallback piece (&lt;0xHH&gt;).</summary>
    public const int TypeByte = 6;
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

    /// <summary>
    /// SentencePiece "dummy prefix": prepend ▁ to every encoded string (GGUF <c>tokenizer.ggml.add_space_prefix</c>; true for llama-style
    /// vocabs, FALSE for gemma 2/3). Chat templates encode each segment separately, so a dummy prefix that the model does not use puts a
    /// stray space token before every "\n" after a turn marker. MEASURED on gemma-3-270m-it: three such tokens in a two-turn prompt, and
    /// the second answer ignored the question ("What is the capital of France?" -> "Hello! I'm happy to help..."; Ollama: "Paris").
    /// </summary>
    public bool AddSpacePrefix { get; set; } = true;

    /// <summary>
    /// SentencePiece's NormalizerSpec.remove_extra_whitespaces: leading, trailing and repeated spaces are removed before encoding. False for GGUF vocabs
    /// (llama.cpp does not apply it); <see cref="FromSentencePieceModel"/> sets it from the model (Gemma 3: false).
    /// </summary>
    public bool RemoveExtraWhitespaces { get; set; }

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

        // Find special token IDs by type and content (FromGGUF then takes the GGUF's own bos / eos / unknown ids). Until 2026-11-02 this read
        // type 2 as control and type 1 as unknown (the enum is 1 normal, 2 unknown, 3 control): no control token was found, the unknown id was the
        // last ordinary piece, and BOS / EOS fell back to ids 1 / 2, which are '"' and '#' in llama 3's and qwen's vocabularies (GgufGenerator
        // stopped at a '#') and <eos> / <bos> swapped in gemma's.
        _bosId = -1; _eosId = -1; _unkId = -1;
        for (int i = 0; i < _tokenTypes.Length; i++)
        {
            if (_tokenTypes[i] == TypeControl)
            {
                var t = tokens[i];
                if (t is "<s>" or "<bos>" or "<|begin_of_text|>") _bosId = i;
                else if (t is "</s>" or "<eos>" or "<|end_of_text|>") _eosId = i;
            }
            if (_tokenTypes[i] == TypeUnknown) _unkId = i;
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
            if (RemoveExtraWhitespaces) text = string.Join(' ', text.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            normalized = (AddSpacePrefix ? "\u2581" : "") + text.Replace(" ", "\u2581");
            return EncodeSentencePieceBpe(normalized);
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

    private Dictionary<char, string[]>? _userDefined;  // user-defined pieces by first char, longest first (kept whole, never merged)

    private bool Mergeable(int id)
    {
        int t = id < _tokenTypes.Length ? _tokenTypes[id] : 0;
        return t is 0 or TypeNormal or TypeUserDefined;                // sentencepiece: pieces_ holds normal / user-defined / unused, and BPE skips unused
    }

    /// <summary>SentencePiece BPE on the escaped text (see the class summary): priority queue of adjacent pairs by (score desc, left asc), linked symbols.</summary>
    private int[] EncodeSentencePieceBpe(string s)
    {
        if (s.Length == 0) return Array.Empty<int>();
        _userDefined ??= Enumerable.Range(0, _vocab.Length).Where(i => i < _tokenTypes.Length && _tokenTypes[i] == TypeUserDefined && _vocab[i].Length > 0)
            .Select(i => _vocab[i]).GroupBy(v => v[0]).ToDictionary(g => g.Key, g => g.OrderByDescending(v => v.Length).ToArray());
        var start = new List<int>(s.Length); var len = new List<int>(s.Length); var frozen = new List<bool>(s.Length);
        for (int pos = 0; pos < s.Length;)
        {
            int ud = 0;
            if (_userDefined.TryGetValue(s[pos], out var candidates))
                foreach (var u in candidates)
                    if (u.Length <= s.Length - pos && string.CompareOrdinal(s, pos, u, 0, u.Length) == 0) { ud = u.Length; break; }
            if (ud > 0) { start.Add(pos); len.Add(ud); frozen.Add(true); pos += ud; continue; }
            int cl = char.IsHighSurrogate(s[pos]) && pos + 1 < s.Length && char.IsLowSurrogate(s[pos + 1]) ? 2 : 1;
            start.Add(pos); len.Add(cl); frozen.Add(false); pos += cl;
        }
        int n = start.Count;
        var prev = new int[n]; var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
        var queue = new PriorityQueue<(int L, int R, int Size), (float Score, int L)>(Comparer<(float Score, int L)>.Create((a, b) => a.Score != b.Score ? b.Score.CompareTo(a.Score) : a.L.CompareTo(b.L)));
        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0 || frozen[l] || frozen[r]) return;
            int size = len[l] + len[r];
            if (_tokenToId.TryGetValue(s.Substring(start[l], size), out int id) && Mergeable(id))
                queue.Enqueue((l, r, size), (id < _scores.Length ? _scores[id] : 0f, l));
        }
        for (int i = 0; i + 1 < n; i++) TryAdd(i, i + 1);
        while (queue.TryDequeue(out var top, out _))
        {
            var (l, r, size) = top;
            if (len[l] == 0 || len[r] == 0 || next[l] != r || len[l] + len[r] != size) continue;   // stale: a neighbour changed since it was queued
            len[l] = size; len[r] = 0;
            next[l] = next[r]; if (next[r] >= 0) prev[next[r]] = l;
            TryAdd(prev[l], l); TryAdd(l, next[l]);
        }
        var result = new List<int>(n);
        for (int i = 0; i >= 0; i = next[i])
        {
            string piece = s.Substring(start[i], len[i]);
            if (_tokenToId.TryGetValue(piece, out int id) && (id >= _tokenTypes.Length || _tokenTypes[id] != TypeUnused)) { result.Add(id); continue; }
            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(piece))
            {
                if (_tokenToId.TryGetValue($"<0x{b:X2}>", out int byteId)) result.Add(byteId);
                else if (_unkId >= 0) result.Add(_unkId);
            }
        }
        return result.ToArray();
    }

    /// <summary>
    /// Read a SentencePiece BPE model file (<c>tokenizer.model</c>, a serialized <c>ModelProto</c>; ComfyUI's Gemma text-encoder files carry one as the U8 tensor
    /// <c>spiece_model</c>): pieces, scores and types, the special ids (trainer_spec unk / bos / eos), and the normalizer's add_dummy_prefix and
    /// remove_extra_whitespaces (defaults true, as in the proto). Unigram models are <see cref="UnigramTokenizer.FromSentencePieceModel"/>; a normalizer other than
    /// "identity" with a character map is refused (its rules are not applied here).
    /// </summary>
    public static SentencePieceTokenizer FromSentencePieceModel(ReadOnlySpan<byte> model)
    {
        var tokens = new List<string>(); var scores = new List<float>(); var types = new List<int>();
        int modelType = 1, unk = -1, bos = -1, eos = -1; bool dummyPrefix = true, removeExtra = true; string normName = ""; int charsmap = 0;
        int p = 0;
        while (p < model.Length)
        {
            ulong key = SpVarint(model, ref p); int field = (int)(key >> 3), wire = (int)(key & 7);
            if (wire != 2) { SpSkip(model, ref p, wire); continue; }
            var msg = SpBytes(model, ref p);
            int q = 0;
            if (field == 1)
            {
                string piece = ""; float score = 0f; int type = TypeNormal;
                while (q < msg.Length)
                {
                    ulong k = SpVarint(msg, ref q); int f = (int)(k >> 3), w = (int)(k & 7);
                    if (f == 1 && w == 2) piece = System.Text.Encoding.UTF8.GetString(SpBytes(msg, ref q));
                    else if (f == 2 && w == 5) { score = BitConverter.ToSingle(msg.Slice(q, 4)); q += 4; }
                    else if (f == 3 && w == 0) type = (int)SpVarint(msg, ref q);
                    else SpSkip(msg, ref q, w);
                }
                tokens.Add(piece); scores.Add(score); types.Add(type);
            }
            else if (field == 2)
            {
                while (q < msg.Length)
                {
                    ulong k = SpVarint(msg, ref q); int f = (int)(k >> 3), w = (int)(k & 7);
                    if (w == 0 && f is 3 or 40 or 41 or 42)
                    {
                        int v = (int)SpVarint(msg, ref q);
                        if (f == 3) modelType = v; else if (f == 40) unk = v; else if (f == 41) bos = v; else eos = v;
                    }
                    else SpSkip(msg, ref q, w);
                }
            }
            else if (field == 3)
            {
                while (q < msg.Length)
                {
                    ulong k = SpVarint(msg, ref q); int f = (int)(k >> 3), w = (int)(k & 7);
                    if (f == 1 && w == 2) normName = System.Text.Encoding.UTF8.GetString(SpBytes(msg, ref q));
                    else if (f == 2 && w == 2) charsmap = SpBytes(msg, ref q).Length;
                    else if (f == 3 && w == 0) dummyPrefix = SpVarint(msg, ref q) != 0;
                    else if (f == 4 && w == 0) removeExtra = SpVarint(msg, ref q) != 0;
                    else SpSkip(msg, ref q, w);
                }
            }
        }
        if (tokens.Count == 0) throw new InvalidDataException("SentencePiece model: no pieces");
        if (modelType != 2) throw new NotSupportedException($"SentencePiece model type {modelType} (1 = Unigram: use UnigramTokenizer.FromSentencePieceModel); this tokenizer is BPE (2)");
        if (charsmap > 0 && normName != "identity") throw new NotSupportedException($"SentencePiece normalizer '{normName}' with a character map: its rules are not applied here");
        var tok = new SentencePieceTokenizer(tokens.ToArray(), scores.ToArray(), types.ToArray()) { AddSpacePrefix = dummyPrefix, RemoveExtraWhitespaces = removeExtra };
        if (unk >= 0) tok._unkId = unk;
        if (bos >= 0) tok._bosId = bos;
        if (eos >= 0) tok._eosId = eos;
        return tok;
    }

    private static ulong SpVarint(ReadOnlySpan<byte> b, ref int p)
    {
        ulong r = 0; int sh = 0;
        while (true)
        {
            if (p >= b.Length || sh > 63) throw new InvalidDataException("SentencePiece model: a truncated varint");
            byte x = b[p++]; r |= (ulong)(x & 0x7F) << sh; sh += 7;
            if (x < 0x80) return r;
        }
    }
    private static ReadOnlySpan<byte> SpBytes(ReadOnlySpan<byte> b, ref int p)
    {
        ulong n = SpVarint(b, ref p);
        if (n > (ulong)(b.Length - p)) throw new InvalidDataException("SentencePiece model: a field runs past the end");
        var o = b.Slice(p, (int)n); p += (int)n; return o;
    }
    private static void SpSkip(ReadOnlySpan<byte> b, ref int p, int wire)
    {
        switch (wire)
        {
            case 0: SpVarint(b, ref p); break;
            case 1: p += 8; break;
            case 2: SpBytes(b, ref p); break;
            case 5: p += 4; break;
            default: throw new InvalidDataException($"SentencePiece model: wire type {wire}");
        }
        if (p > b.Length) throw new InvalidDataException("SentencePiece model: a field runs past the end");
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
    /// Decode token IDs back to text: each token's bytes (<see cref="TokenToBytes"/>) concatenated and read as UTF-8. A SentencePiece vocab drops
    /// the leading space of its word-start marker; a byte-level BPE vocab keeps the text exactly.
    /// Until 2026-11-02 this appended the vocab strings as they are: a byte-level BPE vocab (llama 3, qwen, smollm) came back in GPT-2's byte
    /// alphabet ("TheĠthreeĠprimary...Ċ") and a byte-fallback run (<c>&lt;0xHH&gt;</c>) came back as one char per byte, not as its UTF-8 glyph.
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        var bytes = new List<byte>();
        foreach (int id in tokenIds) bytes.AddRange(TokenToBytes(id));
        string result = System.Text.Encoding.UTF8.GetString(bytes.ToArray());
        if (!_byteLevelBpe && result.StartsWith(' ')) result = result[1..];
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
        if (tokenType == TypeControl) return Array.Empty<byte>(); // control token (BOS/EOS/turn markers): no text, as llama.cpp renders it

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
        var tok = new SentencePieceTokenizer(tokens, scores, tokenTypes, byteLevelBpe, merges);
        // the GGUF's own special ids win over the guesses by name (llama 3.1: eos = <|eot_id|>, not <|end_of_text|>; gemma: bos 2, eos 1)
        int Special(string key, int current) =>
            model.Metadata.TryGetValue(key, out var v) && v is not null && Convert.ToInt64(v) is long id && id >= 0 && id < tokens.Length ? (int)id : current;
        tok._bosId = Special("tokenizer.ggml.bos_token_id", tok._bosId);
        tok._eosId = Special("tokenizer.ggml.eos_token_id", tok._eosId);
        tok._unkId = Special("tokenizer.ggml.unknown_token_id", tok._unkId);
        if (model.Metadata.TryGetValue("tokenizer.ggml.add_space_prefix", out var asp) && asp is bool addSpace)
            tok.AddSpacePrefix = addSpace;                  // absent = the llama SentencePiece default (true)
        return tok;
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
