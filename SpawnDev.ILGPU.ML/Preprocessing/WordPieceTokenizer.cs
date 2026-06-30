using System.Globalization;
using System.Text;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// WordPiece tokenizer for the BERT family (BERT, DistilBERT, MobileBERT, ELECTRA, etc.).
/// Faithful port of the HuggingFace reference algorithm: a BasicTokenizer pass
/// (clean text → space around CJK → optional lowercase + accent strip → whitespace
/// and punctuation splitting) followed by greedy longest-match-first WordPiece
/// subword splitting with a "##" continuation prefix and an [UNK] fallback.
///
/// Encode() returns the raw WordPiece ids WITHOUT [CLS]/[SEP]; the post-processing
/// (adding [CLS] at the front and [SEP] at the end, padding, attention mask) is the
/// caller's job — see <see cref="LoadedTokenizer.EncodeForModel"/>, which adds them
/// via BosTokenId/EosTokenId. This matches how BPETokenizer/SentencePieceTokenizer behave.
/// </summary>
public class WordPieceTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _ids;

    private readonly bool _doLowerCase;
    private readonly bool _stripAccents;
    private readonly bool _tokenizeChineseChars;
    private readonly string _unkToken;
    private readonly string _continuingSubwordPrefix;
    private readonly int _maxInputCharsPerWord;
    private readonly int _unkId;

    /// <summary>Size of the vocabulary.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Create a WordPiece tokenizer.
    /// </summary>
    /// <param name="vocab">Token-to-id mapping (from tokenizer.json model.vocab or vocab.txt order).</param>
    /// <param name="doLowerCase">Lowercase input before tokenizing (true for *-uncased models).</param>
    /// <param name="stripAccents">
    /// Strip accents (NFD then drop combining marks). HuggingFace defaults this to
    /// <paramref name="doLowerCase"/> when unspecified; pass null to follow that default.
    /// </param>
    /// <param name="tokenizeChineseChars">Add whitespace around CJK characters (BERT default true).</param>
    /// <param name="unkToken">Unknown token (default "[UNK]").</param>
    /// <param name="continuingSubwordPrefix">Continuation prefix (default "##").</param>
    /// <param name="maxInputCharsPerWord">Words longer than this map straight to [UNK] (default 100).</param>
    public WordPieceTokenizer(
        Dictionary<string, int> vocab,
        bool doLowerCase = true,
        bool? stripAccents = null,
        bool tokenizeChineseChars = true,
        string unkToken = "[UNK]",
        string continuingSubwordPrefix = "##",
        int maxInputCharsPerWord = 100)
    {
        _vocab = vocab;
        _ids = new Dictionary<int, string>(vocab.Count);
        foreach (var kv in vocab) _ids[kv.Value] = kv.Key; // first-write-wins == lowest id for dup tokens
        _doLowerCase = doLowerCase;
        _stripAccents = stripAccents ?? doLowerCase;
        _tokenizeChineseChars = tokenizeChineseChars;
        _unkToken = unkToken;
        _continuingSubwordPrefix = continuingSubwordPrefix;
        _maxInputCharsPerWord = maxInputCharsPerWord;
        _unkId = vocab.TryGetValue(unkToken, out var uid) ? uid : 0;
    }

    /// <summary>Encode text to WordPiece token ids (no [CLS]/[SEP]).</summary>
    public int[] Encode(string text)
    {
        var ids = new List<int>();
        foreach (var word in BasicTokenize(text))
            foreach (var sub in WordPieceSplit(word))
                ids.Add(_vocab.TryGetValue(sub, out var id) ? id : _unkId);
        return ids.ToArray();
    }

    /// <summary>
    /// Decode token ids back to text. WordPiece decode joins tokens with spaces and
    /// re-attaches continuation pieces (" ##xyz" → "xyz").
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        var sb = new StringBuilder();
        bool first = true;
        foreach (var id in tokenIds)
        {
            if (!_ids.TryGetValue(id, out var tok)) continue;
            if (tok.StartsWith(_continuingSubwordPrefix, StringComparison.Ordinal))
            {
                sb.Append(tok.AsSpan(_continuingSubwordPrefix.Length));
            }
            else
            {
                if (!first) sb.Append(' ');
                sb.Append(tok);
            }
            first = false;
        }
        return sb.ToString();
    }

    // ── BasicTokenizer (HF transformers BasicTokenizer) ──
    // clean text → space around CJK → whitespace split → (lowercase + strip accents) → split on punctuation
    private List<string> BasicTokenize(string text)
    {
        text = CleanText(text);
        if (_tokenizeChineseChars) text = TokenizeChineseChars(text);

        var output = new List<string>();
        foreach (var token in WhitespaceSplit(text))
        {
            var t = token;
            if (_doLowerCase) t = t.ToLowerInvariant();
            if (_stripAccents) t = StripAccents(t);
            output.AddRange(SplitOnPunctuation(t));
        }
        return output;
    }

    // Greedy longest-match-first WordPiece over a single basic token.
    private List<string> WordPieceSplit(string token)
    {
        var sub = new List<string>();
        if (token.Length == 0) return sub;
        if (token.Length > _maxInputCharsPerWord) { sub.Add(_unkToken); return sub; }

        int start = 0;
        bool bad = false;
        while (start < token.Length)
        {
            int end = token.Length;
            string? cur = null;
            while (start < end)
            {
                var piece = token.Substring(start, end - start);
                if (start > 0) piece = _continuingSubwordPrefix + piece;
                if (_vocab.ContainsKey(piece)) { cur = piece; break; }
                end--;
            }
            if (cur == null) { bad = true; break; }
            sub.Add(cur);
            start = end;
        }

        if (bad) { sub.Clear(); sub.Add(_unkToken); }
        return sub;
    }

    private static string CleanText(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            if (c == '\0' || c == '�' || IsControl(c)) continue;
            sb.Append(IsWhitespace(c) ? ' ' : c);
        }
        return sb.ToString();
    }

    private static string TokenizeChineseChars(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            if (IsChineseChar(c)) { sb.Append(' ').Append(c).Append(' '); }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    private static string StripAccents(string text)
    {
        var nfd = text.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(nfd.Length);
        foreach (var c in nfd)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.NonSpacingMark) continue;
            sb.Append(c);
        }
        return sb.ToString();
    }

    private static IEnumerable<string> WhitespaceSplit(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    private static List<string> SplitOnPunctuation(string text)
    {
        var output = new List<string>();
        var cur = new StringBuilder();
        foreach (var c in text)
        {
            if (IsPunctuation(c))
            {
                if (cur.Length > 0) { output.Add(cur.ToString()); cur.Clear(); }
                output.Add(c.ToString());
            }
            else cur.Append(c);
        }
        if (cur.Length > 0) output.Add(cur.ToString());
        return output;
    }

    // ── BERT char classifiers (HF transformers reference) ──
    private static bool IsWhitespace(char c)
    {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return true;
        return CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.SpaceSeparator;
    }

    private static bool IsControl(char c)
    {
        if (c == '\t' || c == '\n' || c == '\r') return false; // treated as whitespace, not control
        var cat = CharUnicodeInfo.GetUnicodeCategory(c);
        return cat == UnicodeCategory.Control || cat == UnicodeCategory.Format;
    }

    private static bool IsPunctuation(char c)
    {
        // BERT treats all non-alphanumeric ASCII as punctuation, plus all Unicode P* categories.
        int cp = c;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
        var cat = CharUnicodeInfo.GetUnicodeCategory(c);
        return cat is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
            or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsChineseChar(char c)
    {
        int cp = c; // BMP coverage (char is UTF-16; supplementary CJK handled by surrogate pairs, rare for BERT)
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
}
