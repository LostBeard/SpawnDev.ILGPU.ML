using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// SentencePiece UNIGRAM tokenizer as the Hugging Face <c>tokenizers</c> library runs it for <c>google/umt5-xxl</c> (the text encoder of Wan 2.2): the normalizer collapses runs of
/// spaces, the Metaspace pre-tokenizer replaces spaces with U+2581 and prefixes one, the Unigram model picks the segmentation with the highest total log-probability (Viterbi over
/// the lattice of every vocabulary piece that matches at every position; a character no piece covers becomes the unknown token with the score of the lowest piece minus 10, and
/// consecutive unknowns fuse), and the post-processor appends the end-of-sequence token. Loaded from <c>tokenizer.json</c>; no byte fallback.
/// (The greedy <see cref="SentencePieceTokenizer"/> is for the llama / gemma vocabularies and is not a Unigram implementation.)
/// </summary>
public sealed class UnigramTokenizer
{
    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly Dictionary<string, int> _ids;
    private readonly int _unk, _eos;
    private readonly int _maxPieceChars;
    private readonly float _unkScore;
    private static readonly Regex Spaces = new(" {2,}", RegexOptions.Compiled);

    public int VocabSize => _pieces.Length;
    public int PadId { get; } = 0;
    public int EosId => _eos;
    public int UnkId => _unk;

    private UnigramTokenizer(string[] pieces, float[] scores, int unk, int eos)
    {
        _pieces = pieces; _scores = scores; _unk = unk; _eos = eos;
        _ids = new Dictionary<string, int>(pieces.Length);
        float min = float.MaxValue;
        for (int i = 0; i < pieces.Length; i++)
        {
            _ids.TryAdd(pieces[i], i);
            _maxPieceChars = Math.Max(_maxPieceChars, pieces[i].Length);
            if (i != unk && i >= 4) min = Math.Min(min, scores[i]);          // the lowest score of the real pieces (the special tokens carry 0)
        }
        _unkScore = min - 10f;
    }

    /// <summary>Read <c>model.vocab</c> ([piece, score] pairs), <c>model.unk_id</c> and the post-processor's end-of-sequence id from a Hugging Face tokenizer.json.</summary>
    public static UnigramTokenizer FromTokenizerJson(Stream json)
    {
        using var doc = JsonDocument.Parse(json);
        var model = doc.RootElement.GetProperty("model");
        if (model.GetProperty("type").GetString() != "Unigram") throw new NotSupportedException("a Unigram tokenizer.json");
        var vocab = model.GetProperty("vocab");
        int n = vocab.GetArrayLength();
        var pieces = new string[n]; var scores = new float[n];
        int i = 0;
        foreach (var e in vocab.EnumerateArray()) { pieces[i] = e[0].GetString()!; scores[i] = (float)e[1].GetDouble(); i++; }
        int unk = model.TryGetProperty("unk_id", out var u) && u.ValueKind == JsonValueKind.Number ? u.GetInt32() : 3;
        int eos = 1;
        if (doc.RootElement.TryGetProperty("post_processor", out var pp) && pp.ValueKind == JsonValueKind.Object && pp.TryGetProperty("special_tokens", out var st)
            && st.TryGetProperty("</s>", out var s) && s.TryGetProperty("ids", out var ids) && ids.GetArrayLength() > 0) eos = ids[0].GetInt32();
        return new UnigramTokenizer(pieces, scores, unk, eos);
    }

    /// <summary>Token ids of <paramref name="text"/> (with the trailing end-of-sequence token when <paramref name="addEos"/>). The text is used as given, apart from the tokenizer's own normalizer.</summary>
    public int[] Encode(string text, bool addEos = true)
    {
        text = Spaces.Replace(text, " ").Replace(' ', '▁');
        if (text.Length == 0 || text[0] != '▁') text = "▁" + text;         // Metaspace, add_prefix_space
        // Viterbi over the code points (a surrogate pair is one position)
        var runes = new List<int>(text.Length);                                      // char offset of every rune start
        for (int c = 0; c < text.Length; c++) { runes.Add(c); if (char.IsHighSurrogate(text[c]) && c + 1 < text.Length) c++; }
        int n = runes.Count;
        runes.Add(text.Length);
        var best = new float[n + 1]; var from = new int[n + 1]; var pid = new int[n + 1];
        Array.Fill(best, float.NegativeInfinity); best[0] = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNegativeInfinity(best[i])) continue;
            bool single = false;
            for (int j = i + 1; j <= n; j++)
            {
                int len = runes[j] - runes[i];
                if (len > _maxPieceChars) break;
                if (!_ids.TryGetValue(text.Substring(runes[i], len), out int id)) continue;
                if (j == i + 1) single = true;
                float s = best[i] + _scores[id];
                if (s > best[j]) { best[j] = s; from[j] = i; pid[j] = id; }
            }
            if (!single)
            {
                float s = best[i] + _unkScore;                                       // no piece covers this character: the unknown token
                if (s > best[i + 1]) { best[i + 1] = s; from[i + 1] = i; pid[i + 1] = _unk; }
            }
        }
        var rev = new List<int>();
        for (int p = n; p > 0; p = from[p]) rev.Add(pid[p]);
        rev.Reverse();
        var outIds = new List<int>(rev.Count + 1);
        foreach (int id in rev) { if (id == _unk && outIds.Count > 0 && outIds[^1] == _unk) continue; outIds.Add(id); }     // consecutive unknowns fuse
        if (addEos) outIds.Add(_eos);
        return outIds.ToArray();
    }

    /// <summary>
    /// The Wan reference's text cleanup before tokenizing (<c>HuggingfaceTokenizer(clean='whitespace')</c>: <c>ftfy.fix_text</c>, HTML unescape twice, strip, then whitespace runs collapse
    /// to one space). The parts of <c>ftfy.fix_text</c> that matter for prompts are reproduced: full-width ASCII variants (U+FF01..FF5E, the ideographic space) become ordinary
    /// ASCII ("，" -> ","), curly quotes straighten, Latin ligatures expand, line breaks unify, control characters go, and the text is NFC-normalised. Mojibake repair is not implemented.
    /// </summary>
    public static string CleanWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            // code points as numbers: FF01..FF5E full-width ASCII, 3000 ideographic space, 2018..201F curly quotes, FB00..FB06 Latin ligatures, 2028/2029 separators
            if (c >= (char)0xFF01 && c <= (char)0xFF5E) sb.Append((char)(c - 0xFEE0));                       // fix_character_width
            else if (c == (char)0x3000) sb.Append(' ');
            else if (c >= (char)0x2018 && c <= (char)0x201B) sb.Append((char)39);                           // uncurl_quotes: single
            else if (c >= (char)0x201C && c <= (char)0x201F) sb.Append((char)34);                           // double
            else if (c == (char)0xFB00) sb.Append(ASCII(102, 102)); else if (c == (char)0xFB01) sb.Append(ASCII(102, 105)); else if (c == (char)0xFB02) sb.Append(ASCII(102, 108));   // fix_latin_ligatures
            else if (c == (char)0xFB03) sb.Append(ASCII(102, 102, 105)); else if (c == (char)0xFB04) sb.Append(ASCII(102, 102, 108)); else if (c == (char)0xFB05 || c == (char)0xFB06) sb.Append(ASCII(115, 116));
            else if (c == (char)0x2028 || c == (char)0x2029 || c == (char)13) sb.Append((char)10);           // fix_line_breaks
            else if (char.IsControl(c) && c != (char)10 && c != (char)9) { /* remove_control_chars */ }
            else sb.Append(c);
        }
        var fixedText = sb.ToString().Normalize(NormalizationForm.FormC);
        fixedText = System.Net.WebUtility.HtmlDecode(System.Net.WebUtility.HtmlDecode(fixedText)).Trim();
        return Regex.Replace(fixedText, @"\s+", " ").Trim();
    }

    private static string ASCII(params int[] codes) => new string(Array.ConvertAll(codes, v => (char)v));

    public string Piece(int id) => _pieces[id];
}
