using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// SentencePiece UNIGRAM tokenizer as the Hugging Face <c>tokenizers</c> library runs it for <c>google/umt5-xxl</c> (the text encoder of Wan 2.2): the normalizer collapses runs of
/// spaces, the Metaspace pre-tokenizer replaces spaces with U+2581 and prefixes one, the Unigram model picks the segmentation with the highest total log-probability (Viterbi over
/// the lattice of every vocabulary piece that matches at every position; a character no piece covers becomes the unknown token with the score of the lowest piece minus 10, and
/// consecutive unknowns fuse), and the post-processor appends the end-of-sequence token. Loaded from <c>tokenizer.json</c> or a SentencePiece model file; no byte fallback.
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

    /// <summary>
    /// Read a SentencePiece model file (<c>spiece.model</c>, a serialized <c>ModelProto</c>; ComfyUI's umT5 files carry one as the U8 tensor <c>spiece_model</c>): every piece and score in
    /// order, the unknown and end-of-sequence ids from the trainer spec (protobuf defaults 0 and 2 when absent), then <paramref name="extraIds"/> T5 sentinel pieces
    /// <c>&lt;extra_id_{extraIds-1}&gt;</c> down to <c>&lt;extra_id_0&gt;</c> with score 0, as Hugging Face's T5 tokenizers append them (google/umt5-xxl: 300, so its tokenizer.json
    /// has 256,300 pieces to the model's 256,000). Encoding is the same as <see cref="FromTokenizerJson"/>'s: the model's own normalizer spec is not read (umT5's is "identity"),
    /// space runs collapse as tokenizer.json's normalizer does.
    /// </summary>
    public static UnigramTokenizer FromSentencePieceModel(ReadOnlySpan<byte> model, int extraIds = 0)
    {
        var pieces = new List<string>(); var scores = new List<float>();
        int unk = 0, eos = 2;                                                        // TrainerSpec defaults: unk_id = 0, eos_id = 2
        int p = 0;
        while (p < model.Length)
        {
            ulong key = ReadVarint(model, ref p);
            int field = (int)(key >> 3), wire = (int)(key & 7);
            if (field == 1 && wire == 2)                                             // repeated SentencePiece pieces = 1 { string piece = 1; float score = 2; Type type = 3; }
            {
                var msg = ReadBytes(model, ref p);
                string piece = ""; float score = 0;
                int q = 0;
                while (q < msg.Length)
                {
                    ulong k = ReadVarint(msg, ref q);
                    int f = (int)(k >> 3), w = (int)(k & 7);
                    if (f == 1 && w == 2) piece = Encoding.UTF8.GetString(ReadBytes(msg, ref q));
                    else if (f == 2 && w == 5) { score = BitConverter.ToSingle(msg.Slice(q, 4)); q += 4; }
                    else Skip(msg, ref q, w);
                }
                pieces.Add(piece); scores.Add(score);
            }
            else if (field == 2 && wire == 2)                                        // TrainerSpec trainer_spec = 2 { int32 unk_id = 40; int32 eos_id = 42; ... }
            {
                var msg = ReadBytes(model, ref p);
                int q = 0;
                while (q < msg.Length)
                {
                    ulong k = ReadVarint(msg, ref q);
                    int f = (int)(k >> 3), w = (int)(k & 7);
                    if ((f == 40 || f == 42) && w == 0) { int v = (int)ReadVarint(msg, ref q); if (f == 40) unk = v; else eos = v; }
                    else Skip(msg, ref q, w);
                }
            }
            else Skip(model, ref p, wire);
        }
        if (pieces.Count == 0) throw new InvalidDataException("not a SentencePiece model: no pieces");
        if ((uint)unk >= (uint)pieces.Count || (uint)eos >= (uint)pieces.Count) throw new InvalidDataException($"SentencePiece model: unk_id {unk} / eos_id {eos} outside the {pieces.Count} pieces");
        for (int e = extraIds - 1; e >= 0; e--) { pieces.Add($"<extra_id_{e}>"); scores.Add(0f); }
        return new UnigramTokenizer(pieces.ToArray(), scores.ToArray(), unk, eos);
    }

    private static ulong ReadVarint(ReadOnlySpan<byte> b, ref int p)
    {
        ulong r = 0;
        for (int s = 0; s < 64; s += 7)
        {
            if (p >= b.Length) throw new InvalidDataException("SentencePiece model: truncated varint");
            byte x = b[p++];
            r |= (ulong)(x & 0x7F) << s;
            if (x < 0x80) return r;
        }
        throw new InvalidDataException("SentencePiece model: varint longer than 10 bytes");
    }

    private static ReadOnlySpan<byte> ReadBytes(ReadOnlySpan<byte> b, ref int p)
    {
        ulong n = ReadVarint(b, ref p);
        if (n > (ulong)(b.Length - p)) throw new InvalidDataException("SentencePiece model: a field runs past the end");
        var s = b.Slice(p, (int)n); p += (int)n;
        return s;
    }

    private static void Skip(ReadOnlySpan<byte> b, ref int p, int wire)
    {
        switch (wire)
        {
            case 0: ReadVarint(b, ref p); break;
            case 1: p += 8; break;
            case 2: ReadBytes(b, ref p); break;
            case 5: p += 4; break;
            default: throw new InvalidDataException($"SentencePiece model: wire type {wire}");
        }
        if (p > b.Length) throw new InvalidDataException("SentencePiece model: a field runs past the end");
    }

    /// <summary>The log-probability of piece <paramref name="id"/>.</summary>
    public float Score(int id) => _scores[id];

    /// <summary>Token ids of <paramref name="text"/> (with the trailing end-of-sequence token when <paramref name="addEos"/>). The text is used as given, apart from the tokenizer's own normalizer.</summary>
    public int[] Encode(string text, bool addEos = true)
    {
        // WhitespaceSplit (tokenizer.json) / remove_extra_whitespaces (SentencePiece): leading and trailing spaces go, so an empty or all-space text has no pieces, only the end token
        text = Spaces.Replace(text, " ").Trim(' ');
        if (text.Length == 0) return addEos ? new[] { _eos } : Array.Empty<int>();
        text = "▁" + text.Replace(' ', '▁');                                          // Metaspace, prepend_scheme always
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
