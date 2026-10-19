using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SpawnDev.ILGPU.ML.Flux2;

/// <summary>
/// The byte-level BPE tokenizer of the Qwen2 / Qwen3 family (151,643 ordinary tokens plus the added ones), read from the model's own <c>tokenizer.json</c>:
/// NFC normalisation, added tokens matched literally, the family's pre-tokenisation regex, GPT-2 byte-to-unicode mapping, merges applied by rank.
/// FLUX.2 klein feeds its Qwen3-4B text encoder the chat-templated prompt (<see cref="ChatPrompt"/>), padded with <c>&lt;|endoftext|&gt;</c> to 512 (<see cref="Pad"/>).
/// Checked token for token against the reference tokenizer (tools/unirig-harness/reference/golden_klein.py and the tokenizer goldens).
/// </summary>
public sealed class Qwen2Tokenizer
{
    private const string PreRegex = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    private static readonly Regex Pre = new(PreRegex, RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _rank = new();
    private readonly Dictionary<string, int> _added;
    private readonly Regex? _addedSplit;
    private readonly Dictionary<string, int[]> _cache = new();
    private static readonly string[] ByteChar = BuildByteChars();

    /// <summary>The id the reference pads with (&lt;|endoftext|&gt;).</summary>
    public int PadId { get; }

    public Qwen2Tokenizer(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(tokenizerJson);
        var model = doc.RootElement.GetProperty("model");
        _vocab = new Dictionary<string, int>();
        foreach (var p in model.GetProperty("vocab").EnumerateObject()) _vocab[p.Name] = p.Value.GetInt32();
        int r = 0;
        foreach (var m in model.GetProperty("merges").EnumerateArray())
        {
            string a, b;
            if (m.ValueKind == JsonValueKind.Array) { a = m[0].GetString()!; b = m[1].GetString()!; }
            else { var s = m.GetString()!; int sp = s.IndexOf(' '); a = s[..sp]; b = s[(sp + 1)..]; }
            _rank[(a, b)] = r++;
        }
        _added = new Dictionary<string, int>();
        if (doc.RootElement.TryGetProperty("added_tokens", out var added))
            foreach (var t in added.EnumerateArray()) _added[t.GetProperty("content").GetString()!] = t.GetProperty("id").GetInt32();
        if (_added.Count > 0)
            _addedSplit = new Regex("(" + string.Join("|", _added.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape)) + ")", RegexOptions.CultureInvariant);
        PadId = _added.TryGetValue("<|endoftext|>", out var pad) ? pad : 151643;
    }

    /// <summary>The user turn the klein pipeline renders (chat template, generation prompt, thinking off).</summary>
    public static string ChatPrompt(string prompt) => "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    /// <summary>Token ids of <paramref name="text"/> (no special tokens are added; the added ones in the text are recognised).</summary>
    public int[] Encode(string text)
    {
        text = text.Normalize(NormalizationForm.FormC);
        var ids = new List<int>();
        if (_addedSplit == null) { EncodeOrdinary(text, ids); return ids.ToArray(); }
        int pos = 0;
        foreach (Match m in _addedSplit.Matches(text))
        {
            if (m.Index > pos) EncodeOrdinary(text.Substring(pos, m.Index - pos), ids);
            ids.Add(_added[m.Value]);
            pos = m.Index + m.Length;
        }
        if (pos < text.Length) EncodeOrdinary(text.Substring(pos), ids);
        return ids.ToArray();
    }

    /// <summary>The ids the pipeline feeds the encoder: the chat-templated prompt truncated and right-padded to <paramref name="length"/>; also the number of real tokens.</summary>
    public (int[] Ids, int Real) EncodePadded(string prompt, int length = 512)
    {
        var ids = Encode(ChatPrompt(prompt));
        int real = Math.Min(ids.Length, length);
        var o = new int[length];
        Array.Fill(o, PadId);
        Array.Copy(ids, o, real);
        return (o, real);
    }

    private void EncodeOrdinary(string text, List<int> ids)
    {
        foreach (Match m in Pre.Matches(text))
        {
            if (_cache.TryGetValue(m.Value, out var hit)) { ids.AddRange(hit); continue; }
            var sb = new StringBuilder();
            foreach (var b in Encoding.UTF8.GetBytes(m.Value)) sb.Append(ByteChar[b]);
            var pieces = Bpe(sb.ToString());
            var piece = new int[pieces.Count];
            for (int i = 0; i < piece.Length; i++) piece[i] = _vocab[pieces[i]];
            _cache[m.Value] = piece;
            ids.AddRange(piece);
        }
    }

    private List<string> Bpe(string word)
    {
        var parts = new List<string>();
        foreach (var ch in word) parts.Add(ch.ToString());   // every mapped byte is one BMP char
        while (parts.Count > 1)
        {
            int best = int.MaxValue, at = -1;
            for (int i = 0; i + 1 < parts.Count; i++)
                if (_rank.TryGetValue((parts[i], parts[i + 1]), out var rk) && rk < best) { best = rk; at = i; }
            if (at < 0) break;
            string merged = parts[at] + parts[at + 1];
            var next = new List<string>(parts.Count - 1);
            for (int i = 0; i < parts.Count; i++)
            {
                if (i == at) { next.Add(merged); i++; }
                else next.Add(parts[i]);
            }
            parts = next;
        }
        return parts;
    }

    /// <summary>GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n.</summary>
    private static string[] BuildByteChars()
    {
        var keep = new List<int>();
        for (int b = '!'; b <= '~'; b++) keep.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) keep.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) keep.Add(b);
        var map = new string[256];
        int n = 0;
        for (int b = 0; b < 256; b++)
            map[b] = keep.Contains(b) ? ((char)b).ToString() : ((char)(256 + n++)).ToString();
        return map;
    }
}
