namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Stateful, UTF-8-safe incremental detokenizer for <see cref="SentencePieceTokenizer"/>. Built for
/// streaming generation (SSE / NDJSON, token-by-token): each <see cref="Push"/> returns only the text
/// that is now COMPLETE, holding back an incomplete trailing multi-byte UTF-8 sequence — or a multi-byte
/// glyph emitted as a run of &lt;0xHH&gt; byte-fallback tokens — until the token that completes it arrives.
///
/// This is the correct path for streaming AND it fixes a latent bug in
/// <see cref="SentencePieceTokenizer.Decode"/>, which appends each byte-fallback byte as its own
/// <c>(char)</c> — a Latin-1 reinterpretation that turns a multi-byte glyph (e.g. "é" = 0xC3 0xA9) into
/// mojibake ("Ã©"). Here the bytes accumulate and decode as UTF-8, so the glyph comes out intact.
/// </summary>
public sealed class SentencePieceStreamingDecoder
{
    private readonly SentencePieceTokenizer _tok;
    private readonly List<byte> _pending = new();
    private bool _trimmedLeadingSpace;

    internal SentencePieceStreamingDecoder(SentencePieceTokenizer tokenizer) => _tok = tokenizer;

    /// <summary>
    /// Feed one generated token id; returns the newly-complete text delta (may be empty while waiting
    /// for the bytes that finish a multi-byte sequence).
    /// </summary>
    public string Push(int id)
    {
        var bytes = _tok.TokenToBytes(id);
        if (bytes.Length > 0) _pending.AddRange(bytes);
        return Drain(flush: false);
    }

    /// <summary>
    /// Flush any buffered bytes at end-of-generation. A leftover incomplete UTF-8 sequence is emitted
    /// as the Unicode replacement character (U+FFFD) by <see cref="System.Text.Encoding.UTF8"/>.
    /// </summary>
    public string Finish() => Drain(flush: true);

    private string Drain(bool flush)
    {
        if (_pending.Count == 0) return string.Empty;

        int complete = flush ? _pending.Count : CompleteUtf8PrefixLength(_pending);
        if (complete == 0) return string.Empty;

        string text = System.Text.Encoding.UTF8.GetString(_pending.GetRange(0, complete).ToArray());
        _pending.RemoveRange(0, complete);

        // Mirror Decode's single leading-space trim (SentencePiece prefixes the first word with ▁→space).
        if (!_trimmedLeadingSpace && text.Length > 0)
        {
            _trimmedLeadingSpace = true;
            if (text[0] == ' ') text = text[1..];
        }
        return text;
    }

    /// <summary>
    /// Length of the longest prefix of <paramref name="buf"/> made up only of COMPLETE UTF-8 sequences
    /// (i.e. excluding a trailing partial multi-byte sequence whose continuation bytes haven't arrived).
    /// </summary>
    private static int CompleteUtf8PrefixLength(List<byte> buf)
    {
        int i = 0, complete = 0;
        while (i < buf.Count)
        {
            byte b = buf[i];
            int need =
                b < 0x80 ? 1 :
                (b & 0xE0) == 0xC0 ? 2 :
                (b & 0xF0) == 0xE0 ? 3 :
                (b & 0xF8) == 0xF0 ? 4 :
                1; // lone continuation / invalid lead → consume 1 (UTF8.GetString substitutes U+FFFD)
            if (i + need > buf.Count) break; // incomplete trailing sequence — hold it for the next token
            i += need;
            complete = i;
        }
        return complete;
    }
}
