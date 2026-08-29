namespace SpawnDev.Phonemizer;

/// <summary>
/// Maps phoneme symbols to the token ids a speech model expects, and back.
/// </summary>
/// <remarks>
/// <para>
/// The last mile of a text-to-speech frontend. <see cref="EnglishPhonemizer.ToSymbols"/> produces IPA
/// symbols; a neural model wants integers. Every consumer was writing that loop itself - four copies of
/// it existed in this repository's own tools alone - and each copy is a chance to get the same two
/// details wrong.
/// </para>
/// <para>
/// ⚠️ The first detail is that <b>a symbol can be whitespace</b>. A ZipVoice vocabulary really does list
/// a space as a token ("&#32;\t3"), so the file must be split on its LAST tab, never on whitespace and
/// never with <c>Split</c>. Splitting the obvious way silently loses the token that separates words.
/// </para>
/// <para>
/// The second is what to do with a symbol the model has no token for. It cannot be spoken at all, so
/// dropping it quietly produces audio that is missing a sound with nothing to explain why.
/// <see cref="Encode"/> therefore throws and names the symbol; <see cref="TryEncode"/> is there for
/// callers that would rather decide for themselves.
/// </para>
/// </remarks>
public sealed class PhonemeVocabulary
{
    private readonly Dictionary<string, long> _toId;
    private readonly Dictionary<long, string> _toSymbol;

    private PhonemeVocabulary(Dictionary<string, long> toId, Dictionary<long, string> toSymbol)
    {
        _toId = toId;
        _toSymbol = toSymbol;
    }

    /// <summary>Number of symbols in the vocabulary.</summary>
    public int Count => _toId.Count;

    /// <summary>
    /// Parse a <c>tokens.txt</c>: one entry per line, the symbol, a tab, then the id.
    /// </summary>
    /// <remarks>
    /// Split on the LAST tab, because the symbol itself may be whitespace. A duplicate symbol keeps its
    /// FIRST id, matching the order the file lists them in.
    /// </remarks>
    public static PhonemeVocabulary Parse(IEnumerable<string> lines)
    {
        var toId = new Dictionary<string, long>(StringComparer.Ordinal);
        var toSymbol = new Dictionary<long, string>();

        foreach (var raw in lines)
        {
            var line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0) continue;

            var cut = line.LastIndexOf('\t');
            if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;

            var symbol = line[..cut];
            toId.TryAdd(symbol, id);
            toSymbol.TryAdd(id, symbol);
        }
        return new PhonemeVocabulary(toId, toSymbol);
    }

    /// <summary>Load a <c>tokens.txt</c> from disk.</summary>
    public static PhonemeVocabulary Load(string path) => Parse(File.ReadLines(path));

    /// <summary>The id for one symbol.</summary>
    public bool TryGetId(string symbol, out long id) => _toId.TryGetValue(symbol, out id);

    /// <summary>The symbol for one id.</summary>
    public bool TryGetSymbol(long id, out string symbol) => _toSymbol.TryGetValue(id, out symbol!);

    /// <summary>
    /// Turn phonemizer output into token ids.
    /// </summary>
    /// <param name="symbols">Symbols, as <see cref="EnglishPhonemizer.ToSymbols"/> returns them.</param>
    /// <exception cref="ArgumentException">
    /// A symbol has no token. That is not recoverable by ignoring it: the model cannot speak the sound,
    /// so the sentence would be rendered missing it with nothing to say why.
    /// </exception>
    public long[] Encode(IReadOnlyList<string> symbols)
    {
        if (!TryEncode(symbols, out var ids, out var missing))
            throw new ArgumentException(
                $"no token for the symbol '{missing}' in this vocabulary of {Count} - the model cannot "
              + "speak that sound, so the text cannot be rendered as given.", nameof(symbols));
        return ids;
    }

    /// <summary>
    /// Turn phonemizer output into token ids, reporting the first symbol that has none.
    /// </summary>
    /// <param name="symbols">Symbols to encode.</param>
    /// <param name="ids">The ids, or empty when a symbol could not be mapped.</param>
    /// <param name="unmappable">The first symbol with no token, or null on success.</param>
    public bool TryEncode(IReadOnlyList<string> symbols, out long[] ids, out string? unmappable)
    {
        ids = [];
        unmappable = null;
        if (symbols is null) return true;

        var result = new long[symbols.Count];
        for (var i = 0; i < symbols.Count; i++)
        {
            if (!_toId.TryGetValue(symbols[i], out var id)) { unmappable = symbols[i]; return false; }
            result[i] = id;
        }
        ids = result;
        return true;
    }

    /// <summary>
    /// Turn token ids back into symbols, for reading a model's input back.
    /// </summary>
    /// <remarks>An id with no symbol comes back as "?", so a decode never throws while debugging.</remarks>
    public string[] Decode(IReadOnlyList<long> ids)
    {
        if (ids is null) return [];
        var symbols = new string[ids.Count];
        for (var i = 0; i < ids.Count; i++)
            symbols[i] = _toSymbol.TryGetValue(ids[i], out var s) ? s : "?";
        return symbols;
    }
}
