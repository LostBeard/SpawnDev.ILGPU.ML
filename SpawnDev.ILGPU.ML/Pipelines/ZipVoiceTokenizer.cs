using SpawnDev.Phonemizer;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Turns English text into the token ids ZipVoice speaks.
/// </summary>
/// <remarks>
/// <para>
/// This is the join between the phonemizer and the model, and without it neither is usable on its own:
/// the phonemizer emits IPA symbols, the model consumes integers, and the mapping between them lives in
/// the model's own <c>tokens.txt</c>.
/// </para>
/// <para>
/// The shipped ZipVoice lexicon is Chinese-only - 68,037 CJK entries and no English at all - which is why
/// English has to be phonemized rather than looked up, and why the reference implementation reaches for
/// espeak-ng (GPL-3) at this exact point. <see cref="SpawnDev.Phonemizer"/> is the MIT replacement.
/// </para>
/// <para>
/// A symbol with no token is reported, never dropped. Silently skipping one produces speech that is
/// subtly wrong with nothing to explain it, which is the failure mode this whole component exists to
/// avoid.
/// </para>
/// </remarks>
public sealed class ZipVoiceTokenizer
{
    private readonly Dictionary<string, long> _symbolToId;
    private readonly EnglishPhonemizer _phonemizer;

    /// <summary>Create a tokenizer over a model's symbol table and a phonemizer.</summary>
    public ZipVoiceTokenizer(IReadOnlyDictionary<string, long> symbolToId, EnglishPhonemizer phonemizer)
    {
        _symbolToId = new Dictionary<string, long>(symbolToId, StringComparer.Ordinal);
        _phonemizer = phonemizer ?? throw new ArgumentNullException(nameof(phonemizer));
    }

    /// <summary>Everything wired up from a model directory: its tokens, the bundled dictionary and rules.</summary>
    /// <param name="modelDirectory">The directory holding the model's <c>tokens.txt</c>.</param>
    public static ZipVoiceTokenizer CreateDefault(string modelDirectory)
        => new(LoadSymbolTable(Path.Combine(modelDirectory, "tokens.txt")), EmbeddedData.CreatePhonemizer());

    /// <summary>Read a <c>tokens.txt</c>: one "symbol TAB id" per line.</summary>
    /// <remarks>
    /// Split on the LAST tab, because the symbol itself can be a space - id 3 in this model is the word
    /// separator, and splitting on the first tab would lose it.
    /// </remarks>
    public static Dictionary<string, long> LoadSymbolTable(string tokensPath)
    {
        var table = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var raw in File.ReadLines(tokensPath))
        {
            var line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0) continue;
            int cut = line.LastIndexOf('\t');
            if (cut < 0 || !long.TryParse(line[(cut + 1)..], out var id)) continue;
            table.TryAdd(line[..cut], id);
        }
        return table;
    }

    /// <summary>Words the dictionary did not contain, from the last call. Sounded out, not skipped.</summary>
    public IReadOnlyList<string> LastUnknownWords => _phonemizer.LastUnknownWords;

    /// <summary>The phonemizer behind this tokenizer, for callers who want to tune its rules.</summary>
    public EnglishPhonemizer Phonemizer => _phonemizer;

    /// <summary>Encode English text into ZipVoice token ids.</summary>
    /// <exception cref="InvalidOperationException">
    /// A phoneme was produced that this model has no token for. Thrown rather than skipped: the caller
    /// would otherwise get audio that is quietly missing a sound, with nothing to point at.
    /// </exception>
    public long[] Encode(string text)
    {
        var symbols = _phonemizer.ToSymbols(text);
        var ids = new long[symbols.Count];
        for (int i = 0; i < symbols.Count; i++)
        {
            if (_symbolToId.TryGetValue(symbols[i], out var id)) { ids[i] = id; continue; }
            throw new InvalidOperationException(
                $"the phonemizer produced '{symbols[i]}', which this model has no token for. " +
                $"Text: \"{text}\"");
        }
        return ids;
    }

    /// <summary>Encode, reporting failure instead of throwing.</summary>
    public bool TryEncode(string text, out long[] ids, out string? problem)
    {
        try
        {
            ids = Encode(text);
            problem = null;
            return true;
        }
        catch (InvalidOperationException ex)
        {
            ids = [];
            problem = ex.Message;
            return false;
        }
    }
}
