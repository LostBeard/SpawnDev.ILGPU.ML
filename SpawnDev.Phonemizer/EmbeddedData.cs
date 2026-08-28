using System.IO.Compression;

namespace SpawnDev.Phonemizer;

/// <summary>
/// The dictionary and letter-to-sound rules, carried inside the assembly.
/// </summary>
/// <remarks>
/// <para>
/// A phonemizer that needs two data files fetched from somewhere is not really dependency-free, and in a
/// browser it is one more thing to host, version and get wrong. Both are embedded, gzipped, and expanded
/// on first use: 915 KB for the dictionary and 340 KB for the rules, against models measured in hundreds
/// of megabytes elsewhere in this stack.
/// </para>
/// <para>
/// <see cref="System.IO.Compression"/> is part of the base class library and works under WebAssembly, so
/// this adds no package reference.
/// </para>
/// </remarks>
public static class EmbeddedData
{
    /// <summary>The bundled CMU Pronouncing Dictionary. See THIRD-PARTY-NOTICES.md.</summary>
    public static PronunciationDictionary LoadDictionary()
        => PronunciationDictionary.Parse(ReadLines("cmudict.dict.gz"));

    /// <summary>The bundled letter-to-sound rules, learned from that dictionary.</summary>
    public static LetterToSound LoadLetterToSound()
        => LetterToSound.Parse(ReadLines("lts-model.txt.gz"));

    /// <summary>A phonemizer with everything wired up, which is what most callers want.</summary>
    public static EnglishPhonemizer CreatePhonemizer()
        => new(LoadDictionary()) { LetterToSound = LoadLetterToSound() };

    private static IEnumerable<string> ReadLines(string resourceName)
    {
        var assembly = typeof(EmbeddedData).Assembly;
        var full = assembly.GetManifestResourceNames()
            .FirstOrDefault(n => n.EndsWith(resourceName, StringComparison.Ordinal))
            ?? throw new InvalidOperationException(
                $"embedded resource '{resourceName}' is missing from {assembly.GetName().Name}");

        using var raw = assembly.GetManifestResourceStream(full)!;
        using var gzip = new GZipStream(raw, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip);
        while (reader.ReadLine() is { } line) yield return line;
    }
}
