using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.Phonemizer;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Kokoro's front end: the published phoneme vocabulary, and the style table that voices it.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THIS IS THE HALF THAT FAILS QUIETLY. A wrong operator throws; a wrong TOKEN ID does not - the model
/// happily speaks a different sound, and the result is fluent audio saying something slightly other than
/// the text. Nothing downstream can detect that: it passes every amplitude, duration and RMS check, and a
/// listener hears an accent rather than a bug. So the vocabulary is pinned by structure here rather than
/// trusted because it was generated correctly once.
/// </para>
/// <para>
/// ⚠️ No GPU, no model, no network - a table and some arithmetic, so these are pure tests.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>The published vocabulary is intact: 115 symbols with the ids Kokoro was trained on.</summary>
    /// <remarks>
    /// Values are from <c>onnx-community/Kokoro-82M-v1.0-ONNX/tokenizer.json</c>. They are asserted rather
    /// than counted because "115 entries" alone would pass a table with every id shifted by one, which is
    /// exactly the shape a bad regeneration takes.
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_VocabularyMatchesThePublishedTable() => await RunPureTest(() =>
    {
        var vocab = KokoroTokenizer.Vocabulary;
        if (vocab.Count != 115)
            throw new Exception($"Kokoro publishes 115 phoneme tokens, this table has {vocab.Count}");

        // Anchors spread across the table, each a different KIND of symbol.
        (string Symbol, long Id)[] anchors =
        [
            ("$", 0),      // padding - what the model is trained to see at both ends
            (";", 1),
            (".", 4),
            ("?", 6),
            (" ", 16),     // ⚠️ A SPACE IS A TOKEN. The table is split on its LAST tab for this one
                           // symbol; losing it runs every word together.
            ("A", 24),     // upper and lower case are DIFFERENT phonemes in this alphabet
            ("a", 43),
            ("z", 68),
        ];
        foreach (var (symbol, expected) in anchors)
        {
            if (!vocab.TryGetId(symbol, out var id))
                throw new Exception($"'{symbol}' is missing from the Kokoro vocabulary");
            if (id != expected)
                throw new Exception($"'{symbol}' should be token {expected}, got {id} - every id after "
                    + "this point is suspect, and a wrong id speaks a different sound rather than failing");
        }

        vocab.TryGetId("A", out var upper);
        vocab.TryGetId("a", out var lower);
        if (upper == lower) throw new Exception("upper and lower case collapsed to one token");
        return Task.CompletedTask;
    });

    /// <summary>Every token sequence is padded at both ends, because Kokoro is trained that way.</summary>
    /// <remarks>
    /// ⚠️ Without the padding Kokoro produces clipped, rushed audio - and the pads also count toward the
    /// style row, so omitting them shifts the voice as well as the timing. Added in one place so no
    /// caller can forget.
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_EncodingPadsBothEnds() => await RunPureTest(() =>
    {
        var (tokens, dropped) = KokoroTokenizer.Encode(new[] { "k", "a", "t" });
        if (dropped != 0) throw new Exception($"{dropped} of those symbols failed to encode");
        if (tokens.Length != 5)
            throw new Exception($"three symbols plus a pad at each end is 5 tokens, got {tokens.Length}");

        KokoroTokenizer.Vocabulary.TryGetId(KokoroTokenizer.PadSymbol, out var pad);
        if (tokens[0] != pad || tokens[^1] != pad)
            throw new Exception($"the sequence must be padded at BOTH ends, got "
                + $"[{string.Join(",", tokens)}] with pad={pad}");

        var (empty, _) = KokoroTokenizer.Encode(Array.Empty<string>());
        if (empty.Length != 2)
            throw new Exception($"an empty utterance is just the two pads, got {empty.Length} tokens");
        return Task.CompletedTask;
    });

    /// <summary>A symbol with no token is COUNTED, never silently dropped.</summary>
    /// <remarks>
    /// 🔴 Kokoro cannot say a sound it has no token for, so there is no correct substitute - but removing
    /// one without saying so produces audio missing a sound with nothing to explain why. That is the most
    /// expensive failure a TTS frontend has, because it reads as a model quality problem rather than a
    /// mapping problem.
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_UnmappableSymbolsAreReported() => await RunPureTest(() =>
    {
        var (tokens, dropped) = KokoroTokenizer.Encode(new[] { "k", " NOT_A_PHONEME", "t" });
        if (dropped != 1) throw new Exception($"the unmappable symbol must be counted, got {dropped}");
        if (tokens.Length != 4)
            throw new Exception($"two real symbols plus two pads is 4 tokens, got {tokens.Length}");
        return Task.CompletedTask;
    });

    /// <summary>Our own phonemizer's output is fully speakable by Kokoro.</summary>
    /// <remarks>
    /// 🔴 THE FEASIBILITY FACT, AND A REGRESSION GUARD ON BOTH SIDES. Operator coverage says the graph can
    /// run and the hub says the weights arrive; neither says the two ALPHABETS agree. MEASURED 2026-09-14
    /// over these five sentences: <b>271 of 271 symbols encodable, 0 dropped</b>. This keeps it that way -
    /// a change to SpawnDev.Phonemizer's inventory that introduced a symbol Kokoro has no token for would
    /// otherwise surface as slightly wrong speech, months later, with no obvious cause.
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_OurPhonemesAreAllSpeakable() => await RunPureTest(() =>
    {
        var phonemizer = EmbeddedData.CreatePhonemizer();
        string[] samples =
        [
            "The capital of France is Paris.",
            "She waited for 2 more minutes, then left without saying anything.",
            "Hello! How are you today? I hope it's going well.",
            "Roughly seventy-three percent of the measurements agreed.",
            "A quick brown fox jumps over the lazy dog.",
        ];

        var unmapped = new SortedSet<string>(StringComparer.Ordinal);
        int total = 0, lost = 0;
        foreach (var text in samples)
        {
            var symbols = phonemizer.ToSymbols(text);
            total += symbols.Count;
            var (_, dropped) = KokoroTokenizer.Encode(symbols);
            lost += dropped;
            foreach (var s in symbols)
                if (!KokoroTokenizer.Vocabulary.TryGetId(s, out _)) unmapped.Add(s);
        }

        // A fixture that produced almost nothing would pass the real assertion trivially.
        if (total < 200)
            throw new Exception($"the fixture produced only {total} symbols - it can no longer demonstrate "
                + "the property");
        if (lost != 0)
            throw new Exception($"{lost} of {total} phonemes have no Kokoro token, so those sounds would be "
                + $"silently missing from the speech: {string.Join(", ", unmapped.Select(s => $"'{s}'"))}");
        return Task.CompletedTask;
    });

    /// <summary>The style table is read as 510 rows of 256, and the row follows the token count.</summary>
    /// <remarks>
    /// ⚠️ THE ROW IS CHOSEN BY LENGTH, which is the detail that looks like a constant and is not. Kokoro
    /// conditions its style on the number of tokens; pinning one row would make short and long lines drift
    /// in timbre with nothing on screen to explain it.
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_StyleRowFollowsTheTokenCount() => await RunPureTest(() =>
    {
        if (KokoroVoicePack.FileBytes != 522_240)
            throw new Exception($"a published voice is 510 x 256 float32 = 522,240 bytes (af_heart.bin "
                + $"measured exactly that), this build says {KokoroVoicePack.FileBytes}");

        // A synthetic pack whose every row holds its own index, so a wrong row is unmistakable.
        var bytes = new byte[KokoroVoicePack.FileBytes];
        for (var row = 0; row < KokoroVoicePack.Rows; row++)
            for (var col = 0; col < KokoroVoicePack.Dim; col++)
                BitConverter.GetBytes((float)row)
                    .CopyTo(bytes, (row * KokoroVoicePack.Dim + col) * sizeof(float));

        var pack = KokoroVoicePack.FromBytes("synthetic", bytes);
        if (pack.StyleFor(35)[0] != 35f)
            throw new Exception($"a 35-token utterance must read row 35, read row {pack.StyleFor(35)[0]}");
        if (pack.StyleFor(0)[0] != 0f) throw new Exception("an empty utterance must read row 0");
        if (pack.StyleFor(35).Length != KokoroVoicePack.Dim)
            throw new Exception($"a style vector is {KokoroVoicePack.Dim} wide, got {pack.StyleFor(35).Length}");

        // ⚠️ CLAMPED, not wrapped. A line longer than the table is legitimate; wrapping would give it the
        // timbre of a very short one, which sounds like the voice changing mid-reply.
        if (pack.StyleFor(10_000)[0] != 509f)
            throw new Exception($"a long utterance must clamp to the last row, read {pack.StyleFor(10_000)[0]}");
        return Task.CompletedTask;
    });

    /// <summary>A voice file that is not the published size is rejected, with the size in the message.</summary>
    /// <remarks>
    /// The realistic failure is not a corrupt float - it is an error page or a truncated transfer saved as
    /// a .bin. Length catches those, and naming the length that arrived is what separates "the download
    /// failed" from "this voice is broken".
    /// </remarks>
    [TestMethod]
    public async Task Kokoro_AWrongSizedVoiceFileIsRejected() => await RunPureTest(() =>
    {
        try
        {
            KokoroVoicePack.FromBytes("truncated", new byte[1024]);
        }
        catch (InvalidDataException ex)
        {
            if (!ex.Message.Contains("1024"))
                throw new Exception($"the message must name the size that arrived: {ex.Message}");
            return Task.CompletedTask;
        }
        throw new Exception("a short voice file was accepted - it would be read as silence");
    });
}
