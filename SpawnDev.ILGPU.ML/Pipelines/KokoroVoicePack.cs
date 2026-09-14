using SpawnDev.Phonemizer;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// A Kokoro voice: the style vectors that make it sound like a particular speaker.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THE FILE IS A TABLE, NOT A VECTOR, and picking the wrong row is the single easiest way to get this
/// wrong. Each published voice is <b>510 x 256 float32</b> - MEASURED: <c>af_heart.bin</c> is exactly
/// 522,240 bytes, which is 510 * 256 * 4 - and the row is chosen by the number of TOKENS in the
/// utterance, not by the voice, the sentence or anything else. Kokoro conditions on length, so a fixed
/// row would make short lines and long lines drift in timbre for no visible reason.
/// </para>
/// <para>
/// ⚠️ Little-endian float32, which is what every browser and every x86/ARM host this runs on uses. The
/// check is on SIZE rather than on content because a truncated or HTML-error-page download is the
/// realistic failure (a 404 page saved as .bin is not 522,240 bytes), while a byte-order mistake would
/// not be caught by any cheap test and does not happen on the platforms this ships to.
/// </para>
/// </remarks>
public sealed class KokoroVoicePack
{
    /// <summary>Style vectors per published voice file.</summary>
    public const int Rows = 510;

    /// <summary>Length of one style vector - the model's <c>style</c> input is [1, 256].</summary>
    public const int Dim = 256;

    /// <summary>Exact byte length of a published voice file.</summary>
    public const int FileBytes = Rows * Dim * sizeof(float);

    /// <summary>The repository the built-in voices live in, as a path the hub serves.</summary>
    /// <remarks>
    /// ⚠️ Through the hub, never huggingface.co directly - the hub caches, supplies the CORS headers a
    /// browser needs, and keeps every visitor out of HuggingFace's rate limiter.
    /// </remarks>
    public const string VoiceRepoId = "onnx-community/Kokoro-82M-v1.0-ONNX";

    /// <summary>Path within the repo for a voice by name, e.g. <c>voices/af_heart.bin</c>.</summary>
    public static string VoicePath(string name) => $"voices/{name}.bin";

    /// <summary>
    /// The English voices published alongside the model, VERIFIED to exist and to be the right size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each was checked against the hub on 2026-09-14: all 28 present, each exactly
    /// <see cref="FileBytes"/> (522,240) bytes. That matters more than it sounds - a voice file of the
    /// wrong length is not a loud failure, it is a silently different <see cref="StyleFor"/> row, so the
    /// list and the size check belong together. Apache-2.0, same licence as the model.
    /// </para>
    /// <para>
    /// ⚠️ A NAME IS NOT A VOICE until its bytes load: this is what the repo publishes, not a promise that
    /// a given deployment has cached it. Treat a miss as "not available here", not as a bad name.
    /// The prefix is the accent and gender the voice was trained for - <c>af_</c>/<c>am_</c> American
    /// female/male, <c>bf_</c>/<c>bm_</c> British female/male.
    /// </para>
    /// </remarks>
    public static IReadOnlyList<string> EnglishVoiceNames { get; } = new[]
    {
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
        "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    };

    /// <summary>The default voice - the one every measurement in this port was taken against.</summary>
    public const string DefaultVoiceName = "af_heart";

    private readonly float[] _all;

    /// <summary>The voice's name, for a picker.</summary>
    public string Name { get; }

    private KokoroVoicePack(string name, float[] all)
    {
        Name = name;
        _all = all;
    }

    /// <summary>
    /// Read a published voice file.
    /// </summary>
    /// <param name="name">Display name for this voice.</param>
    /// <param name="bytes">The raw <c>.bin</c> - 510 x 256 little-endian float32.</param>
    /// <exception cref="InvalidDataException">The file is not the published size.</exception>
    public static KokoroVoicePack FromBytes(string name, byte[] bytes)
    {
        if (bytes == null) throw new ArgumentNullException(nameof(bytes));
        if (bytes.Length != FileBytes)
            throw new InvalidDataException(
                $"'{name}' is {bytes.Length} bytes; a Kokoro voice is exactly {FileBytes} "
                + $"({Rows} x {Dim} float32). A wrong size here is almost always a failed download saved "
                + "as a file - an error page, or a truncated transfer.");

        var all = new float[Rows * Dim];
        Buffer.BlockCopy(bytes, 0, all, 0, bytes.Length);
        return new KokoroVoicePack(name, all);
    }

    /// <summary>
    /// The style vector for an utterance of <paramref name="tokenCount"/> tokens.
    /// </summary>
    /// <remarks>
    /// ⚠️ CLAMPED, not wrapped or thrown. A line longer than the table is legitimate - the published
    /// table simply stops at <see cref="Rows"/> - and the last row is the right answer for anything
    /// beyond it. Wrapping would make a long sentence adopt the timbre of a very short one, which sounds
    /// like the voice changing mid-reply.
    /// </remarks>
    /// <param name="tokenCount">Tokens in the utterance, INCLUDING the padding token at each end.</param>
    public ReadOnlySpan<float> StyleFor(int tokenCount)
    {
        var row = Math.Clamp(tokenCount, 0, Rows - 1);
        return _all.AsSpan(row * Dim, Dim);
    }
}

/// <summary>
/// Kokoro's text front end: IPA phonemes to the token ids the graph expects.
/// </summary>
/// <remarks>
/// <para>
/// Kokoro takes <c>tokens int64[1, sequence_length]</c>, <c>style float32[1, 256]</c> and
/// <c>speed float32[1]</c>, and returns audio directly - there is no separate vocoder. So the whole
/// front end is this: phonemes in, ids out.
/// </para>
/// <para>
/// ⭐ The phonemes come from <c>SpawnDev.Phonemizer</c>, which exists precisely because the rest of the
/// ecosystem reaches for espeak-ng and espeak-ng is GPL-3. Kokoro is named in its README as one of the
/// models this unblocks.
/// </para>
/// <para>
/// ⚠️ THE PADDING TOKEN IS NOT DECORATION. Kokoro is trained with a <c>$</c> (id 0) at BOTH ends and
/// produces clipped, rushed audio without them. They also count toward the style row, so they are added
/// here - once, in the one place that builds a token sequence - rather than left to each caller.
/// </para>
/// </remarks>
public static class KokoroTokenizer
{
    /// <summary>The padding symbol Kokoro expects at both ends of an utterance.</summary>
    public const string PadSymbol = "$";

    /// <summary>Longest token sequence the published style table covers.</summary>
    /// <remarks>
    /// A longer utterance still runs - <see cref="KokoroVoicePack.StyleFor"/> clamps - but the caller is
    /// better off splitting it, because past this point the conditioning stops matching the length.
    /// </remarks>
    public const int MaxTokens = KokoroVoicePack.Rows;

    private static PhonemeVocabulary? _vocabulary;

    /// <summary>Kokoro's published phoneme vocabulary: 115 symbols, ids 0..177.</summary>
    public static PhonemeVocabulary Vocabulary =>
        _vocabulary ??= PhonemeVocabulary.Parse(
            KokoroVocabularyData.Tsv.Split('\n', StringSplitOptions.RemoveEmptyEntries));

    /// <summary>
    /// Turn IPA symbols into Kokoro token ids, padded at both ends.
    /// </summary>
    /// <remarks>
    /// ⚠️ SYMBOLS THE VOCABULARY DOES NOT HAVE ARE DROPPED, and the count of dropped symbols is returned
    /// rather than swallowed. Kokoro genuinely cannot say a sound it has no token for, so there is no
    /// correct substitute - but a frontend that silently removes sounds produces audio that is subtly
    /// wrong with nothing to explain it, which is the failure mode that wastes the most time. The caller
    /// gets the number and decides whether to report it.
    /// </remarks>
    /// <param name="symbols">IPA symbols, e.g. from <c>EnglishPhonemizer.ToSymbols</c>.</param>
    /// <returns>The token ids including padding, and how many symbols had no token.</returns>
    public static (long[] Tokens, int Dropped) Encode(IEnumerable<string> symbols)
    {
        if (symbols == null) throw new ArgumentNullException(nameof(symbols));
        var vocab = Vocabulary;
        if (!vocab.TryGetId(PadSymbol, out var pad))
            throw new InvalidOperationException(
                $"the Kokoro vocabulary has no '{PadSymbol}' padding token - the embedded table is wrong");

        var ids = new List<long> { pad };
        var dropped = 0;
        foreach (var symbol in symbols)
        {
            if (vocab.TryGetId(symbol, out var id)) ids.Add(id);
            else dropped++;
        }
        ids.Add(pad);
        return (ids.ToArray(), dropped);
    }
}
