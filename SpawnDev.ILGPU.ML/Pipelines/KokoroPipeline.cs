using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>What one <see cref="KokoroPipeline.SpeakAsync"/> call produced.</summary>
/// <param name="Samples">Mono PCM in [-1, 1].</param>
/// <param name="SampleRate">Always <see cref="KokoroPipeline.OutputSampleRate"/>.</param>
/// <param name="Tokens">Token count including the two padding tokens - also the style row that was used.</param>
/// <param name="DroppedPhonemes">
/// Phonemes with no Kokoro token. Zero for English through <c>SpawnDev.Phonemizer</c> (MEASURED
/// 271/271), and reported rather than swallowed because a dropped phoneme is a missing SOUND, not an error.
/// </param>
/// <param name="InferenceMs">Wall time of the forward pass.</param>
public readonly record struct KokoroAudio(
    float[] Samples, int SampleRate, int Tokens, int DroppedPhonemes, double InferenceMs)
{
    /// <summary>Duration of the produced audio.</summary>
    public double Seconds => SampleRate > 0 ? Samples.Length / (double)SampleRate : 0;

    /// <summary>
    /// Render time as a multiple of realtime. Below 1.0 is faster than speech.
    /// </summary>
    /// <remarks>
    /// The number that decides whether a voice can stream. ZipVoice MEASURED 3.6x at 320 characters, which
    /// is why a reply spoken in chunks stalls between them: the renderer can never catch up with the
    /// speaker. Reported per call so the answer is never a matter of opinion.
    /// </remarks>
    public double RealtimeFactor => Seconds > 0 ? InferenceMs / 1000.0 / Seconds : 0;
}

/// <summary>
/// Kokoro-82M: phonemes and a voice in, speech out, in ONE pass.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY THIS EXISTS. Captain: "is ZipVoice the best choise for streaming tts? why is our so slow? we
/// needs fast tts. there is no reason for it to be that slow." He was right, and the reason is structural
/// rather than fixable by tuning: ZipVoice's flow decoder is 8,621 nodes run <c>NumSteps</c>=4 times per
/// utterance - ~34,484 node-executions - and this engine costs ~1 ms per node of HOST orchestration in a
/// browser. Kokoro is <b>2,464 nodes, once</b>, with no separate vocoder.
/// </para>
/// <para>
/// ⭐ Nothing had to be added to the engine to run it. MEASURED with
/// <c>DemoConsole -- OPCHECK</c>: 49 operators used, 49 supported, 100%. And nothing had to be added to
/// the frontend either - <c>SpawnDev.Phonemizer</c> exists because the ecosystem phonemizes with
/// espeak-ng (GPL-3), it names Kokoro as a model it unblocks, and MEASURED over five sentences its IPA
/// covers Kokoro's vocabulary completely: 271 of 271 symbols, zero dropped.
/// </para>
/// <para>
/// 🔴 IT CANNOT CLONE A VOICE. That is the whole trade against ZipVoice, and it is not a gap to be closed
/// later - Kokoro speaks the voices it ships with and no others. A product that wants both runs this for
/// the default voice and keeps a cloner for the one somebody recorded.
/// </para>
/// </remarks>
public sealed class KokoroPipeline : IDisposable
{
    /// <summary>Kokoro's output rate. Fixed by the model, not a setting.</summary>
    public const int OutputSampleRate = 24_000;

    private readonly InferenceSession _session;
    private readonly Accelerator _accelerator;
    private readonly string _tokensInput;
    private readonly string _styleInput;
    private readonly string _speedInput;

    private KokoroPipeline(InferenceSession session, Accelerator accelerator)
    {
        _session = session;
        _accelerator = accelerator;
        // 🔴 RESOLVED FROM THE GRAPH, NOT HARDCODED. Two exports of this same model are in circulation and
        // they do not agree on names: onnx-community's serves `input_ids` -> `waveform`, KokoroSharp's
        // ships `tokens` -> `audio`. Same 2,463-node graph, same operators, same weights. Hardcoding
        // either one fails against the other with "Tensor 'input_ids' not found (needed by Shape)", which
        // reads like a corrupt model rather than a naming difference - I lost a run to exactly that,
        // having inspected one file and executed another.
        _tokensInput = Resolve(session, "tokens input", "input_ids", "tokens");
        _styleInput = Resolve(session, "style input", "style", "ref_s");
        _speedInput = Resolve(session, "speed input", "speed");
    }

    /// <summary>First input name the graph actually declares from a list of known aliases.</summary>
    private static string Resolve(InferenceSession session, string what, params string[] aliases)
    {
        foreach (var alias in aliases)
            if (Array.IndexOf(session.InputNames, alias) >= 0) return alias;
        throw new InvalidOperationException(
            $"this Kokoro export has no {what}: looked for {string.Join(" or ", aliases)}, and the graph "
            + $"declares [{string.Join(", ", session.InputNames)}]. Either it is not Kokoro, or it is an "
            + "export with names nothing here knows - add the alias rather than renaming the model.");
    }

    /// <summary>Load the graph from raw ONNX bytes.</summary>
    /// <remarks>
    /// ⚠️ Desktop and tests. In a browser use <see cref="CreateFromStreamAsync"/> instead: this model is
    /// ~326 MB as fp32, and a <c>byte[]</c> of it lands on the .NET WASM managed heap, which is small and
    /// which the crossing itself is the cost of filling. The stream overload hands the bytes to the
    /// session without ever materialising them managed-side.
    /// </remarks>
    public static KokoroPipeline Create(Accelerator accelerator, byte[] modelOnnx)
        => new(InferenceSession.CreateFromFile(accelerator, modelOnnx), accelerator);

    /// <summary>Load the graph from a model STREAM - the browser path.</summary>
    /// <remarks>
    /// ⭐ This is the form to use behind an <c>IModelSource</c>/OPFS cache: the weights go from the cache
    /// to the GPU without a managed copy of the whole model in between. Mirrors how the ZipVoice and
    /// speech engines load.
    /// </remarks>
    public static async Task<KokoroPipeline> CreateFromStreamAsync(Accelerator accelerator, Stream modelOnnx,
        CancellationToken ct = default)
        => new(await InferenceSession.CreateFromOnnxStreamAsync(accelerator, modelOnnx, ct: ct)
            .ConfigureAwait(false), accelerator);

    /// <summary>The loaded session, for diagnostics.</summary>
    public InferenceSession Session => _session;

    /// <summary>
    /// Speak a phoneme sequence in a voice.
    /// </summary>
    /// <param name="phonemes">IPA symbols, e.g. from <c>EnglishPhonemizer.ToSymbols</c>.</param>
    /// <param name="voice">The voice pack - see <see cref="KokoroVoicePack"/>.</param>
    /// <param name="speed">1.0 is the trained rate; the model takes it as an input rather than resampling.</param>
    /// <param name="ct">Cancellation.</param>
    /// <remarks>
    /// <para>
    /// ⚠️ THE STYLE ROW DEPENDS ON THE TOKEN COUNT, so it is selected here - after tokenizing, including
    /// the two padding tokens - and never cached across utterances. A voice pack is a 510-row table, not a
    /// vector; using one row for everything makes short and long lines drift in timbre.
    /// </para>
    /// <para>
    /// ⚠️ Token ids are uploaded as floats because that is how this engine represents every tensor,
    /// including integral ones - the same conversion ZipVoice's encoder does. The graph declares
    /// <c>tokens</c> as int64; the values are small integers and survive the round trip exactly.
    /// </para>
    /// </remarks>
    public async Task<KokoroAudio> SpeakAsync(IEnumerable<string> phonemes, KokoroVoicePack voice,
        float speed = 1.0f, CancellationToken ct = default)
    {
        if (voice == null) throw new ArgumentNullException(nameof(voice));

        var (tokens, dropped) = KokoroTokenizer.Encode(phonemes);
        if (tokens.Length <= 2)
            throw new ArgumentException(
                "there is nothing to say - every phoneme was empty or unmappable, so the sequence is just "
                + "the two padding tokens", nameof(phonemes));

        return await SpeakTokensAsync(tokens, voice, speed, dropped, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Synthesise from token ids directly, skipping this pipeline's phonemizer and tokenizer.
    /// </summary>
    /// <param name="tokens">
    /// Vocabulary ids INCLUDING the padding token at both ends - exactly what the model's
    /// <c>input_ids</c> expects. <see cref="KokoroTokenizer.Encode"/> produces them from phonemes.
    /// </param>
    /// <param name="droppedPhonemes">
    /// Carried through to <see cref="KokoroAudio.DroppedPhonemes"/> for a caller that did its own
    /// tokenizing and knows the count; 0 when there is nothing to report.
    /// </param>
    /// <remarks>
    /// ⚠️ THE POINT OF THIS OVERLOAD IS COMPARABILITY. Rendering from TEXT compares two front ends as well
    /// as two inference engines, so a phonemizer change reads as an engine regression and vice versa.
    /// Feeding fixed ids pins the front end out of the question, which is what makes a reference waveform
    /// from another runtime mean something. It is also the entry point for a caller who has their own
    /// phonemizer, or a language this one does not cover.
    /// </remarks>
    public async Task<KokoroAudio> SpeakTokensAsync(IReadOnlyList<long> tokens, KokoroVoicePack voice,
        float speed = 1.0f, int droppedPhonemes = 0, CancellationToken ct = default)
    {
        if (voice == null) throw new ArgumentNullException(nameof(voice));
        if (tokens == null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count <= 2)
            throw new ArgumentException(
                "there is nothing to say - a sequence of two or fewer ids is just the padding tokens",
                nameof(tokens));

        var dropped = droppedPhonemes;
        var style = voice.StyleFor(tokens.Count).ToArray();

        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var tokenBuffer = _accelerator.Allocate1D(ToFloats(tokens));
        using var styleBuffer = _accelerator.Allocate1D(style);
        using var speedBuffer = _accelerator.Allocate1D(new[] { speed });

        var inputs = new Dictionary<string, Tensor>
        {
            [_tokensInput] = new Tensor(tokenBuffer.View, new[] { 1, tokens.Count }),
            [_styleInput] = new Tensor(styleBuffer.View, new[] { 1, KokoroVoicePack.Dim }),
            [_speedInput] = new Tensor(speedBuffer.View, new[] { 1 }),
        };

        Dictionary<string, Tensor> outputs;
        try
        {
            outputs = await _session.RunAsync(inputs).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // ⚠️ Name the shapes. A shape error surfaces as the failing operator with no indication of
            // what was fed in, and "tokens[1,58] style[1,256] speed[1]" is the whole diagnosis.
            var shapes = string.Join(", ", inputs.Select(kv => $"{kv.Key}[{string.Join(",", kv.Value.Shape)}]"));
            throw new InvalidOperationException($"Kokoro failed with inputs {shapes}: {ex.Message}", ex);
        }
        ct.ThrowIfCancellationRequested();

        var samples = await ReadAsync(outputs[_session.OutputNames[0]]).ConfigureAwait(false);
        clock.Stop();
        return new KokoroAudio(samples, OutputSampleRate, tokens.Count, dropped,
            clock.Elapsed.TotalMilliseconds);
    }

    private static async Task<float[]> ReadAsync(Tensor tensor)
    {
        var host = new float[tensor.ElementCount];
        var data = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync<float>(tensor.Data);
        Array.Copy(data, host, Math.Min(data.Length, host.Length));
        return host;
    }

    private static float[] ToFloats(IReadOnlyList<long> values)
    {
        var floats = new float[values.Count];
        for (var i = 0; i < values.Count; i++) floats[i] = values[i];
        return floats;
    }

    /// <summary>Release the graph.</summary>
    public void Dispose() => _session.Dispose();
}
