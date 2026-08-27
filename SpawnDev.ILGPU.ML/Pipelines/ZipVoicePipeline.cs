using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>The three ONNX graphs ZipVoice is made of, behind one interface.</summary>
/// <remarks>
/// The orchestration around these graphs - mel features, the flow-matching loop, the inverse STFT - is
/// where a port goes wrong, and it is identical whichever engine runs the graphs. Keeping the engine
/// behind an interface means that orchestration is written ONCE and can be run on a reference engine and
/// on ours, so a wrong output can be attributed: same result on both engines means the algorithm is
/// wrong, different results means the engine is.
///
/// Asynchronous because the browser backends are: reading a WebGPU buffer back is a mapAsync, and a
/// synchronous interface would either deadlock there or quietly restrict this pipeline to the desktop.
/// </remarks>
public interface IZipVoiceGraphs : IDisposable
{
    /// <summary>
    /// Text encoder: turns the text and reference tokens into the per-frame conditioning, and in doing
    /// so DECIDES how many frames will be generated.
    /// </summary>
    /// <remarks>
    /// The generated length is not something the caller picks - duration prediction lives inside this
    /// graph, which is why the returned frame count drives the shape of everything downstream.
    /// </remarks>
    /// <param name="tokens">Phoneme/pinyin token ids of the text to speak.</param>
    /// <param name="promptTokens">Token ids of the reference clip's transcript.</param>
    /// <param name="promptFeatureFrames">Number of mel frames in the reference clip.</param>
    /// <param name="speed">Speaking rate multiplier.</param>
    Task<ZipVoiceEncoding> RunEncoderAsync(
        long[] tokens, long[] promptTokens, long promptFeatureFrames, float speed);

    /// <summary>
    /// Flow-matching decoder: the velocity field of the ODE, evaluated at one timestep.
    /// </summary>
    /// <remarks>
    /// Classifier-free guidance is applied INSIDE this graph - it takes the guidance scale as an input
    /// rather than requiring two passes and a blend outside.
    /// </remarks>
    Task<float[]> RunDecoderAsync(
        float t, float[] x, float[] textCondition, float[] speechCondition,
        float guidanceScale, int numFrames, int featDim);

    /// <summary>
    /// Vocos vocoder: mel to a complex spectrogram, expressed as a magnitude and the cosine and sine of
    /// the phase.
    /// </summary>
    /// <param name="melChannelsFirst">Log-mel, [channels, frames] row-major, WITHOUT the feature scale.</param>
    /// <param name="channels">Mel channels.</param>
    /// <param name="frames">Mel frames.</param>
    Task<ZipVoiceSpectrum> RunVocoderAsync(float[] melChannelsFirst, int channels, int frames);
}

/// <summary>The encoder's conditioning output and the generated length it implies.</summary>
public readonly record struct ZipVoiceEncoding(float[] TextCondition, int NumFrames, int FeatDim);

/// <summary>
/// The vocoder's output: a complex spectrogram in polar-ish form, [bins, frames] row-major each.
/// </summary>
/// <remarks>
/// Vocos deliberately stops one step short of a waveform. Predicting magnitude and phase and letting a
/// plain inverse STFT do the resynthesis is what makes it fast - there is no autoregressive sample
/// generation anywhere in it.
/// </remarks>
public readonly record struct ZipVoiceSpectrum(float[] Magnitude, float[] Cos, float[] Sin, int Bins, int Frames);

/// <summary>Synthesised audio plus the timings of each stage.</summary>
public record ZipVoiceResult(
    float[] Audio, int SampleRate,
    double EncoderMs, double DecoderMs, double VocoderMs, double TotalMs)
{
    /// <summary>Length of the generated audio in seconds.</summary>
    public double DurationSeconds => SampleRate > 0 ? (double)Audio.Length / SampleRate : 0;
}

/// <summary>
/// ZipVoice zero-shot voice cloning: speaks new text in the voice of a short reference clip.
/// </summary>
/// <remarks>
/// The pipeline is: encode text + reference transcript to a per-frame conditioning; start from gaussian
/// noise; integrate the flow-matching ODE for a handful of Euler steps to turn that noise into a mel;
/// drop the reference frames off the front; vocode the rest.
/// <para>
/// This class takes TOKEN IDS, not text. Turning English text into tokens needs an espeak-ng
/// grapheme-to-phoneme pass, which is a separate piece - the shipped lexicon covers Chinese only.
/// </para>
/// </remarks>
public sealed class ZipVoicePipeline : IDisposable
{
    private readonly IZipVoiceGraphs _graphs;

    /// <summary>Acoustic and sampling configuration.</summary>
    public ZipVoiceConfig Config { get; }

    /// <summary>
    /// Seed for the noise the ODE starts from, or null for a fresh random start each call.
    /// </summary>
    /// <remarks>
    /// Zero-shot flow matching re-samples noise every call, so output is NOT reproducible by default -
    /// two calls on identical input differ, and can differ audibly. Fixing the seed is what makes a
    /// numeric comparison against a reference engine possible at all.
    /// </remarks>
    public int? NoiseSeed { get; set; }

    /// <summary>
    /// Silence appended to the reference clip before analysis, in seconds.
    /// </summary>
    /// <remarks>
    /// Without it the model carries the last word of the reference into the start of the line it
    /// generates - the reference audio ends mid-breath, so the continuation it is asked to write begins
    /// inside that word. A quarter second of silence gives it somewhere to finish. Set to 0 to compare
    /// against implementations that do not pad.
    /// </remarks>
    public float ReferenceTailSilenceSeconds { get; set; } = 0.25f;

    public ZipVoicePipeline(IZipVoiceGraphs graphs, ZipVoiceConfig? config = null)
    {
        _graphs = graphs ?? throw new ArgumentNullException(nameof(graphs));
        Config = config ?? new ZipVoiceConfig();
    }

    /// <summary>
    /// Speak <paramref name="tokens"/> in the voice of a reference clip.
    /// </summary>
    /// <param name="tokens">Token ids of the text to speak.</param>
    /// <param name="promptTokens">Token ids of the reference clip's exact transcript.</param>
    /// <param name="referenceAudio">Reference clip, mono, in [-1, 1].</param>
    /// <param name="referenceSampleRate">Sample rate of the reference clip.</param>
    public Task<ZipVoiceResult> SynthesizeAsync(
        long[] tokens, long[] promptTokens, float[] referenceAudio, int referenceSampleRate)
    {
        var promptFeatures = ZipVoiceFeatures.ComputePromptFeatures(
            PadReferenceTail(referenceAudio, referenceSampleRate),
            referenceSampleRate, Config, out int promptFrames);
        return SynthesizeFromFeaturesAsync(tokens, promptTokens, promptFeatures, promptFrames);
    }

    /// <summary>
    /// Speak <paramref name="tokens"/> from reference features that have already been computed.
    /// </summary>
    /// <remarks>
    /// Worth having separately: the reference features are the expensive, unchanging part of a voice, so
    /// a speaking robot computes them once per voice rather than once per sentence.
    /// </remarks>
    public async Task<ZipVoiceResult> SynthesizeFromFeaturesAsync(
        long[] tokens, long[] promptTokens, float[] promptFeatures, int promptFrames)
    {
        if (tokens.Length == 0) throw new ArgumentException("No tokens to speak.", nameof(tokens));
        if (promptFrames <= 0) throw new ArgumentException("Reference features are empty.", nameof(promptFrames));

        var total = System.Diagnostics.Stopwatch.StartNew();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var encoding = await _graphs.RunEncoderAsync(tokens, promptTokens, promptFrames, Config.Speed);
        double encoderMs = sw.Elapsed.TotalMilliseconds;

        int numFrames = encoding.NumFrames, featDim = encoding.FeatDim;
        if (numFrames <= promptFrames)
            throw new InvalidOperationException(
                $"The encoder asked for {numFrames} frames but the reference alone is {promptFrames}; " +
                "nothing would be left to speak.");

        int count = numFrames * featDim;

        // The ODE starts from pure noise. Everything the output owes to the reference voice arrives
        // through the conditioning, not through the starting point.
        var x = GaussianNoise(count, NoiseSeed);

        // The reference mel occupies the front of the conditioning and zeros fill the rest: the model is
        // completing a spectrogram whose beginning it has been given.
        var speechCondition = new float[count];
        Array.Copy(promptFeatures, speechCondition, Math.Min(promptFeatures.Length, promptFrames * featDim));

        var timesteps = ZipVoiceFeatures.Timesteps(Config);

        sw.Restart();
        for (int step = 0; step < Config.NumSteps; step++)
        {
            float t = timesteps[step];
            float dt = timesteps[step + 1] - timesteps[step];
            var v = await _graphs.RunDecoderAsync(
                t, x, encoding.TextCondition, speechCondition, Config.GuidanceScale, numFrames, featDim);
            for (int i = 0; i < count; i++) x[i] += v[i] * dt;
        }
        double decoderMs = sw.Elapsed.TotalMilliseconds;

        // Drop the reference frames - the model regenerated them, but they are the voice we were GIVEN,
        // not the text we asked for.
        int keptFrames = numFrames - promptFrames;

        // Vocos wants [channels, frames] and unscaled log-mels, so transpose and undo the feature scale in
        // one pass.
        float invFeatScale = 1f / Config.FeatScale;
        var mel = new float[featDim * keptFrames];
        for (int f = 0; f < keptFrames; f++)
        {
            int src = (promptFrames + f) * featDim;
            for (int c = 0; c < featDim; c++)
                mel[c * keptFrames + f] = x[src + c] * invFeatScale;
        }

        sw.Restart();
        var spectrum = await _graphs.RunVocoderAsync(mel, featDim, keptFrames);
        var audio = Vocode(spectrum, Config);
        double vocoderMs = sw.Elapsed.TotalMilliseconds;

        total.Stop();
        return new ZipVoiceResult(
            audio, Config.SampleRate, encoderMs, decoderMs, vocoderMs, total.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Turn the vocoder's magnitude/cosine/sine output into a waveform.
    /// </summary>
    /// <remarks>
    /// The magnitude and the unit vector (cos, sin) multiply back into a rectangular complex
    /// spectrogram, which the inverse STFT resynthesises. The transpose is not incidental: the vocoder
    /// emits [bins, frames] and the inverse STFT reads frames as rows.
    /// </remarks>
    public static float[] Vocode(ZipVoiceSpectrum spectrum, ZipVoiceConfig config)
    {
        int bins = spectrum.Bins, frames = spectrum.Frames;
        var real = new float[frames * bins];
        var imag = new float[frames * bins];

        for (int b = 0; b < bins; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                float magnitude = spectrum.Magnitude[b * frames + f];
                real[f * bins + b] = magnitude * spectrum.Cos[b * frames + f];
                imag[f * bins + b] = magnitude * spectrum.Sin[b * frames + f];
            }
        }

        return AudioPreprocessor.Istft(
            real, imag, frames, config.NFft, config.HopLength, config.WinLength,
            center: true, normalized: false);
    }

    /// <summary>Standard normal noise, optionally from a fixed seed.</summary>
    /// <remarks>
    /// Box-Muller rather than a sum-of-uniforms approximation: the tails are the part that matters here,
    /// and an approximation would quietly narrow them.
    /// </remarks>
    public static float[] GaussianNoise(int count, int? seed = null)
    {
        var random = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var values = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();   // in (0, 1], so Log never sees zero
            double u2 = random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = 2.0 * Math.PI * u2;
            values[i] = (float)(radius * Math.Cos(angle));
            if (i + 1 < count) values[i + 1] = (float)(radius * Math.Sin(angle));
        }
        return values;
    }

    private float[] PadReferenceTail(float[] referenceAudio, int referenceSampleRate)
    {
        int silence = (int)(ReferenceTailSilenceSeconds * referenceSampleRate);
        if (silence <= 0) return referenceAudio;
        var padded = new float[referenceAudio.Length + silence];
        Array.Copy(referenceAudio, padded, referenceAudio.Length);
        return padded;
    }

    /// <summary>Releases this pipeline's own resources. The graphs are NOT disposed.</summary>
    /// <remarks>
    /// The graphs are handed in by the caller and belong to the caller, exactly as the accelerator does
    /// elsewhere in this library: whoever created it decides when it dies. This used to call
    /// <c>_graphs.Dispose()</c>, which made a pipeline lethal to the object it was given - constructing a
    /// second pipeline over the same graphs threw a NullReferenceException from inside onnxruntime on the
    /// first call, because the sessions had already been torn down by the first pipeline's disposal. That
    /// is not a hypothetical: it is the shape a caller takes when it renders the same graphs repeatedly
    /// with different tokens, which is what the phonemizer sensitivity gate does.
    /// The pipeline currently owns nothing else, so this body is empty rather than absent - it stays
    /// IDisposable so that acquiring scratch buffers later does not become a breaking API change.
    /// </remarks>
    public void Dispose() { }
}
