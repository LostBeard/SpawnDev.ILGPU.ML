namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Everything ZipVoice needs that is not one of its three ONNX graphs: the acoustic configuration,
/// the reference-clip mel features it clones a voice from, and the flow-matching timestep schedule.
/// </summary>
/// <remarks>
/// ZipVoice is a zero-shot voice CLONER. It is conditioned on a short reference clip plus that
/// clip's exact transcript, so the features here are not merely preprocessing - the mel of the
/// reference clip IS the voice. Get the mel convention wrong and the model still runs, still emits
/// speech, and simply does not sound like the reference.
/// </remarks>
public sealed class ZipVoiceConfig
{
    /// <summary>Output sample rate; the vocos vocoder is trained at 24 kHz.</summary>
    public int SampleRate { get; init; } = 24000;

    /// <summary>FFT size for the mel analysis.</summary>
    public int NFft { get; init; } = 1024;

    /// <summary>Hop between mel frames, in samples - one frame per 256 / 24000 s.</summary>
    public int HopLength { get; init; } = 256;

    /// <summary>Analysis window length; equal to <see cref="NFft"/> for this model.</summary>
    public int WinLength { get; init; } = 1024;

    /// <summary>Mel bands. This is also the model's feature dimension.</summary>
    public int NumMels { get; init; } = 100;

    /// <summary>
    /// Scale applied to the log-mel features on the way in, and undone on the way out.
    /// </summary>
    /// <remarks>
    /// The model was trained on log-mels multiplied by this, so it is part of the model contract, not
    /// a tuning knob. The vocoder is NOT - it expects unscaled log-mels, which is why the decoder
    /// output is divided by this again before vocoding.
    /// </remarks>
    public float FeatScale { get; init; } = 0.1f;

    /// <summary>
    /// Warps the flow-matching timestep schedule towards t = 0, where the trajectory curves most.
    /// </summary>
    /// <remarks>
    /// With the default 0.5 the four steps are not evenly spaced in t: they cluster early, spending
    /// resolution where the velocity field changes fastest. Raising it towards 1 gives a uniform
    /// schedule.
    /// </remarks>
    public float TShift { get; init; } = 0.5f;

    /// <summary>
    /// Reference clips quieter than this RMS are scaled up to it before analysis.
    /// </summary>
    /// <remarks>
    /// One-sided on purpose: a quiet clip is brought up so its mel sits in the range the model was
    /// trained on, but a loud one is left alone rather than being squashed.
    /// </remarks>
    public float TargetRms { get; init; } = 0.1f;

    /// <summary>Classifier-free guidance scale for the flow-matching decoder.</summary>
    public float GuidanceScale { get; init; } = 1.0f;

    /// <summary>
    /// Euler steps taken through the flow-matching ODE.
    /// </summary>
    /// <remarks>
    /// Four is right for the DISTILL models, which are trained to land in a handful of steps. The
    /// non-distilled models need considerably more.
    /// </remarks>
    public int NumSteps { get; init; } = 4;

    /// <summary>Speaking rate multiplier; the encoder uses it to set the generated length.</summary>
    public float Speed { get; init; } = 1.0f;

    /// <summary>Number of FFT bins the mel filterbank consumes, <c>NFft / 2 + 1</c>.</summary>
    public int FreqBins => NFft / 2 + 1;
}

/// <summary>
/// Reference-clip feature extraction and the flow-matching schedule for ZipVoice.
/// </summary>
public static class ZipVoiceFeatures
{
    /// <summary>
    /// Compute the log-mel features of a clip, laid out [frames, mels] row-major - the layout the
    /// model's speech_condition input wants.
    /// </summary>
    /// <remarks>
    /// Resamples to the model rate first if needed, then a centred STFT with a periodic Hann window,
    /// MAGNITUDE (not power) through a librosa/HTK filterbank, then log with a 1e-10 floor and the
    /// model's feature scale.
    /// <para>
    /// Magnitude rather than power is easy to get wrong and silent when wrong - squaring would push
    /// every value through the log as a doubling, which reads as a plausible-looking mel that is
    /// uniformly the wrong contrast.
    /// </para>
    /// </remarks>
    /// <param name="samples">Mono samples in [-1, 1].</param>
    /// <param name="sampleRate">Sample rate of <paramref name="samples"/>.</param>
    /// <param name="config">Acoustic configuration.</param>
    /// <param name="numFrames">Number of frames produced.</param>
    public static float[] ComputeMel(float[] samples, int sampleRate, ZipVoiceConfig config, out int numFrames)
    {
        if (samples.Length == 0) throw new ArgumentException("No samples given.", nameof(samples));

        if (sampleRate != config.SampleRate)
            samples = AudioPreprocessor.Resample(samples, sampleRate, config.SampleRate);

        var magnitudes = AudioPreprocessor.ComputeSTFT(
            samples, config.NFft, config.HopLength, center: true, periodicWindow: true);

        numFrames = magnitudes.GetLength(0);
        int freqBins = magnitudes.GetLength(1);

        var filters = MelFilters(config);
        var mel = new float[numFrames * config.NumMels];

        for (int f = 0; f < numFrames; f++)
        {
            int rowOffset = f * config.NumMels;
            for (int m = 0; m < config.NumMels; m++)
            {
                float sum = 0f;
                for (int k = 0; k < freqBins; k++)
                    sum += filters[m, k] * magnitudes[f, k];
                mel[rowOffset + m] = MathF.Log(sum + 1e-10f) * config.FeatScale;
            }
        }

        return mel;
    }

    /// <summary>
    /// Remove dead air from a reference clip, because to ZipVoice dead air is SLOW SPEECH.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ THE DEFECT THIS EXISTS TO FIX, and it is worth understanding before touching any of the numbers,
    /// because nothing about the symptom points at silence. Walking the forward cone of the encoder's
    /// <c>prompt_features_len</c> input in <c>text_encoder.onnx</c>:
    /// </para>
    /// <code>
    /// Cast -> Div(by len(prompt_tokens)) -> Mul(by len(prompt_tokens) + len(tokens)) -> Div(by speed) -> Ceil
    /// </code>
    /// <para>
    /// The model measures FRAMES PER PROMPT TOKEN from the reference and multiplies it by the total token
    /// count. That ratio is the entire duration prediction. Copying the speaker's rate is deliberate - it
    /// is how ZipVoice clones delivery - but <c>prompt_features_len</c> counts every mel frame of the
    /// reference and silence has mel frames too. A reference clip that is half silence declares a speaking
    /// rate half of what the speaker used, and every generated syllable is stretched to match.
    /// </para>
    /// <para>
    /// ⚠️ Nothing downstream can notice. The shapes are right, the output is speech, the transcript still
    /// matches; it simply sounds like a slur. MEASURED (tools/zipvoice-ref-rate, CUDA, a 4.00 s reference
    /// and a 45-token line):
    /// </para>
    /// <list type="table">
    ///   <item><term>reference as recorded</term><description>3.20 s generated</description></item>
    ///   <item><term>+2 s of silence at each end</term><description>6.20 s - <b>1.94x slower</b></description></item>
    ///   <item><term>+2 s of silence in the middle</term><description>4.69 s</description></item>
    ///   <item><term>any of those, through this method</term><description>3.06 - 3.23 s</description></item>
    /// </list>
    /// <para>
    /// ⚠️ Internal pauses are CAPPED, not deleted, and the distinction is the design. A pause carries
    /// rhythm and the model clones rhythm, so splicing every gap to zero would clone a speaker who never
    /// breathes. A two-second think in the middle of a two-word reference is not rhythm, it is dead air
    /// being counted as speech. <paramref name="maxPauseSeconds"/> is the line between them.
    /// </para>
    /// <para>
    /// ⚠️ The gate is RELATIVE to the clip's own loudest frame, never an absolute level. A reference comes
    /// off whatever microphone is in whatever room, and an absolute threshold that works in one room is a
    /// threshold that silently mis-trims in the next. It is also frame RMS rather than sample peak, so one
    /// click cannot set the reference level for the whole clip.
    /// </para>
    /// <para>
    /// ⚠️ An energy gate rather than Silero, deliberately, even though this library ships Silero and Silero
    /// is the better instrument in general. Two reasons, and the second is the one that decided it. A TTS
    /// pipeline that cannot clone at the right speed without a second 643 KB model download is a pipeline
    /// every consumer gets wrong by default. And MEASURED on the same fixture, this gate was the MORE
    /// stable of the two: 382 prompt frames whether the clip arrived clean, +2 s padded or +4 s padded,
    /// where the Silero trim gave 393/391/388 because its own 512-sample framing shifts with the padding.
    /// A caller that already holds a detector is free to trim first and set
    /// <c>ZipVoicePipeline.TrimReferenceSilence</c> to false.
    /// </para>
    /// </remarks>
    /// <param name="samples">The reference clip, mono, in [-1, 1].</param>
    /// <param name="sampleRate">Sample rate of <paramref name="samples"/>.</param>
    /// <param name="gateDbBelowPeak">How far below the loudest frame still counts as speech.</param>
    /// <param name="keepSeconds">Audio kept either side of speech, so a soft onset survives the gate.</param>
    /// <param name="maxPauseSeconds">
    /// How much of an over-long internal pause survives, on top of the <paramref name="keepSeconds"/>
    /// margin already held either side of the speech around it - so a long pause is reduced to roughly
    /// <c>maxPauseSeconds + 2 * keepSeconds</c>, not to <paramref name="maxPauseSeconds"/> exactly. Pauses
    /// shorter than that are untouched.
    /// </param>
    /// <returns>
    /// The trimmed clip, or <paramref name="samples"/> unchanged when the gate found nothing it could act
    /// on - a clip too short to frame, digital silence, or a result so short the gate has clearly misfired.
    /// Returning the input is the right failure: a gutted reference clones badly and does it silently.
    /// </returns>
    public static float[] TrimReferenceSilence(
        float[] samples, int sampleRate,
        double gateDbBelowPeak = 35, double keepSeconds = 0.06, double maxPauseSeconds = 0.20)
    {
        if (samples == null) throw new ArgumentNullException(nameof(samples));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));

        int win = Math.Max(1, sampleRate / 100);            // 10 ms analysis frames
        int frames = samples.Length / win;
        if (frames < 3) return samples;

        var rms = new double[frames];
        double peak = 0;
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            int end = (f + 1) * win;
            for (int i = f * win; i < end; i++) sum += (double)samples[i] * samples[i];
            rms[f] = Math.Sqrt(sum / win);
            if (rms[f] > peak) peak = rms[f];
        }
        if (peak <= 0) return samples;                      // digital silence: nothing to be right about

        double gate = peak * Math.Pow(10, -gateDbBelowPeak / 20.0);
        int keep = (int)Math.Round(keepSeconds * 100);
        int cap = (int)Math.Round(maxPauseSeconds * 100);

        // Widen every loud frame by the keep margin, so a low-energy onset - an /h/, a trailing fricative -
        // is inside the kept region rather than shaved off it.
        var take = new bool[frames];
        for (int f = 0; f < frames; f++)
        {
            if (rms[f] < gate) continue;
            int lo = Math.Max(0, f - keep), hi = Math.Min(frames - 1, f + keep);
            for (int k = lo; k <= hi; k++) take[k] = true;
        }

        int first = Array.IndexOf(take, true);
        if (first < 0) return samples;                      // gate found no speech at all: leave it alone
        int last = Array.LastIndexOf(take, true);

        // Everything before `first` and after `last` is leading and trailing dead air and goes entirely.
        // Between them, a run of quiet frames longer than the cap is truncated to the cap.
        var kept = new List<float>(samples.Length);
        int run = 0;
        for (int f = first; f <= last; f++)
        {
            if (take[f]) run = 0;
            else if (++run > cap) continue;
            int end = (f + 1) * win;
            for (int i = f * win; i < end; i++) kept.Add(samples[i]);
        }

        // A result this short means the gate misfired on something it does not understand. Hand back what
        // we were given: a wrong speaking rate is a bad clone, an empty reference is no clone at all.
        if (kept.Count < win * 10) return samples;
        return kept.ToArray();
    }

    /// <summary>
    /// Compute the mel features of the reference clip, level-matched first.
    /// </summary>
    /// <remarks>
    /// The RMS step is what stops a quietly-recorded reference from cloning as a whisper: the model
    /// reads absolute level out of the log-mel, so a clip 20 dB down is a different conditioning
    /// signal, not just a quieter one.
    /// </remarks>
    public static float[] ComputePromptFeatures(
        float[] samples, int sampleRate, ZipVoiceConfig config, out int numFrames)
    {
        if (samples.Length == 0) throw new ArgumentException("Reference audio is empty.", nameof(samples));

        double sumSq = 0;
        foreach (var s in samples) sumSq += (double)s * s;
        float rms = (float)Math.Sqrt(sumSq / samples.Length);

        if (rms > 0f && rms < config.TargetRms)
        {
            float scale = config.TargetRms / rms;
            var scaled = new float[samples.Length];
            for (int i = 0; i < samples.Length; i++) scaled[i] = samples[i] * scale;
            samples = scaled;
        }

        return ComputeMel(samples, sampleRate, config, out numFrames);
    }

    /// <summary>
    /// The flow-matching timesteps, <c>numSteps + 1</c> of them, from 0 to 1.
    /// </summary>
    /// <remarks>
    /// t_shift warps a uniform grid: <c>shift * t / (1 + (shift - 1) * t)</c>. At shift = 1 it is the
    /// identity; below 1 it pulls the samples towards 0.
    /// </remarks>
    public static float[] Timesteps(ZipVoiceConfig config)
    {
        var timesteps = new float[config.NumSteps + 1];
        for (int i = 0; i <= config.NumSteps; i++)
        {
            float t = (float)i / config.NumSteps;
            timesteps[i] = config.TShift * t / (1f + (config.TShift - 1f) * t);
        }
        return timesteps;
    }

    /// <summary>Mel filterbank for a configuration, cached because it depends only on the config.</summary>
    public static float[,] MelFilters(ZipVoiceConfig config)
    {
        var key = (config.NumMels, config.NFft, config.SampleRate);
        lock (FilterCacheLock)
        {
            if (FilterCache.TryGetValue(key, out var cached)) return cached;
            var filters = AudioPreprocessor.GenerateMelFilterbankLibrosaHtk(
                config.NumMels, config.NFft / 2, config.SampleRate, config.NFft,
                lowFreq: 0f, highFreq: config.SampleRate / 2f);
            FilterCache[key] = filters;
            return filters;
        }
    }

    private static readonly Dictionary<(int, int, int), float[,]> FilterCache = new();
    private static readonly object FilterCacheLock = new();
}
