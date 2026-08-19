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
