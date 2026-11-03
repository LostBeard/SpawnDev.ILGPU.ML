namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>The mel scale of <see cref="MelFilters.Build"/>.</summary>
public enum MelScale
{
    /// <summary>HTK: 2595 log10(1 + f / 700) (torchaudio's default).</summary>
    Htk,
    /// <summary>Slaney: linear below 1 kHz (3 f / 200), log above (librosa's default).</summary>
    Slaney,
}

/// <summary>
/// Triangular mel filter banks, a port of transformers' <c>audio_utils.mel_filter_bank</c> (the same arithmetic, in double): the CLAP, Whisper-style and
/// librosa feature extractors build their filters with it, so a model's features match when the filters are these. Dense, bin-major:
/// <c>filters[bin * mels + m]</c>, bins = n_fft / 2 + 1 (pass it to <see cref="SpectralFrontend.SetBandFilters"/>).
/// </summary>
public static class MelFilters
{
    /// <param name="bins">Frequency bins (n_fft / 2 + 1).</param>
    /// <param name="mels">Filters.</param>
    /// <param name="minHz">Lowest filter edge.</param>
    /// <param name="maxHz">Highest filter edge.</param>
    /// <param name="sampleRate">Sample rate (the bins span 0 .. sampleRate / 2).</param>
    /// <param name="slaneyNorm">Divide each filter by half its width in Hz (librosa's norm="slaney": equal area).</param>
    /// <param name="scale">The mel scale.</param>
    public static float[] Build(int bins, int mels, double minHz, double maxHz, int sampleRate, bool slaneyNorm, MelScale scale)
    {
        // filter_freqs = mel_to_hertz(linspace(hertz_to_mel(min), hertz_to_mel(max), mels + 2))
        double melMin = HzToMel(minHz, scale), melMax = HzToMel(maxHz, scale);
        var filterFreqs = new double[mels + 2];
        for (int i = 0; i < mels + 2; i++) filterFreqs[i] = MelToHz(melMin + (melMax - melMin) * i / (mels + 1), scale);
        // fft_freqs = linspace(0, sampleRate // 2, bins)
        var fftFreqs = new double[bins];
        for (int k = 0; k < bins; k++) fftFreqs[k] = (double)(sampleRate / 2) * k / (bins - 1);
        var f = new float[bins * mels];
        for (int m = 0; m < mels; m++)
        {
            double lo = filterFreqs[m], c = filterFreqs[m + 1], hi = filterFreqs[m + 2];
            double norm = slaneyNorm ? 2.0 / (hi - lo) : 1.0;
            for (int k = 0; k < bins; k++)
            {
                // _create_triangular_filter_bank: max(0, min(down, up)), down = (fft - lo) / (c - lo), up = (hi - fft) / (hi - c)
                double down = (fftFreqs[k] - lo) / (c - lo), up = (hi - fftFreqs[k]) / (hi - c);
                double w = Math.Max(0, Math.Min(down, up));
                f[k * mels + m] = (float)(w * norm);
            }
        }
        return f;
    }

    /// <summary>Hz to mel on <paramref name="scale"/>.</summary>
    public static double HzToMel(double hz, MelScale scale)
    {
        if (scale == MelScale.Htk) return 2595.0 * Math.Log10(1.0 + hz / 700.0);
        const double minLogHz = 1000.0, minLogMel = 15.0;
        double logStep = 27.0 / Math.Log(6.4);
        return hz >= minLogHz ? minLogMel + Math.Log(hz / minLogHz) * logStep : 3.0 * hz / 200.0;
    }

    /// <summary>Mel to Hz on <paramref name="scale"/>.</summary>
    public static double MelToHz(double mel, MelScale scale)
    {
        if (scale == MelScale.Htk) return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
        const double minLogHz = 1000.0, minLogMel = 15.0;
        double logStep = Math.Log(6.4) / 27.0;
        return mel >= minLogMel ? minLogHz * Math.Exp(logStep * (mel - minLogMel)) : 200.0 * mel / 3.0;
    }
}
