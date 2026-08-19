namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Spectral analysis/synthesis pieces used by the vocoder-based speech models (ZipVoice, Vocos):
/// a PERIODIC Hann window, a librosa-style mel filterbank on the HTK scale, and the inverse STFT.
/// </summary>
/// <remarks>
/// These are deliberately separate from the Whisper preprocessing, which needs different conventions
/// in every one of the three: a SYMMETRIC window, the SLANEY mel scale, and no resynthesis at all.
/// Mixing the two is how a mel port silently hands a model a spectrum it was never trained on - the
/// failure is not an exception, it is unintelligible output.
///
/// The conventions here are the ones kaldi-native-fbank uses (the library sherpa-onnx runs ZipVoice
/// through), which are in turn torch's: hann_window(periodic: true), stft(center: true,
/// pad_mode: "reflect"), and istft's weighted overlap-add.
/// </remarks>
public static partial class AudioPreprocessor
{
    /// <summary>
    /// Generate a Hann window, optionally PERIODIC (torch.hann_window's default) rather than symmetric.
    /// </summary>
    /// <remarks>
    /// The only difference is the denominator - <c>size</c> instead of <c>size - 1</c> - but it is not
    /// cosmetic: a periodic window is the one that satisfies the constant-overlap-add condition, so it
    /// is what an analysis/synthesis pair must use for the ISTFT to reconstruct the signal. Whisper's
    /// existing <see cref="GenerateHannWindow(int)"/> stays symmetric, unchanged.
    /// </remarks>
    public static float[] GenerateHannWindow(int size, bool periodic)
    {
        var window = new float[size];
        float denom = periodic ? size : size - 1;
        for (int i = 0; i < size; i++)
            window[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / denom);
        return window;
    }

    /// <summary>Hz to mel on the HTK scale, in kaldi-native-fbank's exact form.</summary>
    /// <remarks>
    /// Mathematically identical to <see cref="HzToMel"/> (2595 * log10 and 1127 * ln are the same
    /// curve), but written the way the reference implementation writes it so the float rounding matches
    /// it bit for bit rather than merely closely.
    /// </remarks>
    public static float HzToMelHtk(float hz) => 1127f * MathF.Log(1f + hz / 700f);

    /// <summary>Mel to Hz on the HTK scale, inverse of <see cref="HzToMelHtk"/>.</summary>
    public static float MelToHzHtk(float mel) => 700f * (MathF.Exp(mel / 1127f) - 1f);

    /// <summary>
    /// Generate a librosa-style mel filterbank on the HTK scale with NO area normalization, returned as
    /// [numBins, numFftBins + 1].
    /// </summary>
    /// <remarks>
    /// This is the is_librosa=true, use_slaney_mel_scale=false, norm="" filterbank - the combination
    /// ZipVoice's mel features are built with.
    /// <para>
    /// The distinction from the classic HTK bank is that the triangle edges live at CONTINUOUS
    /// frequencies rather than being floored to integer FFT bins, so a bin's weight is interpolated
    /// between its neighbours instead of snapped. Flooring shifts every band edge down by up to one bin
    /// width and visibly changes the low bands, where the triangles are only a few bins wide to begin
    /// with.
    /// </para>
    /// </remarks>
    /// <param name="numBins">Number of mel bands (100 for ZipVoice).</param>
    /// <param name="numFftBins">Half the padded window size - the matrix gets numFftBins + 1 columns.</param>
    /// <param name="sampleRate">Sample rate of the audio the STFT was taken from.</param>
    /// <param name="paddedWindowSize">FFT size the bin spacing is derived from (1024 for ZipVoice).</param>
    /// <param name="lowFreq">Low edge of the lowest band, Hz.</param>
    /// <param name="highFreq">High edge of the highest band, Hz; 0 or less means Nyquist.</param>
    public static float[,] GenerateMelFilterbankLibrosaHtk(
        int numBins, int numFftBins, int sampleRate, int paddedWindowSize,
        float lowFreq = 0f, float highFreq = 0f)
    {
        if (numBins < 3) throw new ArgumentOutOfRangeException(nameof(numBins), "Must have at least 3 mel bins.");

        float nyquist = 0.5f * sampleRate;
        if (highFreq <= 0f) highFreq = nyquist;
        if (lowFreq < 0f || lowFreq >= nyquist || highFreq > nyquist || highFreq <= lowFreq)
            throw new ArgumentException($"Bad mel range: low {lowFreq}, high {highFreq}, nyquist {nyquist}.");

        float fftBinWidth = (float)sampleRate / paddedWindowSize;
        float melLow = HzToMelHtk(lowFreq);
        float melHigh = HzToMelHtk(highFreq);

        // numBins + 1, not numBins: the outermost triangles spill past the band edges, so the spacing is
        // set by the number of GAPS between centres rather than the number of centres.
        float melDelta = (melHigh - melLow) / (numBins + 1);

        var filters = new float[numBins, numFftBins + 1];
        for (int bin = 0; bin < numBins; bin++)
        {
            float leftHz = MelToHzHtk(melLow + bin * melDelta);
            float centerHz = MelToHzHtk(melLow + (bin + 1) * melDelta);
            float rightHz = MelToHzHtk(melLow + (bin + 2) * melDelta);

            for (int i = 0; i <= numFftBins; i++)
            {
                float hz = fftBinWidth * i;
                // Strictly inside the triangle - a bin sitting exactly on an edge contributes nothing,
                // which is what keeps adjacent bands from double-counting it.
                if (hz <= leftHz || hz >= rightHz) continue;
                filters[bin, i] = hz <= centerHz
                    ? (hz - leftHz) / (centerHz - leftHz)
                    : (rightHz - hz) / (rightHz - centerHz);
            }
        }

        return filters;
    }

    /// <summary>
    /// Inverse STFT: turn a complex spectrogram back into a waveform by weighted overlap-add, matching
    /// torch.istft.
    /// </summary>
    /// <remarks>
    /// Each frame is inverse-transformed, re-windowed, and summed into place, then the whole signal is
    /// divided by the summed SQUARE of the window at each sample. That denominator is what makes the
    /// reconstruction exact: the analysis pass already multiplied by the window once, so overlap-add
    /// without it leaves the signal amplitude-modulated at the frame rate.
    /// <para>
    /// Vocos does not output a waveform - it outputs a magnitude and the cosine/sine of a phase, which is
    /// a complex spectrogram in disguise. So this method, not the neural net, is the last stage of the
    /// vocoder.
    /// </para>
    /// </remarks>
    /// <param name="real">Real parts, [numFrames, nFft / 2 + 1] row-major.</param>
    /// <param name="imag">Imaginary parts, same layout.</param>
    /// <param name="numFrames">Number of frames in the spectrogram.</param>
    /// <param name="nFft">Transform size.</param>
    /// <param name="hopLength">Hop between frames, in samples.</param>
    /// <param name="winLength">Window length; must equal <paramref name="nFft"/> here.</param>
    /// <param name="center">
    /// The analysis was centred (reflect-padded by nFft / 2), so trim that padding back off.
    /// </param>
    /// <param name="normalized">The analysis divided by sqrt(nFft), so multiply it back in.</param>
    public static float[] Istft(
        float[] real, float[] imag, int numFrames,
        int nFft, int hopLength, int winLength,
        bool center = true, bool normalized = false)
    {
        if (winLength != nFft)
            throw new ArgumentException($"winLength ({winLength}) must equal nFft ({nFft}).", nameof(winLength));
        if (numFrames <= 0) return Array.Empty<float>();

        int freqBins = nFft / 2 + 1;
        if (real.Length < (long)numFrames * freqBins || imag.Length < (long)numFrames * freqBins)
            throw new ArgumentException($"Spectrogram is smaller than {numFrames} x {freqBins}.");

        var window = GenerateHannWindow(winLength, periodic: true);
        int numSamples = nFft + (numFrames - 1) * hopLength;
        var samples = new float[numSamples];
        var denominator = new float[numSamples];

        float inputScale = normalized ? MathF.Sqrt(nFft) : 1f;
        float invN = 1f / nFft;

        var re = new float[nFft];
        var im = new float[nFft];
        var scratchRe = new float[nFft];
        var scratchIm = new float[nFft];

        for (int f = 0; f < numFrames; f++)
        {
            int rowOffset = f * freqBins;

            // Rebuild the full spectrum from the half-spectrum: a real signal's transform is Hermitian,
            // so the upper bins are the conjugates of the lower ones mirrored about Nyquist. DC and
            // Nyquist are their own mirror and carry no imaginary part.
            re[0] = real[rowOffset] * inputScale;
            im[0] = 0f;
            re[nFft / 2] = real[rowOffset + nFft / 2] * inputScale;
            im[nFft / 2] = 0f;
            for (int k = 1; k < nFft / 2; k++)
            {
                float kr = real[rowOffset + k] * inputScale;
                float ki = imag[rowOffset + k] * inputScale;
                re[k] = kr;
                im[k] = ki;
                re[nFft - k] = kr;
                im[nFft - k] = -ki;
            }

            // Inverse transform by the conjugate identity: ifft(x) = conj(fft(conj(x))) / N. Reuses the
            // forward FFT rather than carrying a second transform implementation.
            for (int k = 0; k < nFft; k++) im[k] = -im[k];
            FFT(re, im, nFft, scratchRe, scratchIm);

            int start = f * hopLength;
            for (int i = 0; i < nFft; i++)
            {
                float w = window[i];
                // Imaginary part is zero to rounding for a Hermitian input, so only the real part is kept.
                samples[start + i] += re[i] * invN * w;
                denominator[start + i] += w * w;
            }
        }

        for (int i = 0; i < numSamples; i++)
            if (denominator[i] != 0f) samples[i] /= denominator[i];

        if (!center) return samples;

        int trim = nFft / 2;
        int keptLength = numSamples - 2 * trim;
        if (keptLength <= 0) return Array.Empty<float>();
        var trimmed = new float[keptLength];
        Array.Copy(samples, trim, trimmed, 0, keptLength);
        return trimmed;
    }
}
