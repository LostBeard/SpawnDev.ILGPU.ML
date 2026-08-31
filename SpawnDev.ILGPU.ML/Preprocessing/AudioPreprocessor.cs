namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// CPU-side audio preprocessing for ML model input.
/// Handles waveform manipulation, resampling, and spectrogram generation.
/// Designed for models like Whisper, wav2vec2, and audio classifiers.
/// </summary>
public static partial class AudioPreprocessor
{
    /// <summary>
    /// Whisper model sample rate (16kHz).
    /// </summary>
    public const int WhisperSampleRate = 16000;

    /// <summary>
    /// Whisper model input length (30 seconds at 16kHz = 480,000 samples).
    /// </summary>
    public const int WhisperMaxSamples = 480000;

    /// <summary>
    /// Whisper mel spectrogram bins.
    /// </summary>
    public const int WhisperMelBins = 80;

    /// <summary>
    /// Resample audio between two sample rates with a windowed-sinc (Lanczos-style) kernel that
    /// band-limits the signal BEFORE decimation.
    /// </summary>
    /// <remarks>
    /// ⚠️ This used to be bare linear interpolation, with a comment conceding "for production quality,
    /// consider using a proper sinc resampler". Linear interpolation does not remove content above the
    /// DESTINATION Nyquist, so decimating 48 kHz microphone audio to the 16 kHz Whisper wants folded
    /// everything from 8-24 kHz back down on top of the speech - and sibilants and fricatives live exactly
    /// there. The result is not obviously broken audio; it is audio whose formants have been polluted, so
    /// Whisper returns fluent, confident, completely unrelated text.
    /// <para>
    /// ⚠️ It stayed invisible even though a test DID exercise this method.
    /// <c>AudioPreprocessor_Resample_Frequency</c> called <c>Resample(samples, 44100, 16000)</c> and
    /// asserted the output LENGTH (within one sample) and that values sit in [-1.1, 1.1]. Aliasing
    /// violates neither. Its input was a 440 Hz tone, far below the 8 kHz destination Nyquist, so there
    /// was nothing in it to alias - the test ran the real conversion and COULD NOT FAIL. Test a rate
    /// conversion with content ABOVE the destination Nyquist, or it proves nothing about band-limiting.
    /// </para>
    /// <para>
    /// The kernel cutoff is <c>min(1, dstRate/srcRate)</c> in units of the source Nyquist: 1.0 when
    /// upsampling (pure interpolation), and the decimation ratio when downsampling, which is the
    /// anti-aliasing filter. Weights are normalised per output sample so partial windows at the signal
    /// edges do not produce a gain step.
    /// </para>
    /// </remarks>
    public static float[] Resample(float[] samples, int srcRate, int dstRate)
    {
        if (samples == null || samples.Length == 0) return samples ?? System.Array.Empty<float>();
        if (srcRate <= 0) throw new ArgumentOutOfRangeException(nameof(srcRate));
        if (dstRate <= 0) throw new ArgumentOutOfRangeException(nameof(dstRate));
        if (srcRate == dstRate) return samples;

        double ratio = (double)dstRate / srcRate;
        int outLength = (int)(samples.Length * ratio);
        if (outLength <= 0) return System.Array.Empty<float>();
        var output = new float[outLength];

        // Cutoff relative to the SOURCE Nyquist. Below 1.0 this IS the anti-aliasing low-pass.
        double cutoff = Math.Min(1.0, ratio);
        // Window half-width in source samples. Widening as the cutoff falls keeps the same number of sinc
        // lobes, so the transition band does not degrade when downsampling hard.
        double halfWidth = ResampleLobes / cutoff;

        // Output i sits at source position i * srcRate / dstRate. Reduce the rate pair by its gcd to S/D:
        // the FRACTIONAL part of that position then depends only on (i mod D), so there are just D
        // distinct kernels no matter how long the signal is. 48 kHz -> 16 kHz reduces to 3/1 - ONE phase,
        // the same kernel for every output sample. Computing the window per tap instead costs two trig
        // calls per tap, which for a 30 s clip at ~48 taps per output is tens of millions of them.
        int g = Gcd(srcRate, dstRate);
        int phaseStride = srcRate / g;     // source samples advanced per D outputs
        int phaseCount = dstRate / g;      // distinct fractional offsets

        // Coprime rates make phaseCount as large as dstRate. Past a sane bound the table costs more than
        // it saves, so fall back to evaluating the kernel directly.
        if (phaseCount <= MaxResamplePhases)
        {
            var kernels = new float[phaseCount][];
            var offsets = new int[phaseCount];
            for (int r = 0; r < phaseCount; r++)
            {
                double c = (double)r * phaseStride / phaseCount;
                int f = (int)Math.Ceiling(c - halfWidth);
                int l = (int)Math.Floor(c + halfWidth);
                offsets[r] = f;
                var k = new float[l - f + 1];
                for (int t = 0; t < k.Length; t++)
                {
                    double dt = c - (f + t);
                    k[t] = (float)(Sinc(cutoff * dt) * BlackmanWindow(dt / halfWidth));
                }
                kernels[r] = k;
            }

            for (int i = 0; i < outLength; i++)
            {
                int q = i / phaseCount;
                int r = i - q * phaseCount;
                var k = kernels[r];
                int start = q * phaseStride + offsets[r];

                double acc = 0, norm = 0;
                for (int t = 0; t < k.Length; t++)
                {
                    int j = start + t;
                    if (j < 0 || j >= samples.Length) continue;
                    acc += samples[j] * k[t];
                    norm += k[t];
                }
                output[i] = norm > 1e-9 ? (float)(acc / norm) : 0f;
            }

            return output;
        }

        for (int i = 0; i < outLength; i++)
        {
            double center = i / ratio;
            int first = (int)Math.Ceiling(center - halfWidth);
            int last = (int)Math.Floor(center + halfWidth);
            if (first < 0) first = 0;
            if (last >= samples.Length) last = samples.Length - 1;

            double acc = 0, norm = 0;
            for (int j = first; j <= last; j++)
            {
                double t = center - j;
                double weight = Sinc(cutoff * t) * BlackmanWindow(t / halfWidth);
                acc += samples[j] * weight;
                norm += weight;
            }
            output[i] = norm > 1e-9 ? (float)(acc / norm) : 0f;
        }

        return output;
    }

    /// <summary>Sinc lobes kept either side of an output sample. More lobes = sharper transition band.</summary>
    private const int ResampleLobes = 8;

    /// <summary>Largest kernel table worth precomputing, in distinct phases.</summary>
    private const int MaxResamplePhases = 4096;

    /// <summary>Greatest common divisor, used to reduce a sample-rate pair to its smallest ratio.</summary>
    private static int Gcd(int a, int b)
    {
        while (b != 0) { (a, b) = (b, a % b); }
        return a < 0 ? -a : a;
    }

    /// <summary>Normalised sinc, sin(pi x) / (pi x), with the removable singularity at 0 filled in.</summary>
    private static double Sinc(double x)
    {
        if (Math.Abs(x) < 1e-9) return 1.0;
        double px = Math.PI * x;
        return Math.Sin(px) / px;
    }

    /// <summary>Blackman window over u in [-1, 1]; zero outside, which bounds the kernel.</summary>
    private static double BlackmanWindow(double u)
    {
        if (u <= -1.0 || u >= 1.0) return 0.0;
        return 0.42 + 0.5 * Math.Cos(Math.PI * u) + 0.08 * Math.Cos(2.0 * Math.PI * u);
    }

    /// <summary>
    /// Convert stereo interleaved samples to mono by averaging channels.
    /// </summary>
    public static float[] StereoToMono(float[] stereo)
    {
        var mono = new float[stereo.Length / 2];
        for (int i = 0; i < mono.Length; i++)
        {
            mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) * 0.5f;
        }
        return mono;
    }

    /// <summary>
    /// Convert 16-bit PCM samples to float [-1, 1].
    /// </summary>
    public static float[] PcmInt16ToFloat(short[] pcm)
    {
        var output = new float[pcm.Length];
        for (int i = 0; i < pcm.Length; i++)
        {
            output[i] = pcm[i] / 32768f;
        }
        return output;
    }

    /// <summary>
    /// Convert raw PCM bytes (16-bit little-endian) to float [-1, 1].
    /// </summary>
    public static float[] PcmBytesToFloat(byte[] pcmBytes)
    {
        int sampleCount = pcmBytes.Length / 2;
        var output = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            short sample = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
            output[i] = sample / 32768f;
        }
        return output;
    }

    /// <summary>
    /// Pad or trim audio to a fixed length.
    /// Pads with silence (zeros) if too short, trims if too long.
    /// </summary>
    public static float[] PadOrTrim(float[] samples, int targetLength)
    {
        if (samples.Length == targetLength) return samples;

        var output = new float[targetLength];
        int copyLength = Math.Min(samples.Length, targetLength);
        Array.Copy(samples, output, copyLength);
        return output;
    }

    /// <summary>
    /// Apply a Hann window to a frame of audio samples. Modifies in place.
    /// </summary>
    public static void ApplyHannWindow(float[] frame)
    {
        int n = frame.Length;
        for (int i = 0; i < n; i++)
        {
            float window = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / (n - 1)));
            frame[i] *= window;
        }
    }

    /// <summary>
    /// Generate a Hann window of the specified size.
    /// </summary>
    public static float[] GenerateHannWindow(int size)
    {
        var window = new float[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / (size - 1)));
        }
        return window;
    }

    /// <summary>
    /// Compute the Short-Time Fourier Transform (STFT) magnitude.
    /// Returns a 2D array [numFrames, fftSize/2 + 1] of magnitude values.
    /// </summary>
    /// <param name="samples">Input audio samples</param>
    /// <param name="fftSize">FFT window size (e.g., 400 for Whisper)</param>
    /// <param name="hopSize">Hop between frames (e.g., 160 for Whisper)</param>
    /// <returns>STFT magnitudes [numFrames, fftSize/2 + 1]</returns>
    /// <param name="center">
    /// Frame the signal the way <c>torch.stft(center: true)</c> does - reflect-pad by <c>fftSize/2</c> at both
    /// ends so frame <c>t</c> is CENTRED on sample <c>t * hopSize</c>. Whisper's reference preprocessing
    /// relies on this, and the frame COUNT depends on it: 30s at 16 kHz gives 2998 frames uncentred but the
    /// 3000 the model's input shape demands once centred (3001 frames, last one dropped by the caller).
    /// Left off by default so a plain STFT keeps its existing framing.
    /// </param>
    /// <param name="periodicWindow">
    /// Use a PERIODIC Hann window (torch/librosa default) instead of the symmetric one. Whisper wants
    /// symmetric, so this stays off by default; the vocoder models need periodic for analysis and
    /// synthesis to be inverses of each other.
    /// </param>
    public static float[,] ComputeSTFT(float[] samples, int fftSize, int hopSize, bool center = false, bool periodicWindow = false)
    {
        if (center)
        {
            // Reflect padding, matching numpy/torch 'reflect': mirror WITHOUT repeating the edge sample.
            int pad = fftSize / 2;
            var padded = new float[samples.Length + 2 * pad];
            Array.Copy(samples, 0, padded, pad, samples.Length);
            for (int i = 0; i < pad; i++)
            {
                padded[pad - 1 - i] = samples[Math.Min(i + 1, samples.Length - 1)];
                padded[pad + samples.Length + i] = samples[Math.Max(samples.Length - 2 - i, 0)];
            }
            samples = padded;
        }

        var window = GenerateHannWindow(fftSize, periodicWindow);
        int numFrames = (samples.Length - fftSize) / hopSize + 1;
        int freqBins = fftSize / 2 + 1;
        var stft = new float[numFrames, freqBins];

        var frame = new float[fftSize];
        var real = new float[fftSize];
        var imag = new float[fftSize];
        // Allocated once for the whole STFT: a non-power-of-two length needs an out-of-place pass, and
        // Whisper's 30s window is ~3000 frames.
        var scratchRe = new float[fftSize];
        var scratchIm = new float[fftSize];

        for (int f = 0; f < numFrames; f++)
        {
            int offset = f * hopSize;

            // Extract windowed frame
            for (int i = 0; i < fftSize; i++)
            {
                int idx = offset + i;
                frame[i] = idx < samples.Length ? samples[idx] * window[i] : 0;
            }

            // DFT (real-valued input)
            Array.Copy(frame, real, fftSize);
            Array.Clear(imag, 0, fftSize);
            FFT(real, imag, fftSize, scratchRe, scratchIm);

            // Compute magnitude
            for (int k = 0; k < freqBins; k++)
            {
                stft[f, k] = MathF.Sqrt(real[k] * real[k] + imag[k] * imag[k]);
            }
        }

        return stft;
    }

    /// <summary>
    /// Compute log-mel spectrogram (Whisper-compatible preprocessing).
    /// </summary>
    /// <param name="samples">Audio samples at 16kHz</param>
    /// <param name="nMels">Number of mel bins (80 for Whisper)</param>
    /// <param name="fftSize">FFT size (400 for Whisper = 25ms at 16kHz)</param>
    /// <param name="hopSize">Hop size (160 for Whisper = 10ms at 16kHz)</param>
    /// <returns>Log-mel spectrogram [nMels, numFrames] ready for model input</returns>
    public static float[] ComputeLogMelSpectrogram(float[] samples, int nMels = 80, int fftSize = 400, int hopSize = 160)
    {
        // Pad to 30 seconds for Whisper
        samples = PadOrTrim(samples, WhisperMaxSamples);

        // Whisper frames the signal CENTRED (torch.stft's default) and then discards the final frame, which
        // is what makes 30 seconds come out as exactly 3000 frames - the length the model's input shape is
        // fixed at. Framed uncentred this produced 2998, and the pipeline threw building a [1,80,3000]
        // tensor over 239,840 values.
        var stft = ComputeSTFT(samples, fftSize, hopSize, center: true);
        int numFrames = Math.Max(0, stft.GetLength(0) - 1);
        int freqBins = stft.GetLength(1);

        // Compute power spectrum
        var power = new float[numFrames, freqBins];
        for (int f = 0; f < numFrames; f++)
            for (int k = 0; k < freqBins; k++)
                power[f, k] = stft[f, k] * stft[f, k];

        // Generate mel filterbank
        // Whisper is trained on librosa/slaney filters, not HTK - see GenerateMelFilterbankSlaney.
        var melFilters = GenerateMelFilterbankSlaney(nMels, freqBins, 16000);

        // Apply mel filterbank: [nMels, numFrames]
        var melSpec = new float[nMels * numFrames];
        for (int m = 0; m < nMels; m++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                float sum = 0;
                for (int k = 0; k < freqBins; k++)
                {
                    sum += melFilters[m, k] * power[f, k];
                }
                // Log scale with floor
                melSpec[m * numFrames + f] = MathF.Log10(MathF.Max(sum, 1e-10f));
            }
        }

        // Normalize: scale to [-1, 1] range (Whisper convention)
        float maxVal = float.MinValue;
        for (int i = 0; i < melSpec.Length; i++)
            if (melSpec[i] > maxVal) maxVal = melSpec[i];

        for (int i = 0; i < melSpec.Length; i++)
        {
            melSpec[i] = MathF.Max(melSpec[i], maxVal - 8f); // Dynamic range clipping
            melSpec[i] = (melSpec[i] + 4f) / 4f; // Approximate Whisper normalization
        }

        return melSpec;
    }

    /// <summary>Hz to mel on the SLANEY scale - linear below 1 kHz, logarithmic above.</summary>
    /// <remarks>
    /// This, not <see cref="HzToMel"/>'s HTK formula, is what Whisper is trained against: OpenAI ships a
    /// precomputed <c>mel_filters.npz</c> produced by <c>librosa.filters.mel(...)</c>, whose default is
    /// <c>htk=False</c>. The two scales place the 80 band edges in visibly different places, so an HTK
    /// filterbank hands the encoder a spectrum it has never seen - which does not throw, it just fails to
    /// recognise anything.
    /// </remarks>
    public static float HzToMelSlaney(float hz)
    {
        const float fSp = 200f / 3f;          // 66.67 Hz per mel below the break
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp;   // 15.0
        float logStep = MathF.Log(6.4f) / 27f;
        return hz < minLogHz ? hz / fSp : minLogMel + MathF.Log(hz / minLogHz) / logStep;
    }

    /// <summary>Mel to Hz on the SLANEY scale. Inverse of <see cref="HzToMelSlaney"/>.</summary>
    public static float MelToHzSlaney(float mel)
    {
        const float fSp = 200f / 3f;
        const float minLogHz = 1000f;
        const float minLogMel = minLogHz / fSp;
        float logStep = MathF.Log(6.4f) / 27f;
        return mel < minLogMel ? fSp * mel : minLogHz * MathF.Exp(logStep * (mel - minLogMel));
    }

    /// <summary>
    /// Slaney-scale, area-normalised triangular mel filterbank - the librosa construction Whisper's
    /// shipped filters come from.
    /// </summary>
    /// <remarks>
    /// Two things differ from the simpler <see cref="GenerateMelFilterbank"/> beyond the scale itself, and
    /// both matter: the triangles are evaluated against the EXACT FFT bin frequencies rather than
    /// floor-rounded bin indices (rounding quantises every band edge, and at n_fft=400 the bins are 40 Hz
    /// apart, so the low bands lose most of their shape), and each filter is scaled by
    /// <c>2 / (hz[m+2] - hz[m])</c> so it integrates to a constant regardless of bandwidth. Without that
    /// normalisation the high, wide bands dominate the low, narrow ones.
    /// </remarks>
    public static float[,] GenerateMelFilterbankSlaney(int nMels, int freqBins, int sampleRate)
    {
        // Exact FFT bin centre frequencies: linspace(0, sr/2, freqBins).
        var fftFreqs = new float[freqBins];
        for (int k = 0; k < freqBins; k++) fftFreqs[k] = sampleRate / 2f * k / (freqBins - 1);

        float melMin = HzToMelSlaney(0f);
        float melMax = HzToMelSlaney(sampleRate / 2f);
        var hzPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
            hzPoints[i] = MelToHzSlaney(melMin + (melMax - melMin) * i / (nMels + 1));

        var filters = new float[nMels, freqBins];
        for (int m = 0; m < nMels; m++)
        {
            float left = hzPoints[m], center = hzPoints[m + 1], right = hzPoints[m + 2];
            float leftWidth = center - left, rightWidth = right - center;
            float enorm = 2f / (right - left);      // Slaney normalisation
            for (int k = 0; k < freqBins; k++)
            {
                float f = fftFreqs[k];
                float lower = leftWidth > 0 ? (f - left) / leftWidth : 0f;
                float upper = rightWidth > 0 ? (right - f) / rightWidth : 0f;
                float w = MathF.Min(lower, upper);
                if (w > 0) filters[m, k] = w * enorm;
            }
        }
        return filters;
    }

    /// <summary>
    /// Generate an HTK-scale mel filterbank matrix [nMels, freqBins].
    /// </summary>
    /// <remarks>
    /// Kept for callers that want the HTK scale. Whisper needs <see cref="GenerateMelFilterbankSlaney"/>.
    /// </remarks>
    private static float[,] GenerateMelFilterbank(int nMels, int freqBins, int sampleRate, int fftSize)
    {
        float melMin = HzToMel(0);
        float melMax = HzToMel(sampleRate / 2f);

        // Equally spaced mel points
        var melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
        }

        // Convert back to Hz then to FFT bin indices
        var binIndices = new int[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            float hz = MelToHz(melPoints[i]);
            binIndices[i] = (int)MathF.Floor((fftSize + 1) * hz / sampleRate);
        }

        // Build triangular filters
        var filters = new float[nMels, freqBins];
        for (int m = 0; m < nMels; m++)
        {
            int start = binIndices[m];
            int center = binIndices[m + 1];
            int end = binIndices[m + 2];

            for (int k = start; k < center && k < freqBins; k++)
            {
                if (center > start)
                    filters[m, k] = (float)(k - start) / (center - start);
            }
            for (int k = center; k < end && k < freqBins; k++)
            {
                if (end > center)
                    filters[m, k] = (float)(end - k) / (end - center);
            }
        }

        return filters;
    }

    /// <summary>
    /// Convert frequency in Hz to mel scale.
    /// </summary>
    public static float HzToMel(float hz) => 2595f * MathF.Log10(1f + hz / 700f);

    /// <summary>
    /// Convert mel scale to frequency in Hz.
    /// </summary>
    public static float MelToHz(float mel) => 700f * (MathF.Pow(10f, mel / 2595f) - 1f);

    /// <summary>
    /// In-place FFT for ANY transform length. Input arrays are modified.
    /// </summary>
    /// <remarks>
    /// A radix-2 FFT alone is not enough here: Whisper's <c>n_fft</c> is <b>400</b>, which is not a power of
    /// two, and the radix-2 butterflies then run off the end of the array (with n=400 the last stage
    /// addresses index 511 of a 400-element array, and the bit-reversal only permutes 0..255 because
    /// <c>(int)Log2(400)</c> truncates to 8). That made <see cref="ComputeLogMelSpectrogram"/> throw
    /// <see cref="IndexOutOfRangeException"/> on its OWN default parameters, so the entire CPU Whisper
    /// preprocessing path was unusable.
    /// <para>
    /// Power-of-two lengths keep the original iterative radix-2 path. Everything else is decimated in time
    /// while the length is even and finished with a direct DFT once it turns odd - 400 = 2^4 x 25, so it
    /// costs four splits and a 25-point DFT rather than a 400-point one. This is exact, not an
    /// approximation: zero-padding a 400-point frame to 512 would change the bin spacing and hand the mel
    /// filterbank a different spectrum than Whisper was trained on.
    /// </para>
    /// </remarks>
    /// <param name="scratchRe">Caller-owned scratch of at least <paramref name="n"/> floats. Supplied by the
    /// caller because the STFT calls this once per frame - thousands of times - and allocating inside would
    /// churn the heap for nothing. Ignored on the power-of-two path, which is in-place.</param>
    /// <param name="scratchIm">Second scratch buffer, same size.</param>
    private static void FFT(float[] real, float[] imag, int n, float[]? scratchRe = null, float[]? scratchIm = null)
    {
        if ((n & (n - 1)) != 0)
        {
            var outRe = scratchRe is { } sr && sr.Length >= n ? sr : new float[n];
            var outIm = scratchIm is { } si && si.Length >= n ? si : new float[n];
            FFTAny(real, imag, 0, 1, outRe, outIm, 0, n);
            Array.Copy(outRe, real, n);
            Array.Copy(outIm, imag, n);
            return;
        }

        // Bit-reversal permutation
        int bits = (int)MathF.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = ReverseBits(i, bits);
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        // Butterfly stages
        for (int size = 2; size <= n; size *= 2)
        {
            int halfSize = size / 2;
            float angle = -2f * MathF.PI / size;

            for (int i = 0; i < n; i += size)
            {
                for (int j = 0; j < halfSize; j++)
                {
                    float cos = MathF.Cos(angle * j);
                    float sin = MathF.Sin(angle * j);

                    int even = i + j;
                    int odd = i + j + halfSize;

                    float tr = real[odd] * cos - imag[odd] * sin;
                    float ti = real[odd] * sin + imag[odd] * cos;

                    real[odd] = real[even] - tr;
                    imag[odd] = imag[even] - ti;
                    real[even] += tr;
                    imag[even] += ti;
                }
            }
        }
    }

    /// <summary>
    /// Out-of-place Cooley-Tukey for an arbitrary length: split while even, direct DFT once odd.
    /// Reads the input with a stride so the even/odd interleave costs no copying.
    /// </summary>
    private static void FFTAny(float[] inRe, float[] inIm, int inOff, int stride,
                               float[] outRe, float[] outIm, int outOff, int n)
    {
        if (n == 1)
        {
            outRe[outOff] = inRe[inOff];
            outIm[outOff] = inIm[inOff];
            return;
        }
        if ((n & 1) != 0) { DFT(inRe, inIm, inOff, stride, outRe, outIm, outOff, n); return; }

        int half = n / 2;
        // Even-indexed samples land in the first half of the output, odd-indexed in the second.
        FFTAny(inRe, inIm, inOff, stride * 2, outRe, outIm, outOff, half);
        FFTAny(inRe, inIm, inOff + stride, stride * 2, outRe, outIm, outOff + half, half);

        for (int k = 0; k < half; k++)
        {
            float angle = -2f * MathF.PI * k / n;
            float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
            float er = outRe[outOff + k], ei = outIm[outOff + k];
            float or_ = outRe[outOff + half + k], oi = outIm[outOff + half + k];
            float tr = or_ * cos - oi * sin;
            float ti = or_ * sin + oi * cos;
            outRe[outOff + k] = er + tr;
            outIm[outOff + k] = ei + ti;
            outRe[outOff + half + k] = er - tr;
            outIm[outOff + half + k] = ei - ti;
        }
    }

    /// <summary>Direct O(n^2) DFT - the base case for an odd length (25 for Whisper's 400).</summary>
    private static void DFT(float[] inRe, float[] inIm, int inOff, int stride,
                            float[] outRe, float[] outIm, int outOff, int n)
    {
        for (int k = 0; k < n; k++)
        {
            float sumRe = 0f, sumIm = 0f;
            for (int t = 0; t < n; t++)
            {
                float angle = -2f * MathF.PI * k * t / n;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                float xr = inRe[inOff + t * stride], xi = inIm[inOff + t * stride];
                sumRe += xr * cos - xi * sin;
                sumIm += xr * sin + xi * cos;
            }
            outRe[outOff + k] = sumRe;
            outIm[outOff + k] = sumIm;
        }
    }

    private static int ReverseBits(int val, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (val & 1);
            val >>= 1;
        }
        return result;
    }
}
