namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// CPU-side audio preprocessing for ML model input.
/// Handles waveform manipulation, resampling, and spectrogram generation.
/// Designed for models like Whisper, wav2vec2, and audio classifiers.
/// </summary>
public static class AudioPreprocessor
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
    /// Resample audio from one sample rate to another using linear interpolation.
    /// For production quality, consider using a proper sinc resampler.
    /// </summary>
    public static float[] Resample(float[] samples, int srcRate, int dstRate)
    {
        if (srcRate == dstRate) return samples;

        double ratio = (double)dstRate / srcRate;
        int outLength = (int)(samples.Length * ratio);
        var output = new float[outLength];

        for (int i = 0; i < outLength; i++)
        {
            double srcPos = i / ratio;
            int idx = (int)srcPos;
            double frac = srcPos - idx;

            if (idx + 1 < samples.Length)
                output[i] = (float)(samples[idx] * (1 - frac) + samples[idx + 1] * frac);
            else if (idx < samples.Length)
                output[i] = samples[idx];
        }

        return output;
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
    public static float[,] ComputeSTFT(float[] samples, int fftSize, int hopSize)
    {
        var window = GenerateHannWindow(fftSize);
        int numFrames = (samples.Length - fftSize) / hopSize + 1;
        int freqBins = fftSize / 2 + 1;
        var stft = new float[numFrames, freqBins];

        var frame = new float[fftSize];
        var real = new float[fftSize];
        var imag = new float[fftSize];

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
            FFT(real, imag, fftSize);

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

        // Compute STFT
        var stft = ComputeSTFT(samples, fftSize, hopSize);
        int numFrames = stft.GetLength(0);
        int freqBins = stft.GetLength(1);

        // Compute power spectrum
        var power = new float[numFrames, freqBins];
        for (int f = 0; f < numFrames; f++)
            for (int k = 0; k < freqBins; k++)
                power[f, k] = stft[f, k] * stft[f, k];

        // Generate mel filterbank
        var melFilters = GenerateMelFilterbank(nMels, freqBins, 16000, fftSize);

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

    /// <summary>
    /// Generate a mel-scale filterbank matrix [nMels, freqBins].
    /// </summary>
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
    /// In-place Cooley-Tukey radix-2 FFT. Input arrays are modified.
    /// </summary>
    private static void FFT(float[] real, float[] imag, int n)
    {
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
