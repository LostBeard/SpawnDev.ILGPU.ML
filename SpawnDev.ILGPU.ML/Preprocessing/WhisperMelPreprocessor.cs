using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// GPU Whisper log-mel: PCM at 16 kHz → <c>[80 × 3000]</c> encoder input, oracle-matched to
/// <see cref="AudioPreprocessor.ComputeLogMelSpectrogram"/>.
/// </summary>
/// <remarks>
/// ⚠️ WHY THIS EXISTS. The browser Whisper path already lands PCM as a JS <c>Float32Array</c>, but
/// <c>SpeechRecognitionPipeline</c> still <c>ToArray()</c>s because mel was CPU-only. That host crossing
/// exists solely for the STFT. This class keeps the waveform on-device through pad → centred STFT →
/// power → Slaney mel → log → Whisper normalize.
/// <para>
/// STFT uses a <b>direct real DFT</b> per (frame, bin) — N=400 is not a power of two, so a radix-2 FFT
/// would change bin spacing. Correctness first; Bluestein/sparse mel can land after the oracle is green.
/// Scratch buffers are instance-owned: WebGPU/Wasm forbid disposing buffers that pending dispatches still
/// reference.
/// </para>
/// </remarks>
public sealed class WhisperMelPreprocessor : IDisposable
{
    /// <summary>Whisper STFT window (25 ms at 16 kHz).</summary>
    public const int FftSize = 400;

    /// <summary>Whisper hop (10 ms at 16 kHz).</summary>
    public const int HopSize = 160;

    /// <summary>Onesided bins: <c>FftSize/2 + 1</c>.</summary>
    public const int FreqBins = FftSize / 2 + 1;

    /// <summary>Encoder time axis after dropping the final centred frame.</summary>
    public const int NumFrames = 3000;

    /// <summary>Mel filter count (Whisper).</summary>
    public const int NMels = AudioPreprocessor.WhisperMelBins;

    /// <summary>Flat mel length: <c>NMels * NumFrames</c>.</summary>
    public const int MelLength = NMels * NumFrames;

    private readonly Accelerator _accelerator;
    private readonly AudioKernels _audio;
    private readonly ReductionKernels _reduce;

    private MemoryBuffer1D<float, Stride1D.Dense>? _pcmPad;
    private MemoryBuffer1D<float, Stride1D.Dense>? _power;
    private MemoryBuffer1D<float, Stride1D.Dense>? _filters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _maxBuf;

    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _padOrTrim;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _stftPower;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _normalizeFromMax;

    private bool _disposed;

    public WhisperMelPreprocessor(Accelerator accelerator)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _audio = new AudioKernels(accelerator);
        _reduce = new ReductionKernels(accelerator);
    }

    /// <summary>
    /// PCM at 16 kHz (any length ≤ <see cref="AudioPreprocessor.WhisperMaxSamples"/>, or longer — trimmed)
    /// → log-mel into <paramref name="melOut"/> (<see cref="MelLength"/> floats, layout <c>[nMels, frames]</c>).
    /// </summary>
    public void ComputeLogMelSpectrogram(
        ArrayView1D<float, Stride1D.Dense> pcm,
        int pcmLength,
        ArrayView1D<float, Stride1D.Dense> melOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (pcmLength < 0) throw new ArgumentOutOfRangeException(nameof(pcmLength));
        if (pcm.Length < pcmLength)
            throw new ArgumentException($"pcm view length {pcm.Length} < pcmLength {pcmLength}", nameof(pcm));
        if (melOut.Length < MelLength)
            throw new ArgumentException($"melOut needs ≥ {MelLength} floats, got {melOut.Length}", nameof(melOut));

        EnsureLoaded();

        int target = AudioPreprocessor.WhisperMaxSamples;
        var pcmPad = _pcmPad!.View;
        int copyLen = pcmLength < target ? pcmLength : target;
        _padOrTrim!((Index1D)target, pcm, pcmPad, copyLen, target);

        var power = _power!.View;
        _stftPower!((Index1D)(NumFrames * FreqBins), pcmPad, power, _stftParams!.View);

        _audio.ApplyMelFilterbank(power, _filters!.View, melOut.SubView(0, MelLength),
            NumFrames, FreqBins, NMels);

        _audio.LogScale(melOut.SubView(0, MelLength), MelLength);

        var maxBuf = _maxBuf!.View;
        _reduce.ReduceMax(melOut.SubView(0, MelLength), maxBuf, 1, MelLength, 1);

        _normalizeFromMax!((Index1D)MelLength, melOut.SubView(0, MelLength), maxBuf);
    }

    /// <summary>
    /// Debug/oracle: pad PCM and write centred STFT power <c>[NumFrames × FreqBins]</c> into
    /// <paramref name="powerOut"/>. Same first stages as <see cref="ComputeLogMelSpectrogram"/>.
    /// </summary>
    public void ComputeStftPower(
        ArrayView1D<float, Stride1D.Dense> pcm,
        int pcmLength,
        ArrayView1D<float, Stride1D.Dense> powerOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (powerOut.Length < NumFrames * FreqBins)
            throw new ArgumentException($"powerOut needs ≥ {NumFrames * FreqBins}", nameof(powerOut));
        EnsureLoaded();
        int target = AudioPreprocessor.WhisperMaxSamples;
        int copyLen = pcmLength < target ? pcmLength : target;
        _padOrTrim!((Index1D)target, pcm, _pcmPad!.View, copyLen, target);
        _stftPower!((Index1D)(NumFrames * FreqBins), _pcmPad.View, powerOut.SubView(0, NumFrames * FreqBins),
            _stftParams!.View);
    }

    public MemoryBuffer1D<float, Stride1D.Dense> ComputeLogMelSpectrogram(float[] pcm)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pcm);
        EnsureLoaded();
        using var pcmBuf = _accelerator.Allocate1D(pcm.Length == 0 ? new float[1] : pcm);
        var mel = _accelerator.Allocate1D<float>(MelLength);
        try
        {
            ComputeLogMelSpectrogram(pcmBuf.View, pcm.Length, mel.View);
            return mel;
        }
        catch
        {
            mel.Dispose();
            throw;
        }
    }

    private MemoryBuffer1D<float, Stride1D.Dense>? _stftParams;

    private void EnsureLoaded()
    {
        if (_pcmPad != null) return;

        _pcmPad = _accelerator.Allocate1D<float>(AudioPreprocessor.WhisperMaxSamples);
        _power = _accelerator.Allocate1D<float>(NumFrames * FreqBins);
        _maxBuf = _accelerator.Allocate1D<float>(1);

        // Slaney filters once per accelerator lifetime — same coefficients as the CPU oracle.
        var filters2d = AudioPreprocessor.GenerateMelFilterbankSlaney(NMels, FreqBins, AudioPreprocessor.WhisperSampleRate);
        var flat = new float[NMels * FreqBins];
        for (int m = 0; m < NMels; m++)
            for (int k = 0; k < FreqBins; k++)
                flat[m * FreqBins + k] = filters2d[m, k];
        _filters = _accelerator.Allocate1D(flat);

        // params: [pcmLength, fftSize, hop, freqBins, numFrames, pad]
        _stftParams = _accelerator.Allocate1D(new float[]
        {
            AudioPreprocessor.WhisperMaxSamples,
            FftSize,
            HopSize,
            FreqBins,
            NumFrames,
            FftSize / 2
        });

        _padOrTrim = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PadOrTrimImpl);

        _stftPower = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(StftPowerCenteredImpl);

        _normalizeFromMax = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(NormalizeWhisperFromMaxImpl);
    }

    /// <summary>Zero-pad or truncate into a fixed-length buffer. One thread per output sample.</summary>
    private static void PadOrTrimImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int srcLength,
        int dstLength)
    {
        if (idx >= dstLength) return;
        dst[idx] = idx < srcLength ? src[idx] : 0f;
    }

    /// <summary>
    /// Centred real STFT power: frame <c>f</c> is centred on sample <c>f * hop</c> via reflect pad
    /// (torch.stft center=true / numpy reflect). Drops the final frame so 30 s → exactly 3000 frames.
    /// One thread per (frame, bin); writes <c>re² + im²</c> (skips the magnitude sqrt the CPU takes
    /// before squaring — algebraically identical).
    /// </summary>
    private static void StftPowerCenteredImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> pcm,
        ArrayView1D<float, Stride1D.Dense> power,
        ArrayView1D<float, Stride1D.Dense> p)
    {
        int pcmLength = (int)p[0];
        int fftSize = (int)p[1];
        int hop = (int)p[2];
        int freqBins = (int)p[3];
        int numFrames = (int)p[4];
        int pad = (int)p[5];

        int f = idx / freqBins;
        int k = idx % freqBins;
        if (f >= numFrames) return;

        // Symmetric Hann, matching AudioPreprocessor.GenerateHannWindow(periodic: false).
        float sumRe = 0f, sumIm = 0f;
        float denom = fftSize - 1;
        for (int n = 0; n < fftSize; n++)
        {
            int g = f * hop + n; // index in reflect-padded space
            float x = ReflectSample(pcm, pcmLength, pad, g);
            float w = 0.5f * (1f - MathF.Cos(2f * MathF.PI * n / denom));
            x *= w;

            // Same angle form as ElementWiseKernels.STFTImpl — do not reduce mod L (Nyquist phase residue).
            float angle = -2f * MathF.PI * k * n / fftSize;
            sumRe += x * MathF.Cos(angle);
            sumIm += x * MathF.Sin(angle);
        }

        power[idx] = sumRe * sumRe + sumIm * sumIm;
    }

    /// <summary>
    /// Sample at reflect-padded index <paramref name="g"/>, matching numpy/torch <c>mode='reflect'</c>
    /// (mirror without repeating the edge sample) used by <see cref="AudioPreprocessor.ComputeSTFT"/>.
    /// </summary>
    private static float ReflectSample(
        ArrayView1D<float, Stride1D.Dense> samples,
        int length,
        int pad,
        int g)
    {
        if (g >= pad && g < pad + length)
            return samples[g - pad];

        if (g < pad)
        {
            // padded[pad - 1 - i] = samples[Min(i + 1, length - 1)], i = pad - 1 - g
            int i = pad - 1 - g;
            int src = i + 1;
            if (src >= length) src = length - 1;
            return samples[src];
        }

        // g >= pad + length: padded[pad + length + i] = samples[Max(length - 2 - i, 0)]
        int j = g - (pad + length);
        int srcR = length - 2 - j;
        if (srcR < 0) srcR = 0;
        return samples[srcR];
    }

    /// <summary>Whisper normalize using a 1-element device max (no host round-trip).</summary>
    private static void NormalizeWhisperFromMaxImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> maxBuf)
    {
        float maxValue = maxBuf[0];
        float threshold = maxValue - 8f;
        float val = data[idx];
        if (val < threshold) val = threshold;
        data[idx] = (val + 4f) / 4f;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _pcmPad?.Dispose();
        _power?.Dispose();
        _filters?.Dispose();
        _maxBuf?.Dispose();
        _stftParams?.Dispose();
        _pcmPad = null;
        _power = null;
        _filters = null;
        _maxBuf = null;
        _stftParams = null;
        // Accelerator is application-owned — never dispose it here.
    }
}
