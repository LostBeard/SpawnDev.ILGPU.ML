using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// GPU Whisper log-mel: PCM at 16 kHz → <c>[80 × 3000]</c> encoder input (30 s window, the default), oracle-matched to
/// <see cref="AudioPreprocessor.ComputeLogMelSpectrogram"/>. A shorter window (Smart Turn: 8 s → <c>[80 × 800]</c>)
/// and the two things that model's feature extractor adds are options: keep the END of the audio (zero-pad the start)
/// and normalise the waveform to zero mean / unit variance before the STFT.
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

    /// <summary>Encoder time axis of the default 30 s window, after dropping the final centred frame.</summary>
    public const int NumFrames = 3000;

    /// <summary>Mel filter count (Whisper).</summary>
    public const int NMels = AudioPreprocessor.WhisperMelBins;

    /// <summary>Flat mel length of the default 30 s window: <c>NMels * NumFrames</c>.</summary>
    public const int MelLength = NMels * NumFrames;

    /// <summary>Window length in samples (480000 for Whisper's 30 s).</summary>
    public int WindowSamples { get; }

    /// <summary>Time axis of this instance: <c>WindowSamples / HopSize</c> (3000 for 30 s, 800 for 8 s).</summary>
    public int Frames { get; }

    /// <summary>Flat mel length of this instance: <c>NMels * Frames</c>.</summary>
    public int MelElements => NMels * Frames;

    private readonly bool _alignToEnd;
    private readonly bool _normalizeWaveform;
    private const int MomentPartials = 256;
    private MemoryBuffer1D<float, Stride1D.Dense>? _moments;
    private MemoryBuffer1D<float, Stride1D.Dense>? _stats;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _partialSum;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>? _partialSqDev;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _finishStats;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>? _applyNormalize;
    private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>? _padOrTrimEnd;

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

    /// <param name="accelerator">The application's accelerator (never disposed here).</param>
    /// <param name="windowSamples">Samples in the window; a multiple of <see cref="HopSize"/>. Default 480000 (Whisper, 30 s).</param>
    /// <param name="alignToEnd">Audio longer than the window keeps its LAST <paramref name="windowSamples"/>; shorter audio is zero-padded at the START
    /// (Smart Turn's <c>truncate_audio_to_last_n_seconds</c>). Off: keep the start, pad the end (Whisper).</param>
    /// <param name="normalizeWaveform">Zero mean / unit variance over the whole padded window before the STFT
    /// (<c>(x - mean) / sqrt(var + 1e-7)</c>, HuggingFace <c>do_normalize=True</c>). Off for Whisper.</param>
    public WhisperMelPreprocessor(Accelerator accelerator, int windowSamples = AudioPreprocessor.WhisperMaxSamples,
        bool alignToEnd = false, bool normalizeWaveform = false)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        if (windowSamples <= 0 || windowSamples % HopSize != 0)
            throw new ArgumentOutOfRangeException(nameof(windowSamples), windowSamples, $"must be a positive multiple of {HopSize}");
        WindowSamples = windowSamples;
        Frames = windowSamples / HopSize;
        _alignToEnd = alignToEnd;
        _normalizeWaveform = normalizeWaveform;
        _audio = new AudioKernels(accelerator);
        _reduce = new ReductionKernels(accelerator);
    }

    /// <summary>
    /// PCM at 16 kHz (any length; longer is trimmed to the window)
    /// → log-mel into <paramref name="melOut"/> (<see cref="MelElements"/> floats, layout <c>[nMels, frames]</c>).
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
        int melLength = MelElements;
        if (melOut.Length < melLength)
            throw new ArgumentException($"melOut needs ≥ {melLength} floats, got {melOut.Length}", nameof(melOut));

        EnsureLoaded();

        var pcmPad = PadWindow(pcm, pcmLength);

        var power = _power!.View;
        _stftPower!((Index1D)(Frames * FreqBins), pcmPad, power, _stftParams!.View);

        _audio.ApplyMelFilterbank(power, _filters!.View, melOut.SubView(0, melLength),
            Frames, FreqBins, NMels);

        _audio.LogScale(melOut.SubView(0, melLength), melLength);

        var maxBuf = _maxBuf!.View;
        _reduce.ReduceMax(melOut.SubView(0, melLength), maxBuf, 1, melLength, 1);

        _normalizeFromMax!((Index1D)melLength, melOut.SubView(0, melLength), maxBuf);
    }

    /// <summary>Pad/trim into the window buffer (start- or end-aligned) and, if asked, normalise the waveform.</summary>
    private ArrayView1D<float, Stride1D.Dense> PadWindow(ArrayView1D<float, Stride1D.Dense> pcm, int pcmLength)
    {
        int target = WindowSamples;
        var pcmPad = _pcmPad!.View;
        if (_alignToEnd) _padOrTrimEnd!((Index1D)target, pcm, pcmPad, pcmLength, target);
        else _padOrTrim!((Index1D)target, pcm, pcmPad, pcmLength < target ? pcmLength : target, target);

        if (_normalizeWaveform)
        {
            var moments = _moments!.View;
            var stats = _stats!.View;
            _partialSum!((Index1D)MomentPartials, pcmPad, moments, target);
            _finishStats!((Index1D)1, moments, stats, target, 0);
            _partialSqDev!((Index1D)MomentPartials, pcmPad, stats, moments, target);
            _finishStats!((Index1D)1, moments, stats, target, 1);
            _applyNormalize!((Index1D)target, pcmPad, stats);
        }
        return pcmPad;
    }

    /// <summary>
    /// Debug/oracle: pad PCM and write centred STFT power <c>[Frames × FreqBins]</c> into
    /// <paramref name="powerOut"/>. Same first stages as <see cref="ComputeLogMelSpectrogram"/>.
    /// </summary>
    public void ComputeStftPower(
        ArrayView1D<float, Stride1D.Dense> pcm,
        int pcmLength,
        ArrayView1D<float, Stride1D.Dense> powerOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (powerOut.Length < Frames * FreqBins)
            throw new ArgumentException($"powerOut needs ≥ {Frames * FreqBins}", nameof(powerOut));
        EnsureLoaded();
        var pcmPad = PadWindow(pcm, pcmLength);
        _stftPower!((Index1D)(Frames * FreqBins), pcmPad, powerOut.SubView(0, Frames * FreqBins),
            _stftParams!.View);
    }

    public MemoryBuffer1D<float, Stride1D.Dense> ComputeLogMelSpectrogram(float[] pcm)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(pcm);
        EnsureLoaded();
        using var pcmBuf = _accelerator.Allocate1D(pcm.Length == 0 ? new float[1] : pcm);
        var mel = _accelerator.Allocate1D<float>(MelElements);
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

        _pcmPad = _accelerator.Allocate1D<float>(WindowSamples);
        _power = _accelerator.Allocate1D<float>(Frames * FreqBins);
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
            WindowSamples,
            FftSize,
            HopSize,
            FreqBins,
            Frames,
            FftSize / 2
        });

        _padOrTrim = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PadOrTrimImpl);

        _stftPower = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(StftPowerCenteredImpl);

        _normalizeFromMax = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(NormalizeWhisperFromMaxImpl);

        if (_alignToEnd)
            _padOrTrimEnd = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(PadOrTrimEndImpl);

        if (_normalizeWaveform)
        {
            _moments = _accelerator.Allocate1D<float>(MomentPartials);
            _stats = _accelerator.Allocate1D<float>(2); // [mean, 1 / sqrt(var + 1e-7)]
            _partialSum = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(PartialSumImpl);
            _partialSqDev = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(PartialSqDevImpl);
            _finishStats = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(FinishStatsImpl);
            _applyNormalize = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ApplyNormalizeImpl);
        }
    }

    /// <summary>
    /// End-aligned window: the LAST <paramref name="dstLength"/> samples of the audio, zero-padded at the START when the audio
    /// is shorter. One thread per output sample.
    /// </summary>
    private static void PadOrTrimEndImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> src,
        ArrayView1D<float, Stride1D.Dense> dst,
        int srcLength,
        int dstLength)
    {
        if (idx >= dstLength) return;
        int s = srcLength - dstLength + idx;
        dst[idx] = s >= 0 ? src[s] : 0f;
    }

    /// <summary>Partial sums over contiguous chunks of the window: one thread per chunk.</summary>
    private static void PartialSumImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> partials,
        int n)
    {
        int chunk = (n + MomentPartials - 1) / MomentPartials;
        int start = idx * chunk;
        int end = start + chunk;
        if (end > n) end = n;
        float sum = 0f;
        for (int i = start; i < end; i++) sum += data[i];
        partials[idx] = sum;
    }

    /// <summary>Partial sums of squared deviation from the mean in <c>stats[0]</c> (two-pass: no catastrophic cancellation).</summary>
    private static void PartialSqDevImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> stats,
        ArrayView1D<float, Stride1D.Dense> partials,
        int n)
    {
        float mean = stats[0];
        int chunk = (n + MomentPartials - 1) / MomentPartials;
        int start = idx * chunk;
        int end = start + chunk;
        if (end > n) end = n;
        float sum = 0f;
        for (int i = start; i < end; i++)
        {
            float d = data[i] - mean;
            sum += d * d;
        }
        partials[idx] = sum;
    }

    /// <summary>One thread. pass 0: <c>stats[0] = mean</c>; pass 1: <c>stats[1] = 1 / sqrt(variance + 1e-7)</c>.</summary>
    private static void FinishStatsImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> partials,
        ArrayView1D<float, Stride1D.Dense> stats,
        int n,
        int pass)
    {
        float total = 0f;
        for (int i = 0; i < MomentPartials; i++) total += partials[i];
        if (pass == 0) stats[0] = total / n;
        else stats[1] = 1f / MathF.Sqrt(total / n + 1e-7f);
    }

    private static void ApplyNormalizeImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> stats)
    {
        data[idx] = (data[idx] - stats[0]) * stats[1];
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

        // PERIODIC Hann (denominator N), matching AudioPreprocessor.GenerateHannWindow(periodic: true): torch.hann_window and the
        // HuggingFace Whisper feature extractor. (Symmetric, denominator N-1, was used before 2026-10-14 and was off by up to 4e-2.)
        float sumRe = 0f, sumIm = 0f;
        float denom = fftSize;
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
        _moments?.Dispose();
        _stats?.Dispose();
        _pcmPad = null;
        _power = null;
        _filters = null;
        _maxBuf = null;
        _stftParams = null;
        _moments = null;
        _stats = null;
        // Accelerator is application-owned — never dispose it here.
    }
}
