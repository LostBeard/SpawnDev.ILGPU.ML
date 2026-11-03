using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

using FV = ArrayView1D<float, Stride1D.Dense>;

/// <summary>
/// CLAP's audio features on the device (transformers' <c>ClapFeatureExtractor</c>, <c>truncation="rand_trunc"</c>, <c>padding="repeatpad"</c>, the
/// laion/clap-htsat-unfused preprocessor config): 48 kHz mono to a log-mel spectrogram <c>[1001, 64]</c> (time-major, the model's
/// <c>input_features[0, 0]</c>). Verified against transformers' source (2026-11-03): the clip is repeated a whole number of times and zero-padded to
/// 10 s (480,000 samples); STFT with a periodic Hann window of 1,024, hop 480, centred with reflect padding (1,001 frames); power; Slaney mel filters
/// (50 Hz .. 14 kHz, Slaney norm); 10 log10(max(e, 1e-10)). One deviation: a clip LONGER than 10 s is cut to its centre 10 s, where transformers takes a
/// random window (rand_trunc) - deterministic on purpose.
/// </summary>
public sealed class ClapFeatureExtractor : IDisposable
{
    public const int SampleRate = 48000, MaxSamples = 480000, FftSize = 1024, Hop = 480, Mels = 64, Frames = MaxSamples / Hop + 1;
    private const int Pad = FftSize / 2, Ring = 1 << 19;              // 524,288 >= 480,000 + 1,024
    private readonly Accelerator _acc;
    private readonly SpectralFrontend _frontend;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _padded, _power;
    private readonly Action<Index1D, FV, FV, int, int> _pad;

    public ClapFeatureExtractor(Accelerator accelerator)
    {
        _acc = accelerator;
        _frontend = new SpectralFrontend(accelerator, FftSize, 1, SampleRate, SpectralWindow.Hann);
        _frontend.SetBandFilters(MelFilters.Build(_frontend.Bins, Mels, 50, 14000, SampleRate, slaneyNorm: true, MelScale.Slaney), Mels);
        _padded = accelerator.Allocate1D<float>(Ring);
        _power = accelerator.Allocate1D<float>((long)Frames * _frontend.Bins);
        _pad = accelerator.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int>(PadImpl);
    }

    /// <summary>Features of <paramref name="length"/> samples of 48 kHz mono <paramref name="pcm"/> into <paramref name="features"/> (<c>Frames x Mels</c>, time-major).</summary>
    public void Extract(FV pcm, int length, FV features)
    {
        if (length < 1 || pcm.Length < length) throw new ArgumentException("pcm needs length samples", nameof(pcm));
        if (features.Length < (long)Frames * Mels) throw new ArgumentException($"features needs {Frames * Mels} floats", nameof(features));
        int start = length > MaxSamples ? (length - MaxSamples) / 2 : 0;
        _pad(MaxSamples + 2 * Pad, pcm, _padded.View, length, start);
        // frame f covers padded samples [480 f, 480 f + 1024): its END is 1024 + 480 f, the last one 481,024 = writePos
        _frontend.Power(_padded.View, Ring, MaxSamples + 2 * Pad, FftSize, Hop, Frames, _power.View);
        _frontend.PoolBands(_power.View, Frames, features.SubView(0, (long)Frames * Mels), logFloor: 1e-10f);
    }

    // padded[j] = y[reflect(j - 512)], y the 10 s signal: the clip repeated floor(480000 / L) times then zeros (repeatpad), or a window of a long clip;
    // reflect is numpy's mode='reflect' (no edge repeat)
    private static void PadImpl(Index1D idx, FV pcm, FV padded, int length, int start)
    {
        int j = idx, i = j - Pad;
        if (i < 0) i = -i;
        else if (i >= MaxSamples) i = 2 * MaxSamples - 2 - i;
        float v;
        if (length >= MaxSamples) v = pcm[start + i];
        else
        {
            int reps = MaxSamples / length;
            v = i < reps * length ? pcm[i % length] : 0f;
        }
        padded[j] = v;
    }

    public void Dispose() { _frontend.Dispose(); _padded.Dispose(); _power.Dispose(); }
}
