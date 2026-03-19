using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Kernels;

/// <summary>
/// GPU kernels for audio preprocessing.
/// Replaces CPU-side AudioPreprocessor with GPU-native execution.
/// FFT stays CPU (complex recursive algorithm), but windowing,
/// mel filterbank application, and log scaling run on GPU.
/// </summary>
public class AudioKernels
{
    private readonly Accelerator _accelerator;

    public AudioKernels(Accelerator accelerator) => _accelerator = accelerator;

    // ──────────────────────────────────────────────
    //  Hann window application (in-place)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Apply Hann window to audio frames on GPU.
    /// One thread per sample. Modifies data in place.
    /// </summary>
    public void ApplyHannWindow(
        ArrayView1D<float, Stride1D.Dense> data,
        int frameSize)
    {
        int count = (int)data.Length;
        float piTerm = 2f * MathF.PI / (frameSize - 1);

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                int posInFrame = idx % frameSize;
                float window = 0.5f * (1f - MathF.Cos(piTerm * posInFrame));
                d[idx] *= window;
            });

        kernel((Index1D)count, data);
    }

    // ──────────────────────────────────────────────
    //  Mel filterbank application
    // ──────────────────────────────────────────────

    /// <summary>
    /// Apply mel filterbank matrix to power spectrum on GPU.
    /// Input: power spectrum [numFrames, freqBins]
    /// Filters: mel filterbank [nMels, freqBins]
    /// Output: mel spectrogram [nMels, numFrames]
    /// This is essentially a matrix multiply: output = filters @ power^T
    /// </summary>
    public void ApplyMelFilterbank(
        ArrayView1D<float, Stride1D.Dense> power,
        ArrayView1D<float, Stride1D.Dense> filters,
        ArrayView1D<float, Stride1D.Dense> output,
        int numFrames, int freqBins, int nMels)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> pw, ArrayView1D<float, Stride1D.Dense> flt, ArrayView1D<float, Stride1D.Dense> outp) =>
            {
                int m = idx / numFrames; // mel bin
                int f = idx % numFrames; // frame

                float sum = 0f;
                for (int k = 0; k < freqBins; k++)
                {
                    sum += flt[m * freqBins + k] * pw[f * freqBins + k];
                }

                outp[idx] = sum;
            });

        kernel((Index1D)(nMels * numFrames), power, filters, output);
    }

    // ──────────────────────────────────────────────
    //  Log scale (for log-mel spectrogram)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Apply log scaling to mel spectrogram on GPU.
    /// log10(max(value, floor)) with dynamic range clipping.
    /// </summary>
    public void LogScale(
        ArrayView1D<float, Stride1D.Dense> data,
        int count,
        float floor = 1e-10f)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                float val = d[idx];
                val = val > floor ? val : floor;
                d[idx] = MathF.Log10(val);
            });

        kernel((Index1D)count, data);
    }

    // ──────────────────────────────────────────────
    //  Normalize mel spectrogram (Whisper convention)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Normalize log-mel spectrogram to [-1, 1] range (Whisper convention).
    /// Requires a two-pass approach: first find max, then normalize.
    /// Pass 1: Use ReductionKernels.ReduceMax to find max value.
    /// Pass 2: This kernel applies (value - (max - 8)) / 4, clamped.
    /// </summary>
    public void NormalizeWhisper(
        ArrayView1D<float, Stride1D.Dense> data,
        int count,
        float maxValue)
    {
        float threshold = maxValue - 8f;
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                float val = d[idx];
                val = val > threshold ? val : threshold;
                d[idx] = (val + 4f) / 4f;
            });

        kernel((Index1D)count, data);
    }

    // ──────────────────────────────────────────────
    //  Audio resampling (linear interpolation)
    // ──────────────────────────────────────────────

    /// <summary>
    /// Resample audio on GPU using linear interpolation.
    /// One thread per output sample.
    /// </summary>
    public void Resample(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int inputLength, int outputLength)
    {
        float ratio = (float)inputLength / outputLength;

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> inp, ArrayView1D<float, Stride1D.Dense> outp) =>
            {
                float srcPos = idx * ratio;
                int i0 = (int)srcPos;
                float frac = srcPos - i0;
                int i1 = i0 + 1;
                if (i1 >= inputLength) i1 = inputLength - 1;
                outp[idx] = inp[i0] * (1f - frac) + inp[i1] * frac;
            });

        kernel((Index1D)outputLength, input, output);
    }

    // ──────────────────────────────────────────────
    //  Stereo to mono mixdown
    // ──────────────────────────────────────────────

    public void StereoToMono(
        ArrayView1D<float, Stride1D.Dense> stereo,
        ArrayView1D<float, Stride1D.Dense> mono,
        int monoLength)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> st, ArrayView1D<float, Stride1D.Dense> mo) =>
            {
                mo[idx] = (st[idx * 2] + st[idx * 2 + 1]) * 0.5f;
            });

        kernel((Index1D)monoLength, stereo, mono);
    }

    // ──────────────────────────────────────────────
    //  Power spectrum from STFT magnitudes
    // ──────────────────────────────────────────────

    /// <summary>
    /// Compute power spectrum (magnitude squared) on GPU.
    /// In-place: data[i] = data[i] * data[i]
    /// </summary>
    public void PowerSpectrum(
        ArrayView1D<float, Stride1D.Dense> magnitudes,
        int count)
    {
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            (Index1D idx, ArrayView1D<float, Stride1D.Dense> d) =>
            {
                float v = d[idx];
                d[idx] = v * v;
            });

        kernel((Index1D)count, magnitudes);
    }
}
