using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Preprocessing;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>The analysis window of a <see cref="SpectralFrontend"/> (periodic: the STFT convention, torch.hann_window(n) / scipy get_window).</summary>
public enum SpectralWindow
{
    /// <summary>0.5 - 0.5 cos(2 pi i / n).</summary>
    Hann,
    /// <summary>4-term Blackman-Harris (a0 0.35875, a1 0.48829, a2 0.14128, a3 0.01168): -92 dB side lobes, for transients next to loud tones.</summary>
    BlackmanHarris4,
    /// <summary>No window (tests, already-windowed input).</summary>
    Rectangular,
}

/// <summary>How <see cref="SpectralFrontend.SetBands"/> spaces its band centres between fMin and fMax.</summary>
public enum BandScale
{
    /// <summary>Mel (Slaney's 2595 log10(1 + f / 700)).</summary>
    Mel,
    /// <summary>Equal ratios (log-frequency).</summary>
    Log,
}

/// <summary>
/// Short-time spectra of multi-channel audio on the device (asked for by Riker for AssistZ, 2026-11-02: 48 kHz stereo from a GPU ring buffer, a frame every
/// 1024 samples; and the STFT under <see cref="WhisperMelPreprocessor"/>). Frames are read from a planar ring (<c>[channel][ringLength]</c>, ringLength a
/// power of two) by their END sample: frame k covers samples <c>[end_k - N, end_k)</c> with <c>end_k = firstEnd + k * hop</c> (positions counted from the
/// start of the stream, so they wrap through the ring by <c>& (ringLength - 1)</c>); every end must lie in the ring's live range
/// <c>[writePos - ringLength + N, writePos]</c>, <c>writePos</c> being the count of samples written so far. A power-of-two N runs a radix-2 FFT (the real
/// signal as an N/2-point complex FFT in shared memory, one group of 256 threads per frame and channel); any other N a direct real DFT. Outputs stay on the
/// device: power <c>[frames][channels][N/2 + 1]</c> (linear, or 10 log10(max(p, floor))), and optionally pooled bands <c>[frames][channels][bands]</c>
/// (triangular filters on a mel or log scale) with the per-band level difference of channel 0 over channel 1 in dB <c>[frames][bands]</c>.
/// </summary>
public sealed class SpectralFrontend : IDisposable
{
    /// <summary>Threads per frame of the FFT kernel.</summary>
    public const int FftThreads = 256;
    /// <summary>The largest power-of-two N the FFT kernel takes (its N/2 complex values fill 8 KB of shared memory at 1024; 16 KB at 2048 is WebGPU's minimum).</summary>
    public const int MaxFftSize = 4096;

    private readonly Accelerator _acc;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _window, _twiddle, _split;
    private MemoryBuffer1D<float, Stride1D.Dense>? _bandWeights;
    private MemoryBuffer1D<int, Stride1D.Dense>? _bandRanges;
    private readonly Action<KernelConfig, FV, FV, FV, FV, FV, SpectralArgs>? _fft;
    private readonly Action<Index1D, FV, FV, FV, FV, SpectralArgs>? _dft;
    private Action<Index1D, FV, FV, IV, FV, int, int, int, float>? _bands;
    private Action<Index1D, FV, FV, int, int, float>? _ild;
    private Action<Index1D, FV, float>? _toDb;

    /// <summary>Frame length (samples).</summary>
    public int N { get; }
    /// <summary>Power bins per frame and channel: N / 2 + 1.</summary>
    public int Bins => N / 2 + 1;
    /// <summary>Channels.</summary>
    public int Channels { get; }
    /// <summary>Sample rate (Hz): sets the band edges' bins.</summary>
    public int SampleRate { get; }
    /// <summary>The window.</summary>
    public SpectralWindow Window { get; }
    /// <summary>True when N is a power of two (the FFT); false runs the direct DFT.</summary>
    public bool UsesFft { get; }
    /// <summary>Bands set by <see cref="SetBands"/> (0 before).</summary>
    public int Bands { get; private set; }

    public SpectralFrontend(Accelerator accelerator, int n, int channels, int sampleRate, SpectralWindow window = SpectralWindow.Hann)
    {
        if (n < 4) throw new ArgumentOutOfRangeException(nameof(n), "a frame needs at least 4 samples");
        if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
        _acc = accelerator; N = n; Channels = channels; SampleRate = sampleRate; Window = window;
        UsesFft = (n & (n - 1)) == 0 && n <= MaxFftSize && accelerator.MaxNumThreadsPerGroup >= FftThreads && accelerator.AcceleratorType != AcceleratorType.WebGL;
        _window = accelerator.Allocate1D(WindowTable(n, window));
        if (UsesFft)
        {
            // twiddles of the N/2-point FFT, exp(-2 pi i j / (N/2)) for j < N/4, and of the real split, exp(-2 pi i k / N) for k <= N/2 (cos, sin pairs)
            int h = n / 2;
            var tw = new float[2 * Math.Max(1, h / 2)];
            for (int j = 0; j < h / 2; j++) { double a = -2 * Math.PI * j / h; tw[2 * j] = (float)Math.Cos(a); tw[2 * j + 1] = (float)Math.Sin(a); }
            var sp = new float[2 * (h + 1)];
            for (int k = 0; k <= h; k++) { double a = -2 * Math.PI * k / n; sp[2 * k] = (float)Math.Cos(a); sp[2 * k + 1] = (float)Math.Sin(a); }
            _twiddle = accelerator.Allocate1D(tw);
            _split = accelerator.Allocate1D(sp);
            _fft = accelerator.LoadStreamKernel<FV, FV, FV, FV, FV, SpectralArgs>(FftPowerImpl);
        }
        else
        {
            // the DFT's cos / sin table: entry m = exp(-2 pi i m / N) (index k * i mod N)
            var cs = new float[2 * n];
            for (int m = 0; m < n; m++) { double a = -2 * Math.PI * m / n; cs[2 * m] = (float)Math.Cos(a); cs[2 * m + 1] = (float)Math.Sin(a); }
            _twiddle = accelerator.Allocate1D(cs);
            _split = accelerator.Allocate1D<float>(1);
            _dft = accelerator.LoadAutoGroupedStreamKernel<Index1D, FV, FV, FV, FV, SpectralArgs>(DftPowerImpl);
        }
    }

    /// <summary>The window's samples (periodic), double-precision on the host.</summary>
    public static float[] WindowTable(int n, SpectralWindow window)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++)
        {
            double x = 2 * Math.PI * i / n;
            w[i] = window switch
            {
                SpectralWindow.Hann => (float)(0.5 - 0.5 * Math.Cos(x)),
                SpectralWindow.BlackmanHarris4 => (float)(0.35875 - 0.48829 * Math.Cos(x) + 0.14128 * Math.Cos(2 * x) - 0.01168 * Math.Cos(3 * x)),
                _ => 1f,
            };
        }
        return w;
    }

    /// <summary>
    /// Power spectra of <paramref name="frames"/> frames per channel into <paramref name="power"/> (<c>[frames][channels][Bins]</c>). <paramref name="ring"/> is
    /// planar, <c>[channels][ringLength]</c>; see the class summary for the frame addressing and the live range. <paramref name="logFloor"/> &gt; 0 writes
    /// 10 log10(max(p, logFloor)) instead of p.
    /// </summary>
    public void Power(FV ring, int ringLength, long writePos, long firstEnd, int hop, int frames, FV power, float logFloor = 0f)
    {
        CheckRange(ringLength, writePos, firstEnd, hop, frames);
        if (power.Length < (long)frames * Channels * Bins) throw new ArgumentException($"power needs {(long)frames * Channels * Bins} floats", nameof(power));
        if (ring.Length < (long)Channels * ringLength) throw new ArgumentException($"ring needs {(long)Channels * ringLength} floats", nameof(ring));
        var args = new SpectralArgs
        {
            N = N, Channels = Channels, RingLength = ringLength, Mask = ringLength - 1, Hop = hop, Frames = frames,
            // the end positions wrap through the ring: only their value mod ringLength matters, so they travel as 32-bit
            FirstEndMod = (int)(firstEnd & (ringLength - 1)), LogFloor = logFloor,
        };
        if (UsesFft) _fft!(new KernelConfig(frames * Channels, FftThreads), ring, _window.View, _twiddle.View, _split.View, power, args);
        else _dft!((int)((long)frames * Channels * Bins), ring, _window.View, _twiddle.View, power.SubView(0, (long)frames * Channels * Bins), args);
    }

    /// <summary>
    /// Validates a frame request against the ring: ringLength a power of two at least N, hop &gt; 0, and every end in <c>[writePos - ringLength + N, writePos]</c>.
    /// Public so a caller can check before it schedules work.
    /// </summary>
    public void CheckRange(int ringLength, long writePos, long firstEnd, int hop, int frames)
    {
        if (ringLength < N || (ringLength & (ringLength - 1)) != 0) throw new ArgumentException($"ringLength {ringLength} must be a power of two of at least N = {N}", nameof(ringLength));
        if (frames < 1) throw new ArgumentOutOfRangeException(nameof(frames));
        if (hop < 1 && frames > 1) throw new ArgumentOutOfRangeException(nameof(hop));
        long lastEnd = firstEnd + (long)(frames - 1) * hop, oldest = writePos - ringLength + N;
        if (firstEnd < Math.Max(N, oldest) || lastEnd > writePos)
            throw new ArgumentOutOfRangeException(nameof(firstEnd), $"frame ends {firstEnd}..{lastEnd} leave the ring's live range [{Math.Max(N, oldest)}, {writePos}] (writePos {writePos}, ring {ringLength}, N {N})");
    }

    /// <summary>
    /// Triangular band filters for <see cref="PoolBands"/>: <paramref name="bands"/> bands whose centres are equally spaced on <paramref name="scale"/> between
    /// <paramref name="fMinHz"/> and <paramref name="fMaxHz"/> (edges one step beyond each side), weights by the bins' centre frequencies.
    /// </summary>
    public void SetBands(int bands, float fMinHz, float fMaxHz, BandScale scale = BandScale.Mel)
    {
        if (bands < 1) throw new ArgumentOutOfRangeException(nameof(bands));
        if (!(fMinHz > 0 && fMaxHz > fMinHz && fMaxHz <= SampleRate / 2f)) throw new ArgumentOutOfRangeException(nameof(fMaxHz), $"need 0 < fMin < fMax <= {SampleRate / 2}");
        var (ranges, weights) = BandFilters(Bins, N, SampleRate, bands, fMinHz, fMaxHz, scale);
        _bandWeights?.Dispose(); _bandRanges?.Dispose();
        _bandWeights = _acc.Allocate1D(weights);
        _bandRanges = _acc.Allocate1D(ranges);
        _bands ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, int, int, int, float>(BandsImpl);
        Bands = bands;
    }

    /// <summary>
    /// The host band filters: ranges[2b] = first bin, ranges[2b + 1] = offset of band b's weights in the weight array (band b covers bins
    /// first .. first + (offset[b + 1] - offset[b]) - 1); one extra offset closes the array. Public: the tests' oracle uses the same filters.
    /// </summary>
    public static (int[] Ranges, float[] Weights) BandFilters(int bins, int n, int sampleRate, int bands, float fMinHz, float fMaxHz, BandScale scale)
    {
        Func<double, double> fwd = scale == BandScale.Mel ? f => 2595 * Math.Log10(1 + f / 700) : f => Math.Log(f);
        Func<double, double> inv = scale == BandScale.Mel ? m => 700 * (Math.Pow(10, m / 2595) - 1) : m => Math.Exp(m);
        double lo = fwd(fMinHz), hi = fwd(fMaxHz), step = bands > 1 ? (hi - lo) / (bands - 1) : (hi - lo);
        var ranges = new int[2 * bands + 1];
        var weights = new List<float>();
        for (int b = 0; b < bands; b++)
        {
            double c = inv(lo + b * step), l = inv(lo + (b - 1) * step), r = inv(lo + (b + 1) * step);
            int first = -1;
            int start = weights.Count;
            for (int k = 0; k < bins; k++)
            {
                double f = (double)k * sampleRate / n;
                double wgt = f <= l || f >= r ? 0 : f <= c ? (f - l) / (c - l) : (r - f) / (r - c);
                if (wgt > 0) { if (first < 0) first = k; weights.Add((float)wgt); }
                else if (first >= 0) break;
            }
            if (first < 0)
            {
                // narrower than a bin: the nearest bin, weight 1
                first = (int)Math.Clamp(Math.Round(c * n / sampleRate), 0, bins - 1);
                weights.Add(1f);
            }
            ranges[2 * b] = first; ranges[2 * b + 1] = start;
        }
        ranges[2 * bands] = weights.Count;
        return (ranges, weights.ToArray());
    }

    /// <summary>
    /// Band energies <c>[frames][channels][Bands]</c> from linear power <c>[frames][channels][Bins]</c> (<see cref="Power"/> with logFloor 0); logFloor &gt; 0
    /// writes 10 log10(max(e, logFloor)). With <paramref name="levelDifference"/> (<c>[frames][Bands]</c>, two channels or more) also the level of channel 0 over
    /// channel 1 in dB per band: 10 log10((e0 + floor) / (e1 + floor)), floor = max(logFloor, 1e-12).
    /// </summary>
    public void PoolBands(FV power, int frames, FV bands, float logFloor = 0f, FV levelDifference = default)
    {
        if (Bands == 0 || _bandRanges == null) throw new InvalidOperationException("SetBands first");
        long count = (long)frames * Channels * Bands;
        if (bands.Length < count) throw new ArgumentException($"bands needs {count} floats", nameof(bands));
        bool ild = levelDifference.IsValid && levelDifference.Length > 0;
        if (ild && Channels < 2) throw new ArgumentException("a level difference needs two channels", nameof(levelDifference));
        if (ild && levelDifference.Length < (long)frames * Bands) throw new ArgumentException($"levelDifference needs {(long)frames * Bands} floats", nameof(levelDifference));
        // the ILD needs the linear energies: pool linear when it is asked for, then the dB pass below converts
        _bands!((int)count, power, _bandWeights!.View, _bandRanges.View, bands.SubView(0, count), Bins, Bands, Channels, ild ? 0f : logFloor);
        if (ild)
        {
            _ild ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, int, int, float>(LevelDifferenceImpl);
            _ild((int)((long)frames * Bands), bands, levelDifference.SubView(0, (long)frames * Bands), Bands, Channels, logFloor);
            if (logFloor > 0f)
            {
                // then the energies to dB, one store per thread (WebGL takes no scatter)
                _toDb ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, float>(ToDbImpl);
                _toDb((int)count, bands.SubView(0, count), logFloor);
            }
        }
    }

    public void Dispose()
    {
        _window.Dispose(); _twiddle.Dispose(); _split.Dispose();
        _bandWeights?.Dispose(); _bandRanges?.Dispose();
    }

    /// <summary>The kernels' scalar arguments.</summary>
    public struct SpectralArgs
    {
        public int N, Channels, RingLength, Mask, Hop, Frames, FirstEndMod;
        public float LogFloor;
    }

    // ---- kernels ----

    // the sample at stream position (end - N + i) of channel ch, windowed (32-bit indexes: WebGPU emulates 64-bit arithmetic, and a ring is far below 2^31)
    private static float Sample(FV ring, FV win, SpectralArgs a, int ch, int endMod, int i) =>
        ring[ch * a.RingLength + ((endMod - a.N + i) & a.Mask)] * win[i];

    // radix-2 real FFT power: group g = (frame f, channel ch); z[m] = x[2m] + i x[2m+1] (windowed), Z = FFT_{N/2}(z) in shared memory (bit-reversed load,
    // in-place decimation-in-time stages), then X[k] = (Z[k] + conj Z[h-k]) / 2 - i e^{-2 pi i k / N} (Z[k] - conj Z[h-k]) / 2 for k = 0..h (Z[h] = Z[0])
    private static void FftPowerImpl(FV ring, FV win, FV tw, FV split, FV power, SpectralArgs a)
    {
        int g = Grid.IdxX, t = Group.IdxX;
        int f = g / a.Channels, ch = g % a.Channels;
        int h = a.N >> 1, bins = h + 1;
        var re = SharedMemory.Allocate<float>(MaxFftSize / 2);
        var im = SharedMemory.Allocate<float>(MaxFftSize / 2);
        int endMod = (a.FirstEndMod + f * a.Hop) & a.Mask;
        int bits = 0; while ((1 << bits) < h) bits++;
        for (int m = t; m < h; m += FftThreads)
        {
            int rev = 0, x = m;
            for (int b = 0; b < bits; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
            re[rev] = Sample(ring, win, a, ch, endMod, 2 * m);
            im[rev] = Sample(ring, win, a, ch, endMod, 2 * m + 1);
        }
        Group.Barrier();
        for (int half = 1; half < h; half <<= 1)
        {
            int stride = h / (2 * half);                                // twiddle index step: exp(-2 pi i j / (2 half)) = tw[j * stride]
            for (int bf = t; bf < (h >> 1); bf += FftThreads)
            {
                int j = bf % half, i0 = (bf / half) * 2 * half + j, i1 = i0 + half;
                float wr = tw[2 * (j * stride)], wi = tw[2 * (j * stride) + 1];
                float xr = re[i1] * wr - im[i1] * wi, xi = re[i1] * wi + im[i1] * wr;
                float ur = re[i0], ui = im[i0];
                re[i0] = ur + xr; im[i0] = ui + xi;
                re[i1] = ur - xr; im[i1] = ui - xi;
            }
            Group.Barrier();
        }
        long ob = (long)g * bins;
        for (int k = t; k <= h; k += FftThreads)
        {
            int k1 = k == h ? 0 : k, k2 = k == 0 ? 0 : h - k;
            float ar = re[k1], ai = im[k1], br = re[k2], bi = -im[k2];          // Z[k], conj Z[h - k]
            float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);                  // even part
            float dr = 0.5f * (ar - br), di = 0.5f * (ai - bi);                  // (Z[k] - conj Z[h-k]) / 2
            float cr = split[2 * k], ci = split[2 * k + 1];                     // e^{-2 pi i k / N}
            // odd part = -i * c * d
            float pr = cr * dr - ci * di, pi = cr * di + ci * dr;
            float xr = er + pi, xi = ei - pr;
            float p = xr * xr + xi * xi;
            power[ob + k] = a.LogFloor > 0f ? 10f * MathF.Log10(MathF.Max(p, a.LogFloor)) : p;
        }
    }

    // direct real DFT power, a thread per (frame, channel, bin): any N (Whisper's 400)
    private static void DftPowerImpl(Index1D idx, FV ring, FV win, FV cs, FV power, SpectralArgs a)
    {
        int bins = a.N / 2 + 1;
        int i = idx, k = i % bins, fc = i / bins, f = fc / a.Channels, ch = fc % a.Channels;
        int endMod = (a.FirstEndMod + f * a.Hop) & a.Mask;
        float sr = 0f, si = 0f;
        int m = 0;
        for (int n = 0; n < a.N; n++)
        {
            float x = Sample(ring, win, a, ch, endMod, n);
            sr += x * cs[2 * m]; si += x * cs[2 * m + 1];
            m += k; if (m >= a.N) m -= a.N;
        }
        float p = sr * sr + si * si;
        power[i] = a.LogFloor > 0f ? 10f * MathF.Log10(MathF.Max(p, a.LogFloor)) : p;
    }

    // band energy: a thread per (frame, channel, band), its weights over its bins
    private static void BandsImpl(Index1D idx, FV power, FV weights, IV ranges, FV bands, int bins, int nb, int channels, float logFloor)
    {
        int i = idx, b = i % nb, fc = i / nb;
        // band b's weights are [offset_b, offset_{b+1}); the last band's end is the closing entry ranges[2 nb]
        int first = ranges[2 * b], w0 = ranges[2 * b + 1], w1 = b == nb - 1 ? ranges[2 * nb] : ranges[2 * b + 3];
        int pb = fc * bins + first - w0;
        float e = 0f;
        for (int w = w0; w < w1; w++) e += weights[w] * power[pb + w];
        bands[i] = logFloor > 0f ? 10f * MathF.Log10(MathF.Max(e, logFloor)) : e;
    }

    // per band: channel 0 over channel 1 in dB, from the linear energies
    private static void LevelDifferenceImpl(Index1D idx, FV bands, FV ild, int nb, int channels, float logFloor)
    {
        int i = idx, b = i % nb, f = i / nb;
        long b0 = ((long)f * channels) * nb + b, b1 = b0 + nb;
        float fl = MathF.Max(logFloor, 1e-12f);
        ild[i] = 10f * MathF.Log10((bands[b0] + fl) / (bands[b1] + fl));
    }

    private static void ToDbImpl(Index1D i, FV x, float floor) => x[i] = 10f * MathF.Log10(MathF.Max(x[i], floor));
}
