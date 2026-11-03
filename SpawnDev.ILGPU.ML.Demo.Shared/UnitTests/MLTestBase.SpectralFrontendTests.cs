using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // the test signal: two tones per channel, a chirp and seeded noise; the right channel is the left at half amplitude plus its own tone, so its bands carry a
    // known level difference where the left's tones dominate
    private static float[] SpectralTestSignal(int channels, int ringLength, long writePos, int sampleRate, int seed)
    {
        var rng = new Random(seed);
        var noise = new float[ringLength];
        for (int i = 0; i < ringLength; i++) noise[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
        var ring = new float[channels * ringLength];
        // fill the ring with stream positions writePos - ringLength .. writePos - 1 at their wrapped slots
        for (long pos = writePos - ringLength; pos < writePos; pos++)
        {
            double t = (double)pos / sampleRate;
            double left = 0.6 * Math.Sin(2 * Math.PI * 440 * t) + 0.3 * Math.Sin(2 * Math.PI * 3150.7 * t) + 0.1 * Math.Sin(2 * Math.PI * (200 + 50 * t) * t);
            int slot = (int)(pos & (ringLength - 1));
            ring[slot] = (float)left + noise[slot];
            if (channels > 1) ring[ringLength + slot] = (float)(0.5 * left + 0.2 * Math.Sin(2 * Math.PI * 9001 * t)) + noise[(slot * 7) & (ringLength - 1)];
        }
        return ring;
    }

    // CPU oracle: the windowed frame ending at stream position end, channel ch, its power per bin (double)
    private static double[] OraclePower(float[] ring, int ringLength, int ch, long end, int n, float[] window)
    {
        int bins = n / 2 + 1;
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = ring[ch * ringLength + (int)((end - n + i) & (ringLength - 1))] * (double)window[i];
        var p = new double[bins];
        for (int k = 0; k < bins; k++)
        {
            double sr = 0, si = 0;
            for (int i = 0; i < n; i++) { double a = -2 * Math.PI * ((long)k * i % n) / n; sr += x[i] * Math.Cos(a); si += x[i] * Math.Sin(a); }
            p[k] = sr * sr + si * si;
        }
        return p;
    }

    /// <summary>
    /// <see cref="SpectralFrontend.Power"/> against a CPU double DFT on a ring that has wrapped several times (48 kHz stereo, a 65,536-sample planar ring):
    /// N = 2048 Hann (the radix-2 FFT where the backend has it) with frames every 1024 samples ending AT the wrap, STRADDLING it, at the OLDEST valid end and at
    /// the newest (writePos), batched; N = 2048 Blackman-Harris; N = 400 mono (Whisper's frame: the direct DFT). Each bin within 2e-5 x the frame's total power
    /// (float FFT rounding; Parseval) + 1e-7. A frame ending one sample past writePos, or one sample older than the live range, must throw. Log power:
    /// 10 log10(max(p, floor)). Added 2026-11-02 (Riker's AssistZ request).
    /// </summary>
    [TestMethod]
    public async Task SpectralFrontend_Power_MatchesDft_AcrossTheRingWrap() => await RunTest(async accelerator =>
    {
        const int Rate = 48000, Ring = 65536;
        var notes = new List<string>();
        foreach (var (n, channels, window, hop) in new[] { (2048, 2, SpectralWindow.Hann, 1024), (2048, 2, SpectralWindow.BlackmanHarris4, 1024), (400, 1, SpectralWindow.Hann, 160) })
        {
            long writePos = 3L * Ring + 12345;                          // the stream has wrapped three times
            var host = SpectralTestSignal(channels, Ring, writePos, Rate, 11 + n);
            using var fe = new SpectralFrontend(accelerator, n, channels, Rate, window);
            using var ringBuf = accelerator.Allocate1D(host);
            var win = SpectralFrontend.WindowTable(n, window);
            long wrapEnd = writePos - (writePos & (Ring - 1));             // the newest stream position that is a multiple of the ring: a frame ending AT the wrap
            var batches = new (string Name, long FirstEnd, int Frames)[]
            {
                ("ends at the wrap", wrapEnd, 1),
                ("straddles the wrap", wrapEnd + n / 2, 1),
                ("oldest valid end", writePos - Ring + n, 1),
                ("newest, batched", writePos - 7L * hop, 8),
            };
            foreach (var (name, firstEnd, frames) in batches)
            {
                using var power = accelerator.Allocate1D<float>((long)frames * channels * fe.Bins);
                fe.Power(ringBuf.View, Ring, writePos, firstEnd, hop, frames, power.View);
                await accelerator.SynchronizeAsync();
                var got = await power.CopyToHostAsync<float>(0, power.Length);
                double worst = 0;
                for (int f = 0; f < frames; f++)
                    for (int c = 0; c < channels; c++)
                    {
                        var want = OraclePower(host, Ring, c, firstEnd + (long)f * hop, n, win);
                        double total = want.Sum();
                        for (int k = 0; k < fe.Bins; k++)
                        {
                            double e = Math.Abs(got[(f * channels + c) * fe.Bins + k] - want[k]), bound = 2e-5 * total + 1e-7;
                            worst = Math.Max(worst, e / bound);
                            if (!(e <= bound)) throw new Exception($"N={n} {window} {name}: frame {f} channel {c} bin {k}: {got[(f * channels + c) * fe.Bins + k]} vs {want[k]} (bound {bound:G3}; {(fe.UsesFft ? "FFT" : "DFT")})");
                        }
                    }
                notes.Add($"N={n} {window} {name}: {(fe.UsesFft ? "FFT" : "DFT")} worst {worst:F3}x");
            }
            // log power, on the batched case
            {
                const float Floor = 1e-10f;
                using var power = accelerator.Allocate1D<float>(channels * fe.Bins);
                fe.Power(ringBuf.View, Ring, writePos, writePos, hop, 1, power.View, Floor);
                await accelerator.SynchronizeAsync();
                var got = await power.CopyToHostAsync<float>(0, power.Length);
                var want = OraclePower(host, Ring, 0, writePos, n, win);
                double total = want.Sum(), worstDb = 0;
                for (int k = 0; k < fe.Bins; k++)
                {
                    if (want[k] < 1e-3 * total) continue;                   // compare dB where the bin carries the signal (rounding at -100 dB is noise)
                    worstDb = Math.Max(worstDb, Math.Abs(got[k] - 10 * Math.Log10(Math.Max(want[k], Floor))));
                }
                if (worstDb > 0.01) throw new Exception($"N={n} {window}: log power off by {worstDb:F4} dB");
            }
            // outside the live range: one past the newest, one older than the oldest
            foreach (var (bad, end) in new[] { ("one past writePos", writePos + 1), ("one older than the ring holds", writePos - Ring + n - 1) })
            {
                try { fe.CheckRange(Ring, writePos, end, hop, 1); throw new Exception($"N={n}: a frame {bad} was accepted"); }
                catch (ArgumentOutOfRangeException) { }
            }
        }
        return string.Join("; ", notes);
    });

    /// <summary>
    /// <see cref="SpectralFrontend.PoolBands"/>: 48 mel bands and 32 log bands, 40 Hz .. 20 kHz, N = 2048 stereo at 48 kHz, against the same triangular filters
    /// (<see cref="SpectralFrontend.BandFilters"/>) applied to the CPU oracle's power (within 1e-4 relative + 1e-7), in dB with a floor, and the per-band level
    /// difference of channel 0 over channel 1: the left's 440 Hz tone band must read the 0.5x amplitude ratio, +6.02 dB (within 0.25 dB), and every band must
    /// match 10 log10 of the oracle energies' ratio within 0.05 dB. Added 2026-11-02.
    /// </summary>
    [TestMethod]
    public async Task SpectralFrontend_Bands_AndLevelDifference_MatchOracle() => await RunTest(async accelerator =>
    {
        const int Rate = 48000, Ring = 65536, N = 2048, Hop = 1024, Frames = 4, Ch = 2;
        long writePos = 2L * Ring + 777;
        var host = SpectralTestSignal(Ch, Ring, writePos, Rate, 23);
        using var fe = new SpectralFrontend(accelerator, N, Ch, Rate);
        using var ringBuf = accelerator.Allocate1D(host);
        var win = SpectralFrontend.WindowTable(N, SpectralWindow.Hann);
        long firstEnd = writePos - (Frames - 1) * Hop;
        using var power = accelerator.Allocate1D<float>((long)Frames * Ch * fe.Bins);
        fe.Power(ringBuf.View, Ring, writePos, firstEnd, Hop, Frames, power.View);
        var oracle = new double[Frames, Ch][];
        for (int f = 0; f < Frames; f++) for (int c = 0; c < Ch; c++) oracle[f, c] = OraclePower(host, Ring, c, firstEnd + (long)f * Hop, N, win);
        var notes = new List<string>();
        foreach (var (nb, scale) in new[] { (48, BandScale.Mel), (32, BandScale.Log) })
        {
            fe.SetBands(nb, 40f, 20000f, scale);
            var (ranges, weights) = SpectralFrontend.BandFilters(fe.Bins, N, Rate, nb, 40f, 20000f, scale);
            double OracleBand(int f, int c, int b)
            {
                int first = ranges[2 * b], w0 = ranges[2 * b + 1], w1 = b == nb - 1 ? ranges[2 * nb] : ranges[2 * b + 3];
                double e = 0; for (int w = w0; w < w1; w++) e += weights[w] * oracle[f, c][first + w - w0];
                return e;
            }
            // linear bands
            using var bands = accelerator.Allocate1D<float>((long)Frames * Ch * nb);
            fe.PoolBands(power.View, Frames, bands.View);
            await accelerator.SynchronizeAsync();
            var got = await bands.CopyToHostAsync<float>(0, bands.Length);
            // the power's float error scales with the frame's total (Parseval): a band is held to 1e-4 of itself plus 2e-5 of the frame's total power
            for (int f = 0; f < Frames; f++) for (int c = 0; c < Ch; c++) for (int b = 0; b < nb; b++)
            {
                double want = OracleBand(f, c, b), e = Math.Abs(got[(f * Ch + c) * nb + b] - want), total = oracle[f, c].Sum();
                if (!(e <= 1e-4 * Math.Abs(want) + 2e-5 * total + 1e-7)) throw new Exception($"{scale} {nb}: frame {f} ch {c} band {b}: {got[(f * Ch + c) * nb + b]} vs {want}");
            }
            // dB bands + the level difference
            const float Floor = 1e-10f;
            using var ild = accelerator.Allocate1D<float>((long)Frames * nb);
            fe.PoolBands(power.View, Frames, bands.View, Floor, ild.View);
            await accelerator.SynchronizeAsync();
            var gotDb = await bands.CopyToHostAsync<float>(0, bands.Length);
            var gotIld = await ild.CopyToHostAsync<float>(0, ild.Length);
            double worstIld = 0, worstDb = 0;
            int toneBand = -1; double toneE = 0;
            for (int b = 0; b < nb; b++) { double e = OracleBand(0, 0, b); int first = ranges[2 * b]; if (first * (double)Rate / N <= 440 && 440 <= (first + (b == nb - 1 ? ranges[2 * nb] : ranges[2 * b + 3]) - ranges[2 * b + 1]) * (double)Rate / N && e > toneE) { toneE = e; toneBand = b; } }
            // dB and level differences where the band carries signal (>= 1e-3 of the frame's total): in noise-only bands the float rounding is the reading
            for (int f = 0; f < Frames; f++) for (int b = 0; b < nb; b++)
            {
                double e0 = OracleBand(f, 0, b), e1 = OracleBand(f, 1, b), t0 = 1e-3 * oracle[f, 0].Sum(), t1 = 1e-3 * oracle[f, 1].Sum();
                if (e0 >= t0 && e1 >= t1) worstIld = Math.Max(worstIld, Math.Abs(gotIld[f * nb + b] - 10 * Math.Log10((e0 + Floor) / (e1 + Floor))));
                if (e0 >= t0) worstDb = Math.Max(worstDb, Math.Abs(gotDb[(f * Ch) * nb + b] - 10 * Math.Log10(Math.Max(e0, Floor))));
                if (e1 >= t1) worstDb = Math.Max(worstDb, Math.Abs(gotDb[(f * Ch + 1) * nb + b] - 10 * Math.Log10(Math.Max(e1, Floor))));
            }
            if (worstIld > 0.05 || worstDb > 0.05) throw new Exception($"{scale} {nb}: level difference off by {worstIld:F4} dB, band dB by {worstDb:F4} dB");
            if (toneBand < 0 || Math.Abs(gotIld[toneBand] - 6.0206) > 0.25) throw new Exception($"{scale} {nb}: the 440 Hz band ({toneBand}) reads {(toneBand >= 0 ? gotIld[toneBand] : float.NaN):F3} dB, the 0.5x channel needs +6.02");
            notes.Add($"{scale} {nb}: ILD worst {worstIld:F4} dB, 440 Hz band {gotIld[toneBand]:F3} dB");
        }
        return string.Join("; ", notes);
    });
}
