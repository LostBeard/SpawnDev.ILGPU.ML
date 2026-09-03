using System;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for <see cref="AudioPreprocessor.Resample"/>, which sits between every microphone and every
/// speech model we run.
///
/// <para>
/// WHY THIS FILE EXISTS: <c>Resample</c> was bare linear interpolation, carrying a comment conceding
/// "for production quality, consider using a proper sinc resampler". Linear interpolation does not
/// band-limit, so decimating 48 kHz microphone audio to 16 kHz folded everything from 8-24 kHz back on
/// top of the speech. Whisper answered with fluent, confident, unrelated text
/// ("We're wrapping up.", "...a little bit of sauce and that rolled it up and baked it...").
/// </para>
///
/// <para>
/// ⚠️ It was invisible even though a test already exercised the method.
/// <c>AudioPreprocessor_Resample_Frequency</c> calls <c>Resample(samples, 44100, 16000)</c> and asserts
/// the output LENGTH (within one sample) and that values sit in [-1.1, 1.1]. Aliasing violates neither.
/// Its input is a 440 Hz tone, far below the 8 kHz destination Nyquist, so there is nothing in it to
/// alias - it runs the real conversion and CANNOT FAIL. Having a test for a function is not coverage of
/// what the function must get right: a rate conversion has to be fed content ABOVE the destination
/// Nyquist before "does it band-limit" is even asked.
/// </para>
///
/// <para>
/// The decisive assertion is <see cref="Resample_AboveDestinationNyquist_IsAttenuatedNotAliased"/>: a
/// 10 kHz tone resampled 48k -> 16k must be REMOVED, because 10 kHz cannot exist below an 8 kHz Nyquist.
/// Linear interpolation passes it through folded down to 6 kHz at nearly full amplitude, so that test
/// fails loudly on the old implementation and is the specific regression guard for this bug.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private static float[] Tone(int rate, double hz, double seconds, double amplitude = 0.5)
    {
        var n = (int)(rate * seconds);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(amplitude * Math.Sin(2 * Math.PI * hz * i / rate));
        return x;
    }

    /// <summary>Peak amplitude away from the edges, where any windowed kernel legitimately rolls off.</summary>
    private static double PeakInterior(float[] x, double skipFraction = 0.15)
    {
        int skip = (int)(x.Length * skipFraction);
        double peak = 0;
        for (int i = skip; i < x.Length - skip; i++) peak = Math.Max(peak, Math.Abs(x[i]));
        return peak;
    }

    /// <summary>
    /// THE REGRESSION GUARD. 10 kHz cannot be represented at 16 kHz (Nyquist 8 kHz), so a correct
    /// resampler filters it out. Linear interpolation instead aliases it to 6 kHz at close to full
    /// amplitude - inaudible as a defect, fatal to a mel spectrogram.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_AboveDestinationNyquist_IsAttenuatedNotAliased()
    {
        var tone = Tone(48000, 10000, 0.5);
        var got = AudioPreprocessor.Resample(tone, 48000, 16000);

        var peak = PeakInterior(got);
        // Source amplitude is 0.5. Anything close to that survived as an alias.
        if (peak > 0.1)
            throw new Exception(
                $"a 10 kHz tone survived 48k->16k at peak {peak:F3} (source 0.5). It has aliased to 6 kHz "
                + "instead of being filtered out - the resampler is not band-limiting before decimation");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The other half: filtering must not eat the speech band. A 1 kHz tone is far below the 8 kHz
    /// Nyquist and has to come through at essentially full amplitude - otherwise the "fix" for aliasing
    /// is just a low-pass that destroys the signal.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_BelowDestinationNyquist_PassesThroughAtFullAmplitude()
    {
        var tone = Tone(48000, 1000, 0.5);
        var got = AudioPreprocessor.Resample(tone, 48000, 16000);

        int expectedLen = 48000 / 3 / 2;   // 0.5 s at 16 kHz
        if (Math.Abs(got.Length - expectedLen) > 4)
            throw new Exception($"expected ~{expectedLen} samples, got {got.Length}");

        var peak = PeakInterior(got);
        if (peak < 0.45 || peak > 0.55)
            throw new Exception($"1 kHz tone came through at peak {peak:F3}, expected about 0.5 - "
                              + "the passband is not flat");
        return Task.CompletedTask;
    }

    /// <summary>A constant must survive any correct resampler unchanged; a gain or windowing error shows here.</summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_PreservesDcLevel()
    {
        var dc = new float[48000];
        for (int i = 0; i < dc.Length; i++) dc[i] = 0.25f;

        var got = AudioPreprocessor.Resample(dc, 48000, 16000);
        for (int i = 8; i < got.Length - 8; i++)
            if (Math.Abs(got[i] - 0.25f) > 1e-3f)
                throw new Exception($"sample {i} is {got[i]}, expected the 0.25 DC level to survive");
        return Task.CompletedTask;
    }

    /// <summary>Upsampling is interpolation, not decimation: a 1 kHz tone must survive 16k -> 48k intact.</summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_Upsampling_PreservesTone()
    {
        var tone = Tone(16000, 1000, 0.5);
        var got = AudioPreprocessor.Resample(tone, 16000, 48000);

        if (Math.Abs(got.Length - 24000) > 4)
            throw new Exception($"expected ~24000 samples for 0.5 s at 48 kHz, got {got.Length}");

        var peak = PeakInterior(got);
        if (peak < 0.45 || peak > 0.55)
            throw new Exception($"1 kHz tone upsampled to peak {peak:F3}, expected about 0.5");
        return Task.CompletedTask;
    }

    /// <summary>Equal rates must be a no-op, and must not silently copy or rescale.</summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_EqualRates_IsIdentity()
    {
        var tone = Tone(16000, 440, 0.1);
        var got = AudioPreprocessor.Resample(tone, 16000, 16000);
        if (got.Length != tone.Length)
            throw new Exception($"length changed: {tone.Length} -> {got.Length}");
        for (int i = 0; i < tone.Length; i++)
            if (got[i] != tone[i])
                throw new Exception($"sample {i} changed from {tone[i]} to {got[i]}");
        return Task.CompletedTask;
    }

    /// <summary>
    /// BRANCH COVERAGE for the kernel table. Resample precomputes one kernel per distinct output phase,
    /// and there are <c>dstRate / gcd(srcRate, dstRate)</c> of those - one, for 48k -> 16k. Coprime rates
    /// blow that up, so past 4096 phases it falls back to evaluating the kernel per sample. 48000 -> 16001
    /// is coprime, which forces the fallback.
    ///
    /// <para>
    /// Without this, every resampler test above would take the table path and the fallback would never
    /// execute - the same shape as the original bug, where the one pre-existing test drove the method with
    /// a 440 Hz tone that had nothing to alias. The invariant asserted is the physical one, not "both
    /// branches agree": 10 kHz cannot exist below an 8 kHz Nyquist either way.
    /// </para>
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task Resample_CoprimeRates_FallbackPath_StillBandLimits()
    {
        var tone = Tone(48000, 10000, 0.5);
        var got = AudioPreprocessor.Resample(tone, 48000, 16001);   // gcd 1 -> 16001 phases -> fallback

        var peak = PeakInterior(got);
        if (peak > 0.1)
            throw new Exception(
                $"a 10 kHz tone survived 48000 -> 16001 at peak {peak:F3} (source 0.5) - the per-sample "
                + "fallback path is not band-limiting");

        // And the passband still has to survive that path.
        var low = AudioPreprocessor.Resample(Tone(48000, 1000, 0.5), 48000, 16001);
        var lowPeak = PeakInterior(low);
        if (lowPeak < 0.45 || lowPeak > 0.55)
            throw new Exception($"1 kHz through the fallback path peaked at {lowPeak:F3}, expected ~0.5");
        return Task.CompletedTask;
    }

    /// <summary>
    /// <see cref="StreamingResampler"/> must produce EXACTLY what the whole-buffer call produces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ THE DEFECT THIS GUARDS. <c>Resample</c> renormalises the kernel where the window overruns the
    /// SIGNAL EDGE, so applying it per chunk of a live stream declares every chunk boundary an edge. A
    /// microphone delivers a frame every ~10 ms, so that is an artifact 100 times a second - a steady
    /// broadband tick, which is exactly what a voice-activity detector reports as speech.
    /// <c>MediaInterop.FromAudioDataAsync(data, targetRate)</c> did this on every frame, so ANY caller
    /// asking capture for a target rate got it.
    /// </para>
    /// <para>
    /// ⚠️ Chunk sizes here deliberately do NOT divide the 3:1 decimation factor or each other. A chunk
    /// size that lands on a phase boundary can be correct by accident, so testing only 480-sample frames
    /// would let a genuinely broken streamer pass.
    /// </para>
    /// <para>
    /// ⚠️ EXACT equality, not a tolerance. A near-miss here means the boundary artifact is still present
    /// at reduced amplitude - harder to see, no less wrong - and a tolerance is how it would ship.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 120000)]
    public Task Streaming_MatchesWholeBufferResample()
    {
        // Content ABOVE the destination Nyquist plus content below it: the high tone is what makes an
        // edge artifact visible at all, per the file header.
        var source = Tone(48000, 10000, 0.75, 0.4);
        var speechish = Tone(48000, 1200, 0.75, 0.4);
        for (int i = 0; i < source.Length; i++) source[i] += speechish[i];

        foreach (var chunkSize in new[] { 1, 7, 128, 480, 1024, 4099 })
        {
            var expected = AudioPreprocessor.Resample(source, 48000, 16000);

            var streamer = new StreamingResampler(48000, 16000);
            var got = new System.Collections.Generic.List<float>();
            for (int off = 0; off < source.Length; off += chunkSize)
            {
                int n = Math.Min(chunkSize, source.Length - off);
                var chunk = new float[n];
                Array.Copy(source, off, chunk, 0, n);
                got.AddRange(streamer.Process(chunk));
            }
            got.AddRange(streamer.Flush());

            if (got.Count != expected.Length)
                throw new Exception($"chunk {chunkSize}: streamed {got.Count} samples, whole-buffer "
                                  + $"produced {expected.Length}");

            for (int i = 0; i < expected.Length; i++)
                if (got[i] != expected[i])
                    throw new Exception($"chunk {chunkSize}: sample {i} is {got[i]} streamed vs "
                                      + $"{expected[i]} whole-buffer. The streamer is treating a chunk "
                                      + "boundary as a signal edge.");
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// The coprime FALLBACK path streams exactly too - it computes kernels per sample rather than from
    /// the phase table, so it is a separate implementation and needs its own equality proof.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public Task Streaming_CoprimeFallback_MatchesWholeBufferResample()
    {
        var source = Tone(48000, 3000, 0.3, 0.4);
        var expected = AudioPreprocessor.Resample(source, 48000, 16001);   // gcd 1 -> fallback

        var streamer = new StreamingResampler(48000, 16001);
        var got = new System.Collections.Generic.List<float>();
        const int chunkSize = 997;
        for (int off = 0; off < source.Length; off += chunkSize)
        {
            int n = Math.Min(chunkSize, source.Length - off);
            var chunk = new float[n];
            Array.Copy(source, off, chunk, 0, n);
            got.AddRange(streamer.Process(chunk));
        }
        got.AddRange(streamer.Flush());

        if (got.Count != expected.Length)
            throw new Exception($"streamed {got.Count} samples, whole-buffer produced {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
            if (got[i] != expected[i])
                throw new Exception($"sample {i} is {got[i]} streamed vs {expected[i]} whole-buffer");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The sparse mel filterbank must produce EXACTLY what the dense one produced.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ WHY THE OPTIMISATION IS SAFE, and why that still needs proving. A Slaney mel filter is a triangle:
    /// filter m touches a handful of the 201 frequency bins and is exactly 0 everywhere else. The dense loop
    /// walked all 201 for every one of 80 mels x 3000 frames - about 48 million multiply-adds, ~95% of them
    /// multiplying by zero. Restricting each mel to its non-zero span is bit-identical because
    /// <c>0 * power == 0</c> and <c>x + 0 == x</c> exactly in IEEE 754.
    /// </para>
    /// <para>
    /// ⚠️ EXACT equality, deliberately, and for the reason the streaming-resampler gate exists: a tolerance
    /// here would also accept a bound that clipped a genuinely non-zero edge coefficient, which is a real
    /// mistake this refactor could make (an off-by-one on <c>kHi</c>) and which would shift the spectrum
    /// slightly - the kind of error that degrades a transcript without ever failing anything.
    /// </para>
    /// <para>
    /// Driven with SPEECH-LIKE content rather than a pure tone: a single sine excites almost no mel bins, so
    /// a bound bug could sit in the untouched ones and never show.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public Task Mel_SparseFilterbank_MatchesDenseExactly()
    {
        // ── 1. THE INVARIANT, at the REAL production geometry (80 mels, 201 bins). ──
        // This is what actually makes skipping safe, and asserting it directly is both cheaper and
        // STRONGER than comparing one fixture's values: it holds for every possible input, not just this
        // one. The realistic bug - an off-by-one on the upper bound clipping a genuinely non-zero edge
        // coefficient - is caught here exactly.
        const int nMels = 80, freqBins = 201;
        var filters = AudioPreprocessor.GenerateMelFilterbankSlaney(nMels, freqBins, 16000);
        int widest = 0;
        for (int m = 0; m < nMels; m++)
        {
            int lo = freqBins, hi = -1;
            for (int k = 0; k < freqBins; k++)
                if (filters[m, k] != 0f) { if (k < lo) lo = k; hi = k; }

            for (int k = 0; k < freqBins; k++)
            {
                bool inside = k >= lo && k <= hi;
                if (!inside && filters[m, k] != 0f)
                    throw new Exception($"mel {m}: filters[{m},{k}] = {filters[m, k]} lies OUTSIDE the "
                                      + $"non-zero span [{lo},{hi}] the production loop derives, so the "
                                      + "sparse bounds would drop a real coefficient");
            }
            if (hi >= lo) widest = Math.Max(widest, hi - lo + 1);
        }

        // The optimisation is only worth its complexity if the filters really are narrow. If a future
        // filterbank change made them dense, this should stop claiming a win.
        if (widest >= freqBins)
            throw new Exception($"the widest mel filter spans all {freqBins} bins - the sparse path saves "
                              + "nothing and the dense loop should come back");

        // ── 2. END-TO-END value equality, at a REDUCED mel count. ──
        // ⚠️ nMels is reduced ONLY to keep the runtime sane, and the reason matters: the dense oracle is
        // 80 x 3000 x 201 = ~48 MILLION multiply-adds on the single WASM thread - the very work this
        // change removes - and the test body is synchronous, so it runs inside PMT's Run-button click.
        // MEASURED: at full size it blew Playwright's 30 s click timeout in a full sweep (it passed when
        // run scoped, which is exactly how a too-slow test hides). Frame count is fixed at 3000 by
        // Whisper's padding and cannot be reduced, so mels are the only lever.
        // Step 1 above is what covers the production geometry; this covers the plumbing.
        const int rate = 16000;
        var n = (int)(rate * 1.2);
        var audio = new float[n];
        var rng = new Random(4242);
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)rate;
            double v = 0;
            for (int h = 1; h <= 12; h++) v += Math.Sin(2 * Math.PI * 110 * h * t) / h;   // broadband
            audio[i] = (float)(0.3 * v + 0.02 * (rng.NextDouble() - 0.5));
        }

        const int smallMels = 8;
        var got = AudioPreprocessor.ComputeLogMelSpectrogram(audio, smallMels);
        var expected = DenseReferenceLogMel(audio, smallMels);

        if (got.Length != expected.Length)
            throw new Exception($"sparse produced {got.Length} values, dense reference {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
            if (got[i] != expected[i])
                throw new Exception($"mel[{i}] is {got[i]} sparse vs {expected[i]} dense - the non-zero span "
                                  + "for some mel filter is wrong, which shifts the spectrum without "
                                  + "failing anything downstream");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The DENSE filterbank, kept here as the oracle the optimised path is measured against.
    /// </summary>
    /// <remarks>
    /// ⚠️ Deliberately a duplicate of the pre-optimisation loop rather than a call into the library. An
    /// oracle that shares the code under test cannot detect a change in that code - the whole point is that
    /// this walks all 201 bins unconditionally, forever, however the production path evolves.
    /// </remarks>
    private static float[] DenseReferenceLogMel(float[] samples, int nMels = 80, int fftSize = 400,
        int hopSize = 160)
    {
        samples = AudioPreprocessor.PadOrTrim(samples, AudioPreprocessor.WhisperMaxSamples);
        var stft = AudioPreprocessor.ComputeSTFT(samples, fftSize, hopSize, center: true);
        int numFrames = Math.Max(0, stft.GetLength(0) - 1);
        int freqBins = stft.GetLength(1);

        var power = new float[numFrames, freqBins];
        for (int f = 0; f < numFrames; f++)
            for (int k = 0; k < freqBins; k++)
                power[f, k] = stft[f, k] * stft[f, k];

        var melFilters = AudioPreprocessor.GenerateMelFilterbankSlaney(nMels, freqBins, 16000);

        var melSpec = new float[nMels * numFrames];
        for (int m = 0; m < nMels; m++)
            for (int f = 0; f < numFrames; f++)
            {
                float sum = 0;
                for (int k = 0; k < freqBins; k++) sum += melFilters[m, k] * power[f, k];
                melSpec[m * numFrames + f] = MathF.Log10(MathF.Max(sum, 1e-10f));
            }

        float maxVal = float.MinValue;
        for (int i = 0; i < melSpec.Length; i++) if (melSpec[i] > maxVal) maxVal = melSpec[i];
        for (int i = 0; i < melSpec.Length; i++)
        {
            melSpec[i] = MathF.Max(melSpec[i], maxVal - 8.0f);
            melSpec[i] = (melSpec[i] + 4.0f) / 4.0f;
        }
        return melSpec;
    }

    /// <summary>Equal rates stream through untouched, and must not buffer or delay.</summary>
    [TestMethod(Timeout = 120000)]
    public Task Streaming_EqualRates_IsIdentity()
    {
        var source = Tone(16000, 440, 0.2);
        var streamer = new StreamingResampler(16000, 16000);
        var got = new System.Collections.Generic.List<float>();
        for (int off = 0; off < source.Length; off += 333)
        {
            int n = Math.Min(333, source.Length - off);
            var chunk = new float[n];
            Array.Copy(source, off, chunk, 0, n);
            got.AddRange(streamer.Process(chunk));
        }
        got.AddRange(streamer.Flush());

        if (got.Count != source.Length)
            throw new Exception($"identity stream changed length: {source.Length} -> {got.Count}");
        for (int i = 0; i < source.Length; i++)
            if (got[i] != source[i])
                throw new Exception($"sample {i} changed from {source[i]} to {got[i]}");
        return Task.CompletedTask;
    }

    /// <summary>
    /// The STFT's tabulated twiddle factors produce BIT-IDENTICAL output to computing the trig inline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ WHAT THIS GUARDS. The STFT runs one FFT per frame - 3,000 times for Whisper's fixed 30 s window -
    /// and used to call <c>MathF.Cos</c>/<c>MathF.Sin</c> inside the innermost loops, recomputing the same
    /// values every frame. At n=400 (= 2^4 x 25) that is ~21,600 transcendental calls per frame and roughly
    /// 65 MILLION per transcription on a single WASM thread. MEASURED: 3,301 ms of a 13,926 ms browser
    /// transcription. Tabulating them removes the calls entirely.
    /// </para>
    /// <para>
    /// ⚠️ EXACT equality, not a tolerance, and the reason is specific: tabulation is only legitimate if the
    /// stored float carries the SAME BITS the call site would have produced, which requires building the
    /// table with the same expression in the same association order. A tolerance would happily accept a
    /// table built from, say, <c>-2πk/n</c> reassociated - numerically "close", and a silent change to the
    /// spectrum the mel filterbank sees and therefore to what Whisper is handed.
    /// </para>
    /// <para>
    /// ⚠️ The oracle below is a DELIBERATE DUPLICATE of the pre-tabulation algorithm. It must not call back
    /// into <c>AudioPreprocessor</c>'s FFT, because an oracle that calls the code under test cannot detect a
    /// change in it - the same trap <c>Mel_SparseFilterbank_MatchesDenseExactly</c> avoids.
    /// </para>
    /// <para>
    /// Both transform lengths are covered because they take different code paths: 400 is even-split down to
    /// an odd base with a direct DFT, 512 is the iterative radix-2 path, and each has its own table.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 300000)]
    public Task Stft_TwiddleTables_MatchInlineTrigExactly()
    {
        // Broadband, speech-like content. A pure tone leaves most bins near zero, where a wrong twiddle
        // would not show up in the magnitude.
        const int rate = 16000;
        var n = (int)(rate * 1.2);
        var audio = new float[n];
        var rng = new Random(9182);
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)rate;
            double v = 0;
            for (int h = 1; h <= 12; h++) v += Math.Sin(2 * Math.PI * 110 * h * t) / h;
            audio[i] = (float)(0.3 * v + 0.05 * (rng.NextDouble() - 0.5));
        }

        // ⚠️ The signal is deliberately followed by SILENCE, because that is the shape Whisper always
        // produces - it pads every utterance to a flat 30 s - and the STFT skips frames whose window is
        // entirely zeros. The oracle below computes every frame regardless, so this fixture is what proves
        // the skipped rows equal the computed ones rather than merely being absent.
        var padded = new float[audio.Length * 3];
        Array.Copy(audio, padded, audio.Length);
        audio = padded;

        foreach (var (fftSize, hopSize) in new[] { (400, 160), (512, 160) })
        {
            var got = AudioPreprocessor.ComputeSTFT(audio, fftSize, hopSize, center: true);
            var expected = InlineTrigStft(audio, fftSize, hopSize);

            if (got.GetLength(0) != expected.GetLength(0) || got.GetLength(1) != expected.GetLength(1))
                throw new Exception(
                    $"n={fftSize}: tabulated STFT is [{got.GetLength(0)},{got.GetLength(1)}], "
                  + $"inline-trig reference is [{expected.GetLength(0)},{expected.GetLength(1)}]");

            for (int f = 0; f < got.GetLength(0); f++)
                for (int k = 0; k < got.GetLength(1); k++)
                    if (got[f, k] != expected[f, k])
                        throw new Exception(
                            $"n={fftSize}: frame {f} bin {k} is {got[f, k]} with tabulated twiddles and "
                          + $"{expected[f, k]} computing the trig inline. The tables must reproduce the "
                          + "call site's bits exactly, not merely approximate them.");
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// The STFT exactly as it was before twiddle tabulation: every sine and cosine computed inline.
    /// </summary>
    /// <remarks>
    /// Duplicated on purpose - see the remarks on <see cref="Stft_TwiddleTables_MatchInlineTrigExactly"/>.
    /// The window comes from the library because the window is not what is under test here.
    /// </remarks>
    private static float[,] InlineTrigStft(float[] samples, int fftSize, int hopSize)
    {
        int pad = fftSize / 2;
        var padded = new float[samples.Length + 2 * pad];
        Array.Copy(samples, 0, padded, pad, samples.Length);
        for (int i = 0; i < pad; i++)
        {
            padded[pad - 1 - i] = samples[Math.Min(i + 1, samples.Length - 1)];
            padded[pad + samples.Length + i] = samples[Math.Max(samples.Length - 2 - i, 0)];
        }
        samples = padded;

        var window = AudioPreprocessor.GenerateHannWindow(fftSize);
        int numFrames = (samples.Length - fftSize) / hopSize + 1;
        int freqBins = fftSize / 2 + 1;
        var stft = new float[numFrames, freqBins];

        var frame = new float[fftSize];
        var real = new float[fftSize];
        var imag = new float[fftSize];
        var scratchRe = new float[fftSize];
        var scratchIm = new float[fftSize];

        for (int f = 0; f < numFrames; f++)
        {
            int offset = f * hopSize;
            for (int i = 0; i < fftSize; i++)
            {
                int idx = offset + i;
                frame[i] = idx < samples.Length ? samples[idx] * window[i] : 0;
            }
            Array.Copy(frame, real, fftSize);
            Array.Clear(imag, 0, fftSize);
            RefFft(real, imag, fftSize, scratchRe, scratchIm);
            for (int k = 0; k < freqBins; k++)
                stft[f, k] = MathF.Sqrt(real[k] * real[k] + imag[k] * imag[k]);
        }
        return stft;
    }

    private static void RefFft(float[] real, float[] imag, int n, float[] scratchRe, float[] scratchIm)
    {
        if ((n & (n - 1)) != 0)
        {
            RefFftAny(real, imag, 0, 1, scratchRe, scratchIm, 0, n);
            Array.Copy(scratchRe, real, n);
            Array.Copy(scratchIm, imag, n);
            return;
        }

        int bits = (int)MathF.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = 0, v = i;
            for (int b = 0; b < bits; b++) { j = (j << 1) | (v & 1); v >>= 1; }
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        for (int size = 2; size <= n; size *= 2)
        {
            int halfSize = size / 2;
            float angle = -2f * MathF.PI / size;
            for (int i = 0; i < n; i += size)
                for (int j = 0; j < halfSize; j++)
                {
                    float cos = MathF.Cos(angle * j);
                    float sin = MathF.Sin(angle * j);
                    int even = i + j, odd = i + j + halfSize;
                    float tr = real[odd] * cos - imag[odd] * sin;
                    float ti = real[odd] * sin + imag[odd] * cos;
                    real[odd] = real[even] - tr;
                    imag[odd] = imag[even] - ti;
                    real[even] += tr;
                    imag[even] += ti;
                }
        }
    }

    private static void RefFftAny(float[] inRe, float[] inIm, int inOff, int stride,
                                  float[] outRe, float[] outIm, int outOff, int n)
    {
        if (n == 1) { outRe[outOff] = inRe[inOff]; outIm[outOff] = inIm[inOff]; return; }
        if ((n & 1) != 0) { RefDft(inRe, inIm, inOff, stride, outRe, outIm, outOff, n); return; }

        int half = n / 2;
        RefFftAny(inRe, inIm, inOff, stride * 2, outRe, outIm, outOff, half);
        RefFftAny(inRe, inIm, inOff + stride, stride * 2, outRe, outIm, outOff + half, half);

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

    private static void RefDft(float[] inRe, float[] inIm, int inOff, int stride,
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

}
