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
/// It was invisible because every existing audio test fed audio that was ALREADY 16 kHz, where
/// <c>srcRate == dstRate</c> returns early and no resampling runs. The file path was correct and the
/// microphone path was garbage, from a line no test exercised. THE RATE CONVERSION ITSELF NEEDED A TEST.
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
    /// execute - the same way every audio fixture being 16 kHz meant the resampling body itself never ran.
    /// The invariant asserted is the physical one, not "both branches agree": 10 kHz cannot exist below an
    /// 8 kHz Nyquist either way.
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

}
