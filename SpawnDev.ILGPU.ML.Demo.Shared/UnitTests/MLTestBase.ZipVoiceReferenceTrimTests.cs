using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Dead air in a ZipVoice reference clip is a SPEAKING-RATE error, and this is the gate on removing it.
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ WHAT SHIPPED BROKEN, and why no existing test could see it. The encoder derives frames-per-token
/// from the reference clip and multiplies it by the total token count - that ratio is the entire duration
/// prediction (traced through <c>text_encoder.onnx</c>, and written out on
/// <see cref="ZipVoiceFeatures.TrimReferenceSilence"/>). Silence in the reference has mel frames like
/// anything else, so a clip that is half silence declares a speaking rate half of the speaker's real one
/// and every generated syllable stretches to match. MEASURED at <b>1.94x too slow</b> with 4 s of silence
/// added to a 4 s reference. The hands-free demo hit exactly this and produced something the Captain
/// described as "out of a sci-fi movie" rather than a voice.
/// </para>
/// <para>
/// Nothing in the pipeline can notice: the shapes are right, the audio is speech, the transcript matches.
/// The end-to-end ZipVoice test passed throughout, because it feeds a clip that is already tightly
/// trimmed - so the defect only ever appeared on a REAL microphone.
/// </para>
/// <para>
/// ⚠️ The load-bearing assertion here is <see cref="ZipVoice_ReferenceTrim_IsImmuneToSurroundingSilence"/>:
/// trimming a clip and trimming the SAME clip surrounded by silence must give a byte-identical result.
/// That property IS the fix - it is what makes the predicted duration independent of how much dead air the
/// microphone happened to catch. Asserting only "it got shorter" would pass on a trim that removed some
/// arbitrary amount, which is the thing that cannot be allowed to vary.
/// </para>
/// <para>
/// No model, no download, no accelerator work - synthetic signals only, so this runs in milliseconds on
/// every backend. The frame arithmetic it protects is verified against the real encoder by
/// <c>tools/zipvoice-ref-rate</c>.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    private const int TrimRate = 16000;

    /// <summary>A tone at <paramref name="amplitude"/>, standing in for voiced speech.</summary>
    private static float[] TrimTone(double seconds, float amplitude = 0.5f, double hz = 220)
    {
        var s = new float[(int)(seconds * TrimRate)];
        for (int i = 0; i < s.Length; i++)
            s[i] = amplitude * MathF.Sin((float)(2 * Math.PI * hz * i / TrimRate));
        return s;
    }

    /// <summary>
    /// Silence with a realistic noise floor. Digital zero is the easy case and not the one that ships:
    /// a microphone in a room hands over quiet noise, and a gate that only recognises exact zero would
    /// pass this test and do nothing at all in production.
    /// </summary>
    private static float[] TrimRoomTone(double seconds, float amplitude = 0.0005f, int seed = 12345)
    {
        var rng = new Random(seed);
        var s = new float[(int)(seconds * TrimRate)];
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        return s;
    }

    private static float[] TrimConcat(params float[][] parts)
    {
        var outp = new float[parts.Sum(p => p.Length)];
        int at = 0;
        foreach (var p in parts) { Array.Copy(p, 0, outp, at, p.Length); at += p.Length; }
        return outp;
    }

    private static double TrimSeconds(float[] s) => s.Length / (double)TrimRate;

    /// <summary>
    /// THE gate: the trimmed clip must not depend on how much silence surrounded it.
    /// </summary>
    /// <remarks>
    /// This is the property that makes the cloned speaking rate stable. Without the trim both sides are
    /// the untouched inputs, whose lengths differ by four seconds, and this fails on the first assertion.
    /// </remarks>
    [TestMethod]
    public async Task ZipVoice_ReferenceTrim_IsImmuneToSurroundingSilence() => await RunTest(_ =>
    {
        var speech = TrimConcat(TrimRoomTone(0.10), TrimTone(1.5), TrimRoomTone(0.10));
        var buried = TrimConcat(TrimRoomTone(2.0), speech, TrimRoomTone(2.0));

        var a = ZipVoiceFeatures.TrimReferenceSilence(speech, TrimRate);
        var b = ZipVoiceFeatures.TrimReferenceSilence(buried, TrimRate);

        if (a.Length != b.Length)
            throw new Exception(
                $"the same speech trimmed to {TrimSeconds(a):F2}s on its own and {TrimSeconds(b):F2}s when "
              + "surrounded by silence. The encoder derives frames-per-token from this length, so a "
              + "difference here IS a difference in the cloned speaking rate.");

        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                throw new Exception($"trimmed clips differ at sample {i}: {a[i]} vs {b[i]}");

        // And it did something: the surrounding four seconds are gone.
        if (TrimSeconds(b) > 2.0)
            throw new Exception(
                $"4.0s of silence went in around 1.7s of speech and {TrimSeconds(b):F2}s came out - the "
              + "gate is not removing dead air");
        return Task.CompletedTask;
    });

    /// <summary>Leading and trailing dead air goes, and the speech itself survives intact.</summary>
    [TestMethod]
    public async Task ZipVoice_ReferenceTrim_KeepsTheSpeechItTrimsAround() => await RunTest(_ =>
    {
        var tone = TrimTone(1.0);
        var clip = TrimConcat(TrimRoomTone(1.5), tone, TrimRoomTone(1.5));
        var trimmed = ZipVoiceFeatures.TrimReferenceSilence(clip, TrimRate);

        // The keep margin is 60 ms either side, so the result is the tone plus at most ~0.13 s.
        if (TrimSeconds(trimmed) < 1.0)
            throw new Exception(
                $"trimmed to {TrimSeconds(trimmed):F2}s but there is 1.00s of speech in the clip - the gate "
              + "is eating the voice, which clones worse than the silence did");
        if (TrimSeconds(trimmed) > 1.35)
            throw new Exception($"trimmed to {TrimSeconds(trimmed):F2}s, expected ~1.0-1.15s");

        // The loudest sample must survive: an amplitude drop means the gate cut into voiced audio.
        float peakIn = 0, peakOut = 0;
        foreach (var v in clip) peakIn = Math.Max(peakIn, Math.Abs(v));
        foreach (var v in trimmed) peakOut = Math.Max(peakOut, Math.Abs(v));
        if (peakOut < peakIn * 0.99f)
            throw new Exception($"peak fell from {peakIn:F4} to {peakOut:F4} - the trim cut into speech");
        return Task.CompletedTask;
    });

    /// <summary>
    /// An internal pause is CAPPED, not deleted: rhythm survives, dead air does not.
    /// </summary>
    /// <remarks>
    /// ⚠️ Both halves matter and they pull opposite ways. A two-second think between two words is dead air
    /// inflating the rate; a natural breath is rhythm the model is supposed to clone. Deleting every gap
    /// would clone a speaker who never pauses, which is a different wrong voice rather than a right one.
    /// </remarks>
    [TestMethod]
    public async Task ZipVoice_ReferenceTrim_CapsAnInternalPauseRatherThanDeletingIt() => await RunTest(_ =>
    {
        // "hello ... hello" - the shape that failed in the demo.
        var clip = TrimConcat(TrimTone(0.5), TrimRoomTone(2.0), TrimTone(0.5));
        var trimmed = ZipVoiceFeatures.TrimReferenceSilence(clip, TrimRate, maxPauseSeconds: 0.20);

        // 1.0s of speech + a capped pause + the keep margins either side of both tones.
        if (TrimSeconds(trimmed) > 1.7)
            throw new Exception(
                $"a 2.0s internal pause survived as {TrimSeconds(trimmed) - 1.0:F2}s - dead air inside the "
              + "reference inflates the cloned rate exactly as dead air around it does, and an end-trim "
              + "alone cannot reach it");

        // The pause is capped, NOT removed: the two tones must not be spliced together.
        if (TrimSeconds(trimmed) < 1.15)
            throw new Exception(
                $"trimmed to {TrimSeconds(trimmed):F2}s, which is under 1.0s of speech plus the 0.20s cap - "
              + "the pause was deleted rather than capped, and a cloned voice that never breathes is its "
              + "own defect");
        return Task.CompletedTask;
    });

    /// <summary>
    /// A clip the gate cannot make sense of comes back UNCHANGED rather than gutted.
    /// </summary>
    /// <remarks>
    /// The failure this forbids is the quiet one: an empty or near-empty reference does not throw, it
    /// clones badly. Handing back the caller's audio is the correct answer when the gate has nothing to
    /// say about it.
    /// </remarks>
    [TestMethod]
    public async Task ZipVoice_ReferenceTrim_HandsBackWhatItCannotImprove() => await RunTest(_ =>
    {
        var silence = new float[TrimRate];                       // digital zero: no peak to be relative to
        var same = ZipVoiceFeatures.TrimReferenceSilence(silence, TrimRate);
        if (!ReferenceEquals(same, silence))
            throw new Exception("digital silence must come back untouched, not emptied");

        var tiny = TrimTone(0.01);                               // too short to frame at all
        if (!ReferenceEquals(ZipVoiceFeatures.TrimReferenceSilence(tiny, TrimRate), tiny))
            throw new Exception("a clip shorter than the analysis window must come back untouched");

        // A clip that is ALL speech has nothing to remove, so it must survive essentially whole.
        var loud = TrimTone(2.0);
        var trimmedLoud = ZipVoiceFeatures.TrimReferenceSilence(loud, TrimRate);
        if (TrimSeconds(trimmedLoud) < 1.98)
            throw new Exception(
                $"an unbroken 2.00s of speech trimmed to {TrimSeconds(trimmedLoud):F2}s - the gate is "
              + "removing audio it was given no reason to remove");
        return Task.CompletedTask;
    });

    /// <summary>
    /// The gate is RELATIVE to the clip's own loudness, so a quiet recording is not mistaken for silence.
    /// </summary>
    /// <remarks>
    /// ⚠️ An absolute threshold is the obvious implementation and it is wrong in a way that only shows up
    /// on somebody else's microphone: a reference recorded 20 dB down is entirely below any fixed gate,
    /// and the trim would return an empty clip or refuse. Scaling the whole clip must not change what the
    /// gate decides.
    /// </remarks>
    [TestMethod]
    public async Task ZipVoice_ReferenceTrim_ScalesWithTheClipNotWithAFixedLevel() => await RunTest(_ =>
    {
        var clip = TrimConcat(TrimRoomTone(1.0), TrimTone(1.0), TrimRoomTone(1.0));
        var quiet = clip.Select(v => v * 0.05f).ToArray();       // the same recording, 26 dB down

        var loudTrim = ZipVoiceFeatures.TrimReferenceSilence(clip, TrimRate);
        var quietTrim = ZipVoiceFeatures.TrimReferenceSilence(quiet, TrimRate);

        if (loudTrim.Length != quietTrim.Length)
            throw new Exception(
                $"the same recording trimmed to {TrimSeconds(loudTrim):F2}s loud and "
              + $"{TrimSeconds(quietTrim):F2}s quiet - the gate is reading an absolute level, so it will "
              + "mis-trim on any microphone that is not the one it was tuned on");
        return Task.CompletedTask;
    });
}
