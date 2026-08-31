using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for <see cref="VoiceActivityDetector"/> - where an utterance starts and stops.
///
/// <para>
/// The probabilities are gated separately (<c>MLTestBase.SileroVadTests</c>, against onnxruntime). This
/// file gates the ENDPOINTING built on top of them, which is a different thing entirely: the same
/// probabilities with the wrong hysteresis, or the wrong silence bound, give a detector that either cuts
/// people off mid sentence or never stops listening.
/// </para>
///
/// <para>
/// The reference comes from sherpa-onnx via <c>tools/vad-oracle</c> - a separate C++ implementation of the
/// same endpointing, and the one <c>RoseEars</c> already runs on the robot. That independence is the point:
/// this code is a PORT, so the claim under test is "it behaves the same", and a reference transcribed by
/// hand from the same upstream source would only prove I read that source the same way twice.
/// </para>
///
/// <para>
/// ⚠️ The fixture is three separate utterances with 1.2 s gaps (<c>tools/gen_vad_segment_fixture.py</c>),
/// not the plain librivox clip. That clip is 4 s of near-continuous speech and sherpa finds ONE segment
/// covering nearly all of it - a detector that simply declared everything to be speech would match it
/// perfectly. Requiring three segments is what makes segmentation testable. The gaps are low-level room
/// tone rather than digital silence, because a microphone never produces zeros.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// How far a boundary may sit from the oracle's.
    /// </summary>
    /// <remarks>
    /// 150 ms is under a third of the 500 ms min-silence that closes a turn, so a difference this small
    /// cannot change WHICH turns are found - it is trigger latency and padding detail. Anything larger is
    /// a behavioural difference from the implementation on the robot and wants investigating rather than a
    /// wider tolerance.
    /// </remarks>
    private const double VadBoundaryToleranceSeconds = 0.150;

    [TestMethod(Timeout = 600000)]
    public async Task Vad_Endpointing_MatchesSherpaOnnxSegments() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await assets.GetStringAsync("references/vad/vad_three_utterances_segments.json");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var expected = root.GetProperty("segments").EnumerateArray()
            .Select(s => (Start: s.GetProperty("start_sample").GetInt64(),
                          End: s.GetProperty("end_sample").GetInt64()))
            .ToArray();

        if (expected.Length < 3)
            throw new Exception($"reference holds {expected.Length} segments - the fixture is supposed to "
                              + "contain three utterances, and fewer cannot test segmentation");

        var wavBytes = await assets.GetByteArrayAsync("references/vad/vad_three_utterances.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode references/vad/vad_three_utterances.wav");

        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        using var vad = SileroVad.Create(accelerator, modelBytes);
        // The same numbers RoseEars runs, and the same ones the oracle was given.
        using var detector = new VoiceActivityDetector(vad, new VadOptions
        {
            Threshold = 0.5f,
            MinSilenceDuration = TimeSpan.FromMilliseconds(500),
            MinSpeechDuration = TimeSpan.FromMilliseconds(250),
            MaxSpeechDuration = TimeSpan.FromSeconds(20),
        });

        var got = new List<(long Start, long End)>();
        detector.OnSegment += seg => got.Add((seg.StartSample, seg.StartSample + seg.Samples.Length));

        // Fed in 320-sample chunks on purpose: that is what RTP delivers, it is not a multiple of the
        // model's 512, and reframing is where a stream detector usually goes wrong.
        const int rtpChunk = 320;
        for (int i = 0; i < samples.Length; i += rtpChunk)
            await detector.AcceptWaveformAsync(samples, i, Math.Min(rtpChunk, samples.Length - i));
        await detector.FlushAsync();

        if (got.Count != expected.Length)
            throw new Exception(
                $"found {got.Count} utterances, sherpa-onnx finds {expected.Length}. "
                + $"ours=[{string.Join(", ", got.Select(g => $"{g.Start / 16000.0:F2}-{g.End / 16000.0:F2}s"))}] "
                + $"sherpa=[{string.Join(", ", expected.Select(e => $"{e.Start / 16000.0:F2}-{e.End / 16000.0:F2}s"))}]. "
                + "One segment covering everything means the detector never closes a turn; many small ones "
                + "mean it closes on every dip (check the NEGATIVE threshold hysteresis).");

        double worst = 0; string worstWhere = "";
        for (int i = 0; i < expected.Length; i++)
        {
            double dStart = Math.Abs(got[i].Start - expected[i].Start) / 16000.0;
            double dEnd = Math.Abs(got[i].End - expected[i].End) / 16000.0;
            if (dStart > worst) { worst = dStart; worstWhere = $"segment {i} start"; }
            if (dEnd > worst) { worst = dEnd; worstWhere = $"segment {i} end"; }
        }

        if (worst > VadBoundaryToleranceSeconds)
            throw new Exception(
                $"{worstWhere} differs from sherpa-onnx by {worst * 1000:F0} ms "
                + $"(tolerance {VadBoundaryToleranceSeconds * 1000:F0} ms). "
                + $"ours=[{string.Join(", ", got.Select(g => $"{g.Start / 16000.0:F3}-{g.End / 16000.0:F3}"))}] "
                + $"sherpa=[{string.Join(", ", expected.Select(e => $"{e.Start / 16000.0:F3}-{e.End / 16000.0:F3}"))}]");

        Console.WriteLine($"[Vad] endpointing: {got.Count} utterances, worst boundary difference vs "
                        + $"sherpa-onnx {worst * 1000:F0} ms ({worstWhere})");
    });

    /// <summary>
    /// A turn that is still open when the audio ends must still be emitted.
    /// </summary>
    /// <remarks>
    /// The detector only closes a segment after it has seen enough trailing silence, so without an explicit
    /// flush the last thing anybody says before releasing the button - or before the session ends - is
    /// simply lost. Deliberately a separate test: the path above always ends in silence and would never
    /// exercise it.
    /// </remarks>
    [TestMethod(Timeout = 600000)]
    public async Task Vad_Flush_EmitsSpeechStillInProgress() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        using var vad = SileroVad.Create(accelerator, modelBytes);
        using var detector = new VoiceActivityDetector(vad, new VadOptions());

        var got = new List<SpeechSegment>();
        detector.OnSegment += got.Add;

        // The clip is near-continuous speech and ends mid-sentence, so nothing can close on its own.
        await detector.AcceptWaveformAsync(samples);

        if (got.Count != 0)
            throw new Exception($"{got.Count} segment(s) emitted before Flush - this clip does not contain "
                              + "500 ms of trailing silence, so the detector should still be holding it "
                              + "open. The fixture may have changed.");

        await detector.FlushAsync();

        if (got.Count != 1)
            throw new Exception($"Flush emitted {got.Count} segments, expected exactly 1. Speech that is "
                              + "still open when the audio ends is dropped without this path.");

        double seconds = got[0].DurationSeconds;
        // The clip is 4.0 s and almost all of it is speech; a segment far short of that means the turn was
        // truncated rather than flushed whole.
        if (seconds < 3.0)
            throw new Exception($"flushed segment is {seconds:F2}s of a 4.0s clip - the tail was lost");

        Console.WriteLine($"[Vad] flush: held a turn open through {samples.Length / 16000.0:F2}s of speech, "
                        + $"emitted {seconds:F2}s on flush");
    });
}
