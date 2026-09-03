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
        //
        // ⚠️ SpeechPad is PINNED here rather than inherited, and that is the point of this test being a
        // comparison. The fixture's boundaries were produced by sherpa-onnx at silero's own 30 ms pad, so
        // every parameter that MOVES a boundary has to be fixed to the oracle's value or the comparison
        // stops being one. What this test is for is the state machine - hysteresis, min-silence, when a
        // turn opens and closes - not the padding, which is a deliberate product choice measured against
        // a recogniser instead (see VadOptions.SpeechPad, now 150 ms by default). Leaving it inherited
        // meant a justified change to that default would surface here as a phantom segmentation failure.
        using var detector = new VoiceActivityDetector(vad, new VadOptions
        {
            Threshold = 0.5f,
            MinSilenceDuration = TimeSpan.FromMilliseconds(500),
            MinSpeechDuration = TimeSpan.FromMilliseconds(250),
            MaxSpeechDuration = TimeSpan.FromSeconds(20),
            SpeechPad = TimeSpan.FromMilliseconds(30),
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

    /// <summary>
    /// A bigger <see cref="VadOptions.SpeechPad"/> must actually move the segment start earlier.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ THE DEFECT THIS EXISTS TO FIX. Silero opens a segment on the frame whose probability crosses
    /// <see cref="VadOptions.Threshold"/>, and that frame is not where the word began - a low-energy onset
    /// (an /h/, an unreleased plosive) does not cross until the vowel arrives. With the old 30 ms default
    /// the detector handed over utterances with their first phoneme already cut, and the hands-free demo
    /// transcribed "Hello. What is a chicken?" as <b>"Oh, what is it chicken?"</b>. MEASURED afterwards at
    /// <b>WER 0.123</b> on a Harvard recording, falling to 0.000 at 150 ms - the table is on
    /// <see cref="VadOptions.SpeechPad"/>, and the sweep is <c>tools/vad-onset-check</c>.
    /// </para>
    /// <para>
    /// ⚠️ A DIFFERENTIAL assertion, on purpose. Comparing the segment start against a known onset needs
    /// ground truth for where speech begins, and the first version of this test manufactured that with a
    /// synthetic tone - which Silero correctly refused to call speech, so it found no segment at all and
    /// asserted nothing on any backend. Real speech has no exact onset sample to assert against. Running
    /// the SAME audio at two pads removes the need for one: the difference between the two starts is the
    /// property under test, and it is exact.
    /// </para>
    /// <para>
    /// ⚠️ Also asserts that the DEFAULT is the padded one. Without that this passes just as happily on a
    /// library whose default has quietly gone back to 30 ms, which is exactly the regression it exists to
    /// stop - the WER sweep is what establishes that 150 ms is enough, and nothing else would notice it
    /// being given back.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 600000)]
    public async Task Vad_SpeechPad_MovesTheSegmentStartEarlier() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");
        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        // Real speech: Silero is trained on voices and will not trigger on a tone, however loud.
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var speech = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // Lead-in of room tone, so the detector has somewhere to reach BACK into. Without it a segment
        // would clamp at sample 0 and both pads would give the same answer for the wrong reason.
        const int rate = SileroVad.SampleRate;
        var rng = new Random(20260902);
        var audio = new float[rate + speech.Length];
        for (int i = 0; i < rate; i++) audio[i] = (float)(rng.NextDouble() * 2 - 1) * 0.0005f;
        Array.Copy(speech, 0, audio, rate, speech.Length);

        // ⚠️ ONE SileroVad for all three passes, reset between them, rather than one per pass. Three
        // sessions of a 643 KB model is not much on a desktop and IS too much at the tail of the Wasm
        // lane: this test passed on Wasm in isolation and failed in the full sweep, where ~819 tests of
        // retained Contexts and Accelerators have already strained the managed heap
        // (see the retention note in the ML repo). Reusing the session also exercises the Reset() the
        // sibling tests gate, so the cheaper shape is the better one.
        using var vad = SileroVad.Create(accelerator, modelBytes);

        async Task<long> FirstStartAsync(VadOptions options)
        {
            vad.Reset();
            using var detector = new VoiceActivityDetector(vad, options);
            long first = -1;
            detector.OnSegment += seg => { if (first < 0) first = seg.StartSample; };

            // ⚠️ STOP AT THE FIRST SEGMENT. Every assertion below reads `first` and nothing else, so once
            // it is set the remaining audio is measured waste - and this runs THREE times, on every
            // backend. That is not free where a VAD frame is expensive: MEASURED at 3,109 ms per frame on
            // the Wasm lane, where feeding the whole clip three times is ~468 frames and blows through a
            // 600 s timeout. Chunks are whole 512-sample windows so the framing, and therefore the
            // reported start sample, is identical to feeding the buffer in one call.
            const int chunkFrames = 16;
            int chunk = SileroVad.WindowSize * chunkFrames;
            for (int off = 0; off < audio.Length && first < 0; off += chunk)
            {
                int take = Math.Min(chunk, audio.Length - off);
                await detector.AcceptWaveformAsync(audio.AsSpan(off, take).ToArray());
            }

            // Only needed if no segment closed on its own: Flush emits speech still in progress. Skipped
            // once `first` is set, because flushing after an early exit would report a second span that
            // this test has no use for.
            if (first < 0) await detector.FlushAsync();
            return first;
        }

        long tight = await FirstStartAsync(new VadOptions { SpeechPad = TimeSpan.FromMilliseconds(30) });
        long padded = await FirstStartAsync(new VadOptions { SpeechPad = TimeSpan.FromMilliseconds(150) });
        long shipped = await FirstStartAsync(new VadOptions());          // the DEFAULT is on trial too

        if (tight < 0 || padded < 0)
            throw new Exception("no segment found on real speech - nothing about reach-back can be "
                              + "concluded from this run");

        double movedMs = (tight - padded) / (double)rate * 1000;
        // The pads differ by 120 ms and the detector triggers on the same frame either way, so the starts
        // must differ by 120 ms. Allow one frame of slack rather than demanding the exact sample.
        const double expectedMs = 120, slackMs = 32;
        if (Math.Abs(movedMs - expectedMs) > slackMs)
            throw new Exception(
                $"raising SpeechPad from 30 ms to 150 ms moved the segment start by {movedMs:F0} ms, "
              + $"expected {expectedMs:F0}. A segment that starts at the trigger starts AFTER the word did "
              + "and hands the recogniser an utterance missing its first phoneme - MEASURED at WER 0.123 "
              + "with a 30 ms pad, which is where \"Hello\" became \"Oh\".");

        if (shipped != padded)
            throw new Exception(
                $"the DEFAULT VadOptions started the segment at {shipped} but a 150 ms pad starts it at "
              + $"{padded}. The shipped default is 150 ms because that is the smallest pad measured to "
              + "cost a recogniser nothing (WER 0.000 against 0.123 at 30 ms); a default that has drifted "
              + "back down gives that away with nothing else to notice it.");

        Console.WriteLine($"[Vad] pad: 30 ms starts at {tight / (double)rate:F3}s, 150 ms at "
                        + $"{padded / (double)rate:F3}s - {movedMs:F0} ms earlier, default matches 150 ms");
    });
}
