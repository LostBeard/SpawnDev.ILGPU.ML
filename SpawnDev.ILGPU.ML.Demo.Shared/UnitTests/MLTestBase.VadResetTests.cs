using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Linq;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Does <see cref="SileroVad.Reset"/> actually restore the stream - on the BROWSER backends?
/// </summary>
/// <remarks>
/// <para>
/// ⚠️ THE SYMPTOM THIS CHASES. Across a four-turn hands-free session the speech probability decayed
/// monotonically - 0.999, 0.978, 0.760, 0.504 - on input whose measured level did not change. Turns three
/// and four never sustained above the 0.5 threshold, so the turn never ended and the timer climbed. The
/// demo calls a reset between turns, so a reset that did not fully reset would produce exactly that.
/// </para>
/// <para>
/// ⚠️ WHY THIS IS A TEST AND NOT A TOOL RUN. <c>tools/vad-reset-check</c> already answers this on CUDA -
/// bit-identical with graph capture on AND off, max diff 0.000E+000 - so the leading theory (that
/// <c>Reset</c>'s host-side <c>CopyFromCPU</c> is invisible to a REPLAYED dispatch plan, since capture
/// records GPU dispatches only) is dead there. But CUDA is not where the symptom lives. The demo runs on
/// WebGPU, and a desktop tool cannot reach a browser backend at all: <b>the one lane that could confirm or
/// kill this was the one lane nothing was running it on.</b> That is the gap, and it is the same shape as
/// the WebGPU empty-binding defect that OpenCL tolerated silently - a permissive backend hides a
/// portability bug rather than proving correctness.
/// </para>
/// <para>
/// The check is deterministic and needs no microphone: run frames, <see cref="SileroVad.Reset"/>, run the
/// IDENTICAL frames again. The model is stateful through <c>h</c>/<c>c</c>, so a correct reset makes pass
/// two reproduce pass one exactly. Any drift is state that survived, and the FIRST differing frame says
/// whether it leaked immediately or accumulated.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task Vad_Reset_ReproducesTheStreamExactly() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // Enough frames to get past graph capture's warm/probe/record passes, so the REPLAYED plan is what
        // is under test on the backends where capture is live. Fewer would only ever exercise the walk.
        int frames = Math.Min(48, samples.Length / SileroVad.WindowSize);
        if (frames < 16)
            throw new Exception($"only {frames} frames of audio - too few to reach a replayed plan");

        using var vad = SileroVad.Create(accelerator, modelBytes);
        var frame = new float[SileroVad.WindowSize];

        async Task<float[]> RunPassAsync()
        {
            var p = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                Array.Copy(samples, f * SileroVad.WindowSize, frame, 0, frame.Length);
                p[f] = await vad.ProcessFrameAsync(frame);
            }
            return p;
        }

        var first = await RunPassAsync();
        vad.Reset();
        var second = await RunPassAsync();

        // A detector that answers the same thing to everything would pass an equality check trivially, so
        // establish that these probabilities actually MOVE before trusting that they match.
        float lo = first.Min(), hi = first.Max();
        if (hi - lo < 0.3f)
            throw new Exception(
                $"speech probability only spanned {lo:F3}..{hi:F3} over {frames} frames of real speech. "
              + "A detector this flat is not detecting anything, and comparing two flat passes proves "
              + "nothing about Reset.");

        double worst = 0; int worstAt = -1;
        for (int f = 0; f < frames; f++)
        {
            double d = Math.Abs(first[f] - second[f]);
            if (d > worst) { worst = d; worstAt = f; }
        }

        // Tight on purpose. The two passes run the same graph on the same inputs from the same state, so
        // the only source of difference is state that Reset failed to clear - there is no accumulation of
        // rounding to allow for.
        const double tolerance = 1e-4;
        if (worst > tolerance)
        {
            int firstDiff = -1;
            for (int f = 0; f < frames; f++)
                if (Math.Abs(first[f] - second[f]) > tolerance) { firstDiff = f; break; }
            throw new Exception(
                $"after Reset the same audio gave a different answer: max |pass1 - pass2| = {worst:E3} at "
              + $"frame {worstAt}, first differing frame {firstDiff}. Recurrent state survived the reset, "
              + "which is what makes speech probability decay turn after turn until the endpointer stops "
              + $"firing. pass1[0..4]={string.Join(",", first.Take(4).Select(x => x.ToString("F4")))} "
              + $"pass2[0..4]={string.Join(",", second.Take(4).Select(x => x.ToString("F4")))}");
        }

        Console.WriteLine($"[Vad] reset: {frames} frames, captured={vad.IsCaptured}, "
                        + $"p spans {lo:F3}..{hi:F3}, max |pass1 - pass2| = {worst:E3}");
    });

    /// <summary>
    /// A <see cref="VoiceActivityDetector"/> reset between turns finds the SAME utterance again.
    /// </summary>
    /// <remarks>
    /// One layer above the model: the demo resets the DETECTOR between turns, not the raw model, and the
    /// detector carries a sample clock, a retained buffer and the trigger state machine as well. Getting
    /// the model right and the state machine wrong looks identical from the outside - a second turn that
    /// never ends - so both are gated.
    /// </remarks>
    [TestMethod(Timeout = 600000)]
    public async Task Vad_DetectorReset_FindsTheSameUtteranceOnASecondTurn() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");
        var wavBytes = await assets.GetByteArrayAsync("references/vad/vad_three_utterances.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode references/vad/vad_three_utterances.wav");

        using var vad = SileroVad.Create(accelerator, modelBytes);
        using var detector = new VoiceActivityDetector(vad, new VadOptions());

        async Task<(long Start, long End)[]> TurnAsync()
        {
            var got = new System.Collections.Generic.List<(long, long)>();
            void OnSeg(SpeechSegment s) => got.Add((s.StartSample, s.StartSample + s.Samples.Length));
            detector.OnSegment += OnSeg;
            try
            {
                await detector.AcceptWaveformAsync(samples);
                await detector.FlushAsync();
            }
            finally { detector.OnSegment -= OnSeg; }
            return got.ToArray();
        }

        var turn1 = await TurnAsync();
        detector.Reset();
        var turn2 = await TurnAsync();

        if (turn1.Length == 0)
            throw new Exception("no utterances on the first turn - the fixture holds three");

        if (turn1.Length != turn2.Length)
            throw new Exception(
                $"turn 1 found {turn1.Length} utterances, turn 2 found {turn2.Length} on IDENTICAL audio "
              + "after Reset. A second turn that finds fewer is a microphone that appears to go deaf as a "
              + "conversation goes on, which is what was reported.");

        for (int i = 0; i < turn1.Length; i++)
        {
            // ⚠️ Reset puts the sample clock back to zero, so offsets are comparable between turns. If
            // they were not, the demo's spans would point past the end of a fresh recording.
            if (turn1[i] != turn2[i])
                throw new Exception(
                    $"utterance {i} moved between turns: {turn1[i].Start}-{turn1[i].End} then "
                  + $"{turn2[i].Start}-{turn2[i].End}. Reset must put the clock back to zero, because the "
                  + "spans it returns are offsets into a buffer that starts again.");
        }

        Console.WriteLine($"[Vad] detector reset: {turn1.Length} utterances, identical across two turns");
    });
}
