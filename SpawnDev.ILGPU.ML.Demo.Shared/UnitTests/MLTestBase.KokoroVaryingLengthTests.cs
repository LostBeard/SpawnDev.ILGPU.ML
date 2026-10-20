using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Speaking two DIFFERENT-LENGTH utterances from one pipeline - which is what a conversation is.
/// </summary>
/// <remarks>
/// 🔴 THE GAP THIS CLOSES, AND IT IS MINE. Pipeline_Kokoro_MatchesOnnxRuntimeWaveform speaks the SAME 35
/// tokens every time. Every timing and every correctness result for graph capture came from that fixture, so
/// "capture works" meant "capture works when the utterance never changes". TJ asked the obvious question -
/// *"are you saying it is only good for repeating itself? that does not sound useful at all"* - and the
/// answer was worse than slow.
///
/// <see cref="Graph.SessionGraphCapture"/> holds ONE plan for ONE shape set and says so:
/// <code>
///   throw new InvalidOperationException(
///       "SessionGraphCapture: input shapes changed after capture - use one instance per fixed shape set.");
/// </code>
/// Kokoro's input is <c>input_ids[1, tokenCount]</c> and <see cref="KokoroPipeline"/> holds a single
/// <c>_capture</c>, so a second utterance of a different length does not degrade - it THROWS. A test that
/// only ever repeats one sentence cannot see that, and no other test speaks twice at two lengths.
///
/// ⚠️ This is the shape-of-fixture trap again: the fixture was chosen for convenience (one known reference
/// waveform) and it happened to hold the one variable that mattered constant. Same family as
/// [[fb-choose-fixture-violate]] and the spatial=49 GlobalAvgPool test.
///
/// The assertion is deliberately NOT about speed. It is that a pipeline can say two different things.
/// </remarks>
public abstract partial class MLTestBase
{
    // Two real token sequences of DIFFERENT length. The first is the reference line (35 ids); the second is
    // a genuinely shorter utterance, so the input shape changes between calls the way it does in a
    // conversation.
    private static readonly long[] KokoroShorterTokens =
    {
        0, 50, 83, 54, 156, 76, 16, 102, 68, 16, 156, 72, 61, 4, 0,
    };

    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_Kokoro_SpeaksTwoDifferentLengths() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }

        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        using var pipeline = KokoroPipeline.Create(accelerator, modelBytes);

        // First utterance, 35 tokens. On WebGPU this is where a plan gets recorded.
        // ⚠️ TIMED, because the third pass below is affordable only where a pass is cheap. The CPU backend
        // runs ~312 s per pass, so three passes blew PMT's 600 s outer cap and reported a FAILURE for a
        // backend that was verifying the behaviour perfectly well - the same trap
        // Pipeline_Kokoro_MatchesOnnxRuntimeWaveform already guards with its warm budget.
        var firstClock = Stopwatch.StartNew();
        var first = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
        firstClock.Stop();
        if (first.Samples.Length == 0)
            throw new Exception($"{BackendName}: the first utterance produced no samples");

        // Second utterance, 15 tokens - a DIFFERENT shape. This is the line that used to throw.
        KokoroAudio second;
        try
        {
            second = await pipeline.SpeakTokensAsync(KokoroShorterTokens, pack);
        }
        catch (Exception ex)
        {
            throw new Exception(
                $"{BackendName}: speaking a second utterance of a different length failed - {ex.Message}. "
              + "A pipeline that can only repeat one sentence is not a text-to-speech pipeline. "
              + $"capture status: {pipeline.CaptureStatus}", ex);
        }

        if (second.Samples.Length == 0)
            throw new Exception($"{BackendName}: the second utterance produced no samples");

        // Different token counts must give different durations - otherwise the second call returned the
        // FIRST utterance's audio, which a replayed stale plan would do silently and which a
        // "did it throw" check alone would call a pass.
        if (second.Samples.Length == first.Samples.Length)
            throw new Exception(
                $"{BackendName}: both utterances produced {first.Samples.Length} samples despite having "
              + $"{KokoroReferenceTokens.Length} and {KokoroShorterTokens.Length} tokens. The second call "
              + "returned the first one's audio - a replayed plan that was never re-recorded for the new "
              + "shape.");

        // And the second must be real audio, not silence from a plan bound to the wrong buffers.
        var peak = 0f;
        foreach (var s in second.Samples) { var a = MathF.Abs(s); if (a > peak) peak = a; }
        if (peak < 0.05f)
            throw new Exception(
                $"{BackendName}: the second utterance peaked at {peak:F4} - it ran and produced silence, "
              + "which is what a replay bound to the previous shape's buffers looks like.");

        // Back to the FIRST length again: a real conversation revisits lengths, and a per-shape cache that
        // evicts or corrupts on return would show up here rather than on the first pair.
        //
        // Only where a pass is affordable. A backend at ~312 s/pass would spend a third of an hour proving
        // a property that is about shape bookkeeping, not about that backend - and would be KILLED by the
        // runner's outer cap, reporting a failure it did not have. Its absence is stated, not papered over.
        const int alternateBudgetMs = 60_000;
        if (firstClock.ElapsedMilliseconds <= alternateBudgetMs)
        {
            var third = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
            if (third.Samples.Length != first.Samples.Length)
                throw new Exception(
                    $"{BackendName}: returning to the first length gave {third.Samples.Length} samples, not "
                  + $"{first.Samples.Length}. Shape handling is not stable across alternation.");

            Console.WriteLine($"[KokoroLen] {BackendName} ok: {KokoroReferenceTokens.Length} tok -> "
                            + $"{first.Samples.Length} samples, {KokoroShorterTokens.Length} tok -> "
                            + $"{second.Samples.Length} samples, back to {KokoroReferenceTokens.Length} tok -> "
                            + $"{third.Samples.Length} samples | capture: {pipeline.CaptureStatus}");
        }
        else
        {
            Console.WriteLine($"[KokoroLen] {BackendName} ok: {KokoroReferenceTokens.Length} tok -> "
                            + $"{first.Samples.Length} samples, {KokoroShorterTokens.Length} tok -> "
                            + $"{second.Samples.Length} samples | ALTERNATION PASS SKIPPED: a pass costs "
                            + $"{firstClock.ElapsedMilliseconds} ms here, over the {alternateBudgetMs} ms budget "
                            + $"| capture: {pipeline.CaptureStatus}");
        }
    });

    /// <summary>
    /// Two DIFFERENT sentences with the SAME token count, one after the other on one pipeline, each identical to the
    /// same sentence on a fresh pipeline.
    /// </summary>
    /// <remarks>
    /// Kokoro's audio length is not a function of its input SHAPE: the vocoder runs at a length set by the predicted
    /// durations, i.e. by the input DATA. Two sentences of the same token count share one compiled executor, and the
    /// executor's fold (<see cref="Graph.GraphExecutor.FoldInputIndependentNodes"/>) treated every Shape/Size as a
    /// function of the input shapes - so the second sentence reused the first one's lengths (87,000 samples where it
    /// produced 87,600) for the Reshape in front of the vocoder's Pad, and that Pad ran on a rank-2 tensor with rank-3
    /// pads. MEASURED 2026-10-12 (Vertex): an illegal address under memory pressure; with the freed memory still mapped,
    /// a quiet write past the buffer. "The number 1..." and "The number 2..." are both 60 tokens.
    /// </remarks>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_Kokoro_SameTokenCount_DifferentSentence_MatchesAFreshPipeline() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroModelsBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }
        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        var phonemizer = SpawnDev.Phonemizer.EmbeddedData.CreatePhonemizer();
        var first = phonemizer.ToSymbols("The number 1 is a placeholder used to show nothing.");
        var second = phonemizer.ToSymbols("The number 2 is a placeholder used to show nothing.");

        float[] alone;
        using (var fresh = KokoroPipeline.Create(accelerator, modelBytes))
            alone = (await fresh.SpeakAsync(second, pack)).Samples;

        using var pipeline = KokoroPipeline.Create(accelerator, modelBytes);
        var a = await pipeline.SpeakAsync(first, pack);
        var b = await pipeline.SpeakAsync(second, pack);
        if (a.Tokens != b.Tokens)
            throw new Exception($"{BackendName}: the fixture is wrong - {a.Tokens} and {b.Tokens} tokens; the two sentences must "
                + "share a token count (and so an executor) for this to test anything");
        if (a.Samples.Length == b.Samples.Length)
            throw new Exception($"{BackendName}: the fixture is wrong - both sentences are {a.Samples.Length} samples; their "
                + "durations must differ for this to test anything");
        if (b.Samples.Length != alone.Length)
            throw new Exception($"{BackendName}: the second sentence came out {b.Samples.Length} samples after the first, "
                + $"{alone.Length} on a fresh pipeline - it reused the first sentence's data-dependent lengths");
        float maxDiff = 0f;
        for (int i = 0; i < alone.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(b.Samples[i] - alone[i]));
        if (maxDiff > 1e-3f)
            throw new Exception($"{BackendName}: the second sentence differs from a fresh pipeline's by up to {maxDiff:G4}");
    });
}
