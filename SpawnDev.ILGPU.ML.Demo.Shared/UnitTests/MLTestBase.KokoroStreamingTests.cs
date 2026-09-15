using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Does a spoken reply play without a pause between chunks, and does all of it get read?
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THE QUESTION THE PER-CHUNK RTF CANNOT ANSWER. Every other Kokoro timing test reports RTF for ONE
/// utterance, and RTF is the wrong instrument for streaming speech. The demo chunks a reply per sentence
/// and plays each chunk as it lands, so what the user hears is a QUEUE: a chunk that renders slower than
/// realtime is fine if the chunks before it built up a lead, and a chunk that renders faster than
/// realtime is NOT fine if it is the first one and the buffer is empty behind it.
/// </para>
/// <para>
/// The host cost of a pass is roughly FIXED (~1,850 nodes of dispatch prep regardless of length), so a
/// short sentence has a much worse RTF than a long one for exactly the same work. "Sure." is the chunk
/// that stalls, not the long one. A per-chunk RTF gate would either fail on a sentence that is perfectly
/// fine in context, or pass a reply that audibly stutters.
/// </para>
/// <para>
/// ⭐ SO THIS MODELS THE QUEUE. Playback starts the moment chunk 0 is rendered and then runs in realtime.
/// Chunk i has to be finished before playback arrives at it:
/// <code>
///   renderDoneBy[i]  &lt;=  renderDoneBy[0] + (a[0] + ... + a[i-1])
/// </code>
/// A violation is an UNDERRUN - a real, audible gap - and it is what TJ asked to have verified:
/// "tts in the Ai demo needs to be realtime so that there are not pauses between streaming chunks and so
/// that the entire response is read."
/// </para>
/// <para>
/// ⚠️ THE TOKENS ARE TIMING FIXTURES, NOT SPEECH. The chunks below are built by slicing and reframing the
/// reference phoneme ids to hit a spread of realistic sentence LENGTHS. Kokoro's cost is a function of
/// token count and predicted duration, not of which phonemes; the audio these produce is not meant to be
/// intelligible and nothing here asserts that it is. Pipeline_Kokoro_MatchesOnnxRuntimeWaveform is what
/// gates correctness of the audio - this gates the schedule.
/// </para>
/// <para>
/// ⚠️ CAPTURE IS LEFT AT THE DEFAULT (on) ON PURPOSE. Capture is recur-only and every chunk in a reply is
/// a different length, so in a real conversation almost nothing is ever replayed. Forcing it off would
/// measure a path the demo does not take; forcing it on per-chunk would measure the recorder. Leave it
/// alone and the test measures what a user gets.
/// </para>
/// <para>
/// <b>HeavyModel</b>: <c>PMT_LANES=WebGPU PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Kokoro_StreamingReply
/// PMT_CONSOLE_LOG=KokoroStream dotnet test PlaywrightMultiTest/...</c>
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>
    /// The chunk sizes the SpawnDev.AI demo actually renders, in phoneme tokens.
    /// </summary>
    /// <remarks>
    /// 🔴 THESE ARE PRODUCTION'S NUMBERS, NOT A GUESS AT THEM. They are the chunk lengths the
    /// SpawnDev.AI demo's own splitter actually emitted for a ~900-character reply, MEASURED 2026-09-15:
    /// <b>230 / 311 / 242 / 138 characters</b>. Kokoro's reference line is 31 characters -> 35 phoneme
    /// tokens, i.e. ~1.13 tokens per character, which puts them at ~260 / ~351 / ~273 / ~156 tokens.
    /// <para>
    /// ⚠️ THIS FIXTURE HAS BEEN WRONG TWICE, IN THE SAME WAY, AND BOTH COST A RUN. First it modelled
    /// per-SENTENCE chunks (9-47 tokens), which the demo had not done since a merge step was added.
    /// Then it modelled 180 / 360 / 360, which was the demo's policy right up until a chunk FLOOR was
    /// added to fix the stall this test found. A gate that models a policy the product does not use is
    /// worse than no gate: it reports a stall nobody would hear and hides the one they would. When
    /// Home.SpeakChunkCharacters / SpeakChunkMinimumCharacters change, these change with them.
    /// </para>
    /// <para>
    /// ⚠️ This file cannot reference SpawnDev.AI (layering), which is why the numbers are copied rather
    /// than called. The demo-side gate that DOES call the real chunker is
    /// <c>AiVoiceStreamingTests.ChunkedReplyStreamsWithoutAPause</c>; this one exists to catch an engine
    /// regression without needing the whole demo.
    /// </para>
    /// <para>
    /// ⚠️ Kokoro's context is 510 tokens, so these fit; do not raise them past that without checking.
    /// </para>
    /// </remarks>
    private static readonly int[] KokoroReplyChunkTokenCounts = { 260, 351, 273, 156 };

    /// <summary>
    /// Builds a chunk of <paramref name="tokenCount"/> ids framed the way the front end frames a
    /// sentence: pad, phonemes, the sentence-final period, pad (see Kokoro_EncodingPadsBothEnds).
    /// The interior ids are drawn from the reference line so they are all in-vocabulary.
    /// </summary>
    private static long[] BuildKokoroChunk(int tokenCount, int offset)
    {
        if (tokenCount < 4) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        var interior = KokoroReferenceTokens.Where(t => t != 0 && t != 4).ToArray();
        var chunk = new long[tokenCount];
        chunk[0] = 0;
        for (int i = 1; i < tokenCount - 2; i++)
            chunk[i] = interior[(offset + i) % interior.Length];
        chunk[tokenCount - 2] = 4;   // '.'
        chunk[tokenCount - 1] = 0;
        return chunk;
    }

    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Kokoro_StreamingReply_PlaysWithoutAnUnderrun() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);

        // 🔴 WEBGL AND WASM ARE EXCLUDED FROM THIS ONE, AND THE REASON IS A REAL LIMITATION, NOT SPEED.
        // This test renders FOUR long utterances back to back (260/351/273/156 tokens); every other
        // Kokoro test uses a single 35-token line, so this is the first thing that ever asked WebGL to
        // sustain long-form synthesis. It runs out of memory doing it - MEASURED 2026-09-15, on the
        // FOURTH chunk, after three had succeeded:
        //
        //   [GE node-1690 sync] [WebGL] GL Worker error: Array buffer allocation failed
        //     at new Uint8Array (<anonymous>)
        //     at dispatchKernel (glWorker.js:641)
        //
        // i.e. the per-dispatch staging allocation, ~1,850 nodes per utterance. That it survives three
        // utterances and dies on a SHORTER fourth says the cost accumulates ACROSS calls rather than
        // peaking within one - which points at retention in the WebGL worker, and is worth chasing on
        // its own terms. It is NOT chased here: WebGL cannot render speech faster than speech anyway
        // (the assertion below already excludes it), so all this would buy is a crash instead of a
        // number nobody gates on.
        //
        // ⚠️ RECORDED, NOT BURIED: long-form multi-utterance TTS on the WebGL backend is a known
        // limitation as of 2026-09-15. A consumer doing it will hit this. The other four heavy Kokoro
        // tests still run on WebGL and still pass, so single-utterance correctness there stays covered.
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException(
                "WebGL/Wasm cannot sustain four consecutive long utterances - the WebGL worker's "
              + "per-dispatch staging allocation fails with 'Array buffer allocation failed' on the "
              + "fourth (MEASURED 2026-09-15). Realtime streaming is not targeted on these backends, so "
              + "this test would cost minutes to produce a crash in place of an ungated number.");

        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }

        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        using var pipeline = KokoroPipeline.Create(accelerator, modelBytes);
        // ⚠️ Capture is left at its default (on) and that is MEASURED to be inert here, not assumed:
        // capture is recur-only and every chunk in a reply is a new length, so CaptureFor returns null
        // every time and the graph runs direct. A/B on 2026-09-15: capture on 23,242 ms vs off
        // 23,223 ms over the same reply. (An earlier single capture-on run read 30,560 ms and looked
        // like a 31% capture tax - it was the first-ever run of this test, paying one-time shader
        // compilation for shapes nothing had ever used. One repeat killed the theory.)

        // One throwaway utterance so the measured reply does not also pay one-time session setup
        // (kernel compilation, pool growth, weight upload) that a real conversation pays once at load,
        // long before the user speaks. Its length is deliberately NOT one of the reply's lengths.
        await pipeline.SpeakTokensAsync(BuildKokoroChunk(21, 3), pack);

        var renderMs = new double[KokoroReplyChunkTokenCounts.Length];
        var audioSec = new double[KokoroReplyChunkTokenCounts.Length];

        // Per-chunk attribution. Every chunk is a NEW shape, which is the whole point of the test, so
        // the question is which phase a new shape costs in - shader resolve (new WGSL per size),
        // dispatch prep, readbacks, drains, or device allocation. Guessing between them by inspection
        // has already been wrong once here.
        SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = true;

        for (int i = 0; i < KokoroReplyChunkTokenCounts.Length; i++)
        {
            var tokens = BuildKokoroChunk(KokoroReplyChunkTokenCounts[i], i * 7);

            var exec0 = Graph.GraphExecutor.CumulativeTotalMs;
            var rbMs0 = Graph.GraphExecutor.CumulativeReadbackMs;
            var rbN0 = Graph.GraphExecutor.CumulativeReadbackCount;
            var drMs0 = Graph.GraphExecutor.CumulativeSyncDrainMs;
            var drN0 = Graph.GraphExecutor.CumulativeSyncDrainCount;
            var alloc0 = Tensors.BufferPool.TotalDeviceAllocations;
            var allocMs0 = Tensors.BufferPool.TotalDeviceAllocationMs;
            var sh0 = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuShaderResolveMs;
            var arg0 = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuArgBuildMs;
            var bind0 = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupMs;
            var bindCreate0 = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupCreateMs;
            var enc0 = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuEncodeMs;

            Console.WriteLine($"[KokoroStream] {BackendName}: chunk {i} starting, {tokens.Length} tok, "
                            + $"capture status before: {pipeline.CaptureStatus}");

            var sw = Stopwatch.StartNew();
            KokoroAudio audio;
            try
            {
                audio = await pipeline.SpeakTokensAsync(tokens, pack);
            }
            catch (Exception ex)
            {
                // Name the chunk and whether capture was replaying. "device has been lost" with no
                // context reads as a card problem; "chunk 2, 360 tok, the SECOND time this length was
                // asked for, so capture was replaying" is the diagnosis.
                throw new Exception(
                    $"{BackendName}: chunk {i} ({tokens.Length} tok) FAILED after "
                  + $"{sw.Elapsed.TotalMilliseconds:F0} ms - capture status {pipeline.CaptureStatus}. "
                  + $"Lengths asked for so far: [{string.Join(", ", KokoroReplyChunkTokenCounts.Take(i + 1))}] "
                  + $"- a repeated length is the one that engages graph capture replay. {ex.Message}", ex);
            }
            sw.Stop();
            renderMs[i] = sw.Elapsed.TotalMilliseconds;
            audioSec[i] = audio.Seconds;

            var execMs = Graph.GraphExecutor.CumulativeTotalMs - exec0;
            var rbMs = Graph.GraphExecutor.CumulativeReadbackMs - rbMs0;
            var drMs = Graph.GraphExecutor.CumulativeSyncDrainMs - drMs0;
            Console.WriteLine($"[KokoroStream] {BackendName}: chunk {i} split: executor {execMs:F0} ms | "
                + $"readbacks {Graph.GraphExecutor.CumulativeReadbackCount - rbN0} ({rbMs:F0} ms) | "
                + $"drains {Graph.GraphExecutor.CumulativeSyncDrainCount - drN0} ({drMs:F0} ms) | "
                + $"residual {execMs - rbMs - drMs:F0} ms | outside executor {renderMs[i] - execMs:F0} ms | "
                + $"shader-resolve {SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuShaderResolveMs - sh0:F0} ms, "
                + $"arg-build {SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuArgBuildMs - arg0:F0} ms, "
                + $"bind-group {SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupMs - bind0:F0} ms "
                + $"(of which CreateBindGroup {SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupCreateMs - bindCreate0:F0} ms), "
                + $"encode {SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuEncodeMs - enc0:F0} ms | "
                + $"device allocs {Tensors.BufferPool.TotalDeviceAllocations - alloc0} "
                + $"({Tensors.BufferPool.TotalDeviceAllocationMs - allocMs0:F0} ms) | "
                // Kokoro sets the node cadence above its own node count, so EVERY drain here is the BYTE
                // CAP - which makes the drain count a statement about intermediate memory churn, not about
                // the cadence. These two numbers are what tell those apart.
                + $"deferred {Graph.GraphExecutor.LastRunDeferredReleaseBytes / 1048576.0:F0} MiB, "
                + $"peak backlog {Graph.GraphExecutor.LastRunPeakPendingReleaseBytes / 1048576.0:F0} MiB");

            if (audio.Samples.Length == 0)
                throw new Exception(
                    $"{BackendName}: chunk {i} ({KokoroReplyChunkTokenCounts[i]} tokens) produced NO audio, so the "
                  + "reply would be silently truncated - 'the entire response is read' fails here regardless of timing.");
        }

        // ── The queue ────────────────────────────────────────────────────────────────────────────
        // renderDoneBy[i] = wall-clock ms at which chunk i is ready.
        // playbackReaches[i] = wall-clock ms at which the speaker needs chunk i, which is when chunk 0
        //                     became ready plus the realtime duration of everything before i.
        double firstReadyMs = renderMs[0];
        double cumRenderMs = 0, cumAudioMsBefore = 0, totalAudioSec = 0;
        double worstMarginMs = double.MaxValue;
        int worstChunk = -1;
        var underruns = new List<string>();

        for (int i = 0; i < renderMs.Length; i++)
        {
            cumRenderMs += renderMs[i];
            double playbackReachesMs = firstReadyMs + cumAudioMsBefore;
            double marginMs = playbackReachesMs - cumRenderMs;   // negative = the speaker waited

            Console.WriteLine($"[KokoroStream] {BackendName}: chunk {i} "
                            + $"{KokoroReplyChunkTokenCounts[i],3} tok -> {audioSec[i]:F2}s audio in "
                            + $"{renderMs[i]:F0} ms (RTF {renderMs[i] / 1000.0 / Math.Max(audioSec[i], 1e-6):F2}x) | "
                            + $"ready at {cumRenderMs:F0} ms, needed at {playbackReachesMs:F0} ms, "
                            + $"margin {marginMs:F0} ms");

            if (marginMs < worstMarginMs) { worstMarginMs = marginMs; worstChunk = i; }
            if (marginMs < 0)
                underruns.Add($"chunk {i} ({KokoroReplyChunkTokenCounts[i]} tok) was {-marginMs:F0} ms late");

            cumAudioMsBefore += audioSec[i] * 1000.0;
            totalAudioSec += audioSec[i];
        }

        double replyRtf = (cumRenderMs / 1000.0) / Math.Max(totalAudioSec, 1e-6);
        Console.WriteLine($"[KokoroStream] {BackendName}: {renderMs.Length} chunks, {totalAudioSec:F2}s of audio "
                        + $"rendered in {cumRenderMs:F0} ms (whole-reply RTF {replyRtf:F2}x) | "
                        + $"tightest margin {worstMarginMs:F0} ms at chunk {worstChunk} | "
                        + $"time to first audio {firstReadyMs:F0} ms");

        // ── The coalesced alternative ────────────────────────────────────────────────────────────
        // A pass's host cost is ~fixed (the same ~1,850 nodes of dispatch prep and the same 3-4 GPU
        // drains whatever the length), so rendering N sentences as N passes multiplies that fixed cost
        // by N while the audio only adds up linearly. Rendering the SAME content as ONE pass pays it
        // once. This measures that directly rather than assuming it.
        // Capped at Kokoro's 510-token context - the coalesced control is a reference point, not a
        // proposal to feed the model more than it takes.
        int totalTokens = Math.Min(510, KokoroReplyChunkTokenCounts.Sum() - (KokoroReplyChunkTokenCounts.Length - 1) * 3);
        var oneShot = BuildKokoroChunk(totalTokens, 0);
        var swOne = Stopwatch.StartNew();
        var oneAudio = await pipeline.SpeakTokensAsync(oneShot, pack);
        swOne.Stop();
        double oneMs = swOne.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[KokoroStream] {BackendName}: COALESCED {totalTokens} tok -> {oneAudio.Seconds:F2}s audio "
                        + $"in {oneMs:F0} ms (RTF {oneMs / 1000.0 / Math.Max(oneAudio.Seconds, 1e-6):F2}x) "
                        + $"vs {renderMs.Length} chunks {cumRenderMs:F0} ms for {totalAudioSec:F2}s (RTF {replyRtf:F2}x) "
                        + $"| per-pass fixed cost saved: {cumRenderMs - oneMs:F0} ms");

        // 🔴 THE TIMING ASSERTION IS SCOPED TO THE BACKENDS THAT CAN ACTUALLY STREAM, and that is a
        // statement about the DEVICE, not a way to dodge a red. Kokoro on CPU renders ~260 s per pass and
        // WebGL/Wasm are structurally slow (no shared memory or atomics on WebGL; interpreted non-AOT
        // Wasm) - none of them can render speech faster than speech, so asserting "no underrun" there
        // would encode a requirement that backend can never meet and that nobody ships.
        //
        // ⚠️ THE NUMBERS ARE PRINTED ON EVERY BACKEND REGARDLESS (above), so a regression stays visible
        // where it cannot be gated. What is scoped is the THROW, not the measurement.
        //
        // ⚠️ This was learned the expensive way the same day: two scalar-pool tests were written to run
        // everywhere, passed vacuously on four backends and failed on WebGL for a capability reason, and
        // the vacuous passes were the worse half. Scope a test to what it is actually about.
        bool assertRealtime = accelerator.AcceleratorType
            is AcceleratorType.WebGPU or AcceleratorType.Cuda or AcceleratorType.OpenCL;

        if (!assertRealtime)
        {
            Console.WriteLine($"[KokoroStream] {BackendName}: REPORTED, NOT GATED - this backend is not one "
                            + "realtime speech is targeted on, so the schedule above is information rather "
                            + $"than a pass/fail (whole-reply RTF {replyRtf:F2}x, "
                            + $"{(underruns.Count == 0 ? "no underrun" : $"{underruns.Count} late chunk(s)")}).");
            return;
        }

        if (underruns.Count > 0)
            throw new Exception(
                $"{BackendName}: the reply STALLS - {underruns.Count} of {renderMs.Length} chunks arrive after "
              + $"playback needs them ({string.Join("; ", underruns)}). Whole-reply RTF is {replyRtf:F2}x and the "
              + $"tightest margin is {worstMarginMs:F0} ms at chunk {worstChunk} "
              + $"({KokoroReplyChunkTokenCounts[worstChunk]} tokens, {audioSec[worstChunk]:F2}s of audio for "
              + $"{renderMs[worstChunk]:F0} ms of work). A pass's host cost is roughly fixed regardless of length, "
              + "so the fix is on the chunking side - coalesce sentences until a chunk is worth a pass - not in "
              + "the per-chunk renderer.");

        // The whole reply has to fit, not just chunk-to-chunk: if total render exceeds first-audio plus
        // total duration, playback runs dry before the last sentence however the chunks are spaced.
        if (cumRenderMs > firstReadyMs + totalAudioSec * 1000.0)
            throw new Exception(
                $"{BackendName}: the reply does not fit - {cumRenderMs:F0} ms of rendering for {totalAudioSec:F2}s "
              + $"of audio starting at {firstReadyMs:F0} ms. The speaker runs dry before the last sentence.");
    });
}
