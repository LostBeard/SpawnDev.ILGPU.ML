using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// WHERE does replaying the captured ZipVoice decoder stop matching the direct forward?
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY THIS EXISTS. <c>Pipeline_ZipVoice_SpeaksInTheBrowser</c> proves the defect - MEASURED on WebGPU
/// 2026-09-03: "replaying the captured decoder changed the audio: 73216 of 73216 samples differ, worst
/// 0.504501" - and cannot localise it. It compares AUDIO, at the end of an encoder, four Euler steps of
/// host-side integration and a vocoder, so every candidate cause produces the identical symptom, and the
/// error message has to end with a guess ("the Euler timestep is the obvious candidate").
/// </para>
/// <para>
/// This asks the two questions separately, on the decoder's own output tensor and nothing else:
/// <list type="number">
/// <item><b>Is a replay faithful at the inputs it was CAPTURED with?</b> If not, the recorded plan is
/// missing work outright - a host-side operator writes no dispatch, and a dispatch elided into a
/// capture-time constant has none to record - and no amount of input plumbing will fix it.</item>
/// <item><b>Does a CHANGED input reach the replay?</b> Run the same plan at a different timestep. If (1)
/// passes and (2) fails, the plan is complete and something downstream of <c>t</c> is frozen.</item>
/// </list>
/// The two have different fixes, and the audio-level test cannot tell them apart.
/// </para>
/// <para>
/// ⚠️ SYNTHETIC conditioning on purpose. The decoder is a pure function of its five inputs, so
/// deterministic pseudo-random <c>x</c> / <c>speech_condition</c> exercise the identical graph while
/// skipping the 54 MB vocoder download entirely. The SHAPES come from the real encoder, so they are not
/// invented - the decoder's cost and its capture are both per-shape.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Pipeline_ZipVoice_CaptureReplayFidelity() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.Cuda))
            throw new UnsupportedTestException(
                $"graph capture is CUDA and WebGPU only; {accelerator.AcceleratorType} has nothing to replay");

        using var http = CreateHuggingFaceHttpClient();
        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));

        // ⚠️ The vocoder is loaded but never RUN. Nothing here turns a mel into audio - the comparison is on
        // the decoder's own output tensor - but IlgpuZipVoiceGraphs owns three sessions and constructs all
        // three, so an empty byte[] here is a parse failure, not a saving.
        await WarmArchiveAsync(http, VocoderArchive);
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        using var graphs = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);

        var promptTokens = tokenizer.Encode(LibrivoxTranscript);
        var tokens = tokenizer.Encode("Paint the sockets in the wall dull green.");
        var encoding = await graphs.RunEncoderAsync(tokens, promptTokens, 96, 1.0f);
        int numFrames = encoding.NumFrames, featDim = encoding.FeatDim;
        int count = numFrames * featDim;
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + $"frames {numFrames} x feat {featDim} = {count} elements");

        var x = Pseudo(count, 11);
        var speech = Pseudo(count, 22);
        const float tCapture = 0.25f, tOther = 0.75f, guidance = 1.0f;

        // ── controls: the direct forward at both timesteps ───────────────────────────────────────────
        graphs.EnableGraphCapture = false;
        var directCapture = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var directCaptureAgain = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var directOther = await graphs.RunDecoderAsync(
            tOther, x, encoding.TextCondition, speech, guidance, numFrames, featDim);

        // ⚠️ CALIBRATE THE INSTRUMENT FIRST. Everything below reads a value difference as evidence about
        // capture. If the uncaptured decoder is not deterministic, or if its output does not depend on t at
        // all, none of those readings mean anything - and an uncalibrated null result is worse than none,
        // because it is believable. (That exact mistake cost a day on the audio-level version of this test,
        // where the control was an unseeded noise vector.)
        var (detDiff, detWorst) = Compare(directCapture, directCaptureAgain);
        if (detDiff != 0)
            throw new Exception($"the UNCAPTURED decoder is not deterministic: {detDiff} of "
                              + $"{directCapture.Length} values differ between two identical calls (worst "
                              + $"{detWorst:F6}). No capture verdict below can be read until that is fixed.");
        var (tSensDiff, _) = Compare(directCapture, directOther);
        if (tSensDiff == 0)
            throw new Exception($"the decoder returned the SAME output for t={tCapture} and t={tOther}, so "
                              + "this test cannot detect a frozen timestep - the instrument is blind and a "
                              + "pass would mean nothing.");
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] control: "
                        + $"deterministic, and {tSensDiff} of {count} values change with t");

        // ── 0. WHICH PART OF THE CAPTURE REGIME CHANGES THE ANSWER? ──────────────────────────────────
        //
        // ⚠️ MEASURED 2026-09-03: the CAPTURE PASS - an ordinary forward, no replay involved - already
        // disagrees with a plain forward in all 16,900 values. So the fault is in the regime the capture
        // pass runs under, not in the plan or the replay. The regime is four independent switches, and
        // reasoning about which one it is has been wrong twice; toggling them one at a time is not.
        //
        // Capture is NOT enabled here - these are plain forwards with one flag flipped, so a difference is
        // attributable to that flag alone.
        graphs.EnableGraphCapture = false;
        foreach (var (label, apply, undo) in new (string, Action, Action)[]
        {
            ("SuppressDrains",
                () => Graph.GraphExecutor.SuppressDrains = true,
                () => Graph.GraphExecutor.SuppressDrains = false),
            ("UseCaptureParamSlots",
                () => Graph.GraphExecutor.UseCaptureParamSlots = true,
                () => Graph.GraphExecutor.UseCaptureParamSlots = false),
            ("FusedAttention stable slots",
                () => Kernels.FusedAttentionKernel.UseStableCaptureSlots = true,
                () => Kernels.FusedAttentionKernel.UseStableCaptureSlots = false),
            ("SuppressDrains, deferred release kept",
                () => { Graph.GraphExecutor.SuppressDrains = true;
                        Graph.GraphExecutor.CaptureImmediateReturn = false; },
                () => { Graph.GraphExecutor.SuppressDrains = false;
                        Graph.GraphExecutor.CaptureImmediateReturn = true; }),
            ("SuppressDrains + capture seed (both capture flags)",
                () => { Graph.GraphExecutor.SuppressDrains = true;
                        Graph.GraphExecutor.UseCaptureParamSlots = true; },
                () => { Graph.GraphExecutor.SuppressDrains = false;
                        Graph.GraphExecutor.UseCaptureParamSlots = false; }),
            // ⚠️ SuppressDrains does TWO things at once. This one isolates the SYNCHRONIZE half without it:
            // drains still return buffers, but the interval is pushed past the graph so no
            // SynchronizeAsync fires mid-forward. If this corrupts, the fault is the missing flush and not
            // the missing readbacks - and the readbacks are already skipped on the correct path anyway,
            // because IlgpuZipVoiceGraphs sets CacheShapeReadbacks = true on this session.
            ("no mid-forward SynchronizeAsync (drains still return)",
                () => { Graph.GraphExecutor.SyncIntervalNodes = 1_000_000;
                        Graph.GraphExecutor.MaxPendingReleaseBytes = long.MaxValue; },
                () => { Graph.GraphExecutor.SyncIntervalNodes = 64;
                        Graph.GraphExecutor.MaxPendingReleaseBytes = 512L * 1024 * 1024; }),
            ("ShapeInterpElideDispatch off",
                () => Graph.GraphExecutor.ShapeInterpElideDispatch = false,
                () => Graph.GraphExecutor.ShapeInterpElideDispatch = true),
        })
        {
            apply();
            float[] under;
            try
            {
                under = await graphs.RunDecoderAsync(
                    tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
            }
            finally { undo(); }
            var (d, w) = Compare(directCapture, under);
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] regime "
                            + $"'{label}': {d} of {count} differ (worst {w:F6})");
        }

        // ── 0b. WHERE does the capture regime first diverge? ─────────────────────────────────────────
        //
        // ⚠️ ARMING THE PROBE EVERYWHERE MAKES THE FAILING RUN PASS. It synchronizes after every node, and
        // MEASURED 2026-09-03 that alone is enough: under the full capture regime not one of 4,873 node
        // outputs differed, while the same run's final output differed in all 16,900 values. So the fault is
        // an ORDERING hazard, and the instrument has to leave the suspect region alone.
        //
        // The probe is therefore WINDOWED and the window walked backwards. A late window leaves the
        // corrupting region unperturbed, so the damage is still visible when the window reads it; an early
        // window synchronizes through the region and heals it. The boundary between those two is where the
        // hazard lives. (Env vars cannot do this - they do not reach the Blazor WASM runtime, which is why
        // ML_CF_CAPTURE never worked in a browser lane - so the sweep is in code.)
        {
            // Reference: a fully probed PLAIN forward. Perturbing this one is harmless - it is correct
            // either way, and it is the only run whose values are needed for every window.
            Graph.GraphExecutor.NodeProbeFromIndex = 0;
            Graph.GraphExecutor.NodeProbeFirst64 = new System.Collections.Generic.Dictionary<string, float[]>();
            Graph.GraphExecutor.NodeProbeOrder = new System.Collections.Generic.List<string>();
            Graph.GraphExecutor.NodeProbeCounts = new System.Collections.Generic.Dictionary<string, long>();
            await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
            var baseProbe = Graph.GraphExecutor.NodeProbeFirst64!;
            var order = Graph.GraphExecutor.NodeProbeOrder!;
            var baseCounts = Graph.GraphExecutor.NodeProbeCounts!;
            Graph.GraphExecutor.NodeProbeOrder = null;
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] reference probe: "
                            + $"{order.Count} node outputs recorded");

            // The capture regime needs its arena slots populated by warm passes at the same cursor, exactly
            // as WebGPUGraphCapture does it, or the arena is the variable rather than the thing under test.
            Graph.GraphExecutor.UseCaptureParamSlots = true;
            Kernels.FusedAttentionKernel.UseStableCaptureSlots = true;
            try
            {
                await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);

                foreach (var w in new[] { order.Count, 4000, 3000, 2000, 1000, 0 })
                {
                    Graph.GraphExecutor.NodeProbeFromIndex = Math.Min(w, order.Count);
                    Graph.GraphExecutor.NodeProbeFirst64 = new System.Collections.Generic.Dictionary<string, float[]>();
                    Graph.GraphExecutor.NodeProbeCounts = new System.Collections.Generic.Dictionary<string, long>();
                    Graph.GraphExecutor.SuppressDrains = true;
                    float[] outUnder;
                    try
                    {
                        outUnder = await graphs.RunDecoderAsync(
                            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                    }
                    finally { Graph.GraphExecutor.SuppressDrains = false; }
                    var underProbe = Graph.GraphExecutor.NodeProbeFirst64!;
                    var underCounts = Graph.GraphExecutor.NodeProbeCounts!;
                    var (wd, ww) = Compare(directCapture, outUnder);

                    int firstIdx = -1; string firstName = ""; string detail = "";
                    for (int i = 0; i < order.Count; i++)
                    {
                        var nm = order[i];
                        if (!underProbe.TryGetValue(nm, out var b)) continue;
                        if (!baseProbe.TryGetValue(nm, out var a)) continue;
                        bool same = a.Length == b.Length;
                        for (int k = 0; same && k < a.Length; k++) if (a[k] != b[k]) same = false;
                        if (same) continue;
                        firstIdx = i; firstName = nm;
                        detail = $"plain [{string.Join(",", a.Take(4).Select(v => v.ToString("G6")))}] "
                               + $"(n={baseCounts.GetValueOrDefault(nm, -1)}) vs capture-regime "
                               + $"[{string.Join(",", b.Take(4).Select(v => v.ToString("G6")))}] "
                               + $"(n={underCounts.GetValueOrDefault(nm, -1)})";
                        break;
                    }
                    Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + $"window>={Graph.GraphExecutor.NodeProbeFromIndex}: output {wd} of {count} differ "
                        + $"(worst {ww:F6}); "
                        + (firstIdx < 0 ? "no probed node differs" : $"first differing node #{firstIdx} '{firstName}' {detail}"));
                }
            }
            finally
            {
                Graph.GraphExecutor.UseCaptureParamSlots = false;
                Kernels.FusedAttentionKernel.UseStableCaptureSlots = false;
                Graph.GraphExecutor.NodeProbeFirst64 = null;
                Graph.GraphExecutor.NodeProbeCounts = null;
                Graph.GraphExecutor.NodeProbeFromIndex = 0;
            }
        }

        // ── 1. replay at the CAPTURED inputs ─────────────────────────────────────────────────────────
        graphs.EnableGraphCapture = true;
        graphs.AllowControlFlowCapture = true;
        // ⚠️ EVERY PASS IS KEPT, not just the replay. SessionGraphCapture runs three different regimes for
        // these three calls - an OBSERVE forward, then the CAPTURE pass (drains suppressed, readbacks
        // skipped, shape values seeded from the warm snapshot, dispatch-elide forced on), then replays -
        // and only the third is "a replay". If the CAPTURE PASS itself already disagrees with the direct
        // forward, the plan is a recording of the wrong computation and the replay is faithfully repeating
        // it, which is a completely different bug from a replay that loses work.
        var observePass = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var capturePass = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var replaySame = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var (obsDiff, obsWorst) = Compare(directCapture, observePass);
        var (capDiff, capWorst) = Compare(directCapture, capturePass);
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + $"observe pass: {obsDiff} differ (worst {obsWorst:F6}) | "
                        + $"CAPTURE pass: {capDiff} differ (worst {capWorst:F6}) "
                        + "- both are ordinary forwards, so a difference here is not about replay at all");
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] capture LIVE: "
                        + $"{graphs.DecoderCaptured} - {graphs.DecoderCaptureStatus}");
        // ⚠️ THE FIRST NUMBER TO READ when a replay does not reproduce its own capture. A plan records
        // command-encoder work only; a queue.writeBuffer inside the window is work the replay silently
        // skips. Non-zero here names the cause instead of leaving it to be deduced.
        //
        // ⚠️ AND READ THE SECOND NUMBER TOO. This line printed 0 on 2026-09-03 and that zero was taken as
        // evidence the plan was complete. It was not: the census hooked WebGPUBuffer's upload paths and
        // NOT the packed scalar-params queue.writeBuffer that the dispatch path issues once PER DISPATCH,
        // so the busiest CPU->GPU path in the engine was invisible to the instrument that was supposed to
        // find it. Counting it is what turns "the plan is missing work" from a deduction into a number.
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] host writes inside "
                        + $"the capture window: {WebGPUGraphCapture.HostWritesDuringCapture} "
                        + $"(of which {WebGPUGraphCapture.ScalarParamWritesDuringCapture} are per-dispatch "
                        + "packed scalar params, which the plan RETAINS and are therefore benign) => "
                        + $"unreplayable work = {WebGPUGraphCapture.HostWritesDuringCapture - WebGPUGraphCapture.ScalarParamWritesDuringCapture}");
        if (!graphs.DecoderCaptured)
            throw new Exception("capture never went live, so there is no replay to check: "
                              + graphs.DecoderCaptureStatus);
        var (sameDiff, sameWorst) = Compare(directCapture, replaySame);

        // ── 2. replay at a DIFFERENT timestep ────────────────────────────────────────────────────────
        var replayOther = await graphs.RunDecoderAsync(
            tOther, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var (otherDiff, otherWorst) = Compare(directOther, replayOther);
        // Is the replay simply ignoring t - still producing its capture-time answer?
        var (frozenDiff, _) = Compare(replayOther, replaySame);

        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + $"replay@captured-inputs: {sameDiff} of {count} differ (worst {sameWorst:F6}) | "
                        + $"replay@t={tOther}: {otherDiff} differ (worst {otherWorst:F6}) | "
                        + $"replay@t={tOther} vs replay@t={tCapture}: {frozenDiff} differ "
                        + "(0 => the timestep never reached the plan)");

        if (capDiff != 0)
            throw new Exception(
                $"the CAPTURE PASS itself disagrees with a plain forward: {capDiff} of {count} values differ "
              + $"(worst {capWorst:F6}). Replay is not involved - the capture pass runs with drains "
              + "suppressed, per-node readbacks skipped and shape values seeded from the warm snapshot, so "
              + "the plan being recorded is a recording of the WRONG computation. Fix that before looking "
              + "at replay at all.");
        if (sameDiff != 0)
            throw new Exception(
                $"a replay does not reproduce the forward it recorded: {sameDiff} of {count} values differ "
              + $"(worst {sameWorst:F6}) at the EXACT inputs it was captured with. Input plumbing cannot "
              + "explain this - the plan is missing work the forward does (a host-side operator writes no "
              + "dispatch, and an elided shape op has none to record).");
        if (frozenDiff == 0)
            throw new Exception(
                $"the replay produced byte-identical output for t={tCapture} and t={tOther}: the timestep "
              + "never reaches the recorded plan, so every Euler step integrates the same instant.");
        if (otherDiff != 0)
            throw new Exception(
                $"replay at t={tOther} differs from the direct forward at the same inputs: {otherDiff} of "
              + $"{count} values (worst {otherWorst:F6}). The plan replays its own capture correctly and it "
              + "does respond to t, so what is wrong is downstream of the input copy.");

        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + "replay is faithful at the captured inputs AND at a changed timestep");
    });

    /// <summary>Deterministic pseudo-random conditioning in [-1, 1].</summary>
    private static float[] Pseudo(int n, int seed)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>How many values differ, and by how much at worst.</summary>
    private static (int Diff, float Worst) Compare(float[] a, float[] b)
    {
        if (a.Length != b.Length) return (Math.Max(a.Length, b.Length), float.NaN);
        int diff = 0; float worst = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d != 0f) { diff++; worst = MathF.Max(worst, d); }
        }
        return (diff, worst);
    }
}
