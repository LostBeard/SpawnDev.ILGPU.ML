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

        // ── 0a2. IS THE RECORDING ITSELF THE VARIABLE? ───────────────────────────────────────────────
        //
        // ⚠️ THIS IS THE ONE DIFFERENCE THE BISECT ABOVE STRUCTURALLY CANNOT SEE. Every regime above is a
        // plain forward with flags flipped, and 0b's window sweep is a hand-rolled emulation of the capture
        // regime - warm x2 + the arena + SuppressDrains - which is CLEAN at every window. Meanwhile the
        // REAL WebGPUGraphCapture.TryCaptureAsync pass disagrees with a plain forward in all 16,900 values.
        // Something the real path does is not in the emulation, and the candidates are exactly three:
        // MaxPendingReleaseBytes = 64 MiB, a SynchronizeAsync after each warm pass, and an ACTIVE
        // BeginDispatchCapture around the forward. The first two are cheap to add; the third cannot be
        // expressed as a flag, so it needs this.
        //
        // The experiment: run the SAME forward under the SAME regime, once with a plan recording and once
        // without, and compare. Recording is supposed to be a passive observer - it appends pipeline and
        // bind-group handles to a JS array - so a difference here means it is not passive, and that is the
        // whole bug. The plan is discarded either way; nothing is replayed, so a difference cannot be
        // blamed on replay.
        //
        // ⚠️ 8,141 host writes happen inside this window and ALL of them are the benign retained
        // scalar-params upload (unreplayable work = 0, MEASURED 2026-09-04). So this is NOT re-testing the
        // missing-work theory - that one is already dead, and this is what is left.
        if (accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator webGpuProbe)
        {
            Graph.GraphExecutor.UseCaptureParamSlots = true;
            Kernels.FusedAttentionKernel.UseStableCaptureSlots = true;
            var prevRelease = Graph.GraphExecutor.MaxPendingReleaseBytes;
            Graph.GraphExecutor.MaxPendingReleaseBytes = 64L * 1024 * 1024;
            try
            {
                // Prime exactly as TryCaptureAsync does - warm A, sync, warm B, sync - so the arena slots
                // and pool buckets are in the state the real capture pass finds them in.
                await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                await accelerator.SynchronizeAsync();
                await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                await accelerator.SynchronizeAsync();

                // (i) the regime WITHOUT a recording - the emulation, which is known clean.
                Graph.GraphExecutor.SuppressDrains = true;
                float[] noRecord;
                try { noRecord = await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim); }
                finally { Graph.GraphExecutor.SuppressDrains = false; }

                // (ii) the identical regime WITH a recording active. Plan discarded, never replayed.
                Graph.GraphExecutor.SuppressDrains = true;
                float[] recorded;
                var probePlan = webGpuProbe.BeginDispatchCapture();
                try { recorded = await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim); }
                finally
                {
                    Graph.GraphExecutor.SuppressDrains = false;
                    try { webGpuProbe.EndDispatchCapture().Dispose(); } catch { }
                }
                await accelerator.SynchronizeAsync();

                var (nrD, nrW) = Compare(directCapture, noRecord);
                var (rD, rW) = Compare(directCapture, recorded);
                var (rvD, rvW) = Compare(noRecord, recorded);
                Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] RECORDING ISOLATION: "
                    + $"same regime, plan discarded | without recording {nrD} of {count} differ (worst {nrW:F6}) "
                    + $"| WITH recording {rD} differ (worst {rW:F6}) | the two against each other {rvD} differ "
                    + $"(worst {rvW:F6}) | plan saw {probePlan.DispatchCount} dispatch(es), "
                    + $"{probePlan.HostWriteCount} host write(s) of which {probePlan.ScalarParamWriteCount} scalar-params "
                    + $"=> unreplayable {probePlan.HostWriteCount - probePlan.ScalarParamWriteCount}");
            }
            finally
            {
                Graph.GraphExecutor.MaxPendingReleaseBytes = prevRelease;
                Graph.GraphExecutor.UseCaptureParamSlots = false;
                Kernels.FusedAttentionKernel.UseStableCaptureSlots = false;
            }
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
        // ⚠️ PROBE THE **REAL** CAPTURE PASS, NOT AN EMULATION OF IT. Everything the emulation above can
        // express is now CLEAN by measurement - each regime flag alone, all of them together, the warm
        // priming, the runtime-constant seed (the warm passes DO snapshot it: the condition is
        // UseCaptureParamSlots && !SuppressDrains), an active recording, and unreplayable host writes
        // (0 of 8,141). Yet TryCaptureAsync's capture pass still disagrees with a plain forward in all
        // 16,900 values. So the difference is inside the real path, and the per-node probe - which named
        // Range_1_output_0 on 2026-09-03 - had only ever been pointed at the emulation.
        //
        // The reference is re-recorded here rather than reused from 0b so this block stands alone.
        Graph.GraphExecutor.NodeProbeFromIndex = 0;
        Graph.GraphExecutor.NodeProbeFirst64 = new System.Collections.Generic.Dictionary<string, float[]>();
        Graph.GraphExecutor.NodeProbeOrder = new System.Collections.Generic.List<string>();
        graphs.EnableGraphCapture = false;
        await graphs.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var realRefProbe = Graph.GraphExecutor.NodeProbeFirst64!;
        var realRefOrder = Graph.GraphExecutor.NodeProbeOrder!;
        Graph.GraphExecutor.NodeProbeOrder = null;
        WebGPUGraphCapture.RecordCapturePassOutput = true;
        graphs.EnableGraphCapture = true;

        var observePass = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);

        // Armed across the capture call. TryCaptureAsync runs warm A, warm B and then the capture pass, so
        // the surviving values are the CAPTURE pass's (last writer wins) - which is exactly the pass under
        // suspicion. ⚠️ If probing everything HEALS the real capture the way it heals the emulation, that
        // is itself the finding: the fault is an ordering hazard, and the window has to be walked back.
        Graph.GraphExecutor.NodeProbeFirst64 = new System.Collections.Generic.Dictionary<string, float[]>();
        var capturePass = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var realCapProbe = Graph.GraphExecutor.NodeProbeFirst64!;
        Graph.GraphExecutor.NodeProbeFirst64 = null;
        Graph.GraphExecutor.NodeProbeFromIndex = 0;
        {
            int idx = -1; string nm = ""; string detail = "";
            for (int i = 0; i < realRefOrder.Count; i++)
            {
                var n = realRefOrder[i];
                if (!realCapProbe.TryGetValue(n, out var b) || !realRefProbe.TryGetValue(n, out var a)) continue;
                bool same = a.Length == b.Length;
                for (int k = 0; same && k < a.Length; k++) if (a[k] != b[k]) same = false;
                if (same) continue;
                idx = i; nm = n;
                detail = $"plain [{string.Join(",", a.Take(4).Select(v => v.ToString("G6")))}] vs capture "
                       + $"[{string.Join(",", b.Take(4).Select(v => v.ToString("G6")))}]";
                break;
            }
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] REAL CAPTURE probe: "
                + $"{realRefOrder.Count} reference nodes, {realCapProbe.Count} probed in the capture call; "
                + (idx < 0 ? "NO probed node differs"
                           : $"first differing node #{idx} '{nm}' {detail}"));

            // ⚠️ THE CONTRADICTION THIS RESOLVES. "No probed node differs" and "the returned output differs
            // in ALL 16,900 values" cannot both be about the same numbers unless the output the CALLER gets
            // is not the output the last node WROTE. The probe reads only the first
            // MaxSmallReadbackElements (64) of each node - but the output differs everywhere, first 64
            // included, and the graph's own output IS one of the probed nodes. So the two readings are of
            // DIFFERENT things, and the difference is introduced after the node loop, when the result is
            // extracted.
            //
            // Prime suspect: under SuppressDrains the pool's release is DEFERRED, so the buffer backing the
            // returned tensor can be recycled and rewritten before the caller copies it out. That would
            // corrupt the whole tensor while leaving every node's own write correct - exactly this shape.
            //
            // Printing the same four values from three places is what separates them. If TAIL matches PLAIN
            // and RETURNED does not, the computation is right and the extraction is wrong.
            var lastNode = realRefOrder.Count > 0 ? realRefOrder[^1] : null;
            string tailRef = lastNode != null && realRefProbe.TryGetValue(lastNode, out var tr)
                ? string.Join(",", tr.Take(4).Select(v => v.ToString("G6"))) : "(none)";
            string tailCap = lastNode != null && realCapProbe.TryGetValue(lastNode, out var tc)
                ? string.Join(",", tc.Take(4).Select(v => v.ToString("G6"))) : "(none)";
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] OUTPUT EXTRACTION: "
                + $"last node '{lastNode}' | plain-tail [{tailRef}] | capture-tail [{tailCap}] | "
                + $"plain-RETURNED [{string.Join(",", directCapture.Take(4).Select(v => v.ToString("G6")))}] | "
                + $"capture-RETURNED [{string.Join(",", capturePass.Take(4).Select(v => v.ToString("G6")))}]");
        }

        var replaySame = await graphs.RunDecoderAsync(
            tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
        var (obsDiff, obsWorst) = Compare(directCapture, observePass);
        var (capDiff, capWorst) = Compare(directCapture, capturePass);
        // 🔴 THIS LINE USED TO SAY "both are ordinary forwards, so a difference here is not about replay at
        // all", AND THAT WAS EXACTLY BACKWARDS. `capturePass` is the value returned by the call that
        // performs the capture - and SessionGraphCapture.RunAsync ends with
        //     if (_webGpu != null) return await _webGpu.ReplayAsync(inputs);
        // so that call returns a REPLAY. Naming it the capture pass sent 2026-09-03 and most of 2026-09-04
        // looking for arithmetic that goes wrong under the capture regime; there is none. MEASURED: all
        // 4,873 probed node outputs of the real capture pass match a plain forward, and the true capture
        // output read inside TryCaptureAsync (below) is the thing to compare.
        Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] "
                        + $"observe pass: {obsDiff} differ (worst {obsWorst:F6}) | "
                        + $"post-capture call (⚠️ A REPLAY, not the capture pass): {capDiff} differ "
                        + $"(worst {capWorst:F6})");
        // The capture pass's OWN output, read between the capture forward and the pool releases.
        if (WebGPUGraphCapture.CapturePassOutput is { } capOwn)
        {
            var (ownDiff, ownWorst) = Compare(directCapture, capOwn);
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] TRUE CAPTURE "
                + $"PASS output ({capOwn.Length} values, read inside TryCaptureAsync): {ownDiff} differ "
                + $"(worst {ownWorst:F6}) => "
                + (ownDiff == 0
                    ? "the capture pass is CORRECT; the fault is in REPLAY, and it is not missing host writes "
                      + "(unreplayable work measured 0)"
                    : "the capture pass itself is wrong - the regime bisect above is the place to look"));
        }
        else
        {
            Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] TRUE CAPTURE "
                + "PASS output: NOT RECORDED (RecordCapturePassOutput was off, or capture never went live)");
        }
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
        // ── 1b. IS THE MISSING WORK THE ELIDED DISPATCHES? ───────────────────────────────────────────
        //
        // The capture pass is correct and unreplayable host writes are zero, so the replay is missing work
        // that is neither bad arithmetic nor a queue.writeBuffer. Dispatch-elide is the one mechanism that
        // removes work from the plan BY DESIGN - a CPU-resolved shape op emits no dispatch, so the plan
        // records nothing for it and its buffer keeps whatever an earlier pass left there.
        //
        // A fresh graphs instance, because capture is attempted once per session.
        // ⚠️ This costs a second decoder session and ~1200 extra dispatches per replay. It is a DIAGNOSTIC
        // A/B, not a proposed setting.
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
        {
            WebGPUGraphCapture.ElideDispatchDuringCapture = false;
            try
            {
                using var noElide = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);
                noElide.EnableGraphCapture = true;
                noElide.AllowControlFlowCapture = true;
                await noElide.RunDecoderAsync(tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                var neReplay = await noElide.RunDecoderAsync(
                    tCapture, x, encoding.TextCondition, speech, guidance, numFrames, featDim);
                var (neDiff, neWorst) = Compare(directCapture, neReplay);
                Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] ELIDE A/B: "
                    + $"captured with dispatch-elide OFF -> replay {neDiff} of {count} differ (worst {neWorst:F6}) "
                    + $"| capture {(noElide.DecoderCaptured ? "LIVE" : "NOT live: " + noElide.DecoderCaptureStatus)} => "
                    + (neDiff == 0
                        ? "ELIDED DISPATCHES ARE THE MISSING WORK - record a write for each elided value "
                          + "(the CaptureParamArena.CaptureConstWrite pattern)"
                        : "elide is NOT the cause - look at buffers a WARM pass populated that the plan never rewrites"));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Benchmark] ZipVoiceFidelity [{accelerator.AcceleratorType}] ELIDE A/B failed: {ex.Message}");
            }
            finally { WebGPUGraphCapture.ElideDispatchDuringCapture = true; }
        }

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
