using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Kokoro-82M end to end, against an onnxruntime reference waveform.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 THIS IS THE GATE THE PORT DID NOT HAVE. Nine separate defects were found by running this model and
/// every one of them produced a full, correctly-shaped, non-throwing buffer: a ConvTranspose bias rented
/// from the pool and never zeroed (0.333 added to all 54,620 samples), a fused Sigmoid that was declared,
/// mapped and then silently not applied (every phoneme duration clamped to 1), an STFT hop that shape
/// inference could not see (5 frames instead of 121), an LSTM whose runtime shape re-inference was skipped
/// because an OPTIONAL input was absent, an epsilon deleted from every normalisation by strength reduction
/// reading an int-truncated constant, and an atan2 written longhand that returns the wrong sign of pi at an
/// exactly-zero imaginary part. Not one of them threw. The ONLY thing that separates "this model runs" from
/// "this model is right" is comparing the samples to a reference.
/// </para>
/// <para>
/// ⚠️ EACH ASSERTION BELOW CORRESPONDS TO A DEFECT THAT ACTUALLY SHIPPED, and they are not
/// interchangeable - the length catches the duration and shape bugs, the peak catches the un-zeroed bias
/// (which gave peak 2226) and a silent output, correlation catches the epsilon (0.45) and the atan2
/// (0.95), and the best-fit scale catches noise in the log-magnitude, which inflates loudness through
/// Exp without ever decorrelating the signal.
/// </para>
/// <para>
/// ⚠️ <b>HeavyModel</b>: fetches ~326 MB of model through the hub, so it is excluded from every sweep by
/// default and runs when asked for:
/// <c>PMT_EXCLUDE_CATEGORIES= PMT_FILTER=Pipeline_Kokoro dotnet test PlaywrightMultiTest/...</c>
/// </para>
/// <para>
/// ⭐ PROVENANCE OF THE FIXTURE, so it can be regenerated rather than trusted:
/// <c>node tools/kokoro-oracle.mjs &lt;model.onnx&gt; &lt;af_heart.bin&gt; &lt;tokens-csv&gt;
/// SpawnDev.ILGPU.ML.Demo/wwwroot/test-refs/kokoro-af_heart-paris.f32</c> - onnxruntime, fp32 export,
/// the token ids below. ⚠️ The <b>fp32</b> export specifically: the published fp16 one correlates only
/// 0.918 against it, a bigger deviation than this engine's entire remaining error.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    private const string KokoroHubBase =
        "https://hub.spawndev.com:44365/hf/onnx-community/Kokoro-82M-v1.0-ONNX";

    /// <summary>The line the reference waveform was rendered from, and its exact phoneme tokens.</summary>
    /// <remarks>
    /// ⚠️ The TOKENS are the fixture, not the text. Feeding the reference engine the text would compare two
    /// FRONT ENDS as well as two inference engines, so a phonemizer change would read as an engine
    /// regression. The KokoroFrontEnd tests cover the text-to-token half
    /// separately; this covers the token-to-audio half.
    /// </remarks>
    private const string KokoroReferenceLine = "The capital of France is Paris.";

    private static readonly long[] KokoroReferenceTokens =
    {
        0, 81, 83, 16, 53, 156, 72, 58, 83, 125, 83, 54, 16, 138, 64, 16, 48, 123, 156, 72, 56, 61, 16,
        102, 68, 16, 58, 156, 86, 123, 102, 61, 16, 4, 0,
    };

    /// <summary>
    /// Synthesise the reference line and compare every sample to onnxruntime's.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_Kokoro_MatchesOnnxRuntimeWaveform() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        // The reference is served from the demo's wwwroot; the model comes through the hub, which is the
        // ONLY route to a Hugging Face artefact in this project.
        float[] reference;
        try
        {
            var refBytes = await http.GetByteArrayAsync("test-refs/kokoro-af_heart-paris.f32");
            reference = new float[refBytes.Length / 4];
            Buffer.BlockCopy(refBytes, 0, reference, 0, refBytes.Length);
        }
        catch (HttpRequestException ex)
        {
            // NOT UnsupportedTestException. The reference is COMMITTED next to the demo, so it not being
            // served means the harness is wrong - and a gate that answers "unsupported" is green.
            throw new Exception("test-refs/kokoro-af_heart-paris.f32 is committed but was not served: "
                + ex.Message);
        }

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

        // ⚠️ TWICE WHERE THAT IS AFFORDABLE, because the first run of this graph is not the cost of the
        // graph. Every kernel compiles on its FIRST execution, and on a browser backend that compile is
        // most of a cold run - MEASURED on WebGPU (real RTX 4070, isFallbackAdapter=False): 8.1 s cold,
        // 2.1 s warm. Reporting only the cold figure describes the first reply of a session as though it
        // were every reply, which is the difference between "slower than realtime" and "faster than
        // realtime" for this model.
        //
        // 🔴 But the SECOND pass is a measurement, and correctness is the gate. The CPU backend takes
        // ~312 s per pass, so running it twice cost 624 s against PMT's 600 s outer cap and the row was
        // KILLED - a backend that was verifying correctness perfectly well reported as a failure, for a
        // timing number nobody would ship anyway. So the warm pass runs only where the cold one showed it
        // is affordable, and its absence is stated rather than papered over.
        var cold = Stopwatch.StartNew();
        var audio = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
        cold.Stop();

        const int warmBudgetMs = 60_000;
        Stopwatch? sw = null;
        if (cold.ElapsedMilliseconds <= warmBudgetMs)
        {
            sw = Stopwatch.StartNew();
            audio = await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
            sw.Stop();
        }

        // ── Length. A wrong duration, a wrong STFT hop and a stale LSTM sequence length all land here,
        //    and all of them produce audio rather than an error. EXACT, not approximate: the reference
        //    and this engine are computing the same deterministic frame count from the same tokens.
        if (audio.Samples.Length != reference.Length)
            throw new Exception($"length {audio.Samples.Length} != reference {reference.Length} on "
                + $"{BackendName} - a duration or frame-count defect, not a numeric one");

        var peak = 0f;
        foreach (var s in audio.Samples) { var a = MathF.Abs(s); if (a > peak) peak = a; }
        if (peak < 0.05f)
            throw new Exception($"peak {peak:F4} on {BackendName} - the graph ran and produced silence");
        if (peak > 1.5f)
            throw new Exception($"peak {peak:F4} on {BackendName} - a stale buffer or a missing "
                + "activation; speech does not leave the interval this model's output lives in");

        double rr = 0, oo = 0, ro = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            double r = reference[i], o = audio.Samples[i];
            rr += r * r; oo += o * o; ro += r * o;
        }
        var correlation = ro / Math.Sqrt(rr * oo);
        // The least-squares gain of ours over the reference. Noise entering the LOG magnitude survives Exp
        // as a systematic loudness gain without decorrelating anything, so correlation alone cannot see it
        // - the epsilon defect left correlation at 0.95 and this at 1.3.
        var scale = ro / rr;

        // 0.98 is set by the model, not by taste. Its vocoder takes atan2 of an STFT whose quiet bins are
        // near zero, so the phase there is decided by rounding: perturbing onnxruntime's OWN spectrogram by
        // one part in 1e6 moves its waveform to correlation 0.974 against itself. 0.98 sits above every
        // measured defect (0.45, 0.95) and below that noise floor.
        if (correlation < 0.98)
            throw new Exception($"correlation {correlation:F4} against onnxruntime on {BackendName} "
                + $"(scale {scale:F3}) - the samples are wrong, not merely rounded");
        if (scale < 0.85 || scale > 1.2)
            throw new Exception($"best-fit gain {scale:F3} on {BackendName} (correlation {correlation:F4}) "
                + "- correlated but at the wrong level");

        var warmth = sw == null
            ? $"warm not measured (cold exceeded the {warmBudgetMs / 1000}s budget for a second pass)"
            : $"warm {sw.ElapsedMilliseconds} ms = RTF {sw.Elapsed.TotalSeconds / audio.Seconds:F2}x";
        Console.WriteLine($"[Kokoro] \"{KokoroReferenceLine}\" -> {audio.Samples.Length} samples "
            + $"({audio.Seconds:F2}s), {warmth} (cold {cold.ElapsedMilliseconds} ms = RTF "
            + $"{cold.Elapsed.TotalSeconds / audio.Seconds:F2}x), peak {peak:F3}, correlation "
            + $"{correlation:F4}, gain {scale:F3} on {BackendName}");

        // ⚠️ A DIAGNOSTIC MUST NEVER FAIL A CORRECTNESS GATE. Everything above this line is the gate and
        // has already passed; everything below is reporting, and some of it calls runtime APIs whose WASM
        // support is not something to find out from a red HeavyModel row minutes into a sweep.
        if (sw != null)
        {
            try { await ReportKokoroCostSplitAsync(pipeline, pack, audio.Seconds); }
            catch (Exception ex) { Console.WriteLine($"[KokoroCost] {BackendName}: not reported ({ex.GetType().Name}: {ex.Message})"); }
        }
    });

    /// <summary>
    /// A third, INSTRUMENTED pass that makes the browser report where its own time went.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 🔴 WHY IT CANNOT BE EXTRAPOLATED FROM THE DESKTOP. This engine's browser cost is per-DISPATCH host
    /// orchestration, and its desktop cost is the GPU work - so the two rank the graph in opposite orders.
    /// MEASURED on CUDA: <c>encoder/bert</c> is 602 of 1,850 nodes and <b>1.6%</b> of the time, while
    /// <c>decoder/generator</c> is 639 nodes and 68.5%. Reading that as "optimise the generator" is exactly
    /// the wrong conclusion for a browser, where 602 near-free dispatches are 602 crossings. Any decision
    /// about splitting work across backends has to be made on the browser's numbers, so the browser has to
    /// print them.
    /// </para>
    /// <para>
    /// ⚠️ A SEPARATE PASS, and never the one the RTF is taken from: this puts a Stopwatch around every
    /// node, which is not free on the .NET WASM heap. Reporting a timing figure measured through the
    /// instrument that exists to explain it is how an overhead becomes a conclusion.
    /// </para>
    /// <para>
    /// ⚠️ Read it with <c>PMT_CONSOLE_LOG=Kokoro</c>. PMT summarises browser console output to
    /// "Console: N error(s)", so without that switch these lines are computed and discarded.
    /// </para>
    /// </remarks>
    private async Task ReportKokoroCostSplitAsync(KokoroPipeline pipeline, KokoroVoicePack pack, double seconds)
    {
        // ⚠️ EVERY READ THAT CAN THROW HAPPENS BEFORE ANY STATIC IS SET. The caller catches, but its catch
        // cannot run this method's `finally` - so a throw while capturing the baselines below would leave
        // the per-node Stopwatch and the per-dispatch timestamps armed for every later test on this lane.
        var gc0 = (Alloc: GC.GetTotalAllocatedBytes(false), Pause: GC.GetTotalPauseDuration(),
                   G0: GC.CollectionCount(0), G1: GC.CollectionCount(1), G2: GC.CollectionCount(2));

        var timings = new Dictionary<string, double>();
        Graph.GraphExecutor.CapturedNodeTimingsMs = timings;

        // ⭐ THE FOUR PHASES OF A WEBGPU DISPATCH, which is where a browser's per-node cost actually
        // lives. The accelerator has accumulated these all along and nothing has ever read them for this
        // model. Each phase has a DIFFERENT fix - shader resolve is a cache, arg build is marshalling,
        // bind group is object creation, encode is the BeginComputePass/SetPipeline/SetBindGroup/
        // DispatchWorkgroups/End/Dispose crossings - so "the browser is slow" is only actionable once one
        // of the four is named. Zeros on a desktop backend, which is correct: there are no crossings there.
        var profWasOn = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling;
        var bgCacheWasOn = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching;
        var p0 = (SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuShaderResolveMs,
                  SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuArgBuildMs,
                  SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupMs,
                  SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuEncodeMs);
        SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = true;

        // 🔴 The GC baseline above is the one cost invisible to every other counter here, and the one that
        // scales with what ELSE is resident. .NET WASM's GC is non-concurrent, so a collection stops the
        // orchestrator mid-graph and its pause is a function of the LIVE heap - the other models in the
        // worker, not this one. That is the shape of the open worker gap: identical graph, identical card,
        // 3,889 ms in the demo worker against 1,736 ms in a page with nothing else loaded. A small pause
        // total here exonerates the GC and makes the cost per-crossing.
        try
        {
            var probe = Stopwatch.StartNew();
            await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
            probe.Stop();

            // ⭐ THE BIND-GROUP CACHE A/B, measured here because this is the only place that already knows
            // the four dispatch phases. The pass above shows bind-group creation as the LARGEST of them on
            // WebGPU, and WebGPUBackend.EnableBindGroupCaching is opt-in and OFF - so those groups are
            // built and thrown away once per dispatch, 1,850 times per utterance, for a graph that
            // re-dispatches the same kernels over the same buffers every pass.
            //
            // Safe to flip here: the cache is mutually exclusive with DISPATCH CAPTURE, and this pipeline
            // runs uncaptured (KokoroPipeline.EnableGraphCapture defaults to false and nothing sets it).
            // Restored in the finally below with the profiling flags.
            //
            // ⚠️ Both passes are WARM and identical in shape - this is the 3rd and 4th SpeakTokensAsync of
            // the test - so the difference is the cache and not kernel compilation.
            double cachedMs = -1;
            string? cacheFailure = null;
            long bgHits = 0, bgMisses = 0; int bgEntries = 0;

            Console.WriteLine($"[KokoroCost] {BackendName} gc: "
                + $"{(GC.GetTotalAllocatedBytes(false) - gc0.Alloc) / 1048576.0:F1} MB allocated, "
                + $"gen0 {GC.CollectionCount(0) - gc0.G0}, gen1 {GC.CollectionCount(1) - gc0.G1}, "
                + $"gen2 {GC.CollectionCount(2) - gc0.G2}, "
                + $"pause {(GC.GetTotalPauseDuration() - gc0.Pause).TotalMilliseconds:F0} ms, "
                + $"live heap {GC.GetTotalMemory(false) / 1048576.0:F0} MB");

            var shaderMs = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuShaderResolveMs - p0.Item1;
            var argMs = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuArgBuildMs - p0.Item2;
            var bindMs = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuBindGroupMs - p0.Item3;
            var encMs = SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ProfileCpuEncodeMs - p0.Item4;
            if (shaderMs + argMs + bindMs + encMs > 0.5)
                Console.WriteLine($"[KokoroCost] {BackendName} dispatch phases: shader-resolve {shaderMs:F0} ms, "
                    + $"arg-build {argMs:F0} ms, bind-group {bindMs:F0} ms, encode {encMs:F0} ms "
                    + $"(total {shaderMs + argMs + bindMs + encMs:F0} ms)");

            // ⭐ AND WHETHER THE BIND-GROUP CACHE IS ACTUALLY WORKING, which decides what the bind-group
            // column above MEANS. The key is (pipeline + the exact buffers bound), so a pool that hands a
            // node a different buffer each pass misses every time and pays a CreateBindGroup - a real JS
            // object - per dispatch. A high hit rate makes that column irreducible; a low one makes it a
            // pool-stability problem with an obvious fix. The counters have always been exposed and, like
            // the phase timers, have never been read for this model.
            // ⚠️ SAY WHEN IT IS NOT MEASURED. The demo's copy of this reported "0 hits / 0 misses" for
            // three runs before I noticed that was a failed cast, not an idle cache - and a zero like that
            // gets quoted. A diagnostic that cannot tell "measured zero" from "not measured" is worse than
            // one that is absent.
            if (pipeline.Session.Accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator wgpu)
            {
                var hits = wgpu.BindGroupCacheHits;
                var misses = wgpu.BindGroupCacheMisses;
                Console.WriteLine($"[KokoroCost] {BackendName} bind-group cache: {hits} hits, {misses} misses "
                    + $"({(hits + misses > 0 ? hits * 100.0 / (hits + misses) : 0):F1}% hit), "
                    + $"{wgpu.BindGroupCacheEntryCount} entries");
            }
            else
            {
                Console.WriteLine($"[KokoroCost] {BackendName} bind-group cache: not available "
                    + $"({pipeline.Session.Accelerator.GetType().Name} is not a WebGPUAccelerator)");
            }

            var ex = Graph.GraphExecutor.LastRunTotalMs;
            var rbN = Graph.GraphExecutor.LastRunReadbackCount;
            var rbMs = Graph.GraphExecutor.LastRunReadbackMs;
            var drN = Graph.GraphExecutor.LastRunSyncDrainCount;
            var drMs = Graph.GraphExecutor.LastRunSyncDrainMs;
            Console.WriteLine($"[KokoroCost] {BackendName}: instrumented pass {probe.ElapsedMilliseconds} ms, "
                + $"executor {ex:F0} ms | readbacks {rbN} ({rbMs:F0} ms) | drains {drN} ({drMs:F0} ms) | "
                + $"residual {ex - rbMs - drMs:F0} ms | nodes {pipeline.Session.NodeCount} | "
                + $"audio {seconds:F2}s");

            // ⭐ THE BIND-GROUP CACHE A/B — runs LAST, and its failure is reported rather than thrown.
            //
            // Bind-group creation is the LARGEST of the four dispatch phases on WebGPU for this model, and
            // WebGPUBackend.EnableBindGroupCaching is opt-in and OFF, so those groups are built and thrown
            // away once per dispatch, 1,850 times per utterance, for a graph that re-dispatches the same
            // kernels over the same buffers every pass.
            //
            // 🔴 IT DOES NOT CURRENTLY WORK ON THIS MODEL. MEASURED 2026-09-15: turning it on produced
            // "[WebGPU] 330 GPU error(s) during dispatch" and the graph failed at node 1238 'ReduceSum'
            // (/encoder/predictor). That is a LIBRARY defect worth fixing - it is ~39% of the dispatch
            // cost of a pipeline whose whole goal is beating realtime - and this line is here so the state
            // of it is a measured fact in every sweep rather than folklore.
            //
            // ⚠️ THE FIRST VERSION OF THIS RAN BEFORE THE REPORT AND THREW, so the 330-error failure
            // aborted the whole method and the WebGPU cost numbers - the reason the method exists - never
            // printed. The caller catches, so the gate stayed green and the loss was silent. A diagnostic
            // must never take down the report it belongs to.
            //
            // Safe to flip: the cache is mutually exclusive with DISPATCH CAPTURE and this pipeline runs
            // uncaptured (KokoroPipeline.EnableGraphCapture defaults false, nothing sets it). Restored in
            // the finally.
            if (pipeline.Session.Accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator wgpuAb)
            {
                try
                {
                    SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = true;
                    // A first pass with the cache on only POPULATES it (recur-only: a signature is stored
                    // on its 2nd sighting), so timing a single run would measure the miss path and report
                    // no win even if the cache were perfect.
                    await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
                    await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
                    var abClock = Stopwatch.StartNew();
                    await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
                    abClock.Stop();
                    cachedMs = abClock.Elapsed.TotalMilliseconds;
                    bgHits = wgpuAb.BindGroupCacheHits;
                    bgMisses = wgpuAb.BindGroupCacheMisses;
                    bgEntries = wgpuAb.BindGroupCacheEntryCount;
                }
                catch (Exception abEx)
                {
                    var m = abEx.Message;
                    // 2000, not 300: the first cap truncated away every individual WebGPU validation message
                    // and left only the count, which names no defect. The errors themselves are the diagnosis.
                    // FLATTEN THE NEWLINES: the exception body is the count, then a newline, then one line per
                    // GPU error - and PMT_CONSOLE_LOG keeps only lines CONTAINING the filter word, so every
                    // continuation line was silently dropped and the log showed the COUNT with none of the
                    // errors. A diagnostic that reaches the log as a number with no evidence is no diagnostic.
                    m = m.Replace("\r", "").Replace("\n", " | ");
                    cacheFailure = m.Length > 2000 ? m[..2000] : m;
                }
                finally
                {
                    SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = bgCacheWasOn;
                }
            }

            // GRAPH CAPTURE A/B - the lever that removes ALL the per-dispatch prep, not one phase of it.
            //
            // WebGPU spends the bulk of a Kokoro pass building dispatches on the CPU: arg-build 323 ms +
            // bind-group 640 ms + encode 112 ms = ~1,091 ms of a ~1,539 ms pass, for 1,850 nodes. CUDA runs
            // the same 1,850 nodes in 674 ms. Graph capture records the command plan once and re-executes
            // it, so that whole cost collapses on every pass after the first.
            //
            // 🔴 EVERY OTHER PIPELINE IN THIS LIBRARY DEFAULTS IT ON - AudioPipelines (Whisper) = true,
            // DepthEstimationPipeline = true ("ON by default: consumers forgetting the ..."). KokoroPipeline
            // declares `public bool EnableGraphCapture { get; set; }` and so defaults to FALSE. Kokoro is the
            // one pipeline that never got it.
            //
            // ⚠️ Capture needs several passes before a plan is live (warm / probe / record), so a single
            // timed run would measure recording, not replay. And CaptureStatus is reported rather than
            // inferred - the type's own docs say "requested is not live".
            double uncapturedMs = -1;
            string capStatus = pipeline.CaptureStatus;
            string? captureFailure = null;
            var capWasOn = pipeline.EnableGraphCapture;
            try
            {
                // ⚠️ THE BASELINE IS THE ONE THAT HAS TO BE FORCED NOW. Capture is the DEFAULT, so `probe`
                // above is already a captured pass - comparing it against another captured pass reports
                // ~1.00x and reads as "capture does nothing", which is the opposite of the truth. Turn it
                // OFF for the baseline instead.
                pipeline.EnableGraphCapture = false;
                await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
                var capClock = Stopwatch.StartNew();
                await pipeline.SpeakTokensAsync(KokoroReferenceTokens, pack);
                capClock.Stop();
                uncapturedMs = capClock.Elapsed.TotalMilliseconds;
            }
            catch (Exception capEx)
            {
                var cm = capEx.Message.Replace("\r", "").Replace("\n", " | ");
                captureFailure = cm.Length > 1200 ? cm[..1200] : cm;
            }
            finally
            {
                pipeline.EnableGraphCapture = capWasOn;
            }

            if (captureFailure != null)
                Console.WriteLine($"[KokoroCost] {BackendName} GRAPH CAPTURE A/B: BASELINE FAILED - {captureFailure}");
            else
                Console.WriteLine($"[KokoroCost] {BackendName} GRAPH CAPTURE A/B: "
                    + $"off {uncapturedMs:F0} ms -> on {probe.ElapsedMilliseconds} ms "
                    + $"({(probe.ElapsedMilliseconds > 0 ? uncapturedMs / probe.ElapsedMilliseconds : 0):F2}x) | status={capStatus} | "
                    + $"RTF on={(probe.ElapsedMilliseconds / 1000.0) / seconds:F2}x (<1 = faster than realtime)");

            if (cacheFailure != null)
                Console.WriteLine($"[KokoroCost] {BackendName} BIND-GROUP CACHE A/B: STILL BROKEN — {cacheFailure}");
            else if (cachedMs >= 0)
                Console.WriteLine($"[KokoroCost] {BackendName} BIND-GROUP CACHE A/B: "
                    + $"off {probe.ElapsedMilliseconds} ms -> on {cachedMs:F0} ms "
                    + $"({(cachedMs > 0 ? probe.ElapsedMilliseconds / cachedMs : 0):F2}x) | "
                    + $"{bgHits} hits, {bgMisses} misses, {bgEntries} entries | "
                    + $"RTF on={(cachedMs / 1000.0) / seconds:F2}x (lower is better, <1 = faster than realtime)");

            // Coarse on purpose - the question is "which half", because that is where a placement split
            // between two backends would cut. Same buckets the DemoConsole prints, so the browser row and
            // the CUDA row are directly comparable.
            static string Stage(string key)
            {
                var name = key[(key.IndexOf('_') + 1)..];
                name = name[(name.IndexOf('_') + 1)..];
                if (name.Contains("/generator/")) return "decoder/generator";
                if (name.StartsWith("/decoder")) return "decoder-pre-generator";
                if (name.Contains("/bert")) return "encoder/bert";
                if (name.Contains("/predictor")) return "encoder/predictor";
                if (name.StartsWith("/encoder")) return "encoder-other";
                return "other";
            }
            var buckets = new Dictionary<string, (int Nodes, double Ms)>();
            foreach (var kv in timings)
            {
                var st = Stage(kv.Key);
                var cur = buckets.TryGetValue(st, out var b) ? b : (0, 0d);
                buckets[st] = (cur.Item1 + 1, cur.Item2 + kv.Value);
            }
            var totalMs = buckets.Values.Sum(b => b.Ms);
            foreach (var kv in buckets.OrderByDescending(k => k.Value.Nodes))
                Console.WriteLine($"[KokoroCost] {BackendName}   {kv.Key,-22} {kv.Value.Nodes,5} nodes  "
                    + $"{kv.Value.Ms,9:F1} ms ({kv.Value.Ms * 100.0 / Math.Max(totalMs, 1e-9),4:F1}%)  "
                    + $"{kv.Value.Ms / Math.Max(kv.Value.Nodes, 1),6:F3} ms/node");
        }
        finally
        {
            // ⚠️ STATIC, BOTH OF THEM. Left set, every later test on this lane pays the per-node Stopwatch
            // and the per-dispatch timestamps, and grows a dictionary nothing reads. Restore the profiling
            // flag to what it WAS rather than to false - something else may have turned it on.
            Graph.GraphExecutor.CapturedNodeTimingsMs = null;
            SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableDispatchProfiling = profWasOn;
            // 🔴 RESTORE THE CACHE FLAG TOO. It is a STATIC on the backend, so leaving it on would change
            // the behaviour of every later test on this lane - and it is mutually exclusive with dispatch
            // capture, so the tests it would break are the ones that capture, far from here and for no
            // visible reason. A diagnostic that mutates global state is part of the system under test.
            SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.EnableBindGroupCaching = bgCacheWasOn;
        }
    }
}
