using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Does ZipVoice actually SPEAK in a browser?
///
/// <para>
/// It has spoken on our engine since 2026-08-30 - but only ever in the desktop harness. NOTHING in the
/// browser-capable suite constructed <c>ZipVoicePipeline</c> or <c>IlgpuZipVoiceGraphs</c> before this
/// file, so "ZipVoice works" and "ZipVoice works where the demo runs" were different claims and only the
/// first had evidence. That gap is the same shape as the one that hid a microphone capture which had never
/// been implemented, and the one that let a reversed slice write nothing on every backend.
/// </para>
///
/// <para>
/// ⚠️ Auditing the path first found exactly ONE filesystem dependency in it -
/// <c>ZipVoiceTokenizer.CreateDefault</c> read <c>tokens.txt</c> off disk. The pipeline, the graphs and the
/// feature extraction are all filesystem-free, and <c>IZipVoiceGraphs</c>' own remarks say it is async
/// precisely so it is not "quietly restricted to the desktop" - which, at that last step, it was.
/// <c>CreateFromTokens</c> closes it.
/// </para>
///
/// <para>
/// ⚠️ <b>HeavyModel</b>: this fetches ~185 MB (fm_decoder_int8 124.7 MB, mel_spec_24khz 54.2 MB,
/// text_encoder_int8 5.6 MB). HeavyModel is excluded from every sweep by default, so this runs when asked
/// for and does not tax the release gate.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private const string ZipVoiceRepo = "k2-fsa/ZipVoice";

    /// <summary>
    /// ZipVoice's vocoder - mel spectrogram back to a waveform - reached through the hub's source proxy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ <b>It is NOT <c>wetdog/vocos-mel-24khz-onnx</c>.</b> That repo looks exactly right and is the
    /// wrong direction: it holds the mel EXTRACTOR (audio -> mel), not the vocoder (mel -> audio). The two
    /// files sit 431 bytes apart in size, which is how the wrong one passed for the right one - a decoy that
    /// survives every check except actually listening to the output.
    /// </para>
    /// <para>
    /// ⚠️ The vocoder is published ONLY inside a sherpa-onnx release tarball, and it must be the <b>fp32</b>
    /// package: the int8 package ships an encoder, a decoder and tokens.txt and <b>no vocoder at all</b>
    /// (verified by listing it - 3 members, none of them a vocoder). A <c>.tar.bz2</c> cannot be seeked
    /// into, so the whole archive must be fetched and decompressed by somebody; the hub does it once, on the
    /// machine with the disk, instead of every visitor doing it in a browser tab.
    /// </para>
    /// </remarks>
    private const string VocoderArchive =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2";
    private const string VocoderMember = "sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx";
    private const string SourceProxyBaseUrl = "https://hub.spawndev.com:44365";

    /// <summary>The hub URL that serves one file out of a remote archive as if it were a plain file.</summary>
    private static string ArchiveMemberUrl(string archiveUrl, string member)
        => $"{SourceProxyBaseUrl}/src?url={Uri.EscapeDataString(archiveUrl)}"
         + $"&member={Uri.EscapeDataString(member)}";

    /// <summary>
    /// Ask the hub to cache an archive, and wait until it has, without holding a request open.
    /// </summary>
    /// <remarks>
    /// ⚠️ Do NOT simply request the member and wait. First contact with this archive means the hub fetches
    /// 634 MB and bzip2-decompresses it, which takes minutes - and the gateway in front of the hub answers
    /// <b>504 at 25 seconds</b> while the hub is still working perfectly. A test written the obvious way
    /// fails with a timeout that looks like a broken server and is not one. <c>/src/warm</c> returns 202
    /// immediately and 200 once cached, so waiting happens between requests instead of inside one.
    /// </remarks>
    private static async Task WarmArchiveAsync(HttpClient http, string archiveUrl)
    {
        var warmUrl = $"{SourceProxyBaseUrl}/src/warm?url={Uri.EscapeDataString(archiveUrl)}";
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < TimeSpan.FromMinutes(25))
        {
            var res = await http.GetAsync(warmUrl);
            if (res.StatusCode == System.Net.HttpStatusCode.NotFound)
                throw new UnsupportedTestException(
                    "the deployed hub has no /src/warm yet - redeploy the hub to arm this test");
            if (res.IsSuccessStatusCode && res.StatusCode != System.Net.HttpStatusCode.Accepted)
                return;                                     // 200: cached and ready
            if (res.StatusCode != System.Net.HttpStatusCode.Accepted)
                throw new Exception($"the hub could not cache the archive: {(int)res.StatusCode} "
                                  + await res.Content.ReadAsStringAsync());
            Console.WriteLine($"[ZipVoice] hub still caching the vocoder archive "
                            + $"({sw.Elapsed.TotalSeconds:F0}s): {await res.Content.ReadAsStringAsync()}");
            await Task.Delay(TimeSpan.FromSeconds(10));
        }
        throw new Exception("the hub did not finish caching the vocoder archive within 25 minutes");
    }

    /// <summary>The librivox fixture's transcript, as WE know it independently of any recogniser.</summary>
    /// <remarks>
    /// ⚠️ The reference transcript must be EXACT. `SpeakAsync`'s own remarks are explicit that anything
    /// present in the audio and missing here bleeds into the start of the generated line - so a sloppy
    /// transcript degrades the clone for a reason nobody would be able to see in the output.
    /// </remarks>
    private const string LibrivoxTranscript = "All LibriVox recordings are in the public domain.";

    /// <summary>
    /// 🔴 DIAGNOSTIC, not a correctness test: is executing a branch subgraph what hangs the device
    /// during capture?
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠️ THE AUDIO FROM THIS TEST IS DELIBERATELY WRONG. <c>IfOperator.BypassSubgraphForCaptureProbe</c>
    /// leaves every If's output unwritten, so nothing here asserts anything about sound. The single
    /// question is whether <c>SessionGraphCapture</c> can then record the decoder without
    /// DXGI_ERROR_DEVICE_HUNG.
    /// </para>
    /// <para>
    /// WHY IT IS WORTH A TEST OF ITS OWN. The decoder is 80% of a synthesis and costs roughly 1 ms per node
    /// across ~8,621 nodes; a replayed plan does per-node dispatch in microseconds, so capture is the single
    /// largest lever in the pipeline. It is refused because the graph contains control flow, and lifting
    /// that refusal hung the GPU on the first attempt - even though SubgraphRunner caches its plans and the
    /// branch census says <c>then=21, else=0</c>, i.e. every If takes a branch that is ONE Constant node.
    /// Removing those Ifs by constant folding would need a compiler stage, a weight-hoisting path and a
    /// shape-time evaluator. This says whether that work would pay, for the price of one run.
    /// </para>
    /// <para>
    /// ⚠️ Run it deliberately - it can reset the display driver:
    /// <c>PMT_FILTER=ZipVoice_CaptureEligibility PMT_EXCLUDE_CATEGORIES= </c>
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel,WasmHeavy,Diagnostic")]
    public async Task Pipeline_ZipVoice_CaptureEligibilityProbe() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU
            && accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException("capture is CUDA and WebGPU only; nothing to probe here");

        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var reference = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode the reference clip");

        using var http = CreateHuggingFaceHttpClient();
        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        await WarmArchiveAsync(http, VocoderArchive);
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));

        using var graphs = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);
        using var pipeline = new ZipVoicePipeline(graphs);

        graphs.AllowControlFlowCapture = true;                       // the thing under test
        Operators.IfOperator.BypassSubgraphForCaptureProbe = true;   // no subgraph runs at all
        Operators.IfOperator.ResetBranchCensus();
        try
        {
            var sw = Stopwatch.StartNew();
            var r = await pipeline.SpeakAsync("Paint the sockets in the wall dull green.",
                LibrivoxTranscript, reference, 16000, tokenizer);
            sw.Stop();
            Console.WriteLine($"[Benchmark] ZipVoice PROBE [{accelerator.AcceleratorType}] SURVIVED: "
                + $"capture LIVE={graphs.DecoderCaptured} ({graphs.DecoderCaptureStatus}) | "
                + $"{sw.Elapsed.TotalSeconds:F1}s | decoder {r.DecoderMs:F0}ms "
                + $"(first step {r.DecoderFirstStepMs:F0}ms, rest {r.DecoderRemainingStepsMs:F0}ms) | "
                + $"If then={Operators.IfOperator.ThenBranchCount} else={Operators.IfOperator.ElseBranchCount}");
            Console.WriteLine($"[Benchmark] ZipVoice PROBE verdict: "
                + (graphs.DecoderCaptured
                    ? "capture RECORDS once the subgraph is out of the window - folding the Ifs away WOULD pay"
                    : "capture still did not engage - folding the Ifs away would NOT have been enough"));
        }
        finally
        {
            Operators.IfOperator.BypassSubgraphForCaptureProbe = false;
        }
    });

    // "WasmHeavy" as well as "HeavyModel": this drives ~185 MB of models through a full TTS pipeline, which
    // is exactly the shape the Wasm lane's interpreted-IL budget exists to keep out - it PASSES on every
    // other backend and would only ever be a timeout there. Run it deliberately with
    // PMT_EXCLUDE_CATEGORIES_WASM= plus a name filter.
    [TestMethod(Timeout = 1800000, Category = "HeavyModel,WasmHeavy")]
    public async Task Pipeline_ZipVoice_SpeaksInTheBrowser() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        // ⚠️ The ILGPU CPU backend needs longer than PMT's OUTER console cap
        // (ProjectRunner.ConsoleTestTimeoutMs() = 600,000 ms) for a 124.7 MB int8 decoder - measured: still
        // inside the ENCODER past 600 s, having burned ~6,900 s of CPU time. PMT kills the subprocess, which
        // surfaces as "no 'TEST:' line - subprocess crashed, exit=-1" with EMPTY stderr. That reads like a
        // crash and is not one: exit=-1 is a killed process.
        //
        // This is a skip, not a hidden failure, and the distinction rests on what IS covered: the operator
        // that made this pipeline fail in the first place (MatMulInteger, N-D shapes) is verified ON THE CPU
        // LANE against a CPU reference by Op_MatMulInteger_BatchedActivationKeepsItsRank and
        // Op_MatMulInteger_SizeOneBatchAxisSurvives. The arithmetic is checked on CPU; only the 30-minute
        // end-to-end render is not.
        //
        // To run it deliberately: PMT_CONSOLE_TIMEOUT_MS=3600000 with a name filter.
        if (accelerator.AcceleratorType == AcceleratorType.CPU
            && Environment.GetEnvironmentVariable("ZIPVOICE_ALLOW_CPU") != "1")
            throw new UnsupportedTestException(
                "the ILGPU CPU backend exceeds PMT's 600s outer console cap for this model (measured >600s "
              + "still in the encoder); operator correctness is covered on the CPU lane by "
              + "Op_MatMulInteger_*. Set ZIPVOICE_ALLOW_CPU=1 with PMT_CONSOLE_TIMEOUT_MS to run it.");

        // ── the reference voice: real speech with a transcript we know ───────────────────────────────
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var reference = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // ── the three graphs, int8, straight from HuggingFace ────────────────────────────────────────
        using var http = CreateHuggingFaceHttpClient();
        var sw = Stopwatch.StartNew();
        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/text_encoder_int8.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/fm_decoder_int8.onnx"));
        // Through the hub rather than from HuggingFace: this file exists nowhere as a standalone download.
        // Warmed first so the member request itself is always a warm, fast one - see WarmArchiveAsync.
        await WarmArchiveAsync(http, VocoderArchive);
        var vocoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, ArchiveMemberUrl(VocoderArchive, VocoderMember));
        // The vocoder is 54,157,409 bytes. Checked explicitly because the failure this guards against is a
        // PLAUSIBLE wrong file rather than a missing one, and a wrong vocoder does not throw - it renders
        // noise, which every remaining assertion in this test would happily accept as speech.
        if (vocoderBytes.Length != 54_157_409)
            throw new Exception($"the vocoder is {vocoderBytes.Length} bytes, expected 54,157,409. A "
                              + "different file here does not fail loudly, it just makes noise.");
        var tokensBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(ZipVoiceRepo, "zipvoice_distill/tokens.txt"));
        double fetchMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] fetched "
                        + $"{(encoderBytes.Length + decoderBytes.Length + vocoderBytes.Length) / 1048576.0:F1} MB "
                        + $"in {fetchMs / 1000:F1}s");

        // The symbol table comes from CONTENT, not a path - the whole point of this test.
        var tokenizer = ZipVoiceTokenizer.CreateFromTokens(Encoding.UTF8.GetString(tokensBytes));

        // ⚠️ The control-flow opt-in is a PROPERTY on the graphs (AllowControlFlowCapture), not an
        // environment variable. It used to be ML_CF_CAPTURE=1, which could never work where it matters:
        // environment variables do not reach the Blazor WASM runtime, so in a browser lane it read as unset
        // no matter what was exported - and the browser is the only place the refusal costs ~20x. MEASURED
        // 2026-09-03: a run with ML_CF_CAPTURE=1 exported still reported "refused: graph contains control
        // flow (If)" on WebGPU. The env var survives only as a way to turn it OFF for a desktop A/B.

        using var graphs = IlgpuZipVoiceGraphs.Create(accelerator, encoderBytes, decoderBytes, vocoderBytes);
        // ⚠️ Capture is exercised HERE even though it defaults OFF in production, because the sample-level
        // A/B below is the thing that decides when that default may flip. A gate that only ran the shipping
        // configuration could never tell us whether replay had become faithful - it would just keep
        // reporting that capture was refused. ML_CF_CAPTURE=0 turns it off for a before/after control.
        graphs.AllowControlFlowCapture = Environment.GetEnvironmentVariable("ML_CF_CAPTURE") != "0";
        Console.WriteLine($"[Benchmark] ZipVoice control-flow capture requested: "
                        + $"{graphs.AllowControlFlowCapture}");
        using var pipeline = new ZipVoicePipeline(graphs);

        // 🔴 PIN THE NOISE SEED, or every sample comparison below is meaningless.
        //
        // ⚠️ This is not a workaround for a defect - it is the documented contract. Zero-shot flow matching
        // starts its ODE from FRESH noise every call (NoiseSeed defaults to null -> Random.Shared), so two
        // renders of the same sentence legitimately differ, and differ audibly. MEASURED 2026-09-03 before
        // this line existed: two uncaptured renders of the same line differed in ALL 73,216 samples, worst
        // 0.551049.
        //
        // ⚠️ That nearly cost a wrong conclusion. The capture A/B below reported "replaying the captured
        // decoder changed the audio: 73216 of 73216 samples differ" and it was believed - but the control
        // it compared against was itself different every time, so the comparison indicted capture for
        // something capture had nothing to do with. A null result from an uncalibrated instrument is worse
        // than no result, because it is believable. The determinism check below now proves the instrument
        // before the verdict is read.
        pipeline.NoiseSeed = 20260903;

        // ── speak ────────────────────────────────────────────────────────────────────────────────────
        // ⚠️ Census the If branches over this one utterance. fm_decoder has FIVE Ifs whose else branch is
        // 254 nodes against a then branch of ONE Constant, so which side runs is worth thousands of node
        // executions per Euler step - on the stage that is 82% of a synthesis.
        Operators.IfOperator.ResetBranchCensus();
        // ⚠️ CUMULATIVE, not LastRun. A synthesis is one encoder pass plus NumSteps decoder passes plus a
        // vocoder pass, and every LastRun* field is overwritten by the next RunAsync - so reading them
        // afterwards reports the VOCODER and makes the decoder look free. That is why the older "0
        // readbacks" line here was never evidence about the decoder at all.
        //
        // Readbacks are what decides whether a captured plan can be faithful: a value read to the host lets
        // the executor compute on the CPU and ELIDE the dispatch, and a recorded plan replays dispatches
        // only - so an elided computation freezes at its capture-time value forever. Driving readbacks to
        // zero is the documented precondition for capturing anything.
        Graph.GraphExecutor.CumulativeReset();
        const string line = "Paint the sockets in the wall dull green.";
        sw.Restart();
        var result = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
        double speakMs = sw.Elapsed.TotalMilliseconds;

        var audio = result.Audio;
        if (audio == null || audio.Length == 0)
            throw new Exception("ZipVoice produced NO audio - the pipeline ran and returned nothing.");

        // ── assertions that a broken render cannot satisfy ───────────────────────────────────────────
        // Silence is the failure mode to guard hardest: a pipeline that returns zeros has the right
        // length and the right type, and only an amplitude check can tell it from speech.
        float peak = 0f;
        double energy = 0;
        foreach (var v in audio) { peak = MathF.Max(peak, MathF.Abs(v)); energy += (double)v * v; }
        double rms = Math.Sqrt(energy / audio.Length);
        double seconds = audio.Length / (double)result.SampleRate;

        if (peak < 0.01f)
            throw new Exception($"ZipVoice returned effectively SILENCE (peak {peak:F5} over {audio.Length} "
                              + "samples). A pipeline that produces zeros looks identical to one that works "
                              + "at every check except amplitude.");
        if (rms < 0.005)
            throw new Exception($"ZipVoice output RMS {rms:F5} is too low to be speech (peak {peak:F4}).");
        // A ~40-character line is not going to be under half a second or over twenty.
        if (seconds < 0.5 || seconds > 20.0)
            throw new Exception($"ZipVoice produced {seconds:F2}s for a {line.Length}-character line - the "
                              + "duration prediction inside the encoder decides this, so a wild value means "
                              + "the encoder ran wrong rather than the vocoder.");

        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}]: spoke {seconds:F2}s of audio "
                        + $"at {result.SampleRate} Hz in {speakMs / 1000:F2}s "
                        + $"({speakMs / 1000 / seconds:F2}x realtime, lower is better) "
                        + $"| peak {peak:F3} rms {rms:F4}");

        // The SPLIT, not just the total. Where the time goes decides what is worth optimising, and the
        // pipeline already measures it - printing only the total throws that away and invites the next
        // person to guess. The decoder is the stage to watch: it runs Config.NumSteps Euler steps, and the
        // integration between them happens on the HOST, so each step is a GPU round trip on the largest
        // model in the pipeline (fm_decoder_int8, 124.7 MB). That is the same shape that cost the Silero
        // VAD 22.8x before its readbacks were removed.
        // ⚠️ Readbacks and sync drains, because the browser is 23x slower than CUDA on IDENTICAL code
        // and that gap is far too large to be backend arithmetic. The Silero VAD had the same shape and the
        // cause was readbacks, not dispatch count: driving them 16 -> 0 took it from 177.9 to 7.81 ms/frame
        // (22.8x). Print the counters rather than guessing which it is this time.
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] capture LIVE: "
                        + $"{graphs.DecoderCaptured} - {graphs.DecoderCaptureStatus}");
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] WHOLE synthesis: "
            + $"{Graph.GraphExecutor.CumulativeRunCount} graph runs, "
            + $"{Graph.GraphExecutor.CumulativeReadbackCount} readbacks "
            + $"({Graph.GraphExecutor.CumulativeReadbackMs:F0} ms), "
            + $"{Graph.GraphExecutor.CumulativeSyncDrainCount} drains "
            + $"({Graph.GraphExecutor.CumulativeSyncDrainMs:F0} ms) "
            + "| a readback lets the executor compute host-side and ELIDE the dispatch, which a replayed "
            + "plan cannot reproduce - non-zero here is the reason to suspect a frozen value");
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] ONE decoder step: "
                        + $"{result.DecoderReadbacks} readbacks | {result.DecoderReadbackByOp}");
        int thenN = Operators.IfOperator.ThenBranchCount, elseN = Operators.IfOperator.ElseBranchCount;
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] If branches this "
                        + $"utterance: then={thenN} (1 node each) else={elseN} (254 nodes each) "
                        + $"=> {elseN * 254} extra node executions from the else branch");
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] host cost: "
                        + $"{Graph.GraphExecutor.LastRunReadbackCount} readbacks "
                        + $"({Graph.GraphExecutor.LastRunReadbackMs:F0} ms), "
                        + $"{Graph.GraphExecutor.LastRunSyncDrainCount} sync drains "
                        + $"({Graph.GraphExecutor.LastRunSyncDrainMs:F0} ms) on the LAST graph run");
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] stage split: "
                        + $"encoder {result.EncoderMs:F0} ms, decoder {result.DecoderMs:F0} ms, "
                        + $"vocoder {result.VocoderMs:F0} ms, total {result.TotalMs:F0} ms "
                        + $"| decoder is {100 * result.DecoderMs / Math.Max(1, result.TotalMs):F0}% of it");
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] decoder steps: "
                        + $"first {result.DecoderFirstStepMs:F0} ms, remaining "
                        + $"{result.DecoderRemainingStepsMs:F0} ms over the remaining "
                        + $"steps | {result.NumFrames} frames");

        // ── SPEAKING MORE THAN ONCE, which is what a conversation actually needs ─────────────────────
        //
        // ⚠️ WHY THIS IS PART OF THE GATE. Every Euler step runs at the same shape, so a first-step cost is
        // one-off FOR THAT SHAPE - and the decoder's shape IS the utterance length. A test that speaks one
        // line can never distinguish "the pipeline pays a one-off setup" from "the pipeline pays that setup
        // again for every new thing it says", and those are completely different products: the second one
        // cannot hold a conversation no matter how warm it is.
        //
        // MEASURED in the SpawnDev.AI demo, 2026-09-03, before this existed: a hands-free turn spent 172.5 s
        // in the voice - a background warm synthesis of "Hello." followed by a real reply of a DIFFERENT
        // length, each costing ~85 s. The warm bought nothing, which is precisely the reading this section
        // makes visible.
        //
        // Three renders: a different length, then a REPEAT of the first line. The repeat is the control -
        // if setup is cached per shape, the repeat is cheap; if nothing is cached, all three are equal.
        const string secondLine = "The quick brown fox jumps over the lazy dog while the kettle boils.";
        var second = await pipeline.SpeakAsync(secondLine, LibrivoxTranscript, reference, 16000, tokenizer);
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] utterance 2 (new length): "
                        + $"total {second.TotalMs:F0} ms | decoder first step {second.DecoderFirstStepMs:F0} ms, "
                        + $"remaining {second.DecoderRemainingStepsMs:F0} ms | {second.NumFrames} frames "
                        + $"| capture LIVE {second.DecoderCaptured}");

        var third = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] utterance 3 (repeat of 1): "
                        + $"total {third.TotalMs:F0} ms | decoder first step {third.DecoderFirstStepMs:F0} ms, "
                        + $"remaining {third.DecoderRemainingStepsMs:F0} ms | {third.NumFrames} frames "
                        + $"| capture LIVE {third.DecoderCaptured}");

        // ── CALIBRATE THE INSTRUMENT FIRST: is this pipeline even deterministic? ─────────────────────
        //
        // ⚠️ The capture A/B below concludes "replay changed the audio" from a sample comparison, and that
        // conclusion is only worth anything if two renders of the SAME line agree in the first place. If
        // the pipeline is non-deterministic, that comparison would report a difference no matter what
        // capture did - a null result from an uncalibrated instrument, which is worse than no result
        // because it is believable.
        //
        // The ODE starts from a SEEDED noise vector and every graph is deterministic, so this should hold;
        // asserting it is what makes the capture verdict admissible rather than assumed.
        graphs.EnableGraphCapture = false;
        var det1 = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
        var det2 = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
        graphs.EnableGraphCapture = true;
        int detDiff = 0; float detWorst = 0f;
        if (det1.Audio.Length != det2.Audio.Length)
            throw new Exception($"two uncaptured renders of the same line differ in LENGTH "
                              + $"({det1.Audio.Length} vs {det2.Audio.Length}) - the pipeline is not "
                              + "deterministic, so no sample-level comparison here means anything");
        for (int i = 0; i < det1.Audio.Length; i++)
        {
            float d = MathF.Abs(det1.Audio[i] - det2.Audio[i]);
            if (d != 0f) { detDiff++; detWorst = MathF.Max(detWorst, d); }
        }
        Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] determinism: "
                        + $"{detDiff} of {det1.Audio.Length} samples differ between two uncaptured renders "
                        + $"(worst {detWorst:F6})");
        if (detDiff != 0)
            throw new Exception($"the pipeline is NOT deterministic: two uncaptured renders of the same "
                              + $"line differ in {detDiff} of {det1.Audio.Length} samples (worst "
                              + $"{detWorst:F6}). Fix that before reading anything into the capture A/B - "
                              + "a comparison against a moving target cannot indict capture.");

        // ── CAPTURE MUST NOT CHANGE THE AUDIO ────────────────────────────────────────────────────────
        //
        // ⚠️ THE FAILURE THIS CATCHES CANNOT BE HEARD. A recorded plan can ELIDE the dispatch that fills a
        // small tensor, promoting it to a runtime constant frozen at its capture-time value - which then
        // never updates again. The Euler timestep `t` is exactly such a tensor. The result is not silence
        // and not noise: it is confident, plausible speech that is subtly wrong, so every other assertion
        // in this test would pass it. Only comparing samples against a render that did NOT replay a plan
        // can tell.
        //
        // Utterance 3 is re-rendered with capture off. Same text, same reference, same noise seed, so the
        // two renders must agree EXACTLY - the ODE is deterministic and a replay is meant to be the same
        // arithmetic in a different envelope, not an approximation of it.
        if (third.DecoderCaptured)
        {
            graphs.EnableGraphCapture = false;
            var control = await pipeline.SpeakAsync(line, LibrivoxTranscript, reference, 16000, tokenizer);
            graphs.EnableGraphCapture = true;

            if (control.Audio.Length != third.Audio.Length)
                throw new Exception(
                    $"replayed capture produced {third.Audio.Length} samples, the direct forward "
                  + $"{control.Audio.Length}. A capture must reproduce the graph, not reshape it.");

            int differing = 0;
            float worst = 0f;
            for (int i = 0; i < control.Audio.Length; i++)
            {
                float d = MathF.Abs(control.Audio[i] - third.Audio[i]);
                if (d != 0f) { differing++; worst = MathF.Max(worst, d); }
            }
            if (differing != 0)
                throw new Exception(
                    $"replaying the captured decoder changed the audio: {differing} of "
                  + $"{control.Audio.Length} samples differ, worst {worst:F6}. The likeliest cause is a "
                  + "dispatch elided into a capture-time constant - the Euler timestep is the obvious "
                  + "candidate - which produces plausible speech and would pass every other check here.");

            Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] capture A/B: "
                            + $"replay is BIT-IDENTICAL to the direct forward over {control.Audio.Length} "
                            + $"samples | direct {control.TotalMs:F0} ms vs replayed {third.TotalMs:F0} ms");
        }
        else
        {
            Console.WriteLine($"[Benchmark] ZipVoice [{accelerator.AcceleratorType}] capture A/B SKIPPED: "
                            + $"capture is not live ({graphs.DecoderCaptureStatus})");
        }

        // Both later renders must still be SPEECH, not just fast. A shape-caching change that returns
        // stale or silent audio would otherwise look like a win.
        foreach (var (label, r) in new[] { ("utterance 2", second), ("utterance 3", third) })
        {
            var a = r.Audio;
            if (a == null || a.Length == 0)
                throw new Exception($"ZipVoice {label} produced NO audio.");
            float p = 0f; double e = 0;
            foreach (var v in a) { p = MathF.Max(p, MathF.Abs(v)); e += (double)v * v; }
            if (p < 0.01f || Math.Sqrt(e / a.Length) < 0.005)
                throw new Exception($"ZipVoice {label} is effectively silence (peak {p:F5}) - speaking a "
                                  + "second time must still produce speech, not just return quickly.");
        }
    });
}
