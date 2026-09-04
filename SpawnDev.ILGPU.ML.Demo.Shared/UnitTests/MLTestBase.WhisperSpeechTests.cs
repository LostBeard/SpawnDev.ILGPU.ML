using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end speech-to-TEXT for the /whisper demo: real audio in, real words out.
///
/// <para>
/// WHY THIS FILE EXISTS: `/whisper`'s only cited evidence was
/// <c>Pipeline_WhisperDecoder_Reference_440HzTone</c>, which never runs the model. That test loads
/// <c>decoder_reference_tone_440hz.json</c> and asserts the JSON is internally consistent - the prefix is
/// 4 tokens, the sequence ends with EOT, <c>steps[0].next_token == generated[0]</c>. Every one of those
/// assertions is about the fixture, not about our decoder, so the test passes unchanged if the entire ML
/// library is deleted. It is renamed <c>ReferenceData_WhisperDecoder_FixtureIsWellFormed</c> to say what it
/// really does, and this file supplies the missing thing.
/// </para>
///
/// <para>
/// The audio is the harness reference clip, chosen because its transcript is KNOWN text rather than
/// something that was itself transcribed (see <c>wwwroot/test-audio/librivox-PROVENANCE.md</c>):
/// <c>"All LibriVox recordings are in the public domain."</c> - 16 kHz mono, 4.0 s, Public Domain Mark 1.0.
/// A transcription test whose expected value came out of a transcriber proves only that two transcribers
/// agree.
/// </para>
///
/// <para>
/// The assertion is on CONTENT WORDS plus a 70% word-overlap floor, not an exact string: Whisper picks its
/// own casing and punctuation, and "public domain" vs "public-domain" is not a defect. It is still a real
/// assertion - no other audio produces those words, and a broken mel/encoder/decoder produces none of them.
/// </para>
///
/// <para>
/// MEASURED 2026-08-30, identical on all six backends (CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm):
/// <c>"All legal box recordings are in the public domain."</c> - seven of eight words exact. whisper-tiny
/// mangles the proper noun "LibriVox", which is a model capability limit, not a pipeline defect, so the
/// assertion does not depend on it. The cross-backend agreement is itself evidence: six independent kernel
/// implementations producing the same token stream is not something a broken pipeline does.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Pipeline_Whisper_TranscribesKnownSpeech() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        // ── the audio, with a transcript we know independently ──────────────────────────────────────
        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // 4.0 s at 16 kHz. If this drifts, the fixture changed and the expected words may no longer apply.
        if (samples.Length < 16000 || samples.Length > 16000 * 30)
            throw new Exception($"fixture is {samples.Length / 16000.0:F1}s - expected about 4s");

        // ── the model, loaded the way the page loads it ─────────────────────────────────────────────
        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);
        var repo = ModelHub.KnownModels.WhisperTiny;

        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/encoder_model.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/decoder_model.onnx"));

        using var encoder = InferenceSession.CreateFromFile(accelerator, encoderBytes);
        using var decoder = InferenceSession.CreateFromFile(accelerator, decoderBytes);
        using var pipeline = new SpeechRecognitionPipeline(encoder, decoder, accelerator);

        var tokenizerJson = Encoding.UTF8.GetString(await hf.DownloadFileAsync(repo, "tokenizer.json"));
        pipeline.LoadTokenizer(tokenizerJson);

        // ── transcribe ──────────────────────────────────────────────────────────────────────────────
        var result = await pipeline.TranscribeAsync(samples, 16000);
        var got = result.Text ?? "";
        Console.WriteLine($"[Whisper] transcribed: \"{got}\" ({result.InferenceTimeMs:F0} ms)");

        if (string.IsNullOrWhiteSpace(got))
            throw new Exception("transcription is empty");

        var norm = Regex.Replace(got.ToLowerInvariant(), @"[^a-z0-9 ]", " ");
        norm = Regex.Replace(norm, @"\s+", " ").Trim();

        // ⚠️ NOT asserting "librivox". MEASURED on all six backends, whisper-tiny returns
        //     "All legal box recordings are in the public domain."
        // Seven of eight words are exact and every backend agrees byte for byte; the model simply mangles
        // an unusual PROPER NOUN, which is a whisper-tiny capability limit and not a defect in the mel,
        // the encoder, the decoder or the tokenizer. Asserting on it would test the model's vocabulary
        // rather than our pipeline. Scoping the assertion to the words the model reliably produces is
        // correct scoping - and to stop that becoming an excuse for a weak test, the overlap floor below
        // holds the WHOLE sentence to account.
        var expected = new[] { "recordings", "are", "in", "the", "public", "domain" };
        var missing = expected.Where(w => !norm.Contains(w)).ToArray();
        if (missing.Length > 0)
            throw new Exception(
                $"transcript is missing {string.Join(", ", missing)}. "
                + $"Expected the words of \"All LibriVox recordings are in the public domain.\", got \"{got}\"");

        // Word-level overlap against the full reference. This is what catches a REGRESSION: the failure
        // mode when Whisper is fed bad audio is fluent, confident, entirely unrelated text (a chopped mic
        // recording produced "middle they did like a pepperoni a little bit of sauce..."), which scores
        // near zero here while satisfying any individual keyword check you might have written.
        var refWords = "all librivox recordings are in the public domain".Split(' ');
        var gotWords = norm.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        double overlap = refWords.Count(w => gotWords.Contains(w)) / (double)refWords.Length;
        Console.WriteLine($"[Whisper] word overlap with the known transcript: {overlap:P0}");
        if (overlap < 0.7)
            throw new Exception(
                $"only {overlap:P0} of the known transcript's words appear (floor is 70%). "
                + $"Expected \"All LibriVox recordings are in the public domain.\", got \"{got}\"");
    });

    /// <summary>
    /// The KV-cache decoder must be an OPTIMISATION, not a second implementation with its own answers.
    /// <c>SpeechRecognitionPipeline</c> takes an optional <c>decoder_with_past</c> session and switches to
    /// <c>DecodeWithKVCacheAsync</c>, a genuinely different code path: without it every step re-feeds the
    /// whole token prefix and decoding is quadratic. Speed is the only thing that may differ, so this runs
    /// the same audio through both and requires the transcripts to match EXACTLY.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Pipeline_Whisper_KVCacheDecode_MatchesFullDecode() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        using var http = CreateHuggingFaceHttpClient();
        var hf = new HuggingFaceClient(http);
        var repo = ModelHub.KnownModels.WhisperTiny;

        var encoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/encoder_model.onnx"));
        var decoderBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/decoder_model.onnx"));
        var withPastBytes = await InferenceSession.DownloadBytesChunkedAsync(
            http, HuggingFaceClient.GetDownloadUrl(repo, "onnx/decoder_with_past_model.onnx"));

        var tokenizerJson = Encoding.UTF8.GetString(await hf.DownloadFileAsync(repo, "tokenizer.json"));

        string full, cached;

        // Full decode: no with-past session, so every step re-feeds the prefix.
        using (var enc = InferenceSession.CreateFromFile(accelerator, encoderBytes))
        using (var dec = InferenceSession.CreateFromFile(accelerator, decoderBytes))
        using (var pipeline = new SpeechRecognitionPipeline(enc, dec, accelerator))
        {
            pipeline.LoadTokenizer(tokenizerJson);
            if (pipeline.UsesKVCache)
                throw new Exception("expected the full-decode pipeline to report UsesKVCache == false");
            full = ((await pipeline.TranscribeAsync(samples, 16000)).Text ?? "").Trim();
        }

        // KV-cache decode: the same audio down the DecodeWithKVCacheAsync path.
        using (var enc = InferenceSession.CreateFromFile(accelerator, encoderBytes))
        using (var dec = InferenceSession.CreateFromFile(accelerator, decoderBytes))
        using (var past = InferenceSession.CreateFromFile(accelerator, withPastBytes))
        using (var pipeline = new SpeechRecognitionPipeline(enc, dec, accelerator, past))
        {
            pipeline.LoadTokenizer(tokenizerJson);
            if (!pipeline.UsesKVCache)
                throw new Exception("a decoder_with_past session was supplied but UsesKVCache is false - "
                                  + "the KV path is not actually being taken, so this test proves nothing");
            // ⚠️ CUMULATIVE counters around the whole transcription, so this test reports the SAME split
            // the SpawnDev.AI demo prints. MEASURED 2026-09-03: the demo's decode step costs 900 ms where
            // this test's costs 400 ms for the byte-identical compiled graph (enc 227 / dec 374), and the
            // window-vs-worker hypothesis was DISPROVEN by measurement (managed 0.92x, JS crossings 1.09x,
            // scheduler 0.58-1.00x - the worker is not slower at any of them). Without the readback and
            // drain counts on BOTH sides there is nothing left to compare but the total, which is how that
            // wrong hypothesis survived a day.
            Graph.GraphExecutor.CumulativeReset();
            var kvResult = await pipeline.TranscribeAsync(samples, 16000);
            cached = (kvResult.Text ?? "").Trim();
            Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] host cost: "
                + $"{Graph.GraphExecutor.CumulativeRunCount} graph runs, "
                + $"{Graph.GraphExecutor.CumulativeReadbackCount} readbacks "
                + $"({Graph.GraphExecutor.CumulativeReadbackMs:F0} ms), "
                + $"{Graph.GraphExecutor.CumulativeSyncDrainCount} drains "
                + $"({Graph.GraphExecutor.CumulativeSyncDrainMs:F0} ms), "
                + $"executor {Graph.GraphExecutor.CumulativeTotalMs:F0} ms");

            Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] kv split: "
                + $"encoder {kvResult.EncoderMs:F0}ms ({kvResult.EncoderCaptureStatus}) | "
                + $"prefill {kvResult.PrefillMs:F0}ms | {kvResult.DecodeSteps} decode steps "
                + $"{kvResult.DecodeStepsMs:F0}ms | mel {kvResult.MelTimeMs:F0}ms");
            Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] COMPILED nodes: "
                + $"encoder {kvResult.EncoderNodeCount}, decode step {kvResult.DecoderNodeCount} "
                + "(from the session, not the offline probe)");
        Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] per decode step: "
                + $"setup {kvResult.DecodeSetupMs / Math.Max(1, kvResult.DecodeSteps):F1}ms + graph "
                + $"{kvResult.DecodeGraphMs / Math.Max(1, kvResult.DecodeSteps):F1}ms + argmax "
                + $"{kvResult.DecodeArgmaxMs / Math.Max(1, kvResult.DecodeSteps):F1}ms");

            // ── WHICH OPERATORS the decode step actually spends its time in ─────────────────────────
            //
            // ⚠️ MEASURED in the SpawnDev.AI demo 2026-09-03: a decode step of whisper-TINY, producing ONE
            // token, cost 1,132 ms - setup 0.5 ms, graph 1,061.7 ms, argmax 69.8 ms. So it is neither host
            // bookkeeping nor the argmax round trip; effectively all of it is inside the graph walk. This
            // section names WHICH nodes, because "the graph is slow" is not something anyone can act on.
            //
            // ⚠️ PerOpSync is REQUIRED for this to mean anything. Without it the timing is sync-blocking:
            // async kernel work surfaces at the next sync point rather than at its real producer, so the
            // profile names an innocent node. It makes the run slower, which is the correct trade for
            // attribution - the number to act on is the RANKING, not this run's wall time.
            // ⚠️ WebGPU and CUDA ONLY. PerOpSync forces a full device sync after every node, so this is a
            // THIRD transcription at a deliberately pessimal cost - and on the CPU and OpenCL lanes that
            // blew the test's 900 s budget outright (MEASURED: both timed out while the four other backends
            // passed). The question this profile answers - where per-node dispatch time goes - is a
            // question about the dispatch-bound backends anyway; on CPU there is no dispatch to attribute.
            //
            // ⚠️ An `if` and not an early return: the transcripts are compared AFTER this using-block, so
            // returning here would skip the actual assertion and leave CPU and OpenCL passing vacuously.
            bool profile = accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.Cuda;
            if (!profile)
                Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] node timing SKIPPED "
                                + "(PerOpSync is a third full transcription; WebGPU/CUDA only)");
            if (profile)
            {
            Graph.GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
            Graph.GraphExecutor.PerOpSync = true;
            try { await pipeline.TranscribeAsync(samples, 16000); }
            finally { Graph.GraphExecutor.PerOpSync = false; }

            var timings = Graph.GraphExecutor.CapturedNodeTimingsMs ?? new Dictionary<string, double>();
            Graph.GraphExecutor.CapturedNodeTimingsMs = null;

            // Aggregated by OP TYPE. Per-node keys are too many to read and the actionable question is
            // which KIND of operator dominates - one expensive op type is a kernel to fix, while a flat
            // spread across every type is per-node dispatch overhead and needs a recorded plan instead.
            var byOp = timings
                .Select(kv => (Op: kv.Key.Split('_') is var p && p.Length > 1 ? p[1] : "?", Ms: kv.Value))
                .GroupBy(x => x.Op)
                .Select(g => (Op: g.Key, Total: g.Sum(x => x.Ms), Count: g.Count()))
                .OrderByDescending(x => x.Total)
                .ToList();
            double grand = byOp.Sum(x => x.Total);
            Console.WriteLine($"[Benchmark] Whisper [{accelerator.AcceleratorType}] node timing "
                + $"(PerOpSync, {timings.Count} nodes, {grand:F0}ms total):");
            foreach (var (op, total, count) in byOp.Take(12))
                Console.WriteLine($"[Benchmark]   {op,-22} {total,8:F0}ms  {count,5} nodes  "
                    + $"{total / Math.Max(1, count),6:F2}ms/node  {100 * total / Math.Max(1, grand),5:F1}%");
            }
        }

        Console.WriteLine($"[Whisper] full   : \"{full}\"");
        Console.WriteLine($"[Whisper] kvcache: \"{cached}\"");

        if (full.Length == 0)
            throw new Exception("full decode produced nothing, so there is nothing to compare against");
        if (!string.Equals(full, cached, StringComparison.Ordinal))
            throw new Exception(
                "KV-cache decode disagrees with full decode. "
                + $"full=[{full}] kvcache=[{cached}]");
    });

}
