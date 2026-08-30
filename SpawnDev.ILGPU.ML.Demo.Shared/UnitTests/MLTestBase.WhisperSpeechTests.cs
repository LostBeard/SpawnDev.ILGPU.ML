using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

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
            cached = ((await pipeline.TranscribeAsync(samples, 16000)).Text ?? "").Trim();
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
