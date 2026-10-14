using System;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The two speech pipelines that sit under every voice feature: <see cref="ParakeetTdtPipeline"/> (what was said) and
/// <see cref="SpeakerEmbeddingPipeline"/> (who said it). Each is held to onnxruntime on the SAME audio.
/// </summary>
public abstract partial class MLTestBase
{
    private const string ParakeetRepo = "istupakov/parakeet-tdt-0.6b-v3-onnx";
    private const string SpeakerRepo = "soniqo/ReDimNet2-B6-ONNX-FP32";

    /// <summary>
    /// Parakeet TDT 0.6B v3 transcribes the harness clip exactly as onnxruntime does, from all three entry points of the
    /// pipeline. Desktop CUDA / OpenCL only: the fp32 encoder is 2.4 GB (external data), which is not a browser load, and a CPU
    /// run of 0.6B parameters is not a unit test.
    /// </summary>
    /// <remarks>
    /// The expected text is what onnxruntime-node produced from the same three ONNX graphs (tools/parakeet-oracle.mjs):
    /// "All Lieberbach's recordings are in the public domain." - the proper noun is mangled by the model, in both runtimes
    /// alike, which is the point: this pins OUR pipeline to the reference, not the model to the truth. On the 60-utterance
    /// LibriSpeech bench (tools/stt-bench) the pipeline scores 3.75% WER against Whisper tiny's 11.53%.
    /// </remarks>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Pipeline_ParakeetTdt_MatchesOnnxRuntimeTranscript() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: the 2.4 GB fp32 Parakeet encoder is desktop GPU only");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var samples = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav"))
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");

        // The default HttpClient timeout is 100 s: a 2.4 GB first download through the hub takes longer, so it gets its own client.
        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var source = new HttpClientModelSource(modelHttp);
        async Task<InferenceSession> Open(string file, string? dataFile = null)
        {
            await using var graph = await source.OpenAsync(ParakeetRepo, file);
            if (dataFile == null)
                return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph);
            await using var data = await source.OpenAsync(ParakeetRepo, dataFile);
            return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph, externalDataStream: data);
        }

        var pre = await Open("nemo128.onnx");
        var enc = await Open("encoder-model.onnx", "encoder-model.onnx.data");
        var dec = await Open("decoder_joint-model.onnx");
        string vocabText;
        await using (var v = await source.OpenAsync(ParakeetRepo, "vocab.txt"))
        using (var reader = new System.IO.StreamReader(v))
            vocabText = await reader.ReadToEndAsync();
        using var pipeline = new ParakeetTdtPipeline(pre, enc, dec, accelerator, ParakeetTdtPipeline.ParseVocabulary(vocabText));

        var viaArray = await pipeline.TranscribeAsync(samples, 16000);
        Console.WriteLine($"[Parakeet] \"{viaArray.Text}\" in {viaArray.InferenceTimeMs:F0} ms (mel {viaArray.MelTimeMs:F0}, encoder {viaArray.EncoderMs:F0}, decode {viaArray.DecodeStepsMs:F0} over {viaArray.DecodeSteps} steps)");
        const string expected = "All Lieberbach's recordings are in the public domain.";
        if (viaArray.Text != expected)
            throw new Exception($"transcript differs from onnxruntime's.\n  expected: \"{expected}\"\n  got:      \"{viaArray.Text}\"");

        // The device entry point (the one the others funnel into) must give the same answer.
        using var pcm = accelerator.Allocate1D(samples);
        var viaDevice = await pipeline.TranscribeAsync(pcm.View, samples.Length);
        if (viaDevice.Text != expected)
            throw new Exception($"the device-resident overload differs: \"{viaDevice.Text}\"");

        // 48 kHz audio is resampled on the host first and must still be recognised (voice chat is 48 kHz).
        var up = new float[samples.Length * 3];
        for (int i = 0; i < up.Length; i++) up[i] = samples[Math.Min(samples.Length - 1, i / 3)];
        var viaResample = await pipeline.TranscribeAsync(up, 48000);
        var words = new[] { "recordings", "public", "domain" };
        if (!words.All(w => viaResample.Text.Contains(w, StringComparison.OrdinalIgnoreCase)))
            throw new Exception($"48 kHz input lost the speech: \"{viaResample.Text}\"");

        // A pause too short to be speech returns nothing instead of running the models.
        var nothing = await pipeline.TranscribeAsync(new float[100], 16000);
        if (nothing.Text.Length != 0) throw new Exception($"100 samples of silence produced \"{nothing.Text}\"");
    });

    /// <summary>
    /// ReDimNet2-B6 speaker embeddings: each of three real LibriSpeech clips embeds to the onnxruntime vector (cosine &gt;= 0.9999),
    /// two clips of ONE speaker are close and a clip of ANOTHER speaker is far.
    /// </summary>
    /// <remarks>
    /// Measured on these clips with onnxruntime: same speaker 0.565, different speakers 0.194 and 0.111 (the 40-speaker bench:
    /// closed-set 99.2%, EER 1.3%). The model's input is a fixed 6 s window, so the 2.7-3.1 s clips are repeat-filled by the
    /// pipeline - exactly what the fixture embeddings did. Fast lanes only (CUDA, OpenCL, WebGPU): 1,770 graph nodes is not a
    /// unit test on the CPU, WebGL or Wasm backends. A 12 s clip is also embedded (centre-cropped) to exercise the long path.
    /// </remarks>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Pipeline_SpeakerEmbedding_MatchesOnnxRuntimeAndSeparatesVoices() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: speaker embedding is gated on the fast backends");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        using var fixture = JsonDocument.Parse(await assets.GetStringAsync("test-audio/librispeech-speaker-fixture.json"));
        var root = fixture.RootElement;
        var reference = root.GetProperty("clips").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetSingle()).ToArray());

        using var http = CreateHuggingFaceHttpClient();
        var modelBytes = await InferenceSession.DownloadBytesChunkedAsync(http, HuggingFaceClient.GetDownloadUrl(SpeakerRepo, "ReDimNet2B6.onnx"));
        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes);
        using var pipeline = new SpeakerEmbeddingPipeline(session, accelerator);

        var ours = new System.Collections.Generic.Dictionary<string, float[]>();
        foreach (var name in reference.Keys)
        {
            var samples = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync("test-audio/" + name))
                ?? throw new Exception("could not decode " + name);
            ours[name] = await pipeline.EmbedAsync(samples, 16000);
            var c = SpeakerEmbeddingPipeline.Cosine(ours[name], reference[name]);
            Console.WriteLine($"[Speaker] {name}: cosine vs onnxruntime {c:F6}");
            if (c < 0.9999f)
                throw new Exception($"{name}: our embedding is not onnxruntime's (cosine {c:F6}, need >= 0.9999)");
            var norm = MathF.Sqrt(ours[name].Sum(v => v * v));
            if (MathF.Abs(norm - 1f) > 1e-3f) throw new Exception($"{name}: embedding length {norm:F4}, expected 1 (L2-normalised)");
        }

        var same = root.GetProperty("sameSpeaker").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var diff = root.GetProperty("differentSpeaker").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var sameScore = SpeakerEmbeddingPipeline.Cosine(ours[same[0]], ours[same[1]]);
        var diffScore = SpeakerEmbeddingPipeline.Cosine(ours[diff[0]], ours[diff[1]]);
        Console.WriteLine($"[Speaker] same speaker {sameScore:F3}, different speaker {diffScore:F3}");
        if (sameScore < 0.45f) throw new Exception($"two clips of one speaker scored only {sameScore:F3} (measured 0.565)");
        if (diffScore > 0.30f) throw new Exception($"clips of two speakers scored {diffScore:F3} (measured 0.194)");
        if (sameScore - diffScore < 0.25f) throw new Exception($"voices barely separated: same {sameScore:F3} vs different {diffScore:F3}");

        // Long audio is centre-cropped to the 6 s window: speaker A's two clips played twice over (~11.6 s) must embed closer
        // to A than to speaker B.
        var a = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync("test-audio/" + same[0]))!;
        var b = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync("test-audio/" + same[1]))!;
        var joined = a.Concat(b).Concat(a).Concat(b).ToArray();    // > 6 s
        var longEmbedding = await pipeline.EmbedAsync(joined, 16000);
        var toSame = SpeakerEmbeddingPipeline.Cosine(longEmbedding, ours[same[0]]);
        var toOther = SpeakerEmbeddingPipeline.Cosine(longEmbedding, ours[diff[1]]);
        Console.WriteLine($"[Speaker] {joined.Length / 16000.0:F1} s of speaker A: to A {toSame:F3}, to B {toOther:F3}");
        if (toSame <= toOther + 0.2f)
            throw new Exception($"a long clip of speaker A is not closer to A ({toSame:F3}) than to B ({toOther:F3})");
    });
}
