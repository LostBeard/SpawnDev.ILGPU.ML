using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    private const string NemotronStreamingRepo = "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-2026-04-25";

    /// <summary>
    /// Nemotron Speech Streaming 0.6B (160 ms export) turns audio into text AS IT ARRIVES: fed in 100 ms pieces like a microphone,
    /// words appear while the clip is still being spoken, and the final text is sherpa-onnx's (onnxruntime on CPU) for the same audio.
    /// Desktop CUDA / OpenCL only: the fp32 encoder is 2.4 GB of external data.
    /// </summary>
    /// <remarks>
    /// Expected texts are what sherpa-onnx 1.x printed for these clips with the same ONNX files. The engine is held to them on the
    /// 60-clip LibriSpeech bench too (tools/stt-bench ... streaming: 5.63% WER; sherpa-onnx 6.17%). A defect in a broadcast
    /// (Sub of [72,1] and [1,72]) once raised that to 7.24% without any error, so the text is compared exactly, not by keywords.
    /// </remarks>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Pipeline_StreamingTranscriber_MatchesSherpaAndStreams() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: the 2.4 GB fp32 streaming encoder is desktop GPU only");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var source = new HttpClientModelSource(modelHttp);
        var (encShapes, decShapes, joiShapes) = StreamingTranscriber.InputShapes();
        async Task<InferenceSession> Open(string file, Dictionary<string, int[]> shapes, string? dataFile = null)
        {
            await using var graph = await source.OpenAsync(NemotronStreamingRepo, file);
            if (dataFile == null) return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph, inputShapes: shapes);
            await using var data = await source.OpenAsync(NemotronStreamingRepo, dataFile);
            return await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph, inputShapes: shapes, externalDataStream: data);
        }
        var enc = await Open("encoder.onnx", encShapes, "encoder.data");
        var dec = await Open("decoder.onnx", decShapes);
        var joi = await Open("joiner.onnx", joiShapes);
        string tokensText;
        await using (var t = await source.OpenAsync(NemotronStreamingRepo, "tokens.txt"))
        using (var reader = new System.IO.StreamReader(t))
            tokensText = await reader.ReadToEndAsync();
        using var stream = new StreamingTranscriber(enc, dec, joi, accelerator, ParakeetTdtPipeline.ParseVocabulary(tokensText));

        var clips = new[]
        {
            ("test-audio/librivox-public-domain.wav", "All LibriVox recordings are in the public domain"),
            ("test-audio/librispeech-1089_1089-134686-0003.wav", "Hello Bertie, any good in your mind"),
        };
        foreach (var (path, expected) in clips)
        {
            var samples = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync(path)) ?? throw new Exception($"could not decode {path}");
            stream.Reset();
            long fed = 0, firstTokenAt = -1;
            for (int i = 0; i < samples.Length; i += 1600)
            {
                var piece = samples[i..Math.Min(samples.Length, i + 1600)];
                var u = await stream.AcceptAudioAsync(piece);
                fed += piece.Length;
                if (firstTokenAt < 0 && u.NewTokens.Count > 0) firstTokenAt = fed;
            }
            string whileSpeaking = stream.Text;
            var fin = await stream.FinishAsync();
            Console.WriteLine($"[StreamingTranscriber] {path}: \"{fin.Text}\"; first token after {firstTokenAt / 16} ms of {samples.Length / 16} ms of audio, {whileSpeaking.Length} characters before the flush");
            if (fin.Text != expected)
                throw new Exception($"{path}: transcript differs from sherpa-onnx's.\n  expected: \"{expected}\"\n  got:      \"{fin.Text}\"");
            if (firstTokenAt < 0 || firstTokenAt >= samples.Length)
                throw new Exception($"{path}: no word was produced before the audio ended (first token at {firstTokenAt} of {samples.Length} samples) - this is not streaming");
        }

        // The same stream, Reset and fed one SAMPLE-MISALIGNED way (not a multiple of the 160 ms chunk), must give the same text.
        var again = WavDecoder.DecodeWavFile(await assets.GetByteArrayAsync(clips[0].Item1))!;
        stream.Reset();
        for (int i = 0; i < again.Length; i += 777) await stream.AcceptAudioAsync(again[i..Math.Min(again.Length, i + 777)]);
        var odd = await stream.FinishAsync();
        if (odd.Text != clips[0].Item2) throw new Exception($"pieces of 777 samples after a Reset gave \"{odd.Text}\"");
    });
}
