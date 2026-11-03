using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    // tools/clap-reference/make_fixtures.py's clips, regenerated in the same order of float64 operations, then cast to float32 as numpy does
    private static float[] ClapFixtureClip(string kind, int n, double seconds)
    {
        const int SR = 48000;
        var u = new double[n];
        long state = 1234;
        for (int i = 0; i < n; i++) { state = (state * 1103515245 + 12345) & 0x7FFFFFFF; u[i] = state / 1073741824.0 - 1.0; }
        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / SR;
            x[i] = kind switch
            {
                "tone" => 0.5 * Math.Sin(2 * Math.PI * 440 * t) + 0.2 * Math.Sin(2 * Math.PI * 2000 * t),
                "clicks" => 0.01 * u[i],
                "noise" => 0.3 * u[i],
                "chirp" => 0.4 * Math.Sin(2 * Math.PI * (200.0 * t + (8000.0 - 200.0) / (2 * seconds) * t * t)),
                _ => throw new ArgumentException(kind),
            };
        }
        if (kind == "clicks")
            for (int c = 0; c < n; c += SR / 4)
            {
                int m = Math.Min(n - c, 480);
                for (int k = 0; k < m; k++) x[c + k] += Math.Exp(-k / 60.0) * 0.9;
            }
        return x.Select(v => (float)v).ToArray();
    }

    /// <summary>
    /// The CLAP towers (Xenova/clap-htsat-unfused ONNX: audio_model 8,031 nodes, text_model 1,382; from models.spawndev.com) on InferenceSession against
    /// transformers' ClapModel (laion/clap-htsat-unfused): the audio tower on the fixture's own input_features (cosine &gt;= 0.9999 to its
    /// audio_embeds) and end to end from our <see cref="ClapFeatureExtractor"/> (cosine &gt;= 0.999); the text tower on the fixture's input_ids (cosine
    /// &gt;= 0.9999 to its text_embeds); the logits (logit_scale x audio . text, both L2-normalised) within 0.05 of the fixture's with the same best
    /// prompt. Fast lanes (CUDA, OpenCL, WebGPU). Added 2026-11-03.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Clap_Towers_MatchTransformers() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        var root = doc.RootElement;
        using var http = CreateHuggingFaceHttpClient();
        http.Timeout = TimeSpan.FromMinutes(20);
        string Url(string f) => "https://models.spawndev.com/Xenova/clap-htsat-unfused/resolve/main/onnx/" + f;
        static float[] Unit(float[] v) { double n = Math.Sqrt(v.Sum(x => (double)x * x)); return v.Select(x => (float)(x / n)).ToArray(); }
        static double Cos(float[] a, float[] b) { double d = 0, na = 0, nb = 0; for (int i = 0; i < a.Length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.Sqrt(na * nb); }
        async Task<float[]> ReadOut(Tensor t, int n)
        {
            await accelerator.SynchronizeAsync();
            using var host = accelerator.Allocate1D<float>(n);
            host.View.CopyFrom(t.Data.SubView(0, n));
            await accelerator.SynchronizeAsync();
            return await host.CopyToHostAsync<float>(0, n);
        }
        var notes = new List<string>();
        // text tower
        var textEmbeds = new List<float[]>();
        {
            var bytes = await InferenceSession.DownloadBytesChunkedAsync(http, Url("text_model.onnx"));
            using var text = InferenceSession.CreateFromFile(accelerator, bytes);
            var wantText = root.GetProperty("text_embeds").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
            int p = 0; double worst = 1;
            foreach (var ids in root.GetProperty("input_ids").EnumerateArray())
            {
                var idv = ids.EnumerateArray().Select(v => (float)v.GetInt64()).ToArray();
                using var idBuf = accelerator.Allocate1D(idv);
                var outs = await text.RunAsync(new Dictionary<string, Tensor> { [text.InputNames[0]] = new Tensor(idBuf.View, new[] { 1, idv.Length }) });
                var e = Unit(await ReadOut(outs[text.OutputNames[0]], 512));
                textEmbeds.Add(e);
                worst = Math.Min(worst, Cos(e, wantText[p++]));
            }
            if (worst < 0.9999) throw new Exception($"text tower: worst cosine to transformers' text_embeds {worst:F6}");
            notes.Add($"text {p} prompts cosine >= {worst:F6}");
        }
        // audio tower: on the fixture's features, and end to end from ours
        {
            var bytes = await InferenceSession.DownloadBytesChunkedAsync(http, Url("audio_model.onnx"));
            using var audio = InferenceSession.CreateFromFile(accelerator, bytes);
            using var fx = new ClapFeatureExtractor(accelerator);
            float logitScale = root.GetProperty("logit_scale").GetSingle();
            using var feat = accelerator.Allocate1D<float>(ClapFeatureExtractor.Frames * ClapFeatureExtractor.Mels);
            foreach (var c in root.GetProperty("clips").EnumerateArray())
            {
                string kind = c.GetProperty("kind").GetString()!;
                var wantAudio = c.GetProperty("audio_embeds").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                var wantLogits = c.GetProperty("logits_per_audio").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                var fixtureFeat = c.GetProperty("input_features").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetSingle())).ToArray();
                async Task<float[]> Embed(Tensor input)
                {
                    var outs = await audio.RunAsync(new Dictionary<string, Tensor> { [audio.InputNames[0]] = input });
                    return Unit(await ReadOut(outs[audio.OutputNames[0]], 512));
                }
                using var fxBuf = accelerator.Allocate1D(fixtureFeat);
                var shape = new[] { 1, 1, ClapFeatureExtractor.Frames, ClapFeatureExtractor.Mels };
                var a1 = await Embed(new Tensor(fxBuf.View, shape));
                double c1 = Cos(a1, wantAudio);
                // end to end: our features
                var pcm = ClapFixtureClip(kind, c.GetProperty("samples").GetInt32(), c.GetProperty("seconds").GetDouble());
                using var pcmBuf = accelerator.Allocate1D(pcm);
                fx.Extract(pcmBuf.View, pcm.Length, feat.View);
                var a2 = await Embed(new Tensor(feat.View, shape));
                double c2 = Cos(a2, wantAudio);
                var logits = textEmbeds.Select(t => logitScale * a2.Zip(t).Sum(z => z.First * z.Second)).ToArray();
                double worstLogit = logits.Zip(wantLogits).Max(z => Math.Abs(z.First - z.Second));
                int best = Array.IndexOf(logits, logits.Max()), wantBest = Array.IndexOf(wantLogits, wantLogits.Max());
                if (c1 < 0.9999 || c2 < 0.999 || worstLogit > 0.05 || best != wantBest)
                    throw new Exception($"{kind}: audio tower cosine {c1:F6} on the fixture's features, {c2:F6} end to end; logits off by {worstLogit:F4}; best prompt {best} vs {wantBest}");
                notes.Add($"{kind}: cosine {c1:F6} / {c2:F6} end to end, logits within {worstLogit:F4}");
            }
        }
        return $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
    });

    /// <summary>
    /// <see cref="ClapAudioClassifier"/> end to end through its public API (models host -> <see cref="ClapAudioClassifier.CreateFromHubAsync"/> ->
    /// <see cref="ClapAudioClassifier.ClassifyAsync(float[], IReadOnlyList{string})"/>): on the fixture's four clips against its six prompts, the
    /// probabilities within 0.01 of transformers' ClapModel and the same best label; then a real recording (LibriSpeech speech, 16 kHz resampled
    /// to 48 kHz) within 0.03 of transformers on six "This is a sound of ..." labels, with speech on top. Prints warm per-clip timings. Fast
    /// lanes. Added 2026-11-03.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Clap_Classifier_EndToEnd() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU))
            throw new UnsupportedTestException("the fast lanes (CUDA, OpenCL, WebGPU)");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        var root = doc.RootElement;
        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var clap = await ClapAudioClassifier.CreateFromHubAsync(accelerator, new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(modelHttp));
        double loadS = sw.Elapsed.TotalSeconds;
        var prompts = root.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();
        var notes = new List<string> { $"load {loadS:F1} s" };
        foreach (var c in root.GetProperty("clips").EnumerateArray())
        {
            string kind = c.GetProperty("kind").GetString()!;
            var want = c.GetProperty("probs").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var pcm = ClapFixtureClip(kind, c.GetProperty("samples").GetInt32(), c.GetProperty("seconds").GetDouble());
            var scores = await clap.ClassifyAsync(pcm, prompts);
            sw.Restart();
            scores = await clap.ClassifyAsync(pcm, prompts);   // warm: text embeddings cached, audio session compiled
            double warmMs = sw.Elapsed.TotalMilliseconds;
            double worst = scores.Select((s, i) => Math.Abs(s.Probability - want[i])).Max();
            int best = Array.IndexOf(scores, scores.MaxBy(s => s.Probability)), wantBest = Array.IndexOf(want, want.Max());
            if (worst > 0.01 || best != wantBest)
                throw new Exception($"{kind}: probabilities off by {worst:F4} (best '{prompts[best]}' vs transformers' '{prompts[wantBest]}')");
            notes.Add($"{kind} -> '{prompts[best]}' {scores[best].Probability:F3} (|dp| {worst:F4}, warm {warmMs:F0} ms)");
        }
        // A real recording, in CLAP's training template "This is a sound of ...": with bare labels this checkpoint calls all three of our
        // speech clips "music" (transformers 0.975 / 0.897 / 0.925 on 1089-134686-0003 / 4077-13751-0019 / librivox; ours 0.971) - the model,
        // not the port. Reference: transformers ClapModel on the clip resampled 16 -> 48 kHz with scipy resample_poly(3, 1) (2026-11-03); our
        // resampler is a different windowed sinc, hence 0.03.
        var wav = await assets.GetByteArrayAsync("test-audio/librispeech-1089_1089-134686-0003.wav");
        int rate = BitConverter.ToInt32(wav, 24);
        var speech = AudioPreprocessor.Resample(WavDecoder.DecodeWavFile(wav) ?? throw new Exception("could not decode the LibriSpeech clip"), rate, ClapAudioClassifier.SampleRate);
        var labels = new[] { "speech", "music", "a dog barking", "a gunshot", "a car engine", "rain" }.Select(l => "This is a sound of " + l).ToArray();
        var wantReal = new[] { 0.912714f, 0.016061f, 0.041725f, 0.013036f, 0.015693f, 0.000771f };
        var real = await clap.ClassifyAsync(speech, labels);
        sw.Restart();
        real = await clap.ClassifyAsync(speech, labels);
        double realMs = sw.Elapsed.TotalMilliseconds;
        var top = real.MaxBy(s => s.Probability);
        double worstReal = real.Select((s, i) => Math.Abs(s.Probability - wantReal[i])).Max();
        Console.WriteLine($"[Clap] {BackendName} LibriSpeech: " + string.Join(", ", real.OrderByDescending(s => s.Probability).Select(s => $"{s.Label} {s.Probability:F3}")));
        if (top.Label != labels[0] || worstReal > 0.03)
            throw new Exception($"LibriSpeech: best '{top.Label}' ({top.Probability:F3}), probabilities off transformers' by {worstReal:F4}");
        notes.Add($"LibriSpeech -> '{top.Label}' {top.Probability:F3} (|dp| {worstReal:F4}, warm {realMs:F0} ms)");
        var line = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine($"[Clap] {line}");
        return line;
    });

    /// <summary>
    /// CLAP's RoBERTa tokenizer (<see cref="BPETokenizer.EncodeRoberta"/> on Xenova/clap-htsat-unfused tokenizer.json, from models.spawndev.com)
    /// against transformers' (laion/clap-htsat-unfused, 5.20.2): the fixture's six prompts, and strings that exercise the GPT-2 split rules the
    /// old hand-written pre-tokenizer got wrong: contractions, digit runs, runs of spaces / tabs / newlines, leading spaces, non-ASCII (accents,
    /// an emoji: UTF-8 bytes, not UTF-16 chars cast to bytes). Not a GPU test: runs on the CPU lane only. Added 2026-11-03.
    /// </summary>
    [TestMethod]
    public async Task Clap_Tokenizer_MatchesTransformers() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("tokenization runs on the host; gated once, on the CPU lane");
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        using var http = CreateHuggingFaceHttpClient();
        var tok = BPETokenizer.LoadFromTokenizerJson(await http.GetStringAsync("https://models.spawndev.com/Xenova/clap-htsat-unfused/resolve/main/tokenizer.json"));
        var cases = new List<(string Text, int[] Want)>();
        var prompts = doc.RootElement.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();
        var ids = doc.RootElement.GetProperty("input_ids").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        for (int i = 0; i < prompts.Length; i++) cases.Add((prompts[i], ids[i]));
        // transformers AutoTokenizer("laion/clap-htsat-unfused")(text)["input_ids"], 2026-11-03
        cases.Add(("It's 3 o'clock!  Héllo   world 123456 don't", new[] { 0, 243, 18, 155, 1021, 108, 17036, 328, 1437, 289, 1140, 9905, 1437, 1437, 232, 17072, 34781, 218, 75, 2 }));
        cases.Add(("zombie\tgroan\n\nnext  ", new[] { 0, 329, 42816, 50117, 15821, 260, 50118, 50118, 25616, 1437, 1437, 2 }));
        cases.Add(("émoji 🎉 naïve café", new[] { 0, 1140, 4992, 5186, 8103, 12736, 23171, 39300, 26059, 2 }));
        cases.Add(("  leading spaces", new[] { 0, 1437, 981, 5938, 2 }));
        cases.Add(("GUNSHOT, then: footsteps...", new[] { 0, 534, 4154, 10237, 3293, 6, 172, 35, 18424, 734, 2 }));
        foreach (var (text, want) in cases)
        {
            var got = tok.EncodeRoberta(text);
            if (!got.SequenceEqual(want))
                throw new Exception($"'{text}': got [{string.Join(",", got)}] want [{string.Join(",", want)}]");
        }
        return $"{cases.Count} strings token-identical to transformers";
    });

    /// <summary>
    /// <see cref="ClapFeatureExtractor"/> against transformers' ClapFeatureExtractor (laion/clap-htsat-unfused, rand_trunc / repeatpad): the fixture's
    /// four 48 kHz clips (a two-tone, a click train over noise, white noise, and a 3.3 s chirp that is repeat-padded) regenerated here, their
    /// <c>input_features</c> [1001, 64] in dB. Our FFT runs in float32 against numpy's float64, so each value is held to 0.05 dB where it is above
    /// -60 dB (the signal), to 1 dB between -99 and -60 dB, and below -90 dB where the reference sits at the -100 dB floor (MEASURED 2026-11-03: the
    /// signal within 0.0011-0.0014 dB on CUDA / OpenCL / WebGPU; the chirp's repeat-pad edge put 9-15 floor values a few dB above -100 in float32).
    /// Fixture: tools/clap-reference.
    /// </summary>
    [TestMethod]
    public async Task Clap_Features_MatchTransformers() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/clap-fixture.json"));
        using var fx = new ClapFeatureExtractor(accelerator);
        var notes = new List<string>();
        foreach (var c in doc.RootElement.GetProperty("clips").EnumerateArray())
        {
            string kind = c.GetProperty("kind").GetString()!;
            int n = c.GetProperty("samples").GetInt32();
            double seconds = c.GetProperty("seconds").GetDouble();
            var want = c.GetProperty("input_features").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetSingle())).ToArray();
            if (want.Length != ClapFeatureExtractor.Frames * ClapFeatureExtractor.Mels) throw new Exception($"{kind}: fixture has {want.Length} features");
            var pcm = ClapFixtureClip(kind, n, seconds);
            using var pcmBuf = accelerator.Allocate1D(pcm);
            using var feat = accelerator.Allocate1D<float>(want.Length);
            fx.Extract(pcmBuf.View, n, feat.View);
            await accelerator.SynchronizeAsync();
            var got = await feat.CopyToHostAsync<float>(0, want.Length);
            double worstSignal = 0, worstLow = 0;
            int bad = 0;
            for (int i = 0; i < want.Length; i++)
            {
                double e = Math.Abs(got[i] - want[i]);
                if (want[i] > -60) { worstSignal = Math.Max(worstSignal, e); if (!(e <= 0.05)) bad++; }
                else if (want[i] > -99) { worstLow = Math.Max(worstLow, e); if (!(e <= 1.0)) bad++; }
                else if (!(got[i] <= -90)) bad++;            // the -100 dB floor: power below 1e-10 there, float32 rounding of a near-zero frame lands just above it
            }
            if (bad > 0) throw new Exception($"{kind}: {bad} of {want.Length} features off (worst {worstSignal:F4} dB in the signal, {worstLow:F3} dB in near-silent bands); first: frame 0 got [{string.Join(", ", got.Take(4).Select(v => v.ToString("F3")))}] want [{string.Join(", ", want.Take(4).Select(v => v.ToString("F3")))}]");
            notes.Add($"{kind}: worst {worstSignal:F4} dB (signal), {worstLow:F3} dB (near-silent)");
        }
        return string.Join("; ", notes);
    });
}
