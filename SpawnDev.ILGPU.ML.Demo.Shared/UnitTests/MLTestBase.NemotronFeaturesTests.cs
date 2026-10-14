using System.Text.Json;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>Largest allowed difference, in log-mel units (the values span about -16.6 to +0.4), between the streaming features and NeMo's recipe in float64.</summary>
    private const double NemoFeatureTolerance = 5e-4;

    private sealed record NemoFeatureFixture(float[] Pcm, float[] Features, int Frames);

    private static async Task<NemoFeatureFixture> LoadNemoFeatureFixture(System.Net.Http.HttpClient http)
    {
        using var doc = JsonDocument.Parse(await http.GetStringAsync("test-audio/nemotron-features-fixture.json"));
        var root = doc.RootElement;
        var i16 = Convert.FromBase64String(root.GetProperty("pcm16").GetString()!);
        var pcm = new float[i16.Length / 2];
        for (int i = 0; i < pcm.Length; i++) pcm[i] = BitConverter.ToInt16(i16, i * 2) / 32768f;
        var fb = Convert.FromBase64String(root.GetProperty("features").GetString()!);
        var features = new float[fb.Length / 4];
        Buffer.BlockCopy(fb, 0, features, 0, fb.Length);
        return new NemoFeatureFixture(pcm, features, root.GetProperty("frames").GetInt32());
    }

    /// <summary>Feeds the clip then one second of silence (as the streaming recogniser does to flush the tail) in pieces of <paramref name="piece"/> samples, and returns every frame.</summary>
    private static float[] StreamNemoFeatures(float[] pcm, int piece)
    {
        var fx = new NemoStreamingFeatures();
        var frame = new float[NemoStreamingFeatures.Mels];
        var all = new List<float>();
        var audio = pcm.Concat(new float[16000]).ToArray();
        for (int at = 0; at < audio.Length; at += piece)
        {
            fx.Accept(audio.AsSpan(at, Math.Min(piece, audio.Length - at)));
            while (fx.TryNext(frame)) all.AddRange(frame);
        }
        return all.ToArray();
    }

    /// <summary>
    /// The features that feed the cache-aware streaming FastConformer are NeMo's, to the digit, and do not depend on how the audio happens to be cut into pieces.
    /// Reference: the same recipe in numpy float64 (pre-emphasis 0.97, SYMMETRIC Hann 400 in a 512 FFT, centred zero-padded STFT, 128 Slaney mel, log(x + 2^-24)),
    /// whose transcripts match sherpa-onnx's on the LibriSpeech clips.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task NemotronFeatures_MatchTheNeMoRecipe_WhateverTheChunking()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var fx = await LoadNemoFeatureFixture(http);

        var whole = StreamNemoFeatures(fx.Pcm, fx.Pcm.Length + 16000);
        // The reference pads the right edge with zeros and so emits two more frames whose windows hang over the end of the audio; a stream cannot produce those until more
        // audio arrives (and after one second of silence nobody needs them). Every frame it does produce is compared.
        int produced = whole.Length / NemoStreamingFeatures.Mels;
        if (produced < fx.Frames - 2 || produced > fx.Frames)
            throw new Exception($"{produced} frames, the reference has {fx.Frames} (a stream may lack at most the last two)");
        double worst = 0; int at = 0;
        for (int i = 0; i < whole.Length; i++)
        {
            var d = Math.Abs(whole[i] - fx.Features[i]);
            if (d > worst) { worst = d; at = i; }
        }
        if (worst > NemoFeatureTolerance)
            throw new Exception($"max |d| {worst:E2} (frame {at / NemoStreamingFeatures.Mels}, mel {at % NemoStreamingFeatures.Mels}) > {NemoFeatureTolerance:E1} vs NeMo's recipe");

        // The way the audio arrives must not matter: 20 ms pieces, an awkward prime, and a single sample at a time.
        foreach (var piece in new[] { 320, 777, 1 })
        {
            var pieces = StreamNemoFeatures(fx.Pcm, piece);
            if (pieces.Length != whole.Length) throw new Exception($"pieces of {piece}: {pieces.Length / NemoStreamingFeatures.Mels} frames, whole {whole.Length / NemoStreamingFeatures.Mels}");
            for (int i = 0; i < whole.Length; i++)
                if (pieces[i] != whole[i]) throw new Exception($"pieces of {piece} samples: frame {i / NemoStreamingFeatures.Mels} differs from the whole-clip result ({pieces[i]} vs {whole[i]})");
        }
        Console.WriteLine($"[NemotronFeatures] {fx.Frames} frames, max |d| {worst:E2} vs NeMo's recipe; identical for pieces of 1, 320, 777 and whole");
    }
}
