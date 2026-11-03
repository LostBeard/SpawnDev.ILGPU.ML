using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
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
