using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Smart Turn v3.2 (pipecat-ai, BSD-2): "has the speaker finished their turn?", a Whisper-tiny encoder over the last 8 s of
/// audio. It replaces waiting a fixed half second of silence before answering. Its input is the 8-second variant of the GPU
/// Whisper log-mel, and the reference for that input is the model authors' own feature extractor.
/// </summary>
/// <remarks>
/// Fixture <c>test-audio/smartturn-fixture.json</c> (tools: _speech-research/smartturn_fixture.py): for three clips - shorter than the
/// window, a mid case, and 13.3 s (longer than the window) - the int16 audio, the features pipecat's <c>inference.py</c> feeds the model
/// (transformers <c>WhisperFeatureExtractor(chunk_length=8)</c> with <c>do_normalize=True</c> on audio that is zero-padded at the START or
/// cut to its LAST 8 s), and onnxruntime's output on them.
/// </remarks>
public abstract partial class MLTestBase
{
    private const string SmartTurnRepo = "pipecat-ai/smart-turn-v3";
    private const string SmartTurnFile = "smart-turn-v3.2-gpu.onnx";

    private sealed record SmartTurnCase(string Name, float[] Pcm, float[] Features, float Probability);

    private static async Task<List<SmartTurnCase>> LoadSmartTurnFixture(System.Net.Http.HttpClient assets)
    {
        using var doc = JsonDocument.Parse(await assets.GetStringAsync("test-audio/smartturn-fixture.json"));
        var cases = new List<SmartTurnCase>();
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            var i16 = Convert.FromBase64String(c.GetProperty("pcm16").GetString()!);
            var pcm = new float[i16.Length / 2];
            for (int i = 0; i < pcm.Length; i++) pcm[i] = BitConverter.ToInt16(i16, i * 2) / 32768f;
            var fb = Convert.FromBase64String(c.GetProperty("features").GetString()!);
            var features = new float[fb.Length / 4];
            Buffer.BlockCopy(fb, 0, features, 0, fb.Length);
            cases.Add(new SmartTurnCase(c.GetProperty("name").GetString()!, pcm, features, c.GetProperty("probability").GetSingle()));
        }
        return cases;
    }

    /// <summary>
    /// The 8-second GPU log-mel equals the features pipecat's feature extractor produces: audio start-padded (or cut to its last 8 s),
    /// waveform normalised to zero mean / unit variance over the whole padded window, then Whisper's log-mel at 800 frames.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task SmartTurnMel_Gpu_MatchesPipecatFeatureExtractor() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException("direct-DFT Whisper mel is gated to the fast backends (see WhisperMel_Gpu_MatchesCpuOracle)");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        const int window = 8 * 16000;
        using var mel = new WhisperMelPreprocessor(accelerator, window, alignToEnd: true, normalizeWaveform: true);
        if (mel.Frames != 800 || mel.MelElements != 80 * 800)
            throw new Exception($"8 s window: {mel.Frames} frames / {mel.MelElements} elements, expected 800 / 64000");

        var report = new System.Text.StringBuilder();
        bool failed = false;
        foreach (var c in await LoadSmartTurnFixture(assets))
        {
            if (c.Features.Length != mel.MelElements) throw new Exception($"{c.Name}: fixture has {c.Features.Length} features");
            using var pcm = accelerator.Allocate1D(c.Pcm);
            using var melBuf = accelerator.Allocate1D<float>(mel.MelElements);
            mel.ComputeLogMelSpectrogram(pcm.View, c.Pcm.Length, melBuf.View);
            await accelerator.SynchronizeAsync();
            var got = await melBuf.CopyToHostAsync<float>(0, mel.MelElements);
            double worst = 0;
            for (int i = 0; i < got.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - c.Features[i]));
            Console.WriteLine($"[SmartTurnMel] {c.Name}: max |d| {worst:E2} vs transformers on {BackendName}");
            report.Append($"{c.Name}: max |d| {worst:E2}; ");
            if (worst > SmartTurnMelTol) failed = true;
        }
        if (failed)
            throw new Exception($"over {SmartTurnMelTol:E1} vs the pipecat feature extractor on {BackendName}: {report}");
    });

    /// <summary>Absolute, on the final features (range about [-0.2, 2.0]). MEASURED on CUDA: 1.4e-3 to 1.6e-3 (the direct DFT's fast sin/cos; a numpy replica with the same window is at 1.5e-7 of the HF features).</summary>
    private const double SmartTurnMelTol = 2.5e-3;

    /// <summary>
    /// The whole turn-end chain - 8 s GPU mel, then the Smart Turn graph on our engine - gives the probability onnxruntime gives on the
    /// pipecat features. Desktop GPU only: it downloads the 32 MB model.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Pipeline_SmartTurn_MatchesOnnxRuntimeProbability() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: desktop GPU only (32 MB model download)");
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        using var modelHttp = new System.Net.Http.HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var source = new HttpClientModelSource(modelHttp);
        await using var graph = await source.OpenAsync(SmartTurnRepo, SmartTurnFile);
        using var session = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph,
            inputShapes: new Dictionary<string, int[]> { ["input_features"] = new[] { 1, 80, 800 } });
        using var mel = new WhisperMelPreprocessor(accelerator, 8 * 16000, alignToEnd: true, normalizeWaveform: true);

        var report = new System.Text.StringBuilder();
        bool failed = false;
        foreach (var c in await LoadSmartTurnFixture(assets))
        {
            using var pcm = accelerator.Allocate1D(c.Pcm);
            using var melBuf = accelerator.Allocate1D<float>(mel.MelElements);
            mel.ComputeLogMelSpectrogram(pcm.View, c.Pcm.Length, melBuf.View);
            var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_features"] = new Tensor(melBuf.View, new[] { 1, 80, 800 }) });
            using var host = accelerator.Allocate1D<float>(1);
            await host.View.CopyFromAsync(outs["logits"].Data.SubView(0, 1));
            await accelerator.SynchronizeAsync();
            float p = (await host.CopyToHostAsync<float>(0, 1))[0];
            Console.WriteLine($"[SmartTurn] {c.Name}: p(complete) {p:F4} vs onnxruntime {c.Probability:F4} on {BackendName}");
            session.ReturnOutputs(outs);
            report.Append($"{c.Name}: {p:F5} vs {c.Probability:F5}; ");
            if (Math.Abs(p - c.Probability) > 0.02) failed = true;
        }
        if (failed)
            throw new Exception($"probabilities differ from onnxruntime on {BackendName}: {report}");
    });
}
