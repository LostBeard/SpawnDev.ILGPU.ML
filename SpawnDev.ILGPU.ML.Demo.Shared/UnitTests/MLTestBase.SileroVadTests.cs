using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for Silero VAD - the endpointer the hands-free speech loop is built on.
///
/// <para>
/// It is STATEFUL in the way that is easiest to get silently wrong: its LSTM state arrives as the GRAPH
/// INPUTS <c>h</c> and <c>c</c> and comes back out as <c>new_h</c>/<c>new_c</c> on every 512-sample frame.
/// An engine that cached those inputs, or ran the LSTM against frozen state, would still return a
/// plausible probability for every frame. So this test does not merely check that numbers came out.
/// </para>
///
/// <para>
/// It drives 4 s of real speech through the model one 512-sample frame at a time, threading the state, and
/// makes TWO assertions:
/// <list type="number">
/// <item>every frame's probability matches onnxruntime, and</item>
/// <item>the sequence is FAR from a negative control in which the state is re-zeroed each frame.</item>
/// </list>
/// The control is in the fixture, and <c>tools/gen_silero_vad_reference.py</c> refuses to emit unless the
/// two runs differ by a wide margin - MEASURED at 0.978 on this clip. That is what makes "the state really
/// threads" a testable claim rather than a hope.
/// </para>
///
/// <para>
/// ⚠️ This model was the reason a REVERSED SLICE bug was found: its <c>adaptive_normalization</c> reverses
/// a <c>[1,1,3]</c> axis twice, our Slice returned an all-zeros buffer of the correct shape, and the
/// detector answered with a confident wrong probability. See <c>MLTestBase.SliceReverseTests</c>.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>Silero's native frame at 16 kHz. The model's <c>x</c> input is fixed at [1, 512].</summary>
    private const int SileroWindow = 512;

    [TestMethod(Timeout = 600000)]
    public async Task Vad_SileroVad_MatchesOnnxRuntimeOverRealSpeech() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await assets.GetStringAsync("references/vad/silero_vad_librivox.json");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        int window = root.GetProperty("window").GetInt32();
        int frames = root.GetProperty("frames").GetInt32();
        var expected = root.GetProperty("probs").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        var frozen = root.GetProperty("frozen_state_probs").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();

        if (window != SileroWindow)
            throw new Exception($"fixture window is {window}, expected {SileroWindow}");

        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");
        if (samples.Length / window < frames)
            throw new Exception($"audio holds {samples.Length / window} frames, fixture expects {frames}");

        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
            inputShapes: new Dictionary<string, int[]>
            {
                ["x"] = new[] { 1, window },
                ["h"] = new[] { 2, 1, 64 },
                ["c"] = new[] { 2, 1, 64 },
            });

        var xShape = new[] { 1, window };
        var stateShape = new[] { 2, 1, 64 };
        var h = new float[2 * 1 * 64];
        var c = new float[2 * 1 * 64];
        var frame = new float[window];
        var probs = new float[frames];

        for (int f = 0; f < frames; f++)
        {
            Array.Copy(samples, f * window, frame, 0, window);

            using var xBuf = accelerator.Allocate1D(frame);
            using var hBuf = accelerator.Allocate1D(h);
            using var cBuf = accelerator.Allocate1D(c);

            var outputs = await session.RunAsync(new Dictionary<string, Tensor>
            {
                ["x"] = new Tensor(xBuf.View, xShape),
                ["h"] = new Tensor(hBuf.View, stateShape),
                ["c"] = new Tensor(cBuf.View, stateShape),
            });

            probs[f] = (await ReadTensor(accelerator, outputs, "prob", 1))[0];
            // Feeding new_h/new_c back is the whole point - this is the loop a cached graph input breaks.
            h = await ReadTensor(accelerator, outputs, "new_h", h.Length);
            c = await ReadTensor(accelerator, outputs, "new_c", c.Length);
        }

        // ── 1. every frame matches onnxruntime ───────────────────────────────────────────────────────
        double worst = 0; int worstAt = -1;
        for (int f = 0; f < frames; f++)
        {
            double d = Math.Abs(probs[f] - expected[f]);
            if (d > worst) { worst = d; worstAt = f; }
        }
        if (worst > 2e-3)
            throw new Exception($"Silero VAD: frame {worstAt} probability {probs[worstAt]:F6} vs "
                              + $"onnxruntime {expected[worstAt]:F6} (max |d| {worst:E3} over {frames} frames)");

        // ── 2. and the run is NOT the frozen-state answer ────────────────────────────────────────────
        // Without this a cached h/c would still satisfy assertion 1 wherever the two happen to agree.
        double toFrozen = 0;
        for (int f = 0; f < frames; f++) toFrozen = Math.Max(toFrozen, Math.Abs(probs[f] - frozen[f]));
        double fixtureGap = root.GetProperty("max_threaded_vs_frozen").GetDouble();
        if (toFrozen < fixtureGap * 0.5)
            throw new Exception(
                $"Silero VAD looks like the FROZEN-STATE control: our run differs from it by only "
                + $"{toFrozen:F4}, where a correctly threaded run differs by {fixtureGap:F4}. The LSTM "
                + "state is not being carried between frames - check that h/c are not cached as static "
                + "inputs (OperatorInputReader.ReadCached).");

        // ── 3. and the SHIPPED CLASS agrees with the loop above, frame for frame ────────────────────
        // ⚠️ Everything up to here drives the session directly, which proves the ENGINE matches
        // onnxruntime but touches none of the code a caller actually uses. That gap was not theoretical:
        // routing SileroVad through SessionGraphCapture left WebGPU finding 0 utterances and crashed CUDA
        // with an access violation, and this test - the one named after the class - passed on all six
        // backends throughout. A wrapper defect belongs HERE, at the probability level where it can be
        // read, not three files away as "0 utterances".
        using var shipped = SileroVad.Create(accelerator, modelBytes);
        double worstClass = 0; int worstClassAt = -1;
        for (int f = 0; f < frames; f++)
        {
            Array.Copy(samples, f * window, frame, 0, window);
            float p = await shipped.ProcessFrameAsync(frame);
            double d = Math.Abs(p - probs[f]);
            if (d > worstClass) { worstClass = d; worstClassAt = f; }
        }
        if (worstClass > 1e-5)
            throw new Exception(
                $"SileroVad disagrees with a direct session run at frame {worstClassAt}: "
                + $"{worstClass:E3} apart. The model is fine (assertion 1 passed) - this is the WRAPPER: "
                + "check that h/c thread between frames and that nothing re-runs or reorders the graph.");

        int speech = probs.Count(p => p >= 0.5f);
        Console.WriteLine($"[Vad] Silero {frames} frames, max |d| vs ORT {worst:E2}, "
                        + $"{speech} frames >= 0.5, distance from frozen-state control {toFrozen:F4}, "
                        + $"SileroVad vs direct session {worstClass:E2}");
    });

    private static async Task<float[]> ReadTensor(Accelerator accelerator,
        Dictionary<string, Tensor> outputs, string name, int count)
    {
        if (!outputs.TryGetValue(name, out var t))
            throw new Exception($"Silero VAD: session produced no output named '{name}'");
        if (t.ElementCount < count)
            throw new Exception($"Silero VAD: '{name}' holds {t.ElementCount} values, expected {count}");
        using var host = accelerator.Allocate1D<float>(count);
        await host.View.CopyFromAsync(t.Data.SubView(0, count));
        await accelerator.SynchronizeAsync();
        return await host.CopyToHostAsync<float>(0, count);
    }
}
