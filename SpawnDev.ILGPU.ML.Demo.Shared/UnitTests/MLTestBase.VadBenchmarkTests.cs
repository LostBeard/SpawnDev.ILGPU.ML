using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// How fast is the VAD, per backend, on the backend that will actually run it?
///
/// <para>
/// This exists because the rate is not a nice-to-have for a detector. A frame is 512 samples at 16 kHz =
/// <b>32 ms of audio</b>, so a live microphone hands over a frame every 32 ms forever. Finish slower than
/// that and the loop falls behind without bound - correct probabilities arriving too late to endpoint
/// anything.
/// </para>
///
/// <para>
/// ⚠️ It REPORTS rather than asserts on most backends, deliberately. A wall-clock threshold inside PMT is
/// a flaky gate: the run shares a machine TJ is working on, and WebGL/Wasm are known-slow lanes that
/// CLAUDE.md says must not gate ML progress. So the number is printed and the only hard failure is a
/// detector so slow it could not be used at all. `tools/vad-harness bench` is the instrument for tuning;
/// this is the cross-backend picture that a desktop harness cannot give.
/// </para>
///
/// <para>
/// ⚠️ The output is tagged <c>[Benchmark]</c> because that is the ONLY console text PMT echoes into its
/// log (<c>ProjectRunner.cs</c> filters on it). A number printed any other way is buffered and lost, which
/// is exactly how the browser cost went unmeasured long enough for me to hand TJ a per-frame table derived
/// by dividing a whole test's duration by its frame count - see
/// <c>feedback-a-benchmark-duration-is-not-a-per-unit-rate</c>. Setup dominates that duration; this times
/// only the frames.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>Audio carried by one 512-sample frame at 16 kHz.</summary>
    private const double VadFrameAudioMs = 512 * 1000.0 / 16000.0;

    [TestMethod(Timeout = 900000)]
    public async Task Vad_Benchmark_FrameRate() => await RunTest(async accelerator =>
    {
        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");

        var wavBytes = await assets.GetByteArrayAsync("test-audio/librivox-public-domain.wav");
        var samples = WavDecoder.DecodeWavFile(wavBytes)
            ?? throw new Exception("could not decode test-audio/librivox-public-domain.wav");
        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        using var vad = SileroVad.Create(accelerator, modelBytes);
        var frame = new float[SileroVad.WindowSize];

        // The first frames pay for kernel compilation and buffer-pool growth. Folding those into the mean
        // would make the result depend mostly on how many frames were run.
        Array.Copy(samples, 0, frame, 0, frame.Length);
        for (int i = 0; i < 5; i++) await vad.ProcessFrameAsync(frame);

        const int Frames = 60;
        var times = new double[Frames];
        var sw = new Stopwatch();
        for (int f = 0; f < Frames; f++)
        {
            int at = (f * SileroVad.WindowSize) % (samples.Length - SileroVad.WindowSize);
            Array.Copy(samples, at, frame, 0, frame.Length);
            sw.Restart();
            await vad.ProcessFrameAsync(frame);
            sw.Stop();
            times[f] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double mean = times.Average();
        double p50 = times[Frames / 2];
        double p99 = times[(int)(Frames * 0.99)];
        double realtime = VadFrameAudioMs / mean;

        // p50 and p99 both matter and say different things: the median is whether it keeps up at all, the
        // tail is whether it ever stalls long enough to drop a turn. MEASURED on OpenCL via the desktop
        // harness, p99 was 47 ms against a 4 ms median - a spike well past the 32 ms budget that a mean
        // alone hides completely.
        Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}]: "
                        + $"mean {mean:F2}ms  p50 {p50:F2}ms  p99 {p99:F2}ms  min {times[0]:F2}ms  "
                        + $"| frame budget {VadFrameAudioMs:F1}ms -> {realtime:F2}x realtime "
                        + $"({(mean <= VadFrameAudioMs ? "KEEPS UP" : "TOO SLOW for a live microphone")})");

        // WHERE the frame goes, from the executor's own split. Without this the next person optimising
        // the browser VAD is guessing between three candidates - readback latency, sync drains, and the
        // per-node walk over 125 nodes - and CLAUDE.md is explicit that a GPU bottleneck must be named by
        // measurement rather than reasoned about. These are static and describe the LAST run, so they are
        // read straight after the timed loop.
        double exec = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunTotalMs;
        double readback = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackMs;
        int readbacks = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackCount;
        double drain = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunSyncDrainMs;
        int drains = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunSyncDrainCount;
        Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}] last frame split: "
                        + $"executor {exec:F2}ms  readback {readback:F2}ms x{readbacks}  "
                        + $"syncDrain {drain:F2}ms x{drains}  "
                        + $"| unattributed {Math.Max(0, mean - exec):F2}ms of the {mean:F2}ms mean");

        // ⚠️ NAME the readbacks rather than inferring who owns them. The counter above lives in the
        // executor's shape-lane path (promoting a node output into runtimeConstants), which is NOT the
        // path an operator's own OperatorInputReader call takes - so "16 readbacks" does not by itself
        // implicate the host-side LSTM, however plausible that is. This list is the evidence.
        var names = SpawnDev.ILGPU.ML.Graph.GraphExecutor.LastRunReadbackNames;
        Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}] readback owners: "
                        + (names.Count == 0 ? "(none recorded)" : string.Join(", ", names)));

        // When a dispatch plan is live, the frame is no longer a graph walk and the per-node numbers below
        // describe nothing. What matters instead is the replay's own split, because its terms differ in
        // KIND: the plan call is a single interop crossing, while the sync is a round trip through the JS
        // event loop - and only one of those is ours to shrink.
        if (vad.IsCaptured && vad.LastReplaySplit is { } split)
        {
            Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}] REPLAYING "
                            + $"{vad.DispatchCount} dispatches | inputCopy {split.InputCopyMs:F2}ms  "
                            + $"planCall {split.PlanCallMs:F2}ms  sync {split.SyncMs:F2}ms  "
                            + $"| unattributed {Math.Max(0, mean - split.InputCopyMs - split.PlanCallMs - split.SyncMs):F2}ms "
                            + $"of the {mean:F2}ms mean (state copies + the prob readback)");
        }

        // Where the NON-readback time goes, aggregated by op type. Silero is 125 nodes of trivial
        // arithmetic (about 64K multiply-adds all told), so if the frame is dominated by per-node cost
        // rather than by a few heavy nodes, the lever is orchestration - one dispatch plan replayed -
        // rather than any individual kernel. Opt-in and off unless this block runs.
        var timings = new System.Collections.Generic.Dictionary<string, double>();
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = timings;
        try
        {
            Array.Copy(samples, 0, frame, 0, frame.Length);
            await vad.ProcessFrameAsync(frame);
        }
        finally { SpawnDev.ILGPU.ML.Graph.GraphExecutor.CapturedNodeTimingsMs = null; }

        double timedTotal = timings.Values.Sum();
        var byOp = timings
            .GroupBy(kv => kv.Key.Split('_')[1])
            .Select(g => (Op: g.Key, Ms: g.Sum(x => x.Value), N: g.Count()))
            .OrderByDescending(x => x.Ms).Take(8);
        Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}] per-node total "
                        + $"{timedTotal:F2}ms over {timings.Count} nodes "
                        + $"(mean {(timings.Count > 0 ? timedTotal / timings.Count : 0):F3}ms/node) | by op: "
                        + string.Join(", ", byOp.Select(x => $"{x.Op} {x.Ms:F2}ms x{x.N}")));
        var top = timings.OrderByDescending(kv => kv.Value).Take(5);
        Console.WriteLine($"[Benchmark] Silero VAD [{accelerator.AcceleratorType}] slowest nodes: "
                        + string.Join(", ", top.Select(kv => $"{kv.Key}={kv.Value:F2}ms")));

        // The only hard failure: so slow the detector could not be used for anything, live or offline.
        // Anything short of that is reported and left to judgement, because this shares a machine.
        if (mean > VadFrameAudioMs * 20)
            throw new Exception(
                $"Silero VAD is {mean:F1}ms per frame against a {VadFrameAudioMs:F1}ms budget "
                + $"({realtime:F3}x realtime) - more than 20x too slow to follow a microphone.");
    });
}
