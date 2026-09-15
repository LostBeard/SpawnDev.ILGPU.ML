using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Names the nodes that make a NEW utterance length cost more than a repeat of the same length.
/// </summary>
/// <remarks>
/// 🔴 THIS IS THE ONE THAT MATTERS FOR STREAMING. Graph capture made a REPEATED length ~3.4x faster, but
/// in streaming speech every chunk is a new length, so the repeat case is the case a conversation almost
/// never hits. MEASURED in the SpawnDev.AI demo (WebGPU, RTX 4070):
/// <code>
///   65 tok, new length -> RTF 2.72x
///   35 tok, new length -> RTF 1.95x      &lt;-- slower than realtime: pauses between chunks
///   35 tok, repeated   -> RTF 0.34x
/// </code>
/// A new-length pass costs ~1.83 ms/node against ~0.8 ms/node warm, and shader-resolve is only 91 ms of
/// it - so it is NOT kernel compilation. The executor's own counters leave ~1,843 ms of a 4,165 ms pass
/// unattributed. This test diffs PER-NODE timings between the first run at a shape and a second run at the
/// SAME shape, which is the only thing that can name where that goes.
///
/// ⚠️ It REPORTS, it does not gate. Per-node wall clock on a shared browser device is far too noisy to
/// fail a build on, and a flaky red is worse than no signal. Read it with
/// <c>PMT_CONSOLE_LOG=NewShape</c> - PMT summarises browser console lines away by default, which is how a
/// diagnostic runs every sweep and has its verdict discarded.
///
/// ⚠️ Capture is DISABLED here on purpose. With it on, the second pass would be a recording (three graph
/// runs) rather than a plain warm pass, and the diff would measure the recorder instead of the shape
/// penalty.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy")]
    public async Task Kokoro_NewShapePenalty_PerNodeDiff() => await RunTest(async accelerator =>
    {
        RequireShippableTtsBackend(accelerator);
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        byte[] modelBytes, voiceBytes;
        try
        {
            modelBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/onnx/model.onnx");
            voiceBytes = await http.GetByteArrayAsync($"{KokoroHubBase}/voices/af_heart.bin");
        }
        catch (HttpRequestException ex)
        {
            throw new UnsupportedTestException($"the hub did not serve Kokoro: {ex.Message}");
        }

        var pack = KokoroVoicePack.FromBytes("af_heart", voiceBytes);
        using var pipeline = KokoroPipeline.Create(accelerator, modelBytes);
        pipeline.EnableGraphCapture = false;   // see the remarks - a recording would not be a warm pass

        // A throwaway utterance first, so the cost being measured below is the NEW SHAPE and not one-time
        // session setup (kernel compilation, pool growth, weight upload) that any first call pays.
        await pipeline.SpeakTokensAsync(KokoroShorterTokens, pack);

        async Task<(double ms, Dictionary<string, double> nodes)> PassAsync(long[] tokens)
        {
            var timings = new Dictionary<string, double>();
            Graph.GraphExecutor.CapturedNodeTimingsMs = timings;
            try
            {
                var sw = Stopwatch.StartNew();
                await pipeline.SpeakTokensAsync(tokens, pack);
                sw.Stop();
                return (sw.Elapsed.TotalMilliseconds, new Dictionary<string, double>(timings));
            }
            finally { Graph.GraphExecutor.CapturedNodeTimingsMs = null; }
        }

        // FIRST time at this length, then the SAME length again. Everything else is held constant, so the
        // difference is the shape being new and nothing else.
        var (coldMs, cold) = await PassAsync(KokoroReferenceTokens);
        var (warmMs, warm) = await PassAsync(KokoroReferenceTokens);

        Console.WriteLine($"[NewShape] {BackendName}: first-at-this-length {coldMs:F0} ms, repeat {warmMs:F0} ms, "
                        + $"penalty {coldMs - warmMs:F0} ms ({(warmMs > 0 ? coldMs / warmMs : 0):F2}x) over "
                        + $"{cold.Count} timed nodes");

        // Attribute the penalty per node. Op type is what a fix acts on - one slow node is a kernel, a
        // whole op type spread across hundreds of nodes is a code path.
        var byOp = new Dictionary<string, (double delta, int count)>();
        double totalDelta = 0;
        foreach (var kv in cold)
        {
            warm.TryGetValue(kv.Key, out var w);
            var d = kv.Value - w;
            totalDelta += d;
            // keys look like "0123_OpType_/path/to/output"
            var parts = kv.Key.Split('_', 3);
            var op = parts.Length >= 2 ? parts[1] : "?";
            var cur = byOp.TryGetValue(op, out var e) ? e : (0d, 0);
            byOp[op] = (cur.Item1 + d, cur.Item2 + 1);
        }

        Console.WriteLine($"[NewShape] {BackendName}: per-node deltas sum to {totalDelta:F0} ms of the "
                        + $"{coldMs - warmMs:F0} ms penalty - the remainder is OUTSIDE the node loop "
                        + "(plan building, shape inference, allocation), which is itself the finding.");

        foreach (var kv in byOp.OrderByDescending(k => k.Value.delta).Take(12))
            Console.WriteLine($"[NewShape] {BackendName}:   {kv.Key,-22} {kv.Value.delta,9:F1} ms over "
                            + $"{kv.Value.count,5} nodes");

        // Sanity: the instrument has to have measured something, or the table above is noise dressed as data.
        if (cold.Count == 0)
            throw new Exception(
                $"{BackendName}: CapturedNodeTimingsMs recorded no nodes, so the per-op table is empty and "
              + "this test proved nothing. The hook did not engage.");
    });
}
