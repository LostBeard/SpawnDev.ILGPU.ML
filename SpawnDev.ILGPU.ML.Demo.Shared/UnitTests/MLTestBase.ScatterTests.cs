using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for <c>ScatterElements</c> and <c>ScatterND</c> against onnxruntime.
///
/// <para>
/// WHY THIS FILE EXISTS: both operators copied <c>data</c> to the output and then RETURNED whenever their
/// inputs were not compile-time constants - discarding every update. ScatterND's own comment described it
/// as "fall back to identity". Since a real model computes indices and updates at runtime, that was the
/// only path either operator ever took: they were no-ops emitting a correctly shaped, entirely plausible
/// tensor. Found by <c>tools/audit-operator-support.cs</c>, not by any test.
/// </para>
///
/// <para>
/// ⚠️ INDICES AND UPDATES ARE GRAPH INPUTS IN THE FIXTURES, deliberately. As initializers they would be
/// constant-folded onto the branch that always worked, and the test would pass against the broken code.
/// The generator also asserts the scatter actually CHANGES values, so "did nothing" cannot look like
/// "did the right thing" - it refuses to emit a fixture where output equals data.
/// </para>
///
/// <para>
/// The fixtures avoid duplicate indices on purpose: ONNX leaves their order undefined for
/// <c>reduction="none"</c>, onnxruntime resolves last-wins, and a GPU kernel races - a fixture with
/// duplicates would be legitimately nondeterministic and flake.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private async Task ScatterMatchesOnnxRuntime(string name)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await http.GetStringAsync($"references/scatter/{name}.json");
        var modelBytes = await http.GetByteArrayAsync($"references/scatter/{name}.onnx");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var inputShapes = new Dictionary<string, int[]>();
        foreach (var p in root.GetProperty("inputs").EnumerateObject())
            inputShapes[p.Name] = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        await RunTest(async accelerator =>
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                inputShapes: inputShapes);

            var feeds = new Dictionary<string, Tensor>();
            var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            try
            {
                foreach (var p in root.GetProperty("inputs").EnumerateObject())
                {
                    var data = p.Value.GetProperty("data").EnumerateArray()
                        .Select(e => (float)e.GetDouble()).ToArray();
                    var buf = accelerator.Allocate1D(data);
                    buffers.Add(buf);
                    feeds[p.Name] = new Tensor(buf.View, inputShapes[p.Name]);
                }

                var outputs = await session.RunAsync(feeds);

                var expected = root.GetProperty("outputs").EnumerateObject().First();
                var want = expected.Value.GetProperty("data").EnumerateArray()
                    .Select(e => (float)e.GetDouble()).ToArray();
                if (!outputs.TryGetValue(expected.Name, out var gotTensor))
                    throw new Exception($"{name}: no output named '{expected.Name}'");

                using var host = accelerator.Allocate1D<float>(want.Length);
                await host.View.CopyFromAsync(gotTensor.Data.SubView(0, want.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, want.Length);

                // The pre-fix behaviour was "output == data", so say that plainly when it happens rather
                // than reporting an anonymous numeric mismatch.
                var dataIn = root.GetProperty("inputs").GetProperty("data")
                    .GetProperty("data").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
                bool unchanged = got.Length == dataIn.Length
                    && !got.Where((v, i) => Math.Abs(v - dataIn[i]) > 1e-6f).Any();
                if (unchanged)
                    throw new Exception(
                        $"{name}: output is IDENTICAL to the input data - every update was discarded. That "
                        + "is the exact no-op this test exists for.");

                double worst = 0; int worstAt = -1;
                for (int i = 0; i < want.Length; i++)
                {
                    double d = Math.Abs(got[i] - want[i]);
                    if (d > worst) { worst = d; worstAt = i; }
                }
                if (worst > 1e-4)
                    throw new Exception($"{name}: max |d| {worst:E3} at {worstAt} "
                                      + $"(ORT {want[worstAt]:F6} vs ours {got[worstAt]:F6})");

                Console.WriteLine($"[Scatter] {name}: {want.Length} values, max |d| {worst:E2}");
            }
            finally
            {
                foreach (var b in buffers) b.Dispose();
            }
        });
    }

    [TestMethod(Timeout = 180000)]
    public async Task Scatter_Elements_MatchesOnnxRuntime()
        => await ScatterMatchesOnnxRuntime("tiny_scatter_elements");

    [TestMethod(Timeout = 180000)]
    public async Task Scatter_ND_MatchesOnnxRuntime()
        => await ScatterMatchesOnnxRuntime("tiny_scatter_nd");
}
