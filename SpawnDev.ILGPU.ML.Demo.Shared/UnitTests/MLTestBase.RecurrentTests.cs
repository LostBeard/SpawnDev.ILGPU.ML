using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using System;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Gate for the recurrent operators (LSTM, GRU, RNN) against onnxruntime.
///
/// <para>
/// WHY THIS FILE EXISTS: all three were listed in <c>OperatorRegistry.BuiltinOpTypes</c> and produced
/// NOTHING for any real model. Each opened with
/// <c>if (wVals == null || rVals == null) return;</c> / <c>if (xVals == null) return;</c>, where the values
/// came from <c>ctx.TryGetInputValues</c> - which reads COMPILE-TIME CONSTANTS ONLY. X is the runtime
/// input, so it was always null and the operator returned having left its output buffer untouched. No
/// exception, no warning, just whatever happened to be in the buffer. Found by
/// <c>tools/audit-operator-support.cs</c>, not by any test.
/// </para>
///
/// <para>
/// ⚠️ THE FIXTURE FEEDS X AS A GRAPH INPUT, and that is the whole point. A fixture whose X was a constant
/// would take the one branch that always worked and prove nothing - the same trap as testing a resampler
/// with audio already at the target rate. Weights stay initializers, which is how a real model ships them.
/// </para>
///
/// <para>
/// The expected values come from onnxruntime (<c>tools/gen_lstm_reference.py</c>), so this is an
/// independent target rather than our own arithmetic agreeing with itself.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    private async Task RecurrentMatchesOnnxRuntime(string kind)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var slug = kind.ToLowerInvariant();
        var json = await http.GetStringAsync($"references/recurrent/tiny_{slug}.json");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        int seq = root.GetProperty("seq").GetInt32();
        int batch = root.GetProperty("batch").GetInt32();
        int inputSize = root.GetProperty("input_size").GetInt32();
        int hidden = root.GetProperty("hidden_size").GetInt32();
        var x = root.GetProperty("X").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();

        var modelBytes = await http.GetByteArrayAsync($"references/recurrent/tiny_{slug}.onnx");

        await RunTest(async accelerator =>
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                inputShapes: new System.Collections.Generic.Dictionary<string, int[]>
                {
                    ["X"] = new[] { seq, batch, inputSize },
                });

            using var xBuf = accelerator.Allocate1D(x);
            var outputs = await session.RunAsync(new System.Collections.Generic.Dictionary<string, Tensor>
            {
                ["X"] = new Tensor(xBuf.View, new[] { seq, batch, inputSize }),
            });

            // Y_h is the state every consumer actually reads, and it depends on every timestep, so a
            // partially-run recurrence cannot match it by luck.
            foreach (var name in new[] { "Y_h", "Y" })
            {
                if (!root.GetProperty("outputs").TryGetProperty(name, out var expectedEl)) continue;
                var expected = expectedEl.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
                if (!outputs.TryGetValue(name, out var gotTensor))
                    throw new Exception($"{kind}: session produced no output named '{name}'");

                int count = Math.Min(gotTensor.ElementCount, expected.Length);
                if (count < expected.Length)
                    throw new Exception($"{kind} {name}: output holds {gotTensor.ElementCount} values, "
                                      + $"expected {expected.Length}");
                using var host = accelerator.Allocate1D<float>(count);
                await host.View.CopyFromAsync(gotTensor.Data.SubView(0, count));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, count);

                // An untouched output buffer is all zeros, which is the pre-fix behaviour - call it out by
                // name rather than reporting a generic mismatch.
                bool allZero = got.Take(expected.Length).All(v => v == 0f);
                if (allZero && expected.Any(v => Math.Abs(v) > 1e-6f))
                    throw new Exception(
                        $"{kind} {name} is ALL ZEROS - the operator never computed. This is the exact "
                        + "failure this test exists for: TryGetInputValues returns compile-time constants "
                        + "only, so a runtime X reads as null.");

                double worst = 0; int worstAt = -1;
                for (int i = 0; i < expected.Length; i++)
                {
                    double d = Math.Abs(got[i] - expected[i]);
                    if (d > worst) { worst = d; worstAt = i; }
                }
                if (worst > 2e-4)
                    throw new Exception($"{kind} {name}: max |d| {worst:E3} at {worstAt} "
                                      + $"(ORT {expected[worstAt]:F6} vs ours {got[worstAt]:F6})");

                Console.WriteLine($"[Recurrent] {kind} {name}: {expected.Length} values, max |d| {worst:E2}");
            }
        });
    }

    [TestMethod(Timeout = 180000)]
    public async Task Recurrent_LSTM_MatchesOnnxRuntime() => await RecurrentMatchesOnnxRuntime("LSTM");

    [TestMethod(Timeout = 180000)]
    public async Task Recurrent_GRU_MatchesOnnxRuntime() => await RecurrentMatchesOnnxRuntime("GRU");

    [TestMethod(Timeout = 180000)]
    public async Task Recurrent_RNN_MatchesOnnxRuntime() => await RecurrentMatchesOnnxRuntime("RNN");
}
