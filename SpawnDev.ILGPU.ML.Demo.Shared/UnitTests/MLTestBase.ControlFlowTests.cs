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
/// Gate for ONNX control flow (If, Loop, Scan) against onnxruntime.
///
/// <para>
/// WHY THIS FILE EXISTS: all three took their output shapes from <c>inputs[0]</c> - which is the CONDITION
/// for If, the SCALAR trip count for Loop, and the first STATE for Scan. Never the output. Each therefore
/// allocated a buffer of the wrong size and the branch or body result was silently truncated into it.
/// </para>
///
/// <para>
/// <c>If</c> was found the hard way: ZipVoice's text encoder reaches its relative positional-encoding table
/// through one If, so we returned the scalar 1.0 where onnxruntime returns a [1999, 48] table, and the
/// encoder diverged by 18.6% of peak while every downstream shape stayed self-consistent. On top of that,
/// the subgraph never even reached the operator - it was destroyed by a JSON round trip and arrived as a
/// string, so no branch ever ran. <c>Loop</c> and <c>Scan</c> were then found by
/// <c>tools/audit-operator-support.cs</c> rather than by any test.
/// </para>
///
/// <para>
/// ⚠️ EVERY FIXTURE'S OUTPUT IS DELIBERATELY LARGER THAN <c>inputs[0]</c>. That is the only shape of
/// fixture the old code fails: one whose output happened to match the condition's size would pass against
/// the bug and prove nothing. Same discipline as feeding a resampler content above the destination Nyquist.
/// </para>
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// A fixture value as float. ⚠️ A BOOL arrives as a JSON `true`/`false` token, and GetDouble() throws
    /// on it - an If's condition is exactly that, so reading it as a number failed the test before the
    /// engine was ever exercised.
    /// </summary>
    private static float FixtureFloat(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.True => 1f,
        JsonValueKind.False => 0f,
        _ => (float)e.GetDouble(),
    };

    private async Task ControlFlowMatchesOnnxRuntime(string name)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var json = await http.GetStringAsync($"references/controlflow/{name}.json");
        var modelBytes = await http.GetByteArrayAsync($"references/controlflow/{name}.onnx");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var inputShapes = new Dictionary<string, int[]>();
        foreach (var p in root.GetProperty("inputs").EnumerateObject())
            inputShapes[p.Name] = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        await RunTest(async accelerator =>
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes,
                inputShapes: inputShapes.ToDictionary(kv => kv.Key,
                    kv => kv.Value.Length == 0 ? new[] { 1 } : kv.Value));

            // Feed every declared input. A rank-0 scalar becomes a 1-element tensor, which is how the
            // engine represents it.
            var feeds = new Dictionary<string, Tensor>();
            var buffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            try
            {
                foreach (var p in root.GetProperty("inputs").EnumerateObject())
                {
                    var data = p.Value.GetProperty("data").EnumerateArray()
                        .Select(FixtureFloat).ToArray();
                    if (data.Length == 0) data = new[] { 0f };
                    var shape = inputShapes[p.Name];
                    if (shape.Length == 0) shape = new[] { 1 };
                    var buf = accelerator.Allocate1D(data);
                    buffers.Add(buf);
                    feeds[p.Name] = new Tensor(buf.View, shape);
                }

                var outputs = await session.RunAsync(feeds);

                foreach (var p in root.GetProperty("outputs").EnumerateObject())
                {
                    var expected = p.Value.GetProperty("data").EnumerateArray()
                        .Select(FixtureFloat).ToArray();
                    var expectedShape = p.Value.GetProperty("shape").EnumerateArray()
                        .Select(e => e.GetInt32()).ToArray();

                    if (!outputs.TryGetValue(p.Name, out var gotTensor))
                        throw new Exception($"{name}: no output named '{p.Name}'");

                    // The bug was a too-SMALL buffer, so size is the first thing to check and the message
                    // should say so plainly.
                    if (gotTensor.ElementCount < expected.Length)
                        throw new Exception(
                            $"{name} {p.Name}: output holds {gotTensor.ElementCount} values but "
                            + $"{expected.Length} are expected (shape [{string.Join(",", expectedShape)}]). "
                            + "A buffer sized from inputs[0] instead of the subgraph is what caused this.");

                    using var host = accelerator.Allocate1D<float>(expected.Length);
                    await host.View.CopyFromAsync(gotTensor.Data.SubView(0, expected.Length));
                    await accelerator.SynchronizeAsync();
                    var got = await host.CopyToHostAsync<float>(0, expected.Length);

                    double worst = 0; int worstAt = -1;
                    for (int i = 0; i < expected.Length; i++)
                    {
                        double d = Math.Abs(got[i] - expected[i]);
                        if (d > worst) { worst = d; worstAt = i; }
                    }
                    if (worst > 1e-4)
                        throw new Exception($"{name} {p.Name}: max |d| {worst:E3} at {worstAt} "
                                          + $"(ORT {expected[worstAt]:F6} vs ours {got[worstAt]:F6})");

                    Console.WriteLine($"[ControlFlow] {name} {p.Name}: {expected.Length} values, max |d| {worst:E2}");
                }
            }
            finally
            {
                foreach (var b in buffers) b.Dispose();
            }
        });
    }

    /// <summary>
    /// A Slice whose input LENGTH depends on which If branch ran. Gates the 2026-09-01 defect: the compiler
    /// resolved this Slice's window at COMPILE time from the only branch it can see, the runtime cascade
    /// preferred those stale params, the window collapsed to an EMPTY output - and a zero-element output
    /// SKIPS the operator, leaving every downstream consumer reading a pooled buffer nobody wrote.
    /// The fixture deliberately takes the LONGER branch, which is the one no compiler can fold.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_BranchSlice_MatchesOnnxRuntime() => await ControlFlowMatchesOnnxRuntime("branch_slice");

    /// <summary>
    /// Sign and Abs over the SAME runtime-shaped [N,1] vector. Gates the 2026-09-01 defect where the
    /// executor's allowlist of unary ops permitted to adopt their input's RUNTIME shape contained Abs but
    /// not Sign, so Sign kept a compile-time [1] and collapsed the vector to a single scalar. Both outputs
    /// are checked, so the test fails on the LENGTH disagreement rather than on a value that looks plausible.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_BranchUnary_MatchesOnnxRuntime() => await ControlFlowMatchesOnnxRuntime("branch_unary");

    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_If_MatchesOnnxRuntime() => await ControlFlowMatchesOnnxRuntime("tiny_if");

    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_Loop_MatchesOnnxRuntime() => await ControlFlowMatchesOnnxRuntime("tiny_loop");

    [TestMethod(Timeout = 180000)]
    public async Task ControlFlow_Scan_MatchesOnnxRuntime() => await ControlFlowMatchesOnnxRuntime("tiny_scan");
}
