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

public abstract partial class MLTestBase
{
    /// <summary>
    /// Runs a fixture from references/fusion against its onnxruntime outputs, then checks the compiled operator set.
    /// The three passes these gate change WHAT the engine executes (fewer dispatches) and must not change results.
    /// </summary>
    private async Task FusionFixture(string name, Func<HashSet<string>, string?> checkOps)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync($"references/fusion/{name}.json");
        var modelBytes = await http.GetByteArrayAsync($"references/fusion/{name}.onnx");
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var shapes = root.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name,
            p => p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray());

        await RunTest(async accelerator =>
        {
            using var session = InferenceSession.CreateFromFile(accelerator, modelBytes, inputShapes: shapes);
            var ops = new HashSet<string>(session.OperatorTypes, StringComparer.Ordinal);
            var opProblem = checkOps(ops);
            if (opProblem != null)
                throw new Exception($"{name}: {opProblem} (compiled ops: {string.Join(", ", ops.OrderBy(o => o))}) on {BackendName}");

            var bufs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            try
            {
                var feeds = new Dictionary<string, Tensor>();
                foreach (var p in root.GetProperty("inputs").EnumerateObject())
                {
                    var b = accelerator.Allocate1D(p.Value.GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray());
                    bufs.Add(b);
                    feeds[p.Name] = new Tensor(b.View, shapes[p.Name]);
                }
                var outs = await session.RunAsync(feeds);
                foreach (var p in root.GetProperty("outputs").EnumerateObject())
                {
                    var expected = p.Value.GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
                    var got = outs[p.Name];
                    if (got.ElementCount != expected.Length)
                        throw new Exception($"{name} {p.Name}: {got.ElementCount} values, ORT has {expected.Length} on {BackendName}");
                    using var host = accelerator.Allocate1D<float>(expected.Length);
                    await host.View.CopyFromAsync(got.Data.SubView(0, expected.Length));
                    await accelerator.SynchronizeAsync();
                    var g = await host.CopyToHostAsync<float>(0, expected.Length);
                    double worst = 0;
                    for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(g[i] - expected[i]) / Math.Max(1.0, Math.Abs(expected[i])));
                    if (worst > 1e-5)
                        throw new Exception($"{name} {p.Name}: max rel {worst:E2} vs onnxruntime on {BackendName}");
                }
                session.ReturnOutputs(outs);
            }
            finally { foreach (var b in bufs) b.Dispose(); }
        });
    }

    /// <summary>
    /// torch's exact GELU (Div, Erf, Add, Mul, Mul) - and the two other association orders - fuse into Gelu, and the
    /// results match onnxruntime. 60 dispatches per DINOv2 ViT-S forward (Video Depth Anything, DAv3).
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Fusion_ErfGelu_MatchesOnnxRuntime() => await FusionFixture("erf_gelu",
        ops => ops.Contains("Erf") ? "an Erf survived: FuseErfGelu did not claim every GELU chain" : null);

    /// <summary>torch's GroupNorm export fuses into one GroupNormalization and matches onnxruntime.</summary>
    [TestMethod(Timeout = 180000)]
    public async Task Fusion_GroupNorm_MatchesOnnxRuntime() => await FusionFixture("group_norm",
        ops => ops.Contains("InstanceNormalization") || !ops.Contains("GroupNormalization")
            ? "the GroupNorm chain was not fused into GroupNormalization" : null);

    /// <summary>
    /// A Transpose that only moves size-1 axes is a reshape: the single-consumer one is handed off zero-copy, the one
    /// of a graph input with two consumers takes TransposeOperator's native-copy path. Both must match onnxruntime.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Fusion_TransposeSizeOneAxes_MatchesOnnxRuntime() => await FusionFixture("transpose_view", _ => null);
}
