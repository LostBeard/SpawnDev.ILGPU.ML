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
    private async Task FusionFixture(string name, Func<HashSet<string>, string?> checkOps, double tolerance = 1e-5)
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
                    if (worst > tolerance)
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
    /// MatMul + Add(SCALAR) is not a bias (Parakeet's log-mel ends in `MatMul -> Add(0)`): it must stay two nodes and
    /// match onnxruntime, while MatMul + Add(b[N]) in the same graph still fuses into FusedLinear. Before the fix the
    /// scalar path fused and threw "FusedLinear bounds mismatch ... bias=[] buf=1".
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Fusion_MatMulPlusScalarAdd_IsNotABias_MatchesOnnxRuntime() => await FusionFixture("matmul_scalar_add",
        ops => !ops.Contains("FusedLinear") ? "the real per-column bias no longer fuses into FusedLinear"
             : !ops.Contains("MatMul") || !ops.Contains("Add") ? "the scalar Add was fused into a FusedLinear as if it were a bias" : null);

    /// <summary>
    /// STFT with a 512-sample Hann window, hop 160, on a rank-2 signal (NeMo's log-mel preprocessor): 23 frames x 257 bins,
    /// matching onnxruntime. Before the fixes the window (over the 64-element host-value limit) was dropped and frame_length
    /// fell back to the hop - right shape, 97% wrong values - and the hop folded AFTER shape inference gave a 128-hop frame count.
    /// Tolerance 2e-3, MEASURED not chosen: this kernel is a direct float32 DFT whose twiddle angle reaches ~1,600 rad
    /// (2*pi*k*n/L, k up to 256, n up to 511). CPU, OpenCL and Wasm land under 5e-4; CUDA, WebGPU and WebGL use fast
    /// sin/cos, which lose accuracy at large arguments, and all three measure 8.1e-4 (the kernel's own comment records
    /// why the angle is deliberately not reduced: Kokoro's Nyquist-bin phase). The defects this test exists for are three
    /// orders of magnitude larger: a dropped window was 97% wrong, a 128-hop frame count changes the output SHAPE.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Operator_Stft_LongWindow_MatchesOnnxRuntime() => await FusionFixture("stft_long_window", _ => null, tolerance: 2e-3);

    /// <summary>
    /// Tile of a COMPUTED tensor ([1,1,T] repeated [1,T,1], a conformer's relative-position index) matches onnxruntime.
    /// The general N-D Tile used to require the input's values on the host and threw for anything computed.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Operator_Tile_OfAComputedTensor_MatchesOnnxRuntime() => await FusionFixture("tile_computed", _ => null);

    /// <summary>
    /// A Transpose that only moves size-1 axes is a reshape: the single-consumer one is handed off zero-copy, the one
    /// of a graph input with two consumers takes TransposeOperator's native-copy path. Both must match onnxruntime.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Fusion_TransposeSizeOneAxes_MatchesOnnxRuntime() => await FusionFixture("transpose_view", _ => null);

    /// <summary>
    /// Two operands with the SAME element count but different shapes ([8,1] and [1,8]) broadcast to [8,8] in Add, Sub, Mul, Div, Min and Max,
    /// matching onnxruntime. Sub treated equal counts as equal shapes and subtracted element by element: the Nemotron streaming encoder's
    /// attention mask (Sub of [72,1] and [1,72]) came out wrong, so the model attended where it must not and the word error rate rose from
    /// 5.6% to 7.2% without any error.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Operator_BinaryOps_BroadcastOuterPairsOfEqualSize_MatchOnnxRuntime() => await FusionFixture("outer_broadcast", _ => null);
}
