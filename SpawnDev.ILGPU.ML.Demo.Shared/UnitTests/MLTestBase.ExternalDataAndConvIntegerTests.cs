using ILGPU;
using ILGPU.Runtime;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ConvInteger (what a quantized FastConformer's convolutions are) matches onnxruntime: a NON-square stride with asymmetric pads and
    /// dilation, a depthwise group, and a rank-3 (1-D) convolution, each with real zero points. Integer arithmetic below 2^24, so the
    /// float engine must be EXACT (tolerance 1e-6).
    /// </summary>
    /// <remarks>
    /// The old operator read the activation on the host (only possible for constants of at most 64 elements), fell back to a Conv2D with
    /// an EMPTY bias view (a CUDA illegal memory access that poisons the context for every model in the process - Parakeet TDT's int8
    /// encoder, its first convolution), ignored both zero points, returned zeros for rank 3, and used strides[0] / pads[0] for both axes.
    /// onnxruntime supports only a per-tensor w_zero_point, so the per-channel form (implemented per the ONNX spec) is not asserted here.
    /// </remarks>
    [TestMethod(Timeout = 180000)]
    public async Task Operator_ConvInteger_MatchesOnnxRuntime() => await FusionFixture("convinteger", _ => null, tolerance: 1e-6);

    /// <summary>
    /// A model whose weights live in a .onnx_data file honours <see cref="WeightStorage.Half"/>: the weight of a MatMul (eligible) is stored
    /// fp16, an elementwise Mul's weight and a rank-3 (1-D) Conv's weight stay fp32, and every output matches onnxruntime (the fp16 one within
    /// fp16 rounding).
    /// </summary>
    /// <remarks>
    /// Two defects. The external-data loop never reached the half branch, so for Parakeet TDT's encoder (and Depth Anything V3, and most
    /// onnx-community exports) the opt-in silently did nothing - MEASURED 3,011 MiB resident for the fp32 and the "fp16" encoder alike, 2,012 MiB
    /// once honoured. And the half-eligibility rule admitted ANY group-1 Conv weight, but the half Conv path is 2-D only: Parakeet's pointwise
    /// 1-D convs threw "reached an unsupported path ... rank=3" at execute time.
    /// </remarks>
    [TestMethod(Timeout = 180000)]
    public async Task WeightStorage_Half_AppliesToExternalDataWeights_AndKeepsIneligibleOnesFp32()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/fusion/external_half.json");
        var model = await http.GetByteArrayAsync("references/fusion/external_half.onnx");
        var data = await http.GetByteArrayAsync("references/fusion/external_half.onnx.data");
        using var doc = JsonDocument.Parse(json);
        float[] Read(JsonElement e) => e.GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var x = Read(doc.RootElement.GetProperty("inputs").GetProperty("x"));
        var xc = Read(doc.RootElement.GetProperty("inputs").GetProperty("xc"));
        var shapes = new System.Collections.Generic.Dictionary<string, int[]> { ["x"] = new[] { 1, 8, 256 }, ["xc"] = new[] { 1, 2, 16 } };

        await RunTest(async accelerator =>
        {
            foreach (var storage in new[] { WeightStorage.Source, WeightStorage.Half })
            {
                using var ms = new MemoryStream(model);
                using var ext = new MemoryStream(data);
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, ms, inputShapes: shapes,
                    externalDataStream: ext, weightStorage: storage);
                var wantHalf = storage == WeightStorage.Half ? 1 : 0;
                if (session.HalfWeightCount != wantHalf)
                    throw new Exception($"{storage}: {session.HalfWeightCount} fp16 weight(s), expected {wantHalf} (the MatMul weight only) on {BackendName}");

                using var xb = accelerator.Allocate1D(x);
                using var xcb = accelerator.Allocate1D(xc);
                var outs = await session.RunAsync(new System.Collections.Generic.Dictionary<string, Tensor>
                {
                    ["x"] = new Tensor(xb.View, shapes["x"]), ["xc"] = new Tensor(xcb.View, shapes["xc"]),
                });
                foreach (var (name, tol) in new[] { ("y1", storage == WeightStorage.Half ? 5e-3 : 1e-5), ("y2", 1e-5), ("y3", 1e-5) })
                {
                    var expected = Read(doc.RootElement.GetProperty("outputs").GetProperty(name));
                    using var host = accelerator.Allocate1D<float>(expected.Length);
                    await host.View.CopyFromAsync(outs[name].Data.SubView(0, expected.Length));
                    await accelerator.SynchronizeAsync();
                    var got = await host.CopyToHostAsync<float>(0, expected.Length);
                    double worst = 0, scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));
                    for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / scale);
                    if (worst > tol)
                        throw new Exception($"{storage} {name}: max |d| / max|y| {worst:E2} > {tol:E0} vs onnxruntime on {BackendName}");
                }
                session.ReturnOutputs(outs);
            }
        });
    }
}
