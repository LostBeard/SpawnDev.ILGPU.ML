using System;
using System.Collections.Generic;
using System.IO;
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
    /// <see cref="WeightStorage.Half"/> stores the FP32 weights of low-precision-capable ops as FP16 (compute stays FP32):
    /// the session must report FP16 weights and match onnxruntime within FP16 rounding. The default must store none
    /// and match exactly - the opt-in is the ONLY way the eligible set widens to FusedLinear.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task WeightStorage_Half_StoresFp16AndStaysClose()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/fusion/erf_gelu.json");
        var modelBytes = await http.GetByteArrayAsync("references/fusion/erf_gelu.onnx");
        using var doc = JsonDocument.Parse(json);
        var x = doc.RootElement.GetProperty("inputs").GetProperty("X").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var expected = doc.RootElement.GetProperty("outputs").GetProperty("Y").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();

        await RunTest(async accelerator =>
        {
            foreach (var storage in new[] { WeightStorage.Source, WeightStorage.Half })
            {
                using var ms = new MemoryStream(modelBytes);
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, ms,
                    inputShapes: new Dictionary<string, int[]> { ["X"] = new[] { 8, 16 } }, weightStorage: storage);
                if (storage == WeightStorage.Half ? session.HalfWeightCount == 0 : session.HalfWeightCount != 0)
                    throw new Exception($"{storage}: {session.HalfWeightCount} FP16 weight(s) on {BackendName}");
                using var xb = accelerator.Allocate1D(x);
                var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xb.View, new[] { 8, 16 }) });
                using var host = accelerator.Allocate1D<float>(expected.Length);
                await host.View.CopyFromAsync(outs["Y"].Data.SubView(0, expected.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, expected.Length);
                session.ReturnOutputs(outs);
                double worst = 0, scale = expected.Max(v => Math.Abs((double)v));
                for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / scale);
                double tol = storage == WeightStorage.Half ? 5e-3 : 1e-5;
                if (worst > tol)
                    throw new Exception($"{storage}: max |d| / max|y| {worst:E2} > {tol:E0} vs onnxruntime on {BackendName}");
            }
        });
    }

    /// <summary>
    /// An FP32 model whose weights are STORED as FP16 (tools/onnx-weights-fp16.py: <c>W__fp16 -> Cast(to=FLOAT) -> W</c>,
    /// half the file) must run exactly like its FP32 twin: the Cast is folded at load (the weight upcast once on the GPU),
    /// so the executed graph has the SAME node count as the FP32 model and no FP16 weights by default, and the result
    /// matches onnxruntime on the FP16-weight file. Under <see cref="WeightStorage.Half"/> the folded weight stays FP16.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task WeightStorage_Fp16StoredWeights_CastFoldedAtLoad()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/fusion/half_linear_fp16w.json");
        var fp16Model = await http.GetByteArrayAsync("references/fusion/half_linear_fp16w.onnx");
        var fp32Model = await http.GetByteArrayAsync("references/fusion/half_linear.onnx");
        using var doc = JsonDocument.Parse(json);
        var x = doc.RootElement.GetProperty("inputs").GetProperty("X").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var expected = doc.RootElement.GetProperty("outputs").GetProperty("Y").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var shapes = new Dictionary<string, int[]> { ["X"] = new[] { 4, 256 } };

        await RunTest(async accelerator =>
        {
            int fp32Nodes;
            using (var ms32 = new MemoryStream(fp32Model))
            using (var s32 = await InferenceSession.CreateFromStreamAsync(accelerator, ms32, inputShapes: shapes))
                fp32Nodes = s32.NodeCount;
            foreach (var storage in new[] { WeightStorage.Source, WeightStorage.Half })
            {
                using var ms = new MemoryStream(fp16Model);
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, ms, inputShapes: shapes, weightStorage: storage);
                if (session.NodeCount != fp32Nodes)
                    throw new Exception($"{storage}: {session.NodeCount} nodes vs {fp32Nodes} for the FP32 model - the weight Cast was not folded ({BackendName})");
                if (storage == WeightStorage.Source ? session.HalfWeightCount != 0 : session.HalfWeightCount == 0)
                    throw new Exception($"{storage}: {session.HalfWeightCount} FP16 weight(s) on {BackendName}");
                using var xb = accelerator.Allocate1D(x);
                var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xb.View, new[] { 4, 256 }) });
                using var host = accelerator.Allocate1D<float>(expected.Length);
                await host.View.CopyFromAsync(outs["Y"].Data.SubView(0, expected.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, expected.Length);
                session.ReturnOutputs(outs);
                double worst = 0, scale = expected.Max(v => Math.Abs((double)v));
                for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / scale);
                // Source: the FP32 compute onnxruntime does on the same FP16-rounded weights - exact up to FP32 order
                double tol = storage == WeightStorage.Half ? 5e-3 : 1e-5;
                if (worst > tol)
                    throw new Exception($"{storage}: max |d| / max|y| {worst:E2} > {tol:E0} vs onnxruntime on the FP16-weight model ({BackendName})");
            }
        });
    }

    /// <summary>
    /// <see cref="WeightStorage.Half"/> through a BROWSER load: an FP32 weight (256x512 = 512 KB) far over
    /// <c>BrowserBufferPolicy.StrictHostCopyMaxBytes</c> (64 KB, armed by InferenceSession for an IJSReadStream load).
    /// The FP32->FP16 downcast must happen on the GPU from a JS->GPU streamed temp - a managed downcast pushes the
    /// weight through the .NET heap and the guard refuses it (it did: Half storage could not load in a browser at all).
    /// Desktop lanes load the same file from a FileStream and must give the same answer.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task WeightStorage_Half_BrowserStreamLoad_StaysOffHeap()
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        var json = await http.GetStringAsync("references/fusion/half_linear.json");
        var url = new Uri(http.BaseAddress!, "references/fusion/half_linear.onnx").ToString();
        using var doc = JsonDocument.Parse(json);
        var x = doc.RootElement.GetProperty("inputs").GetProperty("X").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        var expected = doc.RootElement.GetProperty("outputs").GetProperty("Y").GetProperty("data").EnumerateArray().Select(FixtureFloat).ToArray();
        const int weightBytes = 256 * 512 * 4;

        await RunTest(async accelerator =>
        {
            foreach (var storage in new[] { WeightStorage.Source, WeightStorage.Half })
            {
                await using var stream = await OpenSeekableModelStreamAsync(url);
                bool jsStream = stream is SpawnDev.SpawnJS.Toolbox.IJSReadStream;
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, stream,
                    inputShapes: new Dictionary<string, int[]> { ["X"] = new[] { 4, 256 } }, weightStorage: storage);
                if (storage == WeightStorage.Half ? session.HalfWeightCount == 0 : session.HalfWeightCount != 0)
                    throw new Exception($"{storage}: {session.HalfWeightCount} FP16 weight(s) on {BackendName}");
                // On a JS stream the big weight must have gone JS->GPU (the counter only counts that path).
                if (jsStream && session.ZeroCopyWeightBytes < weightBytes)
                    throw new Exception($"{storage}: only {session.ZeroCopyWeightBytes} B zero-copy (< {weightBytes}) on {BackendName}");
                using var xb = accelerator.Allocate1D(x);
                var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["X"] = new Tensor(xb.View, new[] { 4, 256 }) });
                using var host = accelerator.Allocate1D<float>(expected.Length);
                await host.View.CopyFromAsync(outs["Y"].Data.SubView(0, expected.Length));
                await accelerator.SynchronizeAsync();
                var got = await host.CopyToHostAsync<float>(0, expected.Length);
                session.ReturnOutputs(outs);
                double worst = 0, scale = expected.Max(v => Math.Abs((double)v));
                for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - expected[i]) / scale);
                double tol = storage == WeightStorage.Half ? 5e-3 : 1e-5;
                if (worst > tol)
                    throw new Exception($"{storage}: max |d| / max|y| {worst:E2} > {tol:E0} vs onnxruntime on {BackendName} (JS stream: {jsStream})");
                Console.WriteLine($"[WeightStorage] {BackendName} {storage}: rel {worst:E2}, fp16 weights {session.HalfWeightCount}, zero-copy {session.ZeroCopyWeightBytes} B, JS stream {jsStream}");
            }
        });
    }
}
