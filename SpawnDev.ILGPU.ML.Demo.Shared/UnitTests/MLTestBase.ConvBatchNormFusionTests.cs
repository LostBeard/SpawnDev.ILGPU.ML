using System.Diagnostics;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// GraphOptimizer's "convbn" pass (Conv -> BatchNormalization folded into the Conv on its first run): MobileNetV2 three ways on the same inputs, the unfused graph
    /// (ML_OPT_SKIP=convbn), the folded weights, and the unfolded fallback the half / low-precision weights take. The three agree (folding rounds differently, so not bit for bit:
    /// within 1e-5 of the output's range, same top-1), the folded graph runs fewer nodes, and its warm run is reported against the unfused one.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ConvBatchNorm_Fusion_MobileNetV2_MatchesUnfused() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        var rng = new Random(9);
        var pic = new float[3 * 224 * 224]; for (int i = 0; i < pic.Length; i++) pic[i] = (float)(rng.NextDouble() * 2 - 1);
        async Task<(float[] Out, int Nodes, double WarmMs)> RunModel(bool fuse, bool fallback)
        {
            var savedSkip = Environment.GetEnvironmentVariable("ML_OPT_SKIP");
            Environment.SetEnvironmentVariable("ML_OPT_SKIP", fuse ? savedSkip : string.IsNullOrEmpty(savedSkip) ? "convbn" : savedSkip + ",convbn");
            Operators.ConvOperator.ForceBatchNormUnfolded = fallback;
            try
            {
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
                int fusedPairs = Graph.GraphOptimizer.LastConvBatchNormFused;
                async Task<float[]> Once()
                {
                    using var t = OwnedTensor<float>.FromHost(accelerator, pic, new[] { 1, 3, 224, 224 });
                    using var o = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t });
                    return await o[session.OutputNames[0]].ToHostAsync();
                }
                var first = await Once();
                for (int i = 0; i < 4; i++) await Once();
                var times = new List<double>();
                for (int i = 0; i < 15; i++) { var sw = Stopwatch.StartNew(); await Once(); times.Add(sw.Elapsed.TotalMilliseconds); }
                times.Sort();
                var again = await Once();
                for (int i = 0; i < first.Length; i++)
                    if (BitConverter.SingleToInt32Bits(first[i]) != BitConverter.SingleToInt32Bits(again[i]))
                        throw new Exception($"fuse={fuse} fallback={fallback}: run 1 and run 21 differ at {i} ({first[i]} vs {again[i]}): a fold applied twice?");
                return (first, fuse ? fusedPairs : 0, times[times.Count / 2]);
            }
            finally
            {
                Environment.SetEnvironmentVariable("ML_OPT_SKIP", savedSkip);
                Operators.ConvOperator.ForceBatchNormUnfolded = false;
            }
        }
        var plain = await RunModel(fuse: false, fallback: false);
        var folded = await RunModel(fuse: true, fallback: false);
        var unfolded = await RunModel(fuse: true, fallback: true);
        static (double Rel, int Top) Compare(float[] a, float[] b)
        {
            double maxDiff = 0, maxMag = 0; int ta = 0, tb = 0;
            for (int i = 0; i < a.Length; i++) { maxDiff = Math.Max(maxDiff, Math.Abs(a[i] - b[i])); maxMag = Math.Max(maxMag, Math.Abs(a[i])); if (a[i] > a[ta]) ta = i; if (b[i] > b[tb]) tb = i; }
            return (maxDiff / maxMag, ta == tb ? ta : -1);
        }
        var (relFold, topFold) = Compare(plain.Out, folded.Out);
        var (relBack, topBack) = Compare(plain.Out, unfolded.Out);
        var report = $"{accelerator.AcceleratorType}: {folded.Nodes} Conv + BatchNorm pairs fused; folded vs unfused max diff {relFold:E1} of the output's range (top-1 {(topFold >= 0 ? "same" : "DIFFERENT")}), " +
                     $"unfolded fallback {relBack:E1} (top-1 {(topBack >= 0 ? "same" : "DIFFERENT")}); warm median: unfused {plain.WarmMs:F1} ms, folded {folded.WarmMs:F1} ms, fallback {unfolded.WarmMs:F1} ms";
        Console.WriteLine("[ConvBn] " + report);
        if (folded.Nodes < 1) throw new Exception("the pass fused nothing in MobileNetV2: " + report);
        if (relFold > 1e-5 || topFold < 0) throw new Exception("folded output differs from the unfused graph: " + report);
        if (relBack > 1e-6 || topBack < 0) throw new Exception("the unfolded fallback differs from the unfused graph: " + report);
        return report;
    });

    /// <summary>
    /// GraphOptimizer's "convact" pass (Conv -> Relu / Clip(0, 6) applied in the convolution's stores): MobileNetV2 with it off (ML_OPT_SKIP=convact), on, and on with the activation
    /// forced to the separate in-place pass the kernels without the epilogue use. The epilogue is the standalone op's own expression, so all three must agree bit for bit.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task<string> ConvActivation_Fusion_MobileNetV2_BitIdentical() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        var bytes = await http.GetByteArrayAsync("models/mobilenetv2/model.onnx");
        var rng = new Random(13);
        var pic = new float[3 * 224 * 224]; for (int i = 0; i < pic.Length; i++) pic[i] = (float)(rng.NextDouble() * 2 - 1);
        async Task<(float[] Out, int Fused, double WarmMs)> RunModel(bool fuse, bool inPlace)
        {
            var savedSkip = Environment.GetEnvironmentVariable("ML_OPT_SKIP");
            Environment.SetEnvironmentVariable("ML_OPT_SKIP", fuse ? savedSkip : string.IsNullOrEmpty(savedSkip) ? "convact" : savedSkip + ",convact");
            Operators.ConvOperator.ForceActivationInPlace = inPlace;
            try
            {
                using var session = await InferenceSession.CreateFromStreamAsync(accelerator, new MemoryStream(bytes, writable: false));
                int fused = Graph.GraphOptimizer.LastConvActivationFused;
                async Task<float[]> Once()
                {
                    using var t = OwnedTensor<float>.FromHost(accelerator, pic, new[] { 1, 3, 224, 224 });
                    using var o = await session.RunOwnedAsync(new Dictionary<string, Tensor<float>> { [session.InputNames[0]] = t });
                    return await o[session.OutputNames[0]].ToHostAsync();
                }
                var first = await Once();
                for (int i = 0; i < 4; i++) await Once();
                var times = new List<double>();
                for (int i = 0; i < 15; i++) { var sw = Stopwatch.StartNew(); await Once(); times.Add(sw.Elapsed.TotalMilliseconds); }
                times.Sort();
                return (first, fuse ? fused : 0, times[times.Count / 2]);
            }
            finally
            {
                Environment.SetEnvironmentVariable("ML_OPT_SKIP", savedSkip);
                Operators.ConvOperator.ForceActivationInPlace = false;
            }
        }
        var plain = await RunModel(fuse: false, inPlace: false);
        var fused = await RunModel(fuse: true, inPlace: false);
        var inPlaceRun = await RunModel(fuse: true, inPlace: true);
        static int Differ(float[] a, float[] b) { int d = 0; for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) d++; return d; }
        int dFused = Differ(plain.Out, fused.Out), dInPlace = Differ(plain.Out, inPlaceRun.Out);
        var report = $"{accelerator.AcceleratorType}: {fused.Fused} Conv + activation pairs fused; values differing from the unfused graph: epilogue {dFused}, in-place pass {dInPlace} of {plain.Out.Length}; " +
                     $"warm median: unfused {plain.WarmMs:F1} ms, epilogue {fused.WarmMs:F1} ms, in-place {inPlaceRun.WarmMs:F1} ms";
        Console.WriteLine("[ConvAct] " + report);
        if (fused.Fused < 1) throw new Exception("the pass fused nothing in MobileNetV2: " + report);
        if (dFused != 0 || dInPlace != 0) throw new Exception("a fused activation changed the output: " + report);
        return report;
    });
}
