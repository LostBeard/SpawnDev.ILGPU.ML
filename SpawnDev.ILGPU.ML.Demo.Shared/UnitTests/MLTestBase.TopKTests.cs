using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ONNX TopK at production scale (2026-10-01). The operator read K only from the opset-1 <c>k</c> attribute, so every
/// opset-10+ TopK (K is input[1]) ran as k=1; it also ignored <c>axis</c> and <c>largest</c>, and its selection
/// kernel costs k(k+1)/2 passes over each row. RaCo-ALIKED (SpawnScene's feature front end) takes k=2304 of
/// [2,5,65536] and of [2,11520]. Data is integer-valued (0..999), so ties are everywhere and the order among equal
/// values - lower index first, which onnxruntime matches - is checked too. Reference: a CPU sort of (value, index).
/// </summary>
public abstract partial class MLTestBase
{
    static float TopKTestValue(long i) => (float)((((ulong)i * 2654435761UL) % 4294967296UL >> 7) % 1000UL);

    async Task RunTopKCase(Accelerator accelerator, string model, int[] shape, int k, int axis, bool largest, bool kAsInput)
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync($"models/tests/{model}");
        int n = 1;
        foreach (var d in shape) n *= d;
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = TopKTestValue(i);
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        // A constant K is resolved at COMPILE time, so the buffers are planned at k, not the axis-length upper bound.
        if (!kAsInput && session.OutputShapes.TryGetValue("v", out var compiledV) && compiledV[axis] != k)
            throw new Exception($"{model}: compiled v shape [{string.Join(",", compiledV)}] - constant K={k} was not resolved at compile time");
        using var xB = accelerator.Allocate1D(x);
        using var kB = accelerator.Allocate1D(new float[] { k });
        var feeds = new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, shape) };
        if (kAsInput) feeds["k"] = new Tensor(kB.View, new[] { 1 });
        var outs = await session.RunAsync(feeds);
        var v = outs["v"]; var fi = outs["fi"];
        var want = (int[])shape.Clone();
        want[axis] = k;
        if (!v.Shape.SequenceEqual(want) || !fi.Shape.SequenceEqual(want))
            throw new Exception($"{model}: shapes v [{string.Join(",", v.Shape)}] i [{string.Join(",", fi.Shape)}], expected [{string.Join(",", want)}]");
        int outN = n / shape[axis] * k;
        using var vH = accelerator.Allocate1D<float>(outN);
        using var iH = accelerator.Allocate1D<float>(outN);
        await vH.View.CopyFromAsync(v.Data.SubView(0, outN));
        await iH.View.CopyFromAsync(fi.Data.SubView(0, outN));
        await accelerator.SynchronizeAsync();
        var vv = await vH.CopyToHostAsync<float>(0, outN);
        var iv = await iH.CopyToHostAsync<float>(0, outN);

        int outer = 1, inner = 1, axisLen = shape[axis];
        for (int d = 0; d < axis; d++) outer *= shape[d];
        for (int d = axis + 1; d < shape.Length; d++) inner *= shape[d];
        var keys = new long[axisLen];
        int bad = 0; string first = "";
        for (int o = 0; o < outer; o++)
            for (int ii = 0; ii < inner; ii++)
            {
                // (rank value, index): the sort order IS the expected output order.
                for (int p = 0; p < axisLen; p++)
                {
                    long val = (long)x[(o * axisLen + p) * inner + ii];
                    keys[p] = ((largest ? 999 - val : val) << 32) | (long)p;
                }
                Array.Sort(keys);
                for (int j = 0; j < k; j++)
                {
                    int wantIdx = (int)(keys[j] & 0xFFFFFFFF);
                    float wantVal = x[(o * axisLen + wantIdx) * inner + ii];
                    int at = (o * k + j) * inner + ii;
                    if (iv[at] != wantIdx || vv[at] != wantVal)
                    {
                        if (bad++ == 0) first = $"[{o},{j},{ii}] got ({vv[at]}, {iv[at]}) expected ({wantVal}, {wantIdx})";
                    }
                }
            }
        if (bad > 0) throw new Exception($"{model}: {bad} of {outN} outputs wrong, first {first}");
        // Shape(v), where the model has it: the compiler folds it from the COMPILE-TIME shape.
        if (outs.TryGetValue("svf", out var svf))
        {
            using var sH = accelerator.Allocate1D<float>(svf.ElementCount);
            await sH.View.CopyFromAsync(svf.Data.SubView(0, svf.ElementCount));
            await accelerator.SynchronizeAsync();
            var sv = await sH.CopyToHostAsync<float>(0, svf.ElementCount);
            if (!sv.Select(f => (int)f).SequenceEqual(want))
                throw new Exception($"{model}: Shape(v) = [{string.Join(",", sv)}], expected [{string.Join(",", want)}]");
        }
    }

    /// <summary>RaCo's first keypoint TopK: k=2304 of 65,536 per row (the sort path).</summary>
    [TestMethod(Timeout = 120000)]
    public async Task TopK_RaCoRows_K2304() => await RunTest(async accelerator =>
        await RunTopKCase(accelerator, "topk_raco_rows.onnx", new[] { 2, 5, 65536 }, 2304, 2, true, false));

    /// <summary>A middle axis with inner &gt; 1, and largest=0.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task TopK_Axis1_Smallest() => await RunTest(async accelerator =>
        await RunTopKCase(accelerator, "topk_axis1_smallest.onnx", new[] { 2, 300, 7 }, 50, 1, false, false));

    /// <summary>K as a graph INPUT: only the runtime knows it, so the executor must resize the outputs.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task TopK_RuntimeK() => await RunTest(async accelerator =>
        await RunTopKCase(accelerator, "topk_runtime_k.onnx", new[] { 3, 1000 }, 37, 1, true, true));
}
