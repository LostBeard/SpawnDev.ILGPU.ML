using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ReduceL1 / L2 / SumSquare / LogSum / LogSumExp as OPERATORS, through a session (2026-10-01). ReduceL2 squared with
    /// Mul(x, x) and finished with Sqrt(out, out) - one buffer in two storage bindings, which WebGPU rejects
    /// (RaCo-ALIKED's descriptor ReduceL2 failed its first browser forward); the older Op_ReduceL2 tests composed the
    /// kernels by hand with a COPY of the input, so the operator itself never ran. They also read axes only from the
    /// attribute and defaulted to the LAST axis where ONNX reduces ALL. models/tests/reduce_family.onnx (opset 18, axes as
    /// inputs), x [2,3,4] = ((i*7) % 11 - 5) * 0.75: l1 axes [1] keepdims, l2 axes [2], ss axes [1,2], ls = LogSum(|x|)
    /// axes [2], lse no axes (reduce all, scalar), l2b axes [0] keepdims. onnxruntime agrees with the reference here.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task ReduceFamily_Operators_MatchReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/reduce_family.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[24];
        for (int i = 0; i < 24; i++) x[i] = ((i * 7) % 11 - 5) * 0.75f;
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 2, 3, 4 }) });

        // Reference: reduce x [2,3,4] over `axes`, element transform `pre`, final transform `post`, in double.
        static double[] Reduce(float[] x, int[] axes, Func<double, double> pre, Func<double, double> post)
        {
            int[] dims = { 2, 3, 4 };
            int[] outDims = dims.Select((d, i) => axes.Contains(i) ? 1 : d).ToArray();
            var sums = new double[outDims[0] * outDims[1] * outDims[2]];
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 3; b++)
                    for (int c = 0; c < 4; c++)
                    {
                        int oa = axes.Contains(0) ? 0 : a, ob = axes.Contains(1) ? 0 : b, oc = axes.Contains(2) ? 0 : c;
                        sums[(oa * outDims[1] + ob) * outDims[2] + oc] += pre(x[(a * 3 + b) * 4 + c]);
                    }
            return sums.Select(post).ToArray();
        }
        var cases = new (string Name, int[] Shape, double[] Want)[]
        {
            ("l1", new[] { 2, 1, 4 }, Reduce(x, new[] { 1 }, Math.Abs, v => v)),
            ("l2", new[] { 2, 3 }, Reduce(x, new[] { 2 }, v => v * v, Math.Sqrt)),
            ("ss", new[] { 2 }, Reduce(x, new[] { 1, 2 }, v => v * v, v => v)),
            ("ls", new[] { 2, 3 }, Reduce(x, new[] { 2 }, Math.Abs, Math.Log)),
            ("lse", Array.Empty<int>(), Reduce(x, new[] { 0, 1, 2 }, Math.Exp, Math.Log)),
            ("l2b", new[] { 1, 3, 4 }, Reduce(x, new[] { 0 }, v => v * v, Math.Sqrt)),
        };
        var errors = new List<string>();
        foreach (var (name, shape, want) in cases)
        {
            var t = outs[name];
            int n = want.Length;
            // A rank-0 result may come back as [] or [1]; anything else is a wrong reduction.
            bool shapeOk = t.Shape.SequenceEqual(shape) || (shape.Length == 0 && t.Shape.SequenceEqual(new[] { 1 }));
            if (!shapeOk) { errors.Add($"{name} shape [{string.Join(",", t.Shape)}], expected [{string.Join(",", shape)}]"); continue; }
            using var hB = accelerator.Allocate1D<float>(n);
            await hB.View.CopyFromAsync(t.Data.SubView(0, n));
            await accelerator.SynchronizeAsync();
            var got = await hB.CopyToHostAsync<float>(0, n);
            for (int i = 0; i < n; i++)
                if (Math.Abs(got[i] - want[i]) > 1e-4 * (1 + Math.Abs(want[i])))
                    errors.Add($"{name}[{i}] = {got[i]}, expected {want[i]:G9}");
        }
        if (errors.Count > 0) throw new Exception(string.Join(" | ", errors.Take(8)));
    });
}
