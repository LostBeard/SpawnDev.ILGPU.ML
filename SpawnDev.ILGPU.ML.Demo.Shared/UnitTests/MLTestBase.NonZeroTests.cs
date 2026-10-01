using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Sigmoid / Exp / Log(Sigmoid) in the float tails (2026-10-01): Sigmoid clamped x &lt; -80 to 0 (sigmoid(-85) =
    /// 1.2e-37 is a normal float, and Log(Sigmoid) went to -inf); Exp returned +inf for x &gt; 80 (exp(85) = 8.2e36)
    /// and 0 for x &lt; -80. models/tests/sigmoid_exp_tails.onnx; no subnormal results (GPUs may flush them).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Sigmoid_Exp_FloatTails() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/sigmoid_exp_tails.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[] { -87f, -85f, -40f, -14.6f, 0f, 14.6f, 40f, 85f, 88f };
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 9 }) });
        async Task<float[]> Read(string n) { using var h = accelerator.Allocate1D<float>(9); await h.View.CopyFromAsync(outs[n].Data.SubView(0, 9)); await accelerator.SynchronizeAsync(); return await h.CopyToHostAsync<float>(0, 9); }
        var s = await Read("s"); var e = await Read("e"); var ls = await Read("ls");
        var errors = new List<string>();
        void Check(string what, float got, double want, double absTol = 1e-30)
        {
            if (Math.Abs(got - want) > 2e-5 * Math.Abs(want) + absTol) errors.Add($"{what} = {got:R}, expected {want:G9}");
        }
        for (int i = 0; i < 9; i++)
        {
            double sig = 1.0 / (1.0 + Math.Exp(-x[i]));
            Check($"sigmoid({x[i]})", s[i], sig);
            Check($"exp({x[i]})", e[i], Math.Exp(x[i]));
            // log() of a value just below 1 inherits that value's float rounding: 1 - 4.56e-7 is stored to within half a
            // float ulp (6e-8) of 1, so the absolute error of log(sigmoid(14.6)) is ~3e-8 however exact sigmoid is.
            Check($"log(sigmoid({x[i]}))", ls[i], Math.Log(sig), 1.2e-7);
        }
        if (errors.Count > 0) throw new Exception(string.Join(" | ", errors));
    });

    /// <summary>
    /// LogSoftmax along a NON-last axis, with values far enough apart that log(softmax) underflows (2026-10-01): the
    /// operator ran Softmax over rows of shape[axis] (only right for the last axis) and took log(softmax). LightGlue's
    /// log-assignment LogSoftmax(axis=1) of [1,2048,2048] came out wrong by up to 19. models/tests/logsoftmax_axes.onnx:
    /// x [2,5,4] = -((i*29) % 41) * 3 (0 .. -120); y1 axis=1, y2 axis=-1; checked against a double-precision reference.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task LogSoftmax_AnyAxis_LogSpace() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/logsoftmax_axes.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[40];
        for (int i = 0; i < 40; i++) x[i] = -((i * 29) % 41) * 3f;
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 2, 5, 4 }) });
        foreach (var (name, axis) in new[] { ("y1", 1), ("y2", 2) })
        {
            using var h = accelerator.Allocate1D<float>(40);
            await h.View.CopyFromAsync(outs[name].Data.SubView(0, 40));
            await accelerator.SynchronizeAsync();
            var y = await h.CopyToHostAsync<float>(0, 40);
            int[] dims = { 2, 5, 4 };
            for (int t = 0; t < 40; t++)
            {
                int b = t / 20, r = t % 20, m = r / 4, k = r % 4;
                int[] idx = { b, m, k };
                double mx = double.NegativeInfinity, sum = 0;
                for (int a = 0; a < dims[axis]; a++) { idx[axis] = a; mx = Math.Max(mx, x[idx[0] * 20 + idx[1] * 4 + idx[2]]); }
                for (int a = 0; a < dims[axis]; a++) { idx[axis] = a; sum += Math.Exp(x[idx[0] * 20 + idx[1] * 4 + idx[2]] - mx); }
                double want = x[t] - mx - Math.Log(sum);
                if (Math.Abs(y[t] - want) > 1e-4 * (1 + Math.Abs(want)))
                    throw new Exception($"{name} (axis {axis})[{t}] = {y[t]}, expected {want:G9}");
            }
        }
    });

    /// <summary>
    /// NonZero with a data-dependent count, consumed the way LightGlue builds its match list (2026-10-01): NonZero
    /// computed on the CPU from host values, and an input too large to have them (over the 64-element readback) was
    /// assumed ALL non-zero; the output kept the padded [rank, numel] shape either way.
    /// models/tests/nonzero_gathernd.onnx, x [3,700] = ((i*37) % 100) / 100: nz = NonZero(x > 0.5) = [2, 1029],
    /// g = GatherND(x, Transpose(nz)), Shape(nz) = [2, 1029]; and Shape(NonZero(x > 10)) = [2, 0] (onnxruntime agrees).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task NonZero_DataDependentCount_GatherND() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/nonzero_gathernd.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[2100];
        for (int i = 0; i < x.Length; i++) x[i] = (i * 37 % 100) / 100f;
        var want = new List<int>();
        for (int i = 0; i < x.Length; i++) if (x[i] > 0.5f) want.Add(i);
        int count = want.Count;   // 1029
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 3, 700 }) });

        async Task<float[]> Read(Tensor t, int n)
        {
            using var h = accelerator.Allocate1D<float>(Math.Max(1, n));
            if (n > 0) await h.View.SubView(0, n).CopyFromAsync(t.Data.SubView(0, n));
            await accelerator.SynchronizeAsync();
            return n > 0 ? await h.CopyToHostAsync<float>(0, n) : Array.Empty<float>();
        }

        var nz = outs["nzf"]; var g = outs["g"];
        if (!nz.Shape.SequenceEqual(new[] { 2, count })) throw new Exception($"nz shape [{string.Join(",", nz.Shape)}], expected [2,{count}]");
        if (!g.Shape.SequenceEqual(new[] { count })) throw new Exception($"g shape [{string.Join(",", g.Shape)}], expected [{count}]");
        var snz = await Read(outs["snzf"], 2);
        var s0 = await Read(outs["s0f"], 2);
        if (snz[0] != 2 || snz[1] != count) throw new Exception($"Shape(nz) = [{string.Join(",", snz)}], expected [2,{count}]");
        if (s0[0] != 2 || s0[1] != 0) throw new Exception($"Shape(NonZero(x > 10)) = [{string.Join(",", s0)}], expected [2,0]");
        var nzv = await Read(nz, 2 * count);
        var gv = await Read(g, count);
        for (int j = 0; j < count; j++)
        {
            int i = want[j];
            if (nzv[j] != i / 700 || nzv[count + j] != i % 700)
                throw new Exception($"nz[:, {j}] = ({nzv[j]}, {nzv[count + j]}), expected ({i / 700}, {i % 700})");
            if (gv[j] != x[i]) throw new Exception($"g[{j}] = {gv[j]}, expected {x[i]}");
        }
    });
}
