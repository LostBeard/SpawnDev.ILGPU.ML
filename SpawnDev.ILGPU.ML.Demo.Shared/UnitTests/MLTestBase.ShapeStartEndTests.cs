using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// A Constant in the <c>value_ints</c> form (2026-09-30). The loader registered only the <c>value</c>-tensor form, so
    /// such a Constant produced nothing: RaCo-ALIKED + LightGlue+ (SpawnScene's feature front end) builds its per-keypoint
    /// patch Reshape target as Concat([n], value_ints [3], [42], [42]) and failed to load ("Slice crashed",
    /// [532480,12,-3,-3]). models/tests/constant_value_ints.onnx: y = Reshape(x [2,6], Constant(value_ints=[3,4])) = [3,4].
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Constant_ValueIntsForm_IsLoaded() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/constant_value_ints.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[12];
        for (int i = 0; i < 12; i++) x[i] = i;
        using var xB = accelerator.Allocate1D(x);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 2, 6 }) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { 3, 4 }))
            throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [3,4] - the value_ints Constant was not loaded");
        using var yH = accelerator.Allocate1D<float>(12);
        await yH.View.CopyFromAsync(y.Data.SubView(0, 12));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, 12);
        for (int i = 0; i < 12; i++)
            if (yv[i] != i) throw new Exception($"y[{i}] = {yv[i]}, expected {i}");
    });

    /// <summary>
    /// ONNX Shape's opset-15 <c>start</c> / <c>end</c> window (2026-09-30): ShapeOperator, the compiler's fold and the
    /// executor's runtime shape all returned the WHOLE shape. They now share ShapeOperator.Window; this pins that rule
    /// (negative = from the back, clamped, end before start = empty) and the operator's inferred output shape. Model-level
    /// tests could not isolate it: a static-shape model is resolved by the shape interpreter (which was already right),
    /// and a data-dependent one (Shape(NonZero(x))) is masked by NonZero's padded upper-bound shape.
    /// </summary>
    [TestMethod]
    public async Task Shape_StartEnd_Window() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask;
        (int, int) W(int rank, long? start, long? end)
        {
            var a = new Dictionary<string, object>();
            if (start != null) a["start"] = start.Value;
            if (end != null) a["end"] = end.Value;
            return ShapeOperator.Window(rank, a);
        }
        void Expect((int, int) got, (int, int) want, string what)
        {
            if (got != want) throw new Exception($"{what}: window {got}, expected {want}");
        }
        Expect(W(4, null, null), (0, 4), "no attributes = whole shape");
        Expect(W(4, 0, 1), (0, 1), "end=1 = batch dim alone (RaCo)");
        Expect(W(4, -2, null), (2, 4), "start=-2 = last two dims");
        Expect(W(4, 1, -1), (1, 3), "negative end");
        Expect(W(4, -9, 99), (0, 4), "clamped");
        Expect(W(4, 3, 1), (3, 3), "end before start = empty");
        var op = new ShapeOperator(new OperatorRegistry(accelerator));
        var inferred = op.InferOutputShapes(new[] { new[] { 2, 3, 4, 5 } }, new Dictionary<string, object> { ["start"] = 1L, ["end"] = 3L });
        if (inferred.Length != 1 || !inferred[0].SequenceEqual(new[] { 2 }))
            throw new Exception($"InferOutputShapes [{string.Join(",", inferred[0])}], expected [2] for start=1, end=3 of a rank-4 input");
    });
}
