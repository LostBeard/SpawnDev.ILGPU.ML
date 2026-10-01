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
    /// torch's export of a bias-free Conv (2026-10-01): bias = Expand(0, Expand(Shape(W, start=0, end=1), [1])), i.e. a
    /// Shape window over an INITIALIZER. RaCo-ALIKED has 18 of these; its first forward failed with "Tensor 'val_25' not
    /// found (needed by Expand)". models/tests/conv_nobias_shape_window.onnx: W [4,3,1,1] = (i - 5) / 7, y = Conv(x, W, b)
    /// with b = zeros[4] built that way; onnxruntime agrees with the 1x1-conv sum checked here. Second output
    /// y2 = v + Cast(Shape(W, start=1, end=3)) = v + [3, 1] pins the OPTIMIZER's Shape fold window (it folded the whole
    /// [4,3,1,1]); W's own rank-4 batch window cannot, since [4,3,1,1] and [4] both hold four values.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Shape_StartEnd_OfInitializer_ConvNoBias() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/conv_nobias_shape_window.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[12];
        for (int i = 0; i < 12; i++) x[i] = i;
        using var xB = accelerator.Allocate1D(x);
        using var vB = accelerator.Allocate1D(new float[] { 10, 20 });
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["x"] = new Tensor(xB.View, new[] { 1, 3, 2, 2 }),
            ["v"] = new Tensor(vB.View, new[] { 2 }),
        });
        var y = outs["y"];
        var y2 = outs["y2"];
        if (!y2.Shape.SequenceEqual(new[] { 2 }))
            throw new Exception($"y2 shape [{string.Join(",", y2.Shape)}], expected [2] - Shape(W, start=1, end=3) folded to the whole shape");
        using var y2H = accelerator.Allocate1D<float>(2);
        await y2H.View.CopyFromAsync(y2.Data.SubView(0, 2));
        if (!y.Shape.SequenceEqual(new[] { 1, 4, 2, 2 }))
            throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [1,4,2,2]");
        using var yH = accelerator.Allocate1D<float>(16);
        await yH.View.CopyFromAsync(y.Data.SubView(0, 16));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, 16);
        var y2v = await y2H.CopyToHostAsync<float>(0, 2);
        if (y2v[0] != 13 || y2v[1] != 21) throw new Exception($"y2 = [{y2v[0]}, {y2v[1]}], expected [13, 21]");
        for (int o = 0; o < 4; o++)
            for (int p = 0; p < 4; p++)
            {
                float want = 0;
                for (int c = 0; c < 3; c++) want += (o * 3 + c - 5) / 7f * x[c * 4 + p];
                if (MathF.Abs(yv[o * 4 + p] - want) > 1e-4f)
                    throw new Exception($"y[0,{o},{p / 2},{p % 2}] = {yv[o * 4 + p]}, expected {want}");
            }
    });

    /// <summary>
    /// The optimizer's constant folds on FLOAT constants (2026-10-01). Its fold pass reads ConstantData, an int[]
    /// that also holds a truncated copy of every small float constant, so RaCo-ALIKED's conv bias Reshape(bias[64],
    /// [1,-1,1,1]) folded to integers AND to shape [64]; float Mul/Div evaluated in integer arithmetic.
    /// models/tests/fold_float_constants.onnx: y = x + Reshape(bias=[-0.622,0.99,3.26,-0.6], [1,-1,1,1]) per
    /// channel; y2 = v + (0.5*3 + 7/2 + float(int64(2.7))) = v + 7 (onnxruntime agrees).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Fold_FloatConstants_KeepValuesAndShape() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/fold_float_constants.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[16];
        for (int i = 0; i < 16; i++) x[i] = i;
        using var xB = accelerator.Allocate1D(x);
        using var vB = accelerator.Allocate1D(new float[] { 0, 1, 2 });
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["x"] = new Tensor(xB.View, new[] { 1, 4, 2, 2 }),
            ["v"] = new Tensor(vB.View, new[] { 3 }),
        });
        var y = outs["y"]; var y2 = outs["y2"];
        if (!y.Shape.SequenceEqual(new[] { 1, 4, 2, 2 })) throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [1,4,2,2]");
        using var yH = accelerator.Allocate1D<float>(16);
        using var y2H = accelerator.Allocate1D<float>(3);
        await yH.View.CopyFromAsync(y.Data.SubView(0, 16));
        await y2H.View.CopyFromAsync(y2.Data.SubView(0, 3));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, 16);
        var y2v = await y2H.CopyToHostAsync<float>(0, 3);
        var bias = new float[] { -0.622f, 0.99f, 3.26f, -0.6f };
        for (int i = 0; i < 16; i++)
            if (MathF.Abs(yv[i] - (x[i] + bias[i / 4])) > 1e-5f)
                throw new Exception($"y[{i}] = {yv[i]}, expected {x[i] + bias[i / 4]} (channel {i / 4} bias {bias[i / 4]})");
        for (int i = 0; i < 3; i++)
            if (y2v[i] != i + 7) throw new Exception($"y2 = [{string.Join(",", y2v)}], expected [7,8,9]");
    });

    /// <summary>
    /// Integer semantics through optimizer-folded constants and Range (2026-10-01): Range of folded Cast&lt;int64&gt;
    /// constants is int64, so Div by 3 truncates. The folded constants carried no dtype and Range did not propagate
    /// one, so the Div ran as float division (RaCo-ALIKED's keypoint row = index / W came out 195.27, not 195).
    /// models/tests/int_div_of_range.onnx: y = v + float(Range(0, 10, 1) / 3) (onnxruntime: [0,0,0,1,1,1,2,2,2,3]).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task IntDiv_OfRangeOfFoldedCasts_Truncates() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/int_div_of_range.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        using var vB = accelerator.Allocate1D(new float[10]);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["v"] = new Tensor(vB.View, new[] { 10 }) });
        using var h = accelerator.Allocate1D<float>(10);
        await h.View.CopyFromAsync(outs["y"].Data.SubView(0, 10));
        await accelerator.SynchronizeAsync();
        var y = await h.CopyToHostAsync<float>(0, 10);
        var want = new float[] { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3 };
        if (!y.SequenceEqual(want)) throw new Exception($"y = [{string.Join(",", y)}], expected [{string.Join(",", want)}]");
    });

    /// <summary>
    /// Einsum "bkn,kd->bnd" (2026-10-01): RaCo-ALIKED's sub-pixel keypoint refinement = softmax weights [B,9,N]
    /// times the 3x3 offset grid [9,2]. models/tests/einsum_bkn_kd.onnx, a [2,9,5] = ((i*7 % 13) - 6) / 8, constant
    /// B = the offset grid; checked against a direct contraction (onnxruntime agrees).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Einsum_bkn_kd_bnd() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/einsum_bkn_kd.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var a = new float[90];
        for (int i = 0; i < 90; i++) a[i] = ((i * 7) % 13 - 6) / 8f;
        var B = new float[] { -1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1 };
        using var aB = accelerator.Allocate1D(a);
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["a"] = new Tensor(aB.View, new[] { 2, 9, 5 }) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { 2, 5, 2 })) throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [2,5,2]");
        using var h = accelerator.Allocate1D<float>(20);
        await h.View.CopyFromAsync(y.Data.SubView(0, 20));
        await accelerator.SynchronizeAsync();
        var yv = await h.CopyToHostAsync<float>(0, 20);
        for (int b = 0; b < 2; b++)
            for (int n = 0; n < 5; n++)
                for (int d = 0; d < 2; d++)
                {
                    float want = 0;
                    for (int k = 0; k < 9; k++) want += a[(b * 9 + k) * 5 + n] * B[k * 2 + d];
                    float got = yv[(b * 5 + n) * 2 + d];
                    if (MathF.Abs(got - want) > 1e-5f) throw new Exception($"y[{b},{n},{d}] = {got}, expected {want}; y = [{string.Join(",", yv)}]");
                }
    });

    /// <summary>
    /// ONNX Mod sign rules (2026-10-01): fmod=0 (default, integers) is the FLOORED remainder (sign of the divisor),
    /// fmod=1 the truncated one (sign of the dividend). Every path computed truncated, so RaCo-ALIKED's DKD pad
    /// (-H*W) mod 65536 came out -4096 instead of 61440. models/tests/mod_signs.onnx: y0 = Mod(a, b) on graph inputs
    /// (GPU kernel), y1 = Mod(float a, float b, fmod=1), y2 = Mod of the same values as CONSTANTS (host/interpreter
    /// path) + zero. Expected values are onnxruntime's.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Mod_SignRules_BothFmods() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/mod_signs.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        using var aB = accelerator.Allocate1D(new float[] { -7, 7, -7, 7, -66560, 9 });
        using var bB = accelerator.Allocate1D(new float[] { 3, -3, -3, 3, 65536, 4 });
        using var zB = accelerator.Allocate1D(new float[6]);
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["a"] = new Tensor(aB.View, new[] { 6 }), ["b"] = new Tensor(bB.View, new[] { 6 }), ["zero"] = new Tensor(zB.View, new[] { 6 }),
        });
        var want = new Dictionary<string, float[]>
        {
            ["y0"] = new float[] { 2, -2, -1, 1, 64512, 1 },
            ["y1"] = new float[] { -1, 1, -1, 1, -1024, 1 },
            ["y2"] = new float[] { 2, -2, -1, 1, 64512, 1 },
        };
        var errors = new List<string>();
        foreach (var (name, w) in want)
        {
            using var h = accelerator.Allocate1D<float>(6);
            await h.View.CopyFromAsync(outs[name].Data.SubView(0, 6));
            await accelerator.SynchronizeAsync();
            var got = await h.CopyToHostAsync<float>(0, 6);
            if (!got.SequenceEqual(w))
                errors.Add($"{name} = [{string.Join(",", got)}], expected [{string.Join(",", w)}]");
        }
        if (errors.Count > 0) throw new Exception(string.Join("; ", errors));
    });

    /// <summary>
    /// A rank-0 Gather index in the TOP-LEVEL graph (2026-10-01). ONNX: Gather's output rank is data.rank - 1 +
    /// indices.rank, so a scalar index drops the axis. Rank-0 is recorded in ModelGraph.ScalarTensorNames, which only
    /// the subgraph builder ever filled; at top level every scalar index kept the axis, Unsqueeze made [1,1] of it, and
    /// RaCo-ALIKED's [0, W-1] bound pair came out [2,1] ("Shapes [2,512,2] and [2,1] are not broadcastable").
    /// models/tests/gather_scalar_index.onnx, x [3,5], i1 = scalar 1: y = Reshape(x, Concat(Unsqueeze(Gather(Shape(x),
    /// i1)), [3])) = [5,3]; y2 = Unsqueeze(Gather(x, i1)) = [1,5] = row 1; y3 = x2 [4,2] - [zeros_like(s), s - 1] for
    /// s = Cast(Gather(Shape(x2), scalar -1)) = 2, RaCo's bound pair built the same way (onnxruntime agrees).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Gather_ScalarIndex_TopLevel_DropsAxis() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/gather_scalar_index.onnx");
        // With the CPU shape interpreter eliding the whole scalar chain, the value is materialized as rank-1 [len] and
        // the wrong compile-time rank never shows; RaCo's chain dispatched (CastLike/Expand), so run the modes that do.
        bool foldWas = Graph.GraphCompiler.ShapeSubgraphFoldEnabled, elideWas = Graph.GraphExecutor.ShapeInterpElideDispatch;
        try
        {
        foreach (var (fold, elide) in new[] { (true, true), (true, false), (false, false) })
        {
        Graph.GraphCompiler.ShapeSubgraphFoldEnabled = fold;
        Graph.GraphExecutor.ShapeInterpElideDispatch = elide;
        string where = $"fold={fold} elide={elide}";
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        var x = new float[15];
        for (int i = 0; i < 15; i++) x[i] = i;
        using var xB = accelerator.Allocate1D(x);
        var x2 = new float[8];
        for (int i = 0; i < 8; i++) x2[i] = i;
        using var x2B = accelerator.Allocate1D(x2);
        var outs = await session.RunAsync(new Dictionary<string, Tensor>
        {
            ["x"] = new Tensor(xB.View, new[] { 3, 5 }),
            ["x2"] = new Tensor(x2B.View, new[] { 4, 2 }),
        });
        var y = outs["y"]; var y2 = outs["y2"]; var y3 = outs["y3"];
        if (!y3.Shape.SequenceEqual(new[] { 4, 2 }))
            throw new Exception($"{where}: y3 shape [{string.Join(",", y3.Shape)}], expected [4,2]");
        using var y3H = accelerator.Allocate1D<float>(8);
        await y3H.View.CopyFromAsync(y3.Data.SubView(0, 8));
        if (!y.Shape.SequenceEqual(new[] { 5, 3 }))
            throw new Exception($"{where}: y shape [{string.Join(",", y.Shape)}], expected [5,3]");
        if (!y2.Shape.SequenceEqual(new[] { 1, 5 }))
            throw new Exception($"{where}: y2 shape [{string.Join(",", y2.Shape)}], expected [1,5] - the scalar index kept its axis");
        using var y2H = accelerator.Allocate1D<float>(5);
        await y2H.View.CopyFromAsync(y2.Data.SubView(0, 5));
        await accelerator.SynchronizeAsync();
        var v = await y2H.CopyToHostAsync<float>(0, 5);
        var v3 = await y3H.CopyToHostAsync<float>(0, 8);
        for (int i = 0; i < 8; i++)
            if (v3[i] != i - (i % 2 == 1 ? 1 : 0)) throw new Exception($"{where}: y3[{i / 2},{i % 2}] = {v3[i]}, expected {i - (i % 2 == 1 ? 1 : 0)}");
        for (int i = 0; i < 5; i++)
            if (v[i] != 5 + i) throw new Exception($"{where}: y2[0,{i}] = {v[i]}, expected {5 + i}");
        }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = foldWas;
            Graph.GraphExecutor.ShapeInterpElideDispatch = elideWas;
        }
    });

    /// <summary>
    /// A CPU-read scalar the optimizer folded (2026-10-01): RaCo-ALIKED builds aranges as Range(Cast(0), Cast(1792),
    /// Cast(1)); the optimizer folds each Cast of a Constant into a new constant, and Range reads its scalars from the
    /// executor's CPU constants, which never received optimizer-folded values ("Range: scalar inputs not available as
    /// runtime constants"). models/tests/range_of_cast_constants.onnx: y = x + Cast(Range(Cast(0), Cast(4), Cast(1)))
    /// = x + [0, 1, 2, 3] (onnxruntime agrees).
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Range_OfOptimizerFoldedScalars() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/range_of_cast_constants.onnx");
        using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
        using var xB = accelerator.Allocate1D(new float[] { 10, 20, 30, 40 });
        var outs = await session.RunAsync(new Dictionary<string, Tensor> { ["x"] = new Tensor(xB.View, new[] { 4 }) });
        var y = outs["y"];
        if (!y.Shape.SequenceEqual(new[] { 4 }))
            throw new Exception($"y shape [{string.Join(",", y.Shape)}], expected [4]");
        using var yH = accelerator.Allocate1D<float>(4);
        await yH.View.CopyFromAsync(y.Data.SubView(0, 4));
        await accelerator.SynchronizeAsync();
        var yv = await yH.CopyToHostAsync<float>(0, 4);
        for (int i = 0; i < 4; i++)
            if (yv[i] != 10 * (i + 1) + i) throw new Exception($"y[{i}] = {yv[i]}, expected {10 * (i + 1) + i}");
    });

    /// <summary>
    /// The Shape window on a DYNAMIC input, where nothing folds at compile time (2026-10-01): RunAsyncCore's runtime
    /// output shape and the CacheShapeReadbacks hit path both used the WHOLE input shape (the sync RunCore had been
    /// fixed alone). models/tests/shape_window_dynamic.onnx, x [N,3,5]: y = Reshape(x, Concat([-1], Shape(x, start=2)))
    /// = [3N, 5], z = Cast(Shape(x, start=1, end=2)) = [3], w = v + Cast(Shape(x, start=2)) = v + 5, r = Reshape(q [15],
    /// Shape(x, start=1)) = [3, 5] (onnxruntime agrees).
    /// By default the CPU shape interpreter resolves every Shape and ELIDES its dispatch (a GPU consumer gets the value
    /// materialized), so neither path runs: they are reached with the elide off, or with the interpreter off entirely.
    /// A cache HIT additionally needs input-independent folding off, which otherwise binds Shape's result on the second
    /// same-shape run and skips the node. This runs each of those modes. The cache-hit guard fails on the browser lanes:
    /// on the desktop ones a real readback of the (correct) Shape buffer overwrites the published value before use. w gives Shape(x, start=2) a GPU consumer while it still drives y's Reshape, so a
    /// cache hit that publishes the whole shape breaks y. Runs N=2 twice (the second is a cache hit) and then N=4 (a
    /// new shape), with the readback cache on and off.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Shape_StartEnd_DynamicInput_RuntimeAndCachedPaths() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available for this backend");
        var bytes = await http.GetByteArrayAsync("models/tests/shape_window_dynamic.onnx");
        bool foldWas = Graph.GraphCompiler.ShapeSubgraphFoldEnabled, elideWas = Graph.GraphExecutor.ShapeInterpElideDispatch;
        bool indepWas = Graph.GraphExecutor.FoldInputIndependentNodes;
        try
        {
        foreach (var (fold, elide, indep) in new[] { (true, true, true), (true, false, true), (false, false, true), (true, false, false), (false, false, false) })
        foreach (var cache in new[] { true, false })
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = fold;
            Graph.GraphExecutor.ShapeInterpElideDispatch = elide;
            Graph.GraphExecutor.FoldInputIndependentNodes = indep;
            using var session = InferenceSession.CreateFromOnnx(accelerator, bytes);
            session.CacheShapeReadbacks = cache;
            foreach (var n in new[] { 2, 2, 4 })
            {
                var x = new float[n * 15];
                for (int i = 0; i < x.Length; i++) x[i] = i;
                using var xB = accelerator.Allocate1D(x);
                using var vB = accelerator.Allocate1D(new float[] { 100 });
                var q = new float[15];
                for (int i = 0; i < 15; i++) q[i] = i;
                using var qB = accelerator.Allocate1D(q);
                var outs = await session.RunAsync(new Dictionary<string, Tensor>
                {
                    ["x"] = new Tensor(xB.View, new[] { n, 3, 5 }),
                    ["v"] = new Tensor(vB.View, new[] { 1 }),
                    ["q"] = new Tensor(qB.View, new[] { 15 }),
                });
                var y = outs["y"]; var z = outs["z"]; var w = outs["w"]; var r = outs["r"];
                if (!r.Shape.SequenceEqual(new[] { 3, 5 }))
                    throw new Exception($"fold={fold} elide={elide} indep={indep} cache={cache} N={n}: r shape [{string.Join(",", r.Shape)}], expected [3,5]");
                string where = $"fold={fold} elide={elide} indep={indep} cache={cache} N={n}";
                if (!y.Shape.SequenceEqual(new[] { 3 * n, 5 }))
                    throw new Exception($"{where}: y shape [{string.Join(",", y.Shape)}], expected [{3 * n},5]");
                if (z.ElementCount != 1)
                    throw new Exception($"{where}: z shape [{string.Join(",", z.Shape)}], expected [1]");
                if (w.ElementCount != 1)
                    throw new Exception($"{where}: w shape [{string.Join(",", w.Shape)}], expected [1]");
                using var yH = accelerator.Allocate1D<float>(x.Length);
                using var zH = accelerator.Allocate1D<float>(1);
                using var wH = accelerator.Allocate1D<float>(1);
                await wH.View.CopyFromAsync(w.Data.SubView(0, 1));
                await yH.View.CopyFromAsync(y.Data.SubView(0, x.Length));
                await zH.View.CopyFromAsync(z.Data.SubView(0, 1));
                await accelerator.SynchronizeAsync();
                var yv = await yH.CopyToHostAsync<float>(0, x.Length);
                var zv = await zH.CopyToHostAsync<float>(0, 1);
                var wv = await wH.CopyToHostAsync<float>(0, 1);
                if (zv[0] != 3) throw new Exception($"{where}: z = {zv[0]}, expected 3");
                if (wv[0] != 105) throw new Exception($"{where}: w = {wv[0]}, expected 105");
                for (int i = 0; i < yv.Length; i++)
                    if (yv[i] != i) throw new Exception($"{where}: y[{i}] = {yv[i]}, expected {i}");
            }
        }
        }
        finally
        {
            Graph.GraphCompiler.ShapeSubgraphFoldEnabled = foldWas;
            Graph.GraphExecutor.ShapeInterpElideDispatch = elideWas;
            Graph.GraphExecutor.FoldInputIndependentNodes = indepWas;
        }
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
