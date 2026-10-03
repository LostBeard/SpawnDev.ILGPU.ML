using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Real GPU kernel tests for core element-wise operators.
/// Each test allocates GPU buffers, dispatches the kernel, and compares against CPU reference.
/// Operator-level tests via the Execute() path are in MLTestBase.OperatorCoverageTests.cs.
/// Fake Resolve()-only tests have been removed — every test here runs real GPU code.
/// </summary>
public abstract partial class MLTestBase
{
    // Helper for operator-level tests
    private static OnnxOpContext MakeOpCtx(Accelerator acc, Tensor[] inputs, Tensor[] outputs,
        Dictionary<string, object>? attrs = null, string[]? inputNames = null,
        Dictionary<string, float[]>? constants = null,
        HashSet<string>? integerTensorNames = null,
        string[]? outputNames = null,
        OperatorRegistry? registry = null) => new OnnxOpContext
    {
        Inputs = inputs, Outputs = outputs,
        Attributes = attrs ?? new Dictionary<string, object>(),
        Pool = new BufferPool(acc),
        InputNames = inputNames ?? inputs.Select((_, i) => $"input_{i}").ToArray(),
        OutputNames = outputNames ?? outputs.Select((_, i) => $"output_{i}").ToArray(),
        ConstantValues = constants,
        IntegerTensorNames = integerTensorNames,
        HostValues = new Dictionary<string, OnnxValue>(StringComparer.Ordinal),
        Registry = registry,
    };

    [TestMethod] public async Task AllOps_Abs() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-3,-1,0,1,3}); using var o = a.Allocate1D<float>(5); e.Abs(i.View,o.View,5); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{3,1,0,1,3},0f,"Abs:"); });
    [TestMethod] public async Task AllOps_Acos() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,0.5f,1}); using var o = a.Allocate1D<float>(3); e.Acos(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Acos(0),MathF.Acos(0.5f),MathF.Acos(1)},1e-4f,"Acos:"); });
    [TestMethod] public async Task AllOps_Acosh() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{1,2,5}); using var o = a.Allocate1D<float>(3); e.Acosh(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Acosh(1),MathF.Acosh(2),MathF.Acosh(5)},1e-4f,"Acosh:"); });
    [TestMethod] public async Task AllOps_Add() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,2,3}); using var y = a.Allocate1D(new float[]{4,5,6}); using var o = a.Allocate1D<float>(3); e.Add(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{5,7,9},0f,"Add:"); });
    [TestMethod] public async Task AllOps_ArgMax() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{1,5,3,2,4,6}); using var o = a.Allocate1D<float>(2); e.ArgMax(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,2},0f,"ArgMax:"); });
    [TestMethod] public async Task AllOps_Asin() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,0.5f,-0.5f}); using var o = a.Allocate1D<float>(3); e.Asin(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Asin(0),MathF.Asin(0.5f),MathF.Asin(-0.5f)},1e-4f,"Asin:"); });
    [TestMethod] public async Task AllOps_Asinh() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Asinh(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Asinh(-1),MathF.Asinh(0),MathF.Asinh(1)},1e-4f,"Asinh:"); });
    [TestMethod] public async Task AllOps_Atan() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Atan(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Atan(-1),MathF.Atan(0),MathF.Atan(1)},1e-4f,"Atan:"); });
    [TestMethod] public async Task AllOps_Atanh() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-0.5f,0,0.5f}); using var o = a.Allocate1D<float>(3); e.Atanh(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Atanh(-0.5f),MathF.Atanh(0),MathF.Atanh(0.5f)},1e-4f,"Atanh:"); });
    [TestMethod] public async Task AllOps_Ceil() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0.1f,0.9f,-0.1f}); using var o = a.Allocate1D<float>(3); e.Ceil(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,1,0},0f,"Ceil:"); });
    [TestMethod] public async Task AllOps_Celu() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Celu(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Max(0,-1)+MathF.Min(0,MathF.Exp(-1)-1),0f,1f},1e-4f,"Celu:"); });
    [TestMethod] public async Task AllOps_Clip() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-5,-1,0,1,5}); using var o = a.Allocate1D<float>(5); e.Clip(i.View,o.View,5,-2f,2f); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{-2,-1,0,1,2},0f,"Clip:"); });
    [TestMethod] public async Task AllOps_Cos() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,MathF.PI/2,MathF.PI}); using var o = a.Allocate1D<float>(3); e.Cos(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{1f,0f,-1f},1e-4f,"Cos:"); });
    [TestMethod] public async Task AllOps_Cosh() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,1,-1}); using var o = a.Allocate1D<float>(3); e.Cosh(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Cosh(0),MathF.Cosh(1),MathF.Cosh(-1)},1e-4f,"Cosh:"); });
    [TestMethod] public async Task AllOps_Div() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{6,8,10}); using var y = a.Allocate1D(new float[]{2,4,5}); using var o = a.Allocate1D<float>(3); e.Div(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{3,2,2},1e-4f,"Div:"); });
    [TestMethod] public async Task AllOps_Elu() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Elu(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Exp(-1)-1,0f,1f},1e-4f,"Elu:"); });
    [TestMethod] public async Task AllOps_Equal() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,2,3}); using var y = a.Allocate1D(new float[]{1,0,3}); using var o = a.Allocate1D<float>(3); e.Equal(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,0,1},0f,"Equal:"); });
    [TestMethod] public async Task AllOps_Erf() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0}); using var o = a.Allocate1D<float>(1); e.Erf(i.View,o.View,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0},1e-4f,"Erf:"); });
    [TestMethod] public async Task AllOps_Exp() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,1}); using var o = a.Allocate1D<float>(2); e.Exp(i.View,o.View,2); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{1f,MathF.E},1e-4f,"Exp:"); });
    [TestMethod] public async Task AllOps_Floor() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0.9f,1.1f,-0.1f,-0.9f}); using var o = a.Allocate1D<float>(4); e.Floor(i.View,o.View,4); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,1,-1,-1},0f,"Floor:"); });
    [TestMethod] public async Task AllOps_GlobalMaxPool() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{1,5,3,2,4,6}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceMax(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{5,6},0f,"GlobalMaxPool:"); });
    [TestMethod] public async Task AllOps_Greater() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,5,3}); using var y = a.Allocate1D(new float[]{2,2,3}); using var o = a.Allocate1D<float>(3); e.Greater(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,1,0},0f,"Greater:"); });
    [TestMethod] public async Task AllOps_IsInf() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,float.PositiveInfinity,float.NegativeInfinity}); using var o = a.Allocate1D<float>(3); e.IsInf(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,1,1},0f,"IsInf:"); });
    [TestMethod] public async Task AllOps_Less() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,5,3}); using var y = a.Allocate1D(new float[]{2,2,3}); using var o = a.Allocate1D<float>(3); e.Less(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,0,0},0f,"Less:"); });
    [TestMethod] public async Task AllOps_Log() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{1,MathF.E}); using var o = a.Allocate1D<float>(2); e.Log(i.View,o.View,2); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,1},1e-4f,"Log:"); });
    [TestMethod] public async Task AllOps_MatMul() => await RunTest(async a => { var r = new OperatorRegistry(a); using var x = a.Allocate1D(new float[]{1,2,3,4}); using var y = a.Allocate1D(new float[]{5,6,7,8}); using var o = a.Allocate1D<float>(4); r.MatMul.MatMul(x.View,y.View,o.View,2,2,2); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{19,22,43,50},1e-3f,"MatMul:"); });
    [TestMethod]
    public async Task AllOps_MatMul_BatchedActivationsSharedWeight() => await RunTest(async a =>
    {
        // REGRESSION (DAv3 multi-view): a batched activation [N,S,K] @ a SHARED 2-D weight [K,Nn] (a Linear)
        // must flatten ALL N*S rows into M, NOT route to BatchedMatMul (which strides the 2-D weight by batch
        // and reads off its end → the multi-view qkv Linear blew to ~1e19). Each row must match the CPU ref.
        int N = 2, S = 3, K = 4, Nn = 5;
        var aData = RandomFloats(N * S * K, seed: 500, scale: 0.5f);
        var wData = RandomFloats(K * Nn, seed: 501, scale: 0.5f);
        var expected = new float[N * S * Nn];
        for (int row = 0; row < N * S; row++)
            for (int n = 0; n < Nn; n++)
            {
                double s = 0; for (int k = 0; k < K; k++) s += (double)aData[row * K + k] * (double)wData[k * Nn + n];
                expected[row * Nn + n] = (float)s;
            }
        using var aBuf = a.Allocate1D(aData);
        using var wBuf = a.Allocate1D(wData);
        using var oBuf = a.Allocate1D<float>(N * S * Nn);
        var reg = new OperatorRegistry(a);
        var xT = new Tensor(aBuf.View, new[] { N, S, K });
        var wT = new Tensor(wBuf.View, new[] { K, Nn });
        var oT = new Tensor(oBuf.View, new[] { N, S, Nn });
        var ctx = MakeOpCtx(a, new[] { xT, wT }, new[] { oT });
        new SpawnDev.ILGPU.ML.Operators.MatMulOperator(reg).Execute(ctx);
        await a.SynchronizeAsync();
        await AssertCloseGpu(a, oBuf.View, expected, K * 2e-5f, "MatMul batched-A@2D-weight: ");
    });
    [TestMethod]
    public async Task AllOps_MatMul_BroadcastBatch() => await RunTest(async a =>
    {
        // REGRESSION (Video Depth Anything): numpy batch broadcasting. DINOv2's position-embedding resize is
        // [7,37] @ [1,384,37,37] -> [1,384,7,37]; the operator took the batch count from A alone (1) and wrote
        // ONE of the 384 output slices. Shape inference also aligned batch dims from the LEFT. Every case here
        // is checked against a CPU broadcast reference, including the inferred output shape.
        var cases = new (int[] A, int[] B)[]
        {
            (new[] { 3, 4 },          new[] { 1, 5, 4, 2 }),   // A shared across B's batch (the VDA case)
            (new[] { 2, 3, 4 },       new[] { 3, 2, 4, 5 }),   // right-aligned: A's batch pairs with B's LAST batch dim
            (new[] { 2, 1, 3, 4 },    new[] { 1, 3, 4, 2 }),   // both broadcast, on different dims
            (new[] { 2, 3, 3, 4 },    new[] { 1, 1, 4, 5 }),   // B shared through size-1 batch dims
            (new[] { 1, 3, 4 },       new[] { 2, 3, 4, 2 }),   // a leading 1 on A
            (new[] { 2, 3, 2, 1, 3, 4 }, new[] { 1, 3, 1, 4, 4, 2 }),   // 4 independent batch dims: one outer dim walked
        };
        int seed = 600;
        foreach (var (aS, bS) in cases)
        {
            int M = aS[^2], K = aS[^1], N = bS[^1];
            var outS = SpawnDev.ILGPU.ML.Operators.MatMulOperator.BroadcastBatch(aS, bS).Concat(new[] { M, N }).ToArray();
            var reg = new OperatorRegistry(a);
            var inferred = new SpawnDev.ILGPU.ML.Operators.MatMulOperator(reg).InferOutputShapes(new[] { aS, bS }, new())[0];
            var label = $"MatMul [{string.Join(",", aS)}]@[{string.Join(",", bS)}]: ";
            if (!inferred.SequenceEqual(outS))
                throw new Exception($"{label}inferred [{string.Join(",", inferred)}], expected [{string.Join(",", outS)}]");

            var aData = RandomFloats(aS.Aggregate(1, (x, y) => x * y), seed: seed++, scale: 0.5f);
            var bData = RandomFloats(bS.Aggregate(1, (x, y) => x * y), seed: seed++, scale: 0.5f);
            int outCount = outS.Aggregate(1, (x, y) => x * y);
            var expected = new float[outCount];
            int rb = outS.Length - 2;
            var idx = new int[rb];
            for (int ob = 0; ob < outCount / (M * N); ob++)
            {
                // Batch coordinates of output batch ob, then each operand's own (broadcast) batch offset.
                for (int t = ob, d = rb - 1; d >= 0; d--) { idx[d] = t % outS[d]; t /= outS[d]; }
                int Off(int[] s, int mat)
                {
                    int r = s.Length - 2, off = 0, stride = mat;
                    for (int d = r - 1; d >= 0; d--)
                    {
                        int od = d + (rb - r);
                        if (s[d] != 1) off += idx[od] * stride;
                        stride *= s[d];
                    }
                    return off;
                }
                int ao = Off(aS, M * K), bo = Off(bS, K * N);
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                    {
                        double s = 0;
                        for (int k = 0; k < K; k++) s += (double)aData[ao + m * K + k] * bData[bo + k * N + n];
                        expected[ob * M * N + m * N + n] = (float)s;
                    }
            }
            using var aBuf = a.Allocate1D(aData);
            using var bBuf = a.Allocate1D(bData);
            using var oBuf = a.Allocate1D<float>(outCount);
            var ctx = MakeOpCtx(a, new[] { new Tensor(aBuf.View, aS), new Tensor(bBuf.View, bS) },
                new[] { new Tensor(oBuf.View, outS) });
            new SpawnDev.ILGPU.ML.Operators.MatMulOperator(reg).Execute(ctx);
            await a.SynchronizeAsync();
            await AssertCloseGpu(a, oBuf.View, expected, K * 2e-5f, label);
        }
    });
    [TestMethod] public async Task AllOps_Max() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,5,3}); using var y = a.Allocate1D(new float[]{4,2,6}); using var o = a.Allocate1D<float>(3); e.Max(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{4,5,6},0f,"Max:"); });
    [TestMethod] public async Task AllOps_Min() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{1,5,3}); using var y = a.Allocate1D(new float[]{4,2,6}); using var o = a.Allocate1D<float>(3); e.Min(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,2,3},0f,"Min:"); });
    [TestMethod] public async Task AllOps_Mish() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,1,-1}); using var o = a.Allocate1D<float>(3); e.Mish(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{0f,MathF.Tanh(MathF.Log(1+MathF.Exp(1))),-MathF.Tanh(MathF.Log(1+MathF.Exp(-1)))},1e-4f,"Mish:"); });
    [TestMethod] public async Task AllOps_Mul() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{2,3,4}); using var y = a.Allocate1D(new float[]{5,6,7}); using var o = a.Allocate1D<float>(3); e.Mul(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{10,18,28},0f,"Mul:"); });
    [TestMethod] public async Task AllOps_Neg() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-3,0,3}); using var o = a.Allocate1D<float>(3); e.Scale(i.View,o.View,3,-1f); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{3,0,-3},0f,"Neg:"); });
    [TestMethod] public async Task AllOps_Pow() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{2,3,4}); using var y = a.Allocate1D(new float[]{3,2,0.5f}); using var o = a.Allocate1D<float>(3); e.Pow(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{8,9,2},1e-3f,"Pow:"); });
    [TestMethod] public async Task AllOps_Reciprocal() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{2,4,5}); using var o = a.Allocate1D<float>(3); e.Reciprocal(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0.5f,0.25f,0.2f},1e-4f,"Reciprocal:"); });
    [TestMethod] public async Task AllOps_ReduceMax() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{1,5,3,2,4,6}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceMax(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{5,6},0f,"ReduceMax:"); });
    [TestMethod] public async Task AllOps_ReduceMean() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{1,2,3,4,5,6}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceMean(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{2,5},1e-4f,"ReduceMean:"); });
    [TestMethod] public async Task AllOps_ReduceMin() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{5,1,3,6,2,4}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceMin(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{1,2},0f,"ReduceMin:"); });
    [TestMethod] public async Task AllOps_ReduceProd() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{1,2,3,4,5,6}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceProd(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{6,120},1e-2f,"ReduceProd:"); });
    [TestMethod] public async Task AllOps_ReduceSum() => await RunTest(async a => { var r = new OperatorRegistry(a); using var i = a.Allocate1D(new float[]{1,2,3,4,5,6}); using var o = a.Allocate1D<float>(2); r.Reductions.ReduceSum(i.View,o.View,2,3,1); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{6,15},1e-4f,"ReduceSum:"); });
    [TestMethod] public async Task AllOps_Relu() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-2,-1,0,1,2}); using var o = a.Allocate1D<float>(5); e.ReLU(i.View,o.View,5); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,0,0,1,2},0f,"Relu:"); });
    [TestMethod] public async Task AllOps_Selu() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Selu(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{1.0507f*1.67326f*(MathF.Exp(-1)-1),0f,1.0507f},1e-3f,"Selu:"); });
    [TestMethod] public async Task AllOps_Sin() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,MathF.PI/2,MathF.PI}); using var o = a.Allocate1D<float>(3); e.Sin(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{0f,1f,0f},1e-4f,"Sin:"); });
    [TestMethod] public async Task AllOps_Sinh() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,1}); using var o = a.Allocate1D<float>(2); e.Sinh(i.View,o.View,2); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{0f,MathF.Sinh(1)},1e-4f,"Sinh:"); });
    [TestMethod] public async Task AllOps_Softplus() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-1,0,1}); using var o = a.Allocate1D<float>(3); e.Softplus(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{MathF.Log(1+MathF.Exp(-1)),MathF.Log(2),MathF.Log(1+MathF.Exp(1))},1e-4f,"Softplus:"); });
    [TestMethod] public async Task AllOps_Softsign() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{-2,0,2}); using var o = a.Allocate1D<float>(3); e.Softsign(i.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{-2f/3,0f,2f/3},1e-4f,"Softsign:"); });
    [TestMethod] public async Task AllOps_Sqrt() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,1,4,9}); using var o = a.Allocate1D<float>(4); e.Sqrt(i.View,o.View,4); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{0,1,2,3},1e-4f,"Sqrt:"); });
    [TestMethod] public async Task AllOps_Sub() => await RunTest(async a => { var e = GetOrCreateEW(a); using var x = a.Allocate1D(new float[]{5,7,9}); using var y = a.Allocate1D(new float[]{1,2,3}); using var o = a.Allocate1D<float>(3); e.Sub(x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{4,5,6},0f,"Sub:"); });
    [TestMethod] public async Task AllOps_Tan() => await RunTest(async a => { var e = GetOrCreateEW(a); using var i = a.Allocate1D(new float[]{0,MathF.PI/4}); using var o = a.Allocate1D<float>(2); e.Tan(i.View,o.View,2); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new[]{0f,1f},1e-4f,"Tan:"); });
    [TestMethod] public async Task AllOps_Where() => await RunTest(async a => { var e = GetOrCreateEW(a); using var c = a.Allocate1D(new float[]{1,0,1}); using var x = a.Allocate1D(new float[]{10,20,30}); using var y = a.Allocate1D(new float[]{100,200,300}); using var o = a.Allocate1D<float>(3); e.Where(c.View,x.View,y.View,o.View,3); await a.SynchronizeAsync(); await AssertCloseGpu(a,o.View,new float[]{10,200,30},0f,"Where:"); });
}
