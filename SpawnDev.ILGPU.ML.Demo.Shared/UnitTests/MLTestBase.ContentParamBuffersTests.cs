using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Params buffers are content-addressed and write-once (<c>ContentParamBuffers</c>): repeated calls with the same
/// shapes must reuse ONE device buffer, and back-to-back calls with DIFFERENT params, issued without a sync, must each
/// still read their own params. The previous pattern allocated a fresh buffer per call and kept every one until
/// Dispose(): SpawnScene's 83-pass DAv3 cascade accumulated ~19,000 of them and the browser GPU process died out of
/// memory (2026-09-27). Each test checks the numbers AND the buffer count, so neither a race nor the leak can pass.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task ContentParams_Slice_RepeatedCallsReuseOneBufferPerShape() => await RunTest(async accelerator =>
    {
        int[] inShape = { 2, 3, 8 };
        var input = RandomFloats(2 * 3 * 8, seed: 11);
        int[] ends = { 2, 3, 8 }, steps = { 1, 1, 1 };
        int[] startsA = { 0, 0, 0 }, endsA = { 2, 3, 4 };   // first half of the last axis
        int[] startsB = { 0, 1, 4 };                          // rows 1.., second half
        var expA = CpuSlice(input, inShape, startsA, endsA, steps);
        var expB = CpuSlice(input, inShape, startsB, ends, steps);
        var (outShapeA, inStrides, totalA) = SliceGeom(inShape, startsA, endsA, steps);
        var (outShapeB, _, totalB) = SliceGeom(inShape, startsB, ends, steps);

        using var inBuf = accelerator.Allocate1D(input);
        using var outA = accelerator.Allocate1D<float>(totalA);
        using var outB = accelerator.Allocate1D<float>(totalB);
        using var slice = new SliceKernel(accelerator);
        // Interleaved, no sync between calls: every A dispatch must read A's params, every B dispatch B's.
        for (int i = 0; i < 20; i++)
        {
            slice.Slice(inBuf.View, outA.View, startsA, steps, outShapeA, inStrides, 3, totalA);
            slice.Slice(inBuf.View, outB.View, startsB, steps, outShapeB, inStrides, 3, totalB);
        }
        await accelerator.SynchronizeAsync();
        AssertClose(expA, await outA.CopyToHostAsync<float>(0, totalA), 0f, "Slice A: ");
        AssertClose(expB, await outB.CopyToHostAsync<float>(0, totalB), 0f, "Slice B: ");
        if (slice.DistinctParamsBuffers != 2)
            throw new Exception($"40 Slice calls with 2 distinct params hold {slice.DistinctParamsBuffers} params buffers, expected 2");
    });

    [TestMethod]
    public async Task ContentParams_Broadcast_RepeatedCallsReuseOneBufferPerShape() => await RunTest(async accelerator =>
    {
        // a[4,1] + b[1,5] -> [4,5]   and   a[4,5] * b[5] -> [4,5]
        var a1 = RandomFloats(4, seed: 1); var b1 = RandomFloats(5, seed: 2);
        var a2 = RandomFloats(20, seed: 3); var b2 = RandomFloats(5, seed: 4);
        var exp1 = new float[20]; var exp2 = new float[20];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 5; c++) { exp1[r * 5 + c] = a1[r] + b1[c]; exp2[r * 5 + c] = a2[r * 5 + c] * b2[c]; }

        using var ba1 = accelerator.Allocate1D(a1); using var bb1 = accelerator.Allocate1D(b1);
        using var ba2 = accelerator.Allocate1D(a2); using var bb2 = accelerator.Allocate1D(b2);
        using var o1 = accelerator.Allocate1D<float>(20); using var o2 = accelerator.Allocate1D<float>(20);
        var ew = new ElementWiseKernels(accelerator);
        try
        {
            for (int i = 0; i < 15; i++)
            {
                ew.BroadcastBinaryOpND(ba1.View, bb1.View, o1.View, new[] { 4, 1 }, new[] { 1, 5 }, new[] { 4, 5 }, BroadcastOp.Add);
                ew.BroadcastBinaryOpND(ba2.View, bb2.View, o2.View, new[] { 4, 5 }, new[] { 5 }, new[] { 4, 5 }, BroadcastOp.Mul);
            }
            await accelerator.SynchronizeAsync();
            AssertClose(exp1, await o1.CopyToHostAsync<float>(0, 20), 1e-6f, "Broadcast Add: ");
            AssertClose(exp2, await o2.CopyToHostAsync<float>(0, 20), 1e-6f, "Broadcast Mul: ");
            if (ew.DistinctStridesBuffers != 2)
                throw new Exception($"30 broadcast calls with 2 distinct stride sets hold {ew.DistinctStridesBuffers} strides buffers, expected 2");
        }
        finally { ew.Dispose(); }
    });

    [TestMethod]
    public async Task ContentParams_Gather_RepeatedCallsReuseOneBuffer() => await RunTest(async accelerator =>
    {
        // data [3,4,5], gather axis 1 with indices {2,0} -> [3,2,5]
        var data = RandomFloats(60, seed: 9);
        var idx = new float[] { 2, 0 };
        var exp = new float[30];
        for (int o = 0; o < 3; o++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 5; k++) exp[(o * 2 + j) * 5 + k] = data[(o * 4 + (int)idx[j]) * 5 + k];

        using var bd = accelerator.Allocate1D(data); using var bi = accelerator.Allocate1D(idx);
        using var outBuf = accelerator.Allocate1D<float>(30);
        using var gather = new GatherKernel(accelerator);
        for (int i = 0; i < 25; i++)
            gather.GatherGenericFloat(bd.View, bi.View, outBuf.View, numIdx: 2, innerSize: 5, outerSize: 3, axisSize: 4);
        await accelerator.SynchronizeAsync();
        AssertClose(exp, await outBuf.CopyToHostAsync<float>(0, 30), 0f, "Gather: ");
        if (gather.DistinctParamsBuffers != 1)
            throw new Exception($"25 identical Gather calls hold {gather.DistinctParamsBuffers} params buffers, expected 1");
    });
}
