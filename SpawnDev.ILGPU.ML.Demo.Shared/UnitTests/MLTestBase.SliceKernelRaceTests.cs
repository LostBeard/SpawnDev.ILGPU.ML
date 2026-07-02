using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// SliceKernel param-upload race probe (the DAv3 WebGPU range-deviation hunt).
/// The operator registry holds ONE SliceKernel instance = one shared _paramsBuf. Slice()
/// uploads params via CopyFromCPU - on WebGPU that is an IMMEDIATE queue.writeBuffer, while the
/// kernel dispatch is batched into the command encoder and executes at a LATER submit. Two
/// back-to-back Slices with different params and no intervening sync (exactly DAv3's q-rope +
/// k-rope rotate-half pair) can therefore both execute with the SECOND call's params: the first
/// writeBuffer lands, the encoder still holds dispatch A, the second writeBuffer OVERWRITES the
/// params before submit, then A and B both run with params B.
/// Test 1 = kernel math in isolation (one Slice vs CPU reference).
/// Test 2 = the race (two Slices back-to-back, both verified; also runs the DAv3 rotate-half
/// geometry [1,6,1370,32] -> halves at the real production size, per Rule 1).
/// </summary>
public abstract partial class MLTestBase
{
    private static float[] CpuSlice(float[] input, int[] inShape, int[] starts, int[] ends, int[] steps)
    {
        int rank = inShape.Length;
        var outShape = new int[rank];
        for (int d = 0; d < rank; d++) outShape[d] = (ends[d] - starts[d] + steps[d] - 1) / steps[d];
        int outCount = 1; foreach (var s in outShape) outCount *= s;
        var inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inShape[i + 1];
        var output = new float[outCount];
        for (int idx = 0; idx < outCount; idx++)
        {
            int rem = idx, inIdx = 0;
            for (int d = rank - 1; d >= 0; d--)
            {
                int c = rem % outShape[d]; rem /= outShape[d];
                inIdx += (starts[d] + c * steps[d]) * inStrides[d];
            }
            output[idx] = input[inIdx];
        }
        return output;
    }

    private static (int[] outShape, int[] inStrides, int total) SliceGeom(int[] inShape, int[] starts, int[] ends, int[] steps)
    {
        int rank = inShape.Length;
        var outShape = new int[rank];
        for (int d = 0; d < rank; d++) outShape[d] = (ends[d] - starts[d] + steps[d] - 1) / steps[d];
        var inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inShape[i + 1];
        int total = 1; foreach (var s in outShape) total *= s;
        return (outShape, inStrides, total);
    }

    [TestMethod]
    public async Task SliceKernel_Single_MatchesCPU() => await RunTest(async accelerator =>
    {
        // DAv3 rotate-half geometry at PRODUCTION size: [1,6,1370,32] -> [..., 16:32].
        int[] inShape = { 1, 6, 1370, 32 };
        int total = 1 * 6 * 1370 * 32;
        var input = RandomFloats(total, seed: 7);
        int[] starts = { 0, 0, 0, 16 }, ends = { 1, 6, 1370, 32 }, steps = { 1, 1, 1, 1 };
        var expected = CpuSlice(input, inShape, starts, ends, steps);
        var (outShape, inStrides, outTotal) = SliceGeom(inShape, starts, ends, steps);

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(outTotal);
        using var slice = new SliceKernel(accelerator);
        slice.Slice(inBuf.View, outBuf.View, starts, steps, outShape, inStrides, 4, outTotal);
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, outTotal);
        AssertClose(expected, got, 0f, "SliceKernel single (rotate-half geometry): ");
        Console.WriteLine("[SliceRace] single Slice at DAv3 rotate-half geometry - exact match");
    });

    [TestMethod]
    public async Task SliceKernel_MixedRankHistory_MatchesCPU() => await RunTest(async accelerator =>
    {
        // The production ingredient the equal-rank tests miss: the registry's ONE SliceKernel serves
        // MIXED ranks (rank-1 shape slices, rank-5 5-D slices, rank-4 rope slices), so _paramsBuf
        // GROWS (inline-disposing the old buffer while its dispatch may be batched) and later takes
        // sub-length SubView(0,N) writes into an OVERSIZED buffer. DAv3's first divergence executed
        // the rotate-half with steps[3]=32 instead of 1 - params content wrong at execution time.
        // Sequence: rank-1 -> rank-5 (grow) -> rank-4 rotate-half (subview into larger buf) ->
        // rank-1 (stale tail), all batched without intervening syncs, then verify EVERY output.
        using var slice = new SliceKernel(accelerator);

        // rank-1: [8] -> [3:7]
        var in1 = RandomFloats(8, seed: 31);
        int[] s1 = { 3 }, e1 = { 7 }, st1 = { 1 };
        var exp1 = CpuSlice(in1, new[] { 8 }, s1, e1, st1);
        var (os1, is1, t1) = SliceGeom(new[] { 8 }, s1, e1, st1);
        using var in1B = accelerator.Allocate1D(in1);
        using var out1B = accelerator.Allocate1D<float>(t1);
        slice.Slice(in1B.View, out1B.View, s1, st1, os1, is1, 1, t1);

        // rank-5 (grows params buffer 4 -> 20 ints; old 4-int buffer disposed inline while
        // the rank-1 dispatch above may still be batched): [1,2,3,10,8] -> [...,2:8,1:5]
        int[] inShape5 = { 1, 2, 3, 10, 8 };
        int tot5 = 1 * 2 * 3 * 10 * 8;
        var in5 = RandomFloats(tot5, seed: 37);
        int[] s5 = { 0, 0, 0, 2, 1 }, e5 = { 1, 2, 3, 8, 5 }, st5 = { 1, 1, 1, 1, 1 };
        var exp5 = CpuSlice(in5, inShape5, s5, e5, st5);
        var (os5, is5, t5) = SliceGeom(inShape5, s5, e5, st5);
        using var in5B = accelerator.Allocate1D(in5);
        using var out5B = accelerator.Allocate1D<float>(t5);
        slice.Slice(in5B.View, out5B.View, s5, st5, os5, is5, 5, t5);

        // rank-4 rotate-half at production size (SubView(0,16) write into the 20-int buffer):
        int[] inShape4 = { 1, 6, 1370, 32 };
        int tot4 = 1 * 6 * 1370 * 32;
        var in4 = RandomFloats(tot4, seed: 41);
        int[] s4 = { 0, 0, 0, 16 }, e4 = { 1, 6, 1370, 32 }, st4 = { 1, 1, 1, 1 };
        var exp4 = CpuSlice(in4, inShape4, s4, e4, st4);
        var (os4, is4, t4) = SliceGeom(inShape4, s4, e4, st4);
        using var in4B = accelerator.Allocate1D(in4);
        using var out4B = accelerator.Allocate1D<float>(t4);
        slice.Slice(in4B.View, out4B.View, s4, st4, os4, is4, 4, t4);

        // trailing rank-1 (overwrites packed[0..3] while the rank-4 dispatch may still be batched)
        using var out1cB = accelerator.Allocate1D<float>(t1);
        slice.Slice(in1B.View, out1cB.View, s1, st1, os1, is1, 1, t1);

        await accelerator.SynchronizeAsync();
        AssertClose(exp1, await out1B.CopyToHostAsync<float>(0, t1), 0f, "mixed-rank rank-1(first): ");
        AssertClose(exp5, await out5B.CopyToHostAsync<float>(0, t5), 0f, "mixed-rank rank-5: ");
        AssertClose(exp4, await out4B.CopyToHostAsync<float>(0, t4), 0f, "mixed-rank rank-4 rotate-half: ");
        AssertClose(exp1, await out1cB.CopyToHostAsync<float>(0, t1), 0f, "mixed-rank rank-1(trailing): ");
        Console.WriteLine("[SliceRace] mixed-rank history (grow + oversized-subview + stale-tail) - all exact");
    });

    [TestMethod]
    public async Task SliceKernel_OffsetSubViews_MatchesCPU() => await RunTest(async accelerator =>
    {
        // Production tensors are SubViews at NON-ZERO offsets into big pooled buffers (BufferPool),
        // not whole buffers - the one structural ingredient the whole-buffer tests don't cover.
        // Same rotate-half geometry, but input and output live at odd offsets inside larger buffers.
        int[] inShape = { 1, 6, 1370, 32 };
        int total = 1 * 6 * 1370 * 32;
        var input = RandomFloats(total, seed: 21);
        int[] starts = { 0, 0, 0, 16 }, ends = { 1, 6, 1370, 32 }, steps = { 1, 1, 1, 1 };
        var expected = CpuSlice(input, inShape, starts, ends, steps);
        var (outShape, inStrides, outTotal) = SliceGeom(inShape, starts, ends, steps);

        const int inOfs = 12352, outOfs = 7040; // odd-ish but 64-float aligned like pool suballocations
        using var inPool = accelerator.Allocate1D<float>(inOfs + total + 1024);
        using var outPool = accelerator.Allocate1D<float>(outOfs + outTotal + 512);
        inPool.View.SubView(inOfs, total).CopyFromCPU(input);

        using var slice = new SliceKernel(accelerator);
        slice.Slice(inPool.View.SubView(inOfs, total), outPool.View.SubView(outOfs, outTotal),
            starts, steps, outShape, inStrides, 4, outTotal);
        await accelerator.SynchronizeAsync();
        var got = await outPool.CopyToHostAsync<float>(outOfs, outTotal);
        AssertClose(expected, got, 0f, "SliceKernel offset subviews: ");
        Console.WriteLine("[SliceRace] offset-subview Slice at rotate-half geometry - exact match");
    });

    [TestMethod]
    public async Task SliceKernel_BackToBack_NoSync_BothCorrect() => await RunTest(async accelerator =>
    {
        // TWO Slices from ONE kernel instance (registry pattern), DIFFERENT params, NO sync between
        // dispatches - the q-rope/k-rope pattern. If the params upload races the batched dispatch,
        // Slice A executes with B's params: A's output becomes the [0:16] half instead of [16:32].
        int[] inShape = { 1, 6, 1370, 32 };
        int total = 1 * 6 * 1370 * 32;
        var inputA = RandomFloats(total, seed: 11);
        var inputB = RandomFloats(total, seed: 13);
        int[] startsA = { 0, 0, 0, 16 }, endsA = { 1, 6, 1370, 32 };   // second half (rotate-half)
        int[] startsB = { 0, 0, 0, 0 }, endsB = { 1, 6, 1370, 16 };    // first half
        int[] steps = { 1, 1, 1, 1 };
        var expectedA = CpuSlice(inputA, inShape, startsA, endsA, steps);
        var expectedB = CpuSlice(inputB, inShape, startsB, endsB, steps);
        var (outShapeA, inStrides, outTotal) = SliceGeom(inShape, startsA, endsA, steps);
        var (outShapeB, _, _) = SliceGeom(inShape, startsB, endsB, steps);

        using var inBufA = accelerator.Allocate1D(inputA);
        using var inBufB = accelerator.Allocate1D(inputB);
        using var outBufA = accelerator.Allocate1D<float>(outTotal);
        using var outBufB = accelerator.Allocate1D<float>(outTotal);
        using var slice = new SliceKernel(accelerator);

        slice.Slice(inBufA.View, outBufA.View, startsA, steps, outShapeA, inStrides, 4, outTotal);
        slice.Slice(inBufB.View, outBufB.View, startsB, steps, outShapeB, inStrides, 4, outTotal); // no sync between
        await accelerator.SynchronizeAsync();

        var gotA = await outBufA.CopyToHostAsync<float>(0, outTotal);
        var gotB = await outBufB.CopyToHostAsync<float>(0, outTotal);
        AssertClose(expectedB, gotB, 0f, "SliceKernel back-to-back, B: ");
        // A is the race victim: if it ran with B's params its values come from the wrong half.
        AssertClose(expectedA, gotA, 0f, "SliceKernel back-to-back, A (params-race victim if wrong): ");
        Console.WriteLine("[SliceRace] back-to-back Slices, shared instance, no intervening sync - both exact");
    });
}
