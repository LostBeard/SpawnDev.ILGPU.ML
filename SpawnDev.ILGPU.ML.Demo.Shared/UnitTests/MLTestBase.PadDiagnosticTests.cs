using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Diagnostic tests to isolate the Pad kernel's WebGPU failure.
/// Tests progressively simpler patterns to find the exact WGSL codegen issue.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Simplest possible pad: rank=1, constant mode, no out-of-bounds.
    /// Input [4], pad [1, 1] → output [6]. Middle 4 elements should equal input.
    /// </summary>
    [TestMethod]
    public async Task PadDiag_Rank1_Constant() => await RunTest(async accelerator =>
    {
        var input = new float[] { 10, 20, 30, 40 };
        int[] inputShape = { 4 };
        int[] pads = { 1, 1 };

        // Expected: [0, 10, 20, 30, 40, 0]
        var expected = new float[] { 0, 10, 20, 30, 40, 0 };

        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(6);

        var pad = new PadKernel(accelerator);
        pad.Forward(inBuf.View, outBuf.View, inputShape, pads, mode: 0, constantValue: 0f);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 6);
        AssertClose(expected, actual, 1e-6f, "PadDiag rank1 constant: ");
    });

    /// <summary>
    /// Test if a simple kernel that reads a buffer-driven loop bound works on WebGPU.
    /// This isolates the "for (d = 0; d &lt; p[0]; d++)" pattern.
    /// </summary>
    [TestMethod]
    public async Task PadDiag_BufferLoopBound() => await RunTest(async accelerator =>
    {
        // Use a simple accumulation kernel: output[idx] = sum of p[1..p[0]]
        // This tests: reading loop bound from buffer, computed buffer indices in loop
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(BufferLoopBoundKernel);

        // p = [3, 10, 20, 30] → loop 3 times, sum p[1]+p[2]+p[3] = 60
        var paramsData = new int[] { 3, 10, 20, 30 };
        using var paramsBuf = accelerator.Allocate1D(paramsData);
        using var outBuf = accelerator.Allocate1D<float>(4);

        kernel(4, outBuf.View, paramsBuf.View);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 4);
        // Every thread should compute 60.0
        var expected = new float[] { 60, 60, 60, 60 };
        AssertClose(expected, actual, 1e-6f, "BufferLoopBound: ");
    });

    private static void BufferLoopBoundKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int count = p[0];
        float sum = 0f;
        for (int i = 0; i < count; i++)
            sum += p[1 + i];
        output[idx] = sum;
    }

    /// <summary>
    /// Test computed buffer indices: p[base + factor * dynamicVar + loopVar].
    /// This is the exact pattern in PadKernel: p[2 + 4 * rank + d].
    /// </summary>
    [TestMethod]
    public async Task PadDiag_ComputedBufferIndex() => await RunTest(async accelerator =>
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(ComputedIndexKernel);

        // p = [2, 100, 200, 300, 400, 10, 20]
        // rank=p[0]=2, values at p[2 + 2*rank + d] = p[6+0]=10, p[6+1]=20
        // Actually: p[2 + 2*2 + 0] = p[6] = 10, p[2 + 2*2 + 1] = p[7] = 20
        var paramsData = new int[] { 2, 999, 100, 200, 300, 400, 10, 20 };
        using var paramsBuf = accelerator.Allocate1D(paramsData);
        using var outBuf = accelerator.Allocate1D<float>(2);

        kernel(2, outBuf.View, paramsBuf.View);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 2);
        // Thread 0: sum p[2+2*2+0] + p[2+2*2+1] = 10 + 20 = 30
        // Thread 1: same = 30
        var expected = new float[] { 30, 30 };
        AssertClose(expected, actual, 1e-6f, "ComputedBufferIndex: ");
    });

    private static void ComputedIndexKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int rank = p[0];
        float sum = 0f;
        for (int d = 0; d < rank; d++)
            sum += p[2 + 2 * rank + d];
        output[idx] = sum;
    }

    /// <summary>
    /// Test the break-inside-loop pattern: for loop with conditional break and a bool flag.
    /// This is the exact pattern in PadKernel's constant mode: "outOfBounds = true; break;"
    /// </summary>
    [TestMethod]
    public async Task PadDiag_BreakInsideLoop() => await RunTest(async accelerator =>
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(BreakInsideLoopKernel);

        // p = [4, 10, 20, 30, 40]
        // Loop sums p[1..4] but breaks if value > 25 → sum = 10 + 20 = 30
        var paramsData = new int[] { 4, 10, 20, 30, 40 };
        using var paramsBuf = accelerator.Allocate1D(paramsData);
        using var outBuf = accelerator.Allocate1D<float>(2);

        kernel(2, outBuf.View, paramsBuf.View);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 2);
        // Each thread: 10 + 20 = 30 (breaks before 30)
        // broken flag is true → output = -sum = -30
        var expected = new float[] { -30, -30 };
        AssertClose(expected, actual, 1e-6f, "BreakInsideLoop: ");
    });

    private static void BreakInsideLoopKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<int, Stride1D.Dense> p)
    {
        int count = p[0];
        float sum = 0f;
        bool broken = false;
        for (int i = 0; i < count; i++)
        {
            int val = p[1 + i];
            if (val > 25)
            {
                broken = true;
                break;
            }
            sum += val;
        }
        // Use both sum and broken flag to verify the break worked correctly
        output[idx] = broken ? -sum : sum;
    }

    /// <summary>
    /// Dump the WGSL source for PadImpl kernel.
    /// This test loads the kernel, triggers compilation, and prints the generated WGSL.
    /// </summary>
    [TestMethod]
    public async Task PadDiag_DumpWGSL() => await RunTest(async accelerator =>
    {
        // Load the PadKernel to trigger WGSL compilation
        var pad = new PadKernel(accelerator);
        // Run a trivial pad to force kernel compilation
        var input = new float[] { 1f };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(3);
        pad.Forward(inBuf.View, outBuf.View, new[] { 1 }, new[] { 1, 1 }, mode: 0, constantValue: 0f);
        await accelerator.SynchronizeAsync();

        // Collect WGSL for Pad kernel — extract just the main function body
        string? padWgsl = null;
        foreach (var kvp in WebGPUBackend.WGSLRegistry)
        {
            if (kvp.Key.Contains("Pad", StringComparison.OrdinalIgnoreCase))
                padWgsl = kvp.Value.Source;
        }
        if (padWgsl == null)
            throw new Exception("No Pad WGSL found");

        // Find "fn main" and dump from there
        int mainIdx = padWgsl.IndexOf("fn main(");
        if (mainIdx >= 0)
            padWgsl = padWgsl.Substring(mainIdx);

        // Split into chunks and throw each as a separate line
        // to avoid truncation
        throw new Exception($"WGSL_MAIN ({padWgsl.Length} chars):\n{padWgsl}");
    });

    /// <summary>
    /// Test ternary with bool: output = boolVar ? A : B.
    /// Isolates the conditional expression pattern used in PadKernel's final line.
    /// </summary>
    [TestMethod]
    public async Task PadDiag_TernaryBool() => await RunTest(async accelerator =>
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            float>(TernaryBoolKernel);

        var input = new float[] { 1, 2, 3, 4 };
        using var inBuf = accelerator.Allocate1D(input);
        using var outBuf = accelerator.Allocate1D<float>(4);

        kernel(4, inBuf.View, outBuf.View, -99f);
        await accelerator.SynchronizeAsync();

        var actual = await outBuf.CopyToHostAsync<float>(0, 4);
        // idx 0,1: even → flag=true → output=-99
        // idx 2,3: even → flag=true → output=-99
        // Actually: 0%2=0(true), 1%2=1(false), 2%2=0(true), 3%2=1(false)
        var expected = new float[] { -99, 2, -99, 4 };
        AssertClose(expected, actual, 1e-6f, "TernaryBool: ");
    });

    private static void TernaryBoolKernel(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        float constVal)
    {
        bool flag = (idx % 2) == 0;
        output[idx] = flag ? constVal : input[idx];
    }
}
