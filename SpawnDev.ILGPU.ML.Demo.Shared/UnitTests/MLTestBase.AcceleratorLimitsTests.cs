using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Report each backend's launch limits - the numbers that decide which kernel every hot path picks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 🔴 THESE NUMBERS ARE DISPATCH LOGIC, AND NOBODY HAD WRITTEN THEM DOWN. Conv2DKernel,
    /// RegisterBlockedMatMul, FusedLinearKernel and Vec4LoadMatMul all choose between a barrier-heavy
    /// 16x16-group tiled kernel and a naive one on the SAME test:
    /// <c>MaxNumThreadsPerGroup &gt;= 256</c>. So a backend's answer to that one property silently decides
    /// whether it runs the fast path or the slow one, everywhere.
    /// </para>
    /// <para>
    /// MEASURED 2026-09-08, this machine, all six lanes:
    /// <code>
    /// backend  MaxNumThreadsPerGroup  MaxGroupSize        Warp  MaxNumThreads  16x16 tiled
    /// CPU      64                     (64,64,64)          8     768            excluded
    /// Cuda     1024                   (1024,1024,64)      32    70656          ELIGIBLE
    /// OpenCL   1024                   (1024,1024,64)      32    47104          ELIGIBLE
    /// Wasm     256                    (256,1,1)           8     3072           ELIGIBLE
    /// WebGL    1                      (1,1,1)             1     16777215       excluded
    /// WebGPU   1024                   (1024,1024,64)      32    16384          ELIGIBLE
    /// </code>
    /// </para>
    /// <para>
    /// ✅ So Conv2DKernel's comment - tiled is "not WebGL, not CPU" - is ACCURATE, though it excludes CPU by
    /// CAPABILITY (64 &lt; 256) rather than by name as it does WebGL. Worth having measured rather than assumed:
    /// the exclusion rests entirely on a number nobody had written down.
    /// </para>
    /// <para>
    /// 🔴 WHAT THE NUMBERS ACTUALLY EXPLAIN. The CPU lane is not slow because it took a kernel shaped for a
    /// GPU - it is slow because the naive kernel is the ONLY path it has, and it has <b>768 threads in total</b>
    /// against WebGPU's 16,384 and CUDA's 70,656. One thread per output element, each looping inC*kH*kW with no
    /// blocking or vectorisation, is what a 1024x1024 U-2-Net forward or a 1.2B-parameter prefill has to run on.
    /// That is the finding behind the CPU-lane timeouts, and making it fast is kernel work, not a bigger cap.
    /// </para>
    /// <para>
    /// This is a REPORT, not a threshold - it asserts only that the accelerator answers coherently, because
    /// there is no "correct" group size to demand of a backend. Its value is the printed line, which is why
    /// it is cheap and runs on every lane. Read it with PMT_CONSOLE_LOG=AccelLimits.
    /// </para>
    /// </remarks>
    [TestMethod(Timeout = 60000)]
    public async Task AcceleratorLimits_AreReported() => await RunTest(async accelerator =>
    {
        var maxGroup = accelerator.MaxNumThreadsPerGroup;
        var maxGroupDim = accelerator.MaxGroupSize;
        var warp = accelerator.WarpSize;
        var mps = accelerator.MaxNumThreads;
        var sharedPerGroup = accelerator.MaxSharedMemoryPerGroup;

        // 256 = RbBlock * RbBlock, the gate every tiled kernel in this repo uses.
        const int TiledGate = 256;
        bool tiledEligible = maxGroup >= TiledGate;

        Console.WriteLine($"[AccelLimits] {accelerator.AcceleratorType}: "
            + $"MaxNumThreadsPerGroup={maxGroup} MaxGroupSize=({maxGroupDim.X},{maxGroupDim.Y},{maxGroupDim.Z}) "
            + $"WarpSize={warp} MaxNumThreads={mps} SharedMemPerGroup={sharedPerGroup} "
            + $"=> 16x16 tiled kernels {(tiledEligible ? "ELIGIBLE" : "excluded")}");

        // A 16x16 group is launched as Index2D(256, 1) by Conv2DKernel, so the X axis alone must hold 256.
        // A backend can pass the total-threads gate and still be unable to take that shape.
        bool xAxisFits = maxGroupDim.X >= TiledGate;
        if (tiledEligible && !xAxisFits)
            Console.WriteLine($"[AccelLimits] ⚠️ {accelerator.AcceleratorType} passes the total-threads gate "
                + $"({maxGroup} >= {TiledGate}) but its X axis caps at {maxGroupDim.X}, and Conv2DKernel "
                + "launches Index2D(256, 1). Total threads is the WRONG gate for that launch shape.");

        if (maxGroup <= 0) throw new Exception($"MaxNumThreadsPerGroup={maxGroup} - the accelerator is not answering");
        if (maxGroupDim.X <= 0) throw new Exception($"MaxGroupSize.X={maxGroupDim.X} - the accelerator is not answering");
        if (warp <= 0) throw new Exception($"WarpSize={warp} - the accelerator is not answering");
        await Task.CompletedTask;
    });
}
