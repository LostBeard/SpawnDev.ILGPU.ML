using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for AsyncMDE components: SpatialMemoryUnit and async pipeline.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task AsyncMDE_SMU_ConvexCombination() => await RunTest(async accelerator =>
    {
        var smu = new SpatialMemoryKernel(accelerator);
        int n = 4;

        // trust=1 → output=memory, trust=0 → output=fastPath
        var trust = new float[] { 1f, 0f, 0.5f, 0.3f };
        var memory = new float[] { 10f, 10f, 10f, 10f };
        var fast = new float[] { 0f, 0f, 0f, 0f };

        using var tBuf = accelerator.Allocate1D(trust);
        using var mBuf = accelerator.Allocate1D(memory);
        using var fBuf = accelerator.Allocate1D(fast);
        using var oBuf = accelerator.Allocate1D<float>(n);

        smu.Combine(tBuf.View, mBuf.View, fBuf.View, oBuf.View, n);
        await accelerator.SynchronizeAsync();
        var output = await oBuf.CopyToHostAsync<float>(0, n);

        // trust=1: output = 1*10 + 0*0 = 10
        if (MathF.Abs(output[0] - 10f) > 0.01f) throw new Exception($"trust=1: {output[0]:F2}, expected 10");
        // trust=0: output = 0*10 + 1*0 = 0
        if (MathF.Abs(output[1]) > 0.01f) throw new Exception($"trust=0: {output[1]:F2}, expected 0");
        // trust=0.5: output = 0.5*10 + 0.5*0 = 5
        if (MathF.Abs(output[2] - 5f) > 0.01f) throw new Exception($"trust=0.5: {output[2]:F2}, expected 5");
        // trust=0.3: output = 0.3*10 + 0.7*0 = 3
        if (MathF.Abs(output[3] - 3f) > 0.01f) throw new Exception($"trust=0.3: {output[3]:F2}, expected 3");

        Console.WriteLine("[AsyncMDE] SMU convex combination: all trust levels correct");
    });

    [TestMethod]
    public async Task AsyncMDE_EMA_DecayCorrect() => await RunTest(async accelerator =>
    {
        var smu = new SpatialMemoryKernel(accelerator);

        var memory = new float[] { 10f, 20f, 30f };
        var newVal = new float[] { 0f, 0f, 0f };
        float beta = 0.5f;

        using var mBuf = accelerator.Allocate1D(memory);
        using var nBuf = accelerator.Allocate1D(newVal);

        smu.EMAUpdate(mBuf.View, nBuf.View, 3, beta);
        await accelerator.SynchronizeAsync();
        var updated = await mBuf.CopyToHostAsync<float>(0, 3);

        // memory = 0.5 * old + 0.5 * 0 = old/2
        if (MathF.Abs(updated[0] - 5f) > 0.01f) throw new Exception($"EMA[0]={updated[0]:F2}, expected 5");
        if (MathF.Abs(updated[1] - 10f) > 0.01f) throw new Exception($"EMA[1]={updated[1]:F2}, expected 10");
        if (MathF.Abs(updated[2] - 15f) > 0.01f) throw new Exception($"EMA[2]={updated[2]:F2}, expected 15");

        Console.WriteLine("[AsyncMDE] EMA decay: memory correctly blended with new values");
    });

    [TestMethod]
    public async Task Mamba3_MIMO_MatchesSingleIO() => await RunTest(async accelerator =>
    {
        // MIMO with dModel=1 should match standard scan
        var scan = new SelectiveScanKernel(accelerator);
        int seqLen = 4, dState = 2;

        var x = new float[] { 1f, 2f, 3f, 4f };
        var A = new float[] { 0.9f, 0.8f };
        var B = new float[] { 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f };
        var C = new float[] { 0.3f, 0.4f, 0.3f, 0.4f, 0.3f, 0.4f, 0.3f, 0.4f };

        // Standard scan
        using var xBuf = accelerator.Allocate1D(x);
        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D(C);
        using var outStd = accelerator.Allocate1D<float>(seqLen);
        using var stateStd = accelerator.Allocate1D<float>(dState);

        scan.Forward(xBuf.View, aBuf.View, bBuf.View, cBuf.View, outStd.View, stateStd.View,
            1, seqLen, dState);

        // MIMO with dModel=1
        using var outMimo = accelerator.Allocate1D<float>(seqLen);
        using var stateMimo = accelerator.Allocate1D<float>(dState);

        scan.ForwardMIMO(xBuf.View, aBuf.View, bBuf.View, cBuf.View, outMimo.View, stateMimo.View,
            1, seqLen, dState, 1);

        await accelerator.SynchronizeAsync();
        var stdOut = await outStd.CopyToHostAsync<float>(0, seqLen);
        var mimoOut = await outMimo.CopyToHostAsync<float>(0, seqLen);

        float maxDiff = 0;
        for (int i = 0; i < seqLen; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(stdOut[i] - mimoOut[i]));

        if (maxDiff > 0.01f)
            throw new Exception($"MIMO(dModel=1) differs from standard scan: maxDiff={maxDiff:F6}");

        Console.WriteLine($"[Mamba-3] MIMO matches standard scan: maxDiff={maxDiff:E3}");
    });

    [TestMethod]
    public async Task Mamba3_ZeroInput_ZeroOutput() => await RunTest(async accelerator =>
    {
        var scan = new SelectiveScanKernel(accelerator);
        int seqLen = 4, dState = 2;

        var x = new float[seqLen]; // all zeros
        var A = new float[] { 0.9f, 0.8f };
        var B = new float[seqLen * dState]; // zeros
        var C = new float[seqLen * dState]; // zeros
        for (int i = 0; i < seqLen * dState; i++) { B[i] = 0.1f; C[i] = 0.1f; }

        using var xBuf = accelerator.Allocate1D(x);
        using var aBuf = accelerator.Allocate1D(A);
        using var bBuf = accelerator.Allocate1D(B);
        using var cBuf = accelerator.Allocate1D(C);
        using var outBuf = accelerator.Allocate1D<float>(seqLen);
        using var stateBuf = accelerator.Allocate1D<float>(dState);

        scan.Forward(xBuf.View, aBuf.View, bBuf.View, cBuf.View, outBuf.View, stateBuf.View, 1, seqLen, dState);
        await accelerator.SynchronizeAsync();
        var output = await outBuf.CopyToHostAsync<float>(0, seqLen);

        // h_t = A*h_{t-1} + B*0 = A*h_{t-1}, h_0=0 → h_t=0 for all t → y_t=0
        for (int i = 0; i < seqLen; i++)
            if (MathF.Abs(output[i]) > 1e-6f)
                throw new Exception($"Zero input should give zero output: output[{i}]={output[i]:F6}");

        Console.WriteLine("[Mamba-3] Zero input → zero output: linearity check PASS");
    });

    [TestMethod]
    public async Task Mamba3_ConstantMemory_StateSize() => await RunTest(async accelerator =>
    {
        // State size should be dState regardless of sequence length
        var scan = new SelectiveScanKernel(accelerator);
        int dState = 4;

        // Run with seqLen=10 and seqLen=100 — same state buffer size
        foreach (int seqLen in new[] { 10, 100 })
        {
            var rng = new Random(42);
            var x = new float[seqLen];
            var B = new float[seqLen * dState];
            var C = new float[seqLen * dState];
            for (int i = 0; i < seqLen; i++) x[i] = (float)rng.NextDouble();
            for (int i = 0; i < seqLen * dState; i++) { B[i] = 0.1f; C[i] = 0.1f; }
            var A = new float[] { 0.9f, 0.8f, 0.95f, 0.85f };

            using var xBuf = accelerator.Allocate1D(x);
            using var aBuf = accelerator.Allocate1D(A);
            using var bBuf = accelerator.Allocate1D(B);
            using var cBuf = accelerator.Allocate1D(C);
            using var outBuf = accelerator.Allocate1D<float>(seqLen);
            using var stateBuf = accelerator.Allocate1D<float>(dState); // ALWAYS dState, not seqLen

            scan.Forward(xBuf.View, aBuf.View, bBuf.View, cBuf.View, outBuf.View, stateBuf.View,
                1, seqLen, dState);
            await accelerator.SynchronizeAsync();
            var output = await outBuf.CopyToHostAsync<float>(0, seqLen);

            // Should produce valid output without OOM
            bool hasNonZero = output.Any(v => MathF.Abs(v) > 1e-6f);
            if (!hasNonZero)
                throw new Exception($"seqLen={seqLen}: all-zero output");
        }

        Console.WriteLine("[Mamba-3] Constant memory: state=dState regardless of seqLen PASS");
    });

    [TestMethod]
    public async Task AsyncMDE_Pipeline_MemoryWarms() => await RunTest(async accelerator =>
    {
        var pipeline = new SpawnDev.ILGPU.ML.Pipelines.AsyncDepthPipeline(
            accelerator, width: 4, height: 4, slowPathInterval: 2);

        if (pipeline.MemoryWarmed)
            throw new Exception("Memory should not be warmed before first frame");

        using var fastOutput = accelerator.Allocate1D<float>(16);
        using var output = accelerator.Allocate1D<float>(16);

        // First frame should warm memory
        // Zero the buffer without aliasing (WebGPU forbids same buffer in/out)
        var zeros = new float[16];
        using var zerosBuf = accelerator.Allocate1D(zeros);
        zerosBuf.View.CopyTo(fastOutput.View);
        var result = await pipeline.ProcessFrameAsync(fastOutput.View, output.View);

        if (!pipeline.MemoryWarmed)
            throw new Exception("Memory should be warmed after first frame");
        if (result.FrameIndex != 1)
            throw new Exception($"FrameIndex should be 1, got {result.FrameIndex}");

        pipeline.Dispose();
        Console.WriteLine("[AsyncMDE] Pipeline: memory warms on first frame PASS");
    });
}
