using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GPU kernel tests for Blazing Edge features using reference data.
/// Tests RoPE, GroupNorm, SelectiveScan, and QKNorm on actual GPU.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task BlazingEdge_RoPE_GPU_MatchesReference() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var inputBytes = await http.GetByteArrayAsync("references/blazing-edge/rope_input.bin");
        var expectedBytes = await http.GetByteArrayAsync("references/blazing-edge/rope_output.bin");
        var input = new float[inputBytes.Length / 4];
        Buffer.BlockCopy(inputBytes, 0, input, 0, inputBytes.Length);
        var expected = new float[expectedBytes.Length / 4];
        Buffer.BlockCopy(expectedBytes, 0, expected, 0, expectedBytes.Length);

        // [1, 8, 64] = 8 positions, headDim=64
        int numPos = 8, headDim = 64;

        using var inputBuf = accelerator.Allocate1D<float>(input.Length);
        inputBuf.View.CopyFromCPU(input);
        using var outputBuf = accelerator.Allocate1D<float>(input.Length);

        var rope = new RoPEKernel(accelerator);
        rope.Apply(inputBuf.View, outputBuf.View, numPos, headDim);
        await accelerator.SynchronizeAsync();
        var gpuOutput = await outputBuf.CopyToHostAsync<float>(0, input.Length);

        float maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(gpuOutput[i] - expected[i]));

        Console.WriteLine($"[BlazingEdge] RoPE GPU: maxErr={maxErr:E3} vs reference");
        if (maxErr > 0.01f)
            throw new Exception($"RoPE GPU maxErr={maxErr:E3} exceeds tolerance 0.01");
    });

    [TestMethod]
    public async Task BlazingEdge_QKNorm_GPU_UnitLength() => await RunTest(async accelerator =>
    {
        // Verify QK-norm produces unit-length vectors
        var rng = new Random(42);
        int numVecs = 4, dim = 64;
        var data = new float[numVecs * dim];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 10 - 5); // large range

        using var inputBuf = accelerator.Allocate1D<float>(data.Length);
        inputBuf.View.CopyFromCPU(data);
        using var outputBuf = accelerator.Allocate1D<float>(data.Length);

        var qkNorm = new QKNormKernel(accelerator);
        qkNorm.NormalizeRows(inputBuf.View, outputBuf.View, numVecs, dim);
        await accelerator.SynchronizeAsync();
        var normalized = await outputBuf.CopyToHostAsync<float>(0, data.Length);

        // Each row should have unit length
        for (int v = 0; v < numVecs; v++)
        {
            float sumSq = 0;
            for (int d = 0; d < dim; d++)
                sumSq += normalized[v * dim + d] * normalized[v * dim + d];
            float len = MathF.Sqrt(sumSq);
            if (MathF.Abs(len - 1f) > 0.01f)
                throw new Exception($"QKNorm vector[{v}] length={len:F4}, expected ~1.0");
        }

        Console.WriteLine($"[BlazingEdge] QKNorm GPU: {numVecs} vectors normalized to unit length");
    });

    [TestMethod]
    public async Task BlazingEdge_SelectiveScan_GPU_Causal() => await RunTest(async accelerator =>
    {
        // Verify selective scan is causal: output[t] depends only on input[0..t]
        int seqLen = 6, dState = 4;
        var rng = new Random(42);

        var x = new float[seqLen];
        var A = new float[dState];
        var B = new float[seqLen * dState];
        var C = new float[seqLen * dState];
        for (int i = 0; i < seqLen; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < dState; i++) A[i] = 0.9f; // decay
        for (int i = 0; i < seqLen * dState; i++)
        {
            B[i] = (float)(rng.NextDouble() * 0.5);
            C[i] = (float)(rng.NextDouble() * 0.5);
        }

        using var xBuf = accelerator.Allocate1D<float>(seqLen);
        xBuf.View.CopyFromCPU(x);
        using var aBuf = accelerator.Allocate1D<float>(dState);
        aBuf.View.CopyFromCPU(A);
        using var bBuf = accelerator.Allocate1D<float>(seqLen * dState);
        bBuf.View.CopyFromCPU(B);
        using var cBuf = accelerator.Allocate1D<float>(seqLen * dState);
        cBuf.View.CopyFromCPU(C);
        using var outBuf = accelerator.Allocate1D<float>(seqLen);
        using var stateBuf = accelerator.Allocate1D<float>(dState);

        var scan = new SelectiveScanKernel(accelerator);
        scan.Forward(xBuf.View, aBuf.View, bBuf.View, cBuf.View, outBuf.View, stateBuf.View,
            1, seqLen, dState);
        await accelerator.SynchronizeAsync();
        var output = await outBuf.CopyToHostAsync<float>(0, seqLen);

        // Output should be non-zero (SSM produces output)
        bool hasNonZero = output.Any(v => MathF.Abs(v) > 1e-6f);
        if (!hasNonZero)
            throw new Exception("SelectiveScan output all zeros");

        // Verify causality: run with only first 3 tokens, check output[0:3] matches
        var xShort = new float[] { x[0], x[1], x[2] };
        var bShort = new float[3 * dState];
        var cShort = new float[3 * dState];
        Array.Copy(B, bShort, 3 * dState);
        Array.Copy(C, cShort, 3 * dState);

        using var xShortBuf = accelerator.Allocate1D<float>(3);
        xShortBuf.View.CopyFromCPU(xShort);
        using var bShortBuf = accelerator.Allocate1D<float>(3 * dState);
        bShortBuf.View.CopyFromCPU(bShort);
        using var cShortBuf = accelerator.Allocate1D<float>(3 * dState);
        cShortBuf.View.CopyFromCPU(cShort);
        using var outShortBuf = accelerator.Allocate1D<float>(3);
        using var stateShortBuf = accelerator.Allocate1D<float>(dState);

        scan.Forward(xShortBuf.View, aBuf.View, bShortBuf.View, cShortBuf.View,
            outShortBuf.View, stateShortBuf.View, 1, 3, dState);
        await accelerator.SynchronizeAsync();
        var shortOutput = await outShortBuf.CopyToHostAsync<float>(0, 3);

        // First 3 outputs should match (causality)
        for (int i = 0; i < 3; i++)
        {
            float diff = MathF.Abs(output[i] - shortOutput[i]);
            if (diff > 1e-4f)
                throw new Exception($"Causality violated: output[{i}]={output[i]:F6} vs short[{i}]={shortOutput[i]:F6}");
        }

        Console.WriteLine($"[BlazingEdge] SelectiveScan GPU: causal, output={string.Join(",", output.Select(v => v.ToString("F4")))}");
    });
}
