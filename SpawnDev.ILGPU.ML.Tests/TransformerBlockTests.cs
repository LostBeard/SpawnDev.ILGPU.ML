namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// End-to-end TransformerBlock tests — verifies the full composition of
/// LayerNorm → QKV → Attention → Proj → LayerScale → Residual → LayerNorm → MLP → LayerScale → Residual.
/// If this passes on CPU, the composition is correct and the SpawnScene divergence is WebGPU-specific.
/// </summary>
public class TransformerBlockTests : KernelTestBase
{
    private const int C = 384;
    private const int H = 6;
    private const int D = 64;
    private const int MLP_DIM = 1536;

    public TransformerBlockTests(AcceleratorFixture fixture) : base(fixture) { }

    [Theory]
    [InlineData(4, "tiny")]
    [InlineData(37, "small grid")]
    [InlineData(1370, "full DAv3 T_FULL")]
    public void TransformerBlock_Forward_IsStable(int T, string label)
    {
        // Create kernels
        var matMul = new MatMulKernel(Accelerator);
        var layerNorm = new LayerNormKernel(Accelerator);
        var softmax = new SoftmaxKernel(Accelerator);
        var elementWise = new ElementWiseKernels(Accelerator);
        var attention = new AttentionKernels(Accelerator);
        var block = new TransformerBlock(matMul, layerNorm, softmax, elementWise, attention);

        // Create random weights (scaled like a real model)
        var weights = CreateRandomBlockWeights(Accelerator, seed: 42);
        using var tmpBuffers = new TransformerBlock.TempBuffers(Accelerator, T);

        // Create random input
        var inputData = RandomFloats(T * C, seed: 100, scale: 0.5f);
        using var inputBuf = Accelerator.Allocate1D(inputData);
        using var outputBuf = Accelerator.Allocate1D<float>(T * C);

        // Run block
        block.Forward(inputBuf.View, outputBuf.View, weights.Weights, T, tmpBuffers);
        Accelerator.Synchronize();

        var output = outputBuf.GetAsArray1D();

        // Check output is finite
        for (int i = 0; i < output.Length; i++)
            Assert.True(float.IsFinite(output[i]), $"{label}: output[{i}] is {output[i]}");

        // Check output is not all zeros
        float maxAbs = output.Max(v => MathF.Abs(v));
        Assert.True(maxAbs > 1e-6f, $"{label}: output is all zeros");

        // Check std is reasonable (not exploding)
        double sum = 0, sumSq = 0;
        for (int i = 0; i < output.Length; i++) { sum += output[i]; sumSq += (double)output[i] * output[i]; }
        double mean = sum / output.Length;
        double std = Math.Sqrt(sumSq / output.Length - mean * mean);
        Assert.True(std < 100, $"{label}: output std={std:F2} is too large (exploding)");
        Assert.True(std > 1e-6, $"{label}: output std={std:E2} is too small");
    }

    [Fact]
    public void TransformerBlock_TwoBlocks_StdDoesNotExplode()
    {
        int T = 1370;
        var matMul = new MatMulKernel(Accelerator);
        var layerNorm = new LayerNormKernel(Accelerator);
        var softmax = new SoftmaxKernel(Accelerator);
        var elementWise = new ElementWiseKernels(Accelerator);
        var attention = new AttentionKernels(Accelerator);
        var block = new TransformerBlock(matMul, layerNorm, softmax, elementWise, attention);

        // Create two sets of random weights (different blocks)
        var w0 = CreateRandomBlockWeights(Accelerator, seed: 42);
        var w1 = CreateRandomBlockWeights(Accelerator, seed: 43);
        using var tmp = new TransformerBlock.TempBuffers(Accelerator, T);

        var inputData = RandomFloats(T * C, seed: 100, scale: 0.5f);
        using var buf1 = Accelerator.Allocate1D(inputData);
        using var buf2 = Accelerator.Allocate1D<float>(T * C);

        // Run block 0
        block.Forward(buf1.View, buf2.View, w0.Weights, T, tmp);
        Accelerator.Synchronize();
        double std0 = ComputeStd(buf2.GetAsArray1D());

        // Run block 1
        block.Forward(buf2.View, buf1.View, w1.Weights, T, tmp);
        Accelerator.Synchronize();
        double std1 = ComputeStd(buf1.GetAsArray1D());

        // Std should not grow more than 5× between blocks (healthy transformer)
        Assert.True(std1 < std0 * 5, $"Std exploded: block0={std0:F4}, block1={std1:F4} (ratio={std1 / std0:F2})");
    }

    [Fact]
    public void TransformerBlock_Forward_Deterministic()
    {
        int T = 37;
        var matMul = new MatMulKernel(Accelerator);
        var layerNorm = new LayerNormKernel(Accelerator);
        var softmax = new SoftmaxKernel(Accelerator);
        var elementWise = new ElementWiseKernels(Accelerator);
        var attention = new AttentionKernels(Accelerator);
        var block = new TransformerBlock(matMul, layerNorm, softmax, elementWise, attention);

        var weights = CreateRandomBlockWeights(Accelerator, seed: 42);
        var inputData = RandomFloats(T * C, seed: 100, scale: 0.5f);

        // Run twice
        float[] output1, output2;
        using (var tmp = new TransformerBlock.TempBuffers(Accelerator, T))
        {
            using var inBuf = Accelerator.Allocate1D(inputData);
            using var outBuf = Accelerator.Allocate1D<float>(T * C);
            block.Forward(inBuf.View, outBuf.View, weights.Weights, T, tmp);
            Accelerator.Synchronize();
            output1 = outBuf.GetAsArray1D();
        }
        using (var tmp = new TransformerBlock.TempBuffers(Accelerator, T))
        {
            using var inBuf = Accelerator.Allocate1D(inputData);
            using var outBuf = Accelerator.Allocate1D<float>(T * C);
            block.Forward(inBuf.View, outBuf.View, weights.Weights, T, tmp);
            Accelerator.Synchronize();
            output2 = outBuf.GetAsArray1D();
        }

        AssertClose(output1, output2, 0f, "Deterministic: ");
    }

    private static double ComputeStd(float[] data)
    {
        double sum = 0, sumSq = 0;
        for (int i = 0; i < data.Length; i++) { sum += data[i]; sumSq += (double)data[i] * data[i]; }
        double mean = sum / data.Length;
        return Math.Sqrt(sumSq / data.Length - mean * mean);
    }

    /// <summary>
    /// Create random block weights with realistic scales.
    /// Returns a disposable wrapper holding all weight buffers.
    /// </summary>
    private static WeightBuffers CreateRandomBlockWeights(Accelerator acc, int seed)
    {
        var rng = new Random(seed);
        float[] Make(int size, float scale)
        {
            var data = new float[size];
            for (int i = 0; i < size; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
            return data;
        }

        // LayerNorm: gamma ≈ 1.0, beta ≈ 0.0
        var norm1W = Make(C, 0.3f); for (int i = 0; i < C; i++) norm1W[i] = MathF.Abs(norm1W[i]) + 0.7f;
        var norm1B = Make(C, 0.01f);
        var norm2W = Make(C, 0.3f); for (int i = 0; i < C; i++) norm2W[i] = MathF.Abs(norm2W[i]) + 0.7f;
        var norm2B = Make(C, 0.01f);

        // MatMul weights: Xavier init scale ≈ sqrt(2/(fan_in+fan_out))
        var qkvW = Make(C * 3 * C, MathF.Sqrt(2f / (C + 3 * C)));
        var qkvB = Make(3 * C, 0.01f);
        var projW = Make(C * C, MathF.Sqrt(2f / (C + C)));
        var projB = Make(C, 0.01f);
        var fc1W = Make(C * MLP_DIM, MathF.Sqrt(2f / (C + MLP_DIM)));
        var fc1B = Make(MLP_DIM, 0.01f);
        var fc2W = Make(MLP_DIM * C, MathF.Sqrt(2f / (MLP_DIM + C)));
        var fc2B = Make(C, 0.01f);

        // LayerScale: small positive values (typical for DINOv2)
        var ls1 = Make(C, 0.01f); for (int i = 0; i < C; i++) ls1[i] = MathF.Abs(ls1[i]) + 0.001f;
        var ls2 = Make(C, 0.01f); for (int i = 0; i < C; i++) ls2[i] = MathF.Abs(ls2[i]) + 0.001f;

        var bufs = new WeightBuffers
        {
            Norm1WBuf = acc.Allocate1D(norm1W), Norm1BBuf = acc.Allocate1D(norm1B),
            QkvWBuf = acc.Allocate1D(qkvW), QkvBBuf = acc.Allocate1D(qkvB),
            ProjWBuf = acc.Allocate1D(projW), ProjBBuf = acc.Allocate1D(projB),
            Ls1Buf = acc.Allocate1D(ls1),
            Norm2WBuf = acc.Allocate1D(norm2W), Norm2BBuf = acc.Allocate1D(norm2B),
            Fc1WBuf = acc.Allocate1D(fc1W), Fc1BBuf = acc.Allocate1D(fc1B),
            Fc2WBuf = acc.Allocate1D(fc2W), Fc2BBuf = acc.Allocate1D(fc2B),
            Ls2Buf = acc.Allocate1D(ls2),
        };
        bufs.Weights = new TransformerBlock.BlockWeights
        {
            Norm1Weight = bufs.Norm1WBuf.View, Norm1Bias = bufs.Norm1BBuf.View,
            QkvWeight = bufs.QkvWBuf.View, QkvBias = bufs.QkvBBuf.View,
            ProjWeight = bufs.ProjWBuf.View, ProjBias = bufs.ProjBBuf.View,
            Ls1Gamma = bufs.Ls1Buf.View,
            Norm2Weight = bufs.Norm2WBuf.View, Norm2Bias = bufs.Norm2BBuf.View,
            Fc1Weight = bufs.Fc1WBuf.View, Fc1Bias = bufs.Fc1BBuf.View,
            Fc2Weight = bufs.Fc2WBuf.View, Fc2Bias = bufs.Fc2BBuf.View,
            Ls2Gamma = bufs.Ls2Buf.View,
        };
        return bufs;
    }

    private class WeightBuffers : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> Norm1WBuf, Norm1BBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> QkvWBuf, QkvBBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> ProjWBuf, ProjBBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> Ls1Buf;
        public MemoryBuffer1D<float, Stride1D.Dense> Norm2WBuf, Norm2BBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> Fc1WBuf, Fc1BBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> Fc2WBuf, Fc2BBuf;
        public MemoryBuffer1D<float, Stride1D.Dense> Ls2Buf;
        public TransformerBlock.BlockWeights Weights;

        public void Dispose()
        {
            Norm1WBuf?.Dispose(); Norm1BBuf?.Dispose();
            QkvWBuf?.Dispose(); QkvBBuf?.Dispose();
            ProjWBuf?.Dispose(); ProjBBuf?.Dispose();
            Ls1Buf?.Dispose();
            Norm2WBuf?.Dispose(); Norm2BBuf?.Dispose();
            Fc1WBuf?.Dispose(); Fc1BBuf?.Dispose();
            Fc2WBuf?.Dispose(); Fc2BBuf?.Dispose();
            Ls2Buf?.Dispose();
        }
    }
}
