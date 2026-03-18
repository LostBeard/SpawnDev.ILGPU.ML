using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// Shared test fixture: creates Context + CPUAccelerator once per test collection.
/// All tests use the CPU backend — no GPU required, bit-exact deterministic results.
/// </summary>
public class AcceleratorFixture : IDisposable
{
    public Context Context { get; }
    public Accelerator Accelerator { get; }

    public AcceleratorFixture()
    {
        Context = MLContext.CreateContext();
        Accelerator = Context.CreateCPUAccelerator(0);
    }

    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}

[CollectionDefinition("Accelerator")]
public class AcceleratorCollection : ICollectionFixture<AcceleratorFixture> { }

/// <summary>
/// Base class for kernel tests. Provides accelerator access and assertion helpers.
/// </summary>
[Collection("Accelerator")]
public abstract class KernelTestBase
{
    protected readonly Accelerator Accelerator;

    protected KernelTestBase(AcceleratorFixture fixture)
    {
        Accelerator = fixture.Accelerator;
    }

    /// <summary>
    /// Assert two float arrays are element-wise close within tolerance.
    /// Reports the first mismatch with index and values.
    /// </summary>
    protected static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-4f, string label = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxErr = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float err = MathF.Abs(expected[i] - actual[i]);
            if (err > maxErr) { maxErr = err; worstIdx = i; }
        }
        if (maxErr > tolerance)
        {
            Assert.Fail($"{label}Max error {maxErr:E3} at index {worstIdx}: expected={expected[worstIdx]:F6}, actual={actual[worstIdx]:F6} (tolerance={tolerance:E1})");
        }
    }

    /// <summary>Generate random float array with given seed.</summary>
    protected static float[] RandomFloats(int count, int seed = 42, float scale = 1f)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return data;
    }

    /// <summary>CPU reference MatMul: C[M,N] = A[M,K] × B[K,N].</summary>
    protected static float[] CpuMatMul(float[] A, float[] B, int M, int K, int N)
    {
        var C = new float[M * N];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < N; c++)
            {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += A[r * K + k] * B[k * N + c];
                C[r * N + c] = s;
            }
        return C;
    }

    /// <summary>CPU reference LayerNorm: normalize rows of [rows, C], apply gamma + beta.</summary>
    protected static float[] CpuLayerNorm(float[] input, float[] gamma, float[] beta, int rows, int C, float eps = 1e-6f)
    {
        var output = new float[rows * C];
        for (int r = 0; r < rows; r++)
        {
            float sum = 0;
            for (int i = 0; i < C; i++) sum += input[r * C + i];
            float mean = sum / C;
            float varSum = 0;
            for (int i = 0; i < C; i++) { float d = input[r * C + i] - mean; varSum += d * d; }
            float invStd = 1f / MathF.Sqrt(varSum / C + eps);
            for (int i = 0; i < C; i++)
                output[r * C + i] = gamma[i] * ((input[r * C + i] - mean) * invStd) + beta[i];
        }
        return output;
    }
}
