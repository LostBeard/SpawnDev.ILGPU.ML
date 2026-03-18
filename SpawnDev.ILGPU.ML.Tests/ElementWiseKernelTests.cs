namespace SpawnDev.ILGPU.ML.Tests;

public class ElementWiseKernelTests : KernelTestBase
{
    private readonly ElementWiseKernels _ew;

    public ElementWiseKernelTests(AcceleratorFixture fixture) : base(fixture)
    {
        _ew = new ElementWiseKernels(Accelerator);
    }

    [Fact]
    public void GELU_MatchesCpuErf()
    {
        int count = 1000;
        var input = RandomFloats(count, seed: 1, scale: 3f);

        // CPU erf-based GELU reference
        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            double erfVal = ErfApprox(x / Math.Sqrt(2.0));
            expected[i] = (float)(0.5 * x * (1.0 + erfVal));
        }

        using var inBuf = Accelerator.Allocate1D(input);
        using var outBuf = Accelerator.Allocate1D<float>(count);
        _ew.GELU(inBuf.View, outBuf.View, count);
        Accelerator.Synchronize();

        AssertClose(expected, outBuf.GetAsArray1D(), 1e-5f, "GELU: ");
    }

    [Fact]
    public void GELUInPlace_MatchesCpuErf()
    {
        int count = 1000;
        var input = RandomFloats(count, seed: 2, scale: 3f);

        var expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            double erfVal = ErfApprox(x / Math.Sqrt(2.0));
            expected[i] = (float)(0.5 * x * (1.0 + erfVal));
        }

        using var buf = Accelerator.Allocate1D((float[])input.Clone());
        _ew.GELUInPlace(buf.View, count);
        Accelerator.Synchronize();

        AssertClose(expected, buf.GetAsArray1D(), 1e-5f, "GELUInPlace: ");
    }

    [Fact]
    public void BroadcastMul_LayerScale_MatchesCpu()
    {
        int T = 1370, C = 384;
        var input = RandomFloats(T * C, seed: 10);
        var gamma = RandomFloats(C, seed: 11, scale: 0.1f);

        var expected = new float[T * C];
        for (int i = 0; i < T * C; i++)
            expected[i] = input[i] * gamma[i % C];

        using var inBuf = Accelerator.Allocate1D(input);
        using var gBuf = Accelerator.Allocate1D(gamma);
        using var outBuf = Accelerator.Allocate1D<float>(T * C);
        _ew.BroadcastMul(inBuf.View, gBuf.View, outBuf.View, T * C, C);
        Accelerator.Synchronize();

        AssertClose(expected, outBuf.GetAsArray1D(), 1e-5f, "BroadcastMul: ");
    }

    [Fact]
    public void AddBias_MatchesCpu()
    {
        int T = 1370, C = 1152;
        var data = RandomFloats(T * C, seed: 20);
        var bias = RandomFloats(C, seed: 21);

        var expected = (float[])data.Clone();
        for (int i = 0; i < T * C; i++)
            expected[i] += bias[i % C];

        using var buf = Accelerator.Allocate1D((float[])data.Clone());
        using var bBuf = Accelerator.Allocate1D(bias);
        _ew.AddBias(buf.View, bBuf.View, T * C, C);
        Accelerator.Synchronize();

        AssertClose(expected, buf.GetAsArray1D(), 1e-5f, "AddBias: ");
    }

    [Fact]
    public void TransposeLastTwo_RoundTrip()
    {
        int batch = 6, rows = 1370, cols = 64;
        var input = RandomFloats(batch * rows * cols, seed: 30);

        using var inBuf = Accelerator.Allocate1D(input);
        using var transBuf = Accelerator.Allocate1D<float>(batch * rows * cols);
        using var roundBuf = Accelerator.Allocate1D<float>(batch * rows * cols);

        // Forward: [batch, rows, cols] → [batch, cols, rows]
        _ew.TransposeLastTwo(inBuf.View, transBuf.View, batch, rows, cols);
        // Reverse: [batch, cols, rows] → [batch, rows, cols]
        _ew.TransposeLastTwo(transBuf.View, roundBuf.View, batch, cols, rows);
        Accelerator.Synchronize();

        AssertClose(input, roundBuf.GetAsArray1D(), 0f, "TransposeRoundTrip: ");
    }

    private static double ErfApprox(double x)
    {
        double ax = Math.Abs(x);
        const double p = 0.3275911;
        const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        const double a4 = -1.453152027, a5 = 1.061405429;
        double t = 1.0 / (1.0 + p * ax);
        double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        double erfAbs = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.Exp(-ax * ax);
        return x < 0 ? -erfAbs : erfAbs;
    }
}
