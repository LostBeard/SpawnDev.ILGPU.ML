using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// RMSNormalization OPERATOR tests — the production node path (Resolve -> Execute), executed against
/// an independent CPU oracle. These exist because the GGUF builder used to emit "LayerNormalization"
/// for the RMS case, which routed to the MEAN-CENTERED LayerNorm kernel (wrong math) and crashed on
/// the absent bias — never caught because every RMS-decoder test was STRUCTURAL (BuildGraph node
/// inspection, no execution). The discriminator test below is the regression guard: it uses an input
/// where true RMSNorm and mean-centered LayerNorm give DIFFERENT results, so a regression to the old
/// path fails loudly.
/// </summary>
public abstract partial class MLTestBase
{
    // CPU oracle: true RMSNorm (NO mean subtraction). output = x / sqrt(mean(x^2) + eps) * weight.
    private static float[] RmsNormCpu(float[] x, float[]? weight, int rows, int C, float eps)
    {
        var outp = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double sumSq = 0;
            for (int c = 0; c < C; c++) { double v = x[r * C + c]; sumSq += v * v; }
            float invRms = 1f / MathF.Sqrt((float)(sumSq / C) + eps);
            for (int c = 0; c < C; c++)
                outp[r * C + c] = x[r * C + c] * invRms * (weight != null ? weight[c] : 1f);
        }
        return outp;
    }

    [TestMethod]
    public async Task RMSNorm_Weighted_MatchesCPU() => await RunTest(async accelerator =>
    {
        const int rows = 7, C = 48;
        const float eps = 1e-6f;
        var rng = new Random(917);
        var x = new float[rows * C];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        var weight = new float[C];
        for (int i = 0; i < C; i++) weight[i] = (float)(rng.NextDouble() * 1.5 + 0.25);

        var expected = RmsNormCpu(x, weight, rows, C, eps);

        using var inBuf = accelerator.Allocate1D(x);
        using var wBuf = accelerator.Allocate1D(weight);
        using var outBuf = accelerator.Allocate1D<float>(x.Length);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("RMSNormalization");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[]
            {
                new Tensor(inBuf.View, new[] { rows, C }),
                new Tensor(wBuf.View, new[] { C }),
            },
            Outputs = new[] { new Tensor(outBuf.View, new[] { rows, C }) },
            Attributes = new Dictionary<string, object> { ["epsilon"] = eps, ["axis"] = -1 },
            Pool = pool,
            InputNames = new[] { "x", "weight" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, x.Length);

        AssertCloseQuant(got, expected, 2e-4f, "RMSNorm weighted");
        Console.WriteLine("[RMSNorm] weighted (2-input) matches CPU oracle");
    });

    [TestMethod]
    public async Task RMSNorm_Weightless_MatchesCPU() => await RunTest(async accelerator =>
    {
        // gemma4's V-norm: ggml_rms_norm with NO weight (unit gain). 1-input node.
        const int rows = 5, C = 64;
        const float eps = 1e-6f;
        var rng = new Random(404);
        var x = new float[rows * C];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);

        var expected = RmsNormCpu(x, null, rows, C, eps);

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(x.Length);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("RMSNormalization");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { rows, C }) },  // no weight -> weightless
            Outputs = new[] { new Tensor(outBuf.View, new[] { rows, C }) },
            Attributes = new Dictionary<string, object> { ["epsilon"] = eps },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, x.Length);

        AssertCloseQuant(got, expected, 2e-4f, "RMSNorm weightless");
        Console.WriteLine("[RMSNorm] weightless (1-input, gemma4 V-norm) matches CPU oracle");
    });

    [TestMethod]
    public async Task RMSNorm_IsNotMeanCentered_RegressionGuard() => await RunTest(async accelerator =>
    {
        // THE regression guard for the floor bug. Each row is a NON-ZERO-MEAN constant: all elements
        // equal to a value v. True RMSNorm -> each element / sqrt(v^2 + eps) ≈ sign(v) (≈ ±1).
        // Mean-centered LayerNorm (the OLD wrong path) subtracts the row mean (= v) first -> ALL ZEROS.
        // So this input separates the two: assert the RMS answer, which a regression to LayerNorm cannot pass.
        const int rows = 4, C = 32;
        const float eps = 1e-6f;
        var x = new float[rows * C];
        float[] rowVals = { 3f, -2.5f, 7f, 0.5f };
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < C; c++) x[r * C + c] = rowVals[r];

        var expected = RmsNormCpu(x, null, rows, C, eps); // ≈ sign(rowVals[r]) for every element

        using var inBuf = accelerator.Allocate1D(x);
        using var outBuf = accelerator.Allocate1D<float>(x.Length);
        using var registry = new OperatorRegistry(accelerator);
        var op = registry.Resolve("RMSNormalization");
        var pool = new BufferPool(accelerator);
        op.Execute(new OnnxOpContext
        {
            Inputs = new[] { new Tensor(inBuf.View, new[] { rows, C }) },
            Outputs = new[] { new Tensor(outBuf.View, new[] { rows, C }) },
            Attributes = new Dictionary<string, object> { ["epsilon"] = eps },
            Pool = pool,
            InputNames = new[] { "x" },
        });
        await accelerator.SynchronizeAsync();
        var got = await outBuf.CopyToHostAsync<float>(0, x.Length);

        // Each element must be ≈ ±1 (RMS), emphatically NOT ≈ 0 (mean-centered LayerNorm).
        for (int r = 0; r < rows; r++)
        {
            float exp = MathF.Sign(rowVals[r]); // sqrt(v^2)=|v| so v/|v| = sign(v)
            if (MathF.Abs(got[r * C] - exp) > 1e-3f)
                throw new Exception($"row {r}: RMSNorm gave {got[r * C]}, expected ≈ {exp} (±1). " +
                    $"A value near 0 means a regression to MEAN-CENTERED LayerNorm.");
        }
        AssertCloseQuant(got, expected, 2e-4f, "RMSNorm not-mean-centered");
        Console.WriteLine("[RMSNorm] confirmed TRUE RMS (no mean subtraction) — floor-bug regression guard");
    });
}
