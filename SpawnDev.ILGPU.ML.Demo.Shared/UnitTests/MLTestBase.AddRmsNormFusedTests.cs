using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The fused residual Add + RMSNorm (<c>NormalizationKernels.AddRMSNormFused</c>, every decode layer's norm) against a CPU f64 reference:
    /// residualOut = a + b exactly (the same float add), normedOut = x / sqrt(mean(x^2) + eps) * w within float rounding, for 1 and 3 rows of
    /// 2,048 (the qwen3:1.7b decode row), 1,536 and 4,096 values. Runs the tree kernel and, on CUDA, the warp-shuffle kernel
    /// (AddRMSNormFusedShuffleImpl), checking the route each time. Added 2026-11-09 with the shuffle kernel (no direct test existed).
    /// </summary>
    [TestMethod]
    public async Task<string> AddRmsNormFused_MatchesCpu() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("AddRMSNormFused is not used on WebGL (two kernels instead)");
        using var norm = new SpawnDev.ILGPU.ML.Kernels.NormalizationKernels(accelerator);
        var notes = new List<string>();
        bool saved = SpawnDev.ILGPU.ML.Kernels.NormalizationKernels.AddRmsShuffle;
        try
        {
            foreach (var (rows, C) in new[] { (1, 2048), (3, 2048), (1, 1536), (2, 4096) })
            {
                var rng = new Random(17 + rows * 31 + C);
                var a = new float[rows * C]; var b = new float[rows * C]; var w = new float[C];
                for (int i = 0; i < a.Length; i++) { a[i] = (float)(rng.NextDouble() * 4 - 2); b[i] = (float)(rng.NextDouble() * 4 - 2); }
                for (int i = 0; i < C; i++) w[i] = (float)(rng.NextDouble() + 0.5);
                const float eps = 1e-6f;
                var wantRes = new float[rows * C]; var wantNorm = new double[rows * C];
                for (int r = 0; r < rows; r++)
                {
                    double ss = 0;
                    for (int i = 0; i < C; i++) { float x = a[r * C + i] + b[r * C + i]; wantRes[r * C + i] = x; ss += (double)x * x; }
                    double inv = 1.0 / Math.Sqrt(ss / C + eps);
                    for (int i = 0; i < C; i++) wantNorm[r * C + i] = wantRes[r * C + i] * inv * w[i];
                }
                using var aBuf = accelerator.Allocate1D(a); using var bBuf = accelerator.Allocate1D(b); using var wBuf = accelerator.Allocate1D(w);
                using var resBuf = accelerator.Allocate1D<float>(rows * C); using var normBuf = accelerator.Allocate1D<float>(rows * C);
                foreach (bool shuffle in new[] { false, true })
                {
                    SpawnDev.ILGPU.ML.Kernels.NormalizationKernels.AddRmsShuffle = shuffle;
                    string want = shuffle && accelerator.AcceleratorType == AcceleratorType.Cuda ? "shuffle" : "tree";
                    resBuf.MemSetToZero(); normBuf.MemSetToZero();   // a stale result from the other kernel must not pass
                    if (!norm.AddRMSNormFused(aBuf.View, bBuf.View, resBuf.View, normBuf.View, wBuf.View, rows, C, eps, 1))
                        throw new UnsupportedTestException("AddRMSNormFused not available on this backend");
                    string route = norm.LastAddRmsKernel;
                    await accelerator.SynchronizeAsync();
                    var gotRes = await resBuf.CopyToHostAsync<float>(0, rows * C);
                    var gotNorm = await normBuf.CopyToHostAsync<float>(0, rows * C);
                    if (route != want) throw new Exception($"rows={rows} C={C}: AddRMSNormFused ran on '{route}', not '{want}'");
                    double worst = 0;
                    for (int i = 0; i < gotRes.Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(gotRes[i]) != BitConverter.SingleToInt32Bits(wantRes[i]))
                            throw new Exception($"{route} rows={rows} C={C}: residual[{i}] = {gotRes[i]:R}, expected {wantRes[i]:R} (a + b must be exact)");
                        double e = Math.Abs(gotNorm[i] - wantNorm[i]) / Math.Max(1e-3, Math.Abs(wantNorm[i]));
                        worst = Math.Max(worst, e);
                    }
                    if (!(worst < 2e-6)) throw new Exception($"{route} rows={rows} C={C}: normalized output off by {worst:E2} relative (limit 2e-6)");
                    notes.Add($"{route} {rows}x{C} worst {worst:E1}");
                }
            }
        }
        finally { SpawnDev.ILGPU.ML.Kernels.NormalizationKernels.AddRmsShuffle = saved; }
        return string.Join("; ", notes);
    });
}
