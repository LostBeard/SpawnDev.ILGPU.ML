using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Anima;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="AnimaKernels.LoraDown"/> (a LoRA's t = x A^T) against a float64 host product at sizes off its tiling (37 rows, K = 300, rank 7: a K remainder and a
    /// partial last group of four outputs), then <see cref="AnimaKernels.AxpyInPlace"/> and <see cref="AnimaKernels.ActivateInPlace"/> (exact GELU) against the fused
    /// Linear epilogue on the same values. Every backend: the LoRA path runs wherever the Anima DiT does.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task AnimaKernels_LoraPieces_MatchHost() => await RunTest(async accelerator =>
    {
        const int M = 37, K = 300, R = 7, N = 50;
        var rng = new Random(9);
        var xh = new float[M * K]; for (int i = 0; i < xh.Length; i++) xh[i] = (float)(rng.NextDouble() * 2 - 1);
        var ah = new BFloat16[R * K]; for (int i = 0; i < ah.Length; i++) ah[i] = (BFloat16)(float)(rng.NextDouble() * 2 - 1);
        var ak = new AnimaKernels(accelerator);
        using var x = accelerator.Allocate1D(xh); using var a = accelerator.Allocate1D(ah); using var t = accelerator.Allocate1D<float>(M * R);
        var want = new float[M * R];
        for (int m = 0; m < M; m++) for (int j = 0; j < R; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)xh[m * K + k] * (float)ah[j * K + k]; want[m * R + j] = (float)s; }
        string downNote;
        if (ak.LoraDown(x.View, a.View, t.View, M, K, R))
        {
            await accelerator.SynchronizeAsync();
            double e = F2_RelRms(await t.View.CopyToHostAsync(), want);
            Ur_Expect(e < 1e-6, $"LoraDown is {e:E2} from float64");
            downNote = $"LoraDown {e:E1}";
        }
        else downNote = "LoraDown n/a (groups under 256 threads: the Linear runs)";
        // o += s u, then the exact GELU in place, against Linear(..., GeluErf) of the same pre-activation
        var oh = new float[M * N]; var uh = new float[M * N];
        for (int i = 0; i < oh.Length; i++) { oh[i] = (float)(rng.NextDouble() * 6 - 3); uh[i] = (float)(rng.NextDouble() * 2 - 1); }
        using var o = accelerator.Allocate1D(oh); using var u = accelerator.Allocate1D(uh);
        ak.AxpyInPlace(o.View, u.View, 0.5f, M * N);
        ak.ActivateInPlace(o.View, DiTKernels.Act.GeluErf, M * N);
        await accelerator.SynchronizeAsync();
        var got = await o.View.CopyToHostAsync();
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double z = oh[i] + 0.5 * uh[i];
            double gelu = 0.5 * z * (1 + Erf(z / Math.Sqrt(2)));
            worst = Math.Max(worst, Math.Abs(got[i] - gelu));
        }
        Ur_Expect(worst < 2e-6, $"o += s u then GELU is {worst:E2} from float64 (Abramowitz-Stegun erf: 1.5e-7 relative)");
        string line = $"{accelerator.AcceleratorType}: {downNote}; axpy + GELU max |diff| {worst:E1}";
        Console.WriteLine("[Anima] LoRA pieces: " + line);
        return line;

        static double Erf(double v)
        {
            // erf from Numerical Recipes' erfc Chebyshev fit (fractional error under 1.2e-7: with the device's 1.5e-7 the bar of 2e-6 on |z| <= 4 holds)
            double z = Math.Abs(v), t2 = 1.0 / (1.0 + 0.5 * z);
            double r = t2 * Math.Exp(-z * z - 1.26551223 + t2 * (1.00002368 + t2 * (0.37409196 + t2 * (0.09678418 + t2 * (-0.18628806 + t2 * (0.27886807 + t2 * (-1.13520398
                + t2 * (1.48851587 + t2 * (-0.82215223 + t2 * 0.17087277)))))))));
            return v >= 0 ? 1 - r : r - 1;
        }
    });

    /// <summary>
    /// Where a rank-32 LoRA's time goes on one Anima Linear (1024 tokens, the shapes of q_proj 2048 -> 2048 and mlp.layer1 / layer2 2048 <-> 8192): the base Linear,
    /// t = x A^T (N = 32: few output tiles, a long K), u = t B^T (K = 32), and o += s u. Random bf16 data, precise activations; ms per op over 20 repeats after a warm-up.
    /// A measurement for the LoRA path's speed work, no pass bar.
    /// </summary>
    [TestMethod(Timeout = 300000)]
    public async Task Benchmark_AnimaLora_LinearShapes() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("a desktop GPU timing");
        var k = new DiTKernels(accelerator) { PreciseActivations = true };
        var ak = new AnimaKernels(accelerator);
        const int M = 1024, R = 32;
        var rng = new Random(5);
        var lines = new List<string>();
        foreach (var (name, K, N) in new[] { ("q_proj", 2048, 2048), ("mlp.layer1", 2048, 8192), ("mlp.layer2", 8192, 2048) })
        {
            BFloat16[] Rand(long n, float s) { var a = new BFloat16[n]; for (long i = 0; i < n; i++) a[i] = (BFloat16)((float)(rng.NextDouble() * 2 - 1) * s); return a; }
            var xh = new float[(long)M * K]; for (long i = 0; i < xh.Length; i++) xh[i] = (float)(rng.NextDouble() * 2 - 1);
            using var x = accelerator.Allocate1D(xh);
            using var w = accelerator.Allocate1D(Rand((long)N * K, 0.02f));
            using var a = accelerator.Allocate1D(Rand((long)R * K, 0.02f));
            using var b = accelerator.Allocate1D(Rand((long)N * R, 0.001f));
            using var o = accelerator.Allocate1D<float>((long)M * N);
            using var t = accelerator.Allocate1D<float>((long)M * R);
            using var u = accelerator.Allocate1D<float>((long)M * N);
            async Task<double> Time(Action op)
            {
                op(); await accelerator.SynchronizeAsync();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < 20; i++) op();
                await accelerator.SynchronizeAsync();
                return sw.Elapsed.TotalMilliseconds / 20;
            }
            double tBase = await Time(() => k.Linear(x.View, w.View, null, o.View, M, K, N, DiTKernels.Act.None));
            double tA = await Time(() => k.Linear(x.View, a.View, null, t.View, M, K, R, DiTKernels.Act.None));
            var tGemm = await t.View.CopyToHostAsync();
            double tDown = await Time(() => ak.LoraDown(x.View, a.View, t.View, M, K, R));
            var tKernel = await t.View.CopyToHostAsync();
            double diff = F2_RelRms(tKernel, tGemm);
            Ur_Expect(diff < 1e-5, $"{name}: LoraDown differs from the Linear by {diff:E2}");   // both fp32 sums of the same bf16 products, different orders
            double tB = await Time(() => k.Linear(t.View, b.View, null, u.View, M, R, N, DiTKernels.Act.None));
            double tAxpy = await Time(() => ak.AxpyInPlace(o.View, u.View, 1f, M * N));
            lines.Add($"{name} [{N}, {K}]: base {tBase:F3} ms, x A^T {tA:F3} (Linear) / {tDown:F3} (LoraDown, {diff:E1} apart), t B^T {tB:F3}, o += s u {tAxpy:F3} (LoRA {(tDown + tB + tAxpy) / tBase * 100:F0}% of base)");
        }
        string line = $"{accelerator.AcceleratorType}: " + string.Join(" | ", lines);
        Console.WriteLine("[Benchmark] Anima LoRA shapes: " + line);
        return line;
    });
}
