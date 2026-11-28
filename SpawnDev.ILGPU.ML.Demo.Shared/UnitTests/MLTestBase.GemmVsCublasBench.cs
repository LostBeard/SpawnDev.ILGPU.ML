using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The small-M GEMV path of <see cref="DiTKernels.Linear"/> (M below <see cref="DiTKernels.GemvMaxRows"/>) against float64 on every backend, at sizes off its
    /// blocking: M = 1, 2, 3; K = 300 (not a multiple of its 256 threads); N = 13 (a partial block of 8 outputs); with a bf16 bias and each activation.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DiTKernels_SmallMLinear_MatchesFloat64() => await RunTest(async accelerator =>
    {
        const int K = 300, N = 13;
        var rng = new Random(17);
        var k = new DiTKernels(accelerator);
        var wh = new BFloat16[N * K]; for (int i = 0; i < wh.Length; i++) wh[i] = (BFloat16)(float)((rng.NextDouble() * 2 - 1) * 0.1);
        var bh = new BFloat16[N]; for (int i = 0; i < N; i++) bh[i] = (BFloat16)(float)(rng.NextDouble() - 0.5);
        using var w = accelerator.Allocate1D(wh); using var b = accelerator.Allocate1D(bh);
        var report = new List<string>(); double worst = 0;
        foreach (int M in new[] { 1, 2, 3 })
            foreach (var act in new[] { DiTKernels.Act.None, DiTKernels.Act.Silu, DiTKernels.Act.GeluErf })
            {
                var xh = new float[M * K]; for (int i = 0; i < xh.Length; i++) xh[i] = (float)(rng.NextDouble() * 4 - 2);
                using var x = accelerator.Allocate1D(xh); using var o = accelerator.Allocate1D<float>(M * N);
                k.Linear(x.View, w.View, b.View, o.View, M, K, N, act);
                await accelerator.SynchronizeAsync();
                var got = await o.View.CopyToHostAsync();
                double e = 0;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++)
                    {
                        double s = (float)bh[n];
                        for (int kk = 0; kk < K; kk++) s += (double)xh[m * K + kk] * (float)wh[n * K + kk];
                        double want = act switch
                        {
                            DiTKernels.Act.Silu => s / (1 + Math.Exp(-s)),
                            DiTKernels.Act.GeluErf => 0.5 * s * (1 + ErfApprox(s / Math.Sqrt(2))),
                            _ => s,
                        };
                        e = Math.Max(e, Math.Abs(got[m * N + n] - want));
                    }
                worst = Math.Max(worst, e);
                report.Add($"M {M} {act} {e:E1}");
            }
        string line = $"{accelerator.AcceleratorType}: max |diff| {worst:E2} ({string.Join(", ", report)})";
        Console.WriteLine("[Gemv] " + line);
        Ur_Expect(worst < 2e-5, line);   // fp32 sums of 300 products ~1; the device erf is Abramowitz-Stegun (1.5e-7)
        return line;

        static double ErfApprox(double v)
        {
            double z = Math.Abs(v), t = 1.0 / (1.0 + 0.5 * z);
            double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
            return v >= 0 ? 1 - r : r - 1;
        }
    });

    /// <summary>
    /// Our Linear paths against what PyTorch CUDA (cuBLAS, what ComfyUI runs) does on the same card (tools/anima-ref/torch_gemm_compare.py, RTX 4070,
    /// torch 2.14.1+cu126): speed at the DiT shapes (4096 tokens: qkv 1536 -> 4608, mlp-up 1536 -> 6144, mlp-down 6144 -> 1536) and accuracy on a
    /// random [256, 2048] x [2048, 2048] product (bf16-exact weights) against float64. cuBLAS: fp32 16.1-17.1 TFLOPS at 3.1e-7 (1024 rows), TF32 24-27 at
    /// 2.1e-4, bf16 50-54 at 2.6e-3. Paths here: CUDA tensor cores fast (bf16 activations) and precise (bf16 hi + lo), CUDA with tensor cores off (the
    /// tiled fp32 kernel), OpenCL / WebGPU tiled fp32. A measurement, no bar.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Benchmark_Gemm_VsCublas() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.WebGPU)) throw new UnsupportedTestException("a GPU timing");
        var rng = new Random(3);
        float Bf16Exact(double v) => BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits((float)v) & unchecked((int)0xFFFF0000));
        // accuracy case
        const int Ma = 256, Ka = 2048, Na = 2048;
        var xa = new float[Ma * Ka]; for (int i = 0; i < xa.Length; i++) xa[i] = (float)Gauss(rng);
        var wa = new float[Na * Ka]; for (int i = 0; i < wa.Length; i++) wa[i] = Bf16Exact(Gauss(rng) * 0.02);
        var refA = new double[Ma * Na];
        Parallel.For(0, Ma, m => { for (int n = 0; n < Na; n++) { double s = 0; for (int k = 0; k < Ka; k++) s += (double)xa[m * Ka + k] * wa[n * Ka + k]; refA[m * Na + n] = s; } });
        using var xA = accelerator.Allocate1D(xa); using var wA = accelerator.Allocate1D(wa.Select(v => (BFloat16)v).ToArray()); using var oA = accelerator.Allocate1D<float>(Ma * Na);
        // speed buffers
        const int T = 4096, MaxK = 6144, MaxN = 6144;
        using var x = accelerator.Allocate1D(Enumerable.Range(0, T * MaxK).Select(i => (float)Math.Sin(i * 0.37)).ToArray());
        using var w = accelerator.Allocate1D(Enumerable.Range(0, MaxN * 1536).Select(i => BFloat16.FromRawBits((ushort)(0x3C00 + (i % 255)))).ToArray());
        using var o = accelerator.Allocate1D<float>((long)T * MaxN);
        var kinds = new List<(string Name, DiTKernels K)>();
        if (accelerator.AcceleratorType == AcceleratorType.Cuda)
        {
            kinds.Add(("tensor cores fast (bf16 act)", new DiTKernels(accelerator) { PreciseActivations = false }));
            kinds.Add(("tensor cores precise (fp16 hi+lo, scaled)", new DiTKernels(accelerator) { PreciseActivations = true }));
        }
        else kinds.Add(("tiled fp32", new DiTKernels(accelerator)));
        var lines = new List<string>();
        bool mmaWas = DiTKernels.EnableTensorCores, splitWas = DiTKernels.F16SplitActivations;
        try
        {
            for (int pass = 0; pass < (accelerator.AcceleratorType == AcceleratorType.Cuda ? 3 : 1); pass++)
            {
                if (pass == 1) { DiTKernels.F16SplitActivations = false; kinds = new() { ("tensor cores precise (bf16 hi+lo, before)", new DiTKernels(accelerator) { PreciseActivations = true }) }; }
                if (pass == 2) { DiTKernels.F16SplitActivations = splitWas; DiTKernels.EnableTensorCores = false; kinds = new() { ("tiled fp32 (tensor cores off)", new DiTKernels(accelerator)) }; }
                foreach (var (name, k) in kinds)
                {
                    k.Linear(xA.View, wA.View, null, oA.View, Ma, Ka, Na);
                    await accelerator.SynchronizeAsync();
                    var got = await oA.View.CopyToHostAsync();
                    double num = 0, den = 0; for (int i = 0; i < got.Length; i++) { double d = got[i] - refA[i]; num += d * d; den += refA[i] * refA[i]; }
                    var parts = new List<string> { $"{Math.Sqrt(num / den):E2} from float64" };
                    foreach (var (sname, K, N) in new[] { ("qkv", 1536, 4608), ("mlp-up", 1536, 6144), ("mlp-down", 6144, 1536) })
                    {
                        var xv = x.View.SubView(0, (long)T * K); var wv = w.View.SubView(0, (long)N * K); var ov = o.View.SubView(0, (long)T * N);
                        for (int i = 0; i < 3; i++) k.Linear(xv, wv, null, ov, T, K, N);
                        await accelerator.SynchronizeAsync();
                        double best = double.MaxValue;
                        for (int r = 0; r < 10; r++)
                        {
                            var sw = System.Diagnostics.Stopwatch.StartNew();
                            k.Linear(xv, wv, null, ov, T, K, N);
                            await accelerator.SynchronizeAsync();
                            best = Math.Min(best, sw.Elapsed.TotalSeconds);
                        }
                        parts.Add($"{sname} {best * 1000:F2} ms {2.0 * T * K * N / best / 1e12:F1} TFLOPS");
                    }
                    lines.Add($"{name}: " + string.Join(", ", parts));
                }
            }
        }
        finally { DiTKernels.EnableTensorCores = mmaWas; DiTKernels.F16SplitActivations = splitWas; }
        // M = 1 (Anima's adaLN modulations and timestep MLP): the GEMV against the general path, accuracy (float64) and time
        {
            var k1 = new DiTKernels(accelerator) { PreciseActivations = true };
            var parts = new List<string>();
            foreach (bool gemv in new[] { true, false })
            {
                DiTKernels.EnableGemv = gemv;
                var ps = new List<string>();
                foreach (var (sname, K, N) in new[] { ("2048->256", 2048, 256), ("256->6144", 256, 6144), ("2048->6144", 2048, 6144) })
                {
                    var xr = new float[K]; for (int i = 0; i < K; i++) xr[i] = (float)Gauss(rng) * 3f;
                    var wr = new float[N * K]; for (int i = 0; i < wr.Length; i++) wr[i] = Bf16Exact(Gauss(rng) * 0.02);
                    var want = new double[N]; for (int n = 0; n < N; n++) { double s = 0; for (int kk = 0; kk < K; kk++) s += (double)xr[kk] * wr[n * K + kk]; want[n] = s; }
                    using var xb = accelerator.Allocate1D(xr); using var wb = accelerator.Allocate1D(wr.Select(v => (BFloat16)v).ToArray()); using var ob = accelerator.Allocate1D<float>(N);
                    k1.Linear(xb.View, wb.View, null, ob.View, 1, K, N);
                    await accelerator.SynchronizeAsync();
                    var got = await ob.View.CopyToHostAsync();
                    double num = 0, den = 0; for (int i = 0; i < N; i++) { double d = got[i] - want[i]; num += d * d; den += want[i] * want[i]; }
                    double best = double.MaxValue;
                    for (int r = 0; r < 20; r++)
                    {
                        var sw = System.Diagnostics.Stopwatch.StartNew();
                        k1.Linear(xb.View, wb.View, null, ob.View, 1, K, N);
                        await accelerator.SynchronizeAsync();
                        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                    }
                    ps.Add($"{sname} {Math.Sqrt(num / den):E1} {best:F3} ms");
                }
                parts.Add($"M = 1 {(gemv ? "GEMV" : "general")}: " + string.Join(", ", ps));
            }
            DiTKernels.EnableGemv = true;
            lines.Add(string.Join(" | ", parts));
        }
        string line = $"{accelerator.AcceleratorType}: " + string.Join(" | ", lines) + " (cuBLAS fp32 16.1-17.1 TFLOPS 3.1e-7; bf16 50-54 TFLOPS 2.6e-3)";
        Console.WriteLine("[Benchmark] GEMM vs cuBLAS: " + line);
        return line;

        static double Gauss(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());
    });
}
