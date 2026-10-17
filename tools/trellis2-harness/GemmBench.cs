using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>GEMM / attention throughput of the tensor-core kernels at the DiT's real shapes (TFLOPS = 2*M*N*K / time).</summary>
internal static class GemmBench
{
    public static int Run()
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        var k = new DiTKernels(accel);
        var rng = new Random(5);
        float[] Rand(long n) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        foreach (var (M, K, N, name) in new[] { (12288, 1536, 4608, "qkv"), (12288, 1536, 8192, "mlp1"), (12288, 8192, 1536, "mlp2"), (12288, 1536, 1536, "out") })
        {
            using var x = accel.Allocate1D(Rand((long)M * K));
            var wHost = new BFloat16[(long)N * K];
            for (long i = 0; i < wHost.Length; i++) wHost[i] = (BFloat16)(float)((rng.NextDouble() * 2 - 1) * 0.05);
            using var w = accel.Allocate1D(wHost);
            using var o = accel.Allocate1D<float>((long)M * N);
            k.Linear(x.View, w.View, null, o.View, M, K, N); accel.Synchronize();
            var sw = Stopwatch.StartNew();
            const int iters = 10;
            for (int i = 0; i < iters; i++) k.Linear(x.View, w.View, null, o.View, M, K, N);
            accel.Synchronize();
            double sec = sw.Elapsed.TotalSeconds / iters;
            Console.WriteLine($"gemm {name,-5} M={M} K={K} N={N}: {sec * 1000,7:F2} ms  {2.0 * M * N * K / sec / 1e12,6:F2} TFLOPS");
        }
        foreach (var (nq, nk, name) in new[] { (12288, 12288, "self attn"), (12288, 1029, "cross attn") })
        {
            const int H = 12, D = 128, C = H * D;
            using var qkv = accel.Allocate1D(Rand((long)Math.Max(nq, nk) * 3 * C));
            using var o = accel.Allocate1D<float>((long)nq * C);
            k.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, default, o.View, nq, nk, H, D); accel.Synchronize();
            var sw = Stopwatch.StartNew();
            const int iters = 5;
            for (int i = 0; i < iters; i++) k.Attention(qkv.View, 3 * C, 0, qkv.View, 3 * C, C, 2 * C, default, o.View, nq, nk, H, D);
            accel.Synchronize();
            double sec = sw.Elapsed.TotalSeconds / iters;
            Console.WriteLine($"{name,-10} Nq={nq} Nk={nk}: {sec * 1000,7:F2} ms  {4.0 * nq * nk * C / sec / 1e12,6:F2} TFLOPS");
        }
        return 0;
    }
}
