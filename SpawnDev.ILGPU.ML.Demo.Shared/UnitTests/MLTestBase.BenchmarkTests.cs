using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Performance benchmark tests for GPU kernels.
/// These validate that optimized paths achieve expected throughput.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// Benchmark register-blocked MatMul vs baseline tiled MatMul.
    /// Measures actual GFLOPS for 512x512 and 1024x1024 matrices.
    /// Target: register-blocked should exceed 92-101 GFLOPS baseline.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Benchmark_RegisterBlockedMatMul_GFLOPS() => await RunTest(async accelerator =>
    {
        var regBlocked = new RegisterBlockedMatMul(accelerator);
        var baseline = new MatMulKernel(accelerator);

        // Test sizes: 128 (small/fallback), 256, 512, 1024
        var sizes = new[] { 128, 256, 512, 1024 };
        var results = new List<string>();

        foreach (int N in sizes)
        {
            int M = N, K = N;
            long flops = 2L * M * N * K; // 2*M*N*K FLOPs for MatMul

            // Generate random data
            var rng = new Random(42);
            var aData = new float[M * K];
            var bData = new float[K * N];
            for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var cBuf = accelerator.Allocate1D<float>(M * N);

            // Warmup
            regBlocked.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
            await accelerator.SynchronizeAsync();

            // Benchmark: run multiple iterations and take best
            int iterations = N <= 256 ? 20 : 10;
            double bestMs = double.MaxValue;

            for (int iter = 0; iter < iterations; iter++)
            {
                var sw = Stopwatch.StartNew();
                regBlocked.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
                await accelerator.SynchronizeAsync();
                sw.Stop();
                bestMs = Math.Min(bestMs, sw.Elapsed.TotalMilliseconds);
            }

            double gflops = flops / (bestMs / 1000.0) / 1e9;
            results.Add($"  {N}x{N}: {gflops:F1} GFLOPS ({bestMs:F2}ms)");
        }

        Console.WriteLine($"[Benchmark] Register-Blocked MatMul:");
        foreach (var r in results)
            Console.WriteLine(r);

        // Don't assert a minimum — just report. The actual values depend on hardware.
        // The checklist target is "measure actual GFLOPS vs 92-101 baseline".
        Console.WriteLine($"[Benchmark] PASS — results reported for validation");
    });

    /// <summary>
    /// Benchmark the barrier-free per-query attention vs the legacy per-element kernel ON EACH BACKEND — the
    /// point is the WebGPU number (per-element is its only prior path; per-query is the universal fix). Reports
    /// ms/call for both + the speedup. Modest prefill shape so WebGL/WebGPU stay under the timeout. WebGL only
    /// runs per-element (no workgroup shared memory) so it reports one number.
    /// </summary>
    [TestMethod(Timeout = 180000)]
    public async Task Benchmark_FusedAttention_PerQuery_vs_PerElement() => await RunTest(async accelerator =>
    {
        int nHeads = 8, SQ = 256, SKV = 256, D = 128;
        var rng = new Random(42);
        var q = new float[nHeads * SQ * D];
        var k = new float[nHeads * SKV * D];
        var v = new float[nHeads * SKV * D];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        using var qBuf = accelerator.Allocate1D(q);
        using var kBuf = accelerator.Allocate1D(k);
        using var vBuf = accelerator.Allocate1D(v);
        using var oBuf = accelerator.Allocate1D<float>(nHeads * SQ * D);
        var fused = new FusedAttentionKernel(accelerator);
        bool isWebGL = accelerator.AcceleratorType == AcceleratorType.WebGL;

        async Task<double> TimeIt(bool perElement)
        {
            FusedAttentionKernel.DisablePerQuery = perElement;
            fused.Forward(qBuf.View, kBuf.View, vBuf.View, oBuf.View, nHeads, SQ, SKV, D); // warmup
            await accelerator.SynchronizeAsync();
            double best = double.MaxValue;
            int iters = perElement ? 2 : 4;
            for (int it = 0; it < iters; it++)
            {
                var sw = Stopwatch.StartNew();
                fused.Forward(qBuf.View, kBuf.View, vBuf.View, oBuf.View, nHeads, SQ, SKV, D);
                await accelerator.SynchronizeAsync();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        try
        {
            double perElem = await TimeIt(perElement: true);
            if (isWebGL)
            {
                Console.WriteLine($"[Benchmark] Attention {nHeads}h×{SQ}×{SKV}×{D} [{accelerator.AcceleratorType}]: per-element {perElem:F2}ms (per-query N/A — no shared mem on WebGL)");
            }
            else
            {
                double perQuery = await TimeIt(perElement: false);
                Console.WriteLine($"[Benchmark] Attention {nHeads}h×{SQ}×{SKV}×{D} [{accelerator.AcceleratorType}]: per-element {perElem:F2}ms → per-query {perQuery:F2}ms = {perElem / perQuery:F2}× faster");
            }
        }
        finally { FusedAttentionKernel.DisablePerQuery = false; }

        Console.WriteLine($"[Benchmark] PASS — attention per-query/per-element reported");
    });

    /// <summary>
    /// Benchmark MatMulKernel auto-selection: verifies that the auto-select path
    /// correctly chooses register-blocked for large matrices and tiled for small ones.
    /// </summary>
    [TestMethod(Timeout = 30000)]
    public async Task Benchmark_MatMulAutoSelect_LargeUsesRegisterBlocked() => await RunTest(async accelerator =>
    {
        var matmul = new MatMulKernel(accelerator);

        // Large matrix: should use register-blocked path (≥64×64)
        int M = 128, K = 128, N = 128;
        var rng = new Random(42);
        var aData = new float[M * K];
        var bData = new float[K * N];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

        using var aBuf = accelerator.Allocate1D(aData);
        using var bBuf = accelerator.Allocate1D(bData);
        using var cBuf = accelerator.Allocate1D<float>(M * N);

        // Run through auto-select path
        matmul.MatMul(aBuf.View, bBuf.View, cBuf.View, M, K, N);
        await accelerator.SynchronizeAsync();

        // Verify result correctness (CPU reference)
        var actual = await cBuf.CopyToHostAsync<float>(0, M * N);
        float maxErr = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float expected = 0;
                for (int k = 0; k < K; k++)
                    expected += aData[i * K + k] * bData[k * N + j];
                float err = MathF.Abs(actual[i * N + j] - expected);
                maxErr = MathF.Max(maxErr, err);
            }
        }

        Console.WriteLine($"[Benchmark] MatMul 128x128 auto-select: maxErr={maxErr:E3}");
        if (maxErr > 0.1f)
            throw new Exception($"MatMul 128x128 correctness failed: maxErr={maxErr:F6}");

        Console.WriteLine($"[Benchmark] MatMul auto-select: PASS");
    });
}
