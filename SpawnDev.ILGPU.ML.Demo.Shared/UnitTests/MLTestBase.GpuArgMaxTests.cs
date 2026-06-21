using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// GpuArgMax (the per-token greedy argmax that reads back ONE int instead of the ~1 MB vocab row) must
    /// return the SAME index as the CPU path (TextGenerationSampler.Greedy) for every size — including sizes
    /// below the partial count, non-multiples, and the real vocab size — on all 6 backends.
    /// </summary>
    [TestMethod]
    public Task GpuArgMax_MatchesCpuGreedy_VariousSizes() => RunTest(async accelerator =>
    {
        var rng = new Random(1234);
        // 1 (degenerate), < numPartials (1024), non-multiple, exactly numPartials, and a real vocab size.
        int[] sizes = { 1, 7, 64, 1000, 1024, 4096, 50257, 262144 };
        using var argmax = new GpuArgMax(accelerator);
        foreach (int n in sizes)
        {
            var data = new float[n];
            for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 20 - 10);
            // Also exercise the max landing at the very first and very last position.
            if (n > 2) { data[0] = 50f + (float)rng.NextDouble(); }
            using var buf = accelerator.Allocate1D(data);
            int gpu = await argmax.ArgMaxAsync(buf.View, n);
            int cpu = TextGenerationSampler.Greedy(data);
            if (gpu != cpu)
                throw new Exception($"GpuArgMax n={n}: gpu={gpu} (val {data[gpu]:F4}) != cpu={cpu} (val {data[cpu]:F4})");

            // max at the LAST position
            var data2 = new float[n];
            for (int i = 0; i < n; i++) data2[i] = (float)(rng.NextDouble() * 20 - 10);
            data2[n - 1] = 99f;
            using var buf2 = accelerator.Allocate1D(data2);
            int gpu2 = await argmax.ArgMaxAsync(buf2.View, n);
            if (gpu2 != n - 1) throw new Exception($"GpuArgMax n={n} (max at last): gpu={gpu2} != {n - 1}");
        }
    });

    /// <summary>
    /// On a plateau of equal maxima, GpuArgMax must return the LOWEST index — matching CPU greedy's strict
    /// first-max-wins — so greedy decode produces token-identical output to the CPU path.
    /// </summary>
    [TestMethod]
    public Task GpuArgMax_LowestIndexOnTie() => RunTest(async accelerator =>
    {
        const int n = 5000;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = 1f;
        for (int i = 2000; i < 2100; i++) data[i] = 9f; // 100-wide tie plateau at the max value
        using var buf = accelerator.Allocate1D(data);
        using var argmax = new GpuArgMax(accelerator);
        int gpu = await argmax.ArgMaxAsync(buf.View, n);
        int cpu = TextGenerationSampler.Greedy(data); // = 2000
        if (gpu != cpu || gpu != 2000)
            throw new Exception($"GpuArgMax tie: gpu={gpu} cpu={cpu} (expected lowest index 2000)");
    });
}
