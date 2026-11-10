using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="TopKKernels"/>' output contract, the kernels launched directly (the decoder and the index only see part of it): three rows of 20,000 values
    /// (79 chunks) in one launch - five finite values among -Infinity (they come first, best first, and every other slot reads float.MinValue / -1, the count
    /// is 5), a row of values near +10 and one near 0 (each the host's exact top-64, ties to the lower index). Then the same buffers again with the rows
    /// swapped: nothing from the first launch may survive into the second (a floor or a slot left from the previous call). Every call runs twice: with the
    /// serial chunk kernels (ChunkMaxImpl / FloorImpl / SelectImpl, a thread per chunk) and the group ones (ChunkMaxGroupImpl / FloorGroupImpl / SelectGroupImpl, 2026-11-09), each held to
    /// the exact top-64, at T = 0.7 so the chunks' softmax mass is checked too (against an f64 host sum, 1e-5 relative: the group sums pairwise).
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task TopKKernels_ExactTop64_FillsAndForgetsEachCall() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL) throw new UnsupportedTestException("the gather takes atomic slots (not WebGL)");
        const int V = 20000, R = 3, chunk = 256, K = TopKKernels.K;   // 79 chunks (at least K: the floor is the K-th maximum), the last one partial
        int nch = (V + chunk - 1) / chunk, S = 2 * K + 1 + nch;
        var rng = new Random(3);
        float[] Sparse() { var a = Enumerable.Repeat(float.NegativeInfinity, V).ToArray(); foreach (int i in new[] { 5, 700, 701, 15000, 19999 }) a[i] = (float)rng.NextDouble(); a[701] = a[700]; return a; }
        float[] Around(float c) { var a = new float[V]; for (int i = 0; i < V; i++) a[i] = c + (float)Math.Round(rng.NextDouble(), 2); return a; }   // rounded: many ties
        using var logits = accelerator.Allocate1D<float>(R * V);
        using var maxV = accelerator.Allocate1D<float>(R * nch);
        using var maxI = accelerator.Allocate1D<int>(R * nch);
        using var cand = accelerator.Allocate1D<float>(R * S);
        using var selV = accelerator.Allocate1D<float>(R * TopKKernels.SelectCap);
        using var selI = accelerator.Allocate1D<int>(R * TopKKernels.SelectCap);
        using var selN = accelerator.Allocate1D<int>(R);
        var chunkMax = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int>(TopKKernels.ChunkMaxImpl);
        var floor = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(TopKKernels.FloorImpl);
        var select = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, float, int>(TopKKernels.SelectImpl);
        var final = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(TopKKernels.FinalImpl);
        bool canGroup = accelerator.MaxNumThreadsPerGroup >= TopKKernels.GroupChunk;
        var chunkMaxG = canGroup ? accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(TopKKernels.ChunkMaxGroupImpl) : null;
        var selectG = canGroup ? accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, float>(TopKKernels.SelectGroupImpl) : null;
        var floorG = canGroup ? accelerator.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int>(TopKKernels.FloorGroupImpl) : null;
        const float T = 0.7f;
        var notes = new List<string>();
        bool ok = true;
        async Task Run(string name, float[][] rows, bool group)
        {
            name += group ? " (group)" : " (serial)";
            logits.View.CopyFromCPU(rows.SelectMany(r => r).ToArray());
            cand.MemSetToZero();                                     // a result left by the other kernels must not pass
            if (group) chunkMaxG!(new KernelConfig(R * nch, TopKKernels.GroupChunk), logits.View, maxV.View, maxI.View, V, nch);
            else chunkMax(R * nch, logits.View, maxV.View, maxI.View, V, nch, chunk);
            if (group) floorG!(new KernelConfig(TopKKernels.FloorGroups(R, nch), TopKKernels.FloorGroupThreads), maxV.View, maxI.View, cand.View, selN.View, nch, S);
            else floor(TopKKernels.FloorThreads(R, nch), maxV.View, maxI.View, cand.View, selN.View, nch, S);
            if (group) selectG!(new KernelConfig(R * nch, TopKKernels.GroupChunk), logits.View, cand.View, selN.View, selV.View, selI.View, V, nch, S, T);
            else select(R * nch, logits.View, cand.View, selN.View, selV.View, selI.View, V, nch, S, T, chunk);
            final(TopKKernels.FinalThreads(R), selV.View, selI.View, selN.View, cand.View, S);
            var o = await cand.View.CopyToHostAsync();
            for (int r = 0; r < R; r++)
            {
                // the host: finite values best first, ties to the lower index; the rest of the K slots MinValue / -1
                var want = Enumerable.Range(0, V).Where(i => !float.IsNegativeInfinity(rows[r][i])).OrderByDescending(i => rows[r][i]).ThenBy(i => i).Take(K).ToArray();
                int bad = 0; string? first = null;
                for (int j = 0; j < K; j++)
                {
                    float wv = j < want.Length ? rows[r][want[j]] : float.MinValue, wi = j < want.Length ? want[j] : -1;
                    float gv = o[r * S + j], gi = o[r * S + K + j];
                    if (gv == wv && gi == wi) continue;
                    bad++; first ??= $" slot {j}: {gv}/{gi} vs {wv}/{wi}";
                }
                // each chunk's softmax mass at T over the values within 20 T of the row's best, against f64
                double worst = 0;
                float best = rows[r].Max();
                for (int ch = 0; ch < nch; ch++)
                {
                    double m = 0;
                    for (int i = ch * chunk; i < Math.Min(V, (ch + 1) * chunk); i++) if (rows[r][i] >= best - 20f * T) m += Math.Exp((rows[r][i] - best) / (double)T);
                    double got = o[r * S + 2 * K + 1 + ch];
                    worst = Math.Max(worst, Math.Abs(got - m) / Math.Max(1e-30, m));
                }
                bool massOk = worst < 1e-5;           // a serial f32 sum of 256 terms drifts ~1e-6; one missing term moves it ~4e-3
                ok &= bad == 0 && massOk;
                notes.Add($"{name} row {r}: {(bad == 0 ? "exact" : $"{bad} slots WRONG{first}")}, mass worst {worst:E1}{(massOk ? "" : " (limit 1e-5)")}");
            }
        }
        var sparse = Sparse(); var high = Around(10f); var low = Around(0f);
        foreach (bool group in canGroup ? new[] { false, true } : new[] { false })
        {
            await Run("first call", new[] { sparse, high, low }, group);
            await Run("second call, rows swapped", new[] { low, sparse, high }, group);
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        Console.WriteLine("[TopK] " + report);
        if (!ok) throw new Exception(report);
        return report;
    });
}
