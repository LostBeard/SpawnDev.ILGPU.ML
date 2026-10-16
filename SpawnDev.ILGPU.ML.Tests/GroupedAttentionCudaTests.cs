using System.Diagnostics;
using SpawnDev.ILGPU.ML.Kernels;
using Xunit.Abstractions;

namespace SpawnDev.ILGPU.ML.Tests;

/// <summary>
/// The grouped-per-query attention against the register per-query kernel, on CUDA, with the KV cache in bfloat16 as a GGUF decode store keeps it: equal outputs, and the time of each at the
/// shapes of a chat turn (a one-token decode step, a short suffix, a long prefill). It pins the decision in FusedAttentionKernel.UseGrouped: the grouped kernel serves a few queries and
/// loses to the register kernel on many (39 ms against 18 per layer at 300 queries over 3,165 keys). Skipped where there is no CUDA device.
/// </summary>
[Collection("GroupedAttentionFlag")]
public class GroupedAttentionCudaTests(ITestOutputHelper output)
{
    private static float[] Random(int n, int seed, float scale)
    {
        var r = new System.Random(seed);
        var a = new float[n];
        for (var i = 0; i < n; i++) a[i] = ((float)r.NextDouble() * 2f - 1f) * scale;
        return a;
    }

    private static (float[] Out, double Ms) Run(Accelerator acc, bool grouped, float[] q, BFloat16[] k, BFloat16[] v, int nHeads, int kvHeads, int seqQ, int seqKV, int maxSeq, int kvOffset, int repeats, bool split = false, bool seqMajorKV = false, bool seqMajorQ = false, bool seqMajorOut = false)
    {
        const int D = 128;
        FusedAttentionKernel.EnableGroupedAttention = grouped;
        FusedAttentionKernel.EnableSplitKvDecode = split;
        var kernel = new FusedAttentionKernel(acc);
        using var qb = acc.Allocate1D(q);
        using var kb = acc.Allocate1D(k);
        using var vb = acc.Allocate1D(v);
        using var ob = acc.Allocate1D<float>(nHeads * seqQ * D);
        void Once() => kernel.ForwardStrided<BFloat16>(qb.View, kb.View, vb.View, ob.View, nHeads, kvHeads, seqQ, seqKV, D, causal: true, int.MaxValue / 2, kvOffset, 0f, maxSeq * D, seqMajorOut: seqMajorOut, seqMajorQ: seqMajorQ, seqMajorKV: seqMajorKV);
        Once();
        acc.Synchronize();
        var times = new List<double>();
        for (var i = 0; i < repeats; i++)
        {
            var sw = Stopwatch.StartNew();
            Once();
            acc.Synchronize();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        return (ob.GetAsArray1D(), times[times.Count / 2]);
    }

    [Theory]
    [InlineData(1, 3165, 4096, true)]  // one decode step over a 3,165-token context, in the layout the graph really uses: K/V store, Q and output seq-major
    [InlineData(1, 3165, 4096)]       // one decode step over a 3,165-token context
    [InlineData(1, 4500, 4608)]       // and over 4,500 (past the grouped kernel's single-pass limit)
    [InlineData(4, 3165, 4096)]       // a few queries
    [InlineData(150, 3165, 4096)]     // a short suffix after a cached prompt
    [InlineData(300, 3165, 4096)]     // a long one
    public void GroupedAndRegisterAgree_AndTheTimesAreShown(int seqQ, int seqKV, int maxSeq, bool seqMajor = false)
    {
        var context = MLContext.CreateContext();
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        if (device == null) { output.WriteLine("no CUDA device: skipped"); context.Dispose(); return; }
        using var acc = device.CreateAccelerator(context);
        const int nHeads = 32, kvHeads = 8, D = 128;
        var q = Random(nHeads * seqQ * D, 1, 0.8f);
        var k = Random(kvHeads * maxSeq * D, 2, 0.8f).Select(x => (BFloat16)x).ToArray();
        var v = Random(kvHeads * maxSeq * D, 3, 1f).Select(x => (BFloat16)x).ToArray();
        var kvOffset = seqKV - seqQ;
        var reference = Run(acc, false, q, k, v, nHeads, kvHeads, seqQ, seqKV, maxSeq, kvOffset, 5, false, seqMajor, seqMajor, seqMajor);
        var grouped = Run(acc, true, q, k, v, nHeads, kvHeads, seqQ, seqKV, maxSeq, kvOffset, 5, false, seqMajor, seqMajor, seqMajor);
        var splitKv = Run(acc, false, q, k, v, nHeads, kvHeads, seqQ, seqKV, maxSeq, kvOffset, 5, true, seqMajor, seqMajor, seqMajor);
        FusedAttentionKernel.EnableGroupedAttention = false;
        FusedAttentionKernel.EnableSplitKvDecode = true;
        var worst = 0f;
        for (var i = 0; i < reference.Out.Length; i++) worst = MathF.Max(worst, MathF.Abs(reference.Out[i] - grouped.Out[i]));
        var worstSplit = 0f;
        for (var i = 0; i < reference.Out.Length; i++) worstSplit = MathF.Max(worstSplit, MathF.Abs(reference.Out[i] - splitKv.Out[i]));
        output.WriteLine($"q{seqQ} over {seqKV} keys: register {reference.Ms:0.00} ms, grouped {grouped.Ms:0.00} ms, split-KV {splitKv.Ms:0.00} ms; max |diff| grouped {worst:E2}, split-KV {worstSplit:E2}");
        Assert.True(worst < 2e-4f, $"grouped max |diff| {worst:E3}");
        Assert.True(worstSplit < 2e-4f, $"split-KV max |diff| {worstSplit:E3}");
        context.Dispose();
    }

    /// <summary>Split-KV decode at every context length from 512 to 4,700 in steps (chunk boundaries and ragged last chunks), seq-major layouts, one kernel object reused for all of them like a decode loop.</summary>
    [Fact]
    public void SplitKvDecode_AtManyContextLengths_MatchesTheRegisterKernel()
    {
        var context = MLContext.CreateContext();
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        if (device == null) { output.WriteLine("no CUDA device: skipped"); context.Dispose(); return; }
        using var acc = device.CreateAccelerator(context);
        const int nHeads = 32, kvHeads = 8, D = 128, maxSeq = 4800;
        var q = Random(nHeads * D, 1, 0.8f);
        var k = Random(kvHeads * maxSeq * D, 2, 0.8f).Select(x => (BFloat16)x).ToArray();
        var v = Random(kvHeads * maxSeq * D, 3, 1f).Select(x => (BFloat16)x).ToArray();
        using var qb = acc.Allocate1D(q);
        using var kb = acc.Allocate1D(k);
        using var vb = acc.Allocate1D(v);
        using var obRef = acc.Allocate1D<float>(nHeads * D);
        using var obSplit = acc.Allocate1D<float>(nHeads * D);
        var worstAll = 0f;
        var kernel = new FusedAttentionKernel(acc);
        for (var seqKV = 512; seqKV <= 4700; seqKV += 61)
        {
            FusedAttentionKernel.EnableGroupedAttention = false;
            FusedAttentionKernel.EnableSplitKvDecode = false;
            kernel.ForwardStrided<BFloat16>(qb.View, kb.View, vb.View, obRef.View, nHeads, kvHeads, 1, seqKV, D, causal: true, int.MaxValue / 2, seqKV - 1, 0f, maxSeq * D, seqMajorOut: true, seqMajorQ: true, seqMajorKV: true);
            FusedAttentionKernel.EnableSplitKvDecode = true;
            kernel.ForwardStrided<BFloat16>(qb.View, kb.View, vb.View, obSplit.View, nHeads, kvHeads, 1, seqKV, D, causal: true, int.MaxValue / 2, seqKV - 1, 0f, maxSeq * D, seqMajorOut: true, seqMajorQ: true, seqMajorKV: true);
            acc.Synchronize();
            var a = obRef.GetAsArray1D();
            var b = obSplit.GetAsArray1D();
            var worst = 0f;
            for (var i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
            worstAll = MathF.Max(worstAll, worst);
            Assert.True(worst < 2e-4f, $"context {seqKV}: max |diff| {worst:E3}");
        }
        output.WriteLine($"worst difference over all lengths: {worstAll:E2}");
        context.Dispose();
    }

    /// <summary>
    /// A decode loop's worth of split-KV attention calls issued back to back WITHOUT a sync (32 layers x many tokens, a growing context, different K/V per layer, one kernel object): the
    /// outputs must equal the same calls made with a sync after each. A real chat run failed once in three with the kernel on and never with a sync after every op, so this looks for what
    /// only shows when launches overlap the host: scratch reuse between layers, parameter slots rewritten before an earlier launch has read them.
    /// </summary>
    [Fact]
    public void SplitKvDecode_BackToBackWithoutSync_MatchesTheSyncedRun()
    {
        var context = MLContext.CreateContext();
        var device = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        if (device == null) { output.WriteLine("no CUDA device: skipped"); context.Dispose(); return; }
        using var acc = device.CreateAccelerator(context);
        const int nHeads = 32, kvHeads = 8, D = 128, maxSeq = 3600, layers = 32, tokens = 120;
        // Different K/V per layer would be 32 x 7 MB; use 4 distinct layers' data cycled, which still changes the buffers the kernel reads from one launch to the next.
        var kLayers = new List<MemoryBuffer1D<BFloat16, Stride1D.Dense>>();
        var vLayers = new List<MemoryBuffer1D<BFloat16, Stride1D.Dense>>();
        for (var i = 0; i < 4; i++)
        {
            kLayers.Add(acc.Allocate1D(Random(kvHeads * maxSeq * D, 10 + i, 0.8f).Select(x => (BFloat16)x).ToArray()));
            vLayers.Add(acc.Allocate1D(Random(kvHeads * maxSeq * D, 20 + i, 1f).Select(x => (BFloat16)x).ToArray()));
        }
        using var qb = acc.Allocate1D(Random(nHeads * D, 1, 0.8f));
        var perCall = nHeads * D;
        using var outRef = acc.Allocate1D<float>(tokens * layers * perCall);
        using var outAsync = acc.Allocate1D<float>(tokens * layers * perCall);
        var kernel = new FusedAttentionKernel(acc);
        FusedAttentionKernel.EnableGroupedAttention = false;

        void Pass(bool split, bool sync, MemoryBuffer1D<float, Stride1D.Dense> dest)
        {
            FusedAttentionKernel.EnableSplitKvDecode = split;
            for (var t = 0; t < tokens; t++)
                for (var l = 0; l < layers; l++)
                {
                    var seqKV = 2400 + t * 9;
                    kernel.ForwardStrided<BFloat16>(qb.View, kLayers[l % 4].View, vLayers[l % 4].View, dest.View.SubView((t * layers + l) * perCall, perCall), nHeads, kvHeads, 1, seqKV, D,
                        causal: true, int.MaxValue / 2, seqKV - 1, 0f, maxSeq * D, seqMajorOut: true, seqMajorQ: true, seqMajorKV: true);
                    if (sync) acc.Synchronize();
                }
            acc.Synchronize();
        }
        var sw = Stopwatch.StartNew();
        Pass(split: false, sync: true, outRef);
        var refMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        Pass(split: true, sync: false, outAsync);
        var asyncMs = sw.Elapsed.TotalMilliseconds;
        FusedAttentionKernel.EnableSplitKvDecode = true;
        var a = outRef.GetAsArray1D();
        var b = outAsync.GetAsArray1D();
        var worst = 0f;
        var worstAt = -1;
        for (var i = 0; i < a.Length; i++)
        {
            var d = MathF.Abs(a[i] - b[i]);
            if (!(d <= worst)) { worst = d; worstAt = i; }     // NaN-safe
        }
        output.WriteLine($"{tokens * layers} calls: register synced {refMs:0} ms, split-KV unsynced {asyncMs:0} ms; worst difference {worst:E2} at call {(worstAt < 0 ? -1 : worstAt / perCall)}");
        foreach (var kb in kLayers) kb.Dispose();
        foreach (var vb in vLayers) vb.Dispose();
        Assert.True(worst < 2e-4f, $"worst difference {worst:E3} at call {worstAt / perCall}");
        context.Dispose();
    }
}

[CollectionDefinition("GroupedAttentionFlag", DisableParallelization = true)]
public class GroupedAttentionFlagCollection { }
