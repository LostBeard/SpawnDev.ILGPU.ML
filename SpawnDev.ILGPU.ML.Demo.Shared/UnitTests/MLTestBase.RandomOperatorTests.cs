using System.Text.Json;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="InferenceSession.RandomAtMean"/> and <see cref="InferenceSession.RandomSeed"/> (Lore, 2026-11-26: the Stable Diffusion ONNX VAE encoders sample their
    /// posterior with RandomNormalLike, so every encode differed). A graph with RandomNormalLike (mean 2, scale 3) and RandomUniformLike (-1..5): by default two runs
    /// differ; at the mean every value is the mean (2 and 2) on every run; with a seed two runs are identical, still a sample (mean ~2, std ~3), and another seed differs.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task InferenceSession_RandomAtMean_And_RandomSeed() => await RunTest(async accelerator =>
    {
        const int N = 4096;
        var graph = new ModelGraph
        {
            Name = "random-ops",
            Inputs = { new GraphValueInfo { Name = "x", Shape = new[] { 1, N } } },
            Outputs = { new GraphValueInfo { Name = "n", Shape = new[] { 1, N } }, new GraphValueInfo { Name = "u", Shape = new[] { 1, N } } },
            Nodes =
            {
                new GraphNode { OpType = "RandomNormalLike", Inputs = { "x" }, Outputs = { "n" },
                    Attributes = new() { ["mean"] = JsonSerializer.SerializeToElement(2f), ["scale"] = JsonSerializer.SerializeToElement(3f) } },
                new GraphNode { OpType = "RandomUniformLike", Inputs = { "x" }, Outputs = { "u" },
                    Attributes = new() { ["low"] = JsonSerializer.SerializeToElement(-1f), ["high"] = JsonSerializer.SerializeToElement(5f) } },
            },
        };
        using var session = InferenceSession.Create(accelerator, graph, new Dictionary<string, Tensor>());
        using var xBuf = accelerator.Allocate1D<float>(N);
        var feed = new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View, new[] { 1, N }) };
        async Task<(float[] N, float[] U)> Run()
        {
            var outs = await session.RunAsync(feed);
            var n = await outs["n"].Data.SubView(0, N).CopyToHostAsync();
            var u = await outs["u"].Data.SubView(0, N).CopyToHostAsync();
            session.ReturnOutputs(outs);
            return (n, u);
        }
        static bool Same(float[] a, float[] b) { for (int i = 0; i < a.Length; i++) if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) return false; return true; }
        static (double Mean, double Std) Stats(float[] a) { double m = a.Average(v => (double)v); return (m, Math.Sqrt(a.Average(v => (v - m) * (v - m)))); }

        // default (ONNX semantics): a fresh sample each run
        var d1 = await Run(); var d2 = await Run();
        if (Same(d1.N, d2.N) || Same(d1.U, d2.U)) throw new Exception("Without a seed two runs gave the same random values.");

        // at the mean: every value the distribution's mean, every run
        session.RandomAtMean = true;
        var m1 = await Run(); var m2 = await Run();
        session.RandomAtMean = false;
        if (m1.N.Any(v => v != 2f) || m1.U.Any(v => v != 2f)) throw new Exception($"RandomAtMean: normal {m1.N[0]}, uniform {m1.U[0]}; expected 2 and 2 everywhere.");
        if (!Same(m1.N, m2.N) || !Same(m1.U, m2.U)) throw new Exception("RandomAtMean: two runs differ.");

        // a seed: the same values every run, still a sample of the right distribution, a different seed differs, the two nodes differ from each other
        session.RandomSeed = 7;
        var s1 = await Run(); var s2 = await Run();
        session.RandomSeed = 8;
        var t1 = await Run();
        session.RandomSeed = null;
        if (!Same(s1.N, s2.N) || !Same(s1.U, s2.U)) throw new Exception("RandomSeed 7: two runs differ.");
        if (Same(s1.N, t1.N) || Same(s1.U, t1.U)) throw new Exception("RandomSeed 7 and 8 gave the same values.");
        var (nm, ns) = Stats(s1.N); var (um, us) = Stats(s1.U);
        if (Math.Abs(nm - 2) > 0.3 || Math.Abs(ns - 3) > 0.3) throw new Exception($"RandomSeed: the normal sample has mean {nm:F3}, std {ns:F3}; expected ~2, ~3.");
        if (Math.Abs(um - 2) > 0.3 || s1.U.Any(v => v < -1f || v > 5f)) throw new Exception($"RandomSeed: the uniform sample has mean {um:F3} or a value outside -1..5.");
        Console.WriteLine($"[RandomOps] default runs differ; at the mean 2 / 2 everywhere; seed 7 repeats (normal mean {nm:F3} std {ns:F3}, uniform mean {um:F3} std {us:F3}), seed 8 differs");
    });
}
