using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Operators;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The <c>FusedAtan2</c> kernel and the <c>GraphOptimizer.FuseAtan2</c> pass that emits it.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY THIS EXISTS AT ALL. ONNX has no Atan2, so exporters write the quadrant logic out by hand as
/// <c>Div -&gt; Atan -&gt; Greater/Less -&gt; Add/Sub -&gt; Where -&gt; Where</c>. That chain is not atan2:
/// at <c>y == +0</c> with <c>x &lt; 0</c> its <c>y &gt; 0</c> test is false, so it returns <c>-pi</c> where
/// atan2 returns <c>+pi</c>. In a signal graph that is not a corner case - the DC and Nyquist bins of the
/// STFT of a REAL signal have an EXACTLY zero imaginary part, so a whole phase channel takes the wrong sign
/// of pi. MEASURED on Kokoro-82M, whose vocoder feeds STFT phase straight into a convolution: waveform
/// correlation against onnxruntime was 0.9503 with the chain and 0.9936 with atan2, from that one channel.
/// </para>
/// <para>
/// ⚠️ THE ZERO CASES ARE THE WHOLE POINT. A test that only samples generic angles passes against the
/// broken chain - the two functions agree everywhere except on a measure-zero set that a real signal lands
/// on constantly. The tolerance (1e-4, set by WebGL's lower-precision GLSL atan) is irrelevant to that:
/// the error being caught is 2*pi.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>The (y, x) pairs the kernel must get right, with the value atan2 is defined to give.</summary>
    /// <remarks>
    /// Includes both signed zeros for y, which is the case the exported chain cannot express, and x == 0,
    /// where the quotient the chain is built on is an infinity.
    /// </remarks>
    private static (float Y, float X, float Expected)[] Atan2Cases()
    {
        const float PI = MathF.PI;
        return new (float, float, float)[]
        {
            // The case the longhand chain gets wrong: a POSITIVE zero imaginary part with a negative real
            // part is +pi, not -pi.
            (0f, -1f, PI),
            (0f, -0.25f, PI),
            (-0f, -1f, -PI),          // and a NEGATIVE zero is -pi - the two are distinguishable
            (0f, 1f, 0f),
            (-0f, 1f, -0f),
            // x == 0: the quotient is +-infinity, so the chain reaches this through an overflow.
            (1f, 0f, PI / 2f),
            (-1f, 0f, -PI / 2f),
            (2.5f, 0f, PI / 2f),
            // Both zero - the all-zero STFT frame of a silent moment. The chain being replaced computes
            // 0/0 here, which is a NaN.
            (0f, 0f, 0f),
            (-0f, 0f, -0f),
            (0f, -0f, PI),
            (-0f, -0f, -PI),
            // Ordinary quadrants.
            (1f, 1f, PI / 4f),
            (1f, -1f, 3f * PI / 4f),
            (-1f, -1f, -3f * PI / 4f),
            (-1f, 1f, -PI / 4f),
            (0.5f, 2f, MathF.Atan2(0.5f, 2f)),
            (-3f, 0.5f, MathF.Atan2(-3f, 0.5f)),
            (1e-8f, -1e-8f, MathF.Atan2(1e-8f, -1e-8f)),
            (-1e-8f, -2f, MathF.Atan2(-1e-8f, -2f)),
        };
    }

    /// <summary>The kernel itself, against the IEEE definition, on every backend.</summary>
    [TestMethod]
    public async Task Atan2_KernelMatchesIeeeSemantics() => await RunTest(async accelerator =>
    {
        var cases = Atan2Cases();
        var y = cases.Select(c => c.Y).ToArray();
        var x = cases.Select(c => c.X).ToArray();

        using var yBuf = accelerator.Allocate1D(y);
        using var xBuf = accelerator.Allocate1D(x);
        using var oBuf = accelerator.Allocate1D<float>(cases.Length);

        var ew = new ElementWiseKernels(accelerator);
        ew.Atan2(yBuf.View, xBuf.View, oBuf.View, cases.Length);
        await accelerator.SynchronizeAsync();
        var got = await oBuf.CopyToHostAsync<float>(0, cases.Length);

        for (var i = 0; i < cases.Length; i++)
        {
            var (cy, cx, want) = cases[i];
            // 1e-4 absolute, not relative: every expected value here is O(1) radians. The tolerance is
            // set by WebGL, whose GLSL atan is a lower-precision builtin (MEASURED: atan2(1,1) off by
            // 1.1e-5) - and it is still 60,000x tighter than the 2*pi error these cases exist to catch.
            if (MathF.Abs(got[i] - want) > 1e-4f)
                throw new Exception($"atan2({cy}, {cx}) = {got[i]:R}, expected {want:R} on {BackendName}");
        }
        Console.WriteLine($"[Atan2] {cases.Length} IEEE cases match on {BackendName}");
    });

    /// <summary>
    /// The optimizer must recognise the exported chain, collapse it to one node, and - the point - produce
    /// atan2's answer where the chain would produce the wrong sign of pi.
    /// </summary>
    /// <remarks>
    /// 🔴 The assertion that would pass either way is "the output is within 1e-6 of the chain's output".
    /// This asserts the opposite: at y == +0, x &lt; 0 the fused graph must DISAGREE with the chain and
    /// agree with atan2. Disable FuseAtan2 and this test fails on those elements - that is what makes it a
    /// test rather than a description.
    /// </remarks>
    [TestMethod]
    public async Task Atan2Fusion_CollapsesChainAndFixesTheZeroCase() => await RunTest(async accelerator =>
    {
        var cases = Atan2Cases();
        int n = cases.Length;
        var pi = MathF.PI;

        var graph = new ModelGraph
        {
            Name = "atan2_fuse_test",
            Inputs = new()
            {
                new() { Name = "Y", Shape = new[] { n } },
                new() { Name = "X", Shape = new[] { n } },
            },
            Outputs = new() { new() { Name = "phase", Shape = new[] { n } } },
            Initializers = new() { ["pi_c"] = new[] { 1 }, ["zero_c"] = new[] { 1 } },
            Nodes = new()
            {
                N("Div", new[] { "Y", "X" }, new[] { "q" }),
                N("Atan", new[] { "q" }, new[] { "a" }),
                N("Greater", new[] { "Y", "zero_c" }, new[] { "ygt" }),
                N("Less", new[] { "X", "zero_c" }, new[] { "xlt" }),
                N("Add", new[] { "a", "pi_c" }, new[] { "aplus" }),
                N("Sub", new[] { "a", "pi_c" }, new[] { "aminus" }),
                N("Where", new[] { "ygt", "aplus", "aminus" }, new[] { "quad" }),
                N("Where", new[] { "xlt", "quad", "a" }, new[] { "phase" }),
            },
        };
        // pi must be readable as an EXACT float at fusion time. ConstantData is int[] and would report it
        // as 3, which is why FuseAtan2 reads FloatConstantData only.
        graph.FloatConstantData = new() { ["pi_c"] = new[] { pi }, ["zero_c"] = new[] { 0f } };
        graph.ConstantData = new() { ["pi_c"] = new[] { 3 }, ["zero_c"] = new[] { 0 } };

        var optimized = GraphOptimizer.Optimize(graph);
        int fused = optimized.Nodes.Count(nd => nd.OpType == "FusedAtan2");
        if (fused != 1) throw new Exception($"expected 1 FusedAtan2 node, got {fused} on {BackendName}");
        if (optimized.Nodes.Any(nd => nd.OpType is "Atan" or "Where" or "Greater" or "Less"))
            throw new Exception($"fusion left chain nodes behind on {BackendName}: "
                + string.Join(",", optimized.Nodes.Select(nd => nd.OpType)));
        var fa = optimized.Nodes.First(nd => nd.OpType == "FusedAtan2");
        if (fa.Inputs[0] != "Y" || fa.Inputs[1] != "X" || fa.Outputs[0] != "phase")
            throw new Exception($"FusedAtan2 wiring wrong: in=[{string.Join(",", fa.Inputs)}] "
                + $"out={fa.Outputs[0]} on {BackendName}");

        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);   // Compile re-runs Optimize internally
        using var ex = new GraphExecutor(accelerator, compiled, new Dictionary<string, Tensor>());
        using var yB = accelerator.Allocate1D(cases.Select(c => c.Y).ToArray());
        using var xB = accelerator.Allocate1D(cases.Select(c => c.X).ToArray());
        using var host = accelerator.Allocate1D<float>(n);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor>
        {
            ["Y"] = new Tensor(yB.View, new[] { n }),
            ["X"] = new Tensor(xB.View, new[] { n }),
        });
        await host.View.CopyFromAsync(outs["phase"].Data.SubView(0, n));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, n);

        var disagreements = 0;
        for (var i = 0; i < n; i++)
        {
            var (cy, cx, want) = cases[i];
            if (MathF.Abs(got[i] - want) > 1e-4f)
                throw new Exception($"fused graph: atan2({cy}, {cx}) = {got[i]:R}, expected {want:R} "
                    + $"on {BackendName}");
            // What the UNFUSED chain would have produced, so the test can prove it is actually testing
            // something: at least one case must be one the chain gets wrong.
            var chain = cx < 0f ? (cy > 0f ? MathF.Atan(cy / cx) + pi : MathF.Atan(cy / cx) - pi)
                                : MathF.Atan(cy / cx);
            if (MathF.Abs(chain - want) > 1e-4f) disagreements++;
        }
        if (disagreements == 0)
            throw new Exception("no case distinguishes atan2 from the exported chain - this test cannot fail");
        Console.WriteLine($"[Atan2Fusion] chain collapsed to 1 node; {n} cases match atan2, "
            + $"{disagreements} of which the unfused chain gets wrong, on {BackendName}");
    });

    /// <summary>
    /// Strength reduction must NOT delete <c>Add(x, epsilon)</c>.
    /// </summary>
    /// <remarks>
    /// 🔴 ROOT CAUSE THIS GUARDS. <c>ModelGraph.ConstantData</c> is <c>int[]</c>, so every small float
    /// initializer is stored there TRUNCATED - 1e-5 reads back as 0 and 1.9 reads back as 1. Strength
    /// reduction proved "this Add is + 0" from that truncated copy and rewrote the node to Identity, which
    /// silently deleted the epsilon from EVERY normalisation in the graph. MEASURED on Kokoro-82M: without
    /// the epsilon, <c>Sqrt(variance)</c> on the near-constant channels it exists to protect moved by 0.5%,
    /// the following <c>Div</c> amplified that ~1000x, and the f0 branch came out 3% wrong - enough to flip
    /// the sign of near-zero f0, which the sine generator's frac() turns into a whole cycle of phase error.
    /// Waveform correlation against onnxruntime: 0.45.
    /// </remarks>
    [TestMethod]
    public async Task StrengthReduction_KeepsAnEpsilonAdd() => await RunTest(async accelerator =>
    {
        const float eps = 1e-5f;
        const int n = 8;
        var graph = new ModelGraph
        {
            Name = "eps_add_test",
            Inputs = new() { new() { Name = "V", Shape = new[] { n } } },
            Outputs = new() { new() { Name = "out", Shape = new[] { n } } },
            // Mul by 1.9 is here for the same reason: (int)1.9 == 1 made it look like a multiply by one.
            Initializers = new() { ["eps_c"] = new[] { 1 }, ["gain_c"] = new[] { 1 } },
            Nodes = new()
            {
                N("Add", new[] { "V", "eps_c" }, new[] { "shifted" }),
                N("Mul", new[] { "shifted", "gain_c" }, new[] { "out" }),
            },
        };
        graph.FloatConstantData = new() { ["eps_c"] = new[] { eps }, ["gain_c"] = new[] { 1.9f } };
        graph.ConstantData = new() { ["eps_c"] = new[] { 0 }, ["gain_c"] = new[] { 1 } };

        var optimized = GraphOptimizer.Optimize(graph);
        if (optimized.Nodes.Any(nd => nd.OpType == "Identity"))
            throw new Exception($"strength reduction turned a real Add/Mul into Identity on {BackendName}");
        if (!optimized.Nodes.Any(nd => nd.OpType == "Add"))
            throw new Exception($"the epsilon Add was eliminated on {BackendName}");
        if (!optimized.Nodes.Any(nd => nd.OpType == "Mul"))
            throw new Exception($"the 1.9 Mul was eliminated on {BackendName}");

        // And it must still COMPUTE the epsilon - a surviving node that runs as a copy is the same bug.
        var values = new float[n];
        for (var i = 0; i < n; i++) values[i] = 0.125f * i;
        var registry = new OperatorRegistry(accelerator);
        var compiled = new GraphCompiler(registry).Compile(graph);
        // The constants must be REAL tensors: unlike a fusion test, these two nodes survive and run.
        using var epsB = accelerator.Allocate1D(new[] { eps });
        using var gainB = accelerator.Allocate1D(new[] { 1.9f });
        var weights = new Dictionary<string, Tensor>
        {
            ["eps_c"] = new Tensor(epsB.View, new[] { 1 }),
            ["gain_c"] = new Tensor(gainB.View, new[] { 1 }),
        };
        using var ex = new GraphExecutor(accelerator, compiled, weights);
        using var vB = accelerator.Allocate1D(values);
        using var host = accelerator.Allocate1D<float>(n);
        var outs = await ex.RunAsync(new Dictionary<string, Tensor>
        {
            ["V"] = new Tensor(vB.View, new[] { n }),
        });
        await host.View.CopyFromAsync(outs["out"].Data.SubView(0, n));
        await accelerator.SynchronizeAsync();
        var got = await host.CopyToHostAsync<float>(0, n);

        for (var i = 0; i < n; i++)
        {
            var want = (values[i] + eps) * 1.9f;
            if (MathF.Abs(got[i] - want) > 1e-5f)
                throw new Exception($"(v+{eps})*1.9 at {i}: got {got[i]:R}, expected {want:R} on {BackendName}");
        }
        Console.WriteLine($"[StrengthReduce] epsilon Add and 1.9 Mul both survive and compute on {BackendName}");
    });
}
