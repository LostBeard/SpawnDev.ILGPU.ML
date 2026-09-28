using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.UnitTesting;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Node-scoped scratch: an UNNAMED <see cref="BufferPool.Rent"/> made while a graph node executes is handed back
/// after the node (on RunAsync's drain schedule, immediately on sync Run). Before, Return keyed on the name, so
/// every unnamed temp stayed allocated until the pool was disposed - DAv3 leaked 16 buffers per warm forward
/// through ConcatOperator's de-alias copy alone (2026-09-28). The heavy gate is
/// <c>DA3_WarmForward_AllocatesNothing</c>; these are the small graphs that run on every backend.
/// </summary>
public abstract partial class MLTestBase
{
    // [2, 300] -> one 600-element de-alias copy per Concat(x, x): bucket 1024, distinct from every other rent.
    const int ScratchRows = 2, ScratchCols = 300;

    [TestMethod]
    public async Task PoolScratch_ConcatSelf_WarmRunsAllocateNothing() => await RunTest(async accelerator =>
    {
        // y = Relu(Concat(x, x, axis=1)). The same tensor twice is what makes Concat take its de-alias copy.
        var graph = new ModelGraph
        {
            Name = "concat-self",
            Inputs = { new GraphValueInfo { Name = "x", Shape = new[] { ScratchRows, ScratchCols } } },
            Outputs = { new GraphValueInfo { Name = "y", Shape = new[] { ScratchRows, 2 * ScratchCols } } },
            Nodes =
            {
                new GraphNode { OpType = "Concat", Inputs = { "x", "x" }, Outputs = { "c" },
                    Attributes = new() { ["axis"] = JsonSerializer.SerializeToElement(1) } },
                new GraphNode { OpType = "Relu", Inputs = { "c" }, Outputs = { "y" } },
            },
        };
        var x = RandomFloats(ScratchRows * ScratchCols, seed: 41);
        var expected = new float[ScratchRows * 2 * ScratchCols];
        for (int r = 0; r < ScratchRows; r++)
            for (int c = 0; c < 2 * ScratchCols; c++)
                expected[r * 2 * ScratchCols + c] = MathF.Max(0f, x[r * ScratchCols + c % ScratchCols]);

        using var session = InferenceSession.Create(accelerator, graph, new Dictionary<string, Tensor>());
        using var xBuf = accelerator.Allocate1D(x);
        var feed = new Dictionary<string, Tensor> { ["x"] = new Tensor(xBuf.View, new[] { ScratchRows, ScratchCols }) };

        async Task Pass(bool sync, string tag)
        {
            var outs = sync ? session.Run(feed) : await session.RunAsync(feed);
            await accelerator.SynchronizeAsync();
            var y = outs["y"];
            AssertClose(expected, await y.Data.SubView(0, y.ElementCount).CopyToHostAsync(), 0f, $"{tag}: ");
            session.ReturnOutputs(outs);
        }

        foreach (bool sync in new[] { false, true })
        {
            string path = sync ? "Run" : "RunAsync";
            await Pass(sync, $"{path} warm 1");
            await Pass(sync, $"{path} warm 2");
            int settled = session.Executor.AllocatedBufferCount;
            for (int i = 0; i < 6; i++)
            {
                await Pass(sync, $"{path} pass {i + 3}");
                int now = session.Executor.AllocatedBufferCount;
                if (now != settled)
                    throw new Exception($"{path}: pool grew {settled} -> {now} buffers by pass {i + 3} - the Concat de-alias " +
                                        "scratch is not being returned");
            }
        }
    });

    [TestMethod]
    public async Task PoolScratch_EscapedSequencePieceIsNotRecycled() => await RunTest(async accelerator =>
    {
        // SplitToSequence's pieces are unnamed Rents that OUTLIVE their node inside a host Sequence. The two
        // Concats after it each take an unnamed de-alias copy of exactly the piece's size bucket, and the drain
        // runs after every node. If the piece were handed back as scratch, sync Run would recycle it into the
        // first Concat's copy at once; RunAsync defers it to the drain after that Concat, so the SECOND Concat
        // is the one that gets it. Either way SequenceAt would then read z or w instead of x.
        // ⚠️ The second Concat is load-bearing: with only one, RunAsync's freed piece is next rented as
        // SequenceAt's own output, which copies the piece onto itself and still reads x (red-checked).
        int n = 256;
        var graph = new ModelGraph
        {
            Name = "escaped-scratch",
            Inputs =
            {
                new GraphValueInfo { Name = "x", Shape = new[] { 1, n } },
                new GraphValueInfo { Name = "z", Shape = new[] { n } },
                new GraphValueInfo { Name = "w", Shape = new[] { n } },
            },
            Outputs =
            {
                new GraphValueInfo { Name = "piece", Shape = new[] { n } },
                new GraphValueInfo { Name = "zz", Shape = new[] { 2 * n } },
                new GraphValueInfo { Name = "ww", Shape = new[] { 2 * n } },
            },
            Nodes =
            {
                new GraphNode { OpType = "SplitToSequence", Inputs = { "x" }, Outputs = { "seq" },
                    Attributes = new() { ["axis"] = JsonSerializer.SerializeToElement(0), ["keepdims"] = JsonSerializer.SerializeToElement(0) } },
                new GraphNode { OpType = "Concat", Inputs = { "z", "z" }, Outputs = { "zz" },
                    Attributes = new() { ["axis"] = JsonSerializer.SerializeToElement(0) } },
                new GraphNode { OpType = "Concat", Inputs = { "w", "w" }, Outputs = { "ww" },
                    Attributes = new() { ["axis"] = JsonSerializer.SerializeToElement(0) } },
                new GraphNode { OpType = "SequenceAt", Inputs = { "seq" }, Outputs = { "piece" } },
            },
        };
        var x = RandomFloats(n, seed: 5);
        var z = RandomFloats(n, seed: 6);
        var w = RandomFloats(n, seed: 7);
        var expectedZZ = z.Concat(z).ToArray();
        var expectedWW = w.Concat(w).ToArray();

        using var session = InferenceSession.Create(accelerator, graph, new Dictionary<string, Tensor>());
        session.Executor.SyncIntervalNodesOverride = 1;
        using var xBuf = accelerator.Allocate1D(x);
        using var zBuf = accelerator.Allocate1D(z);
        using var wBuf = accelerator.Allocate1D(w);
        var feed = new Dictionary<string, Tensor>
        {
            ["x"] = new Tensor(xBuf.View, new[] { 1, n }),
            ["z"] = new Tensor(zBuf.View, new[] { n }),
            ["w"] = new Tensor(wBuf.View, new[] { n }),
        };

        foreach (bool sync in new[] { false, true })
            for (int i = 0; i < 3; i++)
            {
                string tag = $"{(sync ? "Run" : "RunAsync")} pass {i + 1}";
                var outs = sync ? session.Run(feed) : await session.RunAsync(feed);
                await accelerator.SynchronizeAsync();
                var piece = outs["piece"];
                AssertClose(x, await piece.Data.SubView(0, n).CopyToHostAsync(), 0f, $"{tag} piece: ");
                var zz = outs["zz"];
                AssertClose(expectedZZ, await zz.Data.SubView(0, 2 * n).CopyToHostAsync(), 0f, $"{tag} zz: ");
                var ww = outs["ww"];
                AssertClose(expectedWW, await ww.Data.SubView(0, 2 * n).CopyToHostAsync(), 0f, $"{tag} ww: ");
                session.ReturnOutputs(outs);
            }
    });
}
