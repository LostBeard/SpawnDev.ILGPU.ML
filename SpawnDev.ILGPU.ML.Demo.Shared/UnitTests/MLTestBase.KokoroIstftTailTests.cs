using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="KokoroIstftTail"/> - the 35 graph nodes Kokoro ends with, recognised and run on the host.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY A TEST AND NOT JUST THE END-TO-END GATE. <c>Pipeline_Kokoro_MatchesOnnxRuntimeWaveform</c>
/// proves the truncated graph plus this tail equals onnxruntime, but it is <b>HeavyModel</b> - it fetches
/// ~326 MB and is excluded from every sweep by default. That makes it the wrong place for the tail's own
/// guard to live: the thing most likely to go wrong here is <see cref="KokoroIstftTail.TryDetect"/>
/// quietly matching a graph it should have declined, and a gate nobody runs catches that on the day the
/// model is re-exported rather than the day the code changed.
/// </para>
/// <para>
/// ⭐ BOTH FIXTURES ARE ONNXRUNTIME'S, so this is an oracle test and not a restatement of the
/// implementation. <c>kokoro-af_heart-paris-istft-in.f32</c> is ORT's value for the tensor the truncated
/// graph now ends at, and <c>kokoro-af_heart-paris.f32</c> is ORT's final waveform. Running our tail over
/// the first has to reproduce the second EXACTLY - not approximately: 0 of 54,600 samples may differ.
/// A tolerance here would hide precisely the reordering this class must not do (dividing and scaling in
/// one fused multiply rounds differently, and every sample would still be "close").
/// </para>
/// <para>
/// ⚠️ Runs on every backend lane on purpose even though the tail is host code, because the Wasm lane is
/// the one that matters: this executes on the .NET WASM orchestrator through <c>Vector128</c>, and
/// "bit-exact on the desktop CLR" says nothing about what the browser's SIMD does with it.
/// </para>
/// <para>
/// ⭐ PROVENANCE, so the fixtures can be regenerated rather than trusted:
/// <c>node tools/onnx-intermediates.mjs &lt;model.onnx&gt; &lt;af_heart.bin&gt; &lt;tokens-csv&gt; &lt;dir&gt;
/// /decoder/decoder/generator/istft/stft/Squeeze_1_output_0</c> - that tensor is the ConvTranspose output
/// with its size-1 channel axis dropped, so its bytes are the cut tensor's bytes.
/// <c>node tools/onnx-tail.mjs &lt;model.onnx&gt; ConvTranspose</c> prints the 35 nodes rebuilt below.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>The tensor the truncated graph ends at.</summary>
    private const string IstftCut = "/decoder/decoder/generator/istft/stft/ConvTranspose_output_0";

    /// <summary>
    /// The shipped export's <c>window_sum</c> initializer, verbatim.
    /// </summary>
    /// <remarks>
    /// ⚠️ Index 0 is an exact zero and index 10 an exact one, and both are load-bearing: the zero is
    /// excluded by the graph's <c>Greater</c> (dividing by it would be an infinity) and the one divides
    /// to itself, which is why ORT changes 18 samples here and not 20.
    /// </remarks>
    private static float[] IstftWindowSum() => new[]
    {
        0f, 0.0005988661432638764f, 0.009118626825511456f, 0.04248024895787239f, 0.11936438083648682f,
        0.25f, 0.42838138341903687f, 0.630265474319458f, 0.8181356191635132f, 0.951655387878418f,
        1f, 0.951655387878418f, 0.8181356191635132f, 0.630265474319458f, 0.42838138341903687f,
        0.25f, 0.11936438083648682f, 0.04248024895787239f, 0.009118626825511456f, 0.0005988661432638764f,
    };

    /// <summary>
    /// Kokoro's iSTFT tail as the exporter emits it, node for node.
    /// </summary>
    /// <remarks>
    /// ⚠️ REBUILT FAITHFULLY, not caricatured. A simplified stand-in would pass against a detector that
    /// matches far too loosely, which is the failure this exists to catch - so the wiring, the operand
    /// ORDER, and the constants are the real ones, and the only thing dropped is the model's 2,400 other
    /// nodes and the tensor names' shared prefix.
    /// </remarks>
    private static ModelGraph IstftTailGraph()
    {
        var graph = new ModelGraph
        {
            Name = "kokoro_istft_tail",
            Inputs = new() { new() { Name = "spec", Shape = new[] { 1, 20, 121 } } },
            Outputs = new() { new() { Name = "waveform", Shape = Array.Empty<int>() } },
            Nodes = new()
            {
                // The cut itself. Above the cut, so the detector must NOT count it as part of the tail.
                N("ConvTranspose", new[] { "spec", "conv_w" }, new[] { IstftCut }),

                N("Squeeze",   new[] { IstftCut, "c1" },                     new[] { "sq1" }),
                N("Shape",     new[] { "sq1" },                              new[] { "shp_sq1" }),
                N("Slice",     new[] { "shp_sq1", "c2", "cmax", "c0" },      new[] { "trailing_dims" }),
                N("Gather",    new[] { "shp_sq1", "c0s" },                   new[] { "batch" }),
                N("Range",     new[] { "c0s", "batch", "c1s" },              new[] { "rows" }),
                N("Reshape",   new[] { "rows", "col_shape" },                new[] { "row_idx" }),

                N("Shape",     new[] { IstftCut },                           new[] { "shp_ct" }),
                N("Slice",     new[] { "shp_ct", "cneg1", "cmax", "c0" },    new[] { "len_1d" }),
                N("Squeeze",   new[] { "len_1d", "c0" },                     new[] { "len" }),
                N("Unsqueeze", new[] { "len", "c0" },                        new[] { "len_vec" }),
                N("Slice",     new[] { "window_sum", "c0", "len_vec", "c0", "c1" }, new[] { "ws" }),

                N("Greater",   new[] { "ws", "tiny" },                       new[] { "gt" }),
                N("NonZero",   new[] { "gt" },                               new[] { "nz" }),
                N("Transpose", new[] { "nz" },                               new[] { "idx" }),
                N("Gather",    new[] { "sq1", "idx" },                       new[] { "sel" }),
                N("Gather",    new[] { "ws", "idx" },                        new[] { "sel_ws" }),
                N("Div",       new[] { "sel", "sel_ws" },                    new[] { "normalised" }),

                N("Add",       new[] { "row_idx", "idx" },                   new[] { "rows_b" }),
                N("Shape",     new[] { "rows_b" },                           new[] { "idx_shape" }),
                N("Concat",    new[] { "idx_shape", "trailing_dims" },       new[] { "upd_shape" }),
                N("Reshape",   new[] { "normalised", "upd_shape" },          new[] { "updates" }),
                N("Equal",     new[] { "idx_shape", "minus_ones" },          new[] { "eq" }),
                N("Where",     new[] { "eq", "ones", "idx_shape" },          new[] { "bcast" }),
                N("Expand",    new[] { "row_idx", "bcast" },                 new[] { "rows_e" }),
                N("Unsqueeze", new[] { "rows_e", "cneg1" },                  new[] { "rows_u" }),
                N("Expand",    new[] { "idx", "bcast" },                     new[] { "cols_e" }),
                N("Unsqueeze", new[] { "cols_e", "cneg1" },                  new[] { "cols_u" }),
                N("Concat",    new[] { "rows_u", "cols_u" },                 new[] { "scatter_idx" }),
                N("ScatterND", new[] { "sq1", "scatter_idx", "updates" },    new[] { "scattered" }),

                N("Unsqueeze", new[] { "scattered", "c1" },                  new[] { "chan" }),
                N("Mul",       new[] { "chan", "scale" },                    new[] { "scaled" }),
                N("Slice",     new[] { "scaled", "trim_front", "cmax", "c2", "c1" },  new[] { "front_trimmed" }),
                N("Slice",     new[] { "front_trimmed", "c0", "trim_back", "c2", "c1" }, new[] { "trimmed" }),
                N("Gather",    new[] { "trimmed", "c0s" },                   new[] { "mono" }),
                N("Reshape",   new[] { "mono", "out_shape" },                new[] { "waveform" }),
            },
        };

        graph.FloatConstantData = new()
        {
            ["window_sum"] = IstftWindowSum(),
            // float.Epsilon is a denormal; this is FLT_MIN, which is what the exporter wrote.
            ["tiny"] = new[] { 1.1754943508222875e-38f },
            ["scale"] = new[] { 4f },
        };
        graph.ConstantData = new()
        {
            ["c0"] = new[] { 0 }, ["c0s"] = new[] { 0 }, ["c1"] = new[] { 1 }, ["c1s"] = new[] { 1 },
            ["c2"] = new[] { 2 }, ["cneg1"] = new[] { -1 }, ["cmax"] = new[] { int.MaxValue },
            ["trim_front"] = new[] { 10 }, ["trim_back"] = new[] { -10 },
            ["col_shape"] = new[] { -1, 1 }, ["minus_ones"] = new[] { -1, -1 }, ["ones"] = new[] { 1, 1 },
            ["out_shape"] = new[] { 1, -1 },
        };
        return graph;
    }

    /// <summary>
    /// Detect the tail, run it over onnxruntime's own input, and require onnxruntime's own output back.
    /// </summary>
    [TestMethod]
    public async Task KokoroIstftTail_ReproducesOnnxRuntimeBitExactly() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        float[] input, reference;
        try
        {
            input = await F32(http, "test-refs/kokoro-af_heart-paris-istft-in.f32");
            reference = await F32(http, "test-refs/kokoro-af_heart-paris.f32");
        }
        catch (HttpRequestException ex)
        {
            // NOT UnsupportedTestException - both fixtures are committed next to the demo, so not being
            // served means the harness is wrong, and a gate that answers "unsupported" is green.
            throw new Exception("the committed iSTFT fixtures were not served: " + ex.Message);
        }

        var tail = KokoroIstftTail.TryDetect(IstftTailGraph(), out var why)
            ?? throw new Exception($"TryDetect declined the shipped tail on {BackendName}: {why}");

        // Every number below was READ OUT OF THE GRAPH by TryDetect. Asserting them pins the extraction,
        // not just the arithmetic - a detector that found the wrong Mul would still produce a plausible
        // waveform, just 4x or 1x off, and only a wrong CONSTANT would say so.
        if (tail.CutTensor != IstftCut) throw new Exception($"cut at {tail.CutTensor}, expected {IstftCut}");
        if (tail.RemovedNodes != 35) throw new Exception($"tail is {tail.RemovedNodes} nodes, expected 35");
        if (tail.Scale != 4f) throw new Exception($"scale {tail.Scale}, expected 4");
        if (tail.TrimFront != 10 || tail.TrimBack != 10)
            throw new Exception($"trim {tail.TrimFront}/{tail.TrimBack}, expected 10/10");
        if (tail.DivisorPositions != 18)
            throw new Exception($"{tail.DivisorPositions} normalised positions, expected 18 "
                + "(20 window values, minus the zero the Greater excludes and the exact one that divides to itself)");

        var got = tail.Apply(input);
        if (got.Length != reference.Length)
            throw new Exception($"produced {got.Length} samples, onnxruntime produced {reference.Length}");

        // BIT-EXACT. Not a tolerance - see the class remarks.
        var differing = 0;
        var firstBad = -1;
        double maxDiff = 0;
        for (var i = 0; i < got.Length; i++)
        {
            if (got[i].Equals(reference[i])) continue;
            if (firstBad < 0) firstBad = i;
            differing++;
            var d = Math.Abs((double)got[i] - reference[i]);
            if (d > maxDiff) maxDiff = d;
        }
        if (differing != 0)
            throw new Exception($"{differing} of {got.Length} samples differ from onnxruntime on {BackendName}, "
                + $"first at {firstBad} (ours {got[firstBad]:R} vs {reference[firstBad]:R}), max|diff| {maxDiff:R}");

        await Task.CompletedTask;
    });

    /// <summary>
    /// A tail that is not the tail this was written against must be DECLINED, never approximated.
    /// </summary>
    /// <remarks>
    /// 🔴 THIS IS THE ONE THAT MATTERS. Declining costs speed; matching a graph that computes something
    /// else costs the audio, silently, with the right sample count and a plausible waveform. Each case
    /// below is a way a re-export could realistically differ, and every one has to come back null.
    /// </remarks>
    [TestMethod]
    public async Task KokoroIstftTail_DeclinesAnythingItDoesNotRecognise() => await RunTest(async accelerator =>
    {
        var cases = new (string What, Action<ModelGraph> Break)[]
        {
            ("a node removed from the tail",
                g => g.Nodes.RemoveAt(g.Nodes.FindIndex(n => n.OpType == "Transpose"))),
            // ⚠️ INSERTED INTO THE PATH, not hung off the output. The first version of this case appended a
            // node CONSUMING `waveform`, and TryDetect accepted it - correctly: the reverse walk starts at
            // the graph output and goes backwards, so a node downstream of the output is not in the tail,
            // contributes nothing to the waveform, and is deleted as dead by the same truncation. The case
            // only tests anything once the extra node is on the path from the cut to the output.
            ("an extra node spliced into the tail",
                g =>
                {
                    var last = g.Nodes.First(n => n.Outputs.Contains("waveform"));
                    last.Outputs[0] = "pre_waveform";
                    g.Nodes.Add(N("Relu", new[] { "pre_waveform" }, new[] { "waveform" }));
                }),
            ("a second graph output",
                g => g.Outputs.Add(new GraphValueInfo { Name = "sq1", Shape = Array.Empty<int>() })),
            ("the scale is no longer a constant",
                g => g.FloatConstantData!.Remove("scale")),
            ("the window sum is no longer a constant",
                g => g.FloatConstantData!.Remove("window_sum")),
            ("the trim amount is no longer a constant",
                g => g.ConstantData!.Remove("trim_back")),
            ("no ConvTranspose to cut at",
                g => g.Nodes[g.Nodes.FindIndex(n => n.OpType == "ConvTranspose")].OpType = "Conv"),
        };

        var missed = new List<string>();
        foreach (var (what, breakIt) in cases)
        {
            var graph = IstftTailGraph();
            breakIt(graph);
            if (KokoroIstftTail.TryDetect(graph, out _) != null) missed.Add(what);
        }

        if (missed.Count > 0)
            throw new Exception($"TryDetect ACCEPTED {missed.Count} graph(s) it must decline on {BackendName}: "
                + string.Join("; ", missed));

        // And the unbroken graph still passes - otherwise the seven cases above prove nothing, because a
        // detector that declines everything would sail through them.
        if (KokoroIstftTail.TryDetect(IstftTailGraph(), out var why) == null)
            throw new Exception($"TryDetect declined the UNBROKEN tail on {BackendName}: {why} "
                + "- the decline cases above are therefore meaningless");

        await Task.CompletedTask;
    });

    private static async Task<float[]> F32(System.Net.Http.HttpClient http, string path)
    {
        var bytes = await http.GetByteArrayAsync(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
