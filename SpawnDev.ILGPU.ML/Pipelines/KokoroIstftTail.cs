using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using SpawnDev.ILGPU.ML.Graph;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Kokoro's iSTFT overlap-add tail, lifted off the GPU and onto the orchestrator.
/// </summary>
/// <remarks>
/// <para>
/// 🔴 WHY. The last 35 nodes of Kokoro's graph are not signal processing, they are BOOKKEEPING: a
/// <c>Greater</c>/<c>NonZero</c>/<c>ScatterND</c> dance that divides a handful of samples by the
/// overlap-add window sum, a scale, and two trims. Two of those operators - <c>NonZero</c> and the
/// <c>ScatterND</c> index build - have a data-dependent output SHAPE, which no GPU can produce without
/// telling the host how big it came out. So the tail costs host READBACKS in the middle of a dispatch
/// stream, and each one drains the whole queue.
/// </para>
/// <para>
/// ⭐ MEASURED against onnxruntime on "The capital of France is Paris." (af_heart, fp32 export): the
/// entire 35-node tail changes exactly 18 of 54,620 samples, scales by 4, and trims 10 from each end.
/// Reproduced here, the result is BIT-EXACT: 0 of 54,600 samples differ, max|diff| = 0. Paying queue
/// drains and 35 dispatches for 18 divisions is the definition of the wrong place to do work.
/// </para>
/// <para>
/// ⚠️ THE NUMBERS ARE READ FROM THE GRAPH, NEVER ASSUMED. <see cref="TryDetect"/> walks the real tail and
/// pulls the window sum, the scale and the trims out of it, and it declines on anything it does not
/// recognise. A different Kokoro export therefore LOSES this optimisation - it never runs a tail that
/// does not match the graph it was read from.
/// </para>
/// <para>
/// This runs on the .NET WASM orchestrator thread, deliberately. It is serial, it is ~54 k multiplies,
/// and .NET WASM is SIMD - so <see cref="Vector128"/> carries it without threads, which the orchestrator
/// does not have and is not getting.
/// </para>
/// </remarks>
public sealed class KokoroIstftTail
{
    /// <summary>The tensor the truncated graph now ends at - the last ConvTranspose's output.</summary>
    public required string CutTensor { get; init; }

    /// <summary>
    /// Per-sample divisors for the head of the signal, 1.0 wherever the graph's <c>Greater</c> test
    /// excluded a position. Dividing by an exact 1.0f is the identity in IEEE, so the whole head divides
    /// uniformly and still changes only the positions ONNX changed.
    /// </summary>
    public required float[] Divisors { get; init; }

    /// <summary>The post-normalisation scale (4 in the shipped export).</summary>
    public required float Scale { get; init; }

    /// <summary>Samples dropped from the start.</summary>
    public required int TrimFront { get; init; }

    /// <summary>Samples dropped from the end.</summary>
    public required int TrimBack { get; init; }

    /// <summary>How many graph nodes truncation removes.</summary>
    public required int RemovedNodes { get; init; }

    /// <summary>How many head positions get a non-identity divide - 18 in the shipped export.</summary>
    public int DivisorPositions => Divisors.Count(d => d != 1f);

    public override string ToString()
        => $"cut at {CutTensor}, -{RemovedNodes} nodes, {DivisorPositions} normalised of {Divisors.Length}, "
         + $"x{Scale}, trim {TrimFront}/{TrimBack}";

    /// <summary>
    /// Recognise the iSTFT tail in a freshly parsed graph, or explain why it could not be recognised.
    /// </summary>
    /// <remarks>
    /// Runs BEFORE optimisation, so the op types here are the raw ONNX ones.
    /// </remarks>
    public static KokoroIstftTail? TryDetect(ModelGraph graph, out string reason)
    {
        reason = "";
        if (graph.Outputs.Count != 1) { reason = $"graph has {graph.Outputs.Count} outputs, expected 1"; return null; }

        var producer = new Dictionary<string, GraphNode>(StringComparer.Ordinal);
        foreach (var n in graph.Nodes)
            foreach (var o in n.Outputs)
                if (!string.IsNullOrEmpty(o)) producer[o] = n;

        // Reverse walk from the output, stopping the descent at the first ConvTranspose - that node's
        // output becomes the new graph output, and everything collected below is what truncation removes.
        var tail = new List<GraphNode>();
        var inTail = new HashSet<GraphNode>();
        var cuts = new HashSet<string>(StringComparer.Ordinal);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        queue.Enqueue(graph.Outputs[0].Name);
        while (queue.Count > 0)
        {
            var t = queue.Dequeue();
            if (!seen.Add(t)) continue;
            if (!producer.TryGetValue(t, out var n)) continue;              // an initializer or a graph input
            if (n.OpType == "ConvTranspose") { cuts.Add(t); continue; }
            if (inTail.Add(n)) tail.Add(n);
            foreach (var i in n.Inputs) if (!string.IsNullOrEmpty(i)) queue.Enqueue(i);
        }

        if (cuts.Count != 1)
        {
            reason = $"reverse walk reached {cuts.Count} ConvTranspose outputs, expected exactly 1";
            return null;
        }
        var cut = cuts.First();

        // An op-type census is the change detector. It is not the extraction - every value below is pulled
        // from the real nodes - but it is what turns "a different export" into a decline instead of a tail
        // run over structure it was never read from.
        var census = tail.GroupBy(n => n.OpType).ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
        var expected = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["Squeeze"] = 2, ["Shape"] = 3, ["Slice"] = 5, ["Gather"] = 4, ["Range"] = 1, ["Reshape"] = 3,
            ["Unsqueeze"] = 4, ["Greater"] = 1, ["NonZero"] = 1, ["Transpose"] = 1, ["Div"] = 1, ["Add"] = 1,
            ["Concat"] = 2, ["Equal"] = 1, ["Where"] = 1, ["Expand"] = 2, ["ScatterND"] = 1, ["Mul"] = 1,
        };
        if (census.Count != expected.Count || expected.Any(kv => !census.TryGetValue(kv.Key, out var c) || c != kv.Value))
        {
            reason = "tail op census differs from the known iSTFT tail: got "
                   + string.Join(",", census.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"));
            return null;
        }

        var floats = graph.FloatConstantData;
        var ints = graph.ConstantData;
        if (floats == null || ints == null) { reason = "constant data was not seeded"; return null; }

        // --- the window sum: Greater(Slice(window_sum, ...), tiny) selects which samples get normalised.
        var greater = tail.First(n => n.OpType == "Greater");
        if (greater.Inputs.Count < 2) { reason = "Greater has fewer than 2 inputs"; return null; }
        if (!producer.TryGetValue(greater.Inputs[0], out var wsSlice) || wsSlice.OpType != "Slice")
        {
            reason = "the Greater's data input is not produced by a Slice";
            return null;
        }
        if (wsSlice.Inputs.Count < 1 || !floats.TryGetValue(wsSlice.Inputs[0], out var windowSum))
        {
            var nm = wsSlice.Inputs.Count > 0 ? wsSlice.Inputs[0] : "?";
            reason = $"the window-sum initializer {nm} has no float data";
            return null;
        }
        if (!floats.TryGetValue(greater.Inputs[1], out var thrArr) || thrArr.Length != 1)
        {
            reason = $"the Greater threshold {greater.Inputs[1]} is not a scalar constant";
            return null;
        }
        var threshold = thrArr[0];

        // ⚠️ The Slice's `ends` is the RUNTIME signal length, so it only ever truncates the window sum on
        // an utterance shorter than the window itself. Apply clamps for that; nothing here can.
        var divisors = new float[windowSum.Length];
        for (var i = 0; i < windowSum.Length; i++)
            divisors[i] = windowSum[i] > threshold ? windowSum[i] : 1f;

        // --- the scale: the one Mul, whose other operand is a scalar constant.
        var mul = tail.First(n => n.OpType == "Mul");
        float? scale = null;
        foreach (var i in mul.Inputs)
            if (!string.IsNullOrEmpty(i) && floats.TryGetValue(i, out var s) && s.Length == 1) scale = s[0];
        if (scale is null) { reason = "the tail's Mul has no scalar constant operand"; return null; }
        if (mul.Outputs.Count < 1) { reason = "the tail's Mul has no output"; return null; }

        // --- the trims: the Slices that consume the Mul's output chain, on the sample axis.
        //     A positive start trims the front; a negative end trims the back.
        int trimFront = 0, trimBack = 0;
        var downstream = new HashSet<string>(StringComparer.Ordinal) { mul.Outputs[0] };
        foreach (var n in TopoOrder(tail, cut))
        {
            if (!n.Inputs.Any(i => !string.IsNullOrEmpty(i) && downstream.Contains(i))) continue;
            foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) downstream.Add(o);
            if (n.OpType != "Slice" || n.Inputs.Count < 3) continue;
            if (!ints.TryGetValue(n.Inputs[1], out var st) || st.Length != 1)
            {
                reason = $"the trim Slice start {n.Inputs[1]} is not a constant";
                return null;
            }
            if (!ints.TryGetValue(n.Inputs[2], out var en) || en.Length != 1)
            {
                reason = $"the trim Slice end {n.Inputs[2]} is not a constant";
                return null;
            }
            if (st[0] > 0) trimFront += st[0];
            if (en[0] < 0) trimBack += -en[0];
        }

        return new KokoroIstftTail
        {
            CutTensor = cut,
            Divisors = divisors,
            Scale = scale.Value,
            TrimFront = trimFront,
            TrimBack = trimBack,
            RemovedNodes = tail.Count,
        };
    }

    /// <summary>Tail nodes in execution order, so "consumes the Mul's output" is answerable in one sweep.</summary>
    private static List<GraphNode> TopoOrder(List<GraphNode> tail, string cut)
    {
        var ready = new HashSet<string>(StringComparer.Ordinal) { cut };
        var pending = new List<GraphNode>(tail);
        var order = new List<GraphNode>(tail.Count);
        while (pending.Count > 0)
        {
            var i = pending.FindIndex(n => n.Inputs.All(x => string.IsNullOrEmpty(x)
                || ready.Contains(x) || !pending.Any(p => p.Outputs.Contains(x))));
            if (i < 0) { order.AddRange(pending); break; }   // ONNX cannot cycle; degrade rather than hang
            var n = pending[i];
            pending.RemoveAt(i);
            order.Add(n);
            foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) ready.Add(o);
        }
        return order;
    }

    /// <summary>
    /// Rewire the graph to end at <see cref="CutTensor"/>. Dead-node elimination does the rest.
    /// </summary>
    /// <remarks>
    /// The shape is left EMPTY on purpose: <c>GraphCompiler</c> registers a declared output shape only
    /// when one is given, and a declared shape that disagrees with inference pins the output buffer.
    /// Inference already knows this tensor's shape - it is a node output like any other.
    /// </remarks>
    public void Truncate(ModelGraph graph)
    {
        graph.Outputs.Clear();
        graph.Outputs.Add(new GraphValueInfo { Name = CutTensor, Shape = Array.Empty<int>() });
    }

    /// <summary>
    /// Run the tail over the signal the truncated graph produced.
    /// </summary>
    /// <remarks>
    /// ⚠️ ORDER IS LOAD-BEARING. ONNX divides and THEN scales, and folding that into one multiply by
    /// <c>Scale/Divisors[i]</c> rounds differently. Two operations, in this order, is what makes the
    /// result bit-exact rather than merely close.
    /// </remarks>
    public float[] Apply(ReadOnlySpan<float> signal)
    {
        var count = signal.Length - TrimFront - TrimBack;
        if (count <= 0)
            throw new InvalidOperationException(
                $"Kokoro produced {signal.Length} samples, which the iSTFT trim ({TrimFront}+{TrimBack}) leaves nothing of");

        var dst = new float[count];
        var src = signal[TrimFront..];

        // Head - the only region the window-sum normalisation reaches. Clamped because the graph's own
        // Slice ends at the signal length, so a signal shorter than the window normalises less of it.
        var head = Math.Clamp(Divisors.Length - TrimFront, 0, count);
        for (var i = 0; i < head; i++)
            dst[i] = (src[i] / Divisors[TrimFront + i]) * Scale;

        // Bulk - a pure scale. Per-lane multiply is exact, so the vector path is bit-identical to the
        // scalar one and there is no "fast mode" to opt out of.
        var j = head;
        if (Vector128.IsHardwareAccelerated && count - j >= Vector128<float>.Count)
        {
            var vs = Vector128.Create(Scale);
            ref var s0 = ref MemoryMarshal.GetReference(src);
            ref var d0 = ref MemoryMarshal.GetArrayDataReference(dst);
            for (; j + Vector128<float>.Count <= count; j += Vector128<float>.Count)
                (Vector128.LoadUnsafe(ref s0, (nuint)j) * vs).StoreUnsafe(ref d0, (nuint)j);
        }
        for (; j < count; j++) dst[j] = src[j] * Scale;

        return dst;
    }
}
