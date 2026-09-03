#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
using SpawnDev.ILGPU.ML.Onnx;

// What EXACTLY does a LayerNorm look like in these graphs, and how many nodes is it worth?
//
//   dotnet run tools/probe-layernorm-pattern.cs [-- <url-or-path> ...]
//
// ⚠️ WHY. MEASURED 2026-09-03: a Whisper decode step costs ~1,223 ms, of which ~694 ms is per-node dispatch
// residual across 801 nodes - roughly 0.87 ms per node, and the same ~1 ms/node this engine pays in every
// browser pipeline. Capture cannot help here: decoder_with_past grows its past-K/V by one position every
// step, so no recorded plan is valid twice. That leaves NODE COUNT as the lever.
//
// The node histogram of one decode step is dominated by things that are not arithmetic:
//   Unsqueeze 205, Mul 74, Add 67, Transpose 62, FusedLinear 60, ReduceMean 52, Reshape 44, ...
// ReduceMean/Sub/Pow/Sqrt/Div appearing in near-equal multiples is the signature of LayerNorm written out
// as primitives - opset 14 has no LayerNormalization op, so exporters emit the whole formula. This engine
// HAS a LayerNormalization operator and a row-wise kernel; the optimizer just never fuses back to it
// (it fuses Linear, Attention and ScaledMatMul, and nothing else).
//
// This prints the actual chain hanging off each ReduceMean so the fusion is written against the real
// pattern rather than the textbook one - the axes, the epsilon position, and whether the tail is
// Mul+Add (weight+bias) or bare.

var targets = args.Length > 0 ? args : new[]
{
    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_with_past_model.onnx",
};

var cacheDir = Path.Combine(Path.GetTempPath(), "spawndev-onnx-probe");
Directory.CreateDirectory(cacheDir);
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };

foreach (var target in targets)
{
    byte[] bytes;
    string label;
    if (target.StartsWith("http", StringComparison.OrdinalIgnoreCase))
    {
        var name = string.Join("_", target.Split('/')[^3..]);
        var cached = Path.Combine(cacheDir, name);
        if (!File.Exists(cached)) File.WriteAllBytes(cached, await http.GetByteArrayAsync(target));
        bytes = File.ReadAllBytes(cached);
        label = name;
    }
    else { bytes = File.ReadAllBytes(target); label = Path.GetFileName(target); }

    var info = OnnxLoader.ParseModelInfo(bytes);
    var nodes = info.Nodes;
    Console.WriteLine($"=== {label} | {nodes.Count} nodes ===");

    var byOp = nodes.GroupBy(n => n.OpType).OrderByDescending(g => g.Count());
    Console.WriteLine("op histogram: " + string.Join("  ", byOp.Take(14).Select(g => $"{g.Key}x{g.Count()}")));

    // consumer map
    var consumers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
    for (int i = 0; i < nodes.Count; i++)
        foreach (var inp in nodes[i].Inputs)
            if (!string.IsNullOrEmpty(inp))
            {
                if (!consumers.TryGetValue(inp, out var l)) consumers[inp] = l = new List<int>();
                l.Add(i);
            }

    // Walk forward from each ReduceMean whose input is NOT itself a ReduceMean output (the mean, not the
    // variance) and print the op chain it feeds, which is the LayerNorm body.
    var reduceMeans = nodes.Select((n, i) => (n, i)).Where(x => x.n.OpType == "ReduceMean").ToList();
    Console.WriteLine($"ReduceMean nodes: {reduceMeans.Count}");

    int shown = 0;
    foreach (var (node, idx) in reduceMeans)
    {
        if (shown >= 2) break;
        Console.WriteLine($"  --- chain from [{idx}] ReduceMean '{node.Name}' ---");
        var seen = new HashSet<int>();
        var frontier = new List<int> { idx };
        for (int depth = 0; depth < 10 && frontier.Count > 0; depth++)
        {
            var next = new List<int>();
            foreach (var ni in frontier)
            {
                if (!seen.Add(ni)) continue;
                var n = nodes[ni];
                var attrs = string.Join(",", n.Attributes.Keys);
                Console.WriteLine($"      [{ni}] {n.OpType,-14} in=({string.Join(", ", n.Inputs.Select(Short))})"
                                + (attrs.Length > 0 ? $"  attrs[{attrs}]" : ""));
                foreach (var o in n.Outputs)
                    if (!string.IsNullOrEmpty(o) && consumers.TryGetValue(o, out var cs))
                        next.AddRange(cs.Take(3));
            }
            frontier = next.Distinct().ToList();
        }
        shown++;
    }
    Console.WriteLine();
}

static string Short(string s) => string.IsNullOrEmpty(s) ? "-" : (s.Length <= 28 ? s : "…" + s[^26..]);
