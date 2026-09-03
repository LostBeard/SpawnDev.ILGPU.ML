#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
using SpawnDev.ILGPU.ML.Onnx;

// What control flow does a graph actually contain, and could it be FOLDED AWAY?
//
//   dotnet run tools/probe-control-flow.cs [-- <url-or-path> ...]
//
// ⚠️ WHY THIS EXISTS. `SessionGraphCapture` refuses to record any graph containing If/Loop/Scan, because a
// device allocation inside a capture window is unrecoverable - an uncatchable 0xC0000005 on CUDA and a HUNG
// DEVICE on WebGPU. MEASURED 2026-09-03: ZipVoice's decoder is 82% of a synthesis at ~8.4 s per Euler step,
// its capture is refused for exactly this reason, and lifting the refusal hung the GPU on the first attempt
// (DXGI_ERROR_DEVICE_HUNG). So the productive question is not "can we override the guard" but "can we remove
// the reason for it" - if a branch's condition is decidable at compile time, the If can be replaced by the
// taken branch and the graph becomes capturable legitimately.
//
// This answers, per graph:
//   1. WHICH control-flow nodes exist at all.
//   2. WHERE each If's condition comes from - an initializer (constant), a Shape-derived chain (constant
//      once shapes are known, and this executor is shape-specialised), or genuine runtime data.
//   3. WHAT each branch contains - a single Constant folds trivially; a branch full of real nodes needs
//      splicing, which is a different and much larger job.
//
// Downloads are cached under the scratchpad so re-running costs nothing.

var targets = args.Length > 0 ? args : new[]
{
    "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder_int8.onnx",
    "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder_int8.onnx",
    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_with_past_model.onnx",
    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/encoder_model.onnx",
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
        if (!File.Exists(cached))
        {
            Console.WriteLine($"downloading {target} ...");
            File.WriteAllBytes(cached, await http.GetByteArrayAsync(target));
        }
        bytes = File.ReadAllBytes(cached);
        label = name;
    }
    else
    {
        if (!File.Exists(target)) { Console.WriteLine($"MISSING: {target}\n"); continue; }
        bytes = File.ReadAllBytes(target);
        label = Path.GetFileName(target);
    }

    Console.WriteLine($"=== {label}  ({bytes.Length / 1048576.0:F1} MB) ===");
    var info = OnnxLoader.ParseModelInfo(bytes);
    var nodes = info.Nodes;
    var inits = new HashSet<string>(info.InitializerNames);

    // producer map: output name -> node index
    var producer = new Dictionary<string, int>();
    for (int i = 0; i < nodes.Count; i++)
        foreach (var o in nodes[i].Outputs)
            if (!string.IsNullOrEmpty(o)) producer[o] = i;

    var cf = nodes.Select((n, i) => (n, i)).Where(x => x.n.OpType is "If" or "Loop" or "Scan").ToList();
    Console.WriteLine($"nodes={nodes.Count} opset={info.OpsetVersion} | control flow: {cf.Count}");
    if (cf.Count == 0) { Console.WriteLine("  none - this graph is capture-eligible on control flow.\n"); continue; }

    foreach (var (node, idx) in cf)
    {
        Console.WriteLine($"  [{idx}] {node.OpType} '{node.Name}'");
        if (node.Inputs.Length > 0)
        {
            var cond = node.Inputs[0];
            Console.WriteLine($"      condition input: '{cond}'");
            // Walk back up to 12 producers, reporting the op chain. A chain that bottoms out in Shape /
            // an initializer / a Constant is decidable once input shapes are fixed; one that reaches a
            // graph input carrying DATA is not.
            var chain = new List<string>();
            var cur = cond;
            for (int hop = 0; hop < 12; hop++)
            {
                if (inits.Contains(cur)) { chain.Add($"initializer({cur})"); break; }
                if (!producer.TryGetValue(cur, out var p)) { chain.Add($"graph-input-or-unknown({cur})"); break; }
                var pn = nodes[p];
                chain.Add(pn.OpType);
                if (pn.Inputs.Length == 0) break;
                cur = pn.Inputs[0];
            }
            Console.WriteLine($"      condition chain: {string.Join(" <- ", chain)}");

            // ⚠️ The single-input walk above only follows input[0], which for a comparison is HALF the
            // question. Whether the condition is decidable at compile time depends on BOTH operands, so
            // this prints the full subtree: an operand that bottoms out in an initializer, a Constant or a
            // Shape is decidable once input shapes are fixed (and this executor is shape-specialised); one
            // that reaches a data-carrying graph input is not, and no amount of folding will help.
            void Dump(string name, int depth)
            {
                var pad = new string(' ', 10 + depth * 2);
                if (inits.Contains(name)) { Console.WriteLine($"{pad}initializer '{name}'"); return; }
                if (!producer.TryGetValue(name, out var p))
                { Console.WriteLine($"{pad}GRAPH INPUT or unknown '{name}'"); return; }
                var pn = nodes[p];
                var extra = pn.OpType == "Constant" ? "  (compile-time constant)"
                          : pn.OpType == "Shape" ? "  (decidable once input shapes are fixed)" : "";
                Console.WriteLine($"{pad}{pn.OpType} '{pn.Name}'{extra}");
                if (depth >= 6) { Console.WriteLine($"{pad}  ..."); return; }
                foreach (var inp in pn.Inputs)
                    if (!string.IsNullOrEmpty(inp)) Dump(inp, depth + 1);
            }
            Console.WriteLine("      condition subtree:");
            Dump(cond, 0);
        }

        foreach (var key in new[] { "then_branch", "else_branch", "body" })
        {
            if (!node.Attributes.TryGetValue(key, out var obj) || obj is not OnnxGraphProto sub) continue;
            var ops = sub.Nodes.GroupBy(x => x.OpType)
                              .OrderByDescending(g => g.Count())
                              .Select(g => $"{g.Key}x{g.Count()}");
            Console.WriteLine($"      {key}: {sub.Nodes.Count} nodes"
                            + (sub.Nodes.Count == 0 ? "  (declared output comes from an initializer)" : "")
                            + $"  [{string.Join(" ", ops.Take(8))}]");
        }
    }
    Console.WriteLine();
}
