#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
using SpawnDev.ILGPU.ML.Onnx;

// What is actually INSIDE each If branch, and how big is the value the taken branch returns?
//
//   dotnet run tools/probe-if-branch-contents.cs [model.onnx]
//
// ⚠️ WHY THIS EXISTS AND WHAT IT DECIDES. `probe-if-foldability.cs` answers "is the condition decidable
// once shapes are known" - WHERE the fold has to live. It does not answer what the fold must PRODUCE, and
// that decides the representation:
//
//   * a branch returning a SMALL value can be folded into ModelGraph.ConstantData/FloatConstantData, the
//     same mechanism every other compile-time fold in GraphCompiler already uses. Host-side, elided, free.
//   * a branch returning a LARGE tensor must NOT go there. A value that lives only in runtimeConstants is
//     materialised into a rented buffer via a host write on the call that needs it - and a host write
//     inside a capture window is precisely the hazard WebGPUDispatchPlan.HostWriteCount was added to
//     count. Folding a megabyte positional table into ConstantData would trade control flow for a
//     per-call upload, which is a REGRESSION wearing an optimisation's clothes.
//
// So: read the branch, count the nodes, and print the element count and BYTES of every Constant tensor it
// returns. That number chooses the design. Guessing it would be guessing about the one quantity that
// matters. Read-only, CPU only, no accelerator - safe to run beside a PMT sweep.

var cacheDir = Path.Combine(Path.GetTempPath(), "spawndev-onnx-probe");
var modelName = args.FirstOrDefault(a => a.EndsWith(".onnx")) ?? "main_zipvoice_distill_fm_decoder_int8.onnx";
var path = Path.IsPathRooted(modelName) ? modelName : Path.Combine(cacheDir, modelName);
if (!File.Exists(path)) { Console.WriteLine($"MISSING {path} - run probe-control-flow.cs first"); return; }

Console.WriteLine($"model: {Path.GetFileName(path)} ({new FileInfo(path).Length / 1024 / 1024} MB)");
var model = OnnxParser.Parse(File.ReadAllBytes(path));
var graph = model.Graph!;
Console.WriteLine($"top-level nodes: {graph.Nodes.Count}");

static string Bytes(long n) => n < 1024 ? $"{n} B" : n < 1024 * 1024 ? $"{n / 1024.0:F1} KiB" : $"{n / 1024.0 / 1024.0:F2} MiB";

static void DescribeBranch(string label, OnnxGraphProto? b)
{
    if (b == null) { Console.WriteLine($"    {label}: (absent)"); return; }
    var ops = b.Nodes.GroupBy(n => n.OpType).OrderByDescending(g => g.Count())
                     .Select(g => $"{g.Key}x{g.Count()}");
    Console.WriteLine($"    {label}: {b.Nodes.Count} node(s), {b.Initializers.Count} initializer(s), "
                    + $"outputs [{string.Join(", ", b.Outputs.Select(o => o.Name))}]");
    Console.WriteLine($"      ops: {string.Join(", ", ops)}");

    // The size question. Every Constant node's tensor, and every initializer, with its true byte cost.
    foreach (var n in b.Nodes)
    {
        if (n.OpType != "Constant") continue;
        var t = n.Attributes.FirstOrDefault(a => a.Name == "value")?.T;
        if (t == null) { Console.WriteLine($"      Constant '{n.Outputs.FirstOrDefault()}': non-tensor value"); continue; }
        long elems = t.ElementCount;
        Console.WriteLine($"      Constant -> '{n.Outputs.FirstOrDefault()}': dims [{string.Join(",", t.Dims)}] "
                        + $"= {elems} element(s), {Bytes(elems * 4)} as float32  <-- THIS is the fold's payload");
    }
    foreach (var init in b.Initializers)
        Console.WriteLine($"      initializer '{init.Name}': dims [{string.Join(",", init.Dims)}] = {init.ElementCount} element(s)");
}

int ifCount = 0;
foreach (var n in graph.Nodes)
{
    if (n.OpType is not ("If" or "Loop" or "Scan")) continue;
    ifCount++;
    Console.WriteLine();
    Console.WriteLine($"  [{ifCount}] {n.OpType} '{n.Name}'  cond='{(n.Inputs.Count > 0 ? n.Inputs[0] : "(none)")}'  "
                    + $"outputs [{string.Join(", ", n.Outputs)}]");
    DescribeBranch("then_branch", n.Attributes.FirstOrDefault(a => a.Name == "then_branch")?.G);
    DescribeBranch("else_branch", n.Attributes.FirstOrDefault(a => a.Name == "else_branch")?.G);
    DescribeBranch("body",        n.Attributes.FirstOrDefault(a => a.Name == "body")?.G);
}
Console.WriteLine();
Console.WriteLine(ifCount == 0 ? "no control-flow nodes at the top level" : $"{ifCount} control-flow node(s)");
