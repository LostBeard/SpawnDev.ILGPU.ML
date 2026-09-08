#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;

// Does the If SURVIVE optimization, and is its condition decidable by the time shapes are known?
//
//   dotnet run tools/probe-if-foldability.cs
//
// ⚠️ WHY. ZipVoice's fm_decoder carries five `If` nodes, and SessionGraphCapture must refuse any graph
// containing control flow - which is why the decoder, 82% of a synthesis, is never captured. The condition
// subtree is all initializers and one `Shape`, so it LOOKS compile-time decidable. But
// GraphOptimizer.FoldConstants only folds a node when every input is already constant, and this `Shape`
// reads an ACTIVATION - so at optimizer time it is not constant, and the chain cannot collapse there no
// matter how good the arithmetic evaluator is.
//
// This checks that claim directly rather than reasoning about it, because the answer decides WHERE the fold
// has to live: the optimizer (before shapes are known) or the compiler (after shape inference).

var ctxIL = Context.Create(b => b.Default().EnableAlgorithms());
using var accelerator = ctxIL.GetPreferredDevice(preferCPU: true).CreateAccelerator(ctxIL);
Console.WriteLine($"accelerator: {accelerator.AcceleratorType}");

var cacheDir = Path.Combine(Path.GetTempPath(), "spawndev-onnx-probe");
// ⚠️ Which model is the point. The TEXT ENCODER's `If` nodes were the first ones found, but the decoder is
// 82% of a synthesis (MEASURED 2026-09-03: 51,697 ms of a 62,221 ms speak) and it is the decoder's control
// flow that costs the capture. Default to the decoder; pass a file name to look at another.
var modelName = args.FirstOrDefault(a => a.EndsWith(".onnx")) ?? "main_zipvoice_distill_fm_decoder_int8.onnx";
var path = Path.Combine(cacheDir, modelName);
if (!File.Exists(path)) { Console.WriteLine($"MISSING {path} - run probe-control-flow.cs first"); return; }

using var session = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(path));

Console.WriteLine($"operator types present: control flow = "
    + $"[{string.Join(", ", session.OperatorTypes.Where(o => o is "If" or "Loop" or "Scan"))}]");

// The compiled graph is what capture actually inspects, so this is the state that matters.
//
// ⚠️ `InferenceSession.Graph` and its `ConstantData` are GONE from the public surface, and this probe
// still referenced both - so it stopped compiling and nobody noticed, because a loose .cs in tools/ is in
// no solution and nothing builds it (found 2026-09-08 by compiling every tool in the folder). The compiled
// node view - NodeCount + GetNode - is the same graph capture refuses on, so the question is unchanged.
//
// The constant-VALUE print is not reconstructible from the public API, and it was never the answer anyway.
// What decides where the fold has to live is whether the condition is still COMPUTED AT RUNTIME by a
// surviving node: if it is, the folder could not evaluate it, and that node names exactly what is missing.
int nodeCount = session.NodeCount;
var nodes = Enumerable.Range(0, nodeCount).Select(i => (idx: i, n: session.GetNode(i))).ToList();
Console.WriteLine($"nodes after optimization: {nodeCount}");

var ifs = nodes.Where(x => x.n.opType is "If" or "Loop" or "Scan").ToList();
Console.WriteLine($"control-flow nodes surviving optimization: {ifs.Count}");

foreach (var (idx, n) in ifs)
{
    var cond = n.inputs.Length > 0 ? n.inputs[0] : "(none)";
    var producers = nodes.Where(x => x.n.outputs.Contains(cond, StringComparer.Ordinal)).ToList();
    Console.WriteLine($"  node {idx} {n.opType} condition '{cond}' -> "
        + (producers.Count > 0
            ? $"STILL COMPUTED at runtime by node {producers[0].idx} '{producers[0].n.opType}' "
              + "- the constant folder could not evaluate it"
            : "produced by no surviving node (folded away, or a graph input / initializer)"));
}

Console.WriteLine("surviving op histogram (top 15):");
foreach (var grp in nodes.GroupBy(x => x.n.opType).OrderByDescending(x => x.Count()).Take(15))
    Console.WriteLine($"  {grp.Key,-22} {grp.Count()}");
