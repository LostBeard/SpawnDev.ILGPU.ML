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
var g = session.Graph;
if (g == null) { Console.WriteLine("no ModelGraph exposed"); return; }

Console.WriteLine($"nodes after optimization: {g.Nodes.Count}");
var ifs = g.Nodes.Where(n => n.OpType is "If" or "Loop" or "Scan").ToList();
Console.WriteLine($"control-flow nodes surviving optimization: {ifs.Count}");

foreach (var n in ifs)
{
    var cond = n.Inputs.Count > 0 ? n.Inputs[0] : "(none)";
    bool hasConst = g.ConstantData != null && g.ConstantData.TryGetValue(cond, out var cd) && cd.Length > 0;
    string val = hasConst ? string.Join(",", g.ConstantData![cond]) : "NOT CONSTANT at this stage";
    Console.WriteLine($"  {n.OpType} '{n.Name}' condition '{cond}' -> {val}");
}

// Which of the condition's producers are still present? Anything still in the node list is a node the
// folder could not evaluate, and names exactly what is missing.
var byOp = g.Nodes.GroupBy(n => n.OpType).OrderByDescending(x => x.Count());
Console.WriteLine("surviving op histogram (top 15):");
foreach (var grp in byOp.Take(15)) Console.WriteLine($"  {grp.Key,-22} {grp.Count()}");
