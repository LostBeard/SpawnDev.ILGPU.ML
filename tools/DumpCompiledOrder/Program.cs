// Diagnostic: compile an ONNX model with GraphCompiler's stable topological order and with the old FIFO order, and print the compiled
// nodes around an index side by side (the 2026-11-04 Wasm MoveNet failure at "GE node-192" appeared with the stable order).
// Run: dotnet run --project tools/DumpCompiledOrder -c Release -- <model.onnx> [center=192] [radius=12]
using System.Reflection;
using ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;

var bytes = File.ReadAllBytes(args[0]);
int center = args.Length > 1 ? int.Parse(args[1]) : 192, radius = args.Length > 2 ? int.Parse(args[2]) : 12;
using var context = MLContext.Create().ToContext();
using var acc = context.GetPreferredDevice(preferCPU: true).CreateAccelerator(context);
CompiledNode[] Compile(bool stable)
{
    GraphCompiler.StableTopologicalOrder = stable;
    using var s = InferenceSession.CreateFromOnnx(acc, bytes);
    var g = (CompiledGraph)typeof(InferenceSession).GetField("_compiled", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(s)!;
    return g.Nodes;
}
var a = Compile(true); var b = Compile(false);
GraphCompiler.StableTopologicalOrder = true;
int firstDiff = Enumerable.Range(0, Math.Min(a.Length, b.Length)).FirstOrDefault(i => a[i].OutputNames.FirstOrDefault() != b[i].OutputNames.FirstOrDefault(), -1);
Console.WriteLine($"{a.Length} nodes stable, {b.Length} FIFO; first different position {firstDiff}");
string D(CompiledNode n) => $"{n.OpType}({string.Join(",", n.InputNames.Take(3))}) -> {n.OutputNames.FirstOrDefault()}";
for (int i = Math.Max(0, center - radius); i < Math.Min(a.Length, center + radius); i++)
    Console.WriteLine($"{i,4} {(i == center ? ">>" : "  ")} stable: {D(a[i]),-90} | fifo: {(i < b.Length ? D(b[i]) : "")}");
