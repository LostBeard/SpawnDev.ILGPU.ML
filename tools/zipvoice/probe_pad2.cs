#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
// Walk backwards from the Pad node's pads input to see how the pad amounts are computed, and how long
// that list really is. The engine substituted a 2-element zero array; ONNX wants 2 * rank.
using SpawnDev.ILGPU.ML.Onnx;

var path = @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia\text_encoder.onnx";
var info = OnnxLoader.ParseModelInfo(File.ReadAllBytes(path));

var producers = new Dictionary<string, OnnxNodeInfo>();
foreach (var n in info.Nodes) foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) producers[o] = n;
var inits = new HashSet<string>(info.InitializerNames);

void Walk(string name, int depth, HashSet<string> seen)
{
    if (depth > 6 || string.IsNullOrEmpty(name) || !seen.Add(name)) return;
    var indent = new string(' ', depth * 2);
    var shape = info.ValueShapes.TryGetValue(name, out var s) ? "[" + string.Join(",", s) + "]" : "";

    if (inits.Contains(name)) { Console.WriteLine($"{indent}{name} {shape} = INITIALIZER"); return; }
    if (!producers.TryGetValue(name, out var node)) { Console.WriteLine($"{indent}{name} {shape} = GRAPH INPUT"); return; }

    var attrs = string.Join(" ", node.Attributes.Select(a => $"{a.Key}={Fmt(a.Value)}"));
    Console.WriteLine($"{indent}{name} {shape} <- {node.OpType} {attrs}");
    foreach (var input in node.Inputs) Walk(input, depth + 1, seen);
}

static string Fmt(object v) => v switch
{
    long[] a => "[" + string.Join(",", a) + "]",
    float[] f => "[" + string.Join(",", f) + "]",
    byte[] b => $"bytes({b.Length})",
    _ => v?.ToString() ?? "null",
};

var pad = info.Nodes.First(n => n.OpType == "Pad");
Console.WriteLine($"Pad data  input: {pad.Inputs[0]}");
Console.WriteLine($"Pad pads  input: {pad.Inputs[1]}");
Console.WriteLine();
Console.WriteLine("--- how the pads list is built ---");
Walk(pad.Inputs[1], 0, new HashSet<string>());

Console.WriteLine();
Console.WriteLine("--- the Concat that feeds the data input ---");
var concat = producers[pad.Inputs[0]];
Console.WriteLine($"{concat.OpType} attrs: {string.Join(" ", concat.Attributes.Select(a => $"{a.Key}={Fmt(a.Value)}"))}");
