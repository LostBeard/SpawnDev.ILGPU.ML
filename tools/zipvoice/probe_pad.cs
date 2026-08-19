#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
// What does the Pad node in the ZipVoice text encoder actually receive? The engine threw an index error
// reading a 2-element pads list against a rank-2 input, and ONNX requires 2*rank - so either the graph is
// unusual or our shape for the input is. This reads the graph rather than guessing which.
using SpawnDev.ILGPU.ML.Onnx;

var path = @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia\text_encoder.onnx";
var info = OnnxLoader.ParseModelInfo(File.ReadAllBytes(path));

var producers = new Dictionary<string, OnnxNodeInfo>();
foreach (var n in info.Nodes) foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) producers[o] = n;

foreach (var node in info.Nodes.Where(n => n.OpType == "Pad"))
{
    Console.WriteLine($"=== Pad '{node.Name}' ===");
    Console.WriteLine($"  inputs : {string.Join(", ", node.Inputs)}");
    Console.WriteLine($"  outputs: {string.Join(", ", node.Outputs)}");
    foreach (var attr in node.Attributes) Console.WriteLine($"  attr {attr.Key} = {attr.Value}");

    foreach (var input in node.Inputs)
    {
        if (string.IsNullOrEmpty(input)) continue;
        var shape = info.ValueShapes.TryGetValue(input, out var s) ? "[" + string.Join(",", s) + "]" : "(no declared shape)";
        var kind = info.InitializerNames.Contains(input) ? "initializer"
                 : producers.TryGetValue(input, out var p) ? $"from {p.OpType} '{p.Name}'" : "graph input";
        Console.WriteLine($"  IN  {input,-40} {shape,-18} {kind}");

        // Walk one more level so a Concat feeding the pads list is visible.
        if (producers.TryGetValue(input, out var prod))
        {
            foreach (var pi in prod.Inputs)
            {
                if (string.IsNullOrEmpty(pi)) continue;
                var ps = info.ValueShapes.TryGetValue(pi, out var s2) ? "[" + string.Join(",", s2) + "]" : "(no shape)";
                var pk = info.InitializerNames.Contains(pi) ? "initializer"
                       : producers.TryGetValue(pi, out var pp) ? $"from {pp.OpType}" : "graph input";
                Console.WriteLine($"        <- {pi,-34} {ps,-14} {pk}");
            }
        }
    }
    foreach (var output in node.Outputs)
    {
        var shape = info.ValueShapes.TryGetValue(output, out var s) ? "[" + string.Join(",", s) + "]" : "(no declared shape)";
        Console.WriteLine($"  OUT {output,-40} {shape}");
    }
}
