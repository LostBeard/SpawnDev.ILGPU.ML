#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
// Our engine gave /fm_decoder/0/0/self_attn_weights/Reshape_5_output_0 the shape [4]; for the following
// Add against [4,1,1210,1210] to broadcast, ONNX must be giving it something like [4,1,1,1]. Read the
// node and the tensor that supplies its target shape.
using SpawnDev.ILGPU.ML.Onnx;

var path = @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia\fm_decoder.onnx";
var model = OnnxParser.Parse(File.ReadAllBytes(path));

var byOutput = new Dictionary<string, OnnxNodeProto>();
foreach (var n in model.Graph.Nodes) foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) byOutput[o] = n;
var inits = model.Graph.Initializers.ToDictionary(i => i.Name);

void Show(string name, int depth)
{
    if (depth > 5 || string.IsNullOrEmpty(name)) return;
    var pad = new string(' ', depth * 2);
    if (inits.TryGetValue(name, out var init))
    {
        Console.WriteLine($"{pad}{name} = INITIALIZER dims=[{string.Join(",", init.Dims)}] [{string.Join(",", init.ToFloatArray().Take(12))}]");
        return;
    }
    if (!byOutput.TryGetValue(name, out var node)) { Console.WriteLine($"{pad}{name} = GRAPH INPUT"); return; }

    var value = node.Attributes.FirstOrDefault(a => a.Name == "value")?.T;
    var extra = value != null ? $" value dims=[{string.Join(",", value.Dims)}] [{string.Join(",", value.ToFloatArray().Take(12))}]" : "";
    var attrs = string.Join(" ", node.Attributes.Where(a => a.Name != "value").Select(a =>
        a.Ints != null ? $"{a.Name}=[{string.Join(",", a.Ints)}]" : $"{a.Name}={a.I}"));
    Console.WriteLine($"{pad}{name} <- {node.OpType} {attrs}{extra}");
    foreach (var input in node.Inputs) Show(input, depth + 1);
}

var target = "/fm_decoder/0/0/self_attn_weights/Reshape_5_output_0";
Show(target, 0);
