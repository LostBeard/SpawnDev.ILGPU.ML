#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
// The pad-amount chain is the standard torch F.pad export: build a flat list, reshape to pairs, reverse
// the row order, transpose, flatten. For a rank-2 input it must end up 4 long. Our engine produced 2, so
// print every constant the chain consumes and every attribute, to find which step loses half of it.
using SpawnDev.ILGPU.ML.Onnx;

var path = @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models\sherpa-onnx-zipvoice-distill-zh-en-emilia\text_encoder.onnx";
var bytes = File.ReadAllBytes(path);
var model = OnnxParser.Parse(bytes);

var byOutput = new Dictionary<string, OnnxNodeProto>();
foreach (var n in model.Graph.Nodes) foreach (var o in n.Outputs) if (!string.IsNullOrEmpty(o)) byOutput[o] = n;

var initValues = new Dictionary<string, float[]>();
foreach (var init in model.Graph.Initializers) initValues[init.Name] = init.ToFloatArray();

string[] chain =
{
    "/Constant_3_output_0", "/ConstantOfShape_output_0", "/Concat_1_output_0", "/Reshape_output_0",
    "/Constant_4_output_0", "/Constant_5_output_0", "/Constant_6_output_0", "/Constant_7_output_0",
    "/Constant_8_output_0", "/Slice_output_0", "/Transpose_output_0", "/Reshape_1_output_0",
    "/Constant_9_output_0", "/Cast_output_0", "/Constant_2_output_0",
};

foreach (var name in chain)
{
    if (!byOutput.TryGetValue(name, out var node)) { Console.WriteLine($"{name,-30} (not produced by a node)"); continue; }
    var attrs = string.Join(" ", node.Attributes.Select(Describe));
    Console.WriteLine($"{name,-30} <- {node.OpType,-16} inputs=[{string.Join(", ", node.Inputs)}] {attrs}");

    // A Constant node carries its payload as an attribute tensor, so show what it actually holds.
    var valueAttr = node.Attributes.FirstOrDefault(a => a.Name == "value")?.T;
    if (valueAttr != null)
        Console.WriteLine($"{"",30}    value dims=[{string.Join(",", valueAttr.Dims)}] = [{string.Join(", ", valueAttr.ToFloatArray())}]");
    else if (initValues.TryGetValue(name, out var vals))
        Console.WriteLine($"{"",30}    initializer = [{string.Join(", ", vals)}]");
}

static string Describe(OnnxAttributeProto a)
{
    if (a.T != null) return $"{a.Name}=tensor(dims=[{string.Join(",", a.T.Dims)}], [{string.Join(",", a.T.ToFloatArray())}])";
    if (a.Ints != null) return $"{a.Name}=[{string.Join(",", a.Ints)}]";
    if (a.Floats != null) return $"{a.Name}=[{string.Join(",", a.Floats)}]";
    if (a.S != null) return $"{a.Name}={a.StringValue}";
    return $"{a.Name}={a.I}";
}
