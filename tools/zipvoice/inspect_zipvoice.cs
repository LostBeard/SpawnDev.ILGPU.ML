#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
// Grounds the ZipVoice port: for each of the three ONNX graphs sherpa-onnx runs,
// print the graph I/O signature, the op histogram, and which ops our engine is
// missing. No guessing about what the port needs - this reads the actual models.
using SpawnDev.ILGPU.ML.Onnx;

var root = @"D:\users\tj\Projects\SpawnDev.Reachy\SpawnDev.Reachy\models";
var fp32 = Path.Combine(root, "sherpa-onnx-zipvoice-distill-zh-en-emilia");
var int8 = Path.Combine(root, "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia");

string[] models =
{
    Path.Combine(fp32, "text_encoder.onnx"),
    Path.Combine(fp32, "fm_decoder.onnx"),
    Path.Combine(fp32, "vocos_24khz.onnx"),
    Path.Combine(int8, "encoder.int8.onnx"),
};

var allMissing = new SortedSet<string>();
foreach (var path in models)
{
    if (!File.Exists(path)) { Console.WriteLine($"MISSING FILE: {path}\n"); continue; }
    var bytes = File.ReadAllBytes(path);
    var info = ModelInspectorHelper.InspectOnnx(bytes);
    var compat = ModelInspectorHelper.CheckCompatibility(bytes);

    Console.WriteLine($"=== {Path.GetFileName(path)}  ({info.FileSizeMB}, opset {info.OpsetVersion}, {info.NodeCount} nodes, {info.TotalParametersFormatted} params) ===");
    Console.WriteLine($"producer: {info.ProducerName} {info.ProducerVersion}");
    Console.WriteLine("INPUTS:");
    foreach (var t in info.Inputs) Console.WriteLine($"   {t.Name,-28} {t.DataType,-8} {t.ShapeStr}");
    Console.WriteLine("OUTPUTS:");
    foreach (var t in info.Outputs) Console.WriteLine($"   {t.Name,-28} {t.DataType,-8} {t.ShapeStr}");
    Console.WriteLine($"OPS ({compat.CompatibilityPercent:F0}% supported, {compat.SupportedOps.Length}/{compat.TotalOpsUsed}):");
    foreach (var op in info.Operators.OrderByDescending(o => o.Count))
    {
        var mark = compat.UnsupportedOps.Contains(op.OpType) ? "  <-- NOT SUPPORTED" : "";
        Console.WriteLine($"   {op.OpType,-24} {op.Count,5}{mark}");
    }
    foreach (var m in compat.UnsupportedOps) allMissing.Add(m);
    Console.WriteLine();
}

Console.WriteLine("==== OPS TO IMPLEMENT ACROSS ALL ZIPVOICE GRAPHS ====");
Console.WriteLine(allMissing.Count == 0 ? "   none - every op is already registered" : "   " + string.Join(", ", allMissing));
