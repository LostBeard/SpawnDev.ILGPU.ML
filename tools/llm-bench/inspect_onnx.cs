#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// An ONNX model's graph inputs / outputs (names, element types, dims), opset, node count and op types, with ML's own parser (no Python onnx needed):
//   dotnet run inspect_onnx.cs -c Release -- <model.onnx or https URL> [...]
using SpawnDev.ILGPU.ML.Onnx;

foreach (var src in args)
{
    byte[] bytes = src.StartsWith("http", StringComparison.OrdinalIgnoreCase)
        ? await new HttpClient { Timeout = TimeSpan.FromMinutes(10) }.GetByteArrayAsync(src)
        : await File.ReadAllBytesAsync(src);
    var model = OnnxParser.Parse(bytes);
    var g = model.Graph;
    var inits = new HashSet<string>(g.Initializers.Select(t => t.Name));
    Console.WriteLine($"{src}: {bytes.Length / 1e6:F1} MB, opset {string.Join(",", model.OpsetImports.Select(o => $"{o.Domain}:{o.Version}"))}, {g.Nodes.Count} nodes");
    string Dims(OnnxValueInfoProto v) => "[" + string.Join(", ", v.Shape.Select(d => d.ToString())) + "]";
    foreach (var i in g.Inputs.Where(i => !inits.Contains(i.Name))) Console.WriteLine($"  in  {i.Name} type {i.ElemType} {Dims(i)}");
    foreach (var o in g.Outputs) Console.WriteLine($"  out {o.Name} type {o.ElemType} {Dims(o)}");
    Console.WriteLine("  ops: " + string.Join(" ", g.Nodes.GroupBy(n => n.OpType).OrderByDescending(x => x.Count()).Select(x => $"{x.Key}x{x.Count()}")));
}
