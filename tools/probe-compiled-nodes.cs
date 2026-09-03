#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property JsonSerializerIsReflectionEnabledByDefault=true
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Onnx;

// Do Constant nodes SURVIVE optimization, i.e. does the executor walk them every step?
//
// MEASURED: Whisper's decoder_with_past is 801 raw nodes of which 271 are Constant - a third of the graph -
// and ConstantOperator.Execute is an empty method whose output is already registered as an initializer at
// load. If they survive, every decode step pays the executor's per-node bookkeeping (shape interpretation,
// refcounting, pool churn) for 271 no-ops. At the ~0.87 ms/node this engine costs in a browser that is
// ~236 ms per step, on a step that costs 1,223 ms.
//
// No accelerator needed: this runs the parse -> ModelGraph -> GraphOptimizer path, which is where node
// elimination happens. (Creating an ILGPU Context needs Reflection.Emit, which file-based apps disable.)
var dir = Path.Combine(Path.GetTempPath(), "spawndev-onnx-probe");
foreach (var f in new[] { "main_onnx_decoder_with_past_model.onnx", "main_onnx_encoder_model.onnx",
                          "main_zipvoice_distill_fm_decoder_int8.onnx", "main_zipvoice_distill_text_encoder_int8.onnx" })
{
    var p = Path.Combine(dir, f);
    if (!File.Exists(p)) { Console.WriteLine($"MISSING {f}"); continue; }
    var info = OnnxLoader.ParseModelInfo(File.ReadAllBytes(p));
    var mg = InferenceSession.ConvertToModelGraph(info);
    int before = mg.Nodes.Count;
    var beforeHist = Hist(mg);
    var opt = GraphOptimizer.Optimize(mg);
    Console.WriteLine($"=== {f} ===");
    Console.WriteLine($"  nodes {before} -> {opt.Nodes.Count}");
    Console.WriteLine($"  before: {beforeHist}");
    Console.WriteLine($"  after : {Hist(opt)}");
}

static string Hist(ModelGraph g) => string.Join("  ", g.Nodes.GroupBy(n => n.OpType)
    .OrderByDescending(x => x.Count()).Take(12).Select(x => $"{x.Key}x{x.Count()}"));
