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
//
// 🔴 THIS PROBE IS A MODEL OF THE COMPILE PATH, NOT AN OBSERVATION OF IT. It has been wrong three times:
//   * it reported 1,579 nodes for ZipVoice's text encoder where the running app reported 1,567;
//   * it predicted Whisper's decoder at 478 nodes where the session compiled 465;
//   * it reported "Pow exponent not a known 2" for EVERY LayerNorm candidate, concluding the fusion could
//     not fire - and at runtime it fires on all of them.
//
// The last one is the instructive failure and it is structural, not a bug here: InferenceSession populates
// ModelGraph.ConstantData by READING SMALL INITIALIZERS OFF THE GPU before it calls Compile. This probe
// never loads weights, so ConstantData is empty and every constant-dependent decision in the optimizer
// takes its "value unknown" branch.
//
// USE THIS FOR STRUCTURE ONLY - op histograms, which patterns exist, how many candidates there are. For
// any claim about node COUNT or about whether a constant-dependent pass fires, read
// TranscriptionResult.EncoderNodeCount / DecoderNodeCount from a real run instead.
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
    Console.WriteLine("  layernorm rejects: " + string.Join(", ",
        GraphOptimizer.LastLayerNormRejects.OrderByDescending(k => k.Value).Select(k => $"{k.Key} x{k.Value}")));
}

static string Hist(ModelGraph g) => string.Join("  ", g.Nodes.GroupBy(n => n.OpType)
    .OrderByDescending(x => x.Count()).Take(12).Select(x => $"{x.Key}x{x.Count()}"));
