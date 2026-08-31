#:project D:/users/tj/Projects/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
using SpawnDev.ILGPU.ML.Onnx;

// Does the decoder FFN fusion FIRE for distilgpt2/gpt2? Replicates GraphOptimizer.FuseLinearLayers'
// predicate on the raw parsed nodes (no ConvertToModelGraph -> avoids the WASM JSON-reflection guard):
//   MatMul/Gemm with single-consumer output -> a following Add whose OTHER input is an initializer (bias).
// Also: Gemm arity (3-input = bias built in, no trailing Add to match) + GELU form.
string[] models =
{
    @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\models\distilgpt2\onnx\decoder_model.onnx",
    @"D:\users\tj\Projects\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML\SpawnDev.ILGPU.ML.Demo\wwwroot\models\gpt2\model.onnx",
};

foreach (var path in models)
{
    if (!File.Exists(path)) { Console.WriteLine($"MISSING: {path}\n"); continue; }
    Console.WriteLine($"=== {Path.GetFileName(Path.GetDirectoryName(path))}/{Path.GetFileName(path)} ===");

    var info = OnnxLoader.ParseModelInfo(File.ReadAllBytes(path));
    var nodes = info.Nodes;
    var inits = new HashSet<string>(info.InitializerNames);

    var consumerCount = new Dictionary<string,int>();
    foreach (var n in nodes) foreach (var i in n.Inputs)
        if (!string.IsNullOrEmpty(i)) consumerCount[i] = consumerCount.GetValueOrDefault(i) + 1;

    int gemm2 = 0, gemm3 = 0, matmul = 0, fuseCandidates = 0, gemmTransB = 0;
    var byOp = new Dictionary<string,int>();
    for (int idx = 0; idx < nodes.Count; idx++)
    {
        var n = nodes[idx];
        byOp[n.OpType] = byOp.GetValueOrDefault(n.OpType) + 1;
        if (n.OpType != "MatMul" && n.OpType != "Gemm") continue;
        if (n.OpType == "Gemm")
        {
            if (n.Inputs.Length >= 3 && !string.IsNullOrEmpty(n.Inputs[2])) gemm3++; else gemm2++;
            if (n.Attributes.TryGetValue("transB", out var tb) && tb is long l && l != 0) gemmTransB++;
        }
        else matmul++;

        // FuseLinearLayers predicate: single-consumer output -> following Add(bias=initializer)
        var outName = n.Outputs.Length > 0 ? n.Outputs[0] : null;
        if (outName == null || consumerCount.GetValueOrDefault(outName) != 1) continue;
        for (int j = idx + 1; j < nodes.Count && j <= idx + 5; j++)
        {
            var c = nodes[j];
            if (c.OpType == "Add" && c.Inputs.Contains(outName))
            {
                var bias = c.Inputs[0] == outName ? (c.Inputs.Length > 1 ? c.Inputs[1] : "") : c.Inputs[0];
                if (inits.Contains(bias)) fuseCandidates++;
                break;
            }
        }
    }

    Console.WriteLine($"nodes={nodes.Count} opset={info.OpsetVersion} | Gemm(2-input)={gemm2} Gemm(3-input/bias)={gemm3} transB={gemmTransB} MatMul={matmul}");
    Console.WriteLine($"FuseLinearLayers would fire on {fuseCandidates} MatMul/Gemm->Add(bias) site(s)  <-- 0 means the lever does NOT apply to this model");
    Console.WriteLine($"GELU form: Gelu={byOp.GetValueOrDefault("Gelu")} Erf={byOp.GetValueOrDefault("Erf")} Tanh={byOp.GetValueOrDefault("Tanh")} Pow={byOp.GetValueOrDefault("Pow")}");
    Console.WriteLine();
}
