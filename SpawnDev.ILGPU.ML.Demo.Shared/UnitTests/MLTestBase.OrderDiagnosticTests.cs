using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// DIAGNOSTIC (2026-11-04): MoveNet inference under the compiler's stable topological order and under the old FIFO order, each with and
    /// without per-op sync, so a Wasm-only failure that appeared with the stable order ("[GE node-192 sync] Arg_NullReferenceException") is
    /// pinned to the order and to the kernel that faults. Reports every combination; fails only if the stable order fails where FIFO passes.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task Diagnostic_MoveNet_StableVsFifoOrder() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("as CreateFromFile_MoveNet_Inference");
        var http = GetHttpClient() ?? throw new UnsupportedTestException("HttpClient not available");
        const int W = 256, H = 256;
        var pixels = new int[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
                pixels[y * W + x] = (int)(x * 255f / W) | ((int)(y * 255f / H) << 8) | (128 << 16) | (0xFF << 24);
        var notes = new List<string>();
        bool stableFailsFifoPasses = false;
        bool savedOrder = GraphCompiler.StableTopologicalOrder, savedSync = GraphExecutor.PerOpSync;
        try
        {
            var outcome = new Dictionary<(bool Stable, bool Sync), bool>();
            foreach (var stable in new[] { true, false })
                foreach (var sync in new[] { false, true })
                {
                    GraphCompiler.StableTopologicalOrder = stable;
                    GraphExecutor.PerOpSync = sync;
                    string tag = $"{(stable ? "stable" : "fifo")}{(sync ? "+perOpSync" : "")}";
                    try
                    {
                        using var session = await InferenceSession.CreateFromFileAsync(accelerator, http, "models/movenet-lightning/model.onnx");
                        var result = await new PoseEstimationPipeline(session, accelerator).EstimateAsync(pixels, W, H);
                        notes.Add($"{tag}: ok ({result.Keypoints.Length} keypoints)");
                        outcome[(stable, sync)] = true;
                    }
                    catch (Exception ex)
                    {
                        var m = ex.Message.Length > 400 ? ex.Message[..400] : ex.Message;
                        notes.Add($"{tag}: FAILED {m}");
                        Console.WriteLine($"[OrderDiag] {tag} failed: {ex}");
                        outcome[(stable, sync)] = false;
                    }
                }
            stableFailsFifoPasses = !outcome[(true, false)] && outcome[(false, false)];
        }
        finally { GraphCompiler.StableTopologicalOrder = savedOrder; GraphExecutor.PerOpSync = savedSync; }
        var line = $"{accelerator.AcceleratorType}: " + string.Join(" || ", notes);
        Console.WriteLine("[OrderDiag] " + line);
        if (stableFailsFifoPasses) throw new Exception("the stable order fails where FIFO passes: " + line);
        return line;
    });
}
