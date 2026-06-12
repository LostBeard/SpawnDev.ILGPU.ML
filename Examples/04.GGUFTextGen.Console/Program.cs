// ─────────────────────────────────────────────────────────────────────────────────────────────────
//  SpawnDev.ILGPU.ML — Example 04: GGUF text-gen runner (desktop)
//
//  STREAM-loads a local .gguf (any size — gemma4:12b is 7 GB, past the ~2 GB byte[] cap) and runs a
//  forward pass on the GPU, printing the per-position argmax. This is the desktop validation vehicle for
//  the streaming GGUF loader + the gemma4 attention path; the argmax can be diffed against a llama.cpp /
//  ollama reference for the E2E correctness check.
//
//    dotnet run -- <path/to/model.gguf>                 # default tokens (load + finite-logits probe)
//    dotnet run -- <model.gguf> 2,651,6037,576,9881     # explicit input_ids (comma-separated)
// ─────────────────────────────────────────────────────────────────────────────────────────────────

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

string modelPath = args.FirstOrDefault(a => !a.Contains(',') && (a.EndsWith(".gguf") || File.Exists(a)))
    ?? @"D:\users\tj\Projects\gemma4-12b-Q4_K_M.gguf";

// input_ids: explicit comma-separated arg, else a default probe sequence (bos=2 + arbitrary valid ids).
int[] tokenIds = args.FirstOrDefault(a => a.Contains(','))?.Split(',')
        .Select(s => int.Parse(s.Trim())).ToArray()
    ?? new[] { 2, 1000, 2000, 3000, 4000 };

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model not found: {modelPath}");
    return 1;
}

try
{
    return await RunAsync(modelPath, tokenIds);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
    return 1;
}

async Task<int> RunAsync(string path, int[] ids)
{
    // The application owns the accelerator (library code never disposes it). Prefer CUDA.
    using var context = MLContext.Create().ToContext();
    var cuda = context.GetCudaDevices();
    var opencl = context.GetCLDevices();
    Device device = cuda.Count > 0 ? (Device)cuda[0]
                  : opencl.Count > 0 ? (Device)opencl[0]
                  : context.GetPreferredDevice(preferCPU: false);
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Accelerator : {accelerator.Name} ({accelerator.AcceleratorType})");

    var fi = new FileInfo(path);
    Console.WriteLine($"Model       : {path} ({fi.Length / 1024.0 / 1024.0 / 1024.0:F2} GB)");
    Console.WriteLine($"input_ids   : [{string.Join(", ", ids)}]  (seq={ids.Length})");
    Console.WriteLine();

    // ── STREAM-load (never materializes the 7 GB as a byte[]) ──
    var sw = Stopwatch.StartNew();
    InferenceSession.VerboseLogging = true;
    using var session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, path,
        onProgress: (stage, pct) => { if (pct == 0 || pct == 100) Console.WriteLine($"  [load] {stage} {pct}%"); });
    sw.Stop();
    Console.WriteLine($"\nLoaded in {sw.Elapsed.TotalSeconds:F1}s — {session}\n");

    // ── Forward pass ──
    var idf = ids.Select(i => (float)i).ToArray();
    using var inBuf = accelerator.Allocate1D(idf);
    var input = new Tensor(inBuf.View, new[] { 1, ids.Length }, "input_ids");

    // Kernel-bisection: sync after every node so a GPU trap fires on the EXACT faulting node (the verbose
    // log shows which). Toggle with env GGUF_PEROP_SYNC=1 (slow — one flush per op on a 1437-node graph).
    if (Environment.GetEnvironmentVariable("GGUF_PEROP_SYNC") == "1")
    {
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.PerOpSync = true;
        SpawnDev.ILGPU.ML.Graph.GraphExecutor.VerboseLogging = true;
    }

    sw.Restart();
    var outputs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
    await accelerator.SynchronizeAsync();
    sw.Stop();

    var logits = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
    int vocab = logits.Shape[^1];
    int seq = logits.ElementCount / vocab;
    Console.WriteLine($"Forward in {sw.Elapsed.TotalSeconds:F2}s — logits {string.Join("x", logits.Shape)} (vocab={vocab})");

    var host = new float[logits.ElementCount];
    logits.Data.CopyToCPU(host);

    // Finite check + per-position argmax (the last position is the next-token prediction).
    int nonFinite = host.Count(v => float.IsNaN(v) || float.IsInfinity(v));
    Console.WriteLine(nonFinite == 0 ? "Logits      : all finite ✓" : $"Logits      : {nonFinite} non-finite ✗");
    for (int s = 0; s < seq; s++)
    {
        int baseIdx = s * vocab, arg = 0; float best = host[baseIdx];
        for (int v = 1; v < vocab; v++) if (host[baseIdx + v] > best) { best = host[baseIdx + v]; arg = v; }
        Console.WriteLine($"  pos {s}: argmax = {arg,7}  (logit {best:F4})");
    }
    Console.WriteLine($"\nNEXT-TOKEN argmax (pos {seq - 1}) is the value to diff against a llama.cpp/ollama reference.");
    return nonFinite == 0 ? 0 : 2;
}
