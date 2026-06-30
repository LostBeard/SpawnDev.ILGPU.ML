// End-to-end gpt-oss forward probe: load the real gpt-oss GGUF (ollama cache) via the new GGUFGraphBuilder
// gpt-oss branch (attention sinks + sliding-window + MoE experts), run one forward over a short token
// sequence, and assert the logits are FINITE with a plausible argmax. Proves the whole gpt-oss graph
// (RMSNorm -> attn(GQA+sinks+SWA) -> MoE(top-k SwiGLU-OAI, MXFP4 experts) -> lm_head) executes on real
// weights. (Token-exact correctness vs llama.cpp is a follow-up; this is the "it runs end-to-end" gate.)
//
// Run: dotnet run --project tools/GptOssRun -c Release [--cpu]
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

string blob = @"C:\Users\TJ\.ollama\models\blobs\sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb";
bool forceCpu = args.Contains("--cpu");
if (!File.Exists(blob)) { Console.WriteLine($"BLOB MISSING: {blob}"); return 2; }

using var context = MLContext.Create().ToContext();
Device device;
if (forceCpu) device = context.GetPreferredDevice(preferCPU: true);
else
{
    var cuda = context.GetCudaDevices();
    var ocl = context.GetCLDevices();
    device = cuda.Count > 0 ? (Device)cuda[0] : ocl.Count > 0 ? (Device)ocl[0] : context.GetPreferredDevice(false);
}
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"accelerator: {accelerator.Name} ({accelerator.AcceleratorType})  mem={accelerator.MemorySize / 1048576} MiB");
InferenceSession.VerboseLogging = true;

InferenceSession session;
try
{
    Console.WriteLine("loading gpt-oss (streaming)...");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    session = await InferenceSession.CreateFromGGUFFileAsync(accelerator, blob,
        onProgress: (stage, pct) => { if (pct % 25 == 0) Console.WriteLine($"  [{stage}] {pct}%"); });
    Console.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s");
}
catch (Exception ex)
{
    Console.WriteLine($"LOAD FAILED ({ex.GetType().Name}): {ex.Message}");
    if (!forceCpu && (ex is OutOfMemoryException || ex.Message.Contains("memory", StringComparison.OrdinalIgnoreCase)))
        Console.WriteLine("Likely VRAM exhaustion (gpt-oss-20b ~13.8GB). Retry with --cpu (system RAM).");
    return 1;
}

using (session)
{
    // Short arbitrary-but-valid token sequence (finite-logits probe; vocab=201088).
    int[] ids = { 1, 791, 6342, 374, 264, 1296 };
    using var idBuf = accelerator.Allocate1D(ids.Select(i => (float)i).ToArray());
    var input = new Tensor(idBuf.View, new[] { 1, ids.Length }, "input_ids");

    Console.WriteLine($"forward over {ids.Length} tokens...");
    var swf = System.Diagnostics.Stopwatch.StartNew();
    var outputs = await session.RunAsync(new Dictionary<string, Tensor> { ["input_ids"] = input });
    Console.WriteLine($"forward in {swf.Elapsed.TotalSeconds:F1}s");

    var logits = outputs.TryGetValue("logits", out var l) ? l : outputs.Values.First();
    int vocab = logits.Shape[^1];
    int seq = logits.ElementCount / vocab;
    // Browser-safe async readback. CopyToHostAsync is a MemoryBuffer extension (not on ArrayView), so copy
    // the logits view into a temp buffer (GPU->GPU CopyFrom, valid on all backends) and async-read that —
    // the EinsumOperator pattern. NOT the sync CopyToCPU, which throws on WebGPU/WebGL/Wasm.
    using var logitsBuf = accelerator.Allocate1D<float>(logits.ElementCount);
    logitsBuf.View.CopyFrom(logits.Data);
    var host = await logitsBuf.CopyToHostAsync<float>(0, logits.ElementCount);

    // last position's logits
    int basei = (seq - 1) * vocab;
    int finite = 0; float mx = float.NegativeInfinity; int argmax = -1;
    double sum = 0, sumsq = 0;
    for (int v = 0; v < vocab; v++)
    {
        float x = host[basei + v];
        if (float.IsFinite(x)) finite++;
        if (x > mx) { mx = x; argmax = v; }
        sum += x; sumsq += (double)x * x;
    }
    double mean = sum / vocab, std = Math.Sqrt(Math.Max(0, sumsq / vocab - mean * mean));
    Console.WriteLine($"logits: seq={seq} vocab={vocab}  finite={finite}/{vocab}  argmax={argmax} (logit {mx:F3})  mean={mean:F3} std={std:F3}");
    bool ok = finite == vocab && argmax >= 0 && std > 1e-3;
    Console.WriteLine(ok ? "PASS: gpt-oss forward runs end-to-end with finite, non-degenerate logits."
                         : "FAIL: non-finite or degenerate logits.");
    return ok ? 0 : 1;
}
