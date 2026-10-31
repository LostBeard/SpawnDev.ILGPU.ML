#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// Where a GgufBatchDecoder step goes: the time per step unprofiled, then per kernel (GgufBatchDecoder.ProfileKernels: a sync after every launch), for 1 and N sequences.
//
//   dotnet run batch_profile.cs -c Release -- GGUF [cuda|opencl] [N=8] [steps=32]
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;

string gguf = args[0], which = args.Length > 1 ? args[1] : "cuda";
int N = args.Length > 2 ? int.Parse(args[2]) : 8, steps = args.Length > 3 ? int.Parse(args[3]) : 32;
using var context = MLContext.Create().ToContext();
using var acc = which == "opencl" ? context.GetCLDevices()[0].CreateAccelerator(context) : context.GetCudaDevices()[0].CreateAccelerator(context);
using var dec = await GgufBatchDecoder.CreateAsync(acc, File.OpenRead(gguf), maxSeq: 1024, maxBatch: Math.Max(N, 1));
var tok = SentencePieceTokenizer.FromGGUF(dec.Model)!;
var (ids, _) = ChatTemplates.BuildChatPrompt(dec.Model, tok, new List<(string, string)> { ("user", "Explain in detail how a rainbow forms.") }, thinking: false);
Console.WriteLine($"{acc.Name} ({acc.AcceleratorType}), {dec.Model.Architecture} hidden {dec.Hidden} layers {dec.Layers}, prompt {ids.Length} tokens");
foreach (int n in new[] { 1, N })
{
    foreach (bool profile in new[] { false, true })
    {
        await dec.PrefillAsync(ids, 0f);
        dec.Fork(n);
        dec.KernelTimes.Clear();
        dec.ProfileKernels = profile;
        var next = Enumerable.Repeat(await dec.SampleAsync(0, 1f, new Random(0)), n).ToArray();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int s = 0; s < steps; s++)
        {
            await dec.StepAsync(next, 0f);
            for (int r = 0; r < n; r++) next[r] = await dec.SampleAsync(r, 1f, new Random(0));
        }
        double ms = sw.Elapsed.TotalMilliseconds / steps;
        dec.ProfileKernels = false;
        if (!profile) { Console.WriteLine($"--- {n} sequence(s): {ms:F2} ms/step, {n * 1000 / ms:F1} tok/s"); continue; }
        double total = dec.KernelTimes.Values.Sum() / steps;
        Console.WriteLine($"    profiled {ms:F2} ms/step, kernels {total:F2} ms/step:");
        foreach (var kv in dec.KernelTimes.OrderByDescending(kv => kv.Value))
            Console.WriteLine($"      {kv.Key,-9} {kv.Value / steps,7:F3} ms  {100 * kv.Value / dec.KernelTimes.Values.Sum(),5:F1} %");
    }
}
