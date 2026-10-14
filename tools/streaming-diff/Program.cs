// Compare the engine's cache-aware encoder with onnxruntime, chunk by chunk, on the SAME mel chunks (dumped by _speech-research/nemotron_chunkdump.py).
//   dotnet run --project tools/streaming-diff -c Release -- <modelDir> <dumpDir> [--half]
// Prints, per chunk, max |engine - reference| relative to the reference's own peak, for the encoder output and (for the dumped chunks) the caches after it.
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnDev.ILGPU;

var modelDir = args[0]; var dump = args[1]; var half = args.Contains("--half");
var builder = MLContext.Create(); await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync() ?? throw new Exception("no accelerator");
Console.WriteLine($"device: {accelerator.AcceleratorType} {accelerator.Name}");

var (shapes, _, _) = StreamingTranscriber.InputShapes();
var graph = new FileStream(Path.Combine(modelDir, "encoder.onnx"), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous);
var data = new FileStream(Path.Combine(modelDir, "encoder.data"), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess);
using var enc = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, graph, inputShapes: shapes, externalDataStream: data, weightStorage: half ? WeightStorage.Half : WeightStorage.Source);
enc.OutputsLiveUntilNextRun = true;

var cutsAt = Array.IndexOf(args, "--cuts");
float[] ReadF32(string name) { var b = File.ReadAllBytes(Path.Combine(name.StartsWith("cut_") && cutsAt >= 0 ? args[cutsAt + 1] : dump, name)); var f = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, f, 0, b.Length); return f; }
if (cutsAt >= 0)
{
    // Cut models of the encoder (tools/onnx-bisect-encoder.mjs), run on CHUNK 0 with zero caches: the first tensor that disagrees names the operator at fault.
    var cutDir = args[cutsAt + 1];
    var cutIndex = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(cutDir, "index.json"))).RootElement;
    var c0 = ReadF32("chunks.f32").AsSpan(0, 128 * 25).ToArray();
    foreach (var cut in cutIndex.EnumerateArray())
    {
        int i = cut.GetProperty("i").GetInt32(); var name = cut.GetProperty("name").GetString()!;
        var g = new FileStream(Path.Combine(cutDir, $"cut_{i}.onnx"), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous);
        var d = new FileStream(Path.Combine(modelDir, "encoder.data"), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.Asynchronous | FileOptions.RandomAccess);
        string verdict;
        try
        {
            using var cutSession = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, g, inputShapes: shapes, externalDataStream: d, weightStorage: half ? WeightStorage.Half : WeightStorage.Source);
            cutSession.OutputsLiveUntilNextRun = true;
            using var a0 = accelerator.Allocate1D(c0); using var l0 = accelerator.Allocate1D(new[] { 25f });
            using var z1 = accelerator.Allocate1D<float>(24 * 70 * 1024); using var z2 = accelerator.Allocate1D<float>(24 * 1024 * 8); using var z3 = accelerator.Allocate1D<float>(1);
            z1.MemSetToZero(); z2.MemSetToZero(); z3.MemSetToZero();
            var rr = await cutSession.RunAsync(new Dictionary<string, Tensor>
            {
                ["audio_signal"] = new Tensor(a0.View, new[] { 1, 128, 25 }), ["length"] = new Tensor(l0.View, new[] { 1 }),
                ["cache_last_channel"] = new Tensor(z1.View, new[] { 1, 24, 70, 1024 }), ["cache_last_time"] = new Tensor(z2.View, new[] { 1, 24, 1024, 8 }),
                ["cache_last_channel_len"] = new Tensor(z3.View, new[] { 1 }),
            });
            var t = rr[cutSession.OutputNames[0]];
            var want = ReadF32($"cut_{i}.f32");
            await accelerator.SynchronizeAsync();
            var count = Math.Min((int)t.Data.Length, want.Length);
            using var host = accelerator.Allocate1D<float>(count);
            host.View.CopyFrom(t.Data.SubView(0, count)); await accelerator.SynchronizeAsync();
            var got = host.GetAsArray1D();
            double worst = 0, peak = 1e-12; int at = 0;
            for (int k = 0; k < count; k++) { peak = Math.Max(peak, Math.Abs(want[k])); var dd = Math.Abs(got[k] - want[k]); if (dd > worst) { worst = dd; at = k; } }
            verdict = $"max|d|/peak {worst / peak:E2} (peak {peak:G4}, worst at {at}: {got[at]:G5} vs {want[at]:G5}) shape [{string.Join(",", t.Shape)}]";
        }
        catch (Exception e) { verdict = "engine could not run it: " + e.Message.Split(Environment.NewLine)[0]; }
        Console.WriteLine($"cut {i,2} {name,-62} {verdict}");
    }
    return 0;
}
var chunks = ReadF32("chunks.f32"); var refOut = ReadF32("out.f32");
const int Mels = 128, Win = 25, OutDim = 1024;
int n = chunks.Length / (Mels * Win); int outFrames = refOut.Length / n / OutDim;
Console.WriteLine($"{n} chunks, {outFrames} encoder frames each");

using var audio = accelerator.Allocate1D<float>(Mels * Win);
using var length = accelerator.Allocate1D(new[] { (float)Win });
using var cacheCh = accelerator.Allocate1D<float>(24 * 70 * 1024); using var cacheTm = accelerator.Allocate1D<float>(24 * 1024 * 8); using var cacheLen = accelerator.Allocate1D<float>(1);
cacheCh.MemSetToZero(); cacheTm.MemSetToZero(); cacheLen.MemSetToZero();

double Rel(float[] got, float[] want, int wantOffset, int count)
{
    double worst = 0, peak = 1e-12;
    for (int i = 0; i < count; i++) peak = Math.Max(peak, Math.Abs(want[wantOffset + i]));
    for (int i = 0; i < count; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[wantOffset + i]));
    return worst / peak;
}

for (int k = 0; k < n; k++)
{
    var chunk = new float[Mels * Win]; Array.Copy(chunks, k * Mels * Win, chunk, 0, chunk.Length);
    audio.CopyFromCPU(chunk);
    var r = await enc.RunAsync(new Dictionary<string, Tensor>
    {
        ["audio_signal"] = new Tensor(audio.View, new[] { 1, Mels, Win }), ["length"] = new Tensor(length.View, new[] { 1 }),
        ["cache_last_channel"] = new Tensor(cacheCh.View, new[] { 1, 24, 70, 1024 }), ["cache_last_time"] = new Tensor(cacheTm.View, new[] { 1, 24, 1024, 8 }),
        ["cache_last_channel_len"] = new Tensor(cacheLen.View, new[] { 1 }),
    });
    var o = r["outputs"]; int frames = o.Shape[2];
    using var host = accelerator.Allocate1D<float>(OutDim * frames);
    host.View.CopyFrom(o.Data.SubView(0, OutDim * frames)); await accelerator.SynchronizeAsync();
    var got = host.GetAsArray1D();
    var line = $"chunk {k,2}: out {Rel(got, refOut, k * OutDim * outFrames, OutDim * Math.Min(frames, outFrames)):E2} (frames {frames})";
    cacheCh.View.CopyFrom(r["cache_last_channel_next"].Data.SubView(0, cacheCh.Length));
    cacheTm.View.CopyFrom(r["cache_last_time_next"].Data.SubView(0, cacheTm.Length));
    cacheLen.View.CopyFrom(r["cache_last_channel_next_len"].Data.SubView(0, 1));
    await accelerator.SynchronizeAsync();
    if (File.Exists(Path.Combine(dump, $"cache_ch_{k}.f32")))
    {
        var cc = cacheCh.GetAsArray1D(); var ct = cacheTm.GetAsArray1D();
        var wc = ReadF32($"cache_ch_{k}.f32"); var wt = ReadF32($"cache_t_{k}.f32");
        var len = cacheLen.GetAsArray1D()[0];
        line += $"   cache_channel {Rel(cc, wc, 0, cc.Length):E2}  cache_time {Rel(ct, wt, 0, ct.Length):E2}  len {len}";
    }
    Console.WriteLine(line);
}

return 0;
