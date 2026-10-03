using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Tensors;

namespace ZipVoiceHarness;

/// <summary>
/// Where does a warm forward's time go? Runs a model a few times, then ONE forward with a GPU sync after every node
/// (GraphExecutor.PerOpSync), so each node's time is its own GPU work, and reports it by operator type plus the top
/// nodes and the executed-node count (on WebGPU each executed node is at least one dispatch).
///
///   dotnet run --project tools/zipvoice-harness -c Release -- profile &lt;model.onnx&gt; &lt;input&gt;=&lt;d0,d1,...&gt;[;...] [warm]
///   (an external-data model: model.onnx_data next to model.onnx is picked up automatically)
/// </summary>
static class OpProfile
{
    public static int Run(string modelPath, string inputSpec, int warm)
    {
        var inputs = inputSpec.Split(';', StringSplitOptions.RemoveEmptyEntries).Select(s =>
        {
            var kv = s.Split('=');
            return (Name: kv[0], Shape: kv[1].Split(',').Select(int.Parse).ToArray());
        }).ToArray();
        var mlBuilder = MLContext.Create();
        mlBuilder.AllAcceleratorsAsync().GetAwaiter().GetResult();
        using var mlCtx = mlBuilder.ToContext();
        using var accel = mlCtx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult()
            ?? throw new InvalidOperationException("no accelerator");
        Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");
        var dataPath = modelPath + "_data";
        using var ms = new MemoryStream(File.ReadAllBytes(modelPath));
        using var ext = File.Exists(dataPath) ? File.OpenRead(dataPath) : null;
        using var session = InferenceSession.CreateFromStreamAsync(accel, ms,
            inputShapes: inputs.ToDictionary(i => i.Name, i => i.Shape), externalDataStream: ext).GetAwaiter().GetResult();
        var feeds = new Dictionary<string, Tensor>();
        var bufs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        var rng = new Random(1);
        foreach (var (name, shape) in inputs)
        {
            var data = new float[shape.Aggregate(1, (a, b) => a * b)];
            for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
            var b = accel.Allocate1D(data);
            bufs.Add(b);
            feeds[name] = new Tensor(b.View, shape);
        }
        for (int r = 0; r < warm; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var outs = session.RunAsync(feeds).GetAwaiter().GetResult();
            accel.Synchronize();
            session.ReturnOutputs(outs);
            Console.WriteLine($"warm {r}: {sw.Elapsed.TotalMilliseconds:F1} ms");
        }
        var timings = ProfileOne(accel, () =>
        {
            var outs = session.RunAsync(feeds).GetAwaiter().GetResult();
            accel.Synchronize();
            session.ReturnOutputs(outs);
        });
        Report(timings);
        foreach (var b in bufs) b.Dispose();
        return 0;
    }

    /// <summary>One forward with a sync after every node; returns node key -> ms.</summary>
    public static Dictionary<string, double> ProfileOne(Accelerator accel, Action forward)
    {
        var timings = new Dictionary<string, double>();
        GraphExecutor.CapturedNodeTimingsMs = timings;
        GraphExecutor.PerOpSync = true;
        try { forward(); }
        finally { GraphExecutor.PerOpSync = false; GraphExecutor.CapturedNodeTimingsMs = null; }
        return timings;
    }

    public static void Report(Dictionary<string, double> timings)
    {
        // PROFILE_DUMP=<file>: every node's time, for grouping by module offline.
        if (Environment.GetEnvironmentVariable("PROFILE_DUMP") is { Length: > 0 } dump)
            File.WriteAllLines(dump, timings.OrderBy(kv => kv.Key).Select(kv => $"{kv.Value:F4}	{kv.Key}"));
        double total = timings.Values.Sum();
        Console.WriteLine($"profile  : {timings.Count} executed nodes, {total:F1} ms with a sync per node");
        foreach (var g in timings.GroupBy(kv => kv.Key.Split('_')[1]).Select(g => (Op: g.Key, N: g.Count(), Ms: g.Sum(x => x.Value)))
                     .OrderByDescending(x => x.Ms).Take(20))
            Console.WriteLine($"  {g.Op,-24} x{g.N,4}  {g.Ms,8:F2} ms  {100 * g.Ms / total,5:F1}%");
        Console.WriteLine("  top nodes:");
        foreach (var kv in timings.OrderByDescending(kv => kv.Value).Take(15))
            Console.WriteLine($"    {kv.Value,7:F2} ms  {kv.Key}");
    }
}
