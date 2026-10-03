using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Tensors;

namespace ZipVoiceHarness;

/// <summary>
/// Does a session give back ALL its device memory on Dispose? Runs create -> forward -> dispose N times on one
/// accelerator and, after each cycle (and a full GC), counts the accelerator's LIVE device buffers - ILGPU's private
/// child-object list, read by reflection as CudaGraphCapture's census does. A cycle that leaves buffers behind is
/// a leak every app that reloads a model pays (SpawnScene: RaCo-ALIKED extractor, +10 MB per reload on WebGPU).
///
///   dotnet run --project tools/zipvoice-harness -c Release -- sessioncycle &lt;model.onnx&gt; &lt;input&gt;=&lt;d0,d1,...&gt;[;...] [cycles]
/// </summary>
static class SessionCycle
{
    public static int Run(string modelPath, string inputSpec, int cycles)
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
        var bytes = File.ReadAllBytes(modelPath);

        var baseline = Census(accel);
        Console.WriteLine($"baseline : {baseline.Count} live buffers, {baseline.Sum(b => b.Bytes) / 1048576.0:F1} MB");
        List<(string Type, long Bytes)>? prev = baseline;
        object? lastSession = null;   // CYCLE_WHO: a strong ref to the DISPOSED session, walked as a root
        for (int c = 0; c < cycles; c++)
        {
            var feeds = new Dictionary<string, Tensor>();
            var bufs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            // CYCLE_STREAM=1: load the way a browser app does (CreateFromStreamAsync), not CreateFromFile.
            using var fs = Environment.GetEnvironmentVariable("CYCLE_STREAM") == "1" ? new MemoryStream(bytes) : null;
            using (var session = fs != null
                       ? InferenceSession.CreateFromStreamAsync(accel, fs,
                             inputShapes: inputs.ToDictionary(i => i.Name, i => i.Shape),
                             streamThreshold: int.TryParse(Environment.GetEnvironmentVariable("CYCLE_THRESHOLD"), out var th) ? th : 1024 * 1024).GetAwaiter().GetResult()
                       : InferenceSession.CreateFromFile(accel, bytes,
                             inputShapes: inputs.ToDictionary(i => i.Name, i => i.Shape)))
            {
                var rng = new Random(c);
                foreach (var (name, shape) in inputs)
                {
                    var data = new float[shape.Aggregate(1, (a, b) => a * b)];
                    for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
                    var b = accel.Allocate1D(data);
                    bufs.Add(b);
                    feeds[name] = new Tensor(b.View, shape);
                }
                lastSession = session;
                var outs = session.RunAsync(feeds).GetAwaiter().GetResult();
                accel.Synchronize();
                session.ReturnOutputs(outs);
            }
            foreach (var b in bufs) b.Dispose();
            accel.Synchronize();
            var now = Census(accel);
            long mb = now.Sum(b => b.Bytes) - baseline.Sum(b => b.Bytes);
            Console.WriteLine($"cycle {c,2}  : {now.Count} live children ({now.Count - baseline.Count:+0;-0} vs baseline), " +
                              $"{mb / 1048576.0:+0.0;-0.0} MB vs baseline");
            if (c == 0 && Environment.GetEnvironmentVariable("CYCLE_LIST") == "1")
                foreach (var g in now.Where(x => x.Bytes > 0 && x.Type.StartsWith("MemoryBuffer")).GroupBy(x => x.Bytes).OrderByDescending(g => g.Key))
                    Console.WriteLine($"    live {g.Count()} x {g.Key} bytes ({g.Key / 4} floats)");
            if (c == cycles - 1 || c == 1)
            {
                // What survived this cycle that was not there before it: grouped by type and size.
                var before = prev.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
                var grown = now.GroupBy(x => x).Select(g => (g.Key, Extra: g.Count() - before.GetValueOrDefault(g.Key)))
                    .Where(x => x.Extra > 0).OrderByDescending(x => x.Key.Bytes * x.Extra).Take(15);
                foreach (var (key, extra) in grown)
                    Console.WriteLine($"    +{extra} x {key.Type} {key.Bytes} bytes");
            }
            prev = now;
        }
        if (Environment.GetEnvironmentVariable("CYCLE_WHO") == "1") FindHolders(accel, lastSession);
        return 0;
    }

    /// <summary>
    /// DIAGNOSTIC: for each live MemoryBuffer of the accelerator, the first reference path from a STATIC field (of a
    /// SpawnDev / ILGPU type) to it - i.e. who keeps a disposed session's buffer alive. Breadth-first over instance
    /// fields, arrays and collections; bounded.
    /// </summary>
    static void FindHolders(Accelerator a, object? extraRoot)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var f = typeof(Accelerator).GetField("childObjects", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var targets = new HashSet<object>(ReferenceEqualityComparer.Instance);
        lock (f.GetValue(a)!)
            foreach (var wrObj in (System.Collections.IEnumerable)f.GetValue(a)!)
                if (wrObj is WeakReference<AcceleratorObject> wr && wr.TryGetTarget(out var t) && t is MemoryBuffer mb && !mb.IsDisposed && mb.LengthInBytes >= 12)
                    targets.Add(t);
        Console.WriteLine($"holders: {targets.Count} live buffers > 64 KB");
        var flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly;
        var queue = new Queue<(object Obj, string Path, int Depth)>();
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        // Instance roots first: the accelerator and its context (locals here, so no static reaches them).
        if (extraRoot != null) { seen.Add(extraRoot); queue.Enqueue((extraRoot, "disposedSession", 0)); }
        seen.Add(a); queue.Enqueue((a, "accelerator", 0));
        seen.Add(a.Context); queue.Enqueue((a.Context, "context", 0));
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies().Where(x => x.GetName().Name is string n && (n.StartsWith("SpawnDev") || n.StartsWith("ILGPU"))))
        {
            Type[] types; try { types = asm.GetTypes(); } catch (System.Reflection.ReflectionTypeLoadException e) { types = e.Types.Where(t => t != null).ToArray()!; }
            foreach (var t in types)
            {
                if (t.ContainsGenericParameters) continue;
                foreach (var sf in t.GetFields(flags))
                {
                    object? v; try { v = sf.GetValue(null); } catch { continue; }
                    if (v != null && !v.GetType().IsPrimitive && seen.Add(v)) queue.Enqueue((v, $"{t.Name}.{sf.Name}", 0));
                }
            }
        }
        int found = 0, visited = 0;
        while (queue.Count > 0 && visited < 3_000_000 && found < 12)
        {
            var (obj, path, depth) = queue.Dequeue();
            visited++;
            if (targets.Contains(obj)) { Console.WriteLine($"  {((MemoryBuffer)obj).LengthInBytes,10} bytes <- {path}"); found++; targets.Remove(obj); continue; }
            if (depth > 14) continue;
            var ty = obj.GetType();
            if (ty == typeof(string) || ty.IsPrimitive || ty.IsPointer) continue;
            if (obj is Array arr)
            {
                if (ty.GetElementType()!.IsPrimitive) continue;
                int i = 0;
                foreach (var e in arr) { if (e != null && seen.Add(e)) queue.Enqueue((e, $"{path}[{i}]", depth + 1)); if (++i > 4096) break; }
                continue;
            }
            for (var tt = ty; tt != null && tt != typeof(object); tt = tt.BaseType)
                foreach (var fi in tt.GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly))
                {
                    if (fi.FieldType.IsPrimitive || fi.FieldType.IsPointer || fi.FieldType == typeof(string)) continue;
                    object? v; try { v = fi.GetValue(obj); } catch { continue; }
                    if (v == null || v.GetType().IsPrimitive) continue;
                    if (v is WeakReference) continue;   // weak references do not keep anything alive
                    if (seen.Add(v)) queue.Enqueue((v, $"{path}.{fi.Name}", depth + 1));
                }
        }
        Console.WriteLine($"holders: visited {visited} objects; {targets.Count} buffer(s) not reached from statics");
    }

    /// <summary>Live MemoryBuffer children of the accelerator (type name, bytes), after a full GC.</summary>
    static List<(string Type, long Bytes)> Census(Accelerator a)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var result = new List<(string, long)>();
        var f = typeof(Accelerator).GetField("childObjects",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (f?.GetValue(a) is not System.Collections.IEnumerable list) throw new InvalidOperationException("childObjects not found");
        lock (list)
            foreach (var wrObj in list)
                if (wrObj is WeakReference<AcceleratorObject> wr && wr.TryGetTarget(out var t) && !t.IsDisposed)
                    result.Add((t.GetType().Name, t is MemoryBuffer mb ? mb.LengthInBytes : 0));
        return result;
    }
}
