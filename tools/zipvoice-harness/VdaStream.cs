using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Tensors;

namespace ZipVoiceHarness;

/// <summary>
/// Video Depth Anything streaming gate: N consecutive frames through <see cref="VideoDepthAnythingStream"/> (our
/// GPU-side cache window) against the torch reference driven by VDA's own window bookkeeping
/// (_research/vda-export/make_stream_reference.py). A window bug shows as a depth drift on the frame where the
/// window first differs - frame 1 (cache replication), frame 2.. (the recent 29), frame 42+ (the sliding anchor).
///
///   dotnet run --project tools/zipvoice-harness -c Release -- vdastream &lt;model.onnx&gt; &lt;reference-basename&gt;
///
/// Each frame line also prints the session pool's buffer count: it must go FLAT after the first frames (a growing
/// count is a buffer some path never hands back). Diagnostic switches:
///   VDA_BINDINGS=1   diff two warm frames' buffer bindings (GraphExecutor.DiagBindingLog) - each changed node is a
///                    WebGPU bind-group cache miss per frame
///   VDA_FRESH=1      name every fresh pool allocation in warm frames (BufferPool.TraceFreshAllocNames)
///   VDA_POOLTRACE=1  the pool ownership trace for warm frame 6 (orphaned rents, rebinds)
/// </summary>
static class VdaStream
{
    public static int Run(string modelPath, string refBase)
    {
        var meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(refBase + ".json")).RootElement;
        int n = meta.GetProperty("frames").GetInt32(), h = meta.GetProperty("h").GetInt32(), w = meta.GetProperty("w").GetInt32();
        var pixels = ReadF32(refBase + ".pixels.f32");
        var refDepth = ReadF32(refBase + ".depth.f32");
        int px = 3 * h * w, dx = h * w;

        var mlBuilder = MLContext.Create();
        mlBuilder.AllAcceleratorsAsync().GetAwaiter().GetResult();
        using var mlCtx = mlBuilder.ToContext();
        using var accel = mlCtx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult()
            ?? throw new InvalidOperationException("no accelerator");
        Console.WriteLine($"device   : {accel.AcceleratorType} {accel.Name}");

        // Bind ONLY pixel_values, the way an app would: the cache inputs' dynamic dims are left to the session.
        using var session = InferenceSession.CreateFromFile(accel, File.ReadAllBytes(modelPath),
            inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = new[] { 1, 1, 3, h, w } });
        using var stream = new VideoDepthAnythingStream(session, accel);
        using var pixBuf = accel.Allocate1D<float>(px);
        using var hostBuf = accel.Allocate1D<float>(dx);

        double worst = 0;
        int bad = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        // VDA_BINDINGS=1: log the buffer bindings of two consecutive WARM frames and print every node whose bindings
        // changed between them - each is a WebGPU bind-group cache miss per frame (GraphExecutor.DiagBindingLog).
        bool bindings = Environment.GetEnvironmentVariable("VDA_BINDINGS") == "1";
        int logA = Math.Min(6, n - 2), logB = logA + 1;
        List<string>? logOfA = null;
        for (int f = 0; f < n; f++)
        {
            if (bindings && (f == logA || f == logB)) SpawnDev.ILGPU.ML.Graph.GraphExecutor.DiagBindingLog = new List<string>();
            // VDA_FRESH=1: name every FRESH pool allocation of the warm frames (a steady forward should make none).
            if (Environment.GetEnvironmentVariable("VDA_FRESH") == "1" && f >= 4)
            {
                SpawnDev.ILGPU.ML.Tensors.BufferPool.TraceFreshAllocNames = true;
                lock (SpawnDev.ILGPU.ML.Tensors.BufferPool.RecentFreshAllocNames) SpawnDev.ILGPU.ML.Tensors.BufferPool.RecentFreshAllocNames.Clear();
            }
            // VDA_POOLTRACE=1: the pool's ownership trace for warm frame 6 (names orphaned rents / rebinds).
            if (Environment.GetEnvironmentVariable("VDA_POOLTRACE") == "1")
            {
                SpawnDev.ILGPU.ML.Tensors.BufferPool.TracePoolOwnership = f == 6;
                if (f == 6) SpawnDev.ILGPU.ML.Tensors.BufferPool.ResetPoolOwnershipTrace();
            }
            pixBuf.View.CopyFromCPU(pixels.AsSpan(f * px, px).ToArray());
            var outs = stream.RunAsync(new Tensor(pixBuf.View, new[] { 1, 1, 3, h, w })).GetAwaiter().GetResult();
            hostBuf.View.CopyFrom(outs["depth"].Data.SubView(0, dx));
            accel.Synchronize();
            var got = hostBuf.GetAsArray1D();
            session.ReturnOutputs(outs);
            if (Environment.GetEnvironmentVariable("VDA_FRESH") == "1" && f >= 4 && f <= 7)
                lock (SpawnDev.ILGPU.ML.Tensors.BufferPool.RecentFreshAllocNames)
                    Console.WriteLine($"  fresh allocs frame {f}: {string.Join(", ", SpawnDev.ILGPU.ML.Tensors.BufferPool.RecentFreshAllocNames)}");
            if (bindings && f == logA) { logOfA = SpawnDev.ILGPU.ML.Graph.GraphExecutor.DiagBindingLog; SpawnDev.ILGPU.ML.Graph.GraphExecutor.DiagBindingLog = null; }
            if (bindings && f == logB)
            {
                var logOfB = SpawnDev.ILGPU.ML.Graph.GraphExecutor.DiagBindingLog!;
                SpawnDev.ILGPU.ML.Graph.GraphExecutor.DiagBindingLog = null;
                ReportBindingChanges(logOfA!, logOfB);
            }
            double num = 0, den = 0, maxd = 0, maxr = 0;
            for (int i = 0; i < dx; i++)
            {
                double r = refDepth[f * dx + i], d = got[i] - r;
                num += d * d; den += r * r; maxd = Math.Max(maxd, Math.Abs(d)); maxr = Math.Max(maxr, Math.Abs(r));
            }
            double relRms = Math.Sqrt(num / Math.Max(den, 1e-30)), rel = maxd / Math.Max(maxr, 1e-6);
            worst = Math.Max(worst, rel);
            bool ok = rel < 1e-4;
            if (!ok) bad++;
            Console.WriteLine($"  frame {f,3}: relRMS {relRms:E2}  max rel {rel:E2}{(ok ? "" : "  MISMATCH")}  pool buffers {session.LastExecutorBufferCount}");
        }
        Console.WriteLine($"{n} frames in {sw.ElapsedMilliseconds} ms, worst max rel {worst:E2}");
        Console.WriteLine(bad == 0 ? "RESULT   : PASS" : $"RESULT   : FAIL ({bad} frames)");
        return bad;
    }

    static void ReportBindingChanges(List<string> a, List<string> b)
    {
        Console.WriteLine($"bindings: {a.Count} executed nodes in frame A, {b.Count} in frame B");
        if (a.Count != b.Count) Console.WriteLine("  (node counts differ - comparing line by line up to the shorter)");
        var byOp = new Dictionary<string, int>();
        int changed = 0;
        for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
        {
            if (a[i] == b[i]) continue;
            changed++;
            var op = a[i].Split(' ')[1];
            byOp[op] = byOp.GetValueOrDefault(op) + 1;
            if (changed <= 25) { Console.WriteLine($"  A: {a[i]}"); Console.WriteLine($"  B: {b[i]}"); }
        }
        Console.WriteLine($"bindings changed on {changed} nodes: " + string.Join(", ", byOp.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key} x{kv.Value}")));
    }

    static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }
}
