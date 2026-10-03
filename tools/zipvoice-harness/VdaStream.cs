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
        for (int f = 0; f < n; f++)
        {
            pixBuf.View.CopyFromCPU(pixels.AsSpan(f * px, px).ToArray());
            var outs = stream.RunAsync(new Tensor(pixBuf.View, new[] { 1, 1, 3, h, w })).GetAwaiter().GetResult();
            hostBuf.View.CopyFrom(outs["depth"].Data.SubView(0, dx));
            accel.Synchronize();
            var got = hostBuf.GetAsArray1D();
            session.ReturnOutputs(outs);
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
            Console.WriteLine($"  frame {f,3}: relRMS {relRms:E2}  max rel {rel:E2}{(ok ? "" : "  MISMATCH")}");
        }
        Console.WriteLine($"{n} frames in {sw.ElapsedMilliseconds} ms, worst max rel {worst:E2}");
        Console.WriteLine(bad == 0 ? "RESULT   : PASS" : $"RESULT   : FAIL ({bad} frames)");
        return bad;
    }

    static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }
}
