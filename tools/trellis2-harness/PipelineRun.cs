using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Whole 512 pipeline on the real checkpoints with a stand-in condition (unit-variance noise tokens where DINOv3 features go):
/// measures per-stage time and the PEAK device memory of the process (nvidia-smi polled once a second), the 12 GB gate.
/// Real image features replace the stand-in once DINOv3 is in (gated weights).
/// </summary>
internal static class PipelineRun
{
    private static long QueryUsedMiB()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits") { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
            using var p = Process.Start(psi)!;
            string s = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit();
            return long.Parse(s.Split('\n')[0].Trim());
        }
        catch { return -1; }
    }

    public static int Run512(int seed, string? glbPath = null)
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        long baseline = QueryUsedMiB(), peak = baseline;
        using var cts = new CancellationTokenSource();
        var poll = Task.Run(async () =>
        {
            while (!cts.IsCancellationRequested) { peak = Math.Max(peak, QueryUsedMiB()); try { await Task.Delay(700, cts.Token); } catch { } }
        });
        const int tokens = 1029;
        var rng = new Random(seed);
        var cond = new float[tokens * 1024];
        for (int i = 0; i < cond.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            cond[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        using var cb = accel.Allocate1D(cond);
        var pipe = new Trellis2Pipeline(accel, Path.Combine(Gpu.WeightDir, "TRELLIS.2-4B"), Path.Combine(Gpu.WeightDir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"));
        var total = Stopwatch.StartNew();
        pipe.StageDone = (name, sec) => Console.WriteLine($"  {name,-22} {sec,7:F2}s   (peak so far {peak - baseline} MiB over baseline)");
        using var output = pipe.Run512(cb.View, tokens, seed);
        cts.Cancel(); poll.Wait();
        peak = Math.Max(peak, QueryUsedMiB());
        Console.WriteLine($"total {total.Elapsed.TotalSeconds:F1}s, peak device memory {peak - baseline} MiB over the {baseline} MiB baseline");
        Console.WriteLine($"mesh {output.Mesh.VertexCount} vertices, {output.Mesh.FaceCount} triangles; {output.VoxelCount} voxels with PBR attributes");
        var attrs = output.Attributes.GetAsArray1D();
        var names = new[] { "base R", "base G", "base B", "metallic", "roughness", "alpha" };
        for (int c = 0; c < 6; c++)
        {
            double sum = 0, mn = double.MaxValue, mx = double.MinValue;
            for (int i = c; i < attrs.Length; i += 6) { sum += attrs[i]; mn = Math.Min(mn, attrs[i]); mx = Math.Max(mx, attrs[i]); }
            Console.WriteLine($"  {names[c],-9} mean {sum / output.VoxelCount:F3}  min {mn:F3}  max {mx:F3}");
        }
        if (glbPath != null)
        {
            var glb = pipe.ToGlb(output);
            File.WriteAllBytes(glbPath, glb);
            Console.WriteLine($"wrote {glbPath} ({glb.Length / 1024} KiB)");
        }
        bool finite = attrs.All(float.IsFinite);
        Console.WriteLine(finite ? "PIPELINE512 RAN (outputs finite)" : "PIPELINE512 FAIL: NaN/Inf in the attributes");
        return finite ? 0 : 1;
    }
}
