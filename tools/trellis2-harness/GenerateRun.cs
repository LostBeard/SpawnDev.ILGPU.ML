using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// The real thing: a photo -> (foreground mask if the image is opaque) -> DINOv3 features of the real weights -> the TRELLIS.2 512 pipeline -> GLB,
/// with per-stage time and the peak device memory of the process (the 12 GB gate).
///   generate &lt;raw prefix from reference/img2raw.py&gt; &lt;out.glb&gt;
/// </summary>
internal static class GenerateRun
{
    private static long Used()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits") { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
            using var p = Process.Start(psi)!; string s = p.StandardOutput.ReadToEnd().Trim(); p.WaitForExit();
            return long.Parse(s.Split('\n')[0].Trim());
        }
        catch { return -1; }
    }

    public static int Run(string rawPrefix, string outGlb)
    {
        var size = File.ReadAllText(rawPrefix + ".size").Split(' ');
        int w = int.Parse(size[0]), h = int.Parse(size[1]);
        var bytes = File.ReadAllBytes(rawPrefix + ".u32");
        var px = new int[w * h]; Buffer.BlockCopy(bytes, 0, px, 0, bytes.Length);
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        long baseline = Used(), peak = baseline;
        using var cts = new CancellationTokenSource();
        var poll = Task.Run(async () => { while (!cts.IsCancellationRequested) { peak = Math.Max(peak, Used()); try { await Task.Delay(500, cts.Token); } catch { } } });

        var total = Stopwatch.StartNew();
        bool opaque = px.All(v => ((v >> 24) & 255) == 255);
        using var rgba = accel.Allocate1D(px);
        MemoryBuffer1D<int, Stride1D.Dense>? masked = null;
        var input = rgba.View;
        if (opaque)
        {
            string isnet = Path.Combine(Gpu.WeightDir, "isnet", "isnet-general-use.onnx");
            var sw = Stopwatch.StartNew();
            using (var fg = new ForegroundMask(accel, isnet)) masked = fg.Apply(rgba.View, w, h);
            Console.WriteLine($"  foreground mask         {sw.Elapsed.TotalSeconds,7:F2}s");
            input = masked.View;
            var m = masked.GetAsArray1D();
            var prevPx = new uint[m.Length]; Buffer.BlockCopy(m, 0, prevPx, 0, m.Length * 4);
            File.WriteAllBytes(Path.ChangeExtension(outGlb, ".masked.png"), PngEncoder.EncodeRgba8(prevPx, w, h));
        }
        using var gen = new Trellis2ImageTo3D(accel, new Trellis2Paths
        {
            ModelDir = Path.Combine(Gpu.WeightDir, "TRELLIS.2-4B"),
            SparseStructureDecoder = Path.Combine(Gpu.WeightDir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"),
            DinoWeights = Path.Combine(Gpu.WeightDir, "dinov3-timm-qkvb", "model.safetensors"),
        });
        gen.StageDone = (name, sec) => Console.WriteLine($"  {name,-32} {sec,7:F2}s   (peak so far {peak - baseline} MiB over baseline)");
        byte[] glb;
        var mf = Environment.GetEnvironmentVariable("TRELLIS2_MAXFACES");
        var options = new Trellis2Options { MaxFaces = mf != null ? int.Parse(mf) : new Trellis2Options().MaxFaces };
        if (Environment.GetEnvironmentVariable("TRELLIS2_DUMP") == "1")
        {
            // debug preview data: voxel centres + PBR attributes (read by reference/preview_voxels.py)
            using var o = gen.Generate(input, w, h, options, out var pipe);
            var coords = o.Coords.GetAsArray1D(); var attrs = o.Attributes.GetAsArray1D();
            using var fs = File.Create(Path.ChangeExtension(outGlb, ".voxels.bin"));
            using var bw = new BinaryWriter(fs);
            bw.Write(o.VoxelCount); bw.Write(o.Grid);
            for (int i = 0; i < o.VoxelCount; i++)
            {
                for (int c = 0; c < 3; c++) bw.Write(coords[i * 3 + c]);
                for (int c = 0; c < 6; c++) bw.Write(attrs[i * 6 + c]);
            }
            glb = pipe.ToGlb(o, options.TextureSize, maxFaces: options.MaxFaces);
        }
        else glb = gen.GenerateGlb(input, w, h, options);
        cts.Cancel(); poll.Wait();
        peak = Math.Max(peak, Used());
        File.WriteAllBytes(outGlb, glb);
        masked?.Dispose();
        Console.WriteLine($"total {total.Elapsed.TotalSeconds:F1}s, peak device memory {peak - baseline} MiB over the {baseline} MiB baseline");
        Console.WriteLine($"wrote {outGlb} ({glb.Length / 1024} KiB)");
        return 0;
    }
}
