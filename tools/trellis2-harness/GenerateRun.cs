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
        return Core(accel, px, w, h, outGlb);
    }

    /// <summary>
    /// Text to 3D: SD Turbo (the ImageGenerationPipeline the ML / SpawnDev.AI demos use) draws the object on a plain background, the foreground
    /// mask cuts it out, TRELLIS.2 lifts it to a GLB. The image model is released before the 3D stages load (one model resident at a time).
    ///   text2glb "&lt;prompt&gt;" &lt;out.glb&gt;      (env TRELLIS2_SEED picks the image seed too)
    /// </summary>
    public static int Text(string prompt, string outGlb)
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        int seed = int.TryParse(Environment.GetEnvironmentVariable("TRELLIS2_SEED"), out var sd) ? sd : 42;
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var source = new SpawnDev.ILGPU.ML.Hub.HttpClientModelSource(http, SpawnDev.ILGPU.ML.Hub.FileModelStore.Default());
        using var gen = new Trellis2TextTo3D(accel, Paths(), source, Path.Combine(Gpu.WeightDir, "isnet", "isnet-general-use.onnx"));
        gen.StageDone = (name, sec) => Console.WriteLine($"  {name,-48} {sec,7:F2}s");
        var total = Stopwatch.StartNew();
        var glb = gen.GenerateGlbAsync(prompt, new Trellis2Options(), seed).GetAwaiter().GetResult();
        File.WriteAllBytes(outGlb, glb);
        var img = gen.LastImage!;
        var prev = new uint[img.Width * img.Height]; Buffer.BlockCopy(img.ImageRGBA, 0, prev, 0, prev.Length * 4);
        for (int i = 0; i < prev.Length; i++) prev[i] |= 0xFF000000u;
        File.WriteAllBytes(Path.ChangeExtension(outGlb, ".image.png"), PngEncoder.EncodeRgba8(prev, img.Width, img.Height));
        Console.WriteLine($"total {total.Elapsed.TotalSeconds:F1}s (image seed {gen.LastImageSeed}); wrote {outGlb} ({glb.Length / 1024} KiB)");
        return 0;
    }

    internal static Trellis2Paths Paths() => new()
    {
        ModelDir = Path.Combine(Gpu.WeightDir, "TRELLIS.2-4B"),
        SparseStructureDecoder = Path.Combine(Gpu.WeightDir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"),
        DinoWeights = Path.Combine(Gpu.WeightDir, "dinov3-timm-qkvb", "model.safetensors"),
    };

    private static int Core(Accelerator accel, int[] px, int w, int h, string outGlb)
    {
        long baseline = Used(), peak = baseline;
        using var cts = new CancellationTokenSource();
        bool noPoll = Environment.GetEnvironmentVariable("TRELLIS2_NOPEAK") == "1";   // the nvidia-smi polling itself slows the driver calls it measures
        var poll = Task.Run(async () => { while (!noPoll && !cts.IsCancellationRequested) { peak = Math.Max(peak, Used()); try { await Task.Delay(500, cts.Token); } catch { } } });

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
        // TRELLIS2_SOURCE=stream: load through Trellis2Weights.FromModelSource (the browser path: streamed, non-seekable reads)
        var weights = Environment.GetEnvironmentVariable("TRELLIS2_SOURCE") == "stream" ? Trellis2Weights.FromModelSource(new LocalFolderSource()) : Trellis2Weights.FromFiles(Paths());
        using var gen = new Trellis2ImageTo3D(accel, weights);
        gen.StageDone = (name, sec) => Console.WriteLine($"  {name,-32} {sec,7:F2}s   (peak so far {peak - baseline} MiB over baseline)");
        byte[] glb;
        var mf = Environment.GetEnvironmentVariable("TRELLIS2_MAXFACES");
        var rs = Environment.GetEnvironmentVariable("TRELLIS2_RES");
        var options = new Trellis2Options { MaxFaces = mf != null ? int.Parse(mf) : new Trellis2Options().MaxFaces, Resolution = rs != null ? int.Parse(rs) : 512, Seed = int.TryParse(Environment.GetEnvironmentVariable("TRELLIS2_SEED"), out var sd) ? sd : 42 };
        if (Environment.GetEnvironmentVariable("TRELLIS2_DUMP") == "1")
        {
            // debug preview data: voxel centres + PBR attributes (read by reference/preview_voxels.py)
            using var o = gen.Generate(input, w, h, options, out var pipe);
            if (Environment.GetEnvironmentVariable("TRELLIS2_MESHSTAGES") == "1")
            {
                // every stage's mesh as <out>.<stage>.obj, for reference/mesh_audit.py (which stage makes it open / non-manifold?)
                pipe.MeshStage = (stage, pos, idx) =>
                {
                    using var sw = new StreamWriter(Path.ChangeExtension(outGlb, "." + stage + ".obj"));
                    var inv = System.Globalization.CultureInfo.InvariantCulture;
                    for (int i = 0; i < pos.Length; i += 3) sw.WriteLine(string.Create(inv, $"v {pos[i]:R} {pos[i + 1]:R} {pos[i + 2]:R}"));
                    for (int i = 0; i < idx.Length; i += 3) sw.WriteLine($"f {idx[i] + 1} {idx[i + 1] + 1} {idx[i + 2] + 1}");
                    Console.WriteLine($"  mesh stage '{stage}': {pos.Length / 3} verts, {idx.Length / 3} faces");
                };
            }
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
        else
        {
            glb = gen.GenerateGlb(input, w, h, options);
            // TRELLIS2_REPEAT=n: n more runs in the same process (a server's steady state: kernels already compiled, caches warm)
            for (int r = int.TryParse(Environment.GetEnvironmentVariable("TRELLIS2_REPEAT"), out var rep) ? rep : 0; r > 0; r--)
            {
                var rsw = Stopwatch.StartNew();
                glb = gen.GenerateGlb(input, w, h, new Trellis2Options { MaxFaces = options.MaxFaces, Seed = 42 + r });
                Console.WriteLine($"  repeat run: {rsw.Elapsed.TotalSeconds:F1}s");
            }
        }
        cts.Cancel(); poll.Wait();
        peak = Math.Max(peak, Used());
        File.WriteAllBytes(outGlb, glb);
        masked?.Dispose();
        Console.WriteLine($"total {total.Elapsed.TotalSeconds:F1}s, peak device memory {peak - baseline} MiB over the {baseline} MiB baseline");
        Console.WriteLine($"wrote {outGlb} ({glb.Length / 1024} KiB)");
        return 0;
    }
}
