using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>Where the weight-load seconds go: disk read vs host-to-device copy vs the library's chunked upload (<c>loadbench &lt;file.safetensors&gt;</c>).</summary>
internal static class LoadBench
{
    public static int Run(string path)
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        using var r = SafeTensorsFileReader.Open(path);
        long total = r.Tensors.Sum(t => t.DataLength);
        Console.WriteLine($"{r.Tensors.Count} tensors, {total / 1048576} MiB");
        var buf = new byte[Math.Min(total, 256L << 20)];
        for (int rep = 0; rep < 2; rep++)
        {
            var sw = Stopwatch.StartNew(); long read = 0;
            foreach (var t in r.Tensors)
                for (long off = 0; off < t.DataLength; off += buf.Length) { int n = (int)Math.Min(buf.Length, t.DataLength - off); r.ReadBytes(t, off, buf.AsSpan(0, n)); read += n; }
            double secRead = sw.Elapsed.TotalSeconds;
            using var dev = accel.Allocate1D<byte>(buf.Length);
            sw.Restart();
            for (long done = 0; done < total; done += buf.Length) dev.View.CopyFromCPU(buf);
            accel.Synchronize();
            double secCopy = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"rep {rep}: read {read / 1048576.0 / secRead:F0} MiB/s ({secRead:F2}s)   copy pageable {total / 1048576.0 / secCopy:F0} MiB/s ({secCopy:F2}s)");
        }
        {
            var sw = Stopwatch.StartNew();
            using var w = GpuWeights.LoadFile(accel, path);
            accel.Synchronize();
            Console.WriteLine($"GpuWeights.LoadFile {sw.Elapsed.TotalSeconds:F2}s");
        }
        return 0;
    }
}
