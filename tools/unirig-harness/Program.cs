using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;
using UniRigHarness;

// UniRig auto-rigging gate. UNIRIG_DIR (default E:/spawndev-unirig) holds weights/ (export_weights.py) and golden/ (golden_skeleton.py).
//   encoder     the perceiver encoder against the reference latents
//   UNIRIG_ACCELERATOR=cuda|opencl|cpu picks the device (default: preferred)
var dir = Environment.GetEnvironmentVariable("UNIRIG_DIR") ?? "E:/spawndev-unirig";
var command = args.Length > 0 ? args[0].ToLowerInvariant() : "";
return command switch
{
    "encoder" => Gates.Encoder(dir),
    _ => Usage(command),
};

static int Usage(string bad)
{
    Console.WriteLine($"unknown command '{bad}'. commands: encoder");
    return 2;
}

namespace UniRigHarness
{
    internal static class Gpu
    {
        public static (IDisposable Context, Accelerator Accelerator) Create()
        {
            var builder = MLContext.Create();
            builder.AllAcceleratorsAsync().GetAwaiter().GetResult();
            var ctx = builder.ToContext();
            var want = Environment.GetEnvironmentVariable("UNIRIG_ACCELERATOR");
            Accelerator? accelerator = null;
            if (!string.IsNullOrWhiteSpace(want))
            {
                var match = ctx.Devices.FirstOrDefault(d => string.Equals(d.AcceleratorType.ToString(), want, StringComparison.OrdinalIgnoreCase))
                    ?? throw new InvalidOperationException($"UNIRIG_ACCELERATOR='{want}' matches no device. Available: " + string.Join(", ", ctx.Devices.Select(d => d.AcceleratorType.ToString())));
                accelerator = match.CreateAccelerator(ctx);
            }
            accelerator ??= ctx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult() ?? throw new InvalidOperationException("no accelerator available");
            Console.WriteLine($"engine : {accelerator.AcceleratorType} / {accelerator.Name}");
            return (ctx, accelerator);
        }
    }

    internal static class Gates
    {
        /// <summary>The C# encoder on the golden's points, against the reference latents. bf16 weights alone move the result by relRMS ~2e-3 (check_export.py).</summary>
        public static int Encoder(string dir)
        {
            var g = Path.Combine(dir, "golden", "skeleton");
            var (pts, ps) = Npy.Load<float>(Path.Combine(g, "points.npy"));
            var (nrm, _) = Npy.Load<float>(Path.Combine(g, "normals.npy"));
            var (pre, _) = Npy.Load<long>(Path.Combine(g, "presample_idx.npy"));
            var (fps, _) = Npy.Load<long>(Path.Combine(g, "fps_idx.npy"));
            var (gold, _) = Npy.Load<float>(Path.Combine(g, "latents.npy"));
            int n = ps[0];
            var rows = fps.Select(i => (int)pre[i]).ToArray();           // queries = pre-sampled points at the FPS indices

            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            var sw = Stopwatch.StartNew();
            using var weights = GpuWeights.LoadFile(accel, Path.Combine(dir, "weights", "skeleton.safetensors"),
                name => name.StartsWith("mesh_encoder.encoder.") || name.StartsWith("output_proj."));
            Console.WriteLine($"weights  : {weights.Bytes / 1e6:F0} MB on the device in {sw.Elapsed.TotalSeconds:F1}s");
            using var p = accel.Allocate1D(pts); using var q = accel.Allocate1D(nrm);
            using var enc = new UniRigEncoder(accel, weights);

            double[] times = new double[3]; float[] got = Array.Empty<float>();
            for (int run = 0; run < times.Length; run++)
            {
                sw.Restart();
                var o = enc.Encode(p.View, q.View, n, rows);
                accel.Synchronize();
                times[run] = sw.Elapsed.TotalSeconds;
                if (run == 0) got = o.GetAsArray1D();
            }
            double num = 0, den = 0, max = 0;
            for (int i = 0; i < gold.Length; i++) { double d = got[i] - gold[i]; num += d * d; den += (double)gold[i] * gold[i]; max = Math.Max(max, Math.Abs(d)); }
            double rel = Math.Sqrt(num / den);
            Console.WriteLine($"encoder  : {QueryText(got)}  relRMS vs reference {rel:E2}   max |diff| {max:E2}   time {times[0]:F2}s (cold), {times[1]:F2}s, {times[2]:F2}s");
            bool ok = rel < 0.02;
            Console.WriteLine(ok ? "PASS" : "FAIL (relRMS >= 2e-2)");
            return ok ? 0 : 1;
        }

        private static string QueryText(float[] a) { double s = 0, ss = 0; foreach (var v in a) { s += v; ss += (double)v * v; } double m = s / a.Length; return $"[1024,1024] mean {m:F4} std {Math.Sqrt(ss / a.Length - m * m):F4}"; }
    }
}
