#:project ../../SpawnDev.ILGPU.ML/SpawnDev.ILGPU.ML.csproj
#:property PublishAot=false
// Decode GEMV throughput: FusedDequantMatMul.Forward (M = 1) and ForwardRows (M = 1..8) on the shapes of Qwen2.5-1.5B (hidden 1536, kv 256, ffn 8960, vocab 151936),
// per block type, as effective weight bandwidth (weight bytes read / time). Weights are random blocks with valid fp16 scales (values do not change the time).
//
//   dotnet run gemv_rows_bench.cs -c Release -- [cuda|opencl] [reps=50]
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.GGUF;
using SpawnDev.ILGPU.ML.Kernels;

string which = args.Length > 0 ? args[0] : "cuda";
int reps = args.Length > 1 ? int.Parse(args[1]) : 50;
using var context = MLContext.Create().ToContext();
using var acc = which == "opencl" ? context.GetCLDevices()[0].CreateAccelerator(context) : context.GetCudaDevices()[0].CreateAccelerator(context);
using var mm = new FusedDequantMatMul(acc);
Console.WriteLine($"{acc.Name} ({acc.AcceleratorType}), {reps} reps per point");

var rng = new Random(1);
var shapes = new (string Name, int K, int N)[] { ("q/o 1536x1536", 1536, 1536), ("k/v 1536x256", 1536, 256), ("gate/up 1536x8960", 1536, 8960), ("down 8960x1536", 8960, 1536), ("head 1536x151936", 1536, 151936) };
foreach (var type in new[] { GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K })
{
    Console.WriteLine($"--- {type}");
    foreach (var (name, K, N) in shapes)
    {
        long bytes = GGMLTypes.TypeSize(type, (long)K * N);
        var raw = new byte[(bytes + 3) / 4 * 4];
        rng.NextBytes(raw);
        // valid small fp16 scales where each block keeps them (Q8_0: bytes 0-1 of 34; Q4_K: d, dmin at 0-3 of 144; Q6_K: d at 208-209 of 210)
        int blk = type == GGMLType.Q8_0 ? 34 : type == GGMLType.Q4_K ? 144 : 210;
        for (long b = 0; b + blk <= bytes; b += blk)
        {
            if (type == GGMLType.Q6_K) { raw[b + 208] = 0x00; raw[b + 209] = 0x20; }
            else { raw[b] = 0x00; raw[b + 1] = 0x20; if (type == GGMLType.Q4_K) { raw[b + 2] = 0x00; raw[b + 3] = 0x20; } }
        }
        using var w = acc.Allocate1D(raw);
        var line = $"{name,-20} {bytes / 1048576.0,7:F1} MiB:";
        foreach (int M in new[] { 1, 4, 8 })
        {
            using var x = acc.Allocate1D(Enumerable.Range(0, M * K).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray());
            using var o = acc.Allocate1D<float>((long)M * N);
            double Time(Action run)
            {
                run(); acc.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < reps; r++) run();
                acc.Synchronize();
                return sw.Elapsed.TotalMilliseconds / reps;
            }
            if (M == 1)
            {
                double f = Time(() => mm.Forward(x.View, w.View, o.View, 1, K, N, type));
                line += $"  Forward M=1 {f * 1000,6:F0} us {bytes / (f * 1e6),5:F0} GB/s |";
            }
            double t = Time(() => mm.ForwardRows(x.View, w.View, o.View, M, K, N, type));
            line += $"  Rows M={M} {t * 1000,6:F0} us {bytes / (t * 1e6),5:F0} GB/s";
        }
        Console.WriteLine(line);
    }
}
