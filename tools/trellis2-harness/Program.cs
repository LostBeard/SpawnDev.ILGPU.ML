using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.SafeTensors;
using Trellis2Harness;

// TRELLIS.2 image-to-3D desktop gate (Plans/trellis2-image-to-3d.md).
//
//   dotnet run --project tools/trellis2-harness -c Release -- header  <file.safetensors>
//   dotnet run --project tools/trellis2-harness -c Release -- upload  <file.safetensors> [tensor]
//
// Env: TRELLIS2_ACCELERATOR=cpu|cuda|opencl pins the backend (default = preferred device; read the printed
// "engine :" line, "NVIDIA GeForce RTX 4070" is the same string via CUDA and OpenCL).
//      TRELLIS2_DIR overrides the weight folder (default: the per-user HF cache folder below).

var command = args.Length > 0 ? args[0].ToLowerInvariant() : "";
return command switch
{
    "header" => Commands.Header(args.Length > 1 ? args[1] : ""),
    "upload" => Commands.Upload(args.Length > 1 ? args[1] : "", args.Length > 2 ? args[2] : null),
    "ssflow" => Golden.SsFlow(),
    "mma" => MmaProbe.Run(),
    "attn" => Golden.Attention(),
    "sampler" => Golden.Sampler(),
    "decoders" => Golden.Decoders(),
    "dualgrid" => Golden.DualGrid(),
    "ssdec" => Golden.SsDecoder(),
    "bench" => GemmBench.Run(),
    "atlas" => AtlasGate.Run(),
    "simplify" => SimplifyGate.Run(),
    "loadbench" => LoadBench.Run(args[1]),
    "atlasmesh" => AtlasMeshRun.Run(args[1]),
    "generate" => GenerateRun.Run(args[1], args[2]),
    "text2glb" => GenerateRun.Text(args[1], args[2]),
    "prep" => Golden.Prep(),
    "dino" => Golden.Dino(),
    "dinoreal" => Golden.Dino(true),
    "bake" => BakeGate.Run(args.Length > 1 ? args[1] : null),
    "ptx" => PtxDump.Run(args[1], args[2], args[3]),
    "run512" => PipelineRun.Run512(args.Length > 1 ? int.Parse(args[1]) : 42, args.Length > 2 ? args[2] : null),
    "slat" => Golden.SlatFlow(args.Length > 1 ? args[1] : "shape"),
    _ => Usage(command),
};

static int Usage(string bad)
{
    Console.WriteLine($"unknown command '{bad}'. commands: header <file> | upload <file> [tensor] | ssflow | attn | sampler | decoders | slat shape|tex | mma");
    return 2;
}

namespace Trellis2Harness
{
    internal static class Gpu
    {
        public static readonly string WeightDir = Environment.GetEnvironmentVariable("TRELLIS2_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "spawndev-trellis2");

        /// <summary>Create the accelerator, honouring TRELLIS2_ACCELERATOR. The caller disposes both.</summary>
        public static (IDisposable Context, Accelerator Accelerator) Create()
        {
            var builder = MLContext.Create();
            builder.AllAcceleratorsAsync().GetAwaiter().GetResult();
            var ctx = builder.ToContext();
            var want = Environment.GetEnvironmentVariable("TRELLIS2_ACCELERATOR");
            Accelerator? accelerator = null;
            if (!string.IsNullOrWhiteSpace(want))
            {
                var match = ctx.Devices.FirstOrDefault(d =>
                    string.Equals(d.AcceleratorType.ToString(), want, StringComparison.OrdinalIgnoreCase))
                    ?? throw new InvalidOperationException(
                        $"TRELLIS2_ACCELERATOR='{want}' matches no device. Available: "
                      + string.Join(", ", ctx.Devices.Select(d => d.AcceleratorType.ToString())));
                accelerator = match.CreateAccelerator(ctx);
            }
            accelerator ??= ctx.CreatePreferredAcceleratorAsync().GetAwaiter().GetResult()
                ?? throw new InvalidOperationException("no accelerator available");
            Console.WriteLine($"engine : {accelerator.AcceleratorType} / {accelerator.Name}");
            return (ctx, accelerator);
        }
    }

    internal static class Commands
    {
        public static int Header(string file)
        {
            if (!File.Exists(file)) { Console.WriteLine($"no such file: {file}"); return 2; }
            using var r = SafeTensorsFileReader.Open(file);
            long total = 0, bytes = 0;
            foreach (var t in r.Tensors)
            {
                long n = SafeTensorsFileReader.ElementCount(t);
                total += n; bytes += t.DataLength;
                Console.WriteLine($"{t.DType,-5} [{string.Join(",", t.Shape)}]  {t.Name}");
            }
            Console.WriteLine($"{r.Tensors.Count} tensors, {total:N0} params, {bytes:N0} bytes");
            return 0;
        }

        /// <summary>
        /// Byte-exact upload gate: stream tensor(s) to the device, read them back, compare with the file bytes.
        /// A test that cannot fail is not a test: a flipped byte in the readback is reported by index.
        /// </summary>
        public static int Upload(string file, string? only)
        {
            if (!File.Exists(file)) { Console.WriteLine($"no such file: {file}"); return 2; }
            var (ctx, accel) = Gpu.Create();
            using var _c = ctx; using var _a = accel;
            using var r = SafeTensorsFileReader.Open(file);
            var picks = only == null ? r.Tensors.ToList() : new List<SafeTensorInfo> { r.Get(only) };
            // TRELLIS2_REDCHECK=1 skips the upload: the comparison MUST then report FAIL (proves the gate can fail).
            bool redCheck = Environment.GetEnvironmentVariable("TRELLIS2_REDCHECK") == "1";
            int bad = 0; long checkedBytes = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var t in picks)
            {
                // 2-byte dtypes land as ushort so the width check in Upload is exercised on the real element type
                if (SafeTensorsFileReader.BytesPerElement(t.DType) == 2)
                {
                    using var b16 = accel.Allocate1D<ushort>(SafeTensorsFileReader.ElementCount(t));
                    if (!redCheck) r.Upload(t, b16.View, chunkBytes: 1 << 20);
                    accel.Synchronize();
                    var back = b16.GetAsArray1D();
                    bad += Compare(r, t, System.Runtime.InteropServices.MemoryMarshal.AsBytes(back.AsSpan()));
                }
                else
                {
                    using var buf = accel.Allocate1D<byte>(t.DataLength);
                    if (!redCheck) r.Upload(t, buf.View, chunkBytes: 1 << 20);
                    accel.Synchronize();
                    var back = buf.GetAsArray1D();
                    bad += Compare(r, t, back);
                }
                checkedBytes += t.DataLength;
            }
            Console.WriteLine($"{(bad == 0 ? "PASS" : "FAIL")}: {picks.Count} tensors, {checkedBytes:N0} bytes, {bad} mismatching tensors, {sw.Elapsed.TotalSeconds:F1}s");
            return bad == 0 ? 0 : 1;
        }

        private static int Compare(SafeTensorsFileReader r, SafeTensorInfo t, ReadOnlySpan<byte> back)
        {
            var want = new byte[t.DataLength];
            r.ReadBytes(t, 0, want);
            for (int i = 0; i < want.Length; i++)
                if (want[i] != back[i])
                {
                    Console.WriteLine($"MISMATCH {t.Name} byte {i}: file {want[i]:X2} device {back[i]:X2}");
                    return 1;
                }
            return 0;
        }
    }
}
