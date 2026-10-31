using ILGPU;
using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// Wan 2.2's linear layers at the 832x480 x33 clip's 3510 tokens (attention projections 3072 -> 3072 and -> 9216, feed forward 3072 -> 14336 and 14336 -> 3072, the text's
    /// 512 rows -> 6144) from synthetic 8-bit blocks: the bf16 matrix + float GEMM path against W8A8 (and W8A8 in one dispatch, no row chunks), on every GPU lane. The two must
    /// agree (the activation quantisation only); the times (median of 5 after a warm-up) are reported. No model files.
    /// </summary>
    [TestMethod(Timeout = 900000)]
    public async Task W8A8_Linear_MatchesAndTimes() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("a GPU benchmark (the int8 dot runs its exact form on the CPU lanes: tested by SpawnDev.ILGPU)");
        var shapes = new (int M, int K, int N)[] { (3510, 3072, 3072), (3510, 3072, 9216), (3510, 3072, 14336), (3510, 14336, 3072), (512, 3072, 6144) };
        var rng = new Random(5);
        var fk = new Flux2Kernels(accelerator); var dk = new DiTKernels(accelerator);
        using var q8 = new Q8Linear(accelerator);
        async Task<double> Time(Action run)
        {
            run(); await accelerator.SynchronizeAsync();
            var ts = new List<double>();
            for (int i = 0; i < 5; i++) { var sw = System.Diagnostics.Stopwatch.StartNew(); run(); await accelerator.SynchronizeAsync(); ts.Add(sw.Elapsed.TotalMilliseconds); }
            ts.Sort(); return ts[2];
        }
        var report = new List<string>();
        bool ok = true;
        bool savedT = DiTKernels.EnableTensorCores; long savedChunk = Q8Linear.MaxFlopsPerDispatch;
        try
        {
            DiTKernels.EnableTensorCores = false;
            // two passes over the shapes, the first unrecorded: a GPU that has idled runs its first seconds at low clocks (a cold float run once read 2.5x slower)
            foreach (var pass in new[] { 0, 1 })
            foreach (var (M, K, N) in shapes)
            {
                int blocks = K / 32;
                var words = new uint[(long)N * blocks * 9];
                for (long i = 0; i < words.Length; i++)
                {
                    if (i % 9 == 0)
                    {
                        ushort d = (ushort)BitConverter.HalfToUInt16Bits((System.Half)(rng.NextDouble() * 0.002 + 0.0005)), mn = (ushort)BitConverter.HalfToUInt16Bits((System.Half)(-(rng.NextDouble() * 0.15)));
                        words[i] = d | ((uint)mn << 16);
                    }
                    else words[i] = (uint)rng.Next() ^ ((uint)rng.Next(4) << 30);
                }
                var x = new float[(long)M * K];
                for (long i = 0; i < x.Length; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * (i % 97 == 0 ? 8 : 1));      // a few outliers per row
                using var wq = accelerator.Allocate1D(words); using var xb = accelerator.Allocate1D(x);
                using var o1 = accelerator.Allocate1D<float>((long)M * N); using var o2 = accelerator.Allocate1D<float>((long)M * N);
                using var bf = accelerator.Allocate1D<uint>((long)N * K / 2);               // the float path's bf16 copy of the matrix (Flux2Weights.Matrix)
                double tFloat = await Time(() => { fk.DequantQ8(wq.View, bf.View, N, K); dk.Linear(xb.View, bf.View.Cast<uint, BFloat16>(), null, o1.View, M, K, N); });
                Q8Linear.MaxFlopsPerDispatch = savedChunk;
                double tInt = await Time(() => q8.Linear(xb.View, wq.View, null, o2.View, M, K, N, DiTKernels.Act.None));
                Q8Linear.MaxFlopsPerDispatch = long.MaxValue;
                double tOne = await Time(() => q8.Linear(xb.View, wq.View, null, o2.View, M, K, N, DiTKernels.Act.None));
                Q8Linear.MaxFlopsPerDispatch = savedChunk;
                var a = await o1.View.CopyToHostAsync(); var b = await o2.View.CopyToHostAsync();
                var m = WanCompare(b, a);
                if (pass == 0) continue;
                report.Add($"{M}x{K}x{N}: float {tFloat:F1} ms, W8A8 {tInt:F1} ms ({tFloat / tInt:F2}x), W8A8 one dispatch {tOne:F1} ms; relRMS {m.RelRms:E2}");
                ok &= m.RelRms < 1e-2;
            }
        }
        finally { DiTKernels.EnableTensorCores = savedT; Q8Linear.MaxFlopsPerDispatch = savedChunk; }
        var head = $"{accelerator.AcceleratorType} (int8 dot {(accelerator.Capabilities.PackedInt8DotProduct ? "native" : "emulated")}, activation blocks of {Q8Linear.ActivationBlock})";
        foreach (var l in report) Console.WriteLine($"[W8A8] {head}: {l}");
        if (!OperatingSystem.IsBrowser()) try { var od = Path.Combine(WanDir, "out"); Directory.CreateDirectory(od); File.AppendAllLines(Path.Combine(od, "wan_tests.log"), report.Select(l => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [W8A8 linear] {head}: {l}")); } catch (IOException) { }
        if (!ok) throw new Exception(head + ": " + string.Join("; ", report));
        return head + ": " + string.Join("; ", report);
    });

    /// <summary>
    /// What W8A8 costs a Wan clip against what the 8-bit weights already cost: the Turbo transformer at the /video page's 832x480 x33 (4 steps, seed 8) as the bf16 export,
    /// the 8-bit export on the float GEMM, and the 8-bit export with W8A8: the gate behind Wan's automatic W8A8 (<see cref="Flux2Weights.AutoW8A8"/>): with the default
    /// two-level activation scales its W8A8 video must be within 1.5 dB of the 8-bit float video's PSNR against bf16 (MEASURED 2026-10-31: 32.4 vs 30.8 dB; scales per 32
    /// gave 27.4, which is why they are not the default). Other scales (ML_W8A8_AB) only fail under 20 dB. Run it on OpenCL: fp32 float GEMMs like the browser's, and the
    /// W8A8 path is integer-exact on every backend. Desktop GPU (the 10 GB bf16 Turbo file).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_W8A8_ClipCost() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU (the 10 GB bf16 Turbo file)");
        var bf16 = Path.Combine("turbo", "wan22_ti2v_5b_turbo_dit_bf16.safetensors");
        if (!File.Exists(Path.Combine(WanDir, bf16)) || !File.Exists(Path.Combine(WanDir, Wan22Paths.DitFile("turbo_q8")))) throw new UnsupportedTestException($"Wan Turbo exports not found in {WanDir}");
        async Task<WanVideo> Run(string dit, bool w8a8)
        {
            var saved = Flux2Weights.EnableW8A8; bool savedT = DiTKernels.EnableTensorCores;
            Flux2Weights.EnableW8A8 = w8a8; DiTKernels.EnableTensorCores = false;
            try
            {
                var pipe = new Wan22Pipeline(accelerator, Wan22Paths.InFolder(WanDir, dit, Wan22Paths.TextEncoderFile("q8")));
                return await pipe.GenerateAsync(new WanRequest { Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.", Width = 832, Height = 480, Frames = 33, Steps = 4, GuidanceScale = 1.0, Sampler = WanSampler.FewStep, Seed = 8, RgbaFrames = true });
            }
            finally { Flux2Weights.EnableW8A8 = saved; DiTKernels.EnableTensorCores = savedT; }
        }
        static double Psnr(WanVideo a, WanVideo b)
        {
            double mse = 0; long n = 0;
            for (int f = 0; f < a.Frames; f++)
            {
                byte[] fa = a.GetFrameRgba(f), fb = b.GetFrameRgba(f);
                for (int i = 0; i < fa.Length; i++) { if ((i & 3) == 3) continue; double d = (fa[i] - fb[i]) / 255.0; mse += d * d; n++; }
            }
            return 10 * Math.Log10(1 / (mse / n));
        }
        var exact = await Run(bf16, false);
        var q8 = await Run(Wan22Paths.DitFile("turbo_q8"), false);
        var w8 = await Run(Wan22Paths.DitFile("turbo_q8"), true);
        double q8VsBf16 = Psnr(q8, exact), w8VsBf16 = Psnr(w8, exact), w8VsQ8 = Psnr(w8, q8);
        var report = $"Wan Turbo 832x480 x33 on {accelerator.AcceleratorType}, activation blocks of {Q8Linear.ActivationBlock}: PSNR vs the bf16 video: 8-bit {q8VsBf16:F1} dB, 8-bit W8A8 {w8VsBf16:F1} dB; W8A8 vs 8-bit {w8VsQ8:F1} dB";
        Console.WriteLine("[W8A8] " + report);
        try { var od = Path.Combine(WanDir, "out"); Directory.CreateDirectory(od); File.AppendAllLines(Path.Combine(od, "wan_tests.log"), new[] { $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [W8A8 clip cost] {report}" }); } catch (IOException) { }
        if (!(w8VsBf16 > (Q8Linear.ActivationBlock == 0 ? q8VsBf16 - 1.5 : 20))) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// W8A8 (<see cref="Flux2Weights.EnableW8A8"/>: 8-bit weight blocks times activations quantised per block of 32, on packed int8 dot products) against the same 8-bit DiT
    /// through its bf16 matrix and a float GEMM, on the Wan 2.2 transformer (q8 export, the step-0 inputs of the 640x352 x17 reference run): one forward each, exact fp32 GEMMs
    /// for the reference (no tensor cores). The difference is the activation quantisation alone; it must stay small next to what the 8-bit weights themselves cost (3.5e-3
    /// against bf16). Times of both paths are reported (on CUDA also the tensor-core path, the default there). Desktop GPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_W8A8_MatchesDequantizedGemm() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU (the browser measures W8A8 in Wan_W8A8_OverHttp)");
        var dit8 = Path.Combine(WanDir, Wan22Paths.DitFile("q8")); var fixture = Path.Combine(WanDir, "dump2", "dit_t2v30.npz"); var t5 = Path.Combine(WanDir, "dump", "t5_0.npz");
        if (!File.Exists(dit8) || !File.Exists(fixture) || !File.Exists(t5)) throw new UnsupportedTestException($"Wan files not found in {WanDir}");
        using var fx = ZipFile.OpenRead(fixture); using var t50 = ZipFile.OpenRead(t5);
        var (lat, ls) = WanNpz(fx, "step0_latent_in"); var t = WanNpz(fx, "step0_t").Data[0]; var (ctx, cs) = WanNpz(t50, "context");
        using var weights = GpuWeights.LoadFile(accelerator, dit8);
        using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
        dit.SetGrid(ls[1], ls[2], ls[3]);
        using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
        async Task<(float[] Out, double Ms)> Run(bool w8a8, bool tensorCores)
        {
            var savedW = Flux2Weights.EnableW8A8; bool savedT = DiTKernels.EnableTensorCores;
            try
            {
                Flux2Weights.EnableW8A8 = w8a8; DiTKernels.EnableTensorCores = tensorCores;
                dit.SetContext(0, ctx, cs[0]);
                dit.Forward(lIn.View, t, false, 0, lOut.View); await accelerator.SynchronizeAsync();     // warm-up: kernels compiled
                var sw = System.Diagnostics.Stopwatch.StartNew();
                dit.Forward(lIn.View, t, false, 0, lOut.View); await accelerator.SynchronizeAsync();
                double ms = sw.Elapsed.TotalMilliseconds;
                return (await lOut.View.CopyToHostAsync(), ms);
            }
            finally { Flux2Weights.EnableW8A8 = savedW; DiTKernels.EnableTensorCores = savedT; }
        }
        var exact = await Run(false, false);
        var w8 = await Run(true, false);
        var m = WanCompare(w8.Out, exact.Out);
        var report = $"{accelerator.AcceleratorType} (packed int8 dot {(accelerator.Capabilities.PackedInt8DotProduct ? "native" : "emulated")}): W8A8 vs the dequantised GEMM relRMS {m.RelRms:E2} (maxAbs {m.MaxAbs:E2}); forward {exact.Ms:F0} ms float GEMM, {w8.Ms:F0} ms W8A8";
        if (accelerator.AcceleratorType == AcceleratorType.Cuda) { var tc = await Run(false, true); report += $", {tc.Ms:F0} ms tensor cores"; }
        Console.WriteLine("[W8A8] " + report);
        try { var od = Path.Combine(WanDir, "out"); Directory.CreateDirectory(od); File.AppendAllLines(Path.Combine(od, "wan_tests.log"), new[] { $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [W8A8] {report}" }); } catch (IOException) { }
        if (!(m.RelRms < 2e-2)) throw new Exception(report);
        return report;
    });
}
