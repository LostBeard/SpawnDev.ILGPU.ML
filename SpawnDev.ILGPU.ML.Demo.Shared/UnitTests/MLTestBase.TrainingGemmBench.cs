using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// BENCH (CUDA sm_75+): the tensor-core GEMM the training backward runs (Trellis2.MmaGemm: C[M, N] = A[M, K] W[N, K]^T, A fp32, W 16-bit) at
    /// llama3.1:8b's dX shapes for a 1,600-token step: q / o (K 4096 -> N 4096), down (K 4096 -> N 14336), gate / up (K 14336 -> N 4096).
    /// Reports ms and TFLOPS per variant (bf16 on sm_80+, the Turing f16 + scale kernel) as the baseline for GEMM tuning. Not a gate: it
    /// fails only if a kernel does not run. Added 2026-11-05.
    /// </summary>
    /// <summary>
    /// The packed-operand tensor-core GEMMs (Trellis2.MmaGemm.Kernel{Bf16,F16Sm75}Packed and PackedV2: 16-bit A and W, C = A W^T * sc[1]) against
    /// a float64 reference of the SAME 16-bit operands (so only accumulation order differs), on ragged shapes (M and N not tile multiples).
    /// V2 = double-buffered shared tiles + 64-bit fragment loads. CUDA sm_75+ (bf16 on sm_80+). Added 2026-11-05.
    /// </summary>
    [TestMethod]
    public async Task MmaGemm_PackedKernels_MatchReference() => await RunTest(async accelerator =>
    {
        if (!SpawnDev.ILGPU.ML.Kernels.CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("CUDA sm_75+ tensor cores");
        var notes = new List<string>();
        var rng = new Random(77);
        static float F16ToF32(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);
        static float Bf16ToF32(ushort b) => BitConverter.Int32BitsToSingle(b << 16);
        foreach (var (M, K, N) in new[] { (200, 96, 136), (33, 64, 300), (130, 160, 128) })
        {
            foreach (bool bf16 in new[] { true, false })
            {
                if (bf16 && !SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator)) continue;
                ushort Enc(float v) => bf16 ? (ushort)(BitConverter.SingleToInt32Bits(v) >> 16) : BitConverter.HalfToUInt16Bits((System.Half)v);
                float Dec(ushort u) => bf16 ? Bf16ToF32(u) : F16ToF32(u);
                var a = Enumerable.Range(0, M * K).Select(_ => Enc((float)(rng.NextDouble() * 2 - 1))).ToArray();
                var w = Enumerable.Range(0, N * K).Select(_ => Enc((float)(rng.NextDouble() * 2 - 1))).ToArray();
                var want = new double[M * N];
                for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { double s = 0; for (int k = 0; k < K; k++) s += (double)Dec(a[m * K + k]) * Dec(w[n * K + k]); want[m * N + n] = s; }
                using var ab = accelerator.Allocate1D(a); using var wb = accelerator.Allocate1D(w);
                using var cb = accelerator.Allocate1D<float>(M * N); using var sc = accelerator.Allocate1D(new[] { 1f, 1f });
                int tiles = ((M + Trellis2.MmaGemm.BM - 1) / Trellis2.MmaGemm.BM) * ((N + Trellis2.MmaGemm.BN - 1) / Trellis2.MmaGemm.BN);
                foreach (var ver in new[] { 1, 2, 4 })
                {
                    if (ver == 4 && !SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator)) continue;   // cp.async is sm_80+
                    bool v2 = ver == 2;
                    Action<KernelConfig, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                        ArrayView1D<float, Stride1D.Dense>, int, int, int> kern = ver == 4
                        ? (bf16 ? accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16PackedV4)
                                : accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75PackedV4))
                        : (bf16, v2) switch
                    {
                        (true, false) => accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16Packed),
                        (true, true) => accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16PackedV2),
                        (false, false) => accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75Packed),
                        _ => accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75PackedV2),
                    };
                    cb.MemSetToZero();
                    kern(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), ab.View.Cast<ushort, Trellis2.U4>(), wb.View.Cast<ushort, Trellis2.U4>(), sc.View, cb.View, M, K, N);
                    var got = await cb.View.CopyToHostAsync();
                    double scale = want.Max(Math.Abs), worst = 0;
                    for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
                    string tag = $"{(bf16 ? "bf16" : "f16 sm75")} V{ver} {M}x{K}x{N}";
                    notes.Add($"{tag}: {worst / scale:E1}");
                    // same 16-bit operands on both sides: only fp32-vs-float64 accumulation differs
                    if (worst > 1e-5 * scale) throw new Exception($"{tag}: max |diff| {worst:E2} (max {scale:F2}): " + string.Join("; ", notes));
                }
            }
        }
        var line = "packed tensor-core GEMMs vs float64 (same 16-bit operands): " + string.Join("; ", notes);
        Console.WriteLine("[GemmBench] " + line);
        return line;
    });

    [TestMethod(Timeout = 600000, Category = "Benchmark")]
    public async Task Bench_TrainingGemm_TensorCores() => await RunTest(async accelerator =>
    {
        if (!SpawnDev.ILGPU.ML.Kernels.CudaCaps.TuringTensorCores(accelerator)) throw new UnsupportedTestException("CUDA sm_75+ tensor cores");
        const int M = 1600, reps = 10;
        var notes = new List<string>();
        var rng = new Random(9);
        var disposables = new List<IDisposable>();
        foreach (var (name, K, N) in new[] { ("q/o", 4096, 4096), ("down", 4096, 14336), ("gate/up", 14336, 4096) })
        {
            var a = new float[(long)M * K];
            for (long i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            var w = new ushort[(long)N * K];
            for (long i = 0; i < w.Length; i++) w[i] = 0x3C00;   // 1.0 in f16 (bf16 0x3C00 = 0.0078): timing only
            using var ab = accelerator.Allocate1D(a);
            using var wb = accelerator.Allocate1D(w);
            using var cb = accelerator.Allocate1D<float>((long)M * N);
            using var sc = accelerator.Allocate1D(new[] { 1f, 1f });
            using var noBias = accelerator.Allocate1D<BFloat16>(8);
            using var noNbr = accelerator.Allocate1D<int>(1);
            int tiles = ((M + Trellis2.MmaGemm.BM - 1) / Trellis2.MmaGemm.BM) * ((N + Trellis2.MmaGemm.BN - 1) / Trellis2.MmaGemm.BN);
            var a4 = ab.View.Cast<float, Trellis2.F4>();
            var w4 = wb.View.Cast<ushort, Trellis2.U4>();
            var variants = new List<(string Tag, Action Run)>();
            if (SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator))
            {
                var k = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, int, int, int, int, int, int>(Trellis2.MmaGemm.KernelBf16Dense);
                variants.Add(("bf16", () => k(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a4, w4, noBias.View, cb.View, cb.View, noNbr.View, M, K, N, 0, 0, 0, 0)));
            }
            var k75 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.F4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75Scaled);
            variants.Add(("sm75 f16", () => k75(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a4, w4, sc.View, cb.View, M, K, N)));
            // 16-bit A (pre-packed by the caller): half the A bytes per tile, no conversion in the kernel
            var a16 = new ushort[(long)M * K];
            for (long i = 0; i < a16.Length; i++) a16[i] = 0x3C00;
            var ab16 = accelerator.Allocate1D(a16);
            disposables.Add(ab16);
            var a16v = ab16.View.Cast<ushort, Trellis2.U4>();
            if (SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator))
            {
                var kp = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16Packed);
                variants.Add(("bf16 packed-A", () => kp(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
            }
            var kp75 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75Packed);
            variants.Add(("sm75 f16 packed-A", () => kp75(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
            if (SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator))
            {
                var kv2 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16PackedV2);
                variants.Add(("bf16 packed-A V2", () => kv2(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
            }
            var kv275 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75PackedV2);
            variants.Add(("sm75 f16 packed-A V2", () => kv275(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
            if (SpawnDev.ILGPU.ML.Kernels.CudaCaps.TensorCores(accelerator))
            {
                var kv4 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelBf16PackedV4);
                variants.Add(("bf16 packed-A V4", () => kv4(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
                var kv475 = accelerator.LoadStreamKernel<ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<Trellis2.U4, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<float, Stride1D.Dense>, int, int, int>(Trellis2.MmaGemm.KernelF16Sm75PackedV4);
                variants.Add(("sm75 f16 packed-A V4", () => kv475(new KernelConfig(tiles, Trellis2.MmaGemm.Threads), a16v, w4, sc.View, cb.View, M, K, N)));
            }
            foreach (var (tag, run) in variants)
            {
                run();
                await accelerator.SynchronizeAsync();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < reps; r++) run();
                await accelerator.SynchronizeAsync();
                double ms = sw.Elapsed.TotalMilliseconds / reps;
                double tflops = 2.0 * M * K * N / (ms * 1e-3) / 1e12;
                notes.Add($"{name} {tag}: {ms:F2} ms, {tflops:F1} TFLOPS");
            }
        }
        foreach (var d in disposables) d.Dispose();
        var line = $"training GEMM M={M}: " + string.Join("; ", notes);
        Console.WriteLine("[GemmBench] " + line);
        return line;
    });
}
