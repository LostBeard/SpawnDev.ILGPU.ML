using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Golden-tensor gates: tensors dumped from the REAL TRELLIS.2 reference classes + REAL weights (CPU torch, fp32,
/// scratchpad golden_*.py, written to TRELLIS2_DIR/golden/...) against our GPU kernels. Metric = relRMS
/// (rms(diff) / rms(reference)) and max |diff|; a gate fails above its tolerance.
/// </summary>
internal static class Golden
{
    internal static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var a = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
        return a;
    }

    /// <summary>relRMS and max |diff| of <paramref name="got"/> against <paramref name="want"/>; prints one line.</summary>
    internal static bool Compare(string name, float[] want, float[] got, double relRmsTol)
    {
        if (want.Length != got.Length) { Console.WriteLine($"FAIL {name}: length {got.Length} != {want.Length}"); return false; }
        double se = 0, sw = 0, maxd = 0; int worst = 0; bool nan = false;
        for (int i = 0; i < want.Length; i++)
        {
            double d = got[i] - want[i];
            if (double.IsNaN(got[i]) || double.IsInfinity(got[i])) nan = true;
            se += d * d; sw += (double)want[i] * want[i];
            if (Math.Abs(d) > maxd) { maxd = Math.Abs(d); worst = i; }
        }
        double rel = Math.Sqrt(se / want.Length) / Math.Max(1e-30, Math.Sqrt(sw / want.Length));
        bool ok = !nan && rel <= relRmsTol;
        Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {name,-12} relRMS {rel:E3} (tol {relRmsTol:E1})  max|d| {maxd:E3} at {worst} (want {want[worst]:F5} got {got[worst]:F5}){(nan ? "  NaN/Inf!" : "")}");
        return ok;
    }

    /// <summary>
    /// Flash attention (bf16 tensor cores) against the fp32 reference kernels on random data, at sizes that are NOT
    /// tile multiples (Nq % 64 != 0, Nk % 64 != 0, Nk &lt; 64) and in both layouts the DiT uses (packed qkv, separate q + kv).
    /// </summary>
    public static int Attention()
    {
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        var k = new DiTKernels(accel);
        if (!k.UsesFlashAttention(128)) { Console.WriteLine("flash attention needs CUDA with tensor cores enabled"); return 2; }
        const int H = 12, D = 128, C = H * D;
        bool ok = true;
        foreach (var (nq, nk, packed) in new[] { (1000, 1029, false), (4096, 4096, true), (300, 37, false), (65, 64, true), (1, 5, false), (777, 1500, true) })
        {
            int qStride = packed ? 3 * C : C, kvStride = packed ? 3 * C : 2 * C;
            int kOff = packed ? C : 0, vOff = packed ? 2 * C : C;
            int tokens = packed ? Math.Max(nq, nk) : 0;
            var rng = new Random(nq * 31 + nk);
            float[] Rand(long n) { var a = new float[n]; for (long i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 4 - 2); return a; }
            // Q and KV as the model lays them out. In the packed layout q, k and v are slices of one buffer.
            var qHost = Rand((long)(packed ? tokens : nq) * qStride);
            var kvHost = packed ? qHost : Rand((long)nk * kvStride);
            using var qb = accel.Allocate1D(qHost);
            using var kvb = packed ? qb : accel.Allocate1D(kvHost);
            using var scores = accel.Allocate1D<float>((long)H * nq * nk);
            using var want = accel.Allocate1D<float>((long)nq * C);
            using var got = accel.Allocate1D<float>((long)nq * C);
            k.ReferenceAttention(qb.View, qStride, 0, kvb.View, kvStride, kOff, vOff, scores.View, want.View, nq, nk, H, D);
            k.Attention(qb.View, qStride, 0, kvb.View, kvStride, kOff, vOff, default, got.View, nq, nk, H, D);
            accel.Synchronize();
            ok &= Compare($"attn {nq}x{nk}{(packed ? " packed" : "")}", want.GetAsArray1D(), got.GetAsArray1D(), 8e-3);
        }
        Console.WriteLine(ok ? "ATTENTION PASS" : "ATTENTION FAIL");
        return ok ? 0 : 1;
    }

    internal static int[] ReadI32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var a = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
        return a;
    }

    /// <summary>The shape / texture SLat flow DiT on a sparse token set (voxel coordinates drive the RoPE).</summary>
    public static int SlatFlow(string kind)
    {
        string dir = Path.Combine(Gpu.WeightDir, "golden", "slat_" + kind);
        if (!Directory.Exists(dir)) { Console.WriteLine($"no golden at {dir}; run reference/golden_slat_flow.py {kind}"); return 2; }
        string ckpt = kind == "shape" ? "slat_flow_img2shape_dit_1_3B_512_bf16" : "slat_flow_imgshape2tex_dit_1_3B_512_bf16";
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        using var weights = GpuWeights.Load(accel, Path.Combine(Gpu.WeightDir, "TRELLIS.2-4B", "ckpts", ckpt));
        var feats = ReadF32(Path.Combine(dir, "feats.bin"));
        var coords = ReadI32(Path.Combine(dir, "coords.bin"));
        var cond = ReadF32(Path.Combine(dir, "cond.bin"));
        float t = 700f;
        int n = coords.Length / 3, condTokens = cond.Length / 1024;
        bool mma = DiTKernels.EnableTensorCores && accel.AcceleratorType == AcceleratorType.Cuda;
        double tol = mma ? 1e-2 : 2e-3;   // block taps: residual-stream relRMS
        // The output gate for the bf16 tensor-core path is the reference's OWN bf16 run: ours must not deviate from the fp32
        // golden by more than the released bf16 pipeline does (measured 2.8% shape / 4.2% tex on this input).
        double yTol = 2e-3;
        string bf16Path = Path.Combine(dir, "y_bf16.bin");
        if (mma)
        {
            if (!File.Exists(bf16Path)) { Console.WriteLine($"missing {bf16Path}; run reference/golden_slat_flow.py {kind} with DTYPE=bf16"); return 2; }
            var refBf16 = ReadF32(bf16Path); var refFp32 = ReadF32(Path.Combine(dir, "y.bin"));
            double se = 0, sw0 = 0;
            for (int i = 0; i < refFp32.Length; i++) { double d = refBf16[i] - refFp32[i]; se += d * d; sw0 += (double)refFp32[i] * refFp32[i]; }
            yTol = Math.Sqrt(se / sw0);
        }
        Console.WriteLine($"slat {kind}: {n} tokens, {condTokens} cond tokens, tap tolerance {tol:E1}, output tolerance relRMS {yTol:E2} ({(mma ? "the reference's own bf16 deviation" : "fp32 kernels")})");
        var kernels = new DiTKernels(accel);
        using var dit = new FlowDiT(accel, kernels, weights);
        bool ok = true;
        dit.Tap = (name, view, cnt) =>
        {
            string golden = Path.Combine(dir, name + ".bin");
            if (!File.Exists(golden)) return;      // this golden set did not dump that stage
            accel.Synchronize();
            ok &= Compare(name, ReadF32(golden), view.SubView(0, cnt).GetAsArray1D(), name == "t_emb" ? 1e-5 : tol);
        };
        using var cb = accel.Allocate1D(coords);
        dit.SetTokens(cb.View, n);
        using var fb = accel.Allocate1D(feats);
        using var cdb = accel.Allocate1D(cond);
        using var ob = accel.Allocate1D<float>((long)n * dit.OutChannels);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        dit.Forward(fb.View, t, cdb.View, condTokens, ob.View);
        Console.WriteLine($"forward {sw.Elapsed.TotalSeconds:F2}s");
        sw.Restart();
        dit.Forward(fb.View, t, cdb.View, condTokens, ob.View);
        Console.WriteLine($"forward (warm) {sw.Elapsed.TotalSeconds:F2}s");
        ok &= Compare("y", ReadF32(Path.Combine(dir, "y.bin")), ob.GetAsArray1D(), yTol);
        Console.WriteLine(ok ? $"SLAT {kind.ToUpperInvariant()} PASS" : $"SLAT {kind.ToUpperInvariant()} FAIL");
        return ok ? 0 : 1;
    }

    public static int SsFlow()
    {
        string dir = Path.Combine(Gpu.WeightDir, "golden", "ss_flow");
        if (!Directory.Exists(dir)) { Console.WriteLine($"no golden at {dir}; run scratchpad golden_ss_flow.py"); return 2; }
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var weights = GpuWeights.Load(accel, Path.Combine(Gpu.WeightDir, "TRELLIS.2-4B", "ckpts", "ss_flow_img_dit_1_3B_64_bf16"));
        Console.WriteLine($"weights {weights.Bytes / 1e9:F2} GB on device in {sw.Elapsed.TotalSeconds:F1}s");

        var x = ReadF32(Path.Combine(dir, "x.bin"));
        var cond = ReadF32(Path.Combine(dir, "cond.bin"));
        float t = ReadF32(Path.Combine(dir, "t.bin"))[0];
        int condTokens = cond.Length / 1024;

        // fp32 reference kernels: 2e-3. Tensor cores round GEMM and attention inputs to bf16 (what the reference model itself
        // does): measured 2.5e-3 at the output with bf16 GEMM alone, so the bf16 gate is 5e-3.
        double tol = DiTKernels.EnableTensorCores && accel.AcceleratorType == AcceleratorType.Cuda ? 5e-3 : 2e-3;
        Console.WriteLine($"tolerance relRMS {tol:E1} (tensor cores {(tol > 2e-3 ? "ON" : "off")})");
        var kernels = new DiTKernels(accel);
        using var flow = new SparseStructureFlow(accel, kernels, weights);
        bool ok = true;
        flow.Tap = (name, view, n) =>
        {
            accel.Synchronize();
            var got = view.SubView(0, n).GetAsArray1D();
            ok &= Compare(name, ReadF32(Path.Combine(dir, name + ".bin")), got, name == "t_emb" ? 1e-5 : tol);
        };
        using var xb = accel.Allocate1D(x);
        using var cb = accel.Allocate1D(cond);
        using var ob = accel.Allocate1D<float>(x.Length);
        sw.Restart();
        flow.Forward(xb.View, t, cb.View, condTokens, ob.View);
        Console.WriteLine($"forward {sw.Elapsed.TotalSeconds:F2}s (reference attention + naive linear)");
        ok &= Compare("y", ReadF32(Path.Combine(dir, "y.bin")), ob.GetAsArray1D(), tol);
        Console.WriteLine(ok ? "SSFLOW PASS" : "SSFLOW FAIL");
        return ok ? 0 : 1;
    }
}
