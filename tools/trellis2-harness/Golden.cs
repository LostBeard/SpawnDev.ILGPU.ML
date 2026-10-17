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

        var kernels = new DiTKernels(accel);
        using var flow = new SparseStructureFlow(accel, kernels, weights);
        bool ok = true;
        flow.Tap = (name, view, n) =>
        {
            accel.Synchronize();
            var got = view.SubView(0, n).GetAsArray1D();
            ok &= Compare(name, ReadF32(Path.Combine(dir, name + ".bin")), got, name == "t_emb" ? 1e-5 : 2e-3);
        };
        using var xb = accel.Allocate1D(x);
        using var cb = accel.Allocate1D(cond);
        using var ob = accel.Allocate1D<float>(x.Length);
        sw.Restart();
        flow.Forward(xb.View, t, cb.View, condTokens, ob.View);
        Console.WriteLine($"forward {sw.Elapsed.TotalSeconds:F2}s (reference attention + naive linear)");
        ok &= Compare("y", ReadF32(Path.Combine(dir, "y.bin")), ob.GetAsArray1D(), 2e-3);
        Console.WriteLine(ok ? "SSFLOW PASS" : "SSFLOW FAIL");
        return ok ? 0 : 1;
    }
}
