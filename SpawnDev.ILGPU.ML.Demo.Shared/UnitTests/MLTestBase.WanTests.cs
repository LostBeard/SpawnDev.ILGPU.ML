using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Animation;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Wan 2.2 TI2V-5B (Plans/wan22-ti2v-5b-2026-10-22.md). Each stage is held to tensors dumped from the REFERENCE implementation on identical inputs
/// (tools/wan-harness/reference: ref_t5_dump.py, ref_dit_dump.py, export_dit_bf16.py). The fixtures and the bf16 DiT export live in WAN_DIR (default
/// F:\Projects\wan: <c>wan22_ti2v_5b_dit_bf16.safetensors</c>, <c>dump\*.npz</c>); a missing directory skips the test.
/// </summary>
public abstract partial class MLTestBase
{
    private static string WanDir => Environment.GetEnvironmentVariable("WAN_DIR") ?? @"F:\Projects\wan";

    /// <summary>One array of an .npz written by numpy.savez, as floats with its shape.</summary>
    private static (float[] Data, int[] Shape) WanNpz(ZipArchive zip, string name)
    {
        var e = zip.GetEntry(name + ".npy") ?? throw new InvalidDataException($"the fixture has no '{name}' array");
        using var s = e.Open(); using var ms = new MemoryStream(); s.CopyTo(ms);
        return MotionClip.ReadNpy(ms.ToArray());
    }

    private static (double RelRms, double MaxAbs, double RefMaxAbs) WanCompare(float[] got, float[] want, int offset = 0)
    {
        double se = 0, sr = 0, maxAbs = 0, refMax = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = got[offset + i] - want[i];
            se += d * d; sr += (double)want[i] * want[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d)); refMax = Math.Max(refMax, Math.Abs(want[i]));
        }
        return (Math.Sqrt(se / Math.Max(1e-30, sr)), maxAbs, refMax);
    }

    /// <summary>
    /// The UniPC flow sampler against the reference scheduler: set_timesteps(4, shift 5) gives the reference's integer timesteps and float32 sigmas exactly, and
    /// each step, fed the reference's own guidance-combined prediction and sample, returns the reference's next sample (predictor, corrector, the final order-1 step).
    /// Host math: no device involved. Red check: a wrong shift moves the sigmas, which the exact timestep comparison catches.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Wan_UniPC_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm) throw new UnsupportedTestException("reads the reference fixture from disk");
        var path = Path.Combine(WanDir, "dump", "dit_t2v_640x352_f33.npz");
        if (!File.Exists(path)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var fx = ZipFile.OpenRead(path);
        var (ts, _) = WanNpz(fx, "timesteps"); var (sg, _) = WanNpz(fx, "sigmas");
        var report = new List<string>();
        var results = new List<string>();
        // The reference calls the scheduler inside torch.autocast(bfloat16): its einsum (the UniP / UniC weights applied to the model-output differences) runs in bf16, and its
        // scalars are float32. With EmulateReference the port must agree with the reference to float rounding (that proves the algorithm, step for step); the double-precision
        // default is the more accurate arithmetic and must stay within the reference's own bf16 rounding (about 1e-4 relative on a latent).
        foreach (var (emulate, limit) in new[] { (true, 5e-6), (false, 3e-4) })
        {
            var sched = new FlowUniPCScheduler { EmulateReference = emulate };
            sched.SetTimesteps(ts.Length, 5.0);
            for (int i = 0; i < ts.Length; i++)
                if (sched.Timesteps[i] != (long)ts[i]) throw new Exception($"timestep {i}: {sched.Timesteps[i]} against the reference's {ts[i]}");
            for (int i = 0; i < sg.Length; i++)
                if (Math.Abs(sched.Sigmas[i] - sg[i]) > 1e-6f) throw new Exception($"sigma {i}: {sched.Sigmas[i]} against the reference's {sg[i]}");
            double worst = 0;
            for (int s = 0; s < ts.Length; s++)
            {
                var (lin, _) = WanNpz(fx, $"step{s}_latent_in"); var (nc, _) = WanNpz(fx, $"step{s}_noise_cond"); var (nu, _) = WanNpz(fx, $"step{s}_noise_uncond");
                var (lout, _) = WanNpz(fx, $"step{s}_latent_out");
                var pred = new float[nc.Length];
                for (int i = 0; i < pred.Length; i++) pred[i] = nu[i] + 5.0f * (nc[i] - nu[i]);            // classifier-free guidance, scale 5
                var got = sched.Step(pred, lin);
                var m = WanCompare(got, lout);
                report.Add($"{(emulate ? "reference arithmetic" : "double scalars")}, step {s}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:E2} of {m.RefMaxAbs:F2}");
                worst = Math.Max(worst, m.RelRms);
            }
            results.Add($"{(emulate ? "reference arithmetic" : "double scalars")} worst {worst:E2} (limit {limit:E0})");
            if (!(worst < limit)) { foreach (var line in report) Console.WriteLine("[Wan] UniPC vs reference: " + line); throw new Exception("UniPC deviates from the reference: " + string.Join("; ", report)); }
        }
        foreach (var line in report) Console.WriteLine("[Wan] UniPC vs reference: " + line);
        await Task.CompletedTask;
        return "4 steps; " + string.Join("; ", results);
    });

    /// <summary>
    /// The Wan DiT on the reference's step-0 inputs: the stage taps (patch embedding, time embedding and projection, text embedding, blocks 0 / 1 / 14 / 29, head)
    /// and both noise predictions (conditional and the default negative prompt), with and without a clean first latent frame (the TI2V image conditioning, per-token timesteps).
    /// The reference runs in bf16 (activations too); ours keeps fp32 activations with bf16 weights, so the comparison is a relative RMS against bf16 noise, printed per stage.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_DiT_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 10 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(WanDir, "wan22_ti2v_5b_dit_bf16.safetensors");
        if (!File.Exists(weightsPath)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");

        using var t50 = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_0.npz"));
        using var t5n = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_neg.npz"));
        var (ctx0, cs0) = WanNpz(t50, "context"); var (ctxN, csN) = WanNpz(t5n, "context");

        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
        var report = new List<string>();
        var worst = 0.0;
        foreach (var (tag, firstFrame) in new[] { ("t2v", false), ("i2v", true) })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"dit_{tag}_640x352_f33.npz"));
            var (lat, ls) = WanNpz(fx, "step0_latent_in");      // [48, Fz, h, w]
            int Fz = ls[1], h = ls[2], w = ls[3];
            var (tArr, _) = WanNpz(fx, "step0_t");
            dit.SetGrid(Fz, h, w);
            dit.SetContext(0, ctx0, cs0[0]);
            dit.SetContext(1, ctxN, csN[0]);
            int N = dit.TokenCount, C = dit.Dim;

            using var lIn = accelerator.Allocate1D(lat);
            using var lOut = accelerator.Allocate1D<float>(lat.Length);
            var taps = new Dictionary<string, float[]>();
            dit.Tap = (name, view, count) => { accelerator.Synchronize(); var a = new float[count]; view.SubView(0, count).CopyToCPU(a); taps[name] = a; };
            // CopyToCPU is not available on WebGPU: this test needs a desktop device for the taps
            dit.Forward(lIn.View, tArr[0], firstFrame, 0, lOut.View);
            dit.Tap = null;
            var cond = await lOut.View.CopyToHostAsync();
            dit.Forward(lIn.View, tArr[0], firstFrame, 1, lOut.View);
            var uncond = await lOut.View.CopyToHostAsync();

            void Check(string what, (double RelRms, double MaxAbs, double RefMaxAbs) m)
            {
                report.Add($"{tag} {what}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
                worst = Math.Max(worst, m.RelRms);
            }
            // taps: patch_embedding is the Conv3d output [1, C, F, H', W'] channel-first; the rest are token-major
            var (pe, _) = WanNpz(fx, "step0_patch_embedding");
            var peT = new float[N * C]; for (int c = 0; c < C; c++) for (int n = 0; n < N; n++) peT[n * C + c] = pe[c * N + n];
            Check("patch_embedding", WanCompare(taps["patch_embedding"], peT));
            // the reference's per-token tensors: row 0 is the clean first frame (timestep 0) in the i2v case; our taps are the step's own timestep, the LAST row
            var (te, _) = WanNpz(fx, "step0_time_embedding"); Check("time_embedding (row " + (firstFrame ? "last" : "0") + ")", WanCompare(taps["time_embedding"], te.AsSpan((firstFrame ? N - 1 : 0) * C, C).ToArray()));
            var (tp, _) = WanNpz(fx, "step0_time_projection"); Check("time_projection (row " + (firstFrame ? "last" : "0") + ")", WanCompare(taps["time_projection"], tp.AsSpan((firstFrame ? N - 1 : 0) * 6 * C, 6 * C).ToArray()));
            var (tx, _) = WanNpz(fx, "step0_text_embedding"); Check("text_embedding", WanCompare(taps["text_embedding"], tx));
            foreach (var b in new[] { 0, 1, 14, 29 })
            {
                var (bk, _) = WanNpz(fx, $"step0_block_{b}");
                Check($"block_{b}", WanCompare(taps[$"block_{b}"], bk));
            }
            var (hd, _) = WanNpz(fx, "step0_head"); Check("head", WanCompare(taps["head"], hd));
            var (nc, _) = WanNpz(fx, "step0_noise_cond"); Check("noise_cond", WanCompare(cond, nc));
            var (nu, _) = WanNpz(fx, "step0_noise_uncond"); Check("noise_uncond", WanCompare(uncond, nu));

            if (firstFrame)
            {
                // negative control: the same i2v data with the per-token timesteps ignored must be clearly worse, or this comparison could not see them
                dit.Forward(lIn.View, tArr[0], false, 0, lOut.View);
                var wrong = WanCompare(await lOut.View.CopyToHostAsync(), nc);
                var right = WanCompare(cond, nc);
                report.Add($"{tag} negative control (first frame treated as a normal frame): noise_cond relRMS {wrong.RelRms:E2} (correct run {right.RelRms:E2})");
                if (!(wrong.RelRms > 2 * right.RelRms && wrong.RelRms > 0.02)) throw new Exception("the comparison cannot tell a missing per-token timestep from the real thing: " + report[^1]);
            }
        }
        foreach (var line in report) Console.WriteLine("[Wan] DiT vs reference (bf16 reference activations): " + line);   // one console line per comparison (PMT_CONSOLE_LOG=Wan)
        // the reference's own activations are bf16 (about 1e-2 relative on a 30-block stack): anything above 5e-2 is a real difference
        if (!(worst < 0.05)) throw new Exception("Wan DiT deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });
}
