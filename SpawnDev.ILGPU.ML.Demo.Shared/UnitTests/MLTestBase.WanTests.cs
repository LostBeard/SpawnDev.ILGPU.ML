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
    /// The whole pipeline end to end on a small clip (320x192, 9 frames, 8 steps, the 8-bit DiT): text through umT5, the DiT with guidance and UniPC, the VAE decode. No reference to compare
    /// against at this size and step count (the stages have their own parity tests); this proves they compose, that the video is finite, in range and not constant, and writes three frames
    /// as PNG to WAN_OUT (default WAN_DIR\out) to look at. Image to video is the second half: the first frame must come back close to the picture.
    /// </summary>
    private static int EnvInt(string name, int fallback) => int.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : fallback;

    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_GeneratesAVideo() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the desktop pipeline test needs a desktop GPU (the browser pipeline is separate)");
        var paths = Wan22Paths.InFolder(WanDir);
        if (!File.Exists(paths.Dit) || !File.Exists(paths.Vae) || !File.Exists(paths.TextEncoder) || !File.Exists(paths.Tokenizer)) throw new UnsupportedTestException($"Wan exports not found in {WanDir}");
        string outDir = Environment.GetEnvironmentVariable("WAN_OUT") ?? Path.Combine(WanDir, "out"); Directory.CreateDirectory(outDir);
        var pipe = new Wan22Pipeline(accelerator, paths);
        var stages = new List<string>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        string last = "";
        var progress = new Progress<WanProgress>(p => { if (p.Stage != last) { last = p.Stage; stages.Add($"{p.Stage} at {sw.Elapsed.TotalSeconds:F0} s"); } });
        var video = await pipe.GenerateAsync(new WanRequest
        {
            Prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly.", Width = EnvInt("WAN_W", 320), Height = EnvInt("WAN_H", 192), Frames = EnvInt("WAN_FRAMES", 9), Steps = EnvInt("WAN_STEPS", 8), Seed = 1,
        }, progress);
        void Check(WanVideo v, string tag)
        {
            if (v.Data.Length != 3L * v.Frames * v.Width * v.Height) throw new Exception($"{tag}: wrong shape {v.Frames}x{v.Width}x{v.Height}");
            double min = double.MaxValue, max = double.MinValue, sum = 0, sum2 = 0;
            foreach (var x in v.Data) { if (!float.IsFinite(x)) throw new Exception($"{tag}: a pixel is not finite"); min = Math.Min(min, x); max = Math.Max(max, x); sum += x; sum2 += (double)x * x; }
            double mean = sum / v.Data.Length, std = Math.Sqrt(sum2 / v.Data.Length - mean * mean);
            if (max > 1.0001 || min < -1.0001) throw new Exception($"{tag}: pixels outside [-1, 1] ({min}..{max})");
            if (!(std > 0.02)) throw new Exception($"{tag}: the video is constant (std {std:F4})");
            Console.WriteLine($"[Wan] pipeline {tag}: {v.Frames} frames {v.Width}x{v.Height}, pixel mean {mean:F3}, std {std:F3}");
            foreach (int f in new[] { 0, v.Frames / 2, v.Frames - 1 }) File.WriteAllBytes(Path.Combine(outDir, $"pipeline_{tag}_{f}.png"), Assets.AssetGenerator.EncodePng(v.GetFrameRgba(f), v.Width, v.Height));
        }
        Check(video, "t2v");
        Console.WriteLine($"[Wan] pipeline t2v stages: {string.Join(", ", stages)}; total {sw.Elapsed.TotalSeconds:F0} s");
        return $"t2v ok in {sw.Elapsed.TotalSeconds:F0} s";
    });

    /// <summary>
    /// The DiT from its 8-bit and 4-bit block exports (5.6 GB and 3.1 GB instead of 10 GB; <c>quantize_streaming.py</c>, the FLUX.2 block format, dequantised a matrix at a time into a shared
    /// scratch before the normal GEMM) against the reference's bf16 noise predictions on the same inputs, text-to-video and with a clean first frame. Prints the error per format so
    /// the cost of the quantisation is on record; the bound is loose for 4-bit (visible differences are expected).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_DiT_Quantized_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the DiT runs on a desktop GPU or WebGPU");
        var files = new[] { ("q8", "wan22_ti2v_5b_dit_q8.safetensors", 0.05), ("q4", "wan22_ti2v_5b_dit_q4.safetensors", 0.25) };
        if (!files.All(f => File.Exists(Path.Combine(WanDir, f.Item2)))) throw new UnsupportedTestException($"Wan quantised exports not found in {WanDir}");
        using var t50 = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_0.npz")); using var t5n = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_neg.npz"));
        var (ctx0, cs0) = WanNpz(t50, "context"); var (ctxN, csN) = WanNpz(t5n, "context");
        var report = new List<string>();
        foreach (var (tag, file, bound) in files)
        {
            long before = accelerator.MemorySize;
            using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(WanDir, file));
            using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
            foreach (var (mode, firstFrame) in new[] { ("t2v", false), ("i2v", true) })
            {
                using var fx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"dit_{mode}_640x352_f33.npz"));
                var (lat, ls) = WanNpz(fx, "step0_latent_in"); var (tArr, _) = WanNpz(fx, "step0_t");
                dit.SetGrid(ls[1], ls[2], ls[3]); dit.SetContext(0, ctx0, cs0[0]); dit.SetContext(1, ctxN, csN[0]);
                using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                dit.Forward(lIn.View, tArr[0], firstFrame, 0, lOut.View); var cond = await lOut.View.CopyToHostAsync();
                double ms = sw.Elapsed.TotalMilliseconds;
                dit.Forward(lIn.View, tArr[0], firstFrame, 1, lOut.View); var uncond = await lOut.View.CopyToHostAsync();
                var (nc, _) = WanNpz(fx, "step0_noise_cond"); var (nu, _) = WanNpz(fx, "step0_noise_uncond");
                var mc = WanCompare(cond, nc); var mu = WanCompare(uncond, nu);
                report.Add($"{tag} {mode}: noise_cond relRMS {mc.RelRms:E2}, noise_uncond {mu.RelRms:E2}; weights {weights.Bytes / 1e9:F2} GB on the device; first forward {ms:F0} ms");
                if (!(mc.RelRms < bound && mu.RelRms < bound)) throw new Exception($"the {tag} DiT deviates from the reference beyond its bound {bound}: " + string.Join("; ", report));
            }
        }
        foreach (var line in report) Console.WriteLine("[Wan] quantised DiT vs reference: " + line);
        return string.Join("; ", report);
    });

    /// <summary>
    /// The Wan 2.2 VAE ENCODER for the image-to-video conditioning (one picture, the first chunk of the reference's chunked encode): a random picture [3, 1, 128, 192] to the normalised
    /// latent mean [48, 1, 8, 12], every stage tapped (conv1, the four down-blocks with their stride-2 resamples and average-pooling shortcuts, the middle, the head) against the float32 reference.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_VaeEncode_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the taps read back to the host: a desktop GPU");
        var weightsPath = Path.Combine(WanDir, "wan22_vae_f16.safetensors"); var fxPath = Path.Combine(WanDir, "dump", "vae.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var fx = ZipFile.OpenRead(fxPath);
        var (img, ims) = WanNpz(fx, "img");                         // [3, 1, H, W]
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var vae = new WanVae(accelerator, weights);
        var taps = new Dictionary<string, float[]>();
        vae.Tap = (name, view, T, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)T * H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = a; };
        var mu = await vae.EncodeImageAsync(img, ims[2], ims[3]);
        vae.Tap = null;
        var report = new List<string>();
        double worst = 0;
        foreach (var name in new[] { "enc_conv1", "enc_down0", "enc_down1", "enc_down2", "enc_down3", "enc_middle", "enc_head" })
        {
            var (rf, rs) = WanNpz(fx, name);                      // [1, C, 1, H, W] channel-first
            int C = rs[1], HW = rs[3] * rs[4];
            var t = new float[rf.Length];
            for (int c = 0; c < C; c++) for (int p = 0; p < HW; p++) t[(long)p * C + c] = rf[(long)c * HW + p];
            var m = WanCompare(taps[name], t);
            report.Add($"{name} [{C}, 1, {rs[3]}, {rs[4]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            worst = Math.Max(worst, m.RelRms);
        }
        var (want, _) = WanNpz(fx, "mu");
        var mm = WanCompare(mu, want);
        report.Add($"mu [48, 1, 8, 12]: relRMS {mm.RelRms:E2}, maxAbs {mm.MaxAbs:F3} of {mm.RefMaxAbs:F2}");
        worst = Math.Max(worst, mm.RelRms);
        foreach (var line in report) Console.WriteLine("[Wan] VAE encode vs reference: " + line);
        if (!(worst < 0.05)) throw new Exception("the VAE encode deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });

    /// <summary>
    /// The denoising loop end to end, two ways. TEACHER-FORCED: at every step our DiT (conditional and negative prompt) and our UniPC sampler get the reference's own latent, so errors
    /// do not compound: each step must reproduce the reference's next latent to the level of one bf16 DiT call. FREE-RUNNING: our own trajectory from the reference's noise (and
    /// conditioning latent for image-to-video). The sampler is sensitive: ref_sensitivity.py measured the REFERENCE diverging from a copy of itself started 1.3e-3 away by 1.8e-3,
    /// 1.4e-2, 4.4e-2, 8.2e-2 over the four steps (3.6e-3, 4.8e-2, 2.3e-1, 4.4e-1 with a clean first frame), so the free-running bound is that sensitivity, not parity.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_SamplerLoop_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 10 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(WanDir, "wan22_ti2v_5b_dit_bf16.safetensors");
        if (!File.Exists(weightsPath)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var t50 = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_0.npz")); using var t5n = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_neg.npz"));
        var (ctx0, cs0) = WanNpz(t50, "context"); var (ctxN, csN) = WanNpz(t5n, "context");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
        var report = new List<string>();
        double worstForced = 0;
        // teacher-forced: the reference's bf16 noise per DiT call (3.5e-3 text-to-video, 1e-2 with a clean first frame) amplified by classifier-free guidance (x5) and by the UniPC history at the
        // low-noise steps (the last step is x0 = sample - sigma * prediction): MEASURED 3.1e-2 and 7.6e-2 after four steps
        var forcedBound = new Dictionary<string, double> { ["t2v"] = 0.06, ["i2v"] = 0.12 };
        var freeBound = new Dictionary<string, double[]> { ["t2v"] = new[] { 0.01, 0.06, 0.15, 0.3 }, ["i2v"] = new[] { 0.02, 0.15, 0.5, 0.8 } };     // about 2x the reference's own sensitivity
        foreach (var (tag, firstFrame) in new[] { ("t2v", false), ("i2v", true) })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"dit_{tag}_640x352_f33.npz"));
            var (noise, ns) = WanNpz(fx, "noise"); var (ts, _) = WanNpz(fx, "timesteps");
            int Fz = ns[1], h = ns[2], w = ns[3], per = h * w;
            dit.SetGrid(Fz, h, w);
            dit.SetContext(0, ctx0, cs0[0]); dit.SetContext(1, ctxN, csN[0]);
            float[] z0 = firstFrame ? WanNpz(fx, "z0").Data : Array.Empty<float>();
            var free = (float[])noise.Clone();
            if (firstFrame) for (int c = 0; c < 48; c++) Array.Copy(z0, (long)c * Fz * per, free, (long)c * Fz * per, per);           // latent = (1 - mask) * z0 + mask * latent: frame 0 is the picture
            var schedFree = new FlowUniPCScheduler(); schedFree.SetTimesteps(ts.Length, 5.0);
            var schedForced = new FlowUniPCScheduler(); schedForced.SetTimesteps(ts.Length, 5.0);
            using var lIn = accelerator.Allocate1D<float>(free.Length); using var lOut = accelerator.Allocate1D<float>(free.Length);
            async Task<float[]> Predict(float[] latent, int step)
            {
                lIn.View.CopyFromCPU(latent);
                dit.Forward(lIn.View, schedFree.Timesteps[step], firstFrame, 0, lOut.View); var nc = await lOut.View.CopyToHostAsync();
                dit.Forward(lIn.View, schedFree.Timesteps[step], firstFrame, 1, lOut.View); var nu = await lOut.View.CopyToHostAsync();
                var pred = new float[nc.Length];
                for (int i = 0; i < pred.Length; i++) pred[i] = nu[i] + 5.0f * (nc[i] - nu[i]);                                          // classifier-free guidance, scale 5
                return pred;
            }
            void Restore(float[] latent) { if (firstFrame) for (int c = 0; c < 48; c++) Array.Copy(z0, (long)c * Fz * per, latent, (long)c * Fz * per, per); }
            for (int s = 0; s < ts.Length; s++)
            {
                var (refIn, _) = WanNpz(fx, $"step{s}_latent_in"); var (want, _) = WanNpz(fx, $"step{s}_latent_out");
                // teacher-forced: the reference's own input, our DiT and sampler
                var forced = schedForced.Step(await Predict(refIn, s), refIn); Restore(forced);
                var mf = WanCompare(forced, want);
                // free-running: our own trajectory
                free = schedFree.Step(await Predict(free, s), free); Restore(free);
                var mr = WanCompare(free, want);
                report.Add($"{tag} step {s}: teacher-forced relRMS {mf.RelRms:E2}; free-running {mr.RelRms:E2} (bound {freeBound[tag][s]:E1})");
                worstForced = Math.Max(worstForced, mf.RelRms / forcedBound[tag]);
                if (mr.RelRms > freeBound[tag][s]) throw new Exception($"the free-running trajectory left the sampler's sensitivity band: {report[^1]}");
            }
        }
        foreach (var line in report) Console.WriteLine("[Wan] sampler loop vs reference: " + line);
        if (!(worstForced < 1.0)) throw new Exception("a teacher-forced sampler step deviates from the reference: " + string.Join("; ", report));
        return $"worst teacher-forced error {worstForced:P0} of its bound over {report.Count} steps";
    });

    /// <summary>
    /// The sampler at the production step count: the reference's 30-step text-to-video trajectory (dump2/dit_t2v30.npz, 640x352, 17 frames, ref_dit_dump.py --steps 30) replayed teacher-forced
    /// through our 8-bit DiT and UniPC, every step compared with the reference's latent. The four-step test above cannot see what the later steps of a long schedule do (the multistep history,
    /// the last-step order reduction, the noise level where the latents collapse onto the data).
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_SamplerLong_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(WanDir, "wan22_ti2v_5b_dit_q8.safetensors"); var fixture = Path.Combine(WanDir, "dump2", "dit_t2v30.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fixture)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir}");
        using var t50 = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_0.npz")); using var t5n = ZipFile.OpenRead(Path.Combine(WanDir, "dump", "t5_neg.npz"));
        var (ctx0, cs0) = WanNpz(t50, "context"); var (ctxN, csN) = WanNpz(t5n, "context");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new WanDiT(accelerator, new DiTKernels(accelerator), weights);
        using var fx = ZipFile.OpenRead(fixture);
        var (noise, ns) = WanNpz(fx, "noise"); var (ts, _) = WanNpz(fx, "timesteps");
        dit.SetGrid(ns[1], ns[2], ns[3]); dit.SetContext(0, ctx0, cs0[0]); dit.SetContext(1, ctxN, csN[0]);
        var sched = new FlowUniPCScheduler(); sched.SetTimesteps(ts.Length, 5.0);
        using var lIn = accelerator.Allocate1D<float>(noise.Length); using var lOut = accelerator.Allocate1D<float>(noise.Length);
        var lines = new List<string>(); double worst = 0;
        var schedFree = new FlowUniPCScheduler(); schedFree.SetTimesteps(ts.Length, 5.0);
        var free = (float[])noise.Clone();
        for (int s = 0; s < ts.Length; s++)
        {
            var (refIn, _) = WanNpz(fx, $"step{s}_latent_in"); var (want, _) = WanNpz(fx, $"step{s}_latent_out");
            lIn.View.CopyFromCPU(free);
            dit.Forward(lIn.View, schedFree.Timesteps[s], false, 0, lOut.View); var fc = await lOut.View.CopyToHostAsync();
            dit.Forward(lIn.View, schedFree.Timesteps[s], false, 1, lOut.View); var fu = await lOut.View.CopyToHostAsync();
            for (int i = 0; i < fc.Length; i++) fc[i] = fu[i] + 5.0f * (fc[i] - fu[i]);
            free = schedFree.Step(fc, free);
            var mfree = WanCompare(free, want);
            lIn.View.CopyFromCPU(refIn);
            dit.Forward(lIn.View, sched.Timesteps[s], false, 0, lOut.View); var nc = await lOut.View.CopyToHostAsync();
            dit.Forward(lIn.View, sched.Timesteps[s], false, 1, lOut.View); var nu = await lOut.View.CopyToHostAsync();
            for (int i = 0; i < nc.Length; i++) nc[i] = nu[i] + 5.0f * (nc[i] - nu[i]);
            var got = sched.Step(nc, refIn);
            var m = WanCompare(got, want); worst = Math.Max(worst, m.RelRms);
            lines.Add($"step {s} t={sched.Timesteps[s]:F0}: teacher-forced relRMS {m.RelRms:E2}; free-running {mfree.RelRms:E2}");
        }
        foreach (var l in lines) Console.WriteLine("[Wan] long sampler vs reference: " + l);
        if (!(worst < 0.1)) throw new Exception("a teacher-forced step of the 30-step trajectory deviates from the reference: " + string.Join("; ", lines));
        return $"worst relRMS {worst:E2} over {ts.Length} steps";
    });

    /// <summary>
    /// Decodes the reference's own final latent of the 30-step trajectory with our VAE and writes frames as PNG (WAN_OUT): what the model's output looks like when nothing of ours produced it,
    /// to judge whether the look of our pipeline's video (a fox in snow, oversaturated at 640x352) is the model's or ours.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_DecodeReferenceTrajectory() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU");
        string tag = Environment.GetEnvironmentVariable("WAN_REF_TAG") ?? "t2v30";
        var vaePath = Path.Combine(WanDir, "wan22_vae_f16.safetensors"); var fixture = Path.Combine(WanDir, "dump2", $"dit_{tag}.npz");
        if (!File.Exists(vaePath) || !File.Exists(fixture)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir}");
        string outDir = Environment.GetEnvironmentVariable("WAN_OUT") ?? Path.Combine(WanDir, "out"); Directory.CreateDirectory(outDir);
        using var fx = ZipFile.OpenRead(fixture);
        var (final, s) = WanNpz(fx, "step29_latent_out");
        int Fz = s[1], h = s[2], w = s[3];
        using var vw = GpuWeights.LoadFile(accelerator, vaePath); using var vae = new WanVae(accelerator, vw);
        var video = new WanVideo { Data = await vae.DecodeAsync(final, Fz, h, w), Width = w * 16, Height = h * 16, Frames = 4 * Fz - 3 };
        foreach (int f in new[] { 0, video.Frames / 2, video.Frames - 1 }) File.WriteAllBytes(Path.Combine(outDir, $"reference_{tag}_{f}.png"), Assets.AssetGenerator.EncodePng(video.GetFrameRgba(f), video.Width, video.Height));
        return $"{video.Frames} frames";
    });

    /// <summary>
    /// The whole pipeline against the reference's 30-step trajectory: the same prompt (t5_0), the same starting noise, 8-bit DiT, everything ours (umT5, DiT, UniPC, VAE); the video is compared
    /// with our decode of the reference's final latent. The free-running sampler alone ends 0.11 away (the reference's own sensitivity), so a bound of 0.5 on the decoded video separates
    /// "the same picture" from "a different one".
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Wan_Pipeline_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("desktop GPU");
        var paths = Wan22Paths.InFolder(WanDir); var fixture = Path.Combine(WanDir, "dump2", "dit_t2v30.npz");
        if (!File.Exists(paths.Dit) || !File.Exists(paths.Vae) || !File.Exists(paths.TextEncoder) || !File.Exists(fixture)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir}");
        string outDir = Environment.GetEnvironmentVariable("WAN_OUT") ?? Path.Combine(WanDir, "out"); Directory.CreateDirectory(outDir);
        const string prompt = "A red fox walking through fresh snow at golden hour, the camera follows it slowly, soft light, shallow depth of field.";       // t5_0's text
        using var fx = ZipFile.OpenRead(fixture);
        var (noise, ns) = WanNpz(fx, "noise"); var (final, _) = WanNpz(fx, "step29_latent_out");
        var pipe = new Wan22Pipeline(accelerator, paths);
        var got = await pipe.GenerateAsync(new WanRequest { Prompt = prompt, Width = ns[3] * 16, Height = ns[2] * 16, Frames = 4 * ns[1] - 3, Steps = 30, InitialNoise = noise });
        using var vw = GpuWeights.LoadFile(accelerator, paths.Vae); using var vae = new WanVae(accelerator, vw);
        var want = await vae.DecodeAsync(final, ns[1], ns[2], ns[3]);
        var m = WanCompare(got.Data, want);
        foreach (int f in new[] { 0, got.Frames / 2, got.Frames - 1 }) File.WriteAllBytes(Path.Combine(outDir, $"pipeline_ref_{f}.png"), Assets.AssetGenerator.EncodePng(got.GetFrameRgba(f), got.Width, got.Height));
        Console.WriteLine($"[Wan] pipeline vs reference trajectory: prompt '{prompt}', decoded video relRMS {m.RelRms:E2}");
        if (!(m.RelRms < 0.5)) throw new Exception($"the pipeline's video is not the reference's: relRMS {m.RelRms:E2}");
        return $"relRMS {m.RelRms:E2}";
    });

    /// <summary>
    /// The umT5-XXL encoder (streamed layer by layer, rows of the embedding table read from the file) against the reference's bf16 encoder output for the three dumped prompts,
    /// fed the reference's own token ids: relative RMS of the context [n, 4096] and the peak VRAM of the run (one layer's weights, not 11 GB). Desktop device.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_UmT5_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the layer-streaming encoder test reads the 11 GB export from disk on a desktop GPU");
        var path = Path.Combine(WanDir, "umt5_xxl_enc_bf16.safetensors");
        if (!File.Exists(path)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var enc = new UmT5Encoder(accelerator, path);
        var report = new List<string>();
        double worstBf16 = 0, worstExact = 0;
        foreach (var name in new[] { "0", "1", "neg" })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"t5_{name}.npz"));
            using var fxx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"t5x_{name}.npz"));      // the same tokens through the reference layers upcast to fp32 (exact math on the same bf16 weights)
            var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask"); var (want, ws) = WanNpz(fx, "context");
            int n = (int)mask.Sum();
            var taps = new Dictionary<string, float[]>();
            enc.Tap = (stage, view, count) => { accelerator.Synchronize(); var a = new float[count]; view.SubView(0, count).CopyToCPU(a); taps[stage] = a; };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var got = await enc.EncodeAsync(ids.Take(n).Select(v => (int)v).ToArray());
            enc.Tap = null;
            var mb = WanCompare(got, want);                                                         // the reference as it runs: bf16 activations
            var (exact, _) = WanNpz(fxx, "fp32_final"); var (bfx, _) = WanNpz(fxx, "bf16_final");
            var me = WanCompare(got, exact);                                                        // the exact math
            var floor = WanCompare(bfx, exact);                                                     // the reference's own rounding against the exact math
            var l0 = WanCompare(taps["layer_0"], WanNpz(fxx, "fp32_layer_0").Data); var l12 = WanCompare(taps["layer_12"], WanNpz(fxx, "fp32_layer_12").Data);
            report.Add($"prompt {name} ({n} tokens): vs the bf16 reference {mb.RelRms:E2}; vs exact math {me.RelRms:E2} (the bf16 reference itself is {floor.RelRms:E2} from exact); layer_0 {l0.RelRms:E1}, layer_12 {l12.RelRms:E1}; {sw.Elapsed.TotalSeconds:F1} s");
            worstBf16 = Math.Max(worstBf16, mb.RelRms); worstExact = Math.Max(worstExact, me.RelRms);
        }
        foreach (var line in report) Console.WriteLine("[Wan] umT5 vs reference: " + line);
        // vs the bf16 reference: its own bf16 rounding (2.2e-2 to 2.8e-2 from exact math on these prompts). vs exact math on the same weights: float rounding with the fp32 GEMM
        // (4e-6 MEASURED with TRELLIS2_MMA=0 and on OpenCL), 5e-3 to 7.5e-3 with the CUDA tensor cores (they round the activations to bf16, as the reference does)
        if (!(worstBf16 < 0.05 && worstExact < 0.015)) throw new Exception("umT5 deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The Unigram tokenizer against the reference's token ids (Hugging Face tokenizers, google/umt5-xxl) for the three dumped prompts: two English sentences and the 126-token
    /// Chinese default negative prompt (full-width punctuation, no spaces: every position is a segmentation choice). Host only; reads tokenizer.json from WAN_DIR.
    /// Red check: a tokenizer that ignores the Metaspace prefix produces different ids.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Wan_UnigramTokenizer_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm) throw new UnsupportedTestException("reads the reference fixtures from disk");
        var tokPath = Path.Combine(WanDir, "google", "umt5-xxl", "tokenizer.json");
        if (!File.Exists(tokPath)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var fs = File.OpenRead(tokPath);
        var tok = Preprocessing.UnigramTokenizer.FromTokenizerJson(fs);
        var report = new List<string>();
        foreach (var name in new[] { "0", "1", "neg" })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(WanDir, "dump", $"t5_{name}.npz"));
            var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask");
            int n = (int)mask.Sum();
            var textEntry = fx.GetEntry("text.npy")!; using var ts = textEntry.Open(); using var tms = new MemoryStream(); ts.CopyTo(tms);
            // a numpy unicode scalar: a header, then UCS-4 code points
            var raw = tms.ToArray(); int hlen = BitConverter.ToUInt16(raw, 8); var bytes = raw.AsSpan(10 + hlen);
            var sb = new System.Text.StringBuilder(); for (int i = 0; i + 3 < bytes.Length; i += 4) { int cp = BitConverter.ToInt32(bytes.Slice(i, 4)); if (cp == 0) break; sb.Append(char.ConvertFromUtf32(cp)); }
            var got = tok.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(sb.ToString()));
            var want = ids.Take(n).Select(v => (int)v).ToArray();
            bool same = got.SequenceEqual(want);
            report.Add($"prompt {name}: {n} reference tokens, ours {got.Length}, {(same ? "identical" : "DIFFERENT at " + Enumerable.Range(0, Math.Min(got.Length, want.Length)).FirstOrDefault(i => got[i] != want[i]))}");
            if (!same) throw new Exception("tokenizer mismatch: " + string.Join("; ", report) + $" ours: [{string.Join(",", got.Take(40))}] reference: [{string.Join(",", want.Take(40))}]");
        }
        foreach (var line in report) Console.WriteLine("[Wan] tokenizer vs reference: " + line);
        await Task.CompletedTask;
        return string.Join("; ", report);
    });

    /// <summary>
    /// The Wan 2.2 VAE decoder (chunked, causal, cached convolutions, fp16 weights) against the reference's float32 decode of the same small latent [48, 3, 8, 12] -> video
    /// [3, 9, 128, 192]: every stage tap (conv2, conv1, middle, the four up-blocks, the head) and the final clamped frames. Desktop device (the taps read back).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_VaeDecode_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the taps read back to the host: a desktop GPU");
        var weightsPath = Path.Combine(WanDir, "wan22_vae_f16.safetensors"); var fxPath = Path.Combine(WanDir, "dump", "vae.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"Wan fixtures not found in {WanDir} (set WAN_DIR)");
        using var fx = ZipFile.OpenRead(fxPath);
        var (z, zs) = WanNpz(fx, "z");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dec = new WanVae(accelerator, weights);
        var taps = new Dictionary<string, List<float[]>>();
        var dims = new Dictionary<string, (int C, int H, int W)>();
        dec.Tap = (name, view, T, H, W, C) =>
        {
            accelerator.Synchronize();
            var a = new float[(long)T * H * W * C]; view.SubView(0, a.Length).CopyToCPU(a);
            if (!taps.TryGetValue(name, out var l)) taps[name] = l = new List<float[]>();
            l.Add(a); dims[name] = (C, H, W);
        };
        var video = await dec.DecodeAsync(z, zs[1], zs[2], zs[3]);
        dec.Tap = null;
        var report = new List<string>();
        double worst = 0;
        foreach (var name in new[] { "dec_conv2", "dec_conv1", "dec_middle", "dec_up0", "dec_up1", "dec_up2", "dec_up3", "dec_head" })
        {
            var (rf, rs) = WanNpz(fx, name);                      // [1, C, T, H, W] channel-first
            int C = rs[1], T = rs[2], HW = rs[3] * rs[4];
            var ours = taps[name].SelectMany(x => x).ToArray();   // [T*HW, C] channel-last
            var t = new float[rf.Length];
            for (int c = 0; c < C; c++) for (int p = 0; p < T * HW; p++) t[(long)p * C + c] = rf[(long)c * T * HW + p];
            var m = WanCompare(ours, t);
            report.Add($"{name} [{C}, {T}, {rs[3]}, {rs[4]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            worst = Math.Max(worst, m.RelRms);
        }
        var (rv, _) = WanNpz(fx, "video");
        var mv = WanCompare(video, rv);
        report.Add($"video [3, 9, 128, 192]: relRMS {mv.RelRms:E2}, maxAbs {mv.MaxAbs:F3} of {mv.RefMaxAbs:F2}");
        foreach (var line in report) Console.WriteLine("[Wan] VAE decode vs reference: " + line);
        worst = Math.Max(worst, mv.RelRms);
        if (!(worst < 0.05)) throw new Exception("the VAE decode deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });

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
