using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// LTX-Video 2B 0.9.8 distilled (Plans/ltx-video-2b-2026-10-22.md). Each stage is held to tensors dumped from the REFERENCE implementation (diffusers) on identical inputs
/// (tools/ltx-harness/reference: ref_t5_dump.py, ref_dit_dump.py, export_t5_bf16.py, export_dit_bf16.py). The fixtures and the bf16 exports live in LTX_DIR (default
/// F:\Projects\ltx: <c>t5_xxl_enc_bf16.safetensors</c>, <c>ltx_dit_bf16.safetensors</c>, <c>ref\*.npz</c>, <c>google\t5-v1_1-xxl\tokenizer.json</c>); a missing directory skips the test.
/// </summary>
public abstract partial class MLTestBase
{
    private static string LtxDir => Environment.GetEnvironmentVariable("LTX_DIR") ?? @"F:\Projects\ltx";
    private const float LtxFrameRate = 25f;

    /// <summary>The Unigram tokenizer of google/t5-v1_1-xxl against the reference's token ids for the three dumped prompts. Host only.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Ltx_T5Tokenizer_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        var tokPath = Path.Combine(LtxDir, "google", "t5-v1_1-xxl", "tokenizer.json");
        if (!File.Exists(tokPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var fs = File.OpenRead(tokPath);
        var tok = Preprocessing.UnigramTokenizer.FromTokenizerJson(fs);
        var report = new List<string>();
        foreach (var name in new[] { "0", "1", "2" })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"t5_{name}.npz"));
            var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask");
            int n = (int)mask.Sum();
            var textEntry = fx.GetEntry("text.npy")!; using var ts = textEntry.Open(); using var tms = new MemoryStream(); ts.CopyTo(tms);
            var raw = tms.ToArray(); int hlen = BitConverter.ToUInt16(raw, 8); var bytes = raw.AsSpan(10 + hlen);
            var sb = new System.Text.StringBuilder(); for (int i = 0; i + 3 < bytes.Length; i += 4) { int cp = BitConverter.ToInt32(bytes.Slice(i, 4)); if (cp == 0) break; sb.Append(char.ConvertFromUtf32(cp)); }
            var got = tok.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(sb.ToString()));
            var want = ids.Take(n).Select(v => (int)v).ToArray();
            bool same = got.SequenceEqual(want);
            report.Add($"prompt {name}: {n} reference tokens, ours {got.Length}, {(same ? "identical" : "DIFFERENT")}");
            if (!same) throw new Exception("tokenizer mismatch: " + string.Join("; ", report) + $" ours: [{string.Join(",", got.Take(40))}] reference: [{string.Join(",", want.Take(40))}]");
        }
        foreach (var line in report) Console.WriteLine("[Ltx] tokenizer vs reference: " + line);
        await Task.CompletedTask;
        return string.Join("; ", report);
    });

    /// <summary>
    /// The RoPE phase table (LTXVideoRotaryPosEmbed: full-width interleaved pairs, entry 3 * j + axis, float32 math) against the reference's cos / sin on the dumped grid. Host only.
    /// Red check: swap the axis order inside an entry (k % 3 -> 2 - k % 3) and the comparison fails.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Ltx_Rope_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        var fxPath = Path.Combine(LtxDir, "ref", "dit_t2v.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var fx = ZipFile.OpenRead(fxPath);
        var (noise, ns) = WanNpz(fx, "noise");                                          // [128, Fz, h, w]
        var (cos, cs) = WanNpz(fx, "rope_cos"); var (sin, _) = WanNpz(fx, "rope_sin");  // [N, 2048], each value repeated for the two members of a pair
        int Fz = ns[1], h = ns[2], w = ns[3], dim = cs[1];
        var table = LtxKernels.RopePhases(Fz, h, w, dim, 8f / LtxFrameRate, 32f, 32f);
        int pairs = dim / 2, N = Fz * h * w;
        var gotCos = new float[(long)N * pairs]; var gotSin = new float[(long)N * pairs]; var wantCos = new float[gotCos.Length]; var wantSin = new float[gotCos.Length];
        for (long i = 0; i < gotCos.Length; i++)
        {
            long n = i / pairs, p = i % pairs;
            gotCos[i] = table[(n * pairs + p) * 2]; gotSin[i] = table[(n * pairs + p) * 2 + 1];
            wantCos[i] = cos[n * dim + 2 * p]; wantSin[i] = sin[n * dim + 2 * p];
            if (cos[n * dim + 2 * p] != cos[n * dim + 2 * p + 1]) throw new Exception("the reference table does not repeat each value for both members of a pair");
        }
        var mc = WanCompare(gotCos, wantCos); var ms = WanCompare(gotSin, wantSin);
        Console.WriteLine($"[Ltx] rope vs reference: cos relRMS {mc.RelRms:E2} maxAbs {mc.MaxAbs:E2}; sin relRMS {ms.RelRms:E2} maxAbs {ms.MaxAbs:E2} ({N} tokens x {pairs} pairs)");
        if (!(mc.MaxAbs < 1e-3 && ms.MaxAbs < 1e-3)) throw new Exception($"the RoPE table deviates from the reference: cos maxAbs {mc.MaxAbs:E2}, sin maxAbs {ms.MaxAbs:E2}");
        await Task.CompletedTask;
        return $"cos maxAbs {mc.MaxAbs:E2}, sin maxAbs {ms.MaxAbs:E2}";
    });

    /// <summary>
    /// The T5-v1.1-XXL encoder (the umT5 encoder class with the shared position bias of block 0, streamed layer by layer) against the reference's bf16 encoder output for the three dumped
    /// prompts, fed the reference's own token ids. Desktop device.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_T5_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the layer-streaming encoder test reads the export from disk on a desktop GPU");
        var path = Path.Combine(LtxDir, "t5_xxl_enc_bf16.safetensors");
        if (!File.Exists(path)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var enc = await UmT5Encoder.CreateAsync(accelerator, path);
        var report = new List<string>();
        double worst = 0;
        foreach (var name in new[] { "0", "1", "2" })
        {
            using var fx = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"t5_{name}.npz"));
            var (ids, _) = WanNpz(fx, "ids"); var (mask, _) = WanNpz(fx, "mask"); var (want, ws) = WanNpz(fx, "context");     // [128, 4096]: the real rows first
            int n = (int)mask.Sum();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var got = await enc.EncodeAsync(ids.Take(n).Select(v => (int)v).ToArray());
            var m = WanCompare(got, want.AsSpan(0, n * 4096).ToArray());                        // the padding rows are masked out of the DiT's keys: only the real rows count
            report.Add($"prompt {name} ({n} tokens): relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}; {sw.Elapsed.TotalSeconds:F1} s");
            worst = Math.Max(worst, m.RelRms);
        }
        foreach (var line in report) Console.WriteLine("[Ltx] T5 vs reference: " + line);
        if (!(worst < 0.05)) throw new Exception("the T5 encoder deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The VAE decoder (fp16 weights, fp32 activations, replicate-padded 3x3x3 convolutions as gather GEMMs, timestep-conditioned resnets, residual upsamplers) against the reference's
    /// float32 decode of a small latent [128, 3, 4, 6] at decode timestep 0.05: every stage tap (conv_in, mid block, the three up stages, conv_out) and the final frames
    /// [3, 17, 128, 192]. Desktop GPU (the taps read back).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_VaeDecode_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the taps read back to the host: a desktop GPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_vae_f16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "vae_small.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var fx = ZipFile.OpenRead(fxPath);
        var (latent, ls) = WanNpz(fx, "latent"); var (tArr, _) = WanNpz(fx, "timestep");               // [128, Fz, h, w]
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var vae = new LtxVae(accelerator, weights);
        var taps = new Dictionary<string, float[]>();
        vae.Tap = (name, view, T, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)T * H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = a; };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var video = await vae.DecodeAsync(latent, ls[1], ls[2], ls[3], tArr[0]);
        double ms = sw.Elapsed.TotalMilliseconds;
        vae.Tap = null;
        var report = new List<string>();
        double worst = 0;
        foreach (var name in new[] { "conv_in", "mid_block", "up_0", "up_1", "up_2", "conv_out" })
        {
            var (rf, rs) = WanNpz(fx, name);                                                         // [C, T, H, W] channel-first
            int C = rs[0], vox = rs[1] * rs[2] * rs[3];
            var t = new float[rf.Length];
            for (int c = 0; c < C; c++) for (int p = 0; p < vox; p++) t[(long)p * C + c] = rf[(long)c * vox + p];
            var m = WanCompare(taps[name], t);
            report.Add($"{name} [{C}, {rs[1]}, {rs[2]}, {rs[3]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            worst = Math.Max(worst, m.RelRms);
        }
        var (want, vs) = WanNpz(fx, "video");                                                        // [3, 17, 128, 192]
        var mv = WanCompare(video, want);
        report.Add($"video [{vs[0]}, {vs[1]}, {vs[2]}, {vs[3]}]: relRMS {mv.RelRms:E2}, maxAbs {mv.MaxAbs:F3} of {mv.RefMaxAbs:F2}; {ms:F0} ms");
        worst = Math.Max(worst, mv.RelRms);
        foreach (var line in report) Console.WriteLine("[Ltx] VAE decode vs reference: " + line);
        if (!(worst < 0.02)) throw new Exception("the VAE decode deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });

    private static readonly string[] LtxDitTaps = { "time_embedding", "time_projection", "text_embedding", "patch_embedding", "block_0", "block_1", "block_14", "block_27", "head" };

    /// <summary>
    /// The DiT, one forward at step 0 of the dumped trajectory, against the reference's bf16 velocity prediction and every tapped intermediate (time embedding and projection, caption
    /// projection, patch projection, blocks 0 / 1 / 14 / 27, the head). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_DiT_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "dit_t2v.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var (lat, ls) = WanNpz(fx, "step0_latent_in"); var (tArr, _) = WanNpz(fx, "step0_t");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
        dit.SetGrid(ls[1], ls[2], ls[3], 8f / LtxFrameRate, 32f, 32f);
        dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
        var taps = new Dictionary<string, float[]>();
        dit.Tap = (name, view, count) => { accelerator.Synchronize(); var a = new float[count]; view.SubView(0, count).CopyToCPU(a); taps[name] = a; };
        using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        dit.Forward(lIn.View, tArr[0], 0, lOut.View);
        var got = await lOut.View.CopyToHostAsync();
        double ms = sw.Elapsed.TotalMilliseconds;
        dit.Tap = null;
        var report = new List<string>();
        foreach (var name in LtxDitTaps)
        {
            var (want, _) = WanNpz(fx, "step0_" + name);
            int count = taps[name].Length;
            var w2 = name == "text_embedding" ? want.AsSpan(0, count).ToArray() : want;          // the reference holds all 128 caption rows, ours the real ones
            var m = WanCompare(taps[name], w2);
            report.Add($"{name}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
        }
        var (wantPred, _) = WanNpz(fx, "step0_pred");                                               // [128, Fz, h, w], channel-major like ours
        var mp = WanCompare(got, wantPred);
        report.Add($"prediction [128, {ls[1]}, {ls[2]}, {ls[3]}]: relRMS {mp.RelRms:E2}, maxAbs {mp.MaxAbs:F3} of {mp.RefMaxAbs:F2}; first forward {ms:F0} ms");
        foreach (var line in report) Console.WriteLine("[Ltx] DiT vs reference: " + line);
        if (!(mp.RelRms < 0.05)) throw new Exception("the DiT deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The distilled sampler loop teacher-forced: at every one of the 7 steps our DiT gets the reference's own latent and our Euler update must reproduce the reference's next latent
    /// (the update is exact arithmetic on the DiT output, so the error is one bf16 DiT call). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_SamplerLoop_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "dit_t2v.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var (noise, ns) = WanNpz(fx, "noise"); var (sigmas, _) = WanNpz(fx, "sigmas"); var (timesteps, _) = WanNpz(fx, "timesteps");
        var ours = LtxSampler.WithTerminal(LtxSampler.DistilledFirstPass);
        for (int i = 0; i < ours.Length; i++)
            if (MathF.Abs(ours[i] - sigmas[i]) > 1e-6f) throw new Exception($"our sigma {i} ({ours[i]}) differs from the reference scheduler's ({sigmas[i]})");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
        dit.SetGrid(ns[1], ns[2], ns[3], 8f / LtxFrameRate, 32f, 32f);
        dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
        using var lIn = accelerator.Allocate1D<float>(noise.Length); using var lOut = accelerator.Allocate1D<float>(noise.Length);
        var free = (float[])noise.Clone();
        var report = new List<string>();
        double worstForced = 0;
        for (int s = 0; s < sigmas.Length - 1; s++)
        {
            var (refIn, _) = WanNpz(fx, $"step{s}_latent_in"); var (want, _) = WanNpz(fx, $"step{s}_latent_out");
            float t = LtxSampler.Timestep(ours[s]);
            if (MathF.Abs(t - timesteps[s]) > 1e-3f) throw new Exception($"our timestep {s} ({t}) differs from the reference's ({timesteps[s]})");
            lIn.View.CopyFromCPU(refIn);
            dit.Forward(lIn.View, t, 0, lOut.View); var v = await lOut.View.CopyToHostAsync();
            var forced = (float[])refIn.Clone(); LtxSampler.Step(forced, v, ours[s], ours[s + 1]);
            var mf = WanCompare(forced, want);
            lIn.View.CopyFromCPU(free);
            dit.Forward(lIn.View, t, 0, lOut.View); var vf = await lOut.View.CopyToHostAsync();
            LtxSampler.Step(free, vf, ours[s], ours[s + 1]);
            var mr = WanCompare(free, want);
            report.Add($"step {s} (sigma {ours[s]:F4} -> {ours[s + 1]:F4}): teacher-forced relRMS {mf.RelRms:E2}; free-running {mr.RelRms:E2}");
            worstForced = Math.Max(worstForced, mf.RelRms);
        }
        foreach (var line in report) Console.WriteLine("[Ltx] sampler loop vs reference: " + line);
        if (!(worstForced < 0.02)) throw new Exception("a teacher-forced sampler step deviates from the reference: " + string.Join("; ", report));
        return $"worst teacher-forced error {worstForced:E2} over {report.Count} steps";
    });
}
