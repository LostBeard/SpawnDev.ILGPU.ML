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
/// D:\Projects\ltx: <c>t5_xxl_enc_bf16.safetensors</c>, <c>ltx_dit_bf16.safetensors</c>, <c>ref\*.npz</c>, <c>google\t5-v1_1-xxl\tokenizer.json</c>); a missing directory skips the test.
/// </summary>
public abstract partial class MLTestBase
{
    private static string LtxDir => Environment.GetEnvironmentVariable("LTX_DIR") ?? @"D:\Projects\ltx";
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
        // The angles reach 15708 rad (10000 * pi / 2): ONE float32 ulp of a frequency is 1e-3 rad, so two float32 implementations (torch's pow and cos, ours) cannot agree closer than that.
        // The exact (double) table says which one is closer: ours must be no further from it than the reference is (plus the same float32 slack).
        int nf = dim / 6, pad = dim % 6;
        var truthCos = new float[gotCos.Length]; var truthSin = new float[gotCos.Length];
        double sT = 8.0 / LtxFrameRate;
        for (int n = 0, f = 0; f < Fz; f++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++, n++)
                    for (int p = 0; p < pairs; p++)
                    {
                        double a = 0;
                        if (p >= pad / 2)
                        {
                            int k = p - pad / 2, jf = k / 3, axis = k % 3;
                            double freq = Math.Pow(10000.0, jf / (double)(nf - 1)) * Math.PI / 2;
                            double g = axis == 0 ? f * sT / 20.0 : axis == 1 ? i * 32.0 / 2048.0 : j * 32.0 / 2048.0;
                            a = freq * (g * 2 - 1);
                        }
                        truthCos[(long)n * pairs + p] = p < pad / 2 ? 1f : (float)Math.Cos(a); truthSin[(long)n * pairs + p] = p < pad / 2 ? 0f : (float)Math.Sin(a);
                    }
        var tOurs = WanCompare(gotCos, truthCos); var tRef = WanCompare(wantCos, truthCos); var tOursS = WanCompare(gotSin, truthSin); var tRefS = WanCompare(wantSin, truthSin);
        var lines = new[]
        {
            $"vs the reference: cos relRMS {mc.RelRms:E2} maxAbs {mc.MaxAbs:E2}; sin relRMS {ms.RelRms:E2} maxAbs {ms.MaxAbs:E2} ({N} tokens x {pairs} pairs)",
            $"vs exact (double) math: ours cos maxAbs {tOurs.MaxAbs:E2} sin {tOursS.MaxAbs:E2}; the reference's cos maxAbs {tRef.MaxAbs:E2} sin {tRefS.MaxAbs:E2}",
        };
        LtxLog("Rope", lines);
        if (!(mc.MaxAbs < 4e-3 && ms.MaxAbs < 4e-3 && tOurs.MaxAbs <= tRef.MaxAbs * 1.5 + 1e-4 && tOursS.MaxAbs <= tRefS.MaxAbs * 1.5 + 1e-4))
            throw new Exception("the RoPE table deviates from the reference: " + string.Join("; ", lines));
        await Task.CompletedTask;
        return string.Join("; ", lines);
    });

    /// <summary>
    /// The T5-v1.1-XXL encoder (the umT5 encoder class with the shared position bias of block 0, streamed layer by layer) against the reference's bf16 encoder output for the three dumped
    /// prompts, fed the reference's own token ids. Desktop device.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_T5_MatchesReference() => await RunTest(accelerator => LtxT5Check(accelerator, "t5_xxl_enc_bf16.safetensors", 0.05));

    /// <summary>The same check on the 8-bit block export (5.0 GB instead of 8.9 GB): the quantisation's error on top of the bf16 rounding.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_T5_Quantized_MatchesReference() => await RunTest(accelerator => LtxT5Check(accelerator, "t5_xxl_enc_q8.safetensors", 0.07));

    private async Task<string> LtxT5Check(Accelerator accelerator, string file, double bound)
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the layer-streaming encoder test reads the export from disk on a desktop GPU");
        var path = Path.Combine(LtxDir, file);
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
        LtxLog("T5 " + file, report);
        if (!(worst < bound)) throw new Exception("the T5 encoder deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

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
        // the RGBA frames converted on the device (LtxRequest.RgbaFrames) against the host conversion of the same float frames
        var rgbaFrames = await vae.DecodeRgbaFramesAsync(latent, ls[1], ls[2], ls[3], tArr[0]);
        var asFloat = new Wan.WanVideo { Data = video, Width = vs[3], Height = vs[2], Frames = vs[1] };
        int rgbaWorst = 0; long rgbaOff = 0;
        for (int f = 0; f < rgbaFrames.Length; f++) { var hostRgba = asFloat.GetFrameRgba(f); for (int i = 0; i < hostRgba.Length; i++) { int d = Math.Abs(hostRgba[i] - rgbaFrames[f][i]); if (d > 0) rgbaOff++; rgbaWorst = Math.Max(rgbaWorst, d); } }
        report.Add($"video [{vs[0]}, {vs[1]}, {vs[2]}, {vs[3]}]: relRMS {mv.RelRms:E2}, maxAbs {mv.MaxAbs:F3} of {mv.RefMaxAbs:F2}; {ms:F0} ms");
        report.Add($"RGBA frames on the device vs the host conversion: {rgbaFrames.Length} frames, {rgbaOff} bytes differ, by at most {rgbaWorst}");
        if (rgbaWorst > 1 || rgbaFrames.Length != vs[1]) throw new Exception("the device RGBA frames differ from the host conversion: " + report[^1]);
        worst = Math.Max(worst, mv.RelRms);
        LtxLog("VaeDecode", report);
        if (!(worst < 0.02)) throw new Exception("the VAE decode deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });

    /// <summary>
    /// The VAE ENCODER (image-to-video and video conditioning: fp16 weights, fp32 activations, causal 3x3x3 convolutions as gather GEMMs, residual space-to-depth downsamplers) against the
    /// reference's float32 encode of a 256x192 picture (one frame: every temporal step works on a single frame) and a 17-frame 128x96 clip (the temporal downsamplers' first-frame
    /// prepend over real time): every stage tap (conv_in, the four down stages, the mid block, conv_out) and the normalised latent (the posterior mean). Desktop GPU (the taps read back).
    /// Fixtures: ref_vae_encode_dump.py --size 1x192x256 --tag img, --size 17x96x128 --seed 12 --tag vid; weights export_vae_encoder_f16.py.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_VaeEncode_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the taps read back to the host: a desktop GPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_vae_encoder_f16.safetensors");
        if (!File.Exists(weightsPath) || !File.Exists(Path.Combine(LtxDir, "ref", "vae_enc_img.npz"))) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var enc = new LtxVaeEncoder(accelerator, weights);
        var report = new List<string>();
        double worst = 0;
        foreach (var tag in new[] { "img", "vid", "img288" }.Where(f => File.Exists(Path.Combine(LtxDir, "ref", $"vae_enc_{f}.npz"))))      // img288: 192x288, a 6 x 9 latent
        {
            using var fx = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"vae_enc_{tag}.npz"));
            var (video, vs) = WanNpz(fx, "video");                                                   // [3, F, H, W]
            var taps = new Dictionary<string, float[]>();
            enc.Tap = (name, view, T, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)T * H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = a; };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var latent = await enc.EncodeAsync(video, vs[1], vs[2], vs[3]);
            double ms = sw.Elapsed.TotalMilliseconds;
            enc.Tap = null;
            foreach (var name in new[] { "conv_in", "down_0", "down_1", "down_2", "down_3", "mid_block", "conv_out" })
            {
                var (rf, rs) = WanNpz(fx, name);                                                     // [C, T, H, W] channel-first
                int C = rs[0], vox = rs[1] * rs[2] * rs[3];
                var t = new float[rf.Length];
                for (int c = 0; c < C; c++) for (int p = 0; p < vox; p++) t[(long)p * C + c] = rf[(long)c * vox + p];
                var m = WanCompare(taps[name], t);
                report.Add($"{tag} {name} [{C}, {rs[1]}, {rs[2]}, {rs[3]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
                worst = Math.Max(worst, m.RelRms);
            }
            var (want, ls) = WanNpz(fx, "latent");
            var ml = WanCompare(latent, want);
            report.Add($"{tag} latent [{ls[0]}, {ls[1]}, {ls[2]}, {ls[3]}]: relRMS {ml.RelRms:E2}, maxAbs {ml.MaxAbs:F3} of {ml.RefMaxAbs:F2}; {ms:F0} ms");
            worst = Math.Max(worst, ml.RelRms);
        }
        LtxLog("VaeEncode", report);
        if (!(worst < 0.02)) throw new Exception("the VAE encode deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count} comparisons";
    });

    /// <summary>
    /// The x2 spatial latent upscaler (fp16 weights, fp32 activations, zero-padded 3x3x3 and per-frame 3x3 convolutions as gather GEMMs, GroupNorm resblocks, 2x2 pixel shuffle) against
    /// diffusers' LTXLatentUpsamplePipeline in float32 on the CPU (ref_upsampler_dump.py; the input is the final latent of the image-to-video fixture, [128, 4, 10, 15]): every stage tap
    /// (on the denormalised latent) and the model-space output with AdaIN (factor 1, the recipe; it moves the output 0.38) and without, bound 5e-3. The source weights are float32 and
    /// the export fp16, which alone moves the reference 7.1e-4; so the port's own math is held to a reference with the weights rounded to fp16 (<c>--round-f16</c>, fixture
    /// <c>upsampler_f16w.npz</c>): bound 1e-4 without tensor cores (CUDA with them off, or OpenCL). Desktop GPU (the taps read back).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_LatentUpsampler_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.WebGPU or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the taps read back to the host: a desktop GPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_upsampler_f16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "upsampler.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var fx = ZipFile.OpenRead(fxPath);
        var (latent, ls) = WanNpz(fx, "latent");
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var up = new LtxLatentUpsampler(accelerator, weights);
        var taps = new Dictionary<string, float[]>();
        up.Tap = (name, view, T, H, W, C) => { accelerator.Synchronize(); var a = new float[(long)T * H * W * C]; view.SubView(0, a.Length).CopyToCPU(a); taps[name] = a; };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var got = await up.UpsampleAsync(latent, ls[1], ls[2], ls[3], 1f);
        double ms = sw.Elapsed.TotalMilliseconds;
        up.Tap = null;
        var got0 = await up.UpsampleAsync(latent, ls[1], ls[2], ls[3], 0f);
        var report = new List<string>();
        double exact = 0;
        var f16Path = Path.Combine(LtxDir, "ref", "upsampler_f16w.npz");
        if (!File.Exists(f16Path)) throw new UnsupportedTestException("fixture upsampler_f16w.npz missing (ref_upsampler_dump.py --round-f16)");
        {
            bool saved = DiTKernels.EnableTensorCores;
            try
            {
                DiTKernels.EnableTensorCores = false;
                using var upExact = new LtxLatentUpsampler(accelerator, weights);
                using var f16 = ZipFile.OpenRead(f16Path);
                var me = WanCompare(await upExact.UpsampleAsync(latent, ls[1], ls[2], ls[3], 1f), WanNpz(f16, "output").Data);
                report.Add($"fp32 math vs the reference with the weights rounded to fp16: output relRMS {me.RelRms:E2}, maxAbs {me.MaxAbs:F4}");
                exact = me.RelRms;
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        double worst = 0;
        foreach (var name in new[] { "initial", "res", "upsampled", "post", "final" })
        {
            var (rf, rs) = WanNpz(fx, name);                                                         // [C, T, H, W] channel-first
            int C = rs[0], vox = rs[1] * rs[2] * rs[3];
            var t = new float[rf.Length];
            for (int c = 0; c < C; c++) for (int p = 0; p < vox; p++) t[(long)p * C + c] = rf[(long)c * vox + p];
            var m = WanCompare(taps[name], t);
            report.Add($"{name} [{C}, {rs[1]}, {rs[2]}, {rs[3]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            worst = Math.Max(worst, m.RelRms);
        }
        foreach (var (name, data) in new[] { ("output", got), ("output_noadain", got0) })
        {
            var (want, ws) = WanNpz(fx, name);
            var m = WanCompare(data, want);
            report.Add($"{name} [{ws[0]}, {ws[1]}, {ws[2]}, {ws[3]}]: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            worst = Math.Max(worst, m.RelRms);
        }
        report.Add($"{ms:F0} ms");
        LtxLog("LatentUpsampler", report);
        if (!(worst < 5e-3 && exact < 1e-4)) throw new Exception("the latent upsampler deviates from the reference: " + string.Join("; ", report));
        return $"worst relRMS {worst:E2} over {report.Count - 1} comparisons";
    });

    /// <summary>
    /// The live-preview colours (<see cref="LatentPreview"/>: linear maps fitted on clips this library made, R^2 0.95 / 0.94) on a clip they were NOT fitted on (a balloon over an
    /// autumn valley, generated by each model and packed as latent + the mean colour of each latent voxel's pixels: LTX ref\preview_holdout.npz, Wan dump\preview_holdout.npz).
    /// R^2 per colour must be above 0.8, the preview finite. Host only. (A random latent is not a fair input: it is nothing the sampler produces.)
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task LatentPreview_Colours_PredictTheDecodedVideo() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.CPU) throw new UnsupportedTestException("host-only check: runs on the CPU lane");
        var cases = new[] { ("LTX", Path.Combine(LtxDir, "ref", "preview_holdout.npz"), LatentPreview.Ltx), ("Wan", Path.Combine(WanDir, "dump", "preview_holdout.npz"), LatentPreview.Wan) };
        var report = new List<string>();
        bool ok = true;
        foreach (var (name, path, lp) in cases)
        {
            if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found");
            using var fx = ZipFile.OpenRead(path);
            var (z, zs) = WanNpz(fx, "latent"); var (col, _) = WanNpz(fx, "colour");                 // [C, Fz, h, w], [3, Fz, h, w]
            int Fz = zs[1], h = zs[2], w = zs[3], hw = h * w;
            var sse = new double[3]; var sum = new double[3]; var sum2 = new double[3]; long n = 0;
            for (int f = 0; f < Fz; f++)
            {
                var rgba = lp.FrameRgba(z, Fz, h, w, f);
                for (int p = 0; p < hw; p++, n++)
                    for (int c = 0; c < 3; c++)
                    {
                        double target = col[((long)c * Fz + f) * hw + p], pred = rgba[p * 4 + c] / 127.5 - 1.0;
                        sse[c] += (pred - target) * (pred - target); sum[c] += target; sum2[c] += target * target;
                    }
            }
            var r2 = Enumerable.Range(0, 3).Select(c => 1 - sse[c] / (sum2[c] - sum[c] * sum[c] / n)).ToArray();
            report.Add($"{name} held-out clip: R^2 (R, G, B) {r2[0]:F3} / {r2[1]:F3} / {r2[2]:F3} over {n} voxels");
            ok &= r2.All(x => x > 0.8);
        }
        foreach (var l in report) Console.WriteLine("[Preview] " + l);
        try { var od = Path.Combine(LtxDir, "out"); Directory.CreateDirectory(od); File.AppendAllLines(Path.Combine(od, "ltx_tests.log"), report.Select(l => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [Preview] {l}")); } catch (IOException) { }
        await Task.CompletedTask;
        if (!ok) throw new Exception("the live-preview colours do not predict the decoded video: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    private static readonly string[] LtxDitTaps = { "time_embedding", "time_projection", "text_embedding", "patch_embedding", "block_0", "block_1", "block_14", "block_27", "head" };

    /// <summary>
    /// The DiT, one forward at step 0 of the dumped trajectory, against the reference's bf16 velocity prediction and every tapped intermediate (time embedding and projection, caption
    /// projection, patch projection, blocks 0 / 1 / 14 / 27, the head). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_DiT_MatchesReference() => await RunTest(accelerator => LtxDitCheck(accelerator, "t2v"));

    /// <summary>The same check at production size: 480x320, 121 frames = 2400 tokens (the first pass of a 704x480 clip), where the flash attention runs several query chunks (fixtures: ref_dit_dump.py --size 480x320 --frames 121 --tag big).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_DiT_ProductionScale_MatchesReference() => await RunTest(accelerator => LtxDitCheck(accelerator, "big"));

    private async Task<string> LtxDitCheck(Accelerator accelerator, string fixtureTag)
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}.npz");
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
        // two references on the same inputs: the exact math (float32 on the same bf16 weights) and the pipeline as it runs (bf16 activations, which is itself far from the exact math here)
        double[] worst = new double[2];
        foreach (var (idx, tag, file) in new[] { (0, "fp32 reference", $"dit_{fixtureTag}_fp32.npz"), (1, "bf16 reference", $"dit_{fixtureTag}.npz") })
        {
            var refPath = Path.Combine(LtxDir, "ref", file);
            if (!File.Exists(refPath)) { report.Add($"{tag}: fixture {file} missing"); worst[idx] = double.MaxValue; continue; }
            using var rx = ZipFile.OpenRead(refPath);
            foreach (var name in LtxDitTaps)
            {
                var (want, _) = WanNpz(rx, "step0_" + name);
                int count = taps[name].Length;
                var w2 = name == "text_embedding" ? want.AsSpan(0, count).ToArray() : want;          // the reference holds all 128 caption rows, ours the real ones
                var m = WanCompare(taps[name], w2);
                report.Add($"vs {tag} {name}: relRMS {m.RelRms:E2}, maxAbs {m.MaxAbs:F3} of {m.RefMaxAbs:F2}");
            }
            var (wantPred, _) = WanNpz(rx, "step0_pred");                                           // [128, Fz, h, w], channel-major like ours
            var mp = WanCompare(got, wantPred);
            report.Add($"vs {tag} prediction [128, {ls[1]}, {ls[2]}, {ls[3]}]: relRMS {mp.RelRms:E2}, maxAbs {mp.MaxAbs:F3} of {mp.RefMaxAbs:F2}; first forward {ms:F0} ms");
            worst[idx] = mp.RelRms;
        }
        // the bf16 pipeline against its own exact math, for the record
        if (File.Exists(Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}_fp32.npz")))
        {
            using var a32 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}_fp32.npz")); using var a16 = ZipFile.OpenRead(fxPath);
            var floor = WanCompare(WanNpz(a16, "step0_pred").Data, WanNpz(a32, "step0_pred").Data);
            report.Add($"the bf16 reference vs the fp32 reference (its own rounding): relRMS {floor.RelRms:E2}");
        }
        // The port itself, free of precision noise: the same forward with the bf16 tensor cores OFF (fp32 GEMMs) must sit on the exact math. The tensor-core forward above rounds
        // every activation to bf16 (as the reference does), which around the model's 2500-sized outlier activations in the middle blocks costs ~1.5e-2; a LOGIC error would not vanish here.
        double exactRel = double.NaN;
        if (accelerator.AcceleratorType == AcceleratorType.Cuda && File.Exists(Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}_fp32.npz")))
        {
            bool saved = DiTKernels.EnableTensorCores;
            try
            {
                DiTKernels.EnableTensorCores = false;
                using var dit2 = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
                dit2.SetGrid(ls[1], ls[2], ls[3], 8f / LtxFrameRate, 32f, 32f);
                dit2.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
                dit2.Forward(lIn.View, tArr[0], 0, lOut.View);
                var exact = await lOut.View.CopyToHostAsync();
                using var a32 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}_fp32.npz"));
                exactRel = WanCompare(exact, WanNpz(a32, "step0_pred").Data).RelRms;
                report.Add($"with the tensor cores off (fp32 GEMMs) vs the fp32 reference: prediction relRMS {exactRel:E2}");
            }
            finally { DiTKernels.EnableTensorCores = saved; }
        }
        LtxLog("DiT " + fixtureTag, report);
        if (!(worst[0] < 0.02 && worst[1] < 0.15) || !(double.IsNaN(exactRel) || exactRel < 2e-3)) throw new Exception("the DiT deviates from the reference: " + string.Join("; ", report));
        return string.Join("; ", report);
    }

    /// <summary>
    /// The 8-bit DiT export (2.0 GB instead of 3.6 GB; <c>quantize_streaming.py</c>, the FLUX.2 block format, dequantised a matrix at a time into a shared scratch) against the fp32
    /// reference's velocity prediction at production size, and the cost of the quantisation on record. Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_DiT_Quantized_MatchesReference() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_q8.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", "dit_big.npz"); var fx32Path = Path.Combine(LtxDir, "ref", "dit_big_fp32.npz");
        if (!File.Exists(weightsPath) || !File.Exists(fxPath) || !File.Exists(fx32Path)) throw new UnsupportedTestException($"LTX fixtures not found in {LtxDir} (set LTX_DIR)");
        using var t5 = ZipFile.OpenRead(Path.Combine(LtxDir, "ref", "t5_0.npz")); using var fx = ZipFile.OpenRead(fxPath); using var fx32 = ZipFile.OpenRead(fx32Path);
        var (ctx, _) = WanNpz(t5, "context"); var (mask, _) = WanNpz(t5, "mask"); int rows = (int)mask.Sum();
        var (lat, ls) = WanNpz(fx, "step0_latent_in"); var (tArr, _) = WanNpz(fx, "step0_t");
        long before = accelerator.MemorySize;
        using var weights = GpuWeights.LoadFile(accelerator, weightsPath);
        using var dit = new LtxDiT(accelerator, new DiTKernels(accelerator), weights);
        dit.SetGrid(ls[1], ls[2], ls[3], 8f / LtxFrameRate, 32f, 32f);
        dit.SetContext(0, ctx.AsSpan(0, rows * 4096).ToArray(), rows);
        using var lIn = accelerator.Allocate1D(lat); using var lOut = accelerator.Allocate1D<float>(lat.Length);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        dit.Forward(lIn.View, tArr[0], 0, lOut.View);
        var got = await lOut.View.CopyToHostAsync();
        double ms = sw.Elapsed.TotalMilliseconds;
        var m32 = WanCompare(got, WanNpz(fx32, "step0_pred").Data); var m16 = WanCompare(got, WanNpz(fx, "step0_pred").Data);
        var lines = new[] { $"q8 DiT prediction [128, {ls[1]}, {ls[2]}, {ls[3]}]: vs the fp32 reference relRMS {m32.RelRms:E2}, vs the bf16 reference {m16.RelRms:E2}; weights {weights.Bytes / 1e9:F2} GB on the device; first forward {ms:F0} ms" };
        LtxLog("DiT q8", lines);
        if (!(m32.RelRms < 0.06 && m16.RelRms < 0.15)) throw new Exception("the 8-bit DiT deviates from the reference beyond its bound: " + string.Join("; ", lines));
        return string.Join("; ", lines);
    });

    /// <summary>Print the lines and append them to LTX_DIR\out\ltx_tests.log (the PMT summary hides the console output of a passing desktop test).</summary>
    private static void LtxLog(string what, IEnumerable<string> lines)
    {
        var all = lines.ToList();
        try
        {
            var dir = Path.Combine(LtxDir, "out"); Directory.CreateDirectory(dir);
            File.AppendAllLines(Path.Combine(dir, "ltx_tests.log"), all.Select(l => $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} [{what}] {l}"));
        }
        catch (IOException) { }
        foreach (var l in all) Console.WriteLine($"[Ltx] {what}: {l}");
    }

    /// <summary>
    /// The distilled sampler loop teacher-forced: at every one of the 7 steps our DiT gets the reference's own latent and our Euler update must reproduce the reference's next latent
    /// (the update is exact arithmetic on the DiT output, so the error is one bf16 DiT call). Desktop GPU or WebGPU.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_SamplerLoop_MatchesReference() => await RunTest(accelerator => LtxSamplerCheck(accelerator, "t2v"));

    /// <summary>The sampler loop at production size (480x320, 121 frames, 2400 tokens).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Ltx_SamplerLoop_ProductionScale_MatchesReference() => await RunTest(accelerator => LtxSamplerCheck(accelerator, "big"));

    private async Task<string> LtxSamplerCheck(Accelerator accelerator, string fixtureTag)
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm or AcceleratorType.CPU or AcceleratorType.Velocity)
            throw new UnsupportedTestException("the 3.9 GB DiT runs on a desktop GPU or WebGPU");
        var weightsPath = Path.Combine(LtxDir, "ltx_dit_bf16.safetensors"); var fxPath = Path.Combine(LtxDir, "ref", $"dit_{fixtureTag}.npz");
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
        var report = new List<string>();
        double[] worstForced = new double[2];
        // against the exact math (float32 reference, the bound) and against the bf16 reference as it runs (recorded; the bf16 reference is itself far from the exact math at the low-noise steps)
        foreach (var (idx, tag, file) in new[] { (0, "fp32 reference", $"dit_{fixtureTag}_fp32.npz"), (1, "bf16 reference", $"dit_{fixtureTag}.npz") })
        {
            var refPath = Path.Combine(LtxDir, "ref", file);
            if (!File.Exists(refPath)) { report.Add($"{tag}: fixture {file} missing"); worstForced[idx] = double.MaxValue; continue; }
            using var rx = ZipFile.OpenRead(refPath);
            var free = (float[])noise.Clone();
            for (int s = 0; s < sigmas.Length - 1; s++)
            {
                var (refIn, _) = WanNpz(rx, $"step{s}_latent_in"); var (want, _) = WanNpz(rx, $"step{s}_latent_out");
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
                report.Add($"vs {tag} step {s} (sigma {ours[s]:F4} -> {ours[s + 1]:F4}): teacher-forced relRMS {mf.RelRms:E2}; free-running {mr.RelRms:E2}");
                worstForced[idx] = Math.Max(worstForced[idx], mf.RelRms);
            }
        }
        LtxLog("SamplerLoop " + fixtureTag, report);
        if (!(worstForced[0] < 0.01 && worstForced[1] < 0.15)) throw new Exception("a teacher-forced sampler step deviates from the reference: " + string.Join("; ", report));
        return $"worst teacher-forced error {worstForced[0]:E2} (fp32 reference), {worstForced[1]:E2} (bf16 reference) over {sigmas.Length - 1} steps";
    }
}
