using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Qwen-Image from ComfyUI's files (QI_DIR, default E:/spawndev-qwenimage, the Comfy-Org/Qwen-Image_ComfyUI split_files layout under comfy\split_files).
/// </summary>
public abstract partial class MLTestBase
{
    private static string Qi_Require(Accelerator accelerator, string relative)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("QI_DIR") ?? "E:/spawndev-qwenimage";
        var path = Path.Combine(dir, "comfy", "split_files", relative);
        if (!File.Exists(path)) throw new UnsupportedTestException($"{path} not found (QI_DIR)");
        return path;
    }

    /// <summary>
    /// Qwen-Image's VAE (<c>qwen_image_vae.safetensors</c>: the Wan 2.1 VAE, bf16, ComfyUI's names) loads through <see cref="WanComfyFormat.AdaptVaeAsync"/> into <see cref="WanVae"/>'s Wan 2.1
    /// mode (16 latent channels, 8x) and reproduces a natural picture through encode -> decode: the FLUX.2 reference's 512 x 512 cat. A sanity gate until ComfyUI's own
    /// encode / decode dumps pin the numbers; the control (the Wan 2.2 latent constants, 48 channels) cannot load at all, so the channel count is asserted instead.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task QwenImage_ComfyVae_RoundTrip() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("vae", "qwen_image_vae.safetensors"));
        var f2 = F2Dir() ?? throw new UnsupportedTestException("the test picture (F2_DIR golden/cat/image.npy) is not on this machine");
        var (image, _) = Ur_Npy<float>(Path.Combine(f2, "golden", "cat", "image.npy"));               // [3, 512, 512] in [-1, 1]
        using var w = GpuWeights.LoadFile(accelerator, path);
        Ur_Expect(WanComfyFormat.IsComfyVae(w), "ComfyUI's VAE layout");
        await WanComfyFormat.AdaptVaeAsync(accelerator, w);
        using var vae = new WanVae(accelerator, w);
        Ur_Expect(vae.IsWan21 && vae.LatentChannels == 16 && vae.SpatialRatio == 8, $"the Wan 2.1 VAE ({vae.IsWan21}, {vae.LatentChannels} channels, {vae.SpatialRatio}x)");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var z = await vae.EncodeImageAsync(image, 512, 512);
        double encS = sw.Elapsed.TotalSeconds; sw.Restart();
        var back = await vae.DecodeAsync(z, 1, 64, 64);
        double decS = sw.Elapsed.TotalSeconds;
        Ur_Expect(z.Length == 16 * 64 * 64 && back.Length == 3 * 512 * 512, $"shapes: latent {z.Length}, picture {back.Length}");
        double sq = 0; int n = 512 * 512;
        for (int i = 0; i < 3 * n; i++)
        {
            double a = Math.Clamp(Math.Round((image[i] / 2 + 0.5) * 255), 0, 255), b = Math.Clamp(Math.Round((back[i] / 2 + 0.5) * 255), 0, 255);
            sq += (a - b) * (a - b);
        }
        double psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(sq / (3.0 * n), 1e-9));
        double zMean = z.Average(), zStd = Math.Sqrt(z.Select(v => (v - zMean) * (v - zMean)).Average());
        var report = $"encode {encS:F1} s -> decode {decS:F1} s of the 512 px cat: {psnr:F1} dB PSNR; normalised latent [16, 64, 64] mean {zMean:F3} std {zStd:F3}";
        Console.WriteLine("[QwenImage] ComfyUI VAE: " + report);
        Ur_Expect(psnr > 33, report);                                         // MEASURED (RTX 4070, CUDA, 2026-11-14): 35.4 dB; QwenImage_ComfyVae_MatchesComfyUi pins the exact numbers
        return report;
    });


    private static string Qi_Dumps()
    {
        var d = Path.Combine(Environment.GetEnvironmentVariable("QI_DIR") ?? "E:/spawndev-qwenimage", "comfy", "comfyui-dumps");
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"SpawnGen's Qwen-Image dumps not found in {d}");
        return d;
    }

    /// <summary>
    /// ComfyUI's VAE Encode / Decode with qwen_image_vae.safetensors (SpawnGen's dumps, float32 CPU): picture A (256 x 192) encodes to ComfyUI's LATENT [16, 1, 24, 32] and that latent
    /// decodes to ComfyUI's picture. Whether LATENT holds the raw or the normalised latent is measured: both readings are reported, the losing one must not match.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task QwenImage_ComfyVae_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("vae", "qwen_image_vae.safetensors")); var d = Qi_Dumps();
        const int H = 192, W = 256, h = 24, w = 32;
        using var wts = GpuWeights.LoadFile(accelerator, path);
        await WanComfyFormat.AdaptVaeAsync(accelerator, wts);
        using var vae = new WanVae(accelerator, wts);
        var (mean, std) = await vae.GetLatentNormalizationAsync();
        var img = F2_Raw(Path.Combine(d, "f2_img_1x192x256x3.raw"), 3L * H * W);
        var wantLat = F2_Raw(Path.Combine(d, "qi_enc_256x192_samples_f32_1x16x1x24x32.raw"), 16L * h * w);
        var z = await vae.EncodeImageAsync(F2_NhwcToChw(img, H, W), H, W);                    // normalised
        var raw = new float[z.Length];
        for (int c = 0; c < 16; c++) for (int p = 0; p < h * w; p++) raw[c * h * w + p] = z[c * h * w + p] * std[c] + mean[c];
        double eRaw = F2_RelRms(raw, wantLat), eNorm = F2_RelRms(z, wantLat);
        // decode ComfyUI's latent (raw -> normalised for our decoder), compare NHWC [0, 1]
        var zIn = new float[wantLat.Length];
        for (int c = 0; c < 16; c++) for (int p = 0; p < h * w; p++) zIn[c * h * w + p] = (wantLat[c * h * w + p] - mean[c]) / std[c];
        var pic = await vae.DecodeAsync(zIn, 1, h, w);
        var nhwc = new float[3L * H * W];
        for (int c = 0; c < 3; c++) for (int p = 0; p < H * W; p++) nhwc[(long)p * 3 + c] = Math.Clamp((pic[(long)c * H * W + p] + 1f) / 2f, 0f, 1f);
        double eDec = F2_RelRms(nhwc, F2_Raw(Path.Combine(d, "qi_dec_of_enc_256x192_images_f32_1x192x256x3.raw"), 3L * H * W));
        var report = $"encode of A vs ComfyUI's LATENT: raw reading {eRaw:E2}, normalised reading {eNorm:E2}; decode of ComfyUI's latent {eDec:E2}";
        Console.WriteLine("[QwenImage] ComfyUI VAE: " + report);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): LATENT is the RAW latent: raw reading 1.75e-4, normalised 0.649; decode 5.02e-5
        Ur_Expect(eRaw < 5e-4 && eNorm > 0.1 && eDec < 2e-4, report);
        return report;
    });

    /// <summary>
    /// Qwen-Image's text conditioning: ComfyUI's qwen_2.5_vl_7b_fp8_scaled.safetensors (Qwen2.5-VL-7B's language model, 8-bit matrices kept 8-bit, no vision tower) through
    /// <see cref="Flux2.Flux2TextEncoder"/>, against ComfyUI's CLIPTextEncode (type qwen_image) rows: the template's ids must be ComfyUI's (46 for the prompt, 39 for the empty one), the rows
    /// after the first 34 are the conditioning (12 and 5). Which hidden state is measured: after layer 28 with the final norm, without it, and after layer 27; two must lose.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task QwenImage_TextEncoder_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")); var d = Qi_Dumps();
        var f2 = F2Dir() ?? throw new UnsupportedTestException("the Qwen tokenizer (F2_DIR klein4b/tokenizer) is not on this machine");
        var tok = new Flux2.Qwen2Tokenizer(File.ReadAllText(Path.Combine(f2, "klein4b", "tokenizer", "tokenizer.json")));
        const string Prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
        const string Suffix = "<|im_end|>\n<|im_start|>assistant\n";
        const int Drop = 34, Hd = 3584;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux2.Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 28, finalNorm: true);
        using var raw = new Flux2.Flux2TextEncoder(accelerator, w, new[] { 27, 28 }) { PreciseActivations = true };                  // the states after layers 27 and 28, no norm
        using var normed = new Flux2.Flux2TextEncoder(accelerator, w, new[] { 28 }, finalNorm: true) { PreciseActivations = true };  // after layer 28 and the final norm (Hugging Face's last hidden state)
        var report = new List<string> { $"28 layers + final norm, {w.Bytes / 1e9:F2} GB on the device, loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        var best = new Dictionary<string, double> { ["layer 28 + final norm"] = 0, ["layer 28, no norm"] = 0, ["layer 27"] = 0 };
        foreach (var (prompt, idsFile, dump, rows) in new[] { ("a red apple on a wooden table", "qi_ids_prompt.json", "qi_cond_0_cond_f32_1x12x3584.raw", 12), ("", "qi_ids_empty.json", "qi_cond_empty_0_cond_f32_1x5x3584.raw", 5) })
        {
            using var idsDoc = System.Text.Json.JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(d, idsFile)));
            var wantIds = idsDoc.RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var (ids, real) = tok.EncodePadded(prompt, p => Prefix + p + Suffix);
            Ur_Expect(ids.Take(real).SequenceEqual(wantIds), $"'{prompt}': our {real} ids differ from ComfyUI's {wantIds.Length}");
            Ur_Expect(real - Drop == rows, $"'{prompt}': {real} - {Drop} = {real - Drop} rows, ComfyUI kept {rows}");
            var want = F2_Raw(Path.Combine(d, dump), (long)rows * Hd);
            sw.Restart();
            var a = await raw.Encode(ids, real).SubView((long)Drop * 2 * Hd, (long)rows * 2 * Hd).CopyToHostAsync();
            var n = await normed.Encode(ids, real).SubView((long)Drop * Hd, (long)rows * Hd).CopyToHostAsync();
            var l27 = new float[rows * Hd]; var l28 = new float[rows * Hd];
            for (int r = 0; r < rows; r++) { Array.Copy(a, r * 2L * Hd, l27, r * (long)Hd, Hd); Array.Copy(a, r * 2L * Hd + Hd, l28, r * (long)Hd, Hd); }
            var readings = new Dictionary<string, double> { ["layer 28 + final norm"] = F2_RelRms(n, want), ["layer 28, no norm"] = F2_RelRms(l28, want), ["layer 27"] = F2_RelRms(l27, want) };
            foreach (var (key, v) in readings) best[key] = Math.Max(best[key], v);
            report.Add($"'{prompt}' ({real} ids = ComfyUI's, {rows} rows kept, {sw.Elapsed.TotalSeconds:F1} s): {string.Join(", ", readings.Select(kv => $"{kv.Key} {kv.Value:E2}"))}");
        }
        foreach (var line in report) Console.WriteLine("[QwenImage] text encoder vs ComfyUI: " + line);
        var win = best.OrderBy(kv => kv.Value).First();
        // MEASURED (RTX 4070, CUDA, 2026-11-14): layer 28 + final norm wins, 4.52e-5 / 6.95e-5 with the exact scaled decode and precise activations (folded scales: 3.09e-3 / 4.97e-3);
        // without the norm 1.15, after layer 27 1.95
        Ur_Expect(win.Key == "layer 28 + final norm" && win.Value < 2e-4 && best.Where(kv => kv.Key != win.Key).All(kv => kv.Value > 0.1), $"one reading must match ({win.Key} {win.Value:E2}) and the others lose: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// Qwen-Image's 20B transformer (qwen_image_fp8_e4m3fn.safetensors, 8-bit floats without scales; blocks past QI_RESIDENT_GB stream from the file) against ComfyUI's model calls
    /// of a 2-step 256 px run at cfg 4 (SpawnGen's dumps, float32 CPU): call 0 the prompt's 12 rows, call 1 the empty prompt's 5 rows, both at sigma 1. One open serves both lengths
    /// (per-call text length), and a third call with the 12 rows again must reproduce call 0 exactly. The time convention (sigma or 1 - sigma) and the output convention (x - sigma v,
    /// x + sigma v, or v itself) are measured; the losing readings are the controls.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task QwenImage_Transformer_MatchesComfyUiCalls() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("diffusion_models", "qwen_image_fp8_e4m3fn.safetensors")); var d = Qi_Dumps();
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("QI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 7.0;
        const int rows = 16, cols = 16, lat = 16 * 32 * 32;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var tr = await QwenImage.QwenImageTransformer.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 12, rows, cols, (long)(residentGb * 1e9));
        var report = new List<string> { $"weights: {tr.ResidentBlocks} of 60 blocks resident, {tr.ResidentBytes / 1e9:F2} GB on the device, {tr.StreamedBytesPerForward / 1e9:F2} GB streamed per call; opened in {sw.Elapsed.TotalSeconds:F1} s" };
        var k = new Flux1.Flux1Kernels(accelerator);
        using var x = accelerator.Allocate1D<float>(lat); using var zero = accelerator.Allocate1D<float>(lat); zero.MemSetToZero();
        using var tokens = accelerator.Allocate1D<float>(rows * cols * 64L); using var den = accelerator.Allocate1D<float>(lat);
        using var cond = accelerator.Allocate1D<float>(12L * 3584);
        double worst = 0, control = double.MaxValue; float[]? call0 = null; string? winner = null;
        foreach (var (c, n, rerun) in new[] { (0, 12, false), (1, 5, false), (0, 12, true) })
        {
            float sigma = F2_Raw(Path.Combine(d, $"qi_call_{c}_timestep_f32_1.raw"), 1)[0];
            var want = F2_Raw(Path.Combine(d, $"qi_call_{c}_output_f32_1x16x1x32x32.raw"), lat);
            x.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"qi_call_{c}_input_f32_1x16x1x32x32.raw"), lat));
            cond.View.SubView(0, (long)n * 3584).CopyFromCPU(F2_Raw(Path.Combine(d, $"qi_call_{c}_c_c_crossattn_f32_1x{n}x3584.raw"), (long)n * 3584));
            k.Patchify(x.View, tokens.View, rows, cols);
            var readings = new List<(string Name, double Rel)>();
            var times = rerun ? new[] { ("t = sigma", sigma) } : new[] { ("t = sigma", sigma), ("t = 1 - sigma", 1f - sigma) };
            foreach (var (tName, t) in times)
            {
                sw.Restart();
                await tr.ForwardAsync(tokens.View, cond.View, t, n);
                foreach (var (oName, src, factor) in new[] { ("x - sigma * v", x, sigma), ("x + sigma * v", x, -sigma), ("v", zero, -1f) })
                {
                    k.Denoised(src.View, tr.Output, den.View, factor, rows, cols);
                    var got = await den.View.CopyToHostAsync();
                    if (!rerun && c == 0 && tName == "t = sigma" && oName == "x - sigma * v") call0 = got;
                    if (rerun && oName == "x - sigma * v")
                    {
                        double drift = F2_RelRms(got, call0!);
                        report.Add($"call 0 again after the 5-row call: {drift:E2} from the first (must be 0)");
                        Ur_Expect(drift == 0, string.Join("; ", report));
                    }
                    readings.Add(($"{tName}, {oName}", F2_RelRms(got, want)));
                }
                report.Add($"call {c} ({n} rows, sigma {sigma:G5}) {tName}: forward {sw.Elapsed.TotalSeconds:F1} s");
            }
            if (rerun) continue;
            var best = readings.OrderBy(r => r.Rel).First();
            winner ??= best.Name;
            Ur_Expect(best.Name == winner, $"call {c}: {best.Name} won, call 0 had {winner}");
            worst = Math.Max(worst, best.Rel);
            control = Math.Min(control, readings.Where(r => r != best).Min(r => r.Rel));
            report.Add($"call {c}: best {best.Name} {best.Rel:E2}; others {string.Join(", ", readings.Where(r => r != best).Select(r => $"{r.Name} {r.Rel:E2}"))}");
        }
        foreach (var line in report) Console.WriteLine("[QwenImage] transformer: " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): t = sigma, x - sigma * v wins, 2.42e-4 / 2.90e-4 with the default precise activations (bf16 activations 4.32e-2 / 3.18e-2); the others 3.4 to 21
        Ur_Expect(worst < 1e-3 && control > 0.2 && winner == "t = sigma, x - sigma * v", string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// ComfyUI's whole 2-step run (KSampler euler, cfg 4, sigmas 1 -> 0.756 -> 0 of ModelSamplingAuraFlow 3.1) rebuilt on <see cref="QwenImage.QwenImageTransformer"/>: step 1 from the
    /// dumped call outputs (the gate above checks them), step 2 our own prompt and negative calls at sigma 0.756, then ComfyUI's Euler and CFG arithmetic; the result must be ComfyUI's
    /// final LATENT. That also measures the latent format: LATENT as the model-space latent itself, or as raw = model * std + mean (Wan 2.1's normalisation); the other must miss.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task QwenImage_TwoStepRun_LandsOnComfyUiLatent() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("diffusion_models", "qwen_image_fp8_e4m3fn.safetensors")); var d = Qi_Dumps();
        var vaePath = Qi_Require(accelerator, Path.Combine("vae", "qwen_image_vae.safetensors"));
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("QI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 7.0;
        const int rows = 16, cols = 16, lat = 16 * 32 * 32, hw = 32 * 32;
        const float Cfg = 4f;
        using var sigmasDoc = System.Text.Json.JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(d, "qi_sigmas_simple2_auraflow3.1.json")));
        var sig = sigmasDoc.RootElement.GetProperty("sigmas").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        Ur_Expect(sig.Length == 3 && sig[2] == 0f, $"a 2-step schedule ending at 0 ({string.Join(", ", sig)})");
        // step 1 (sigma 1): ComfyUI's own two calls; Euler: x1 = x0 + (s1 - s0) (x0 - den) / s0, den = uncond + cfg (cond - uncond)
        var x0 = F2_Raw(Path.Combine(d, "qi_call_0_input_f32_1x16x1x32x32.raw"), lat);
        var c0 = F2_Raw(Path.Combine(d, "qi_call_0_output_f32_1x16x1x32x32.raw"), lat); var u0 = F2_Raw(Path.Combine(d, "qi_call_1_output_f32_1x16x1x32x32.raw"), lat);
        var x1 = new float[lat];
        for (int i = 0; i < lat; i++) { float dn = u0[i] + Cfg * (c0[i] - u0[i]); x1[i] = x0[i] + (sig[1] - sig[0]) * (x0[i] - dn) / sig[0]; }
        // step 2 (sigma 0.756): our calls; the last Euler step to sigma 0 is the denoised estimate itself
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var tr = await QwenImage.QwenImageTransformer.OpenAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path), 12, rows, cols, (long)(residentGb * 1e9));
        var k = new Flux1.Flux1Kernels(accelerator);
        using var x = accelerator.Allocate1D(x1); using var tokens = accelerator.Allocate1D<float>(rows * cols * 64L); using var den = accelerator.Allocate1D<float>(lat);
        using var cond = accelerator.Allocate1D<float>(12L * 3584);
        k.Patchify(x.View, tokens.View, rows, cols);
        var outs = new List<float[]>();
        foreach (var (c, n) in new[] { (0, 12), (1, 5) })
        {
            cond.View.SubView(0, (long)n * 3584).CopyFromCPU(F2_Raw(Path.Combine(d, $"qi_call_{c}_c_c_crossattn_f32_1x{n}x3584.raw"), (long)n * 3584));
            await tr.ForwardAsync(tokens.View, cond.View, sig[1], n);
            k.Denoised(x.View, tr.Output, den.View, sig[1], rows, cols);
            outs.Add(await den.View.CopyToHostAsync());
        }
        var x2 = new float[lat];
        for (int i = 0; i < lat; i++) x2[i] = outs[1][i] + Cfg * (outs[0][i] - outs[1][i]);
        // the latent format
        using var vw = GpuWeights.LoadFile(accelerator, vaePath);
        await WanComfyFormat.AdaptVaeAsync(accelerator, vw);
        using var vae = new WanVae(accelerator, vw);
        var (mean, std) = await vae.GetLatentNormalizationAsync();
        var asRaw = new float[lat];
        for (int c = 0; c < 16; c++) for (int p = 0; p < hw; p++) asRaw[c * hw + p] = x2[c * hw + p] * std[c] + mean[c];
        var want = F2_Raw(Path.Combine(d, "qi_out_samples_f32_1x16x1x32x32.raw"), lat);
        double eSame = F2_RelRms(x2, want), eRaw = F2_RelRms(asRaw, want);
        var report = $"2-step euler cfg {Cfg} (sigmas {string.Join(" / ", sig.Select(s => s.ToString("G4")))}), our step-2 calls in {sw.Elapsed.TotalSeconds:F1} s: final latent vs ComfyUI's LATENT as model space {eSame:E2}, as raw = model * std + mean {eRaw:E2}";
        Console.WriteLine("[QwenImage] " + report);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): as raw = model * std + mean 5.54e-5, as the model space itself 0.61: the transformer works in the normalised latent space
        Ur_Expect(eRaw < 2e-4 && eSame > 0.1, report);
        return report;
    });
}
