using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.1's autoencoder from ComfyUI's <c>ae.safetensors</c> (Comfy-Org/z_image_turbo split_files/vae; Black Forest Labs' FLUX.1 AE, Apache-2.0: shared by the FLUX.1 and Z-Image
/// templates) on the native LDM VAE (<see cref="SdVae"/>): 16 latent channels and no quant convolutions. Files in F1_DIR (default E:/spawndev-flux1) \comfy. Desktop GPU lanes.
/// </summary>
public abstract partial class MLTestBase
{
    private static string F1_RequireAe(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F1_DIR") ?? "E:/spawndev-flux1";
        if (!File.Exists(Path.Combine(dir, "comfy", "ae.safetensors"))) throw new UnsupportedTestException($"ComfyUI's ae.safetensors not found in {dir}\\comfy (F1_DIR)");
        return dir;
    }

    /// <summary>
    /// The autoencoder loads (16 latent channels, 4 levels, encoder and decoder, no quant convolutions) and reproduces a natural picture through encode -> decode: the FLUX.2
    /// reference's 512 x 512 cat (F2_DIR golden). A sanity gate until ComfyUI's own encode / decode dumps pin the numbers.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux1_ComfyAe_RoundTrip() => await RunTest(async accelerator =>
    {
        var dir = F1_RequireAe(accelerator);
        var f2 = F2Dir() ?? throw new UnsupportedTestException("the test picture (F2_DIR golden/cat/image.npy) is not on this machine");
        var (image, _) = Ur_Npy<float>(Path.Combine(f2, "golden", "cat", "image.npy"));               // [3, 512, 512] in [-1, 1]
        using var vae = await SdVae.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", "ae.safetensors");
        Ur_Expect(vae.LatentChannels == 16 && vae.Levels == 4 && vae.HasEncoder && vae.HasDecoder, $"16 latent channels, 4 levels, encoder and decoder ({vae.LatentChannels}, {vae.Levels}, {vae.HasEncoder}, {vae.HasDecoder})");
        Ur_Expect(!vae.Weights.Has("quant_conv.weight") && !vae.Weights.Has("post_quant_conv.weight"), "FLUX.1's autoencoder has no quant convolutions");
        // the file is fp32: every conv weight must carry its low fp16 half (SdWeights.KeepLowHalves), and the halves must hold the part fp16 dropped (not zeros)
        var lo = vae.Weights.Lo("encoder.down.2.block.0.conv1.weight") ?? throw new Exception("no low half kept for an fp32 conv weight");
        var loHost = await lo.CopyToHostAsync();
        double loMax = 0; int loNonZero = 0;
        foreach (var v in loHost) { double a = Math.Abs((float)v); if (a > 0) loNonZero++; loMax = Math.Max(loMax, a); }
        Console.WriteLine($"[Flux1] ComfyUI ae: low halves of encoder.down.2.block.0.conv1.weight: {loNonZero} of {loHost.Length} non-zero, max |lo| {loMax:E2}");
        Ur_Expect(loNonZero > loHost.Length / 2, $"the low halves are mostly zero ({loNonZero} of {loHost.Length} non-zero): the fp32 -> fp16 -> fp32 round trip did not round on this backend");
        var z = await vae.EncodeAsync(image, 512, 512);
        var back = await vae.DecodeAsync(z, 64, 64);
        double sq = 0; int n = 512 * 512;
        for (int i = 0; i < 3 * n; i++)
        {
            double a = Math.Clamp(Math.Round((image[i] / 2 + 0.5) * 255), 0, 255), b = Math.Clamp(Math.Round((back[i] / 2 + 0.5) * 255), 0, 255);
            sq += (a - b) * (a - b);
        }
        double psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(sq / (3.0 * n), 1e-9));
        double zMean = z.Average(), zStd = Math.Sqrt(z.Select(v => (v - zMean) * (v - zMean)).Average());
        var report = $"encode -> decode of the 512 px cat: {psnr:F1} dB PSNR; latent [16, 64, 64] mean {zMean:F3} std {zStd:F3}";
        Console.WriteLine("[Flux1] ComfyUI ae: " + report);
        Ur_Expect(psnr > 30, report);                                         // a 16-channel autoencoder round trip; ComfyUI's dumps (below) pin the exact numbers
        return report;
    });

    /// <summary>
    /// The encoder's stage taps (channel-last float32) for picture A through ComfyUI's ae.safetensors, written to F1_DIR\comfy\taps\&lt;backend&gt;: compared on the host with SpawnGen's
    /// ComfyUI encoder dumps (_DevComms 2026-11-14-0700 ...-ae-encoder-spots) to find where a few latent cells of A and C part from ComfyUI's.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux1_ComfyAe_DumpEncoderTaps() => await RunTest(async accelerator =>
    {
        var dir = F1_RequireAe(accelerator); var d = Path.Combine(dir, "comfy", "comfyui-dumps");
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"ComfyUI's measured ae dumps not found in {d}");
        using var vae = await SdVae.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", "ae.safetensors");
        var outDir = Path.Combine(dir, "comfy", "taps", accelerator.AcceleratorType.ToString());
        Directory.CreateDirectory(outDir);
        static void Write(string path, float[] a) { var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); File.WriteAllBytes(path, b); }
        var names = new List<string>();
        vae.Tap = (name, view, h, w, c) =>
        {
            accelerator.Synchronize();
            var a = new float[(long)h * w * c]; view.SubView(0, a.Length).CopyToCPU(a);
            Write(Path.Combine(outDir, $"tap_{name}_{h}x{w}x{c}.raw"), a); names.Add(name);
        };
        var img = F2_Raw(Path.Combine(d, "f2_img_1x192x256x3.raw"), 192L * 256 * 3);
        var z = await vae.EncodeAsync(F2_NhwcToChw(img, 192, 256), 192, 256);
        vae.Tap = null;
        Write(Path.Combine(outDir, "mean_16x24x32.raw"), z);
        var report = $"{names.Count} encoder taps ({string.Join(", ", names)}) + the mean -> {outDir}";
        Console.WriteLine("[Flux1] " + report);
        Ur_Expect(names.Contains("enc_conv_out") && names.Contains("enc_mid_attn_1"), report);
        return report;
    });

    /// <summary>
    /// The FLUX.1 dev transformer from ComfyUI's all-in-one <c>flux1-dev-fp8.safetensors</c> (8-bit floats, no scales) against ComfyUI's own two model calls of a 2-step 256 px run
    /// (SpawnGen's dumps, F1_DIR\comfy\comfyui-dumps\dev: ComfyUI ran the transformer in float32 on the CPU): the same T5 conditioning [256, 4096], pooled CLIP-L [768], guidance
    /// 3.5, model-space input [16, 32, 32] and sigma give the denoised estimate x - sigma * v. Blocks beyond F1_RESIDENT_GB (default 7) are streamed from the file every call.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux1_Transformer_MatchesComfyUiCalls() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F1_DIR") ?? "E:/spawndev-flux1";
        string comfy = Path.Combine(dir, "comfy"), d = Path.Combine(comfy, "comfyui-dumps", "dev"), file = "flux1-dev-fp8.safetensors";
        if (!File.Exists(Path.Combine(comfy, file)) || !Directory.Exists(d)) throw new UnsupportedTestException($"needs {comfy}\\{file} (Comfy-Org/flux1-dev) and SpawnGen's dumps in {d}");
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("F1_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 7.0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux1.Flux1Weights.OpenAsync(accelerator, new LocalFolderModelSource(comfy), "", file, (long)(residentGb * 1e9));
        double loadS = sw.Elapsed.TotalSeconds;
        const int T = 256, rows = 16, cols = 16;
        using var tr = new Flux1.Flux1Transformer(accelerator, w, T, rows * cols);
        tr.SetPositions(rows, cols);
        var k = new Flux1.Flux1Kernels(accelerator);
        using var txt = accelerator.Allocate1D(F2_Raw(Path.Combine(d, "flux1_cond_0_cond_f32_1x256x4096.raw"), 256L * 4096));
        using var pooled = accelerator.Allocate1D(F2_Raw(Path.Combine(d, "flux1_call_0_c_y_f32_1x768.raw"), 768));
        using var x = accelerator.Allocate1D<float>(16L * 32 * 32); using var tokens = accelerator.Allocate1D<float>(256L * 64); using var den = accelerator.Allocate1D<float>(16L * 32 * 32);
        var report = new List<string> { $"weights: {w.ResidentBlocks} of 57 blocks resident, {w.ResidentBytes / 1e9:F2} GB on the device, {w.StreamedBytesPerForward / 1e9:F2} GB streamed per call; opened in {loadS:F1} s" };
        double worst = 0, control = 0;
        for (int c = 0; c < 2; c++)
        {
            float sigma = F2_Raw(Path.Combine(d, $"flux1_call_{c}_timestep_f32_1.raw"), 1)[0];
            x.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"flux1_call_{c}_input_f32_1x16x32x32.raw"), 16L * 32 * 32));
            k.Patchify(x.View, tokens.View, rows, cols);
            sw.Restart();
            await tr.ForwardAsync(tokens.View, txt.View, pooled.View, sigma, 3.5f);
            k.Denoised(x.View, tr.Output, den.View, sigma, rows, cols);
            var got = await den.View.CopyToHostAsync();
            double rel = F2_RelRms(got, F2_Raw(Path.Combine(d, $"flux1_call_{c}_output_f32_1x16x32x32.raw"), 16L * 32 * 32));
            worst = Math.Max(worst, rel);
            report.Add($"call {c} (sigma {sigma:G6}): denoised vs ComfyUI {rel:E2}, {sw.Elapsed.TotalSeconds:F1} s");
            if (c == 1)
            {
                // control: the same call with guidance 1 instead of ComfyUI's 3.5 must not match (the guidance embedder is read)
                await tr.ForwardAsync(tokens.View, txt.View, pooled.View, sigma, 1f);
                k.Denoised(x.View, tr.Output, den.View, sigma, rows, cols);
                control = F2_RelRms(await den.View.CopyToHostAsync(), F2_Raw(Path.Combine(d, $"flux1_call_{c}_output_f32_1x16x32x32.raw"), 16L * 32 * 32));
                report.Add($"control, call {c} at guidance 1: {control:E2}");
            }
        }
        foreach (var line in report) Console.WriteLine("[Flux1] transformer: " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): 3.32e-5 and 5.49e-5 with the default precise activations (bf16 activations: 4.71e-3 / 5.34e-3; RX 590 OpenCL fp32: 4.7e-6 / 8.5e-6, SpawnGen)
        Ur_Expect(worst < 2e-4, string.Join("; ", report));
        Ur_Expect(control > 4 * worst, "the guidance control must not match: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The T5-XXL inside ComfyUI's all-in-one <c>flux1-dev-fp8.safetensors</c> (under <c>text_encoders.t5xxl.transformer.</c>, plain 8-bit floats, the embedding table too) through
    /// <see cref="Wan.UmT5Encoder"/> against ComfyUI's CLIPTextEncode conditioning (SpawnGen's dumps): [256, 4096], the prompt's tokens and the end token padded with the pad token (0)
    /// to 256 rows. The real-token rows are reported apart from the padding rows.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux1_T5FromCheckpoint_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F1_DIR") ?? "E:/spawndev-flux1";
        string comfy = Path.Combine(dir, "comfy"), d = Path.Combine(comfy, "comfyui-dumps", "dev"), file = "flux1-dev-fp8.safetensors";
        var tokPath = Environment.GetEnvironmentVariable("F1_T5_TOKENIZER") ?? "D:/Projects/ltx/google/t5-v1_1-xxl/tokenizer.json";
        if (!File.Exists(Path.Combine(comfy, file)) || !Directory.Exists(d) || !File.Exists(tokPath)) throw new UnsupportedTestException($"needs {comfy}\\{file}, the dumps in {d} and the T5 tokenizer {tokPath}");
        using var fs = File.OpenRead(tokPath);
        var tok = Preprocessing.UnigramTokenizer.FromTokenizerJson(fs);
        using var t5 = await Wan.UmT5Encoder.CreateAsync(accelerator, new LocalFolderModelSource(comfy), "", file);
        t5.PreciseActivations = true;                                            // ComfyUI ran it in fp32
        var report = new List<string>();
        double worst = 0, worstReal = 0;
        foreach (var (prompt, dump) in new[] { ("a red apple on a wooden table", "flux1_cond_0_cond_f32_1x256x4096.raw"), ("", "flux1_cond_empty_0_cond_f32_1x256x4096.raw") })
        {
            var real = tok.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(prompt));
            var ids = new int[256]; Array.Copy(real, ids, real.Length);                 // the pad token is 0
            var ctx = await t5.EncodeAsync(ids);
            var want = F2_Raw(Path.Combine(d, dump), 256L * 4096);
            double all = F2_RelRms(ctx, want), realRows = F2_RelRms(ctx, want, real.Length * 4096), padRows = F2_RelRms(ctx, want, (256 - real.Length) * 4096, real.Length * 4096);
            worst = Math.Max(worst, all); worstReal = Math.Max(worstReal, realRows);
            report.Add($"'{prompt}' ({real.Length} tokens): all rows {all:E2}, real tokens {realRows:E2}, padding rows {padRows:E2}");
        }
        foreach (var line in report) Console.WriteLine("[Flux1] T5 from the checkpoint vs ComfyUI: " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): prompt real tokens 6.13e-3, all rows 1.28e-2; empty prompt (the end token alone) 1.26e-3, all rows 1.99e-2. The empty prompt
        // read as a space piece + end token (the tokenizer before its whitespace fix) was 1.46 on the real rows: that is this gate's red-check.
        // PreciseActivations (MEASURED 2026-11-14): prompt real tokens 3.17e-4, all rows 1.05e-3; empty prompt 5.6e-6 / 1.5e-4 (bf16 activations: 6.1e-3 / 1.3e-2; RX 590 OpenCL 8.3e-6 / 1.4e-5)
        Ur_Expect(worstReal < 1e-3 && worst < 3e-3, string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// The T5 tokenizer's whitespace handling against the HF tokenizer.json pipeline (WhitespaceSplit, then Metaspace): an empty or all-space prompt has no pieces, only the end token
    /// (ComfyUI's empty-prompt conditioning is one <c>&lt;/s&gt;</c> row then padding), and leading / trailing spaces add nothing. Host only: no model weights.
    /// </summary>
    [TestMethod]
    public async Task Flux1_T5Tokenizer_EmptyAndEdgeSpaces() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL or AcceleratorType.CPU)) throw new UnsupportedTestException("reads the tokenizer from disk: desktop");
        var tokPath = Environment.GetEnvironmentVariable("F1_T5_TOKENIZER") ?? "D:/Projects/ltx/google/t5-v1_1-xxl/tokenizer.json";
        if (!File.Exists(tokPath)) throw new UnsupportedTestException($"needs the T5 tokenizer {tokPath}");
        using var fs = File.OpenRead(tokPath);
        var tok = Preprocessing.UnigramTokenizer.FromTokenizerJson(fs);
        string Ids(string s) => "[" + string.Join(",", tok.Encode(s)) + "]";
        var report = $"'' {Ids("")}, '   ' {Ids("   ")}, 'apple' {Ids("apple")}, ' apple  ' {Ids(" apple  ")}";
        Ur_Expect(Ids("") == "[1]" && Ids("   ") == "[1]", "an empty prompt is the end token alone: " + report);
        Ur_Expect(Ids(" apple  ") == Ids("apple"), "edge spaces add no pieces: " + report);
        Console.WriteLine("[Flux1] T5 tokenizer: " + report);
        await Task.CompletedTask;
        return report;
    });

    /// <summary>
    /// The CLIP-L inside ComfyUI's <c>flux1-dev-fp8.safetensors</c> (under <c>text_encoders.clip_l.transformer.</c>, fp16) loaded by <see cref="StableDiffusion.SdTextEncoder"/>, and its pooled
    /// vector against the <c>y</c> ComfyUI handed the FLUX.1 model (SpawnGen's dump of the model call): which reading ComfyUI uses (the final-norm end-token row, with or without the
    /// text projection) is measured here, not assumed.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux1_ClipLFromCheckpoint_PooledMatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F1_DIR") ?? "E:/spawndev-flux1";
        string comfy = Path.Combine(dir, "comfy"), d = Path.Combine(comfy, "comfyui-dumps", "dev"), file = "flux1-dev-fp8.safetensors";
        var tokPath = Path.Combine(SdDir, "tokenizer", "clip-vit-large-patch14", "tokenizer.json");
        if (!File.Exists(Path.Combine(comfy, file)) || !Directory.Exists(d) || !File.Exists(tokPath)) throw new UnsupportedTestException($"needs {comfy}\\{file}, the dumps in {d} and the CLIP tokenizer {tokPath}");
        var tok = Preprocessing.BPETokenizer.LoadFromTokenizerJson(await File.ReadAllTextAsync(tokPath));
        using var clip = await StableDiffusion.SdTextEncoder.LoadAsync(accelerator, new LocalFolderModelSource(comfy), "", file);
        clip.ExactActivations = true;                                            // ComfyUI ran it in fp32
        var ids = tok.EncodeCLIP("a red apple on a wooden table");
        var want = F2_Raw(Path.Combine(d, "flux1_call_0_c_y_f32_1x768.raw"), 768);
        var plain = await clip.PooledAsync(ids, project: false);
        var projected = await clip.PooledAsync(ids, project: true);
        double ePlain = F2_RelRms(plain, want), eProj = F2_RelRms(projected, want);
        var report = $"pooled CLIP-L vs ComfyUI's y: final-norm end-token row {ePlain:E2}, through text_projection {eProj:E2}";
        Console.WriteLine("[Flux1] " + report);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): 1.34e-6 with ExactActivations, the same for both readings (the final-norm end-token row, with or without text_projection); tensor cores 3.33e-4
        Ur_Expect(Math.Min(ePlain, eProj) < 1e-5, report);
        return report;
    });

    /// <summary>
    /// ComfyUI's own VAE Encode / Decode with ae.safetensors, measured by SpawnGen on the ComfyUI reference (float32, CPU; F1_DIR\comfy\comfyui-dumps, Lore's note there): LATENT is
    /// [16, H/8, W/8] channel-major, the RAW encoder mean (FLUX.1's scale 0.3611 / shift 0.1159 are applied on the way into the model, not in LATENT); a picture that is not a
    /// multiple of 8 is cut to its centred multiple. Checked: encode of A (256 x 192), B (200 x 120) and C (205 x 125 -> the centred 200 x 120), decode of ComfyUI's latent of A
    /// and of a synthetic latent (NHWC [0, 1] clamped). Control that must NOT match: the scaled / shifted reading of our latent.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux1_ComfyAe_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var dir = F1_RequireAe(accelerator); var d = Path.Combine(dir, "comfy", "comfyui-dumps");
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"ComfyUI's measured ae dumps not found in {d}");
        using var vae = await SdVae.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", "ae.safetensors");
        var report = new List<string>();
        async Task<double> EncodeVs(string imgFile, int imgH, int imgW, string latFile)
        {
            int H = imgH / 8 * 8, W = imgW / 8 * 8;
            var img = F2_Raw(Path.Combine(d, imgFile), (long)imgH * imgW * 3);
            var z = await vae.EncodeAsync(F2_NhwcToChw(img, imgH, imgW, (imgH - H) / 2, (imgW - W) / 2, H, W), H, W);
            return F2_RelRms(z, F2_Raw(Path.Combine(d, latFile), 16L * (H / 8) * (W / 8)));
        }
        double eA = await EncodeVs("f2_img_1x192x256x3.raw", 192, 256, "ae_enc_256x192_samples_f32_1x16x24x32.raw");
        double eB = await EncodeVs("f2_img_1x120x200x3.raw", 120, 200, "ae_enc_200x120_samples_f32_1x16x15x25.raw");
        double eC = await EncodeVs("ae_img_1x125x205x3.raw", 125, 205, "ae_enc_205x125_samples_f32_1x16x15x25.raw");
        // control: FLUX.1's model-side scaling applied to our latent must NOT be what ComfyUI keeps
        var imgA = F2_Raw(Path.Combine(d, "f2_img_1x192x256x3.raw"), 192L * 256 * 3);
        var zA = await vae.EncodeAsync(F2_NhwcToChw(imgA, 192, 256), 192, 256);
        var scaled = zA.Select(v => (v - 0.1159f) * 0.3611f).ToArray();
        double cScaled = F2_RelRms(scaled, F2_Raw(Path.Combine(d, "ae_enc_256x192_samples_f32_1x16x24x32.raw"), 16L * 24 * 32));
        report.Add($"encode vs ComfyUI: A {eA:E2}, B {eB:E2}, C (centred crop) {eC:E2}; control, scaled + shifted reading {cScaled:E2}");
        async Task<double> DecodeVs(string latFile, string imgFile)
        {
            var lat = F2_Raw(Path.Combine(d, latFile), 16L * 24 * 32);
            var pic = await vae.DecodeAsync(lat, 24, 32);
            var want = F2_Raw(Path.Combine(d, imgFile), 192L * 256 * 3);
            var got = new float[want.Length];
            for (int c = 0; c < 3; c++) for (int p = 0; p < 192 * 256; p++) got[p * 3 + c] = Math.Clamp((pic[c * 192 * 256 + p] + 1f) / 2f, 0f, 1f);
            return F2_RelRms(got, want);
        }
        double dA = await DecodeVs("ae_enc_256x192_samples_f32_1x16x24x32.raw", "ae_dec_of_enc_256x192_images_f32_1x192x256x3.raw");
        double dS = await DecodeVs("ae_lat_1x16x24x32.raw", "ae_dec_synthetic_24x32_images_f32_1x192x256x3.raw");
        report.Add($"decode vs ComfyUI: latent of A {dA:E2}, synthetic {dS:E2}");
        foreach (var line in report) Console.WriteLine("[Flux1] ComfyUI ae: " + line);
        // bounds PROVISIONAL until measured: ComfyUI runs fp32 weights, this library fp16 weights with fp32 activations (FLUX.2's VAE: 5.8e-4 / 1.1e-4)
        // MEASURED (RTX 4070, 2026-11-14, SpawnDev.ILGPU 5.4.32, fp32 weights as fp16 hi + lo): CUDA A 2.81e-4, B 2.03e-5, C 7.95e-5; OpenCL 2.69e-4 / 1.96e-5 / 8.59e-5 (fp16 weights: up to 5e-3)
        Ur_Expect(eA < 6e-4 && eB < 6e-5 && eC < 2e-4, $"encode: A {eA:E2}, B {eB:E2}, C {eC:E2}");
        // MEASURED: decode of A 9.6e-7 (CUDA and OpenCL), synthetic 1.5e-5 / 1.8e-5
        Ur_Expect(dA < 5e-6 && dS < 5e-5, $"decode: A {dA:E2}, synthetic {dS:E2}");
        Ur_Expect(cScaled > 0.1, $"the scaled reading must not match: {cScaled:E2}");
        return string.Join("; ", report);
    });
}
