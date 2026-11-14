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
        Ur_Expect(eA < 0.02 && eB < 0.02 && eC < 0.02, $"encode: A {eA:E2}, B {eB:E2}, C {eC:E2}");
        Ur_Expect(dA < 0.02 && dS < 0.02, $"decode: A {dA:E2}, synthetic {dS:E2}");
        Ur_Expect(cScaled > 0.1, $"the scaled reading must not match: {cScaled:E2}");
        return string.Join("; ", report);
    });
}
