using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.2's VAE from ComfyUI's <c>flux2-vae.safetensors</c> (Comfy-Org repackaged; the klein repos ship Black Forest Labs' names, the dev repo diffusers names; the same fp32
/// weights) on the library's native LDM VAE (<see cref="SdVae"/>), with the batch-norm statistics read from the file (<see cref="Flux2VaeNorm.FromSafeTensorsAsync"/>). The files in
/// F2_DIR\comfy (flux2-vae.safetensors = the klein repo's, flux2-vae-dev.safetensors = the dev repo's). Desktop GPU lanes.
/// </summary>
public abstract partial class MLTestBase
{
    private static string F2_RequireComfyVae(Accelerator accelerator)
    {
        var dir = F2_Require(accelerator);
        if (!File.Exists(Path.Combine(dir, "comfy", "flux2-vae.safetensors")) || !File.Exists(Path.Combine(dir, "comfy", "flux2-vae-dev.safetensors")))
            throw new UnsupportedTestException($"ComfyUI's flux2-vae.safetensors (Comfy-Org/flux2-klein) and flux2-vae-dev.safetensors (Comfy-Org/flux2-dev) not found in {dir}\\comfy");
        return dir;
    }

    /// <summary>
    /// Both namings load (32 latent channels, 4 levels, encoder and decoder) to the same tensors, bit for bit after the fp16 conversion; the batch-norm statistics read from either
    /// file are equal and within bf16 rounding of our export's vae_norm.json (made from the bf16 diffusers checkpoint; ComfyUI's files carry the fp32 values).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux2_ComfyVae_BothNamingsLoad() => await RunTest(async accelerator =>
    {
        var dir = F2_RequireComfyVae(accelerator); var src = new LocalFolderModelSource(Path.Combine(dir, "comfy"));
        using var a = await SdVae.LoadAsync(accelerator, src, "", "flux2-vae.safetensors");
        using var b = await SdVae.LoadAsync(accelerator, src, "", "flux2-vae-dev.safetensors");
        foreach (var (v, n) in new[] { (a, "klein"), (b, "dev") })
            Ur_Expect(v.LatentChannels == 32 && v.Levels == 4 && v.HasEncoder && v.HasDecoder, $"{n}: 32 latent channels, 4 levels, encoder and decoder ({v.LatentChannels}, {v.Levels}, {v.HasEncoder}, {v.HasDecoder})");
        var na = a.Weights.Names.OrderBy(x => x, StringComparer.Ordinal).ToArray(); var nb = b.Weights.Names.OrderBy(x => x, StringComparer.Ordinal).ToArray();
        Ur_Expect(na.SequenceEqual(nb), $"the same tensor names: only klein {string.Join(", ", na.Except(nb).Take(5))}; only dev {string.Join(", ", nb.Except(na).Take(5))}");
        long values = 0;
        foreach (var name in na)
        {
            static int[] Squeeze(int[] s) { int n = s.Length; while (n > 1 && s[n - 1] == 1) n--; return s[..n]; }    // diffusers stores the attention projections as [C, C], LDM as [C, C, 1, 1]
            Ur_Expect(Squeeze(a.Weights.Shape(name)).SequenceEqual(Squeeze(b.Weights.Shape(name))), $"{name}: shapes differ ([{string.Join(",", a.Weights.Shape(name))}] vs [{string.Join(",", b.Weights.Shape(name))}])");
            var x = await a.Weights[name].CopyToHostAsync(); var y = await b.Weights[name].CopyToHostAsync();
            for (int i = 0; i < x.Length; i++) if (x[i].RawValue != y[i].RawValue) throw new Exception($"{name}[{i}]: {(float)x[i]} vs {(float)y[i]}");
            values += x.Length;
        }
        var ka = await Flux2VaeNorm.FromSafeTensorsAsync(src, "", "flux2-vae.safetensors"); var kb = await Flux2VaeNorm.FromSafeTensorsAsync(src, "", "flux2-vae-dev.safetensors");
        Ur_Expect(ka.Mean.SequenceEqual(kb.Mean) && ka.Std.SequenceEqual(kb.Std), "the batch-norm statistics of the two files are equal");
        var ours = F2_Norm(dir);
        double worstMean = 0, worstStd = 0;
        for (int i = 0; i < 128; i++)
        {
            worstMean = Math.Max(worstMean, Math.Abs(ka.Mean[i] - ours.Mean[i]) / Math.Max(Math.Abs(ka.Mean[i]), 1e-30));
            worstStd = Math.Max(worstStd, Math.Abs(ka.Std[i] - ours.Std[i]) / ka.Std[i]);
        }
        Ur_Expect(worstMean <= 1.0 / 256 && worstStd <= 1.0 / 256, $"vae_norm.json within bf16 rounding (mean {worstMean:E2}, std {worstStd:E2})");
        var report = $"{na.Length} tensors, {values:N0} values bit-identical across the two namings; batch-norm vs vae_norm.json: mean {worstMean:E2}, std {worstStd:E2} relative";
        Console.WriteLine("[Flux2] ComfyUI VAE: " + report);
        return report;
    });

    /// <summary>
    /// The native VAE against the ONNX export on the SAME weights (both from the bf16 diffusers checkpoint: this isolates the implementation) and ComfyUI's fp32 file against the reference
    /// pipeline: decode of the reference's vae_in [32, 64, 64] to its picture, encode of the reference's prepared edit picture to its raw latent (ref_z_0) and, through the batch-norm and
    /// 2 x 2 patchify (<see cref="Flux2Kernels.RefTokens"/>), to its transformer tokens (ref_latents).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyVae_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_RequireComfyVae(accelerator); var g = Path.Combine(dir, "golden", "cat"); var ge = Path.Combine(dir, "golden", "edit");
        var (vaeIn, _) = Ur_Npy<float>(Path.Combine(g, "vae_in.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var report = new List<string>();
        // ---- decode: ONNX (bf16 weights) vs native on the same bf16 weights vs native on ComfyUI's fp32 weights
        float[] onnx;
        using (var session = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } }))
        {
            var pipe = new Flux2Pipeline(accelerator, null!, null!, null!, session, F2_Norm(dir));
            onnx = new float[3 * 512 * 512];
            await pipe.DecodeAsync(vaeIn, 32, onnx);
        }
        float[] nativeSame, nativeComfy;
        using (var v = await SdVae.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "klein4b", "vae")), "", "diffusion_pytorch_model.safetensors"))
            nativeSame = await v.DecodeAsync(vaeIn, 64, 64);
        using var comfy = await SdVae.LoadAsync(accelerator, new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", "flux2-vae.safetensors");
        nativeComfy = await comfy.DecodeAsync(vaeIn, 64, 64);
        double dSame = F2_RelRms(nativeSame, onnx), dComfyRef = F2_RelRms(nativeComfy, image), dOnnxRef = F2_RelRms(onnx, image), dComfyOnnx = F2_RelRms(nativeComfy, onnx);
        report.Add($"decode: native vs ONNX on the same bf16 weights {dSame:E2}; vs the reference picture: ComfyUI fp32 native {dComfyRef:E2}, ONNX {dOnnxRef:E2}; ComfyUI native vs ONNX {dComfyOnnx:E2}");
        // ---- encode: the reference's prepared edit picture to its raw latent and its packed, normalised tokens
        var (refIn, refInShape) = Ur_Npy<float>(Path.Combine(ge, "ref_in_0.npy")); var (refZ, _) = Ur_Npy<float>(Path.Combine(ge, "ref_z_0.npy")); var (refLat, _) = Ur_Npy<float>(Path.Combine(ge, "ref_latents.npy"));
        int H = refInShape[^2], W = refInShape[^1], lh = H / 8, lw = W / 8;
        var z = await comfy.EncodeAsync(refIn, H, W);
        var norm = await Flux2VaeNorm.FromSafeTensorsAsync(new LocalFolderModelSource(Path.Combine(dir, "comfy")), "", "flux2-vae.safetensors");
        using var zb = accelerator.Allocate1D(z); using var mean = accelerator.Allocate1D(norm.Mean); using var std = accelerator.Allocate1D(norm.Std);
        int rows = lh / 2, cols = lw / 2;
        using var lat = accelerator.Allocate1D<float>(128L * rows * cols); using var tok = accelerator.Allocate1D<float>(128L * rows * cols);
        var k = new Flux2Kernels(accelerator);
        k.VaeToLatent(zb.View, mean.View, std.View, lat.View, lh, lw);       // ref_z: [128, H/16, W/16] patchified + batch-normalised, as ComfyUI's LATENT
        k.LatentToTokens(lat.View, tok.View, rows, cols);                     // ref_latents: the same values as tokens
        var latent = await lat.View.CopyToHostAsync(); var tokens = await tok.View.CopyToHostAsync();
        double dZ = F2_RelRms(latent, refZ, latent.Length), dTok = F2_RelRms(tokens, refLat, tokens.Length);
        // ---- LATENT -> VAE input -> LATENT
        using var back = accelerator.Allocate1D<float>(32L * lh * lw);
        k.LatentToVae(lat.View, mean.View, std.View, back.View, rows, cols);
        var zBack = await back.View.CopyToHostAsync();
        double dRound = F2_RelRms(zBack, z);
        report.Add($"encode {W}x{H}: LATENT vs the reference's ref_z {dZ:E2}, tokens vs ref_latents {dTok:E2}; LATENT -> VAE input round trip {dRound:E2}");
        foreach (var line in report) Console.WriteLine("[Flux2] ComfyUI VAE " + line);
        // bounds: PROVISIONAL until measured (the same-weights decode is float rounding; the rest carries bf16-vs-fp32 weights and the reference's own bf16 run)
        Ur_Expect(dSame < 1e-2, $"native vs ONNX on the same weights {dSame:E2}");
        Ur_Expect(dComfyRef < 0.05 && dZ < 0.05 && dTok < 0.05, $"ComfyUI VAE vs the reference: decode {dComfyRef:E2}, latent {dZ:E2}, tokens {dTok:E2}");
        Ur_Expect(dRound < 1e-5, $"round trip {dRound:E2}");
        return string.Join("; ", report);
    });

    private static float[] F2_Raw(string path, long count)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * 4) throw new InvalidDataException($"{path}: {bytes.Length} bytes, expected {count * 4}");
        var v = new float[count]; Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length); return v;
    }

    /// <summary>A picture NHWC in [0, 1] (ComfyUI's IMAGE) to the VAE's [3, H, W] in [-1, 1].</summary>
    private static float[] F2_NhwcToChw(float[] nhwc, int H, int W, int y0 = 0, int x0 = 0, int h = -1, int w = -1)
    {
        if (h < 0) { h = H; w = W; }
        var o = new float[3L * h * w];
        for (int c = 0; c < 3; c++) for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) o[((long)c * h + y) * w + x] = nhwc[(((long)(y0 + y)) * W + x0 + x) * 3 + c] * 2f - 1f;
        return o;
    }

    /// <summary>
    /// ComfyUI's own FLUX.2 VAE Encode / Decode, measured by SpawnGen on the ComfyUI reference (float32, CPU) with this file (F2_DIR\comfy\comfyui-dumps; Lore's note there): LATENT is
    /// [128, H/16, W/16] channel-major, the encoder mean patchified 2 x 2 AND batch-normalised (<see cref="Flux2Kernels.VaeToLatent"/>; the transformer's tokens transposed); Decode takes that
/// layout (<see cref="Flux2Kernels.LatentToVae"/>) and gives NHWC in [0, 1], clamped. Checked here:
    /// encode of picture A (256 x 192) and of picture B (200 x 120, which ComfyUI cuts to the centred 192 x 112), decode of ComfyUI's latent of A and of a synthetic latent, and the
    /// LATENT -> tokens -> LATENT round trip. The un-normalised and the dy / dx-swapped readings are controls that must NOT match.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ComfyVae_MatchesComfyUiLatents() => await RunTest(async accelerator =>
    {
        var dir = F2_RequireComfyVae(accelerator); var d = Path.Combine(dir, "comfy", "comfyui-dumps");
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"ComfyUI's measured VAE dumps not found in {d}");
        var src = new LocalFolderModelSource(Path.Combine(dir, "comfy"));
        using var vae = await SdVae.LoadAsync(accelerator, src, "", "flux2-vae.safetensors");
        var norm = await Flux2VaeNorm.FromSafeTensorsAsync(src, "", "flux2-vae.safetensors");
        var k = new Flux2Kernels(accelerator);
        using var mean = accelerator.Allocate1D(norm.Mean); using var std = accelerator.Allocate1D(norm.Std);
        var report = new List<string>();
        async Task<float[]> EncodeToLatent(float[] chw, int H, int W)
        {
            var z = await vae.EncodeAsync(chw, H, W);
            using var zb = accelerator.Allocate1D(z); using var lat = accelerator.Allocate1D<float>(128L * (H / 16) * (W / 16));
            k.VaeToLatent(zb.View, mean.View, std.View, lat.View, H / 8, W / 8);
            return await lat.View.CopyToHostAsync();
        }
        // ---- encode A
        var imgA = F2_Raw(Path.Combine(d, "f2_img_1x192x256x3.raw"), 192L * 256 * 3);
        var latA = F2_Raw(Path.Combine(d, "enc_256x192_samples_f32_1x128x12x16.raw"), 128L * 12 * 16);
        var mineA = await EncodeToLatent(F2_NhwcToChw(imgA, 192, 256), 192, 256);
        double eA = F2_RelRms(mineA, latA);
        var rawA = new float[mineA.Length]; var swapA = new float[mineA.Length];
        for (int ch = 0; ch < 128; ch++)
        {
            int sw = (ch & ~3) | ((ch & 1) << 1) | ((ch >> 1) & 1);           // dy and dx exchanged
            for (int p = 0; p < 192; p++) { rawA[ch * 192 + p] = mineA[ch * 192 + p] * norm.Std[ch] + norm.Mean[ch]; swapA[ch * 192 + p] = mineA[sw * 192 + p]; }
        }
        double cRaw = F2_RelRms(rawA, latA), cSwap = F2_RelRms(swapA, latA);
        report.Add($"encode A 256x192: vs ComfyUI {eA:E2} (controls, must not match: not batch-normalised {cRaw:E2}, dy/dx swapped {cSwap:E2})");
        // ---- encode B: ComfyUI cuts 200 x 120 to the centred 192 x 112 (offset 4, 4)
        var imgB = F2_Raw(Path.Combine(d, "f2_img_1x120x200x3.raw"), 120L * 200 * 3);
        var latB = F2_Raw(Path.Combine(d, "enc_200x120_samples_f32_1x128x7x12.raw"), 128L * 7 * 12);
        var mineB = await EncodeToLatent(F2_NhwcToChw(imgB, 120, 200, 4, 4, 112, 192), 112, 192);
        double eB = F2_RelRms(mineB, latB);
        report.Add($"encode B 200x120 (centred 192x112 crop): vs ComfyUI {eB:E2}");
        // ---- decode ComfyUI's latents: Unpatchify, decode, to NHWC [0, 1] clamped
        async Task<double> DecodeVs(string latFile, string imgFile)
        {
            var lat = F2_Raw(Path.Combine(d, latFile), 128L * 12 * 16);
            using var lb = accelerator.Allocate1D(lat); using var raw = accelerator.Allocate1D<float>(32L * 24 * 32);
            k.LatentToVae(lb.View, mean.View, std.View, raw.View, 12, 16);
            var z = await raw.View.CopyToHostAsync();
            var pic = await vae.DecodeAsync(z, 24, 32);
            var want = F2_Raw(Path.Combine(d, imgFile), 192L * 256 * 3);
            var got = new float[want.Length];
            for (int c = 0; c < 3; c++) for (int p = 0; p < 192 * 256; p++) got[p * 3 + c] = Math.Clamp((pic[c * 192 * 256 + p] + 1f) / 2f, 0f, 1f);
            return F2_RelRms(got, want);
        }
        double dA = await DecodeVs("enc_256x192_samples_f32_1x128x12x16.raw", "dec_of_enc_256x192_images_f32_1x192x256x3.raw");
        double dS = await DecodeVs("f2_lat_1x128x12x16.raw", "dec_synthetic_12x16_images_f32_1x192x256x3.raw");
        report.Add($"decode ComfyUI's latent of A vs ComfyUI's picture {dA:E2}, synthetic latent {dS:E2}");
        // ---- LATENT -> tokens -> LATENT
        using var la = accelerator.Allocate1D(latA); using var tk = accelerator.Allocate1D<float>(192L * 128); using var lb2 = accelerator.Allocate1D<float>(latA.Length);
        k.LatentToTokens(la.View, tk.View, 12, 16);
        k.TokensToLatent(tk.View, lb2.View, 12, 16);
        var tokens = await tk.View.CopyToHostAsync(); var back = await lb2.View.CopyToHostAsync();
        long tokDiff = 0;
        for (int cell = 0; cell < 192; cell++) for (int ch = 0; ch < 128; ch++) if (tokens[cell * 128 + ch] != latA[ch * 192 + cell]) tokDiff++;
        long backDiff = 0; for (int i = 0; i < latA.Length; i++) if (back[i] != latA[i]) backDiff++;
        report.Add($"LATENT -> tokens: {tokDiff} values differ from the transpose; round trip: {backDiff} differ");
        foreach (var line in report) Console.WriteLine("[Flux2] ComfyUI layout: " + line);
        // bounds PROVISIONAL until measured: ComfyUI runs the fp32 weights, this library fp16 weights with fp32 activations
        Ur_Expect(eA < 0.02 && eB < 0.02, $"encode vs ComfyUI: A {eA:E2}, B {eB:E2}");
        Ur_Expect(dA < 0.02 && dS < 0.02, $"decode vs ComfyUI: A {dA:E2}, synthetic {dS:E2}");
        Ur_Expect(tokDiff == 0 && backDiff == 0, $"transpose exact: tokens {tokDiff}, round trip {backDiff}");
        Ur_Expect(cRaw > 0.1 && cSwap > 0.1, $"the controls must not match: not batch-normalised {cRaw:E2}, dy/dx swapped {cSwap:E2}");
        return string.Join("; ", report);
    });
}
