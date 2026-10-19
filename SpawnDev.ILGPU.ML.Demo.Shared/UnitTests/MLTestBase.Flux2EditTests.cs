using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.2 klein reference-image editing against goldens from the real diffusers pipeline (tools/unirig-harness/reference/golden_klein_edit.py: golden/edit/ holds the reference picture ref_0.png, its
/// preprocessed pixels, its latents, the DiT steps and the edited picture; weights/vae_encoder.onnx from export_flux2_vae_encoder.py). Skipped when those are not on this machine.
/// </summary>
public abstract partial class MLTestBase
{
    private static string F2_RequireEdit(Accelerator accelerator)
    {
        var dir = F2_Require(accelerator);
        if (!File.Exists(Path.Combine(dir, "golden", "edit", "ref_latents.npy")) || !File.Exists(Path.Combine(dir, "weights", "vae_encoder.onnx")))
            throw new UnsupportedTestException("FLUX.2 editing goldens or vae_encoder.onnx not found (golden_klein_edit.py, export_flux2_vae_encoder.py)");
        return dir;
    }

    private static Flux2ReferenceImage F2_Reference(string pngPath)
    {
        var png = File.ReadAllBytes(pngPath);
        var (w, h) = PngDecoder.GetDimensions(png);
        return new Flux2ReferenceImage(w, h, PngDecoder.DecodePixels(png) ?? throw new Exception("the reference png must be 8-bit and not interlaced"));
    }

    [TestMethod]
    public async Task Flux2_ReferencePrep_ShrinksOnlyAboveTheAreaLimitAndCropsToMultiplesOf16() => await RunPureTest(async () =>
    {
        Ur_Expect(Flux2ReferencePrep.PreparedSize(512, 512) == (512, 512), "a 512 picture is left alone");
        Ur_Expect(Flux2ReferencePrep.PreparedSize(517, 301) == (512, 288), "sides are floored to multiples of 16 (centre crop)");
        Ur_Expect(Flux2ReferencePrep.PreparedSize(2048, 1024) == (1440, 720), $"above 1024x1024 of area it is shrunk to that area first ({Flux2ReferencePrep.PreparedSize(2048, 1024)})");
        // a 40 x 20 picture crops to 32 x 16 around the centre
        var rgba = new byte[40 * 20 * 4];
        for (int y = 0; y < 20; y++) for (int x = 0; x < 40; x++) { int o = (y * 40 + x) * 4; rgba[o] = (byte)(x * 6); rgba[o + 3] = 255; }
        var (chw, w, h) = Flux2ReferencePrep.Prepare(new Flux2ReferenceImage(40, 20, rgba));
        Ur_Expect(w == 32 && h == 16, $"32 x 16 ({w} x {h})");
        Ur_Expect(Math.Abs(chw[0] - (4 * 6 / 255f * 2f - 1f)) < 1e-6f, "the crop starts 4 columns in (left = (40 - 32) / 2) and pixels map to [-1, 1]");
        // Lanczos keeps a flat picture flat
        var flat = new byte[64 * 64 * 4]; Array.Fill(flat, (byte)90);
        var small = Flux2ReferencePrep.Lanczos(flat, 64, 64, 20, 20);
        Ur_Expect(small.All(v => v == 90), "a flat picture stays flat when shrunk");
        await Task.CompletedTask;
    });

    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ReferenceEncoder_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_RequireEdit(accelerator); var g = Path.Combine(dir, "golden", "edit");
        var (refIn, _) = Ur_Npy<float>(Path.Combine(g, "ref_in_0.npy")); var (refLat, _) = Ur_Npy<float>(Path.Combine(g, "ref_latents.npy")); var (refIds, _) = Ur_Npy<long>(Path.Combine(g, "ref_ids.npy"));
        var reference = F2_Reference(Path.Combine(g, "ref_0.png"));
        var (chw, w, h) = Flux2ReferencePrep.Prepare(reference);
        double prepErr = 0; for (int i = 0; i < chw.Length; i++) prepErr = Math.Max(prepErr, Math.Abs(chw[i] - refIn[i]));
        Ur_Expect(prepErr < 1e-5, $"the preprocessed pixels equal the reference's (max diff {prepErr:E2})");
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var encBytes = File.ReadAllBytes(Path.Combine(dir, "weights", "vae_encoder.onnx"));
        var pipe = new Flux2Pipeline(accelerator, null!, null!, null!, vae, F2_Norm(dir))
        {
            VaeEncoderFactory = (hh, ww) => Task.FromResult(InferenceSession.CreateFromFile(accelerator, encBytes, inputShapes: new Dictionary<string, int[]> { ["image"] = new[] { 1, 3, hh, ww } })),
        };
        var (tokens, rows, cols) = await pipe.EncodeReferenceAsync(reference);
        double rel = F2_RelRms(tokens, refLat);
        Console.WriteLine($"[Flux2] reference encoder on {accelerator.AcceleratorType}: {rows}x{cols} tokens, relRMS {rel:E2} vs the reference's bf16 encode");
        Ur_Expect(rows * cols * 128 == refLat.Length && rows == h / 16 && cols == w / 16, "token count and shape");
        Ur_Expect(rel < 0.03, $"reference latents relRMS {rel:E2}");
        var ids = Flux2Pipeline.ImageIds(rows, cols, 10);
        Ur_Expect(ids.Length == refIds.Length && Enumerable.Range(0, ids.Length).All(i => ids[i] == refIds[i]), "the reference's position ids (t = 10, row, column, 0)");
    });

    /// <summary>The whole edit from the reference's prompt, start noise and reference picture: encoder, 4 transformer steps over [noise | reference tokens], VAE; the picture against the reference's.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_ReferenceEdit_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_RequireEdit(accelerator); var g = Path.Combine(dir, "golden", "edit");
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        var prompt = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "meta.json"))).RootElement.GetProperty("prompt").GetString()!;
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var encBytes = File.ReadAllBytes(Path.Combine(dir, "weights", "vae_encoder.onnx"));
        var pipe = new Flux2Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder.safetensors")))),
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "transformer.safetensors")))), vae, F2_Norm(dir))
        {
            VaeEncoderFactory = (hh, ww) => Task.FromResult(InferenceSession.CreateFromFile(accelerator, encBytes, inputShapes: new Dictionary<string, int[]> { ["image"] = new[] { 1, 3, hh, ww } })),
        };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux2Request { Prompt = prompt, Steps = 4, Noise = noise, References = new[] { F2_Reference(Path.Combine(g, "ref_0.png")) }, Progress = (s, i, n) => Console.WriteLine($"[Flux2] {s} {i}/{n}") });
        double sq = 0, abs = 0; int n = 512 * 512;
        Ur_Expect(result.Width == 512 && result.Height == 512, "the picture has the reference's size");
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); abs += Math.Abs(have - want);
            }
        double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        Console.WriteLine($"[Flux2] reference edit on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, PSNR {psnr:F1} dB, mean |diff| {abs / (3.0 * n):F2} grey levels vs the reference picture");
        using (var f = File.Create(Path.Combine(g, "csharp.ppm")))
        {
            f.Write(System.Text.Encoding.ASCII.GetBytes("P6\n512 512\n255\n"));
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        Ur_Expect(psnr > 22, $"edited picture PSNR {psnr:F1} dB vs the reference");
    });
}
