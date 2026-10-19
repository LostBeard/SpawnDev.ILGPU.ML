using System.Text.Json;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX 3 Image Klein (black-forest-labs; code Apache-2.0, weights FLUX Kommunity License) against the reference code's own classes
/// (tools/unirig-harness/reference/flux3/: flux3_dit.py is the vendored reference DiT, golden_flux3_small.py runs a SMALL copy of it with random weights in float32 on the CPU; flux3_convert.py fuses a
/// checkpoint into the layout <see cref="Flux3Transformer"/> reads). F3_DIR default E:/spawndev-flux3 (golden/small/); skipped when it is not on this machine.
/// </summary>
public abstract partial class MLTestBase
{
    private static string? F3Dir()
    {
        var dir = Environment.GetEnvironmentVariable("F3_DIR") ?? "E:/spawndev-flux3";
        return File.Exists(Path.Combine(dir, "golden", "small", "weights.safetensors")) ? dir : null;
    }

    private static string F3_Require(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("FLUX 3 goldens run on the desktop GPU lanes (CUDA, OpenCL)");
        return F3Dir() ?? throw new UnsupportedTestException("FLUX 3 goldens not found (F3_DIR, default E:/spawndev-flux3; tools/unirig-harness/reference/flux3/golden_flux3_small.py)");
    }

    /// <summary>The small random-weight DiT against the reference code: the whole forward (token modes, early blocks per mode, joint blocks, flow-map time input, final layer) and three block outputs.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux3_SmallTransformer_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F3_Require(accelerator); var g = Path.Combine(dir, "golden", "small");
        using var cfgDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "config.json"))); var c = cfgDoc.RootElement;
        var cfg = new Flux3Config(Dim: c.GetProperty("dim").GetInt32(), Heads: c.GetProperty("heads").GetInt32(), EarlyDepth: c.GetProperty("early").GetInt32(),
            SingleDepth: c.GetProperty("single").GetInt32(), MlpHidden: c.GetProperty("mlp").GetInt32(), TextChannels: c.GetProperty("text").GetInt32());
        float[] Load(string name) => Ur_Npy<float>(Path.Combine(g, name + ".npy")).Item1;
        var image = Load("image"); var cond = Load("cond"); var text = Load("text");
        var imageIds = Load("image_ids"); var condIds = Load("cond_ids"); var textIds = Load("text_ids");
        float t = Load("t")[0], tNext = Load("t_next")[0];
        var want = Load("out");
        int nImg = image.Length / 128, nCond = cond.Length / 128, nTxt = text.Length / cfg.TextChannels;
        using var w = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(g, "weights.safetensors")));
        using var tr = new Flux3Transformer(accelerator, w, cfg, nImg, nCond, nTxt) { RoundTimestepToBf16 = false };
        var taps = new Dictionary<string, float[]>();
        tr.Probe = (name, view, rows, cols) =>
        {
            if (name == "early2" || name == "single1" || name == $"single{cfg.SingleDepth}") { accelerator.Synchronize(); taps[name] = view.SubView(0, (long)rows * cols).GetAsArray1D(); }
        };
        tr.Forward(image, cond, text, t, tNext, imageIds, condIds, textIds);
        await accelerator.SynchronizeAsync();
        var got = await tr.Output.SubView(0, (long)nImg * 128).CopyToHostAsync();
        double rel = F2_RelRms(got, want);
        Console.WriteLine($"[Flux3] small DiT on {accelerator.AcceleratorType}: velocity relRMS {rel:E2} ({nImg}+{nCond}+{nTxt} tokens)");
        // the taps: the reference's early block 2 of the image mode is the image rows only; the single blocks are the whole stream
        var tapEarly = Load("tap_early_image2"); var tapS1 = Load("tap_single1"); var tapSL = Load("tap_single_last");
        double relEarly = taps.TryGetValue("early2", out var te) ? F2_RelRms(te, tapEarly, tapEarly.Length) : double.NaN;
        double relS1 = F2_RelRms(taps["single1"], tapS1), relSL = F2_RelRms(taps[$"single{cfg.SingleDepth}"], tapSL);
        Console.WriteLine($"[Flux3] taps: early image block 2 {relEarly:E2}, single block 1 {relS1:E2}, last single block {relSL:E2}");
        // OpenCL runs the portable fp32 GEMM (agreement at float noise, 1e-5); CUDA's tensor-core GEMM rounds its inputs to bf16 (a few 1e-3, the same level as the FLUX.2 gates)
        double tol = accelerator.AcceleratorType == AcceleratorType.Cuda ? 1.5e-2 : 1e-4;
        Ur_Expect(relEarly < tol, $"early image block 2 relRMS {relEarly:E2}");
        Ur_Expect(relS1 < tol, $"single block 1 relRMS {relS1:E2}");
        Ur_Expect(relSL < tol, $"last single block relRMS {relSL:E2}");
        Ur_Expect(rel < tol, $"velocity relRMS {rel:E2}");
    });

    // ---- the real checkpoint (gated; golden/t2i written by golden_flux3.py, weights by quantize_flux3.py / quantize_flux3_text.py / export_flux3_vae.py) -----------------------------------------

    private static string F3_RequireReal(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("FLUX 3 runs on the desktop GPU lanes (CUDA, OpenCL)");
        var dir = Environment.GetEnvironmentVariable("F3_DIR") ?? "E:/spawndev-flux3";
        foreach (var f in new[] { "golden/t2i/text_tokens.npy", "weights/transformer_q8.safetensors", "weights/text_encoder_q8.safetensors", "weights/vae_decoder.onnx", "weights/vae_norm.json", "repo/text_encoder/tokenizer.json" })
            if (!File.Exists(Path.Combine(dir, f))) throw new UnsupportedTestException($"FLUX 3 {f} not found (F3_DIR, default E:/spawndev-flux3; tools/unirig-harness/reference/flux3/)");
        return dir;
    }

    /// <summary>The Qwen3-VL text tower (8-bit blocks) against the reference's stacked hidden states of the same caption: token ids, then the [80, 20480] conditioning matrix.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux3_TextEncoder_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator); var g = Path.Combine(dir, "golden", "t2i");
        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "meta.json"))).RootElement;
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "repo", "text_encoder", "tokenizer.json")));
        var (goldIds, _) = Ur_Npy<float>(Path.Combine(g, "text_ids.npy")); var (goldTokens, _) = Ur_Npy<float>(Path.Combine(g, "text_tokens.npy"));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var pipe = new Flux3Pipeline(accelerator, tok, () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder_q8.safetensors")))), null!, vae, F3_Norm(dir));
        var (ids, real) = pipe.Tokenize(meta.GetProperty("caption").GetString()!);
        Ur_Expect(ids.Length == goldIds.Length && Enumerable.Range(0, ids.Length).All(i => ids[i] == (int)goldIds[i]), $"token ids equal the reference's ({ids.Length} tokens, {real} real)");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (rows, n) = await pipe.EncodeTextAsync(meta.GetProperty("caption").GetString()!);
        double rel = F2_RelRms(rows, goldTokens);
        double relReal = F2_RelRms(rows, goldTokens, real * Flux3TextEncoder.EmbedWidth);
        Console.WriteLine($"[Flux3] text encoder on {accelerator.AcceleratorType}: {n} tokens in {sw.Elapsed.TotalSeconds:F1} s, relRMS {rel:E2} (real rows {relReal:E2}) vs the reference's bf16 run");
        Ur_Expect(rel < 0.05, $"conditioning relRMS {rel:E2}");
    });

    private static Flux2VaeNorm F3_Norm(string dir) => Flux2VaeNorm.FromJson(File.ReadAllText(Path.Combine(dir, "weights", "vae_norm.json")));

    /// <summary>The whole text to image from the reference's caption and start noise: text encoder, 16 flow-map steps (8-bit transformer), VAE; the picture against the reference's.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux3_Pipeline_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator); var g = Path.Combine(dir, "golden", "t2i");
        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "meta.json"))).RootElement;
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "repo", "text_encoder", "tokenizer.json")));
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        int size = meta.GetProperty("size").GetInt32();
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var pipe = new Flux3Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder_q8.safetensors")))),
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "transformer_q8.safetensors")))), vae, F3_Norm(dir));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux3Request { Caption = meta.GetProperty("caption").GetString()!, Width = size, Height = size, Steps = meta.GetProperty("steps").GetInt32(), Alpha = (float)meta.GetProperty("alpha").GetDouble(), Noise = noise,
            Progress = (s, i, n) => Console.WriteLine($"[Flux3] {s} {i}/{n}") });
        double sq = 0, abs = 0; int n = size * size;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); abs += Math.Abs(have - want);
            }
        double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        Console.WriteLine($"[Flux3] pipeline on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, PSNR {psnr:F1} dB, mean |diff| {abs / (3.0 * n):F2} grey levels vs the reference picture");
        using (var f = File.Create(Path.Combine(g, "csharp.ppm")))
        {
            f.Write(System.Text.Encoding.ASCII.GetBytes("P6\n" + size + " " + size + "\n255\n"));
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        Ur_Expect(psnr > 20, $"picture PSNR {psnr:F1} dB vs the reference");
    });
}
