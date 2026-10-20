using System.Text.Json;
using BBP = global::SpawnDev.ILGPU.BrowserBufferPolicy;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Flux3;
using SpawnDev.ILGPU.ML.Tensors;
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

    /// <summary>Weight bits the real-checkpoint gates load (F3_BITS = 4 or 8; default 8).</summary>
    private static string F3_Bits => Environment.GetEnvironmentVariable("F3_BITS") == "4" ? "4" : "8";

    /// <summary>Euler steps the picture gates use instead of the reference's 16 (F3_STEPS): the flow-map model takes fewer, at some cost.</summary>
    private static int? F3_Steps => int.TryParse(Environment.GetEnvironmentVariable("F3_STEPS"), out var n) && n > 0 ? n : null;

    private static string F3_RequireReal(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("FLUX 3 runs on the desktop GPU lanes (CUDA, OpenCL)");
        var dir = Environment.GetEnvironmentVariable("F3_DIR") ?? "E:/spawndev-flux3";
        foreach (var f in new[] { "golden/t2i/text_tokens.npy", $"weights/transformer_q{F3_Bits}.safetensors", $"weights/text_encoder_q{F3_Bits}.safetensors", "weights/vae_decoder.onnx", "weights/vae_norm.json", "repo/text_encoder/tokenizer.json" })
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
        var pipe = new Flux3Pipeline(accelerator, tok, () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"text_encoder_q{F3_Bits}.safetensors")))), null!, vae, F3_Norm(dir));
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

    /// <summary>
    /// The decoder built for 32 x 32 latent cells (256 px tiles, 0.8 GB of feature maps instead of 3.2 GB: what lets a browser tab keep the transformer resident beside it; Chrome drops the WebGPU device at ~70% of the card's memory)
    /// against the decoder built for the whole 64 x 64 latent, on the real latents of the reference run (the start noise integrated along the reference's per-step velocities), and the whole decode against the reference picture.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Flux3_SmallTileVae_MatchesTheWholeDecode() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator); var g = Path.Combine(dir, "golden", "t2i");
        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "meta.json"))).RootElement;
        int steps = meta.GetProperty("steps").GetInt32(), size = meta.GetProperty("size").GetInt32(), cells = size / 16;
        var (latents, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var schedule = Flux3Pipeline.Schedule(steps, meta.GetProperty("alpha").GetDouble());
        for (int i = 0; i < steps; i++)
        {
            var (pred, _) = Ur_Npy<float>(Path.Combine(g, $"pred_{i}.npy")); float dt = (float)(schedule[i + 1] - schedule[i]);
            for (int k = 0; k < latents.Length; k++) latents[k] += dt * pred[k];
        }
        var bytes = File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx"));
        using var whole = InferenceSession.CreateFromFile(accelerator, bytes, inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var norm = F3_Norm(dir);
        var wholePipe = new Flux2Pipeline(accelerator, null!, null!, null!, whole, norm);
        var a = await wholePipe.DecodeAsync(wholePipe.ToVaeInput(latents, cells, cells), cells, cells);
        int n = size * size; double sqRef = 0;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255);
                sqRef += (a.Rgba[p * 4 + c] - want) * (a.Rgba[p * 4 + c] - want);
            }
        double psnrRef = 10 * Math.Log10(255.0 * 255.0 / Math.Max(sqRef / (3.0 * n), 1e-9));
        Console.WriteLine($"[Flux3] VAE on {accelerator.AcceleratorType}: whole decode {psnrRef:F1} dB against the reference picture");
        Ur_Expect(psnrRef > 25, $"the integrated latents decode to the reference picture ({psnrRef:F1} dB): the test's latents are right");
        foreach (var (tile, overlap) in new[] { (32, 16), (40, 16), (48, 16), (56, 16) })
        {
            using var small = InferenceSession.CreateFromFile(accelerator, bytes, inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, tile, tile } });
            var tiledPipe = new Flux2Pipeline(accelerator, null!, null!, null!, small, norm) { VaeTileLatent = tile, VaeTileOverlap = overlap };
            var b = await tiledPipe.DecodeAsync(tiledPipe.ToVaeInput(latents, cells, cells), cells, cells);
            double sq = 0, sqTile = 0;
            for (int p = 0; p < n; p++)
                for (int c = 0; c < 3; c++)
                {
                    double d = a.Rgba[p * 4 + c] - b.Rgba[p * 4 + c], want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255);
                    sq += d * d; sqTile += (b.Rgba[p * 4 + c] - want) * (b.Rgba[p * 4 + c] - want);
                }
            double psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(sq / (3.0 * n), 1e-9));
            Console.WriteLine($"[Flux3] VAE tiles of {tile} cells ({tile * 8} px), {overlap} overlap: {psnr:F1} dB against the whole decode, {10 * Math.Log10(255.0 * 255.0 / Math.Max(sqTile / (3.0 * n), 1e-9)):F1} dB against the reference picture");
            // measured on this latent: 32 cells 29.9 dB, 40 cells 31.7, 48 cells 35.7, 56 cells 43.3 (the 4-bit weights themselves are ~19 dB from the reference picture); the demo's 48 must stay above 33
            if (tile == 48) Ur_Expect(psnr > 33, $"48-cell tile decode PSNR {psnr:F1} dB against the whole decode");
        }
    });

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
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"text_encoder_q{F3_Bits}.safetensors")))),
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"transformer_q{F3_Bits}.safetensors")))), vae, F3_Norm(dir));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux3Request { Caption = meta.GetProperty("caption").GetString()!, Width = size, Height = size, Steps = F3_Steps ?? meta.GetProperty("steps").GetInt32(), Alpha = (float)meta.GetProperty("alpha").GetDouble(), Noise = noise,
            Progress = (s, i, n) => Console.WriteLine($"[Flux3] {s} {i}/{n} at {sw.Elapsed.TotalSeconds:F2} s") });
        double sq = 0, abs = 0; int n = size * size;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); abs += Math.Abs(have - want);
            }
        double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        Console.WriteLine($"[Flux3] pipeline on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, PSNR {psnr:F1} dB, mean |diff| {abs / (3.0 * n):F2} grey levels vs the reference picture");
        using (var f = File.Create(Path.Combine(g, $"csharp_q{F3_Bits}{(F3_Steps is { } st ? "_s" + st : "")}.ppm")))
        {
            f.Write(System.Text.Encoding.ASCII.GetBytes("P6\n" + size + " " + size + "\n255\n"));
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        if (F3_Steps == null) Ur_Expect(psnr > 20, $"picture PSNR {psnr:F1} dB vs the reference");
    });

    private static Flux2ReferenceImage F3_Reference(string path)
    {
        var (raw, shape) = Ur_Npy<float>(path);                                  // [H, W, 3] in 0..255
        int h = shape[0], w = shape[1]; var rgba = new byte[w * h * 4];
        for (int p = 0; p < w * h; p++) { rgba[p * 4] = (byte)raw[p * 3]; rgba[p * 4 + 1] = (byte)raw[p * 3 + 1]; rgba[p * 4 + 2] = (byte)raw[p * 3 + 2]; rgba[p * 4 + 3] = 255; }
        return new Flux2ReferenceImage(w, h, rgba);
    }

    private static async Task<Func<int, int, Task<InferenceSession>>> F3_EncoderFactory(Accelerator accelerator, string dir)
    {
        var bytes = await File.ReadAllBytesAsync(Path.Combine(dir, "weights", "vae_encoder.onnx"));
        return (h, w) => Task.FromResult(InferenceSession.CreateFromFile(accelerator, bytes, inputShapes: new Dictionary<string, int[]> { ["image"] = new[] { 1, 3, h, w } }));
    }

    /// <summary>Two reference pictures (one with sides that are not multiples of 16: the centre crop) as image_cond tokens against the reference autoencoder's: tokens at float noise next to bf16 rounding, ids exactly.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux3_References_MatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator); var g = Path.Combine(dir, "golden", "refs");
        if (!File.Exists(Path.Combine(g, "cond.npy"))) throw new UnsupportedTestException("FLUX 3 golden/refs not found (golden_flux3_refs.py)");
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "repo", "text_encoder", "tokenizer.json")));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        using var pipe = new Flux3Pipeline(accelerator, tok, null!, null!, vae, F3_Norm(dir)) { VaeEncoderFactory = await F3_EncoderFactory(accelerator, dir) };
        var (goldTokens, _) = Ur_Npy<float>(Path.Combine(g, "cond.npy")); var (goldIds, _) = Ur_Npy<float>(Path.Combine(g, "cond_ids.npy"));
        var (tokens, ids) = await pipe.EncodeReferencesAsync(new[] { F3_Reference(Path.Combine(g, "ref_raw_0.npy")), F3_Reference(Path.Combine(g, "ref_raw_1.npy")) });
        Ur_Expect(tokens.Length == goldTokens.Length, $"{tokens.Length / 128} reference tokens, the reference has {goldTokens.Length / 128}");
        Ur_Expect(ids.Length == goldIds.Length && Enumerable.Range(0, ids.Length).All(i => ids[i] == goldIds[i]), "position ids equal the reference's");
        double rel = F2_RelRms(tokens, goldTokens);
        Console.WriteLine($"[Flux3] reference tokens on {accelerator.AcceleratorType}: {tokens.Length / 128} tokens, relRMS {rel:E2} vs the reference autoencoder (bf16 rounded)");
        Ur_Expect(rel < 2e-2, $"reference tokens relRMS {rel:E2}");
    });

    /// <summary>The whole pipeline with a reference picture (the fox, as an edit of its own caption): runs and draws something that is neither flat nor identical to the text-only picture; the picture is saved for a look.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux3_Pipeline_WithReference_Draws() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator); var g = Path.Combine(dir, "golden", "t2i"); var gr = Path.Combine(dir, "golden", "refs");
        if (!File.Exists(Path.Combine(gr, "ref_raw_0.npy"))) throw new UnsupportedTestException("FLUX 3 golden/refs not found (golden_flux3_refs.py)");
        var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(g, "meta.json"))).RootElement;
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "repo", "text_encoder", "tokenizer.json")));
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy"));
        int size = meta.GetProperty("size").GetInt32();
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        using var pipe = new Flux3Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"text_encoder_q{F3_Bits}.safetensors")))),
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"transformer_q{F3_Bits}.safetensors")))), vae, F3_Norm(dir)) { VaeEncoderFactory = await F3_EncoderFactory(accelerator, dir) };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux3Request { Caption = meta.GetProperty("caption").GetString()!, Width = size, Height = size, Steps = meta.GetProperty("steps").GetInt32(), Alpha = (float)meta.GetProperty("alpha").GetDouble(), Noise = noise,
            References = new[] { F3_Reference(Path.Combine(gr, "ref_raw_0.npy")) }, Progress = (s, i, n) => Console.WriteLine($"[Flux3] {s} {i}/{n}") });
        var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy")); int n = size * size; double sq = 0, mean = 0;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); mean += have;
            }
        mean /= 3.0 * n; double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        double sd = 0; for (int p = 0; p < n; p++) for (int c = 0; c < 3; c++) sd += Math.Pow(result.Rgba[p * 4 + c] - mean, 2); sd = Math.Sqrt(sd / (3.0 * n));
        Console.WriteLine($"[Flux3] pipeline with a reference on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, {psnr:F1} dB against the text-only picture, grey-level spread {sd:F1}");
        using (var f = File.Create(Path.Combine(g, $"csharp_ref_q{F3_Bits}.ppm")))
        {
            f.Write(System.Text.Encoding.ASCII.GetBytes($"P6 {size} {size} 255 "));
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        Ur_Expect(sd > 20, $"the picture is not flat (spread {sd:F1})");
    });

    private const string F3Server = "http://127.0.0.1:8767/";

    /// <summary>A plain HTTP file server as a model source: every repo maps to the server root (tests only; the demo uses the hub source with its OPFS cache).</summary>
    private sealed class F3HttpSource(HttpClient http) : SpawnDev.ILGPU.ML.Hub.IModelSource
    {
        public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
        {
            if (filePath.EndsWith(".onnx")) return new MemoryStream(await http.GetByteArrayAsync(F3Server + filePath, cancellationToken));   // the ONNX reader needs a seekable stream
            return await http.GetStreamAsync(F3Server + filePath, cancellationToken);
        }
        public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) => http.GetByteArrayAsync(F3Server + filePath, cancellationToken);
    }

    /// <summary>
    /// The whole pipeline built from a model source and run on any lane with a GPU, the browser WebGPU lane included: 4-bit stages streamed from a local file server
    /// (node tools/unirig-harness/serve-files.mjs E:/spawndev-flux3/hf_upload 8767), a caption, a picture that is not blank; then the same caption with a reference picture. Timings go to the log.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux3_OverHttp_DrawsAPicture() => await RunTest(async accelerator =>
    {
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");
        try { await http.GetByteArrayAsync(F3Server + "vae_norm.json"); }
        catch (Exception e) { throw new UnsupportedTestException($"the local file server is not running ({e.GetType().Name}): node tools/unirig-harness/serve-files.mjs E:/spawndev-flux3/hf_upload 8767"); }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var pipe = await Flux3Pipeline.CreateAsync(accelerator, new F3HttpSource(http), "local");
        pipe.KeepTransformerResident = true;
        double create = sw.Elapsed.TotalSeconds; sw.Restart();
        const string caption = "Full-color studio photograph, eye-level shot of a small red ceramic teapot with a white lid, centred on a plain light grey background, soft light from the upper left.";
        Flux2Image Check(Flux2Image result, string what)
        {
            double sum = 0, sumSq = 0; int px = result.Width * result.Height;
            for (int i = 0; i < px; i++) { double l = result.Rgba[i * 4] + result.Rgba[i * 4 + 1] + result.Rgba[i * 4 + 2]; sum += l; sumSq += l * l; }
            double mean = sum / px, std = Math.Sqrt(Math.Max(0, sumSq / px - mean * mean));
            Console.WriteLine($"[Flux3] {what} on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, {result.Width}x{result.Height}, mean luminance {mean / 3:F0}, std {std / 3:F0}");
            Ur_Expect(std / 3 > 10, $"{what}: picture is flat (luminance std {std / 3:F1})");
            return result;
        }
        var first = Check(await pipe.RunAsync(new Flux3Request { Caption = caption, Seed = 3, Width = 512, Height = 512, Progress = (s, i, n) => Console.WriteLine($"[Flux3] {s} {i}/{n} at {sw.Elapsed.TotalSeconds:F1} s") }), "text to image");
        sw.Restart();
        var refPixels = new byte[256 * 192 * 4];                                     // a synthetic reference: a diagonal gradient with a dark disc
        for (int y = 0; y < 192; y++)
            for (int x = 0; x < 256; x++)
            {
                int o = (y * 256 + x) * 4; bool disc = (x - 128) * (x - 128) + (y - 96) * (y - 96) < 50 * 50;
                refPixels[o] = (byte)(disc ? 30 : 60 + x / 2); refPixels[o + 1] = (byte)(disc ? 30 : 120 + y / 3); refPixels[o + 2] = (byte)(disc ? 30 : 200 - x / 3); refPixels[o + 3] = 255;
            }
        Check(await pipe.RunAsync(new Flux3Request { Caption = caption, Seed = 3, Width = 512, Height = 384, References = new[] { new Flux2ReferenceImage(256, 192, refPixels) } }), "with a reference picture");
        Ur_Expect(first.Width == 512 && first.Height == 512, "512 px picture");
    });

    /// <summary>The prompt upsampler asks the model with Black Forest Labs' system prompt and a canvas line, and returns the caption alone (reasoning blocks, fences and quotes removed).</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Flux3_PromptUpsampler_BuildsTheRequestAndCleansTheAnswer() => await RunTest(async accelerator =>
    {
        string? gotSystem = null, gotUser = null;
        var up = new Flux3PromptUpsampler((system, user) => { gotSystem = system; gotUser = user; return Task.FromResult("<think>plan the scene</think>\n\"Full-color outdoor photograph of a red fox on snow.\"\n"); });
        var caption = await up.ExpandAsync("a red fox in the snow", 512, 384);
        Ur_Expect(caption == "Full-color outdoor photograph of a red fox on snow.", $"cleaned caption: {caption}");
        Ur_Expect(gotSystem == Flux3PromptUpsampler.TextToImagePrompt && gotSystem!.Contains("prompt upsampler of a text-to-image model"), "the system prompt is the reference's");
        Ur_Expect(gotUser == "TARGET CANVAS: 512:384\na red fox in the snow", $"user message: {gotUser}");
        await up.ExpandAsync("x");
        Ur_Expect(gotUser == "x", "no canvas line without a size");
        Ur_Expect(Flux3PromptUpsampler.Clean("```\nA caption.\n```") == "A caption.", "code fence removed");
    });

    /// <summary>The caption can be shown as it is written: the upsampler hands the pipeline's token callback the CLEANED text so far (no reasoning block, no quotes), never when the caller did not ask, and the answer it returns is unchanged.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Flux3_PromptUpsampler_StreamsTheCleanedCaptionAndReturnsTheSameAnswer() => await RunTest(async accelerator =>
    {
        const string full = "<think>plan the scene</think>\n\"Full-color outdoor photograph of a red fox on snow, frame-left.\"\n";
        bool askedForTokens = false;
        var up = new Flux3PromptUpsampler(async (system, user, onToken) =>
        {
            askedForTokens = onToken != null;
            if (onToken != null)
                for (int n = 1, i = 10; i < full.Length; n++, i += 6) await onToken(n, full[..i]);   // a model writing it in pieces (the first piece already holds the whole <think> tag: a tokenizer emits it as one token)
            return full;
        });

        var seen = new List<(int Tokens, string Text)>();
        var streamed = await up.ExpandAsync("a red fox in the snow", 512, 384, (tokens, text) => { seen.Add((tokens, text)); return Task.CompletedTask; });
        Ur_Expect(askedForTokens, "the token callback reaches the model when the caller wants the text");
        Ur_Expect(seen.Count >= 8, $"the text arrived in pieces ({seen.Count})");
        Ur_Expect(seen.Zip(seen.Skip(1), (a, b) => b.Tokens > a.Tokens).All(ok => ok), "tokens count up");
        // While the model is still reasoning, the text so far shows its <think> block growing (a progress cue the demo keeps on purpose). The moment </think> has
        // arrived, the reasoning is dropped and the CAPTION streams: from then on the text never contains the reasoning and only grows.
        int firstAnswer = seen.FindIndex(s => !s.Text.Contains("<think>"));
        Ur_Expect(firstAnswer > 0, $"the reasoning was visible first ({firstAnswer})");
        var thinking = seen.Take(firstAnswer).ToList();
        Ur_Expect(thinking.Zip(thinking.Skip(1), (a, b) => b.Text.Length >= a.Text.Length).All(ok => ok), "the reasoning grows while it is written");
        var answer = seen.Skip(firstAnswer).ToList();
        Ur_Expect(answer.Count >= 3, $"the caption arrived in pieces ({answer.Count})");
        Ur_Expect(answer.All(s => !s.Text.Contains("<think>") && !s.Text.Contains("plan the scene")), "once the reasoning has closed it is never shown again");
        Ur_Expect(answer.Zip(answer.Skip(1), (a, b) => b.Text.Length >= a.Text.Length).All(ok => ok), "the caption only grows");
        // (this fixture quotes its answer, and a piece shows the opening quote until the closing one arrives; the real system prompt says "no quotes around it")
        Ur_Expect(seen[^1].Text.TrimStart('"').StartsWith("Full-color outdoor photograph of a red fox"), $"the last piece is the caption's start: {seen[^1].Text}");

        var plain = await up.ExpandAsync("a red fox in the snow", 512, 384);
        Ur_Expect(!askedForTokens, "no token callback is passed when the caller did not ask");
        Ur_Expect(streamed == plain && plain == "Full-color outdoor photograph of a red fox on snow, frame-left.", $"streaming does not change the answer: '{streamed}' vs '{plain}'");
    });

    /// <summary>
    /// The real thing: a real GGUF model (Qwen2.5 0.5B, the one the chat tests use) writes a caption through <see cref="Flux3PromptUpsampler.FromPipeline"/> and the text arrives while it is being written, growing, ending as the returned caption.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel,WasmHeavy", RetryCount = 2)]
    public async Task Flux3_PromptUpsampler_WithARealModel_StreamsTheCaptionAsItIsWritten() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.WebGPU && accelerator.AcceleratorType != AcceleratorType.Cuda)
            throw new UnsupportedTestException($"{accelerator.AcceleratorType}: the streamed-caption check runs on the campaign target (WebGPU) + desktop reference (CUDA)");
        var http = GetHttpClient();
        if (http == null) throw new UnsupportedTestException("HttpClient not available");

        var client = new ModelsHostStreamLoader(http);
        try
        {
            var hub = client;
            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(12));
            var model = await hub.OpenAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", deselect: false, cts.Token);
            await using (model.Stream)
            using (var pipe = await SpawnDev.ILGPU.ML.Pipelines.GgufTextGenerationPipeline.CreateFromStreamAsync(accelerator, model.Stream, maxSeqLen: 2048, ct: cts.Token))
            {
                var seen = new List<(int Tokens, string Text)>();
                var caption = await Flux3PromptUpsampler.FromPipeline(pipe, maxNewTokens: 48)
                    .ExpandAsync("a red fox in the snow", 512, 512, (tokens, text) => { seen.Add((tokens, text)); return Task.CompletedTask; });
                Ur_Expect(!string.IsNullOrWhiteSpace(caption), "the model wrote a caption");
                Ur_Expect(seen.Count >= 8, $"the caption arrived token by token, not at the end ({seen.Count} callbacks)");
                Ur_Expect(seen.Zip(seen.Skip(1), (a, b) => b.Tokens > a.Tokens).All(ok => ok), "token counts increase");
                Ur_Expect(seen[0].Text.Length < caption.Length, $"the first piece ({seen[0].Text.Length} chars) is shorter than the finished caption ({caption.Length})");
                Ur_Expect(seen[^1].Text == caption || caption.StartsWith(seen[^1].Text) || seen[^1].Text.StartsWith(caption), "the last streamed piece is the caption (the final token may be an end marker)");
                Console.WriteLine($"[Flux3 streamed caption][{accelerator.AcceleratorType}] {seen.Count} pieces, {caption.Length} chars: {caption}");
            }
        }
        catch (UnsupportedTestException) { throw; }
        catch (Exception ex) when (ex.Message.Contains("No connection") || ex.Message.Contains("network")
            || ex.Message.Contains("magnet") || ex.Message.Contains("preparing") || ex is TimeoutException)
        {
            throw new UnsupportedTestException($"Hub/network unavailable: {ex.Message}");
        }
        finally { await client.DisposeAsync(); }
    });

    /// <summary>
    /// Where one denoising step goes: the real transformer (F3_BITS weights) at a 512 px picture's size (1024 image tokens + 80 text), every stage timed with a device sync on either side. Prints the table; asserts nothing
    /// but that the step ran. CUDA uses the tensor-core GEMM; OpenCL runs the portable tiled GEMM and flash attention the browser's WebGPU lane runs.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux3_Transformer_Profile() => await RunTest(async accelerator =>
    {
        var dir = F3_RequireReal(accelerator);
        int side = int.TryParse(Environment.GetEnvironmentVariable("F3_PROFILE_SIDE"), out var sd) ? sd : 512, rows = side / 16, nImg = rows * rows, nTxt = 80;
        var rng = new Random(5);
        float[] Rand(int n, float scale) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale; return a; }
        var img = Rand(nImg * Flux3Config.InChannels, 1f); var text = Rand(nTxt * 20480, 1f);
        var imageIds = Flux3Pipeline.ImageIds(rows, rows); var textIds = new float[nTxt * 4]; for (int i = 0; i < nTxt; i++) textIds[i * 4 + 3] = i;
        var loadWatch = System.Diagnostics.Stopwatch.StartNew();
        using var w = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", $"transformer_q{F3_Bits}.safetensors")));
        using var tr = new Flux3Transformer(accelerator, w, new Flux3Config(), nImg, 0, nTxt);
        await accelerator.SynchronizeAsync();
        Console.WriteLine($"[Flux3Profile] {accelerator.AcceleratorType}, {side} px ({nImg} image tokens + {nTxt} text), weights q{F3_Bits}: load {loadWatch.Elapsed.TotalSeconds:F1} s");
        tr.Forward(img, Array.Empty<float>(), text, 1f, 0.9f, imageIds, Array.Empty<float>(), textIds);       // warm: kernels compile, scratch allocates
        await accelerator.SynchronizeAsync();
        var plain = System.Diagnostics.Stopwatch.StartNew();
        tr.Forward(img, Array.Empty<float>(), text, 0.9f, 0.8f, imageIds, Array.Empty<float>(), textIds);
        await accelerator.SynchronizeAsync();
        double plainMs = plain.Elapsed.TotalMilliseconds;
        tr.Profile = new Dictionary<string, (int Calls, double Ms)>();
        tr.Forward(img, Array.Empty<float>(), text, 0.8f, 0.7f, imageIds, Array.Empty<float>(), textIds);
        await accelerator.SynchronizeAsync();
        double sum = tr.Profile.Values.Sum(v => v.Ms);
        Console.WriteLine($"[Flux3Profile] one step, unprofiled: {plainMs:F0} ms; profiled stages add up to {sum:F0} ms");
        foreach (var (name, (calls, ms)) in tr.Profile.OrderByDescending(kv => kv.Value.Ms))
            Console.WriteLine($"[Flux3Profile]   {name,-20} {calls,4} calls {ms,8:F1} ms {100 * ms / sum,5:F1} %");
        Ur_Expect(plainMs > 0, "a step ran");
    });

    /// <summary>
    /// The FLUX 3 VAE decoder (from the hub, OPFS cache) built for several tile sizes and run on random latents in the browser: every size must finish with a finite picture and the device must survive.
    /// Found in the 3D page: a decoder built for 48 x 48 latent cells lost the WebGPU device at its first tile (Chrome 151, 2 of 2 runs) while 64 x 64 ran; this is the quick repro (no transformer, no text encoder).
    /// </summary>
    [TestMethod(Timeout = 1_800_000, Category = "HeavyModel")]
    public async Task Flux3_Browser_VaeTileSizes_FinishOnWebGPU() => await RunTest(async accelerator =>
    {
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser) throw new UnsupportedTestException("browser-only (OPFS)");
        if (accelerator.AcceleratorType.ToString().IndexOf("WebGPU", StringComparison.OrdinalIgnoreCase) < 0) throw new UnsupportedTestException($"WebGPU only; this lane is {accelerator.AcceleratorType}");
        using var source = new SpawnDev.ILGPU.ML.Hub.HubModelSource(js);
        const string repo = "LostBeard/flux-3-image-klein-q4-ILGPU";
        var rng = new Random(11);
        foreach (int tile in new[] { 64, 48, 56, 40, 32 })
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var vae = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, await source.OpenAsync(repo, Flux3Pipeline.VaeFile),
                inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, tile, tile } });
            double build = sw.Elapsed.TotalSeconds; sw.Restart();
            var lat = new float[32 * tile * tile]; for (int i = 0; i < lat.Length; i++) lat[i] = (float)((rng.NextDouble() * 2 - 1) * 0.8);
            using var latBuf = accelerator.Allocate1D(lat);
            var outputs = await vae.RunAsync(new Dictionary<string, Tensor> { [vae.InputNames[0]] = new Tensor(latBuf.View, new[] { 1, 32, tile, tile }, vae.InputNames[0]) });
            var o = outputs[vae.OutputNames[0]];
            var px = await o.Data.SubView(0, 3L * (tile * 8) * (tile * 8)).CopyToHostAsync();
            int bad = 0; double sum = 0; foreach (var v in px) { if (!float.IsFinite(v)) bad++; else sum += Math.Abs(v); }
            Console.WriteLine($"[Flux3Vae] tile {tile} cells ({tile * 8} px): build {build:F1} s, decode {sw.Elapsed.TotalSeconds:F1} s, {bad} non-finite of {px.Length}, mean |value| {sum / px.Length:F3}");
            Ur_Expect(bad == 0, $"tile {tile}: {bad} non-finite values in the decoded picture");
        }

        // the pipeline's own decode of a 512 px picture's latents: tile 64 is the whole picture in one run, smaller tiles go through the gather / blend / finalize kernels
        var norm = Flux2VaeNorm.FromJson(System.Text.Encoding.UTF8.GetString(await source.FetchBytesAsync(repo, Flux3Pipeline.NormFile)));
        var packed = new float[32 * 32 * Flux3Config.InChannels]; for (int i = 0; i < packed.Length; i++) packed[i] = (float)((rng.NextDouble() * 2 - 1) * 0.8);
        foreach (int tile in new[] { 64, 48, 32 })
        {
            using var vae = await InferenceSession.CreateFromOnnxStreamAsync(accelerator, await source.OpenAsync(repo, Flux3Pipeline.VaeFile),
                inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, tile, tile } });
            var pipe = new Flux2Pipeline(accelerator, null!, null!, null!, vae, norm) { VaeTileLatent = tile };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var picture = await pipe.DecodeAsync(pipe.ToVaeInput(packed, 32, 32), 32, 32);
            int distinct = picture.Rgba.Take(4096).Distinct().Count();
            Console.WriteLine($"[Flux3Vae] pipeline decode, tile {tile} cells: {picture.Width}x{picture.Height} in {sw.Elapsed.TotalSeconds:F1} s, {distinct} distinct byte values in the first 4096");
            Ur_Expect(picture.Width == 512 && distinct > 8, $"tile {tile}: the picture is not flat");
        }
    });

    /// <summary>
    /// The weights of both big FLUX 3 stages, opened through the page's own source (HubModelSource: OPFS cache over the hub) and uploaded with <see cref="Flux2Weights.LoadAsync"/>, must go JS -> GPU and never through the .NET heap:
    /// SpawnDev.ILGPU counts every host -> device CopyFromCPU on a browser backend (<c>BrowserBufferPolicy.HostCopyBytes</c>) and the zero-copy stream path counts what it moved (<c>StreamBytes</c>). Prints both with the read / write split,
    /// and fails if more than a few MB came through the managed heap. First run downloads about 5.8 GB into the browser's OPFS.
    /// </summary>
    [TestMethod(Timeout = 7_200_000, Category = "HeavyModel")]
    public async Task Flux3_Browser_WeightsStayOutOfTheManagedHeap() => await RunTest(async accelerator =>
    {
        var js = SpawnDev.SpawnJS.SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser) throw new UnsupportedTestException("browser-only (OPFS)");
        if (accelerator.AcceleratorType.ToString().IndexOf("WebGPU", StringComparison.OrdinalIgnoreCase) < 0) throw new UnsupportedTestException($"WebGPU only; this lane is {accelerator.AcceleratorType}");
        using var source = new SpawnDev.ILGPU.ML.Hub.HubModelSource(js);
        const string repo = "LostBeard/flux-3-image-klein-q4-ILGPU";
        foreach (var file in new[] { "text_encoder_q4.safetensors", "transformer_q4.safetensors" })
        {
            // warm the OPFS cache first so the measured load is the cached one (the download is not what is being judged)
            await using (var warm = await source.OpenAsync(repo, file)) { }
            BBP.ResetHostCopyCensus();
            BBP.ResetStreamUploadTiming();
            BBP.TraceStreamUploadTiming = true;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            Flux2Weights w;
            try { w = await Flux2Weights.LoadAsync(accelerator, source, repo, file); }
            finally { BBP.TraceStreamUploadTiming = false; }
            await accelerator.SynchronizeAsync();
            double sec = sw.Elapsed.TotalSeconds;
            long heapBytes = BBP.HostCopyBytes, streamed = BBP.StreamBytes;
            Console.WriteLine($"[Flux3Weights] {file}: loaded in {sec:F1} s; JS->GPU {streamed / 1048576.0:F0} MiB in {BBP.StreamChunks} chunks (stream read {BBP.StreamReadMs / 1000:F1} s, GPU write {BBP.StreamWriteMs / 1000:F1} s); " +
                $"through the .NET heap: {heapBytes / 1048576.0:F2} MiB in {BBP.HostCopyCount} copies, largest {BBP.LargestHostCopyBytes / 1024.0:F0} KiB");
            w.Dispose();
            Ur_Expect(heapBytes < 4 * 1048576, $"{file}: {heapBytes / 1048576.0:F1} MiB of weights went through the .NET heap (the stream must be a JS stream so each chunk goes JS -> GPU)");
        }
    });
}
