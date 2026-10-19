using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.2 klein 4B (black-forest-labs, Apache-2.0) against goldens written by the real diffusers / transformers classes (tools/unirig-harness/reference/golden_klein.py; weights by export_flux2.py).
/// F2_DIR default E:/spawndev-flux2 (klein4b/ = the checkpoint, weights/ = the exports, golden/cat/ = the reference run); skipped when those are not on this machine. Each stage is gated on its
/// own with the reference's previous stage as input: tokenizer ids, text embeddings, one transformer step, the VAE decode, then the whole pipeline. The reference runs in bf16 (the checkpoint's
/// dtype), this port keeps bf16 weights with fp32 activations, so agreement is bf16-noise level (a few 1e-3 relative RMS), not bit-exact.
/// </summary>
public abstract partial class MLTestBase
{
    private static string? F2Dir()
    {
        var dir = Environment.GetEnvironmentVariable("F2_DIR") ?? "E:/spawndev-flux2";
        return Directory.Exists(Path.Combine(dir, "klein4b", "tokenizer")) && File.Exists(Path.Combine(dir, "golden", "cat", "prompt_embeds.npy")) ? dir : null;
    }

    private static string F2_Require(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("FLUX.2 goldens run on the desktop GPU lanes (CUDA, OpenCL)");
        return F2Dir() ?? throw new UnsupportedTestException("FLUX.2 klein checkpoint, exports and goldens not found (F2_DIR, default E:/spawndev-flux2; tools/unirig-harness/reference/golden_klein.py, export_flux2.py)");
    }

    private static double F2_RelRms(float[] got, float[] gold, int count = -1, int offset = 0)
    {
        if (count < 0) count = gold.Length - offset;
        double num = 0, den = 0;
        for (int i = offset; i < offset + count; i++) { double d = got[i] - gold[i]; num += d * d; den += (double)gold[i] * gold[i]; }
        return Math.Sqrt(num / den);
    }

    /// <summary>Token ids of ten prompts (plain, contractions, accents, CJK, emoji, whitespace runs, newlines, an added token inside the text, empty) against the reference tokenizer, and the padded form the pipeline feeds.</summary>
    [TestMethod(Timeout = 120000)]
    public async Task Flux2_Qwen2Tokenizer_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator);
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        var cases = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(Path.Combine(dir, "golden", "tokenizer.json")));
        int n = 0;
        foreach (var c in cases.EnumerateArray())
        {
            var prompt = c.GetProperty("prompt").GetString()!;
            var text = c.GetProperty("text").GetString()!;
            var want = c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Ur_Expect(Qwen2Tokenizer.ChatPrompt(prompt) == text, $"chat template for '{prompt}'");
            var got = tok.Encode(text);
            Ur_Expect(got.SequenceEqual(want), $"ids for '{(prompt.Length > 40 ? prompt[..40] + "..." : prompt)}': got {got.Length} ids, reference {want.Length}; first difference at {Enumerable.Range(0, Math.Min(got.Length, want.Length)).FirstOrDefault(i => got[i] != want[i], -1)}");
            n++;
        }
        var (ids, real) = tok.EncodePadded("a photo of a cat sitting on a red sofa");
        var (goldIds, _) = Ur_Npy<long>(Path.Combine(dir, "golden", "cat", "prompt_ids.npy"));
        Ur_Expect(ids.Select(v => (long)v).SequenceEqual(goldIds), "padded ids of the golden prompt");
        Ur_Expect(real == 22, $"real tokens {real}");
        Console.WriteLine($"[Flux2] Qwen2 tokenizer: {n} prompts and the padded 512 ids identical to the reference");
        await Task.CompletedTask;
    });

    /// <summary>Qwen3-4B up to layer 27 on the golden prompt: the three tapped hidden states against the reference's prompt_embeds, real rows and padding rows separately.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux2_TextEncoder_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var (idsL, _) = Ur_Npy<long>(Path.Combine(g, "prompt_ids.npy")); var (mask, _) = Ur_Npy<long>(Path.Combine(g, "prompt_mask.npy"));
        var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "prompt_embeds.npy"));
        int real = (int)mask.Sum();
        var ids = idsL.Select(v => (int)v).ToArray();
        using var weights = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder.safetensors")));
        using var enc = new Flux2TextEncoder(accelerator, weights);
        enc.Encode(ids, real);
        var got = await enc.Output.CopyToHostAsync();
        var report = new List<string>();
        for (int tap = 0; tap < 3; tap++)
        {
            double a = 0, b = 0, c = 0, d = 0;
            for (int r = 0; r < 512; r++)
                for (int k = 0; k < Flux2TextEncoder.Hidden; k++)
                {
                    int i = r * Flux2TextEncoder.EmbedWidth + tap * Flux2TextEncoder.Hidden + k;
                    double e = got[i] - gold[i];
                    if (r < real) { a += e * e; b += (double)gold[i] * gold[i]; } else { c += e * e; d += (double)gold[i] * gold[i]; }
                }
            report.Add($"layer {9 * (tap + 1)}: real rows {Math.Sqrt(a / b):E2}, padding rows {Math.Sqrt(c / d):E2}");
        }
        double all = F2_RelRms(got, gold);
        Console.WriteLine($"[Flux2] text encoder on {accelerator.AcceleratorType}: {string.Join("; ", report)}; all {all:E2} (rope table bf16-rounded: {Flux2TextEncoder.RoundRopeToBf16})");
        Ur_Expect(all < 0.05, $"prompt embeddings relRMS {all:E2} vs the reference (bf16 reference, fp32 activations here)");
    });

    /// <summary>The 3.9B flow transformer on the golden step 0 (noise, prompt embeddings, t = 1000): every checkpoint the reference run kept (timestep embedding, input embeddings, each double block, single blocks 1/5/10/15/20)
    /// then the prediction, and step 1 from the reference's own latents (a different timestep).</summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Flux2_Transformer_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (embeds, _) = Ur_Npy<float>(Path.Combine(g, "prompt_embeds.npy"));
        var (ts, _) = Ur_Npy<float>(Path.Combine(g, "timesteps.npy")); var (imgIds64, _) = Ur_Npy<long>(Path.Combine(g, "img_ids.npy")); var (txtIds64, _) = Ur_Npy<long>(Path.Combine(g, "txt_ids.npy"));
        var imgIds = imgIds64.Select(v => (int)v).ToArray(); var txtIds = txtIds64.Select(v => (int)v).ToArray();
        int nImg = imgIds.Length / 4, nTxt = txtIds.Length / 4;
        using var weights = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "transformer.safetensors")));
        using var tr = new Flux2Transformer(accelerator, weights, nTxt, nImg);
        var seen = new Dictionary<string, float[]>();
        tr.Probe = (name, v, rows, cols) =>
        {
            if (name.StartsWith("single") && !new[] { "single1", "single5", "single10", "single15", "single20" }.Contains(name)) return;
            seen[name] = v.SubView(0, (long)rows * cols).GetAsArray1D();
        };
        tr.Forward(noise, embeds, ts[0], imgIds, txtIds);
        var got0 = await tr.Output.SubView(0, (long)nImg * Flux2Transformer.InChannels).CopyToHostAsync();
        var (pred0, _) = Ur_Npy<float>(Path.Combine(g, "pred_0.npy"));
        var report = new List<string>();
        void Cmp(string mine, string theirs)
        {
            var path = Path.Combine(g, theirs + ".npy");
            if (!File.Exists(path)) return;
            report.Add($"{mine} {F2_RelRms(seen[mine], Ur_Npy<float>(path).Data):E2}");
        }
        Cmp("temb", "dbg_temb"); Cmp("embedded_img", "dbg_x"); Cmp("embedded_txt", "dbg_ctx");
        for (int k = 1; k <= Flux2Transformer.DoubleBlocks; k++) { Cmp($"double{k}_img", $"dbg_double{k}_img"); Cmp($"double{k}_txt", $"dbg_double{k}_txt"); }
        foreach (var k in new[] { 1, 5, 10, 15, 20 }) Cmp($"single{k}", $"dbg_single{k}");
        double rel0 = F2_RelRms(got0, pred0);
        Console.WriteLine($"[Flux2] transformer step 0 on {accelerator.AcceleratorType}: {string.Join(", ", report)}, prediction {rel0:E2}");
        // step 1 from the reference's latents after step 0
        var (lat0, _) = Ur_Npy<float>(Path.Combine(g, "latents_0.npy")); var (pred1, _) = Ur_Npy<float>(Path.Combine(g, "pred_1.npy"));
        tr.Probe = null;
        tr.Forward(lat0, embeds, ts[1], imgIds, txtIds);
        var got1 = await tr.Output.SubView(0, (long)nImg * Flux2Transformer.InChannels).CopyToHostAsync();
        double rel1 = F2_RelRms(got1, pred1);
        Console.WriteLine($"[Flux2] transformer step 1 (t = {ts[1]}): prediction {rel1:E2}");
        Ur_Expect(rel0 < 0.05, $"step 0 prediction relRMS {rel0:E2} vs the bf16 reference");
        Ur_Expect(rel1 < 0.05, $"step 1 prediction relRMS {rel1:E2} vs the bf16 reference");
    });

    /// <summary>The flow schedule: shifted sigmas and timesteps for 4 steps at 1024 image tokens against the reference scheduler's.</summary>
    [TestMethod(Timeout = 60000)]
    public async Task Flux2_Scheduler_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var (sig, _) = Ur_Npy<float>(Path.Combine(g, "sigmas.npy")); var (ts, _) = Ur_Npy<float>(Path.Combine(g, "timesteps.npy"));
        var mine = Flux2Scheduler.Sigmas(1024, 4);
        for (int i = 0; i < sig.Length; i++) Ur_Expect(Math.Abs(mine[i] - sig[i]) < 1e-6, $"sigma {i}: {mine[i]} vs {sig[i]}");
        for (int i = 0; i < ts.Length; i++) Ur_Expect(Math.Abs(Flux2Scheduler.Timestep(mine, i) - ts[i]) < 1e-2, $"timestep {i}: {Flux2Scheduler.Timestep(mine, i)} vs {ts[i]}");
        Console.WriteLine($"[Flux2] scheduler: sigmas [{string.Join(", ", mine.Select(v => v.ToString("F6")))}] match the reference");
        await Task.CompletedTask;
    });

    private static Flux2VaeNorm F2_Norm(string dir) => Flux2VaeNorm.FromJson(File.ReadAllText(Path.Combine(dir, "weights", "vae_norm.json")));

    /// <summary>Un-normalise + unpatchify of the reference's final latents against its vae_in, then the VAE decoder (ONNX through the library's session) against the reference picture.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux2_Vae_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var (packed, _) = Ur_Npy<float>(Path.Combine(g, "latents_3.npy")); var (vaeIn, _) = Ur_Npy<float>(Path.Combine(g, "vae_in.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var bytes = File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx"));
        using var vae = InferenceSession.CreateFromFile(accelerator, bytes, inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var pipe = new Flux2Pipeline(accelerator, null!, null!, null!, vae, F2_Norm(dir));
        var mine = pipe.ToVaeInput(packed, 32);
        double relIn = F2_RelRms(mine, vaeIn);
        var got = new float[3 * 512 * 512];
        var img = await pipe.DecodeAsync(vaeIn, 32, got);
        double rel = F2_RelRms(got, image), maxAbs = 0;
        for (int i = 0; i < got.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - image[i]));
        Console.WriteLine($"[Flux2] VAE on {accelerator.AcceleratorType}: un-normalise + unpatchify {relIn:E2}, decoder relRMS {rel:E2}, max |diff| {maxAbs:F3} (range [-1,1])");
        Ur_Expect(relIn < 0.05, $"vae input relRMS {relIn:E2}");
        Ur_Expect(rel < 0.05, $"decoded picture relRMS {rel:E2}");
    });

    /// <summary>The whole pipeline from the reference's prompt and start noise: encoder, 4 transformer steps, VAE; the picture against the reference's (the reference runs bf16 throughout, so a few grey levels of drift are expected).</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_Pipeline_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        var pipe = new Flux2Pipeline(accelerator, tok,
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "text_encoder.safetensors")))),
            () => Task.FromResult(new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "transformer.safetensors")))), vae, F2_Norm(dir));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux2Request { Prompt = "a photo of a cat sitting on a red sofa", Size = 512, Steps = 4, Noise = noise, Progress = (s, i, n) => Console.WriteLine($"[Flux2] {s} {i}/{n} at {sw.Elapsed.TotalSeconds:F1} s") });
        double sq = 0, abs = 0; int n = 512 * 512;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); abs += Math.Abs(have - want);
            }
        double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        Console.WriteLine($"[Flux2] pipeline on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, PSNR {psnr:F1} dB, mean |diff| {abs / (3.0 * n):F2} grey levels vs the reference picture");
        using (var f = File.Create(Path.Combine(g, "csharp.ppm")))
        {
            var head = System.Text.Encoding.ASCII.GetBytes("P6\n512 512\n255\n"); f.Write(head);
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        Ur_Expect(psnr > 22, $"picture PSNR {psnr:F1} dB vs the reference");
    });

    /// <summary>The 4-bit block dequantiser against a CPU decode of the same words (bit-identical after bf16 rounding), on a real transformer matrix.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task Flux2_Q4Dequant_MatchesCpu() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator);
        var path = Path.Combine(dir, "weights", "transformer_q4.safetensors");
        if (!File.Exists(path)) throw new UnsupportedTestException("transformer_q4.safetensors not found (quantize_flux2.py)");
        const string name = "single_transformer_blocks.0.attn.to_out.weight";
        using var weights = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, path, n => n == name));
        Ur_Expect(weights.IsQuantized(name), "the matrix is stored quantised");
        var got = weights.Matrix(name).Cast<BFloat16, uint>().GetAsArray1D();
        // CPU decode of the same words
        using var raw = GpuWeights.LoadFile(accelerator, path, n => n == name);
        var words = raw.U32(name).GetAsArray1D(); var shape = raw.Shape(name); int N = shape[0], K = shape[1] / 5 * 32;
        static float Half(uint h) => (float)BitConverter.UInt16BitsToHalf((ushort)h);
        static uint Bf16(float f) { uint b = BitConverter.SingleToUInt32Bits(f); return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16; }
        int bad = 0;
        for (int row = 0; row < N && bad == 0; row += 37)
            for (int col = 0; col < K; col += 2)
            {
                uint P(int c) { long b5 = ((long)row * (K / 32) + c / 32) * 5; float d = Half(words[b5] & 0xFFFF), m = Half(words[b5] >> 16); int e = c % 32; return Bf16(d * ((words[b5 + 1 + e / 8] >> (4 * (e % 8))) & 15u) + m); }
                uint want = P(col) | (P(col + 1) << 16);
                if (got[((long)row * K + col) / 2] != want) { bad++; Console.WriteLine($"[Flux2] dequant mismatch at row {row} col {col}"); break; }
            }
        Console.WriteLine($"[Flux2] Q4 dequant [{N}, {K}] on {accelerator.AcceleratorType}: bit-identical to the CPU decode: {bad == 0}");
        Ur_Expect(bad == 0, "GPU dequantisation differs from the CPU decode");
        await Task.CompletedTask;
    });

    /// <summary>The whole pipeline on the 4-bit exports (whichever exist: transformer_q4 / text_encoder_q4), reference prompt and noise: the picture against the bf16 reference picture.</summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Flux2_Pipeline_Q4_StaysCloseToTheReference() => await RunTest(async accelerator =>
    {
        var dir = F2_Require(accelerator); var g = Path.Combine(dir, "golden", "cat");
        var enc = Path.Combine(dir, "weights", "text_encoder_q4.safetensors"); var trf = Path.Combine(dir, "weights", "transformer_q4.safetensors");
        if (!File.Exists(trf)) throw new UnsupportedTestException("transformer_q4.safetensors not found (quantize_flux2.py)");
        if (!File.Exists(enc)) enc = Path.Combine(dir, "weights", "text_encoder.safetensors");
        var (noise, _) = Ur_Npy<float>(Path.Combine(g, "noise.npy")); var (image, _) = Ur_Npy<float>(Path.Combine(g, "image.npy"));
        var tok = new Qwen2Tokenizer(File.ReadAllText(Path.Combine(dir, "klein4b", "tokenizer", "tokenizer.json")));
        using var vae = InferenceSession.CreateFromFile(accelerator, File.ReadAllBytes(Path.Combine(dir, "weights", "vae_decoder.onnx")), inputShapes: new Dictionary<string, int[]> { ["latents"] = new[] { 1, 32, 64, 64 } });
        long peak = 0;
        var pipe = new Flux2Pipeline(accelerator, tok,
            () => { var w = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, enc)); peak = Math.Max(peak, w.Bytes); return Task.FromResult(w); },
            () => { var w = new Flux2Weights(accelerator, GpuWeights.LoadFile(accelerator, trf)); peak = Math.Max(peak, w.Bytes); return Task.FromResult(w); }, vae, F2_Norm(dir));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipe.RunAsync(new Flux2Request { Prompt = "a photo of a cat sitting on a red sofa", Size = 512, Steps = 4, Noise = noise, Progress = (s, i, n) => Console.WriteLine($"[Flux2] {s} {i}/{n} at {sw.Elapsed.TotalSeconds:F1} s") });
        double sq = 0, abs = 0; int n = 512 * 512;
        for (int p = 0; p < n; p++)
            for (int c = 0; c < 3; c++)
            {
                double want = Math.Clamp(Math.Round((image[c * n + p] / 2 + 0.5) * 255), 0, 255), have = result.Rgba[p * 4 + c];
                sq += (have - want) * (have - want); abs += Math.Abs(have - want);
            }
        double mse = sq / (3.0 * n), psnr = 10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9));
        Console.WriteLine($"[Flux2] Q4 pipeline ({Path.GetFileName(enc)} + {Path.GetFileName(trf)}) on {accelerator.AcceleratorType}: {sw.Elapsed.TotalSeconds:F1} s, largest stage {peak / 1048576.0:F0} MiB of weights, PSNR {psnr:F1} dB, mean |diff| {abs / (3.0 * n):F2} grey levels vs the bf16 reference picture");
        using (var f = File.Create(Path.Combine(g, "csharp_q4.ppm")))
        {
            f.Write(System.Text.Encoding.ASCII.GetBytes("P6\n512 512\n255\n"));
            var rgb = new byte[3 * n]; for (int p = 0; p < n; p++) { rgb[p * 3] = result.Rgba[p * 4]; rgb[p * 3 + 1] = result.Rgba[p * 4 + 1]; rgb[p * 3 + 2] = result.Rgba[p * 4 + 2]; }
            f.Write(rgb);
        }
        Ur_Expect(psnr > 15, $"4-bit picture PSNR {psnr:F1} dB vs the reference");
    });
}
