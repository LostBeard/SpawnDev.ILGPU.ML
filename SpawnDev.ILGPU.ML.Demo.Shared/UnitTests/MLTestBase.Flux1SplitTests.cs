using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// FLUX.1 from ComfyUI's standalone files (the split templates: Load Diffusion Model + DualCLIPLoader type flux + Load VAE), measured by SpawnGen on ComfyUI's flux1_krea_dev graph
/// at 256 px, 2 steps (float32 CPU; F1_DIR\comfy\comfyui-dumps\split): <c>flux1-krea-dev_fp8_scaled.safetensors</c> (8-bit floats with ComfyUI's per-matrix scale_weight),
/// <c>t5xxl_fp8_e4m3fn_scaled.safetensors</c>, <c>clip_l.safetensors</c>, all under F1_DIR\comfy\split (the Hugging Face paths). These gate the fp8_scaled paths: the transformer's
/// scales (StreamedModelWeights) and the scaled T5 (a different file from the checkpoint's T5: ComfyUI's own rows differ by 0.38).
/// </summary>
public abstract partial class MLTestBase
{
    private static (string Split, string Dumps) F1s_Require(Accelerator accelerator, string relative)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("F1_DIR") ?? "E:/spawndev-flux1";
        string split = Path.Combine(dir, "comfy", "split"), d = Path.Combine(dir, "comfy", "comfyui-dumps", "split");
        if (!File.Exists(Path.Combine(split, relative)) || !Directory.Exists(d)) throw new UnsupportedTestException($"needs {split}\\{relative} and SpawnGen's split-graph dumps in {d}");
        return (split, d);
    }

    /// <summary>The fp8_scaled T5 (<c>t5xxl_fp8_e4m3fn_scaled.safetensors</c>) against ComfyUI's rows from it (prompt and empty prompt, all 256 rows), precise activations.</summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task Flux1Split_T5Fp8Scaled_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var (split, d) = F1s_Require(accelerator, "t5xxl_fp8_e4m3fn_scaled.safetensors");
        var tokPath = Environment.GetEnvironmentVariable("F1_T5_TOKENIZER") ?? "D:/Projects/ltx/google/t5-v1_1-xxl/tokenizer.json";
        if (!File.Exists(tokPath)) throw new UnsupportedTestException($"needs the T5 tokenizer {tokPath}");
        using var fs = File.OpenRead(tokPath);
        var tok = Preprocessing.UnigramTokenizer.FromTokenizerJson(fs);
        using var t5 = await Wan.UmT5Encoder.CreateAsync(accelerator, new LocalFolderModelSource(split), "", "t5xxl_fp8_e4m3fn_scaled.safetensors");
        t5.PreciseActivations = true;
        var report = new List<string>();
        double worstReal = 0, worst = 0;
        foreach (var (prompt, dump) in new[] { ("a red apple on a wooden table", "f1s_cond_0_cond_f32_1x256x4096.raw"), ("", "f1s_cond_empty_0_cond_f32_1x256x4096.raw") })
        {
            var real = tok.Encode(Preprocessing.UnigramTokenizer.CleanWhitespace(prompt));
            var ids = new int[256]; Array.Copy(real, ids, real.Length);
            var ctx = await t5.EncodeAsync(ids);
            var want = F2_Raw(Path.Combine(d, dump), 256L * 4096);
            double all = F2_RelRms(ctx, want), realRows = F2_RelRms(ctx, want, real.Length * 4096);
            worst = Math.Max(worst, all); worstReal = Math.Max(worstReal, realRows);
            report.Add($"'{prompt}' ({real.Length} tokens): all rows {all:E2}, real tokens {realRows:E2}");
        }
        foreach (var line in report) Console.WriteLine("[Flux1Split] fp8_scaled T5 vs ComfyUI: " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): with PreciseActivations the scaled matrices run exact (WanComfyFormat.AdaptT5Async exactScaled: 8-bit values unscaled, the scale on the input
        // or on each fused part's output columns): real tokens 2.25e-4 / 7.4e-6, all rows 6.0e-4 / 1.8e-4. Folded into bf16(f8 * scale) they were 4.1e-3 / 8.3e-4.
        Ur_Expect(worstReal < 1e-3 && worst < 3e-3, string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>The standalone <c>clip_l.safetensors</c> (DualCLIPLoader's file: Hugging Face names, no prefix) through SdTextEncoder: its pooled vector against ComfyUI's y.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Flux1Split_ClipL_PooledMatchesComfyUi() => await RunTest(async accelerator =>
    {
        var (split, d) = F1s_Require(accelerator, "clip_l.safetensors");
        var tokPath = Path.Combine(SdDir, "tokenizer", "clip-vit-large-patch14", "tokenizer.json");
        if (!File.Exists(tokPath)) throw new UnsupportedTestException($"needs the CLIP tokenizer {tokPath}");
        var tok = Preprocessing.BPETokenizer.LoadFromTokenizerJson(await File.ReadAllTextAsync(tokPath));
        using var clip = await StableDiffusion.SdTextEncoder.LoadAsync(accelerator, new LocalFolderModelSource(split), "", "clip_l.safetensors");
        clip.ExactActivations = true;                                            // ComfyUI ran it in fp32
        var ids = tok.EncodeCLIP("a red apple on a wooden table");
        var plain = await clip.PooledAsync(ids, project: false);
        double e = F2_RelRms(plain, F2_Raw(Path.Combine(d, "f1s_cond_0_pooled_output_f32_1x768.raw"), 768));
        var report = $"standalone clip_l pooled vs ComfyUI's: {e:E2}";
        Console.WriteLine("[Flux1Split] " + report);
        Ur_Expect(e < 1e-5, report);                                          // MEASURED (RTX 4070, CUDA, 2026-11-14): 1.34e-6 with ExactActivations, the checkpoint's CLIP-L exactly
        return report;
    });
}
