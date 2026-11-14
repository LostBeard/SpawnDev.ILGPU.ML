using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.ZImage;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Z-Image Turbo's transformer from ComfyUI's <c>z_image_turbo_bf16.safetensors</c> against ComfyUI's own model calls of a 2-step 256 px run (SpawnGen's dumps,
/// ZI_DIR\comfy\comfyui-dumps: the transformer ran in float32 on the CPU): the same caption rows [15, 2560], model-space input [16, 32, 32] and sigma give the denoised estimate.
/// The time convention (the reference's 1 - sigma vs sigma) and the output's sign (the reference's model predicts minus the velocity) are measured: the losing readings are reported as controls.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task ZImage_Transformer_MatchesComfyUiCalls() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("ZI_DIR") ?? "E:/spawndev-zimage";
        string comfy = Path.Combine(dir, "comfy"), d = Path.Combine(comfy, "comfyui-dumps"), file = "z_image_turbo_bf16.safetensors";
        if (!File.Exists(Path.Combine(comfy, file)) || !Directory.Exists(d)) throw new UnsupportedTestException($"needs {comfy}\\{file} (Comfy-Org/z_image_turbo) and SpawnGen's dumps in {d}");
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("ZI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 7.0;
        const int rows = 16, cols = 16, capRows = 15;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var tr = await ZImageTransformer.OpenAsync(accelerator, new LocalFolderModelSource(comfy), "", file, rows, cols, capRows, (long)(residentGb * 1e9));
        var report = new List<string> { $"weights: {tr.ResidentBlocks} of 34 blocks resident, {tr.ResidentBytes / 1e9:F2} GB on the device, {tr.StreamedBytesPerForward / 1e9:F2} GB streamed per call; opened in {sw.Elapsed.TotalSeconds:F1} s" };
        var z = new ZImageKernels(accelerator);
        using var cap = accelerator.Allocate1D(F2_Raw(Path.Combine(d, "z_cond_0_cond_f32_1x15x2560.raw"), capRows * 2560L));
        using var x = accelerator.Allocate1D<float>(16L * 32 * 32); using var den = accelerator.Allocate1D<float>(16L * 32 * 32);
        double worst = 0, control = double.MaxValue;
        for (int c = 0; c < 2; c++)
        {
            float sigma = F2_Raw(Path.Combine(d, $"z_call_{c}_timestep_f32_1.raw"), 1)[0];
            var want = F2_Raw(Path.Combine(d, $"z_call_{c}_output_f32_1x16x32x32.raw"), 16L * 32 * 32);
            x.View.CopyFromCPU(F2_Raw(Path.Combine(d, $"z_call_{c}_input_f32_1x16x32x32.raw"), 16L * 32 * 32));
            var readings = new List<(string Name, double Rel)>();
            foreach (var (tName, modelTime) in c == 0 ? new[] { ("t = 1 - sigma", 1f - sigma), ("t = sigma", sigma) } : new[] { ("t = 1 - sigma", 1f - sigma) })
            {
                sw.Restart();
                await tr.ForwardAsync(x.View, cap.View, modelTime);
                foreach (var (sName, factor) in new[] { ("x + sigma * out", sigma), ("x - sigma * out", -sigma) })
                {
                    z.Denoised(x.View, tr.Output, den.View, factor, rows, cols);
                    readings.Add(($"{tName}, {sName}", F2_RelRms(await den.View.CopyToHostAsync(), want)));
                }
                report.Add($"call {c} (sigma {sigma:G5}) {tName}: forward {sw.Elapsed.TotalSeconds:F1} s");
            }
            var best = readings.OrderBy(r => r.Rel).First();
            Ur_Expect(best.Name == "t = 1 - sigma, x + sigma * out", $"call {c}: the documented convention must win, {best.Name} did");
            worst = Math.Max(worst, best.Rel);
            control = Math.Min(control, readings.Where(r => r != best).Min(r => r.Rel));
            report.Add($"call {c}: best {best.Name} {best.Rel:E2}; others {string.Join(", ", readings.Where(r => r != best).Select(r => $"{r.Name} {r.Rel:E2}"))}");
        }
        foreach (var line in report) Console.WriteLine("[ZImage] " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): t = 1 - sigma, x + sigma * out wins, 5.62e-3 and 3.56e-3 (the reference ran fp32 activations on the bf16 weights); the other readings 1.46 to 2.85
        Ur_Expect(worst < 0.012, string.Join("; ", report));
        Ur_Expect(control > 0.5, "the losing conventions must not match: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// Z-Image's text conditioning: ComfyUI's <c>qwen_3_4b.safetensors</c> (the FLUX.2 klein copy, F2_DIR\comfy) through <see cref="Flux2.Flux2TextEncoder"/> reading the hidden state after
    /// layer 35 of 36 (diffusers' <c>hidden_states[-2]</c>, no final norm), the prompt in the chat template without the thinking block, against ComfyUI's CLIPTextEncode (type lumina2)
    /// rows: 15 for "a red apple on a wooden table", 8 for the empty prompt (unpadded). The token counts are checked first; the state after layer 34 is the control that must not match.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task ZImage_TextEncoder_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var zDir = Environment.GetEnvironmentVariable("ZI_DIR") ?? "E:/spawndev-zimage";
        var f2Dir = Environment.GetEnvironmentVariable("F2_DIR") ?? "E:/spawndev-flux2";
        string d = Path.Combine(zDir, "comfy", "comfyui-dumps"), comfy = Path.Combine(f2Dir, "comfy"), file = "qwen_3_4b.safetensors", tokPath = Path.Combine(f2Dir, "klein4b", "tokenizer", "tokenizer.json");
        if (!File.Exists(Path.Combine(comfy, file)) || !File.Exists(tokPath) || !Directory.Exists(d)) throw new UnsupportedTestException($"needs {comfy}\\{file}, the Qwen tokenizer {tokPath} and SpawnGen's dumps in {d}");
        var tok = new Flux2.Qwen2Tokenizer(File.ReadAllText(tokPath));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux2.Flux2Weights.LoadAsync(accelerator, new LocalFolderModelSource(comfy), "", file, 35);
        using var enc = new Flux2.Flux2TextEncoder(accelerator, w, new[] { 34, 35 });
        const int H = Flux2.Flux2TextEncoder.Hidden;
        var report = new List<string> { $"35 layers, {w.Bytes / 1e9:F2} GB on the device, loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        double worst = 0, control = double.MaxValue;
        foreach (var (prompt, dump, rows) in new[] { ("a red apple on a wooden table", "z_cond_0_cond_f32_1x15x2560.raw", 15), ("", "z_cond_empty_0_cond_f32_1x8x2560.raw", 8) })
        {
            var (ids, real) = tok.EncodePadded(prompt, Flux2.Qwen2Tokenizer.ChatPromptNoThinking);
            Ur_Expect(real == rows, $"'{prompt}': {real} tokens, ComfyUI gave {rows} rows");
            sw.Restart();
            var all = await enc.Encode(ids, real).CopyToHostAsync();
            var want = F2_Raw(Path.Combine(d, dump), (long)rows * H);
            var l35 = new float[rows * H]; var l34 = new float[rows * H];
            for (int r = 0; r < rows; r++) { Array.Copy(all, r * 2L * H + H, l35, r * (long)H, H); Array.Copy(all, r * 2L * H, l34, r * (long)H, H); }
            double e35 = F2_RelRms(l35, want), e34 = F2_RelRms(l34, want), eTail = F2_RelRms(l35, want, (rows - 1) * H, H);
            worst = Math.Max(worst, e35); control = Math.Min(control, e34);
            report.Add($"'{prompt}' ({real} tokens, {sw.Elapsed.TotalSeconds:F1} s): after layer 35 {e35:E2} (rows 1+ {eTail:E2}), control after layer 34 {e34:E2}");
        }
        foreach (var line in report) Console.WriteLine("[ZImage] text encoder vs ComfyUI: " + line);
        // MEASURED (RTX 4070, CUDA, 2026-11-14): after layer 35 3.84e-3 and 3.86e-3 (ComfyUI ran the encoder on the CPU); the control after layer 34 0.19
        Ur_Expect(worst < 0.008 && control > 0.1, string.Join("; ", report));
        return string.Join("; ", report);
    });
}
