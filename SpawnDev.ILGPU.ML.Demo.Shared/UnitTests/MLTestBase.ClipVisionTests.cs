using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.ClipVision;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// CLIP vision ViT-H/14 (<see cref="ClipVisionEncoder"/>, ComfyUI's clip_vision_h.safetensors from Comfy-Org/Wan_2.1_ComfyUI_repackaged; WAN14_DIR, default F:/spawndev-wan14b) against
    /// ComfyUI's CLIPVisionEncode (crop "none") of picture A (SpawnGen's dumps, float32 CPU; WAN14_DUMPS, default WAN14_DIR/comfyui-dumps): fed ComfyUI's own pixel tensor, the
    /// embeddings, the pre-norm, last_hidden_state, penultimate_hidden_states (what Wan 2.1 i2v reads), the pooled class token and image_embeds must match. Control: the
    /// state one layer earlier must not pass for the penultimate one.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task ClipVisionH_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var dir = Environment.GetEnvironmentVariable("WAN14_DIR") ?? "F:/spawndev-wan14b";
        var file = Path.Combine(dir, "comfy", "split_files", "clip_vision", "clip_vision_h.safetensors");
        var d = Environment.GetEnvironmentVariable("WAN14_DUMPS") ?? Path.Combine(dir, "comfyui-dumps");
        if (!File.Exists(file)) throw new UnsupportedTestException($"{file} not found (WAN14_DIR)");
        if (!File.Exists(Path.Combine(d, "wan_cv_cvo_penultimate_hidden_states_f32_1x257x1280.raw"))) throw new UnsupportedTestException($"SpawnGen's CLIP vision dumps not found in {d} (WAN14_DUMPS)");
        string M(string module, string shape) => Path.Combine(d, $"wan_cv_cv_{module}_{shape}.raw");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var enc = await ClipVisionEncoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(file)!), "", Path.GetFileName(file));
        Ur_Expect(enc.Width == 1280 && enc.Layers == 32 && enc.Heads == 16 && enc.ImageSize == 224 && enc.Tokens == 257 && enc.ProjectionWidth == 1024,
            $"ViT-H/14: width {enc.Width}, {enc.Layers} layers, {enc.Heads} heads, {enc.ImageSize} px, {enc.Tokens} tokens, projection {enc.ProjectionWidth}");
        var report = new List<string> { $"loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        var taps = new Dictionary<string, float[]>();
        enc.Probe = (name, view, rows, cols) => { if (name is "embeddings" or "pre_layrnorm" or "pooled" or "layer30") taps[name] = view.SubView(0, (long)rows * cols).CopyToHostAsync().GetAwaiter().GetResult(); };
        using var px = accelerator.Allocate1D(F2_Raw(M("vision_model_0_kw-pixel_values0", "f32_1x3x224x224"), 3L * 224 * 224));
        sw.Restart();
        enc.Encode(px.View);
        var last = await enc.LastHiddenState.SubView(0, 257L * 1280).CopyToHostAsync();
        var pen = await enc.PenultimateHiddenState.SubView(0, 257L * 1280).CopyToHostAsync();
        var emb = await enc.ImageEmbeds.SubView(0, 1024).CopyToHostAsync();
        double t = sw.Elapsed.TotalSeconds;
        var wantPen = F2_Raw(Path.Combine(d, "wan_cv_cvo_penultimate_hidden_states_f32_1x257x1280.raw"), 257L * 1280);
        var r = new (string Name, double E)[]
        {
            ("embeddings", F2_RelRms(taps["embeddings"], F2_Raw(M("vision_model_embeddings_0_out0", "f32_1x257x1280"), 257L * 1280))),
            ("pre_layrnorm", F2_RelRms(taps["pre_layrnorm"], F2_Raw(M("vision_model_pre_layrnorm_0_out0", "f32_1x257x1280"), 257L * 1280))),
            ("last_hidden_state", F2_RelRms(last, F2_Raw(Path.Combine(d, "wan_cv_cvo_last_hidden_state_f32_1x257x1280.raw"), 257L * 1280))),
            ("penultimate_hidden_states", F2_RelRms(pen, wantPen)),
            ("pooled (post_layernorm)", F2_RelRms(taps["pooled"], F2_Raw(M("vision_model_post_layernorm_0_out0", "f32_1x1280"), 1280))),
            ("image_embeds", F2_RelRms(emb, F2_Raw(Path.Combine(d, "wan_cv_cvo_image_embeds_f32_1x1024.raw"), 1024))),
        };
        double control = F2_RelRms(taps["layer30"], wantPen);
        report.Add($"encode {t:F1} s: " + string.Join(", ", r.Select(x => $"{x.Name} {x.E:E2}")) + $"; control (layer 30 as the penultimate) {control:E2}");
        foreach (var line in report) Console.WriteLine("[ClipVision] ViT-H vs ComfyUI: " + line);
        double worst = r.Max(x => x.E);
        // MEASURED (RTX 4070, 2026-11-15): CUDA penultimate 4.50e-5, last 4.18e-5, image_embeds 2.20e-6, embeddings 6.3e-7; OpenCL 4.43e-5 / 4.08e-5 / 2.24e-6; control 0.434
        Ur_Expect(worst < 1.5e-4 && control > 10 * worst, string.Join("; ", report));
        return string.Join("; ", report);
    });
}
