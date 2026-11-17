using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.AceStep;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The code LM's sampling path (<see cref="AceStepCodeLm"/>: two <see cref="Flux2.Qwen3KvDecoder"/> caches, prefill + single-token steps) teacher-forced on ComfyUI's run:
    /// the positive prompt and the NEGATIVE prompt with ComfyUI's left padding stripped, then ComfyUI's first 11 chosen codes; each step's final-norm state against
    /// ComfyUI's model.norm taps (both rows). Then a short seeded generation: every code in range, the same seed gives the same codes.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_CodeLm_KvCache_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var lmPath = Ace_Require(accelerator, Path.Combine("text_encoders", "qwen_1.7b_ace15.safetensors"));
        var d = Ace_Dumps();
        using var idsDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(d, "ace_tl_codes1_norm_ids.json")));
        var calls = idsDoc.RootElement.GetProperty("calls");
        int[] Ids(int call)
        {
            var e = calls[call].GetProperty("ids");
            static IEnumerable<int> Flat(System.Text.Json.JsonElement x) => x.ValueKind == System.Text.Json.JsonValueKind.Array ? x.EnumerateArray().SelectMany(Flat) : new[] { x.GetInt32() };
            if (e.ValueKind == System.Text.Json.JsonValueKind.String) { using var inner = System.Text.Json.JsonDocument.Parse(e.GetString()!); return Flat(inner.RootElement).ToArray(); }
            return Flat(e).ToArray();
        }
        int[] pos = Ids(2), negPadded = Ids(3), neg = AceStepCodeLm.StripLeftPadding(negPadded);
        const int Steps = 11, W = 2048;
        var fed = Enumerable.Range(4, Steps).Select(c => Ids(c)[0]).ToArray();
        string Tap(int call, string shape) => Path.Combine(d, $"ace_tl_codes1_norm_qwen3_2b_transformer_model_norm_{call}_out0_f32_{shape}.raw");
        var src = new LocalFolderModelSource(Path.GetDirectoryName(lmPath)!);
        using var lm = await AceStepCodeLm.LoadAsync(accelerator, src, "", Path.GetFileName(lmPath), 512, 64);
        var report = new List<string>(); double worst = 0;
        lm.Positive.Prefill(pos); lm.Negative.Prefill(neg);
        var p0 = F2_Raw(Tap(0, "2x478x2048"), 2L * 478 * W);
        double ep = F2_RelRms(await lm.Positive.LastHidden.CopyToHostAsync(), p0[(477 * W)..(478 * W)]);
        double en = F2_RelRms(await lm.Negative.LastHidden.CopyToHostAsync(), p0[((478 + 477) * W)..((478 + 478) * W)]);
        worst = Math.Max(ep, en);
        report.Add($"prompt pass last row: positive {ep:E2}, negative (unpadded, {negPadded.Length - neg.Length} pads stripped) {en:E2}");
        var steps = new List<string>();
        for (int k = 1; k <= Steps; k++)
        {
            lm.Positive.Step(fed[k - 1]); lm.Negative.Step(fed[k - 1]);
            var tap = F2_Raw(Tap(k, "2x1x2048"), 2L * W);
            double a = F2_RelRms(await lm.Positive.LastHidden.CopyToHostAsync(), tap[..W]), b = F2_RelRms(await lm.Negative.LastHidden.CopyToHostAsync(), tap[W..]);
            worst = Math.Max(worst, Math.Max(a, b)); steps.Add($"{a:E1}/{b:E1}");
        }
        report.Add($"steps 1..{Steps} (positive / negative): {string.Join(" ", steps)}");
        // sampling: 10 codes twice with one seed: in range and reproducible
        var c1 = await lm.GenerateCodesAsync(pos, neg, 10, 1234); var c2 = await lm.GenerateCodesAsync(pos, neg, 10, 1234);
        bool same = c1.SequenceEqual(c2), inRange = c1.All(c => c >= 0 && c < AceStepCodeLm.CodeCount);
        report.Add($"seeded sampling: {string.Join(" ", c1)} (reproducible {same}, in range {inRange})");
        foreach (var line in report) Console.WriteLine("[AceStep] code LM (KV cache) vs ComfyUI: " + line);
        Ur_Expect(worst < 2e-4 && same && inRange, string.Join("; ", report));                // MEASURED 2026-11-17 CUDA: prompt 5.8e-5 / 3.8e-5, steps <= 9.2e-7
        return string.Join("; ", report);
    });

    /// <summary>
    /// The Oobleck VAE encoder (<see cref="OobleckEncoder"/>, ace_1.5_vae's encoder) against ComfyUI's VAEEncodeAudio (SpawnGen's dumps, _DevComms 2026-11-17-1100): the
    /// codes-off graph's decoded audio [2, 480000] in, the posterior [128, 250] (encoder.layers.7's output: mean | scale) and the returned latent [64, 250]. Also reported:
    /// digital silence, against ComfyUI's encode of it (ComfyUI's silence table, the refer_audio its calls carry, is NOT that encode: 3.6e-2 apart in ComfyUI itself).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_VaeEncode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("vae", "ace_1.5_vae.safetensors"));
        var d = Ace_Dumps();
        using var enc = await OobleckEncoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        Ur_Expect(enc.HopLength == 1920 && enc.LatentChannels == 64 && enc.AudioChannels == 2, $"layout hop {enc.HopLength}, latent {enc.LatentChannels}, audio {enc.AudioChannels}");
        const int frames = 250;
        var lines = new List<string>(); double worst = 0;
        foreach (var (name, audio) in new[] { ("g0audio", F2_Raw(Path.Combine(d, "ace_g0_audio_waveform_f32_1x2x480000.raw"), 2L * 480000)), ("zeros", new float[2 * 480000]) })
        {
            using var dA = accelerator.Allocate1D(audio);
            using var post = await enc.EncodeAsync(dA.View, frames);
            var ours = await post.View.CopyToHostAsync();                                        // [128, 250]
            var tap = F2_Raw(Path.Combine(d, $"ace_venc_{name}_taps_encoder_layers_7_0_f32_1x128x250.raw"), 128L * frames);
            var samples = F2_Raw(Path.Combine(d, $"ace_venc_{name}_samples_f32_1x64x250.raw"), 64L * frames);
            double ePost = F2_RelRms(ours, tap), eMean = F2_RelRms(ours[..(64 * frames)], tap[..(64 * frames)]);
            double eSamples = F2_RelRms(ours[..(64 * frames)], samples), tapMeanVsSamples = F2_RelRms(tap[..(64 * frames)], samples);
            if (name == "g0audio") worst = ePost;
            lines.Add($"{name}: posterior [128, 250] {ePost:E2} (mean {eMean:E2}); our mean vs ComfyUI's latent {eSamples:E2} (ComfyUI's own mean vs its latent {tapMeanVsSamples:E2})");
        }
        foreach (var line in lines) Console.WriteLine("[AceStep] VAE encode vs ComfyUI: " + line);
        Ur_Expect(worst < 2e-5, string.Join("; ", lines));                                    // MEASURED 2026-11-17 CUDA: posterior 3.0e-6 (zeros 1.6e-6); ComfyUI SAMPLES the posterior (its latent is 4e-4 .. 6e-4 from its own mean)
        return string.Join("; ", lines);
    });
}
