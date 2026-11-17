using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.AceStep;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
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
