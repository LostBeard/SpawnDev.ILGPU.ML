using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.AceStep;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ACE-Step 1.5 (ComfyUI's audio_ace_step_1_5_split template) against SpawnGen's ComfyUI dumps (float32 CPU; _DevComms 2026-11-17-0010; Plans/ace-step-1.5-2026-11-16.md).
/// Files: ACE_DIR (default F:/spawndev-acestep): comfy/split_files/{vae, text_encoders, diffusion_models}/..., comfyui-dumps/ace_*.
/// </summary>
public abstract partial class MLTestBase
{
    private static string Ace_Dir() => Environment.GetEnvironmentVariable("ACE_DIR") ?? "F:/spawndev-acestep";
    private static string Ace_Dumps() => Path.Combine(Ace_Dir(), "comfyui-dumps");

    private static string Ace_Require(Accelerator accelerator, string relative)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var p = Path.Combine(Ace_Dir(), "comfy", "split_files", relative);
        if (!File.Exists(p)) throw new UnsupportedTestException($"{p} not found (Comfy-Org/ACE-Step-1.5)");
        return p;
    }

    /// <summary>
    /// The Oobleck VAE decoder (<see cref="OobleckDecoder"/>, ace_1.5_vae) against ComfyUI's VAEDecodeAudio of the codes-off graph's final latent [1, 64, 250]:
    /// stereo [2, 480000] at 48 kHz (1920 samples per latent frame).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task AceStep_VaeDecode_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Ace_Require(accelerator, Path.Combine("vae", "ace_1.5_vae.safetensors"));
        var d = Ace_Dumps();
        var latent = F2_Raw(Path.Combine(d, "ace_g0_out_samples_f32_1x64x250.raw"), 64L * 250);
        var expected = F2_Raw(Path.Combine(d, "ace_g0_audio_waveform_f32_1x2x480000.raw"), 2L * 480000);
        using var vae = await OobleckDecoder.LoadAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        Ur_Expect(vae.HopLength == 1920 && vae.LatentChannels == 64 && vae.AudioChannels == 2, $"layout hop {vae.HopLength}, latent {vae.LatentChannels}, audio {vae.AudioChannels}");
        using var dLat = accelerator.Allocate1D(latent);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var audio = await vae.DecodeAsync(dLat.View, 250);
        var ours = await audio.View.CopyToHostAsync();
        double e = F2_RelRms(ours, expected);
        float peakOurs = ours.Max(MathF.Abs), peakTheirs = expected.Max(MathF.Abs);
        string report = $"audio [2, 480000] {e:E2} (peak ours {peakOurs:F4}, ComfyUI {peakTheirs:F4}; {sw.Elapsed.TotalSeconds:F1} s)";
        Console.WriteLine("[AceStep] VAE decode vs ComfyUI: " + report);
        Ur_Expect(e < 2e-5, report);                                                          // MEASURED 2026-11-16 CUDA: 3.1e-6 (peaks equal, 0.6337)
        return report;
    });
}
