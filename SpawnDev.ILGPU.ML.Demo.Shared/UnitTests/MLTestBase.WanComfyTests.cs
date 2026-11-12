using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// ComfyUI's repackaged Wan 2.2 files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, Apache-2.0) loaded through <see cref="Wan22Pipeline.LoadWeightsAsync"/>, which rewrites
/// ComfyUI's layout into ours (<see cref="WanComfyFormat"/>), must equal our own export of the same weights. Desktop; the files in WAN_DIR\comfy.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>
    /// The VAE: every fp16 tensor (the convolutions permuted from torch's [co, ci, k...] to [co, taps, ci], the biases) bit-identical to our export, the RMS-norm
    /// gammas (fp16 there, fp32 in ours) within fp16 rounding, the latent normalisation exactly the reference's constants, and the same set of names.
    /// </summary>
    [TestMethod]
    public async Task Wan_ComfyVae_LoadsAsOurExport() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string comfyDir = Path.Combine(WanDir, "comfy");
        if (!File.Exists(Path.Combine(comfyDir, "wan2.2_vae.safetensors")) || !File.Exists(Path.Combine(WanDir, "wan22_vae_f16.safetensors")))
            throw new UnsupportedTestException($"needs {comfyDir}\\wan2.2_vae.safetensors (ComfyUI's) and our wan22_vae_f16.safetensors in {WanDir}");
        using var comfy = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(comfyDir), "", "wan2.2_vae.safetensors");
        using var ours = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(WanDir), "", "wan22_vae_f16.safetensors");
        var a = ours.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var b = comfy.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        if (!a.SequenceEqual(b)) throw new Exception($"names differ: only ours {string.Join(", ", a.Except(b).Take(5))}; only ComfyUI's {string.Join(", ", b.Except(a).Take(5))}");
        int f16Exact = 0, f32Close = 0;
        double worstGamma = 0;
        foreach (var name in a)
        {
            if (!ours.Shape(name).SequenceEqual(comfy.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", ours.Shape(name))}] vs [{string.Join(",", comfy.Shape(name))}]");
            if (ours.IsF16(name))
            {
                if (!comfy.IsF16(name)) throw new Exception($"{name}: fp16 in ours, not in the adapted ComfyUI file");
                var x = await ours.F16(name).CopyToHostAsync(); var y = await comfy.F16(name).CopyToHostAsync();
                for (int i = 0; i < x.Length; i++)
                    if (x[i].RawValue != y[i].RawValue) throw new Exception($"{name}[{i}]: {(float)x[i]} vs {(float)y[i]} (fp16 bits differ)");
                f16Exact++;
            }
            else if (ours.IsF32(name))
            {
                if (!comfy.IsF32(name)) throw new Exception($"{name}: fp32 in ours, not in the adapted ComfyUI file");
                var x = await ours.F32(name).CopyToHostAsync(); var y = await comfy.F32(name).CopyToHostAsync();
                bool latent = name.StartsWith("latents_", StringComparison.Ordinal);
                for (int i = 0; i < x.Length; i++)
                {
                    double rel = Math.Abs(x[i] - y[i]) / Math.Max(1e-30, Math.Abs(x[i]));
                    if (latent ? x[i] != y[i] : rel > 1.0 / 2048) throw new Exception($"{name}[{i}]: {x[i]} vs {y[i]}");
                    if (!latent) worstGamma = Math.Max(worstGamma, rel);
                }
                f32Close++;
            }
            else throw new Exception($"{name}: neither fp16 nor fp32 in our export");
        }
        return $"{a.Length} tensors: {f16Exact} fp16 bit-identical, {f32Close} fp32 (gammas within fp16 rounding, worst rel {worstGamma:E1}; latent mean / std exact)";
    });
}
