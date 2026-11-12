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
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
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

    /// <summary>
    /// The DiT: ComfyUI's fp16 file adapted (q / k / v fused, fp16 -> bf16, the reshapes) against our bf16 export of the same reference weights, read from disk
    /// tensor by tensor (both are 10 GB: only the adapted one is on the device). Both are roundings of the reference's fp32 weights: ours fp32 -> bf16 directly,
    /// ComfyUI's fp32 -> fp16 (a finer grid) -> bf16 here, and that double rounding can land one bf16 step away (found 2026-11-12: e.g. bits BC73 vs BC74). So
    /// every value must be bit-identical or one bf16 step away with the same sign, or, below fp16's normal range (|x| &lt; 6.1e-5), within fp16's subnormal step.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Wan_ComfyDit_LoadsAsOurExport() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL))
            throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        string comfyDir = Path.Combine(WanDir, "comfy"), oursPath = Path.Combine(WanDir, "wan22_ti2v_5b_dit_bf16.safetensors");
        if (!File.Exists(Path.Combine(comfyDir, "wan2.2_ti2v_5B_fp16.safetensors")) || !File.Exists(oursPath))
            throw new UnsupportedTestException($"needs {comfyDir}\\wan2.2_ti2v_5B_fp16.safetensors (ComfyUI's) and our wan22_ti2v_5b_dit_bf16.safetensors in {WanDir}");
        using var comfy = await Wan22Pipeline.LoadWeightsAsync(accelerator, new LocalFolderModelSource(comfyDir), "", "wan2.2_ti2v_5B_fp16.safetensors");
        using var ours = SpawnDev.ILGPU.ML.SafeTensors.SafeTensorsFileReader.Open(oursPath);
        var oursByName = ours.Tensors.ToDictionary(t => t.Name);
        var a = oursByName.Keys.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var b = comfy.Names.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        if (!a.SequenceEqual(b)) throw new Exception($"names differ: only ours {string.Join(", ", a.Except(b).Take(5))}; only ComfyUI's {string.Join(", ", b.Except(a).Take(5))}");
        long values = 0, subnormal = 0, oneStep = 0;
        foreach (var name in a)
        {
            var t = oursByName[name];
            if (!t.Shape.SequenceEqual(comfy.Shape(name))) throw new Exception($"{name}: shape [{string.Join(",", t.Shape)}] vs [{string.Join(",", comfy.Shape(name))}]");
            if (t.DType != "BF16") throw new Exception($"{name}: our export is {t.DType}, expected BF16");
            var raw = new byte[t.DataLength];
            ours.ReadBytes(t, 0, raw);
            var yArr = await comfy.BF16(name).CopyToHostAsync();
            var (st, sub) = CompareBf16ThroughFp16(name, raw, yArr);
            oneStep += st; subnormal += sub;
            values += raw.Length / 2;
        }
        return $"{a.Length} tensors, {values:N0} values: {values - oneStep - subnormal:N0} bit-identical, {oneStep:N0} one bf16 step away ({100.0 * oneStep / values:F3}%: the double rounding through fp16), {subnormal:N0} below fp16's normal range";
    });

    // our bf16 bytes vs the adapted ComfyUI bf16 values: equal bits, one bf16 step apart with the same sign (the double rounding fp32 -> fp16 -> bf16), or below
    // fp16's normal range within its subnormal step; returns (one step, subnormal) counts
    private static (long OneStep, long Subnormal) CompareBf16ThroughFp16(string name, byte[] raw, BFloat16[] yArr)
    {
        var x = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(raw);
        var y = System.Runtime.InteropServices.MemoryMarshal.Cast<BFloat16, ushort>(yArr);
        if (x.Length != y.Length) throw new Exception($"{name}: {x.Length} vs {y.Length} values");
        long step = 0, sub = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] == y[i]) continue;
            if ((x[i] & 0x8000) == (y[i] & 0x8000) && Math.Abs(x[i] - y[i]) == 1) { step++; continue; }
            float xv = BitConverter.Int32BitsToSingle(x[i] << 16), yv = BitConverter.Int32BitsToSingle(y[i] << 16);
            if (!(Math.Abs(xv) < 6.11e-5f && Math.Abs(xv - yv) <= 6.0e-8f))
                throw new Exception($"{name}[{i}]: ours {xv:E4} vs ComfyUI's {yv:E4} (bits {x[i]:X4} / {y[i]:X4}): neither one bf16 step nor an fp16-subnormal rounding");
            sub++;
        }
        return (step, sub);
    }
}
