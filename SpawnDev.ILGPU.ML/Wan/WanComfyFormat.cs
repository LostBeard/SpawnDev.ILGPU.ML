using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnDev.ILGPU.ML.Wan;

/// <summary>
/// ComfyUI's repackaged Wan 2.2 files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, Apache-2.0: the official weights in ComfyUI's tensor layout) loaded as they are:
/// <see cref="Wan22Pipeline.LoadWeightsAsync(Accelerator, Hub.IModelSource, string, string, CancellationToken)"/> recognises the layout and rewrites the tensors
/// into this library's on the device, so a ComfyUI workflow's own files run without a separate export. Read from the files' headers only (2026-11-12).
/// VAE (<c>split_files/vae/wan2.2_vae.safetensors</c>): the same tensor names as our export; the convolutions are torch's [co, ci, kt, kh, kw] where ours are the
/// gather layout [co, taps, ci] (<c>tools/wan-harness/reference/export_vae_f16.py</c>), the RMS-norm gammas are fp16 [C, 1, 1(, 1)] where ours are fp32 [C], and
/// the latent normalisation (<c>latents_mean</c> / <c>latents_std</c>, the constants of the reference's <c>Wan2_2_VAE</c>) is not in the file.
/// </summary>
public static class WanComfyFormat
{
    /// <summary>True when <paramref name="w"/> is ComfyUI's Wan 2.2 VAE layout (torch-shaped convolutions, no latent normalisation) rather than ours.</summary>
    public static bool IsComfyVae(GpuWeights w) => !w.Has("latents_mean") && w.Has("conv1.weight") && w.Shape("conv1.weight").Length == 5;

    /// <summary>Rewrite ComfyUI's Wan 2.2 VAE layout into ours, in place (see the class summary). Returns once the device has finished with the old buffers.</summary>
    public static async Task AdaptVaeAsync(Accelerator accelerator, GpuWeights w)
    {
        var sk = new SparseKernels(accelerator);
        var displaced = new List<IDisposable>();
        try
        {
            foreach (var name in w.Names.ToList())
            {
                var s = w.Shape(name);
                if (name.EndsWith(".gamma", StringComparison.Ordinal) && w.IsF16(name))
                {
                    // fp16 [C, 1, 1(, 1)] -> fp32 [C]: tiny, through the host
                    var h = await w.F16(name).CopyToHostAsync();
                    var f = new float[h.Length];
                    for (int i = 0; i < h.Length; i++) f[i] = (float)h[i];
                    if (w.SetF32(name, accelerator.Allocate1D(f), new[] { f.Length }) is { } old) displaced.Add(old);
                }
                else if (name.EndsWith(".weight", StringComparison.Ordinal) && s.Length >= 4 && w.IsF16(name))
                {
                    // torch conv [co, ci, k...] -> the gather layout [co, taps, ci] (taps = the product of the kernel dims; a 1-tap kernel is the same element order)
                    int co = s[0], ci = s[1], taps = 1;
                    for (int d = 2; d < s.Length; d++) taps *= s[d];
                    if (taps == 1) { w.SetShape(name, new[] { co, 1, ci }); continue; }
                    var dst = accelerator.Allocate1D<global::ILGPU.Half>((long)co * taps * ci);
                    sk.PermuteConvWeight(w.F16(name), dst.View, ci, co, taps);
                    if (w.SetF16(name, dst, new[] { co, taps, ci }) is { } old) displaced.Add(old);
                }
            }
            if (w.SetF32("latents_mean", accelerator.Allocate1D(LatentsMean), new[] { 48 }) is { } m) displaced.Add(m);
            if (w.SetF32("latents_std", accelerator.Allocate1D(LatentsStd), new[] { 48 }) is { } d2) displaced.Add(d2);
            await accelerator.SynchronizeAsync();        // the permutes have read the old buffers
        }
        finally
        {
            foreach (var b in displaced) b.Dispose();
        }
    }

    // the reference Wan2_2_VAE's latent normalisation (decode: z * std + mean), as our export stores it (fp32, shortest round-trip literals)
    private static readonly float[] LatentsMean = { -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.157f, -0.0098f, 0.0375f, -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.123f, -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.052f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f };
    private static readonly float[] LatentsStd = { 0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.499f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.06f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f };
}
