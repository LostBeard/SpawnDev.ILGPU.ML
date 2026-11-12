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
    // ---------------------------------------------------------------------------------------------------------------------------------------------------
    // DiT (split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors): the reference's names with q / k / v unfused (our export fuses
    // self_attn.qkv = cat(q, k, v) and cross_attn.kv = cat(k, v): tools/wan-harness/reference/export_dit_bf16.py), fp16 where ours is the reference's bf16,
    // modulation [1, 6, C] / head.modulation [1, 2, C] / patch_embedding [C, 48, 1, 2, 2] where ours is [6, C] / [2, C] / [C, 192] (the same element order).
    // ---------------------------------------------------------------------------------------------------------------------------------------------------

    /// <summary>True when <paramref name="w"/> is ComfyUI's Wan 2.2 DiT layout (unfused q / k / v) rather than ours.</summary>
    public static bool IsComfyDit(GpuWeights w) => w.Has("blocks.0.self_attn.q.weight") && !w.Has("blocks.0.self_attn.qkv.weight");

    private static Action<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>? _toBf16;
    private static Accelerator? _toBf16Acc;
    // fp16 -> bf16 (ILGPU's (BFloat16)float cast rounds to nearest even; a bf16 value that went through fp16 comes back exactly unless fp16 made it subnormal)
    private static void ToBf16Impl(Index1D i, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst) => dst[i] = (BFloat16)(float)src[i];
    private static void ToBf16(Accelerator acc, ArrayView1D<global::ILGPU.Half, Stride1D.Dense> src, ArrayView1D<BFloat16, Stride1D.Dense> dst)
    {
        if (!ReferenceEquals(_toBf16Acc, acc)) { _toBf16 = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<global::ILGPU.Half, Stride1D.Dense>, ArrayView1D<BFloat16, Stride1D.Dense>>(ToBf16Impl); _toBf16Acc = acc; }
        _toBf16!((int)src.Length, src, dst);
    }

    /// <summary>
    /// Rewrite ComfyUI's Wan 2.2 DiT layout into ours, in place: q / k / v fused, every tensor fp16 -> bf16, the three reshapes. One tensor at a time with a device
    /// wait before its fp16 buffer is freed, so the peak is the fp16 model plus one tensor (the 10 GB file fits a 12 GB card).
    /// </summary>
    public static async Task AdaptDitAsync(Accelerator accelerator, GpuWeights w)
    {
        async Task Retire(IDisposable? old) { if (old == null) return; await accelerator.SynchronizeAsync(); old.Dispose(); }
        int blocks = 0;
        while (w.Has($"blocks.{blocks}.self_attn.q.weight")) blocks++;
        for (int b = 0; b < blocks; b++)
        {
            foreach (var (fused, parts) in new[] { ("self_attn.qkv", new[] { "self_attn.q", "self_attn.k", "self_attn.v" }), ("cross_attn.kv", new[] { "cross_attn.k", "cross_attn.v" }) })
                foreach (var suffix in new[] { ".weight", ".bias" })
                {
                    string p = $"blocks.{b}.";
                    var names = parts.Select(x => p + x + suffix).ToArray();
                    long total = names.Sum(n => w.F16(n).Length);
                    var dst = accelerator.Allocate1D<BFloat16>(total);
                    long off = 0;
                    foreach (var n in names) { var src = w.F16(n); ToBf16(accelerator, src, dst.View.SubView(off, src.Length)); off += src.Length; }
                    var s0 = w.Shape(names[0]);
                    var shape = s0.Length == 2 ? new[] { s0[0] * names.Length, s0[1] } : new[] { (int)total };
                    w.SetBF16(p + fused + suffix, dst, shape);
                    foreach (var n in names) await Retire(w.Remove(n));
                }
        }
        foreach (var name in w.Names.ToList())
        {
            if (!w.IsF16(name)) continue;
            var s = w.Shape(name);
            int[] shape = name.EndsWith("modulation", StringComparison.Ordinal) && s.Length == 3 && s[0] == 1 ? new[] { s[1], s[2] }
                : name == "patch_embedding.weight" && s.Length == 5 ? new[] { s[0], s[1] * s[2] * s[3] * s[4] } : s;
            var src = w.F16(name);
            var dst = accelerator.Allocate1D<BFloat16>(src.Length);
            ToBf16(accelerator, src, dst.View);
            await Retire(w.SetBF16(name, dst, shape));
        }
        await accelerator.SynchronizeAsync();
    }

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
