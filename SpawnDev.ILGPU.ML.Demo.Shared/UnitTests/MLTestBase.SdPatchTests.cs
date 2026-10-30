using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ComfyUI's UNet patches on SD 1.5 against diffusers with ComfyUI's code put in (tools/sd-harness/reference/ref_sd_patches.py -> ref\sd_sd15patches_small_f16w.npz,
    /// 32x40 latent: the deepest level is 4x5, so the Fourier window meets an odd side): FreeU (1.1, 1.2, 0.9, 0.2), FreeU_V2 (1.3, 1.4, 0.9, 0.2) and Perturbed-Attention
    /// Guidance's perturbed pass (the middle block's self-attention returns v) and HyperTile (tiled self-attention). Each patch must match its reference, move the prediction well away from the plain one
    /// (it ran), and switching it off must give the plain prediction bit-identically.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_UNetPatches_FreeUAndPag_MatchComfyUI() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15patches_small_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"{fxPath} not found (tools/sd-harness/reference/ref_sd_patches.py)");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        Task<float[]> Eps() => unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var report = new List<string>();
        bool ok = true;
        var plain = await Eps();
        var mp = WanCompare(plain, WanNpz(fx, "eps_plain").Data);
        report.Add($"plain {mp.RelRms:E2}");
        ok &= mp.RelRms < 2e-3;
        foreach (var (name, setup) in new (string, Action<bool>)[]
        {
            ("freeu", on => unet.FreeU = on ? new SdFreeU(1.1f, 1.2f, 0.9f, 0.2f) : null),
            ("freeu2", on => unet.FreeU = on ? new SdFreeU(1.3f, 1.4f, 0.9f, 0.2f, V2: true) : null),
            ("pag", on => unet.PerturbMiddleSelfAttention = on),
            // 32x40 latent, tile 128 px = 16 latent: the full-resolution self-attention runs in 2 x 2 tiles (ComfyUI's transposed h x w grid); depth 1 stays whole
            ("hypertile", on => unet.HyperTile = on ? new SdHyperTile(TileSize: 128, SwapSize: 1, MaxDepth: 1) : null),
        })
        {
            setup(true);
            var got = await Eps();
            setup(false);
            var want = WanNpz(fx, "eps_" + name).Data;
            var m = WanCompare(got, want);
            double moved = WanCompare(want, plain).RelRms;
            bool off = SdSame(plain, await Eps());
            report.Add($"{name} {m.RelRms:E2} (maxAbs {m.MaxAbs:E2}; the reference is {moved:E2} from plain; off again: {(off ? "plain" : "DIFFERENT")})");
            ok &= m.RelRms < 2e-3 && moved > 20 * m.RelRms && off;
        }
        SdLog("UNet patches", report);
        if (!ok) throw new Exception("UNet patches: " + string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// A latent whose sides are not multiples of 8 (36x44: the up blocks resize to the next skip connection's size, ComfyUI's Upsample(output_shape)) and ComfyUI's
    /// PatchModelAddDownscale (deep shrink: block 3, factor 2, after skip, bicubic) on SD 1.5 against diffusers with ComfyUI's code put in
    /// (tools/sd-harness/reference/ref_sd_deepshrink.py -> ref\sd_sd15shrink_36x44_f16w.npz). Switching deep shrink off gives the plain prediction bit-identically.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_DeepShrinkAndOddLatent_MatchComfyUI() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15shrink_36x44_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"{fxPath} not found (tools/sd-harness/reference/ref_sd_deepshrink.py)");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        Task<float[]> Eps() => unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var plain = await Eps();
        var mp = WanCompare(plain, WanNpz(fx, "eps_plain").Data);
        unet.DeepShrink = new SdDeepShrink();
        var shrink = await Eps();
        unet.DeepShrink = null;
        var want = WanNpz(fx, "eps_shrink").Data;
        var ms = WanCompare(shrink, want);
        double moved = WanCompare(want, plain).RelRms;
        bool off = SdSame(plain, await Eps());
        var report = $"{ls[1]}x{ls[2]} latent: plain {mp.RelRms:E2}; deep shrink {ms.RelRms:E2} (maxAbs {ms.MaxAbs:E2}; the reference is {moved:E2} from plain); off again: {(off ? "plain" : "DIFFERENT")}";
        SdLog("deep shrink", new[] { report });
        if (!(mp.RelRms < 2e-3 && ms.RelRms < 2e-3 && moved > 20 * ms.RelRms && off)) throw new Exception(report);
        return report;
    });
}
