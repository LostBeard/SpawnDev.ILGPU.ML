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
    /// Guidance's perturbed pass (the middle block's self-attention returns v) and HyperTile: ComfyUI's node (queries tiled, each query tile attending to the band of untiled tokens
    /// with its index, as ComfyUI's attention reshapes k / v by q's batch; SpawnGen found the difference: 40 % from ComfyUI's picture with true tiles) and true tiled attention
    /// (<c>TileKeysAndValues</c>). Each patch must match its reference, move the prediction well away from the plain one (it ran), and switching it off must give the plain
    /// prediction bit-identically.
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
            ("hypertile_kv", on => unet.HyperTile = on ? new SdHyperTile(TileSize: 128, SwapSize: 1, MaxDepth: 1, TileKeysAndValues: true) : null),
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
    /// Self-Attention Guidance's pieces (ComfyUI's SelfAttentionGuidance) on SD 1.5 at 32x40 against diffusers with ComfyUI's code (ref_sd_patches.py): recording the
    /// middle block's first self-attention leaves the prediction bit-identical; its per-key mass (softmax mean over heads, summed over queries) matches; and
    /// SdGuidance.SagBlur reproduces ComfyUI's create_blur_map, from the reference mass (the host math alone) and from ours (the whole chain).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_SagCaptureAndBlurMap_MatchComfyUI() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15patches_small_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException($"{fxPath} not found (tools/sd-harness/reference/ref_sd_patches.py)");
        using var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadTextEncoder = false, LoadVae = false });
        var unet = ck.UNet!;
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        int C = ls[0], lh = ls[1], lw = ls[2];
        var plain = await unet.ForwardAsync(lat, lh, lw, t, ctx, cs[0]);
        unet.RecordMiddleSelfAttention = true;
        var rec = await unet.ForwardAsync(lat, lh, lw, t, ctx, cs[0]);
        var mass = await unet.ReadMiddleAttentionAsync();
        unet.RecordMiddleSelfAttention = false;
        var refMass = WanNpz(fx, "sag_mass").Data; var refDeg = WanNpz(fx, "sag_degraded").Data;
        if (mass == null || mass.Length != refMass.Length) throw new Exception($"mass: {mass?.Length} tokens, reference {refMass.Length}");
        var mm = WanCompare(mass, refMass);
        var degRef = SdGuidance.SagBlur(lat, C, lh, lw, refMass, 2f, 1f);
        var degOurs = SdGuidance.SagBlur(lat, C, lh, lw, mass, 2f, 1f);
        var dr = WanCompare(degRef, refDeg); var dOurs = WanCompare(degOurs, refDeg);
        int above = refMass.Count(v => v > 1f), flips = mass.Zip(refMass).Count(p => p.First > 1f != p.Second > 1f);
        double blurEffect = WanCompare(refDeg, lat).RelRms;
        var report = $"recording: {(SdSame(plain, rec) ? "prediction unchanged" : "PREDICTION CHANGED")}; mass ({mass.Length} tokens, {above} above 1) {mm.RelRms:E2}, threshold flips {flips}; "
            + $"blur map from the reference mass {dr.RelRms:E2}, from ours {dOurs.RelRms:E2} (the blur changes x0 by {blurEffect:E2})";
        SdLog("SAG", new[] { report });
        if (!(SdSame(plain, rec) && mm.RelRms < 2e-3 && dr.RelRms < 1e-5 && flips == 0 && dOurs.RelRms < 1e-5 && above > 0 && blurEffect > 1e-2)) throw new Exception(report);
        return report;
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
