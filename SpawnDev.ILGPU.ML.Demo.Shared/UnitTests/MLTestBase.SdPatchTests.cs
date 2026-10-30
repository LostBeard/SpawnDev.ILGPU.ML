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
    /// Guidance's perturbed pass (the middle block's self-attention returns v). Each patch must match its reference, move the prediction well away from the plain one
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
}
