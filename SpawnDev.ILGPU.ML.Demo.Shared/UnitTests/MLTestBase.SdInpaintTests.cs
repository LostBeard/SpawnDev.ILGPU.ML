using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// The SD 1.5 inpainting UNet (9 input channels: latent, mask, masked image latent; ComfyUI's InpaintModelConditioning) loaded with the split loader from the diffusers
    /// file (SD_DIR\inpaint\diffusion_pytorch_model.fp16.safetensors, stable-diffusion-v1-5/stable-diffusion-inpainting unet/) against diffusers' UNet2DConditionModel on
    /// the same inputs (tools/sd-harness/reference/ref_sd_inpaint.py -> ref\sd_sd15inpaint_small_f16w.npz, 32x40, fp32 math on the fp16 weights).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15Inpaint_UNet_MatchesReference() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, Sd15File);
        var unetPath = Path.Combine(SdDir, "inpaint", "diffusion_pytorch_model.fp16.safetensors");
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15inpaint_small_f16w.npz");
        if (!File.Exists(unetPath) || !File.Exists(fxPath)) throw new UnsupportedTestException($"{unetPath} or {fxPath} not found");
        await using var fs = new FileStream(unetPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var unet = await SdUNet.LoadAsync(accelerator, fs, "sd15-inpainting");
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (cc, ccs) = WanNpz(fx, "concat"); var (ctx, cs) = WanNpz(fx, "ctx");
        var t = WanNpz(fx, "timestep").Data[0]; var want = WanNpz(fx, "eps").Data;
        if (unet.InChannels != 9 || unet.ConcatChannels != 5 || ccs[0] != 5) throw new Exception($"InChannels {unet.InChannels}, ConcatChannels {unet.ConcatChannels}, fixture concat {ccs[0]}");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var eps = await unet.ForwardAsync(lat, cc, ls[1], ls[2], t, ctx, cs[0]);
        var m = WanCompare(eps, want);
        var report = $"SD 1.5 inpainting UNet ({ls[1]}x{ls[2]}, t={t}): eps relRMS {m.RelRms:E2} maxAbs {m.MaxAbs:E2} of {m.RefMaxAbs:F2}, {sw.ElapsedMilliseconds} ms (first call incl. compile)";
        // the same UNet as an LDM single file (model.diffusion_model.*, written by SaveAsync) loads through SdCheckpoint and predicts bit-identically
        var single = SdOutPath("inpaint_unet_ldm.safetensors");
        await SdSaveTo(single, s => unet.SaveAsync(s));
        bool same;
        using (var ck = await SdLoad(accelerator, single, new SdCheckpointOptions()))
        {
            var eps2 = await ck.UNet!.ForwardAsync(lat, cc, ls[1], ls[2], t, ctx, cs[0]);
            same = SdSame(eps, eps2);
            report += $"; as an LDM single file through SdCheckpoint: {ck.Family}, ConcatChannels {ck.UNet.ConcatChannels}, prediction {(same ? "bit-identical" : "DIFFERENT")}";
        }
        File.Delete(single);
        SdLog("inpainting", new[] { report });
        if (!(m.RelRms < 2e-3) || !same) throw new Exception(report);
        return report;
    });
}
