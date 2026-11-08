using System.IO.Compression;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// ComfyUI's attention-multiply nodes (Lore, SpawnGen) on SD 1.5: UNetSelfAttentionMultiply (q 1.5, k 0.5, v 2, out 0.8), UNetCrossAttentionMultiply (q 0.7, k 1.3,
    /// v 1, out 1.2) and CLIPAttentionMultiply (q 1.1, k 0.9, v 1.4, out 0.6) through <see cref="SdUNet.ScaleAsync"/> / <see cref="SdTextEncoder.ScaleAsync"/>. Every saved
    /// tensor must be the file's weight times the factor ComfyUI's OWN key rules give its checkpoint name (suffix <c>{attn}.to_q.weight</c>, <c>self_attn.q_proj.bias</c>,
    /// ...; computed on the host in fp32, within 1 fp16 unit in the last place - ComfyUI rounds stochastically), q / k / v scaled inside the fused qkv tensors; and
    /// ResetAsync must give the plain prediction back bit for bit.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task<string> Sd15_AttentionMultiply_MatchComfyFormulas() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15_small_f16w.npz");
        if (!File.Exists(fxPath)) throw new UnsupportedTestException("SD 1.5 fixture not found");
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        var self = SdMerge.UNetAttentionMultiply("attn1", 1.5f, 0.5f, 2f, 0.8f);
        var cross = SdMerge.UNetAttentionMultiply("attn2", 0.7f, 1.3f, 1f, 1.2f);
        var clip = SdMerge.CLIPAttentionMultiply(1.1f, 0.9f, 1.4f, 0.6f);

        // ComfyUI's own key rules on the CHECKPOINT names (nodes_attention_multiply.py), independent of this library's name mapping
        static float? ComfyUNet(string key, string attn, float q, float k, float v, float o)
        {
            if (key.EndsWith($"{attn}.to_q.bias") || key.EndsWith($"{attn}.to_q.weight")) return q;
            if (key.EndsWith($"{attn}.to_k.bias") || key.EndsWith($"{attn}.to_k.weight")) return k;
            if (key.EndsWith($"{attn}.to_v.bias") || key.EndsWith($"{attn}.to_v.weight")) return v;
            if (key.EndsWith($"{attn}.to_out.0.bias") || key.EndsWith($"{attn}.to_out.0.weight")) return o;
            return null;
        }
        static float? ComfyClip(string key)
        {
            if (key.EndsWith("self_attn.q_proj.weight") || key.EndsWith("self_attn.q_proj.bias")) return 1.1f;
            if (key.EndsWith("self_attn.k_proj.weight") || key.EndsWith("self_attn.k_proj.bias")) return 0.9f;
            if (key.EndsWith("self_attn.v_proj.weight") || key.EndsWith("self_attn.v_proj.bias")) return 1.4f;
            if (key.EndsWith("self_attn.out_proj.weight") || key.EndsWith("self_attn.out_proj.bias")) return 0.6f;
            return null;
        }

        var report = new List<string>();
        var outPath = SdOutPath("attn_multiply_sd15.safetensors");
        bool resetSame;
        using (var ck = await SdLoad(accelerator, path, new SdCheckpointOptions { LoadVae = false }))
        {
            var unet = ck.UNet!; var text = ck.TextEncoder!;
            var plain = await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            int nu = await unet.ScaleAsync(n => self(n) ?? cross(n));
            int nt = await text.ScaleAsync(clip);
            var scaled = await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            report.Add($"scaled {nu} UNet tensors, {nt} text tensors; prediction moved {WanCompare(scaled, plain).RelRms:E2}");
            await SdSaveTo(outPath, s => ck.SaveAsync(s, new SdSaveOptions { VPrediction = false }));
            await unet.ResetAsync(); await text.ResetAsync();
            resetSame = SdSame(plain, await unet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]));
            report.Add($"after ResetAsync: {(resetSame ? "bit-identical" : "DIFFERENT")}");
        }
        long total = 0, exact = 0, off1 = 0, bad = 0; int scaledTensors = 0; string? worst = null;
        using (var src = new SdHostFile(path))
        using (var got = new SdHostFile(outPath))
            foreach (var name in got.Tensors.Keys)
            {
                float f = 1f;
                if (name.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                    f = ComfyUNet(name, "attn1", 1.5f, 0.5f, 2f, 0.8f) ?? ComfyUNet(name, "attn2", 0.7f, 1.3f, 1f, 1.2f) ?? 1f;
                else if (name.StartsWith("cond_stage_model.", StringComparison.Ordinal))
                    f = ComfyClip(name) ?? 1f;
                if (f != 1f) scaledTensors++;
                var w = src.Half(name); var g = got.Half(name);
                long badHere = 0;
                for (int i = 0; i < g.Length; i++)
                {
                    var want = BitConverter.HalfToUInt16Bits((System.Half)(SdH(w[i]) * f));
                    if (want == g[i]) exact++;
                    else if (SdNearHalf(want, g[i], 1)) off1++;
                    else badHere++;
                }
                if (badHere > 0) { bad += badHere; worst ??= $"{name} x{f} ({badHere} of {g.Length})"; }
                total += g.Length;
            }
        File.Delete(outPath);
        report.Add($"saved {total} values: {exact} exact, {off1} within 1 ulp, {bad} off{(worst != null ? " first " + worst : "")}; {scaledTensors} checkpoint tensors carry a factor");
        SdLog("attention multiply sd15", report);
        // SD 1.5: 16 transformer blocks x (attn1 q, k, v, out w + b = 5; attn2 q, k, out w + b = 4, v is x1) = 144; CLIP-L 12 layers x 8 = 96
        if (bad != 0 || !resetSame || scaledTensors != 144 + 96) throw new Exception("attention multiply check failed: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
