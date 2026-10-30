using System.IO.Compression;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SafeTensors;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Stable Diffusion model merging and saving (ComfyUI's ModelMerge* / CLIPMerge* / CheckpointSave / ModelSave / CLIPSave / VAESave, for SpawnGen). The saved files are
/// checked on the host against the SOURCE files, tensor by tensor: a save must reproduce every tensor the loaders read, bit for bit at fp16, under the source's own name
/// and shape; a merge must be ComfyUI's formula on the fp16 weights. Files in SD_DIR (see MLTestBase.SdTests.cs); desktop GPU lanes.
/// </summary>
public abstract partial class MLTestBase
{
    /// <summary>A .safetensors file on the host: its tensors by name, read back as fp16 bit patterns (the source type rounded to fp16 as the loader does).</summary>
    private sealed class SdHostFile : IDisposable
    {
        private readonly FileStream _fs;
        public readonly Dictionary<string, SafeTensorInfo> Tensors;
        public SdHostFile(string path)
        {
            _fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
            var lenBytes = new byte[8]; _fs.ReadExactly(lenBytes);
            long hs = BitConverter.ToInt64(lenBytes);
            var header = new byte[hs]; _fs.ReadExactly(header);
            var (list, _) = SafeTensorsFileReader.ParseHeader(Path.GetFileName(path), header, 8 + hs, _fs.Length - 8 - hs);
            Tensors = list.ToDictionary(t => t.Name, StringComparer.Ordinal);
        }
        public ushort[] Half(string name)
        {
            var t = Tensors[name];
            var raw = new byte[t.DataLength];
            _fs.Position = t.DataOffset; _fs.ReadExactly(raw);
            switch (t.DType)
            {
                case "F16": { var h = new ushort[raw.Length / 2]; Buffer.BlockCopy(raw, 0, h, 0, raw.Length); return h; }
                case "F32":
                    {
                        var h = new ushort[raw.Length / 4];
                        for (int i = 0; i < h.Length; i++) h[i] = BitConverter.HalfToUInt16Bits((System.Half)BitConverter.ToSingle(raw, i * 4));
                        return h;
                    }
                case "BF16":
                    {
                        var h = new ushort[raw.Length / 2];
                        for (int i = 0; i < h.Length; i++) h[i] = BitConverter.HalfToUInt16Bits((System.Half)BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, i * 2) << 16));
                        return h;
                    }
                default: throw new NotSupportedException($"{name}: {t.DType}");
            }
        }
        public void Dispose() => _fs.Dispose();
    }

    private static bool SdSame(float[] a, float[] b) => a.AsSpan().SequenceEqual(b);

    private static float SdH(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);

    /// <summary>fp16 bit patterns a and b differ by at most <paramref name="ulps"/> units in the last place (same sign; +0 / -0 equal).</summary>
    private static bool SdNearHalf(ushort a, ushort b, int ulps)
    {
        if (a == b || ((a | b) & 0x7FFF) == 0) return true;
        if ((a & 0x8000) != (b & 0x8000)) return false;
        return Math.Abs((a & 0x7FFF) - (b & 0x7FFF)) <= ulps;
    }

    private static string SdOutPath(string file)
    {
        var dir = Path.Combine(SdDir, "out"); Directory.CreateDirectory(dir);
        return Path.Combine(dir, file);
    }

    private static async Task SdSaveTo(string path, Func<Stream, Task> save)
    {
        await using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
        await save(fs);
    }

    /// <summary>
    /// CheckpointSave round trip on SD 1.5 (HF CLIP text encoder, fp32 file) and SD 2.1 base (OpenCLIP text encoder: the fused in_proj, the transposed text_projection):
    /// load, save as fp16, then on the host every saved tensor must be the source file's tensor of the same name and shape rounded to fp16, bit for bit; every source
    /// tensor of the UNet and VAE must be saved, and the text tensors not saved may only be ones the loader does not read. The saved file loads back and its UNet gives a
    /// bit-identical noise prediction.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd_Save_RoundTripsCheckpoints() => await RunTest(async accelerator =>
    {
        RequireSdFile(accelerator, Sd15File);
        var report = new List<string>();
        foreach (var (file, fxName, tag) in new[] { (Sd15File, "sd_sd15_small_f16w.npz", "sd15"), (Sd21BaseFile, "sd_sd21b_small_f16s.npz", "sd21b") })
        {
            var src = Path.Combine(SdDir, file);
            var fxPath = Path.Combine(SdDir, "ref", fxName);
            if (!File.Exists(src) || !File.Exists(fxPath)) throw new UnsupportedTestException($"{file} or {fxName} not found in {SdDir}");
            var outPath = SdOutPath($"save_roundtrip_{tag}.safetensors");
            using var fx = ZipFile.OpenRead(fxPath);
            var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
            float[] before;
            using (var ck = await SdLoad(accelerator, src, new SdCheckpointOptions()))
            {
                before = await ck.UNet!.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
                await SdSaveTo(outPath, s => ck.SaveAsync(s));
            }
            // host: every saved tensor = the source's, rounded to fp16
            int checkedN = 0, badShape = 0; long badValues = 0;
            var notInSource = new List<string>(); var missing = new List<string>();
            using (var a = new SdHostFile(src))
            using (var b = new SdHostFile(outPath))
            {
                foreach (var (name, ti) in b.Tensors)
                {
                    if (!a.Tensors.TryGetValue(name, out var si)) { notInSource.Add(name); continue; }
                    if (!si.Shape.SequenceEqual(ti.Shape) || ti.DType != "F16") { badShape++; continue; }
                    var want = a.Half(name); var got = b.Half(name);
                    for (int i = 0; i < want.Length; i++) if (want[i] != got[i]) badValues++;
                    checkedN++;
                }
                foreach (var name in a.Tensors.Keys)
                {
                    if (b.Tensors.ContainsKey(name)) continue;
                    bool weight = name.StartsWith("model.diffusion_model.", StringComparison.Ordinal) || name.StartsWith("first_stage_model.", StringComparison.Ordinal)
                        || (name.StartsWith("cond_stage_model.", StringComparison.Ordinal) && !name.EndsWith(".position_ids", StringComparison.Ordinal) && !name.EndsWith("logit_scale", StringComparison.Ordinal));
                    if (weight) missing.Add(name);
                }
                report.Add($"{tag}: {b.Tensors.Count} tensors saved ({new FileInfo(outPath).Length / 1048576} MB), {checkedN} compared, {badValues} values differ, {badShape} shape/dtype wrong, "
                    + $"{notInSource.Count} not in the source{(notInSource.Count > 0 ? " (" + string.Join(", ", notInSource.Take(3)) + ")" : "")}, "
                    + $"{missing.Count} weights not saved{(missing.Count > 0 ? " (" + string.Join(", ", missing.Take(3)) + ")" : "")}");
            }
            // the saved file loads back and gives the same prediction
            float[] after;
            using (var ck2 = await SdLoad(accelerator, outPath, new SdCheckpointOptions()))
                after = await ck2.UNet!.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            bool same = before.AsSpan().SequenceEqual(after);
            report.Add($"{tag}: reloaded UNet prediction {(same ? "bit-identical" : $"DIFFERENT ({WanCompare(after, before).RelRms:E2})")}");
            if (badValues != 0 || badShape != 0 || notInSource.Count != 0 || missing.Count != 0 || !same || checkedN == 0)
                throw new Exception("save round trip failed: " + string.Join("; ", report));
            File.Delete(outPath);
        }
        SdLog("save round trip", report);
        return string.Join("; ", report);
    });

    /// <summary>
    /// ModelMergeBlocks (input 0.2, middle 0.5, out 0.8) on the SD 2.1 base UNet with the 768-v UNet, and CLIPMergeSimple 0.3 on their text encoders: every saved
    /// tensor must be ComfyUI's <c>W1 * ratio + W2 * (1 - ratio)</c> on the two files' fp16 weights (computed on the host in fp32, within 1 fp16 unit in the last place), with
    /// each UNet tensor taking its block's ratio. ResetAsync then gives the original UNet's prediction bit-identically.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd21_MergeBlocksAndClip_MatchComfyFormulas() => await RunTest(async accelerator =>
    {
        var pathA = RequireSdFile(accelerator, Sd21BaseFile);
        var pathB = Path.Combine(SdDir, Sd21vFile);
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd21b_small_f16s.npz");
        if (!File.Exists(pathB) || !File.Exists(fxPath)) throw new UnsupportedTestException($"{Sd21vFile} or the SD 2.1 fixture not found");
        var opt = new SdCheckpointOptions { LoadVae = false };
        var outPath = SdOutPath("merge_sd21.safetensors");
        var blocks = new[] { new KeyValuePair<string, float>("input", 0.2f), new("middle", 0.5f), new("out", 0.8f) };
        const float clipRatio = 0.3f;
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        var report = new List<string>();
        var resetPath = SdOutPath("merge_sd21_reset.safetensors");
        var intactPath = SdOutPath("merge_sd21_a_after_b.safetensors");
        bool resetSame;
        using (var a = await SdLoad(accelerator, pathA, opt))
        {
            var plain = await a.UNet!.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            var plain2 = await a.UNet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            bool repeatable = SdSame(plain, plain2);
            report.Add($"the plain prediction twice: {(repeatable ? "bit-identical" : $"DIFFERENT ({WanCompare(plain2, plain).RelRms:E2}, max {WanCompare(plain2, plain).MaxAbs:E2})")}");
            using (var b = await SdLoad(accelerator, pathB, opt))
            {
                // loading a second model must leave the first one's weights untouched (OpenCL 10-30: a reset left garbage in A's buffers)
                await SdSaveTo(intactPath, s => a.SaveAsync(s, new SdSaveOptions { VPrediction = false }));
                var ru = await a.UNet.MergeAsync(b.UNet!, SdMerge.Blocks(blocks));
                var rt = await a.TextEncoder!.MergeAsync(b.TextEncoder!, SdMerge.Simple(clipRatio));
                report.Add($"merged {ru.Merged} UNet tensors ({ru.NoCounterpart.Count} without counterpart), {rt.Merged} text tensors ({rt.NoCounterpart.Count})");
                if (ru.NoCounterpart.Count != 0 || rt.NoCounterpart.Count != 0) throw new Exception("merge skipped tensors: " + string.Join(", ", ru.NoCounterpart.Concat(rt.NoCounterpart).Take(5)));
            }
            var merged = await a.UNet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            await SdSaveTo(outPath, s => a.SaveAsync(s, new SdSaveOptions { VPrediction = false }));
            await a.UNet.ResetAsync(); await a.TextEncoder!.ResetAsync();
            var reset = await a.UNet.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
            resetSame = SdSame(plain, reset);
            report.Add($"merged prediction vs plain {WanCompare(merged, plain).RelRms:E2}; after ResetAsync {(resetSame ? "bit-identical" : $"DIFFERENT ({WanCompare(reset, plain).RelRms:E2}, max {WanCompare(reset, plain).MaxAbs:E2})")}");
            // every tensor after the reset must be the file's again (fp16-rounded): which ones are not, and whether they hold the merged values or something else,
            // says how a reset went wrong
            await SdSaveTo(resetPath, s => a.SaveAsync(s, new SdSaveOptions { VPrediction = false }));
        }
        var damaged = new List<string>();
        using (var src = new SdHostFile(pathA))
        using (var ia = new SdHostFile(intactPath))
            foreach (var name in ia.Tensors.Keys)
            {
                var w = src.Half(name); var g = ia.Half(name);
                int diff = 0, firstBad = -1, lastBad = -1;
                for (int i = 0; i < w.Length; i++) if (w[i] != g[i]) { diff++; if (firstBad < 0) firstBad = i; lastBad = i; }
                if (diff > 0) damaged.Add($"{name} [{string.Join("x", ia.Tensors[name].Shape)}]: {diff} of {w.Length} differ, elements {firstBad}..{lastBad}");
            }
        File.Delete(intactPath);
        SdLog("merge sd21 A after loading B", damaged.Count == 0 ? new[] { "every tensor of A is the file's" } : damaged);
        report.Add($"A after loading B: {damaged.Count} tensors differ from A's file");
        var notRestored = new List<string>();
        using (var src = new SdHostFile(pathA))
        using (var mg = new SdHostFile(outPath))
        using (var rs = new SdHostFile(resetPath))
            foreach (var name in rs.Tensors.Keys)
            {
                var w = src.Half(name); var g = rs.Half(name); var m = mg.Half(name);
                int diff = 0, isMerged = 0, firstBad = -1, lastBad = -1;
                for (int i = 0; i < w.Length; i++)
                    if (w[i] != g[i]) { diff++; if (g[i] == m[i]) isMerged++; if (firstBad < 0) firstBad = i; lastBad = i; }
                if (diff > 0) notRestored.Add($"{name} [{string.Join("x", rs.Tensors[name].Shape)}]: {diff} of {w.Length} differ ({isMerged} hold the merged value), elements {firstBad}..{lastBad}");
            }
        File.Delete(resetPath);
        SdLog("merge sd21 reset", notRestored.Count == 0 ? new[] { "after ResetAsync every tensor is the file's" } : notRestored);
        report.Add($"after ResetAsync: {notRestored.Count} tensors differ from the file{(notRestored.Count > 0 ? " (first: " + notRestored[0] + "; all in sd_tests.log)" : "")}");
        bool back = resetSame && notRestored.Count == 0 && damaged.Count == 0;
        long total = 0, exact = 0, off1 = 0, bad = 0, changed = 0; int tensors = 0;
        string? worst = null;
        var ratioUsed = new Dictionary<float, int>();
        using (var fa = new SdHostFile(pathA))
        using (var fb = new SdHostFile(pathB))
        using (var fo = new SdHostFile(outPath))
        {
            foreach (var name in fo.Tensors.Keys)
            {
                float r;
                if (name.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                {
                    var k = name["model.diffusion_model.".Length..];
                    r = blocks[0].Value; int best = 0;
                    foreach (var (p, v) in blocks) if (p.Length > best && k.StartsWith(p, StringComparison.Ordinal)) { r = v; best = p.Length; }
                }
                else r = clipRatio;
                ratioUsed[r] = ratioUsed.GetValueOrDefault(r) + 1;
                var w1 = fa.Half(name); var w2 = fb.Half(name); var got = fo.Half(name);
                long badHere = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    var want = BitConverter.HalfToUInt16Bits((System.Half)(SdH(w1[i]) * r + SdH(w2[i]) * (1f - r)));
                    if (w1[i] != w2[i]) changed++;
                    if (want == got[i]) exact++;
                    else if (SdNearHalf(want, got[i], 1)) off1++;
                    else badHere++;
                }
                if (badHere > 0) { bad += badHere; worst ??= $"{name} ({badHere})"; }
                total += got.Length; tensors++;
            }
        }
        report.Add($"saved {tensors} tensors, {total} values: {exact} exact, {off1} within 1 ulp, {bad} off{(worst != null ? " first " + worst : "")}; {changed} differ between the two models; "
            + $"ratios used {string.Join(", ", ratioUsed.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}x{kv.Value}"))}");
        SdLog("merge sd21", report);
        if (bad != 0 || changed == 0 || ratioUsed.Count != 4 || tensors == 0 || !back) throw new Exception("merge check failed: " + string.Join("; ", report));
        File.Delete(outPath);
        return string.Join("; ", report);
    });

    /// <summary>
    /// Merges and LoRAs apply in call order, as in ComfyUI's graph (SD 1.5 loaded twice, LCM-LoRA): after a merge, a LoRA set is replaceable and comes off back to the
    /// merged weights; a merge on top of a LoRA bakes it in (LoRA x0.8 then ModelMergeSimple 0.5 with the plain model = the LoRA at x0.4, and taking the LoRAs off
    /// changes nothing); ResetAsync returns to the file's weights bit-identically.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task Sd15_MergeAndLora_ApplyInCallOrder() => await RunTest(async accelerator =>
    {
        var path = RequireSdFile(accelerator, Sd15File);
        var loraPath = Path.Combine(SdDir, "lora", "lcm-lora-sdv1-5.safetensors");
        var fxPath = Path.Combine(SdDir, "ref", "sd_sd15_small_f16w.npz");
        if (!File.Exists(loraPath) || !File.Exists(fxPath)) throw new UnsupportedTestException("LoRA or SD 1.5 fixture not found");
        FileStream Open(string f) => new(f, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        SdLoraSpec Lcm(float s) => new(ct => Task.FromResult<Stream>(Open(loraPath)), "lcm", s);
        using var fx = ZipFile.OpenRead(fxPath);
        var (lat, ls) = WanNpz(fx, "latent"); var (ctx, cs) = WanNpz(fx, "ctx"); var t = WanNpz(fx, "timestep").Data[0];
        await using var s1 = Open(path); using var a = await SdUNet.LoadAsync(accelerator, s1, Sd15File);
        await using var s2 = Open(path); using var b = await SdUNet.LoadAsync(accelerator, s2, Sd15File);
        Task<float[]> Eps() => a.ForwardAsync(lat, ls[1], ls[2], t, ctx, cs[0]);
        var report = new List<string>();
        var checks = new List<(string What, bool Ok)>();

        var plain = await Eps();
        await a.ApplyLorasAsync(new[] { Lcm(0.8f) }); var lora08 = await Eps();
        await a.ApplyLorasAsync(new[] { Lcm(0.4f) }); var lora04 = await Eps();
        await a.ApplyLorasAsync(Array.Empty<SdLoraSpec>());
        checks.Add(("LoRA off = plain", SdSame(plain, await Eps())));

        // merge -> LoRA: merging with the same weights at 0.5 is the identity in fp16, then the LoRA is replaceable and comes off to the merged weights
        await a.MergeAsync(b, SdMerge.Simple(0.5f));
        checks.Add(("merge with itself = plain", SdSame(plain, await Eps())));
        await a.ApplyLorasAsync(new[] { Lcm(0.8f) });
        checks.Add(("merge then LoRA x0.8 = LoRA x0.8", SdSame(lora08, await Eps())));
        await a.ApplyLorasAsync(Array.Empty<SdLoraSpec>());
        checks.Add(("its LoRA off = merged", SdSame(plain, await Eps())));

        // LoRA -> merge: (W + 0.8 D) * 0.5 + W * 0.5 = W + 0.4 D, and the LoRA is baked in
        await a.ApplyLorasAsync(new[] { Lcm(0.8f) });
        await a.MergeAsync(b, SdMerge.Simple(0.5f));
        var baked = await Eps();
        double vs04 = WanCompare(baked, lora04).RelRms, vs08 = WanCompare(baked, lora08).RelRms, ref04 = WanCompare(lora04, plain).RelRms;
        report.Add($"LoRA x0.8 then merge 0.5: vs LoRA x0.4 {vs04:E2}, vs x0.8 {vs08:E2} (x0.4 vs plain {ref04:E2})");
        checks.Add(("LoRA then merge = LoRA x0.4", vs04 < ref04 / 20));
        await a.ApplyLorasAsync(Array.Empty<SdLoraSpec>());
        checks.Add(("baked LoRA stays", SdSame(baked, await Eps())));

        await a.ResetAsync();
        checks.Add(("reset = plain", SdSame(plain, await Eps())));
        report.Add(string.Join(", ", checks.Select(c => $"{c.What}: {(c.Ok ? "ok" : "FAIL")}")));
        SdLog("merge + LoRA order", report);
        if (checks.Any(c => !c.Ok)) throw new Exception("merge / LoRA order failed: " + string.Join("; ", report));
        return string.Join("; ", report);
    });
}
