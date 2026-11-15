using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.QwenImage;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>SpawnGen's Qwen-Image Edit 2509 dumps of ComfyUI (QE_DUMPS, default F:/spawndev-qwenimage-edit/comfyui-dumps; _DevComms 2026-11-15-1425 Lore -> Tuvok).</summary>
    private static string Qe_Dumps()
    {
        var d = Environment.GetEnvironmentVariable("QE_DUMPS") ?? "F:/spawndev-qwenimage-edit/comfyui-dumps";
        if (!Directory.Exists(d)) throw new UnsupportedTestException($"SpawnGen's Qwen-Image Edit dumps not found in {d} (QE_DUMPS)");
        return d;
    }

    private static int[] Qe_I64(string path)
    {
        var b = File.ReadAllBytes(path);
        var o = new int[b.Length / 8];
        for (int i = 0; i < o.Length; i++) o[i] = checked((int)BitConverter.ToInt64(b, i * 8));
        return o;
    }

    /// <summary>The 2509 edit transformer file (QE_DIR, default F:/spawndev-qwenimage-edit, Comfy-Org/Qwen-Image-Edit_ComfyUI's split_files layout under comfy).</summary>
    private static string Qe_Transformer(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("reads the files from disk: a desktop GPU");
        var p = Path.Combine(Environment.GetEnvironmentVariable("QE_DIR") ?? "F:/spawndev-qwenimage-edit", "comfy", "split_files", "diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors");
        if (!File.Exists(p)) throw new UnsupportedTestException($"{p} not found (QE_DIR)");
        return p;
    }

    /// <summary>
    /// A latent [16, h, w] as transformer tokens [(h'/2) (w'/2), 64] (column c * 4 + dy * 2 + dx, as <see cref="Flux1.Flux1Kernels.Patchify"/>), odd sides padded to even by
    /// <paramref name="pad"/> ("zero", "replicate" = the edge row / column, "circular" = wrap to row / column 0), optionally normalised per channel ((v - mean) / std).
    /// </summary>
    private static (float[] Tokens, int Rows, int Cols) Qe_Tokens(float[] lat, int h, int w, string pad, float[]? mean, float[]? std)
    {
        int H = h + (h & 1), W = w + (w & 1), rows = H / 2, cols = W / 2;
        var t = new float[(long)rows * cols * 64];
        for (int c = 0; c < 16; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int sy = y, sx = x; bool zero = false;
                    if (y >= h) { if (pad == "zero") zero = true; else sy = pad == "replicate" ? h - 1 : y - h; }
                    if (x >= w) { if (pad == "zero") zero = true; else sx = pad == "replicate" ? w - 1 : x - w; }
                    float v = zero ? 0f : lat[(long)c * h * w + (long)sy * w + sx];
                    if (mean != null && !zero) v = (v - mean[c]) / std![c];
                    t[((long)(y / 2) * cols + x / 2) * 64 + c * 4 + (y & 1) * 2 + (x & 1)] = v;
                }
        return (t, rows, cols);
    }

    /// <summary>
    /// Qwen-Image Edit 2509's transformer (qwen_image_edit_2509_fp8_e4m3fn: the base model's architecture, its own weights) with reference latents
    /// (<see cref="QwenImageTransformer.OpenAsync(Accelerator, IModelSource, string, string, int, int, int, IReadOnlyList{ValueTuple{int, int}}, long, CancellationToken)"/>) against
    /// ComfyUI's model calls (SpawnGen's dumps, float32 CPU, sigma 1, cfg 4): 1 picture (reference A [16, 111, 148], cond 212 / uncond 204 rows) and 2 pictures (A and B
    /// [16, 99, 165], 417 / 409 rows). How ComfyUI feeds a reference is measured on call 0 of the 1-picture run: odd sides padded with zeros, the edge or by wrapping, and the raw VAE
    /// latent or the model's normalised one; the winner then has to hold for the other three calls. Base model conventions (QwenImage_Transformer_MatchesComfyUiCalls): t = sigma,
    /// the output is x - sigma v.
    /// </summary>
    [TestMethod(Timeout = 7200000, Category = "HeavyModel")]
    public async Task QwenEdit_Transformer_MatchesComfyUiCalls() => await RunTest(async accelerator =>
    {
        var path = Qe_Transformer(accelerator); var d = Qe_Dumps();
        var vaePath = Qi_Require(accelerator, Path.Combine("vae", "qwen_image_vae.safetensors"));
        double residentGb = double.TryParse(Environment.GetEnvironmentVariable("QI_RESIDENT_GB"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 6.0;
        const int lat = 16 * 24 * 32;
        float[] mean, std;
        using (var vw = GpuWeights.LoadFile(accelerator, vaePath))
        {
            await Wan.WanComfyFormat.AdaptVaeAsync(accelerator, vw);
            using var vae = new Wan.WanVae(accelerator, vw);
            (mean, std) = await vae.GetLatentNormalizationAsync();
        }
        var refA = F2_Raw(Path.Combine(d, "qe_graph1_call_0_c_ref_latents_0_f32_1x16x1x111x148.raw"), 16L * 111 * 148);
        var refB = F2_Raw(Path.Combine(d, "qe_graph2_call_0_c_ref_latents_1_f32_1x16x1x99x165.raw"), 16L * 99 * 165);
        var k = new Flux1.Flux1Kernels(accelerator);
        var report = new List<string>(); double worst = 0, control = double.MaxValue; string? winner = null;
        var src = new LocalFolderModelSource(Path.GetDirectoryName(path)!);

        async Task<double> Call(QwenImageTransformer tr, string run, int call, int n, List<float[]> refs, string pad, bool normalise)
        {
            var x = F2_Raw(Path.Combine(d, $"qe_{run}_call_{call}_input_f32_1x16x1x24x32.raw"), lat);
            float sigma = F2_Raw(Path.Combine(d, $"qe_{run}_call_{call}_timestep_f32_1.raw"), 1)[0];
            var parts = new List<float[]> { Qe_Tokens(x, 24, 32, "zero", null, null).Tokens };
            int[] rh = { 111, 99 }, rw = { 148, 165 };
            for (int r = 0; r < refs.Count; r++) parts.Add(Qe_Tokens(refs[r], rh[r], rw[r], pad, normalise ? mean : null, normalise ? std : null).Tokens);
            var tokens = parts.SelectMany(p => p).ToArray();
            Ur_Expect(tokens.Length == (long)tr.InputTokens * 64, $"{tokens.Length / 64} tokens, the transformer takes {tr.InputTokens}");
            using var dTok = accelerator.Allocate1D(tokens); using var dX = accelerator.Allocate1D(x); using var den = accelerator.Allocate1D<float>(lat);
            using var cond = accelerator.Allocate1D(F2_Raw(Path.Combine(d, $"qe_{run}_call_{call}_c_c_crossattn_f32_1x{n}x3584.raw"), (long)n * 3584));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await tr.ForwardAsync(dTok.View, cond.View, sigma, n);
            k.Denoised(dX.View, tr.Output, den.View, sigma, 12, 16);
            double e = F2_RelRms(await den.View.CopyToHostAsync(), F2_Raw(Path.Combine(d, $"qe_{run}_call_{call}_output_f32_1x16x1x24x32.raw"), lat));
            report.Add($"{run} call {call} ({n} rows, {refs.Count} ref, pad {pad}, {(normalise ? "normalised" : "raw")}): {e:E2} ({sw.Elapsed.TotalSeconds:F1} s)");
            return e;
        }

        // the rotary layout of text + pictures: where the text starts (after the largest of ALL pictures, diffusers; or after the noise latent alone) and which frame a
        // reference sits on (k, diffusers' index; or 0); pictures centred as the base model
        static int[] Layout(int nTxt, (int R, int C)[] pics, bool textAfterAll, bool refFrameK)
        {
            int start = 0;
            foreach (var (r, c) in textAfterAll ? pics : pics.Take(1)) start = Math.Max(start, Math.Max(r / 2, c / 2));
            var ids = new int[(nTxt + pics.Sum(p => p.R * p.C)) * 3];
            for (int j = 0; j < nTxt; j++) ids[j * 3] = ids[j * 3 + 1] = ids[j * 3 + 2] = start + j;
            int row = nTxt;
            for (int q = 0; q < pics.Length; q++)
            {
                var (r, c) = pics[q]; int y0 = r - r / 2, x0 = c - c / 2;
                for (int y = 0; y < r; y++) for (int x = 0; x < c; x++) { ids[row * 3] = refFrameK ? q : 0; ids[row * 3 + 1] = y - y0; ids[row * 3 + 2] = x - x0; row++; }
            }
            return ids;
        }
        var layouts = new (string Name, bool All, bool FrameK)[] { ("text after all pictures, ref k on frame k (diffusers)", true, true), ("text after the noise latent, ref k on frame k", false, true),
            ("text after all pictures, refs on frame 0", true, false), ("text after the noise latent, refs on frame 0", false, false) };
        // ComfyUI's reference format, measured on SpawnGen's img_in tap (_DevComms 2026-11-15-1730): the reference latent as the call carries it (not normalised again), odd sides
        // wrapped (circular); 0.0 from ComfyUI's packed rows (zero padding 0.11, edge 0.20)
        const string Pad = "circular"; const bool Norm = false;
        var one = new[] { (12, 16), (56, 74) }; var two = new[] { (12, 16), (56, 74), (50, 83) };
        Ur_Expect(Layout(212, one, false, true).SequenceEqual(QwenImageTransformer.Ids(212, one)) && Layout(212, one, true, true).SequenceEqual(QwenImageTransformer.Ids(212, one, true)),
            "the library's Ids must be the 'text after the noise latent, ref k on frame k' layout (and diffusers' with textAfterAllPictures)");
        (string Name, bool All, bool FrameK, double E) best;
        // 1 picture: the layout on call 0, then call 1 with the winner
        using (var tr = await QwenImageTransformer.OpenAsync(accelerator, src, "", Path.GetFileName(path), 212, 12, 16, new[] { (56, 74) }, (long)(residentGb * 1e9)))
        {
            report.Add($"1 picture: {tr.ResidentBlocks} of 60 blocks resident, {tr.InputTokens} picture tokens");
            var readings = new List<(string Name, bool All, bool FrameK, double E)>();
            foreach (var (name, all, frameK) in layouts)
            {
                tr.UsePositionIds(Layout(212, one, all, frameK), 212);
                double e = await Call(tr, "graph1", 0, 212, new List<float[]> { refA }, Pad, Norm);
                report.Add($"  layout: {name}");
                readings.Add((name, all, frameK, e));
            }
            best = readings.OrderBy(r => r.E).First();
            winner = best.Name; worst = best.E; control = readings.Where(r => r != best).Min(r => r.E);
            report.Add($"layout: '{best.Name}' wins ({best.E:E2}); the next best {control:E2}");
            Ur_Expect(!best.All && best.FrameK, $"ComfyUI's layout (the library default) must win, '{best.Name}' did: " + string.Join("; ", report));
            tr.ResetPositionIds();                                                                  // the library's own Ids from here on
            worst = Math.Max(worst, await Call(tr, "graph1", 1, 204, new List<float[]> { refA }, Pad, Norm));
        }
        // control: the noise latent alone (no reference) must miss
        using (var tr0 = await QwenImageTransformer.OpenAsync(accelerator, src, "", Path.GetFileName(path), 212, 12, 16, (long)(residentGb * 1e9)))
        {
            double e0 = await Call(tr0, "graph1", 0, 212, new List<float[]>(), Pad, Norm);
            report.Add($"control, no reference latent: {e0:E2}");
            control = Math.Min(control, e0);
        }
        // 2 pictures, the same layout
        using (var tr2 = await QwenImageTransformer.OpenAsync(accelerator, src, "", Path.GetFileName(path), 417, 12, 16, new[] { (56, 74), (50, 83) }, (long)(residentGb * 1e9)))
        {
            report.Add($"2 pictures: {tr2.InputTokens} picture tokens");
            // which frame the SECOND reference sits on: k = 2 (the library default) or 1, like the first; measured on call 0
            int[] RefsOnFrame1(int n)
            {
                var ids = QwenImageTransformer.Ids(n, two);
                for (int row = n + 12 * 16 + 56 * 74; row < n + tr2.InputTokens; row++) ids[row * 3] = 1;
                return ids;
            }
            double eK = await Call(tr2, "graph2", 0, 417, new List<float[]> { refA, refB }, Pad, Norm);
            report.Add("  2 pictures, reference k on frame k (1, 2)");
            tr2.UsePositionIds(RefsOnFrame1(417), 417);
            double e1 = await Call(tr2, "graph2", 0, 417, new List<float[]> { refA, refB }, Pad, Norm);
            report.Add("  2 pictures, both references on frame 1");
            tr2.ResetPositionIds();
            report.Add($"second reference: frame 2 {eK:E2}, frame 1 {e1:E2}");
            worst = Math.Max(worst, Math.Min(eK, e1)); control = Math.Min(control, Math.Max(eK, e1));
            if (e1 < eK) tr2.UsePositionIds(RefsOnFrame1(409), 409);
            worst = Math.Max(worst, await Call(tr2, "graph2", 1, 409, new List<float[]> { refA, refB }, Pad, Norm));
        }
        foreach (var line in report) Console.WriteLine("[QwenEdit] transformer vs ComfyUI: " + line);
        Ur_Expect(worst < 1e-3 && control > 10 * worst, string.Join("; ", report));             // bound to be pinned to the measurement
        return string.Join("; ", report);
    });

    /// <summary>
    /// Qwen2.5-VL's vision tower (<see cref="QwenVisionEncoder"/>, the visual.* tensors of ComfyUI's qwen_2.5_vl_7b_fp8_scaled) against ComfyUI's TextEncodeQwenImageEditPlus run
    /// (SpawnGen's dumps, float32 CPU): fed ComfyUI's own patch rows (picture A 768 x 1176, grid [24, 32]; picture B 792 x 1176, grid [22, 36]), the patch embedding, the merger's
    /// input (the last block's output, window order) and the tower's output [tokens / 4, 3584] must match. A alone, then A and B in ONE call (ComfyUI runs one call per picture:
    /// full attention per picture and windows that never cross pictures must give the same rows).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "HeavyModel")]
    public async Task QwenEdit_VisionTower_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var path = Qi_Require(accelerator, Path.Combine("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")); var d = Qe_Dumps();
        string V(string run, int pic, string module, string shape) => Path.Combine(d, $"qe_{run}_vis_qwen25_7b_transformer_visual{module}_{pic}_{shape}.raw");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux2Weights.LoadVisionAsync(accelerator, new LocalFolderModelSource(Path.GetDirectoryName(path)!), "", Path.GetFileName(path));
        using var enc = new QwenVisionEncoder(accelerator, w, 768 + 792);
        var report = new List<string> { $"vision tower {w.Bytes / 1e9:F2} GB on the device, loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        var gA = Qe_I64(V("te1", 0, "", "in1_i64_1x3")); var gB = Qe_I64(V("te2", 1, "", "in1_i64_1x3"));
        Ur_Expect(gA.SequenceEqual(new[] { 1, 24, 32 }) && gB.SequenceEqual(new[] { 1, 22, 36 }), $"grids {string.Join(",", gA)} / {string.Join(",", gB)}");
        var inA = F2_Raw(V("te1", 0, "", "in0_f32_768x1176"), 768L * 1176); var inB = F2_Raw(V("te2", 1, "", "in0_f32_792x1176"), 792L * 1176);
        double worst = 0;
        // A alone, with the intermediate checkpoints
        float[]? embed = null, last = null;
        enc.Probe = (name, view, rows, cols) =>
        {
            if (name == "embed") embed = view.SubView(0, (long)rows * cols).CopyToHostAsync().GetAwaiter().GetResult();
            if (name == $"block{QwenVisionEncoder.Depth}") last = view.SubView(0, (long)rows * cols).CopyToHostAsync().GetAwaiter().GetResult();
        };
        using (var dIn = accelerator.Allocate1D(inA))
        {
            sw.Restart();
            var outA = await enc.EncodePatches(dIn.View, new[] { (24, 32) }).CopyToHostAsync();
            double t = sw.Elapsed.TotalSeconds;
            double eEmb = F2_RelRms(embed!, F2_Raw(V("te1", 0, "_patch_embed", "out0_f32_768x1280"), 768L * 1280));
            double eLast = F2_RelRms(last!, F2_Raw(V("te1", 0, "_merger", "in0_f32_768x1280"), 768L * 1280));
            double eOut = F2_RelRms(outA, F2_Raw(V("te1", 0, "", "out0_f32_192x3584"), 192L * 3584));
            worst = Math.Max(worst, eOut);
            report.Add($"A alone ({t:F1} s): patch embed {eEmb:E2}, merger input (block {QwenVisionEncoder.Depth}) {eLast:E2}, output [192, 3584] {eOut:E2}");
        }
        enc.Probe = null;
        // A and B together
        var both = new float[inA.Length + inB.Length]; inA.CopyTo(both, 0); inB.CopyTo(both, inA.Length);
        using (var dIn = accelerator.Allocate1D(both))
        {
            sw.Restart();
            var outAB = await enc.EncodePatches(dIn.View, new[] { (24, 32), (22, 36) }).CopyToHostAsync();
            double t = sw.Elapsed.TotalSeconds;
            double eA = F2_RelRms(outAB.AsSpan(0, 192 * 3584).ToArray(), F2_Raw(V("te2", 0, "", "out0_f32_192x3584"), 192L * 3584));
            double eB = F2_RelRms(outAB.AsSpan(192 * 3584, 198 * 3584).ToArray(), F2_Raw(V("te2", 1, "", "out0_f32_198x3584"), 198L * 3584));
            worst = Math.Max(worst, Math.Max(eA, eB));
            report.Add($"A + B in one call ({t:F1} s): A {eA:E2}, B [198, 3584] {eB:E2}");
        }
        foreach (var line in report) Console.WriteLine("[QwenEdit] vision tower vs ComfyUI: " + line);
        // MEASURED (RTX 4070, 2026-11-15): CUDA A 6.13e-5, B 5.04e-5 (merger input 1.05e-4); OpenCL A 2.49e-5, B 1.73e-5; patch embedding 8.0e-7 on both; A alone = A in the pair
        Ur_Expect(worst < 2e-4, string.Join("; ", report));
        return string.Join("; ", report);
    });
    /// <summary>
    /// TextEncodeQwenImageEditPlus's conditioning: ComfyUI's ids (the embedding table sees vision_start and vision_end adjacent), the merged picture rows of
    /// <see cref="QwenVisionEncoder"/> placed between them, the Qwen2.5-VL language model with M-RoPE positions (<see cref="Flux2TextEncoder.MRopePositions"/>), the first 64 rows dropped,
    /// against ComfyUI's conditioning: 1 picture (212 / 204 rows with / without the prompt) and 2 pictures (417 / 409). Control: the same rows laid out as a 1 x N strip (wrong
    /// positions) must lose.
    /// </summary>
    [TestMethod(Timeout = 3600000, Category = "HeavyModel")]
    public async Task QwenEdit_TextEncoderWithPictures_MatchesComfyUi() => await RunTest(async accelerator =>
    {
        var tePath = Qi_Require(accelerator, Path.Combine("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")); var d = Qe_Dumps();
        var src = new LocalFolderModelSource(Path.GetDirectoryName(tePath)!);
        const int VisionStart = 151652, ImagePad = 151655, EndOfText = 151643, Drop = 64, Hd = 3584;
        string V(string run, int pic, string shape) => Path.Combine(d, $"qe_{run}_vis_qwen25_7b_transformer_visual_{pic}_{shape}.raw");
        // the pictures' merged rows from our vision tower on ComfyUI's patch rows (gated by QwenEdit_VisionTower_MatchesComfyUi), then the tower is freed for the 7B language model
        float[] rowsA, rowsB;
        using (var vw = await Flux2Weights.LoadVisionAsync(accelerator, src, "", Path.GetFileName(tePath)))
        using (var enc = new QwenVisionEncoder(accelerator, vw, 768 + 792))
        {
            var both = new float[(768 + 792) * QwenVisionEncoder.PatchIn];
            F2_Raw(V("te1", 0, "in0_f32_768x1176"), 768L * 1176).CopyTo(both, 0);
            F2_Raw(V("te2", 1, "in0_f32_792x1176"), 792L * 1176).CopyTo(both, 768 * 1176);
            using var dIn = accelerator.Allocate1D(both);
            var o = await enc.EncodePatches(dIn.View, new[] { (24, 32), (22, 36) }).CopyToHostAsync();
            rowsA = o.AsSpan(0, 192 * Hd).ToArray(); rowsB = o.AsSpan(192 * Hd, 198 * Hd).ToArray();
        }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var w = await Flux2Weights.LoadAsync(accelerator, src, "", Path.GetFileName(tePath), 28, finalNorm: true);
        using var te = new Flux2TextEncoder(accelerator, w, new[] { 28 }, finalNorm: true) { PreciseActivations = true };
        using var dA = accelerator.Allocate1D(rowsA); using var dB = accelerator.Allocate1D(rowsB);
        var report = new List<string> { $"language model {w.Bytes / 1e9:F2} GB, loaded in {sw.Elapsed.TotalSeconds:F1} s" };
        var grids = new[] { (12, 16), (11, 18) }; var views = new[] { dA.View, dB.View };
        double worst = 0, controlBest = double.MaxValue;
        foreach (var (run, idsFile, condFile, rows) in new[] {
            ("te1", "qe_te1_ids.json", "qe_te1_cond_0_cond_f32_1x212x3584.raw", 212), ("te1 negative", "qe_te1_neg_ids.json", "qe_te1_neg_cond_0_cond_f32_1x204x3584.raw", 204),
            ("te2", "qe_te2_ids.json", "qe_te2_cond_0_cond_f32_1x417x3584.raw", 417), ("te2 negative", "qe_te2_neg_ids.json", "qe_te2_neg_cond_0_cond_f32_1x409x3584.raw", 409) })
        {
            using var doc = System.Text.Json.JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(d, idsFile)));
            var comfy = doc.RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var ids = new List<int>(); var starts = new List<int>(); int pic = 0;
            foreach (var id in comfy)
            {
                ids.Add(id);
                if (id != VisionStart) continue;
                var (gh, gw) = grids[pic++]; starts.Add(ids.Count);
                ids.AddRange(Enumerable.Repeat(ImagePad, gh * gw));                                   // placeholders: their rows are replaced by the picture's
            }
            int real = ids.Count;
            Ur_Expect(real - Drop == rows, $"{run}: {comfy.Length} ids + pictures = {real}, minus {Drop} = {real - Drop}; ComfyUI kept {rows}");
            var padded = ids.Concat(Enumerable.Repeat(EndOfText, Flux2TextEncoder.SeqLen - real)).ToArray();
            var want = F2_Raw(Path.Combine(d, condFile), (long)rows * Hd);
            var pics = starts.Select((s, k) => (s, views[k], grids[k].Item1, grids[k].Item2)).ToList();
            var strip = starts.Select((s, k) => (s, views[k], 1, grids[k].Item1 * grids[k].Item2)).ToList();
            sw.Restart();
            double e = F2_RelRms(await te.Encode(padded, real, pics).SubView((long)Drop * Hd, (long)rows * Hd).CopyToHostAsync(), want);
            double t = sw.Elapsed.TotalSeconds;
            var paddedRows = await te.Encode(padded, real, pics).SubView((long)Drop * Hd, (long)rows * Hd).CopyToHostAsync();
            // the same prompt without padding (per-call length): the prompt's rows must not depend on the padding after it
            sw.Restart();
            var trimmed = await te.Encode(ids.ToArray(), real, pics).SubView((long)Drop * Hd, (long)rows * Hd).CopyToHostAsync();
            double tTrim = sw.Elapsed.TotalSeconds;
            double trimDiff = 0; for (int i = 0; i < trimmed.Length; i++) trimDiff = Math.Max(trimDiff, Math.Abs(trimmed[i] - paddedRows[i]));
            double c = F2_RelRms(await te.Encode(padded, real, strip).SubView((long)Drop * Hd, (long)rows * Hd).CopyToHostAsync(), want);
            worst = Math.Max(worst, e); controlBest = Math.Min(controlBest, c);
            report.Add($"{run} ({starts.Count} picture(s), {real} rows, {t:F1} s; unpadded {tTrim:F1} s, max |unpadded - padded| {trimDiff:E2}): M-RoPE {e:E2}, 1 x N strip control {c:E2}");
            Ur_Expect(trimDiff <= 1e-5, $"{run}: unpadded rows differ from the padded run by {trimDiff:E2}");
        }
        foreach (var line in report) Console.WriteLine("[QwenEdit] text encoder with pictures vs ComfyUI: " + line);
        // MEASURED (RTX 4070, 2026-11-15): CUDA 1 picture 1.51e-4 / 1.55e-4 (negative), 2 pictures 3.08e-4 / 3.12e-4; OpenCL 4.67e-5 / 4.80e-5, 8.02e-5 / 8.14e-5;
        // the 1 x N strip control 0.10 - 0.38: ComfyUI positions the pictures by transformers' get_rope_index rule
        Ur_Expect(worst < 6e-4 && controlBest > 10 * worst, string.Join("; ", report));
        return string.Join("; ", report);
    });

    /// <summary>
    /// <see cref="SkinTokensKernels.RopeTables"/> with <see cref="SkinTokensKernels.AxesRopeTables"/> (Qwen2.5-VL's M-RoPE and its vision tower's 2D rope) against a double-precision
    /// evaluation of the reference's formula (float32 angle pos * inv_freq, accurate cos / sin), on random rows and positions up to 600 (a text prompt with pictures reaches a few
    /// hundred): 3 axes with sections [16, 24, 24] over head width 128, and 2 axes over head width 80 (pairs 0-19 by the row, 20-39 by the column). Bound 1e-6: fp32 rounding only.
    /// (GPU sin / cos evaluated in the kernel were 3.8e-5 off on WebGPU and WebGL at these angles, which is why the tables come from the host.) A text-only prompt's 3-axis tables
    /// must equal plain 1D tables bit for bit.
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_RopeAxes_MatchesHostAndPlainRope() => await RunTest(async accelerator =>
    {
        var u = new SkinTokensKernels(accelerator);
        var rng = new Random(11);
        var notes = new List<string>(); bool ok = true;
        async Task Case(string name, int rows, int H, int D, int A, int b1, int b2, float theta, bool pairedTable)
        {
            int stride = 2 * H * D + 7, off = 3;                                                    // an odd stride and offset, as a fused q | k | v row has
            var buf = new float[rows * stride]; for (int i = 0; i < buf.Length; i++) buf[i] = (float)(rng.NextDouble() * 2 - 1);
            var inv = new float[D / 2];
            if (pairedTable) for (int i = 0; i < D / 4; i++) inv[i] = inv[i + D / 4] = 1f / MathF.Pow(theta, 2f * i / (D / 2));
            else for (int k = 0; k < D / 2; k++) inv[k] = 1f / MathF.Pow(theta, 2f * k / D);
            var pos = new int[rows * A]; for (int i = 0; i < pos.Length; i++) pos[i] = rng.Next(0, 600);
            var (cos, sin) = SkinTokensKernels.AxesRopeTables(pos, rows, A, inv, b1, b2);
            using var dBuf = accelerator.Allocate1D(buf); using var dCos = accelerator.Allocate1D(cos); using var dSin = accelerator.Allocate1D(sin);
            u.RopeTables(dBuf.View, dCos.View, dSin.View, rows, stride, off, H, D);
            var got = await dBuf.View.CopyToHostAsync();
            double worst = 0;
            for (int n = 0; n < rows; n++) for (int h = 0; h < H; h++) for (int k = 0; k < D / 2; k++)
            {
                int axis = k < b1 ? 0 : (k < b2 ? 1 : 2);
                float ang = pos[n * A + axis] * inv[k];                                             // the reference's float32 angle
                double c = Math.Cos(ang), s = Math.Sin(ang);
                long b = (long)n * stride + off + h * D;
                double x0 = buf[b + k], x1 = buf[b + D / 2 + k];
                worst = Math.Max(worst, Math.Max(Math.Abs(got[b + k] - (x0 * c - x1 * s)), Math.Abs(got[b + D / 2 + k] - (x1 * c + x0 * s))));
            }
            for (long i = 0; i < buf.Length; i++)                                                  // nothing outside the rotated columns moves
            {
                long col = i % stride;
                if ((col < off || col >= off + H * D) && got[i] != buf[i]) { worst = double.PositiveInfinity; break; }
            }
            ok &= worst < 1e-6;
            notes.Add($"{name}: worst {worst:E2}");
        }
        await Case("M-RoPE 3 axes, 28 heads of 128, sections 16 / 24 / 24", 37, 28, 128, 3, 16, 40, 1_000_000f, false);
        await Case("vision 2D rope, 16 heads of 80, pairs 0-19 h / 20-39 w", 41, 16, 80, 2, 20, 40, 10000f, true);
        // text-only: the 3-axis tables of a prompt without pictures equal plain 1D tables (position = row) bit for bit
        {
            int rows = 512, D = 128;
            var inv = new float[D / 2]; for (int k = 0; k < D / 2; k++) inv[k] = 1f / MathF.Pow(1_000_000f, 2f * k / D);
            var (c3, s3) = SkinTokensKernels.AxesRopeTables(Flux2TextEncoder.MRopePositions(rows, Array.Empty<(int, int, int)>()), rows, 3, inv, 16, 40);
            var (c1, s1) = SkinTokensKernels.AxesRopeTables(Enumerable.Range(0, rows).ToArray(), rows, 1, inv, D / 2, D / 2);
            int diff = 0; for (int i = 0; i < c3.Length; i++) if (c3[i] != c1[i] || s3[i] != s1[i]) diff++;
            ok &= diff == 0;
            notes.Add($"text-only M-RoPE tables vs 1D: {diff} differ");
        }
        var report = $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
        if (!ok) throw new Exception(report);
        return report;
    });

    /// <summary>
    /// <see cref="Flux1.Flux1Kernels.PatchifyWrap"/> (the device packing of Qwen-Image Edit's reference latents) bit for bit against the host packing that reproduced ComfyUI's
    /// img_in rows exactly (odd sides wrapped): latents of 111 x 148 (odd rows), 99 x 165 (both odd) and 24 x 32 (even, = <see cref="Flux1.Flux1Kernels.Patchify"/>), written at an
    /// offset into one larger token buffer as a call packs them.
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_PatchifyWrap_MatchesComfyUiPacking() => await RunTest(async accelerator =>
    {
        var k = new Flux1.Flux1Kernels(accelerator);
        var rng = new Random(3);
        var notes = new List<string>();
        foreach (var (h, w) in new[] { (111, 148), (99, 165), (24, 32) })
        {
            var lat = new float[16 * h * w]; for (int i = 0; i < lat.Length; i++) lat[i] = (float)(rng.NextDouble() * 8 - 4);
            var (want, rows, cols) = Qe_Tokens(lat, h, w, "circular", null, null);
            const int Offset = 192 * 64;                                                          // after a noise latent's 12 x 16 tokens
            using var dLat = accelerator.Allocate1D(lat); using var dTok = accelerator.Allocate1D<float>(Offset + want.Length + 64);
            dTok.MemSetToZero();
            var (r, c) = k.PatchifyWrap(dLat.View, dTok.View.SubView(Offset, want.Length), h, w);
            var got = await dTok.View.CopyToHostAsync();
            int diff = 0; for (int i = 0; i < want.Length; i++) if (got[Offset + i] != want[i]) diff++;
            bool untouched = got.Take(Offset).All(v => v == 0f) && got.Skip(Offset + want.Length).All(v => v == 0f);
            if (r != rows || c != cols || diff != 0 || !untouched) throw new Exception($"{h} x {w}: grid {r} x {c} (want {rows} x {cols}), {diff} values differ, outside the slice untouched {untouched}");
            notes.Add($"{h} x {w} -> {r} x {c} tokens: bit-identical");
        }
        return $"{accelerator.AcceleratorType}: " + string.Join("; ", notes);
    });

    /// <summary>
    /// The host-side layouts of Qwen-Image Edit: <see cref="Flux2TextEncoder.MRopePositions"/> on a hand-worked prompt (text, a 2 x 3 picture, text, a 3 x 2 picture: transformers'
    /// <c>get_rope_index</c> rule), and <see cref="QwenVisionEncoder.Layout"/> for two pictures (Lore's ComfyUI grids 24 x 32 and 22 x 36): the window order is a permutation of the
    /// merge units, each window lies inside one 4 x 4 unit tile of its own picture, windows and pictures partition the tokens, and the positions are the processor's block-major
    /// patch coordinates.
    /// </summary>
    [TestMethod]
    public async Task QwenEdit_Layouts_MatchTheReferenceRules() => await RunTest(async accelerator =>
    {
        await Task.CompletedTask;
        var notes = new List<string>();
        // M-RoPE: 2 text, picture 2 x 3 at 2, 1 text, picture 3 x 2 at 9, 2 text = 17 tokens
        var pos = Flux2TextEncoder.MRopePositions(17, new[] { (2, 2, 3), (9, 3, 2) });
        var want = new List<int[]> { new[] { 0, 0, 0 }, new[] { 1, 1, 1 } };
        for (int y = 0; y < 2; y++) for (int x = 0; x < 3; x++) want.Add(new[] { 2, 2 + y, 2 + x });
        want.Add(new[] { 5, 5, 5 });                                                                // p = 2 + max(2, 3)
        for (int y = 0; y < 3; y++) for (int x = 0; x < 2; x++) want.Add(new[] { 6, 6 + y, 6 + x });
        want.Add(new[] { 9, 9, 9 }); want.Add(new[] { 10, 10, 10 });                                // p = 6 + max(3, 2)
        for (int t = 0; t < 17; t++) for (int a = 0; a < 3; a++)
            if (pos[t * 3 + a] != want[t][a]) throw new Exception($"M-RoPE token {t} axis {a}: {pos[t * 3 + a]}, want {want[t][a]}");
        notes.Add("M-RoPE positions: 17/17 tokens as the rule gives");

        var grids = new[] { (24, 32), (22, 36) };
        var (vp, wi, windows, pics) = QwenVisionEncoder.Layout(grids);
        int n = 24 * 32 + 22 * 36, m = n / 4;
        if (pics.Length != 3 || pics[1] != 24 * 32 || pics[2] != n) throw new Exception($"picture segments {string.Join(",", pics)}");
        if (wi.Length != m || wi.Distinct().Count() != m || wi.Min() != 0 || wi.Max() != m - 1) throw new Exception("window index is not a permutation of the merge units");
        if (windows[0] != 0 || windows[^1] != n || windows.Zip(windows.Skip(1)).Any(p => p.Second <= p.First || (p.Second - p.First) % 4 != 0)) throw new Exception("window segments");
        // each window: units of one picture inside one 4 x 4 tile
        int units0 = 24 * 32 / 4;
        for (int s = 1; s < windows.Length; s++)
        {
            var units = Enumerable.Range(windows[s - 1] / 4, (windows[s] - windows[s - 1]) / 4).Select(i => wi[i]).ToList();
            bool second = units[0] >= units0;
            int lw = second ? 18 : 16, b = second ? units0 : 0;
            var tiles = units.Select(u => { if ((u >= units0) != second) return (-1, -1); int v = u - b; return (v / lw / 4, v % lw / 4); }).Distinct().ToList();
            if (tiles.Count != 1 || tiles[0].Item1 < 0) throw new Exception($"window {s} spans {tiles.Count} tiles or two pictures");
        }
        // windows: picture 1 12 x 16 units = 3 x 4 tiles of 16; picture 2 11 x 18 = 3 x 5 tiles, the last row and column partial
        if (windows.Length - 1 != 12 + 15) throw new Exception($"{windows.Length - 1} windows, want 27");
        // positions: token 0..3 of picture 1 are patches (0,0) (0,1) (1,0) (1,1); token 4 starts merge block (0, 1) = patch (0, 2)
        int[] p0 = { 0, 0, 0, 1, 1, 0, 1, 1, 0, 2 };
        for (int i = 0; i < p0.Length; i++) if (vp[i] != p0[i]) throw new Exception($"vision positions {string.Join(",", vp.Take(10))}");
        notes.Add($"vision layout: {m} units, {windows.Length - 1} windows inside 4 x 4 tiles, 2 pictures, block-major positions");
        return string.Join("; ", notes);
    });
}
