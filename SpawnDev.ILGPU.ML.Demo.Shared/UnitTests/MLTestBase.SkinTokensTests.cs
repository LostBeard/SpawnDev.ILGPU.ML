using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// SkinTokens / TokenRig (VAST-AI-Research/SkinTokens, MIT) against goldens written by the REAL reference classes in float32 (tools/unirig-harness/reference/golden_skintokens.py,
/// ST_DIR default E:/spawndev-unirig/skintokens; skipped when the checkpoint export or the goldens are not on this machine). Every stage is gated on its own with the previous
/// stage's golden as input, then the whole pass: mesh prefix, decoder logits (teacher forced), cond latents, skin weights from the golden tokens, and the generated token sequence.
/// </summary>
public abstract partial class MLTestBase
{
    private static string? StDir()
    {
        var dir = Environment.GetEnvironmentVariable("UNIRIG_DIR") ?? "E:/spawndev-unirig";
        return File.Exists(Path.Combine(dir, "weights", "skintokens.safetensors")) && File.Exists(Path.Combine(dir, "skintokens", "golden", "mesh_cond.npy")) ? dir : null;
    }

    private static string St_Require(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType is not (AcceleratorType.Cuda or AcceleratorType.OpenCL)) throw new UnsupportedTestException("SkinTokens goldens run on the desktop GPU lanes (CUDA, OpenCL)");
        return StDir() ?? throw new UnsupportedTestException("SkinTokens export and goldens not found (UNIRIG_DIR, default E:/spawndev-unirig; tools/unirig-harness/reference/export_weights.py skintokens, golden_skintokens.py)");
    }

    private static double St_RelRms(float[] got, float[] gold, int count = -1)
    {
        if (count < 0) count = gold.Length;
        double num = 0, den = 0;
        for (int i = 0; i < count; i++) { double d = got[i] - gold[i]; num += d * d; den += (double)gold[i] * gold[i]; }
        return Math.Sqrt(num / den);
    }

    private static int[] St_Rows(long[] presample, long[] fps) => fps.Select(i => (int)presample[i]).ToArray();

    /// <summary>The mesh encoder + output_proj on the golden's 54000 points against the reference's prefix.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task SkinTokens_MeshPrefix_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (pre, _) = Ur_Npy<long>(Path.Combine(g, "mesh_presample_idx.npy")); var (fps, _) = Ur_Npy<long>(Path.Combine(g, "mesh_fps_idx.npy")); var (gold, _) = Ur_Npy<float>(Path.Combine(g, "mesh_cond.npy"));
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"), name => name.StartsWith("mesh_encoder.") || name.StartsWith("output_proj."));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var enc = new SkinTokensMeshEncoder(accelerator, weights);
        var seen = new Dictionary<string, float[]>();
        enc.Probe = (name, v, rows, cols) => seen[name] = v.SubView(0, (long)rows * cols).GetAsArray1D();
        var o = enc.Encode(p.View, q.View, ps[0], St_Rows(pre, fps));
        var got = await o.View.CopyToHostAsync();
        double rel = St_RelRms(got, gold);
        var stages = new List<string>();
        foreach (var name in new[] { "cross", "self0", "self1", "self2", "self3", "self4", "self5", "self6", "self7" })
            stages.Add($"{name} {St_RelRms(seen[name], Ur_Npy<float>(Path.Combine(g, "probe_mesh." + name + ".npy")).Data):E2}");
        Console.WriteLine("[SkinTokens] mesh encoder stages relRMS: " + string.Join(", ", stages));
        var (goldLat, _) = Ur_Npy<float>(Path.Combine(g, "mesh_latents.npy"));
        double relLat = St_RelRms(await enc.LastLatents!.View.CopyToHostAsync(), goldLat);
        Console.WriteLine($"[SkinTokens] mesh encoder latents relRMS {relLat:E2}, prefix after output_proj {rel:E2} on {accelerator.AcceleratorType}");
        Ur_Expect(relLat < 0.03, $"mesh encoder latents relRMS {relLat:E2} vs the reference (tensor cores: bf16 activations cost ~1e-2; 2e-4 without)");
        Ur_Expect(rel < 0.03, $"mesh prefix relRMS {rel:E2} vs the reference");
    });

    /// <summary>The Qwen3 decoder, teacher-forced on the reference's own prefix and token sequence: logits against the reference's, and the argmax at every position.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task SkinTokens_Decoder_TeacherForcedLogitsMatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (prefix, _) = Ur_Npy<float>(Path.Combine(g, "mesh_cond.npy")); var (ids64, _) = Ur_Npy<long>(Path.Combine(g, "ids.npy")); var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "logits_teacher.npy"));
        var ids = ids64.Select(v => (int)v).ToArray(); int rows = gs[0], V = gs[1];
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"), name => name.StartsWith("transformer."));
        using var pre = accelerator.Allocate1D(prefix); using var dec = new SkinTokensDecoder(accelerator, weights, 512 + ids.Length + 4);
        var got = await dec.PrefillAsync(pre.View, 512, ids, rows);
        double rel = St_RelRms(got, gold);
        // what the generator actually does: the argmax under the grammar (skeleton rows), over the skin codes (skin rows), the end token (last row)
        var allowed = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(Path.Combine(g, "allowed_skeleton.json")));
        int sw = allowed.GetProperty("switch_position").GetInt32(); var sets = allowed.GetProperty("allowed_after_prefix_len").EnumerateArray().Select(a => a.EnumerateArray().Select(x => x.GetInt32()).ToHashSet()).ToList();
        int agree = 0, checkedRows = 0;
        for (int r = 1; r < rows - 1; r++)
        {
            int best = -1; float bv = float.NegativeInfinity;
            for (int v = 0; v < V - 1; v++)
            {
                bool ok = r < sw ? sets[r].Contains(v) : v >= UniRigTokenizer.VocabSize;
                if (ok && got[r * V + v] > bv) { bv = got[r * V + v]; best = v; }
            }
            checkedRows++; if (best == ids[r + 1]) agree++;
        }
        var perRow = new List<(double Rel, int Row)>();
        for (int r = 0; r < rows; r++) { double nn = 0, dd = 0; for (int v = 0; v < V; v++) { double d = got[r * V + v] - gold[r * V + v]; nn += d * d; dd += (double)gold[r * V + v] * gold[r * V + v]; } perRow.Add((Math.Sqrt(nn / dd), r)); }
        Console.WriteLine($"[SkinTokens] decoder per-row relRMS: rows 0-4 {string.Join(", ", perRow.Take(5).Select(x => x.Rel.ToString("E1")))}; worst {string.Join(", ", perRow.OrderByDescending(x => x.Rel).Take(6).Select(x => $"row {x.Row}={x.Rel:E1}"))}; median {perRow.OrderBy(x => x.Rel).ElementAt(rows / 2).Rel:E1}");
        Ur_Expect(rel < 0.05, $"decoder logits relRMS {rel:E2} vs the reference");
        Ur_Expect(agree >= checkedRows - 4, $"the constrained argmax reproduced {agree}/{checkedRows} generated tokens (relRMS {rel:E2}; near-ties may flip one or two)");
        Console.WriteLine($"[SkinTokens] decoder relRMS {rel:E2}, {agree}/{checkedRows} tokens on {accelerator.AcceleratorType}");
    });

    /// <summary>The VAE's cond encoder on the golden's points and sampled queries against the reference's [384, 512] latents.</summary>
    [TestMethod(Timeout = 300000, Category = "HeavyModel")]
    public async Task SkinTokens_CondLatents_MatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (pre, _) = Ur_Npy<long>(Path.Combine(g, "cond_presample_idx.npy")); var (fps, _) = Ur_Npy<long>(Path.Combine(g, "cond_fps_idx.npy")); var (gold, _) = Ur_Npy<float>(Path.Combine(g, "cond_latents.npy"));
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"), name => name.StartsWith("vae."));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var vae = new SkinTokensVae(accelerator, weights);
        var o = vae.EncodeCond(p.View, q.View, ps[0], St_Rows(pre, fps));
        var got = await o.View.CopyToHostAsync();
        double rel = St_RelRms(got, gold);
        Ur_Expect(rel < 0.02, $"cond latents relRMS {rel:E2} vs the reference");
        Console.WriteLine($"[SkinTokens] cond latents relRMS {rel:E2} on {accelerator.AcceleratorType}");
    });

    /// <summary>The VAE decoder on the reference's own cond latents and skin tokens: every bone's weights at every one of the 54000 points.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task SkinTokens_SkinWeights_MatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (cond, _) = Ur_Npy<float>(Path.Combine(g, "cond_latents.npy")); var (ids64, _) = Ur_Npy<long>(Path.Combine(g, "ids.npy")); var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "skin_pred.npy"));
        var ids = ids64.Select(v => (int)v).ToArray(); int n = gs[0], J = gs[1];
        int first = Array.IndexOf(ids, UniRigTokenizer.Eos) + 1;
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"), name => name.StartsWith("vae."));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var c = accelerator.Allocate1D(cond); using var vae = new SkinTokensVae(accelerator, weights);
        using var skin = accelerator.Allocate1D<float>((long)n * J);
        vae.SetCond(c.View);
        var (gz, _) = Ur_Npy<float>(Path.Combine(g, "skin_z.npy")); var (gkv, _) = Ur_Npy<float>(Path.Combine(g, "skin_kv0.npy"));
        var (z0, kv0) = vae.BoneState(Enumerable.Range(0, 4).Select(t => ids[first + t] - UniRigTokenizer.VocabSize).ToArray());
        double relZ = St_RelRms(z0, gz), relKv = St_RelRms(kv0, gkv);
        Console.WriteLine($"[SkinTokens] bone 0: FSQ latents relRMS {relZ:E2}, decoder state after 10 blocks relRMS {relKv:E2}");
        Ur_Expect(relZ < 1e-2, $"FSQ latents relRMS {relZ:E2}"); Ur_Expect(relKv < 2e-2, $"decoder state relRMS {relKv:E2}");
        vae.BeginSkin(p.View, q.View, n);
        for (int j = 0; j < J; j++)
            vae.DecodeBone(Enumerable.Range(0, 4).Select(t => ids[first + j * 4 + t] - UniRigTokenizer.VocabSize).ToArray(), skin.View, j, J);
        accelerator.Synchronize();
        var got = await skin.View.CopyToHostAsync();
        double rel = St_RelRms(got, gold);
        double worst = 0; for (int i = 0; i < gold.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - gold[i]));
        Ur_Expect(rel < 0.02, $"skin weights relRMS {rel:E2} vs the reference (worst absolute {worst:F4})");
        Console.WriteLine($"[SkinTokens] skin weights {n}x{J}: relRMS {rel:E2}, worst abs {worst:F4} on {accelerator.AcceleratorType}");
    });

    /// <summary>
    /// Everything on the golden's points, greedy, with exact (fp32, no tensor cores) kernels: the SAME token sequence as the reference's and the same skin weights. One
    /// intended difference: the shipped logits processor counts the switch token among the skin tokens, so it ends the sequence one token early and its last bone is decoded
    /// with the END id (33035 - 267 = 32768, which the FSQ digits wrap to code 0) as its fourth token. The port writes all four tokens of every bone; that bone's weights
    /// are therefore not compared.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task SkinTokens_EndToEnd_MatchesTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (mpre, _) = Ur_Npy<long>(Path.Combine(g, "mesh_presample_idx.npy")); var (mfps, _) = Ur_Npy<long>(Path.Combine(g, "mesh_fps_idx.npy"));
        var (cpre, _) = Ur_Npy<long>(Path.Combine(g, "cond_presample_idx.npy")); var (cfps, _) = Ur_Npy<long>(Path.Combine(g, "cond_fps_idx.npy"));
        var (ids64, _) = Ur_Npy<long>(Path.Combine(g, "ids.npy")); var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "skin_pred.npy"));
        var refIds = ids64.Select(v => (int)v).ToArray(); int n = gs[0], J = gs[1];
        bool tc = DiTKernels.EnableTensorCores, tiled = DiTKernels.EnableTiled;
        DiTKernels.EnableTensorCores = false; DiTKernels.EnableTiled = false;
        try
        {
            using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"));
            using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var gen = new SkinTokensGenerator(accelerator, weights);
            var r = await gen.GenerateAsync(p.View, q.View, ps[0], St_Rows(mpre, mfps), St_Rows(cpre, cfps));
            int same = 0; for (int i = 0; i < Math.Min(r.Ids.Length, refIds.Length); i++) { if (r.Ids[i] != refIds[i]) break; same++; }
            Ur_Expect(r.Skeleton.Count == J, $"{r.Skeleton.Count} bones, the reference found {J} (the sequences agree on their first {same} of {refIds.Length} tokens)");
            Ur_Expect(same == refIds.Length - 1 && r.Ids.Length == refIds.Length + 1 && r.Ids[^1] == SkinTokensGenerator.EndToken,
                $"expected the reference's {refIds.Length - 1} tokens then the missing fourth skin token of the last bone and the end token; identical for {same}, lengths {r.Ids.Length} / {refIds.Length}");
            // the weights of every bone but the last (whose reference decode used the wrapped END id)
            double num = 0, den = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < J - 1; j++) { double d = r.Skin[i * J + j] - gold[i * J + j]; num += d * d; den += (double)gold[i * J + j] * gold[i * J + j]; }
            double rel = Math.Sqrt(num / den);
            Ur_Expect(rel < 5e-3, $"skin weights of bones 0..{J - 2} relRMS {rel:E2} vs the reference");
            Console.WriteLine($"[SkinTokens] end to end (exact kernels): {r.Skeleton.Count} bones, {same}/{refIds.Length} identical tokens, skin relRMS {rel:E2} on {accelerator.AcceleratorType}");
        }
        finally { DiTKernels.EnableTensorCores = tc; DiTKernels.EnableTiled = tiled; }
    });

    /// <summary>The same pass with the fast kernels (CUDA tensor cores, bf16 inputs): the sequence may differ from the reference's at near-ties, so only what must hold is checked.</summary>
    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task SkinTokens_FastKernels_ProduceAValidRig() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (mpre, _) = Ur_Npy<long>(Path.Combine(g, "mesh_presample_idx.npy")); var (mfps, _) = Ur_Npy<long>(Path.Combine(g, "mesh_fps_idx.npy"));
        var (cpre, _) = Ur_Npy<long>(Path.Combine(g, "cond_presample_idx.npy")); var (cfps, _) = Ur_Npy<long>(Path.Combine(g, "cond_fps_idx.npy"));
        var (gold, gs) = Ur_Npy<float>(Path.Combine(g, "skin_pred.npy")); int n = gs[0], refJ = gs[1];
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var gen = new SkinTokensGenerator(accelerator, weights);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var r = await gen.GenerateAsync(p.View, q.View, ps[0], St_Rows(mpre, mfps), St_Rows(cpre, cfps));
        int J = r.Skeleton.Count;
        Ur_Expect(J >= 10 && J <= 3 * refJ, $"{J} bones (the reference found {refJ})");
        Ur_Expect(r.Skeleton.Parents[0] == -1 && Enumerable.Range(1, J - 1).All(j => r.Skeleton.Parents[j] >= 0 && r.Skeleton.Parents[j] < j), "parents are not a tree in creation order");
        Ur_Expect(r.Skin.Length == n * J && r.Skin.All(v => float.IsFinite(v) && v >= 0f && v <= 1f), "skin weights are not finite values in [0,1]");
        double mass = 0; for (int i = 0; i < n; i++) { double rs = 0; for (int j = 0; j < J; j++) rs += r.Skin[i * J + j]; mass += rs; }
        Ur_Expect(mass / n > 0.2 && mass / n < 5, $"mean row mass {mass / n:F2}");
        Console.WriteLine($"[SkinTokens] fast kernels: {J} bones (reference {refJ}), {sw.Elapsed.TotalSeconds:F1} s on {accelerator.AcceleratorType}");
    });

    /// <summary>The VAE stage by stage on the golden's points: where the port first leaves the reference (the cond encoder, then bone 0 of the skin decoder).</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task SkinTokens_VaeStages_MatchTheReference() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator); var g = Path.Combine(dir, "skintokens", "golden");
        var (pts, ps) = Ur_Npy<float>(Path.Combine(g, "points.npy")); var (nrm, _) = Ur_Npy<float>(Path.Combine(g, "normals.npy"));
        var (pre, _) = Ur_Npy<long>(Path.Combine(g, "cond_presample_idx.npy")); var (fps, _) = Ur_Npy<long>(Path.Combine(g, "cond_fps_idx.npy"));
        var (cond, _) = Ur_Npy<float>(Path.Combine(g, "cond_latents.npy")); var (ids64, _) = Ur_Npy<long>(Path.Combine(g, "ids.npy"));
        var ids = ids64.Select(v => (int)v).ToArray(); int first = Array.IndexOf(ids, UniRigTokenizer.Eos) + 1;
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"), name => name.StartsWith("vae."));
        using var p = accelerator.Allocate1D(pts); using var q = accelerator.Allocate1D(nrm); using var vae = new SkinTokensVae(accelerator, weights);
        var got = new Dictionary<string, float[]>();
        vae.Probe = (name, view, rows, cols) => { if (!got.ContainsKey(name)) got[name] = view.SubView(0, (long)rows * cols).GetAsArray1D(); };
        vae.EncodeCond(p.View, q.View, ps[0], St_Rows(pre, fps));
        using var c = accelerator.Allocate1D(cond); using var skin = accelerator.Allocate1D<float>((long)ps[0] * 40);
        vae.SetCond(c.View); vae.BeginSkin(p.View, q.View, ps[0]);
        vae.DecodeBone(Enumerable.Range(0, 4).Select(t => ids[first + t] - UniRigTokenizer.VocabSize).ToArray(), skin.View, 0, 40);
        var lines = new List<string>(); string firstBad = "";
        foreach (var (mine, theirs) in new[] { ("cond.proj_in.q", "probe_cond.proj_in"), ("cond.block0", "probe_cond.block0"), ("cond.block1", "probe_cond.block1"), ("cond.block2", "probe_cond.block2"),
                                               ("cond.norm_out", "probe_cond.norm_out"), ("dec.proj_query", "probe_dec.proj_query"), ("dec.block10", "probe_dec.block10"), ("dec.norm_out", "probe_dec.norm_out") })
        {
            var (gold, gs) = Ur_Npy<float>(Path.Combine(g, theirs + ".npy"));
            var mineArr = got[mine]; int n = Math.Min(mineArr.Length, gold.Length);
            if (mine == "cond.proj_in.q") n = 384 * 768;                       // the reference's proj_in probe holds the 384 queries then keys
            double rel = St_RelRms(mineArr, gold, n);
            lines.Add($"{mine} relRMS {rel:E2}"); if (rel > 2e-2 && firstBad == "") firstBad = mine;
        }
        if (got.TryGetValue("cond.proj_in.kv", out var kvMine)) { var (gold, _) = Ur_Npy<float>(Path.Combine(g, "probe_cond.proj_in.npy")); lines.Add($"cond.proj_in.kv (first {Math.Min(kvMine.Length, gold.Length - 384 * 768) / 768} keys) relRMS {St_RelRms(kvMine, gold.AsSpan(384 * 768).ToArray(), Math.Min(kvMine.Length, gold.Length - 384 * 768)):E2}"); }
        Console.WriteLine("[SkinTokens] VAE stages: " + string.Join("; ", lines));
        Ur_Expect(firstBad == "", $"first stage beyond 2e-2: {firstBad}. All: {string.Join("; ", lines)}");
    });

    /// <summary>SkinTokens end to end on a TRELLIS-generated humanoid GLB (GLB in, rigged GLB out): a valid skinned glTF whose right arm follows a posed shoulder and whose other side stays still.</summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task SkinTokens_Rigger_RigsAGeneratedHumanoid() => await RunTest(async accelerator =>
    {
        var dir = St_Require(accelerator);
        var humanoid = Path.Combine(dir, "inputs", "humanoid.glb");
        if (!File.Exists(humanoid)) throw new UnsupportedTestException("inputs/humanoid.glb (a TRELLIS.2 generated T-pose humanoid) not found");
        var glb = File.ReadAllBytes(humanoid);
        using var weights = GpuWeights.LoadFile(accelerator, Path.Combine(dir, "weights", "skintokens.safetensors"));
        using var rigger = new SkinTokensRigger(accelerator, weights);
        var res = await rigger.RigGlbAsync(glb, seed: 0);
        var keep = Path.Combine(dir, "tmp"); if (Directory.Exists(keep)) File.WriteAllBytes(Path.Combine(keep, "skintokens_rigged_humanoid.glb"), res.Glb);   // for inspection (validate_glb.mjs, pose_check.py)
        var sk = res.Skeleton;
        Ur_Expect(sk.Count is >= 15 and <= 60, $"{sk.Count} joints for a humanoid");
        Ur_Expect(sk.Parents.Count(p => p < 0) == 1 && sk.Parents.Select((p, i) => p < i).All(b => b), "one root, parents before children");

        var mesh = RiggedGlb.ReadMesh(res.Glb); var orig = RiggedGlb.ReadMesh(glb);
        Ur_Expect(mesh.Positions.SequenceEqual(orig.Positions), "the rigged file keeps the mesh");
        int V = mesh.Positions.Length / 3, J = sk.Count;
        float lo = mesh.Positions.Where((_, i) => i % 3 == 1).Min(), hi = mesh.Positions.Where((_, i) => i % 3 == 1).Max();
        for (int j = 0; j < J; j++)
            Ur_Expect(sk.Joints[j * 3 + 1] > lo - 0.05f * (hi - lo) && sk.Joints[j * 3 + 1] < hi + 0.05f * (hi - lo), $"joint {j} lies within the model's height");

        // read the skin back out of the file and pose it: rotate the outermost shoulder on +x
        int jsonLen = (int)BitConverter.ToUInt32(res.Glb, 12), bin = 20 + jsonLen + 8;
        using var doc = JsonDocument.Parse(res.Glb.AsMemory(20, jsonLen)); var root = doc.RootElement;
        var prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        int Off(int a) { var acc = root.GetProperty("accessors")[a]; return bin + root.GetProperty("bufferViews")[acc.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32(); }
        int jo = Off(prim.GetProperty("JOINTS_0").GetInt32()), wo = Off(prim.GetProperty("WEIGHTS_0").GetInt32());
        for (int v = 0; v < V; v++)
        {
            float s = 0; for (int k = 0; k < 4; k++) s += BitConverter.ToSingle(res.Glb, wo + v * 16 + k * 4);
            Ur_Expect(Math.Abs(s - 1) < 1e-4, $"vertex {v} weights sum to {s}");
        }
        float span = sk.Joints.Where((_, i) => i % 3 == 0).Max();
        int shoulder = -1;
        for (int j = 0; j < J && shoulder < 0; j++) if (sk.Joints[j * 3] > 0.2f * span && sk.Parents[j] >= 0 && sk.Joints[sk.Parents[j] * 3] <= 0.2f * span) shoulder = j;
        Ur_Expect(shoulder >= 0, "a joint out on the right arm");
        var follows = new bool[J]; follows[shoulder] = true; for (int j = 0; j < J; j++) if (sk.Parents[j] >= 0 && follows[sk.Parents[j]]) follows[j] = true;
        float height = hi - lo; int armVerts = 0, leftMoved = 0, bodyMoved = 0;
        for (int v = 0; v < V; v++)
        {
            float follow = 0; for (int k = 0; k < 4; k++) if (follows[BitConverter.ToUInt16(res.Glb, jo + v * 8 + k * 2)]) follow += BitConverter.ToSingle(res.Glb, wo + v * 16 + k * 4);
            float x = mesh.Positions[v * 3];
            if (follow > 0.5f) armVerts++;
            if (x < -0.1f * height && follow > 0.01f) leftMoved++;
            if (Math.Abs(x) < 0.08f * height && follow > 0.25f) bodyMoved++;
        }
        Ur_Expect(armVerts > V * 0.05, $"only {armVerts} of {V} vertices follow the right arm");
        Ur_Expect(leftMoved == 0, $"{leftMoved} vertices on the left half would move with the right shoulder");
        Ur_Expect(bodyMoved < V * 0.01, $"{bodyMoved} vertices of the torso, head and legs would move noticeably with the right shoulder");
        Console.WriteLine($"[SkinTokens] rigged humanoid: {J} joints, {armVerts} of {V} vertices follow the right arm, left half {leftMoved}, body {bodyMoved} on {accelerator.AcceleratorType}");
    });
}
