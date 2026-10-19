using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Assets;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Region editing of a generated TRELLIS.2 asset (<see cref="Trellis2Pipeline.RetextureRegionAsync"/>): the tokens outside the region are pinned to the asset's own texture latent, the ones inside
/// are regenerated under a new picture. The real model on CUDA (skipped without the checkpoints, like <c>Trellis2_ImageToGlb_EndToEnd</c>).
/// </summary>
public abstract partial class MLTestBase
{
    private static void Te_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    [TestMethod]
    public async Task AssetGenerator_RegionFromSpec_ParsesAndCombinesRegions() => await RunPureTest(async () =>
    {
        var top = AssetGenerator.RegionFromSpec("z>0.1");
        Te_Expect(top(0, 0, 0.2f) && !top(0, 0, 0.05f), "an axis bound");
        var box = AssetGenerator.RegionFromSpec("box:-0.1,-0.1,-0.1,0.1,0.1,0.1");
        Te_Expect(box(0, 0, 0) && !box(0.2f, 0, 0), "a box");
        var ball = AssetGenerator.RegionFromSpec("sphere:0.3,0,0,0.1");
        Te_Expect(ball(0.35f, 0, 0) && !ball(0, 0, 0), "a sphere");
        var both = AssetGenerator.RegionFromSpec("z>0.4; x<-0.4");
        Te_Expect(both(0, 0, 0.5f) && both(-0.5f, 0, 0) && !both(0, 0, 0), "a union");
        foreach (var bad in new[] { "", "w>1", "box:1,2", "sphere:1" })
        {
            try { AssetGenerator.RegionFromSpec(bad); throw new Exception($"'{bad}' must be refused"); } catch (FormatException) { }
        }
        await Task.CompletedTask;
    });

    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Trellis2_RegionRetexture_KeepsTheRestAndChangesTheRegion() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("Trellis2 runs on CUDA");
        var dir = Environment.GetEnvironmentVariable("TRELLIS2_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "spawndev-trellis2");
        var paths = new Trellis2Paths
        {
            ModelDir = Path.Combine(dir, "TRELLIS.2-4B"),
            SparseStructureDecoder = Path.Combine(dir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"),
            DinoWeights = Path.Combine(dir, "dinov3-timm-qkvb", "model.safetensors"),
        };
        if (!File.Exists(paths.DinoWeights) || !File.Exists(paths.SparseStructureDecoder + ".safetensors") || !Directory.Exists(Path.Combine(paths.ModelDir, "ckpts")))
            throw new UnsupportedTestException("TRELLIS.2 checkpoints not found (set TRELLIS2_DIR)");

        const int W = 384, H = 320;
        int[] Picture(int baseR, int baseG, int baseB)
        {
            var px = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    double dx = (x - W / 2.0) / (W * 0.30), dy = (y - H / 2.0) / (H * 0.36), d = dx * dx + dy * dy;
                    int a = d < 1 ? 255 : 0;
                    int r = baseR + (int)(30 * Math.Sin(x * 0.05)), g = baseG + (int)(30 * Math.Cos(y * 0.07)), b = baseB - y / 8;
                    px[y * W + x] = Math.Clamp(r, 0, 255) | (Math.Clamp(g, 0, 255) << 8) | (Math.Clamp(b, 0, 255) << 16) | (a << 24);
                }
            return px;
        }
        using var gen = new Trellis2ImageTo3D(accelerator, paths);
        using var first = RgbaUpload.FromManaged(accelerator, Picture(200, 60, 50), W, H);
        var (source, pipe) = await gen.GenerateAsync(first.View, W, H, new Trellis2Options { TextureSize = 1024 });
        using var _s = source;
        Te_Expect(source.Edit != null && source.Edit.Count > 100, "a generation keeps its latents");
        using var second = RgbaUpload.FromManaged(accelerator, Picture(40, 170, 220), W, H);
        var (edited, _) = await gen.EditRegionAsync(source, second.View, W, H, (x, y, z) => x > 0f, seed: 3);
        using var _e = edited;

        Te_Expect(edited.VoxelCount == source.VoxelCount, $"the same voxels ({source.VoxelCount} -> {edited.VoxelCount}): the shape is unchanged");
        var ca = await source.Coords.View.SubView(0, (long)source.VoxelCount * 3).CopyToHostAsync();
        var aa = await source.Attributes.View.SubView(0, (long)source.VoxelCount * 6).CopyToHostAsync();
        var cb = await edited.Coords.View.SubView(0, (long)edited.VoxelCount * 3).CopyToHostAsync();
        var ab = await edited.Attributes.View.SubView(0, (long)edited.VoxelCount * 6).CopyToHostAsync();
        Te_Expect(ca.AsSpan().SequenceEqual(cb), "the voxels are at the same coordinates");
        double farOut = 0, inside = 0; int nOut = 0, nIn = 0;
        for (int i = 0; i < source.VoxelCount; i++)
        {
            float x = (ca[i * 3] + 0.5f) / source.Grid - 0.5f;
            double d = 0; for (int c = 0; c < 3; c++) d += Math.Abs(aa[i * 6 + c] - ab[i * 6 + c]) / 3;
            if (x < -0.12f) { farOut = Math.Max(farOut, d); nOut++; }              // well outside the region (the decoder looks a few tokens around a voxel)
            else if (x > 0.12f) { inside += d; nIn++; }
        }
        Te_Expect(nOut > 100 && nIn > 100, $"voxels on both sides ({nOut} outside, {nIn} inside)");
        inside /= nIn;
        Te_Expect(farOut < 0.02, $"the part far outside the region keeps its colour (worst mean-channel change {farOut:F4})");
        Te_Expect(inside > 0.08, $"the region takes the new picture's colours (mean change {inside:F3})");
        Console.WriteLine($"[RegionEdit] {source.Edit!.Count} tokens, {nOut} voxels kept (worst change {farOut:F4}), {nIn} voxels in the region (mean change {inside:F3})");
        await Task.CompletedTask;
        return $"{source.Edit!.Count} tokens, {nOut} voxels kept (worst change {farOut:F4}), {nIn} voxels in the region (mean change {inside:F3})";
    });

    [TestMethod(Timeout = 900000, Category = "HeavyModel")]
    public async Task Trellis2_ShapeEdit_ChangesTheRegionsGeometryAndKeepsTheRest() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("Trellis2 runs on CUDA");
        var dir = Environment.GetEnvironmentVariable("TRELLIS2_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "spawndev-trellis2");
        var paths = new Trellis2Paths
        {
            ModelDir = Path.Combine(dir, "TRELLIS.2-4B"),
            SparseStructureDecoder = Path.Combine(dir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"),
            DinoWeights = Path.Combine(dir, "dinov3-timm-qkvb", "model.safetensors"),
        };
        if (!File.Exists(paths.DinoWeights) || !File.Exists(paths.SparseStructureDecoder + ".safetensors") || !Directory.Exists(Path.Combine(paths.ModelDir, "ckpts")))
            throw new UnsupportedTestException("TRELLIS.2 checkpoints not found (set TRELLIS2_DIR)");

        const int W = 384, H = 320;
        int[] Blob(double rx, double ry, int r, int g, int b)
        {
            var px = new int[W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    double dx = (x - W / 2.0) / (W * rx), dy = (y - H / 2.0) / (H * ry), d = dx * dx + dy * dy;
                    int a = d < 1 ? 255 : 0;
                    px[y * W + x] = Math.Clamp(r + (int)(30 * Math.Sin(x * 0.05)), 0, 255) | (Math.Clamp(g + (int)(30 * Math.Cos(y * 0.07)), 0, 255) << 8) | (Math.Clamp(b - y / 8, 0, 255) << 16) | (a << 24);
                }
            return px;
        }
        using var gen = new Trellis2ImageTo3D(accelerator, paths);
        using var first = RgbaUpload.FromManaged(accelerator, Blob(0.30, 0.36, 200, 60, 50), W, H);
        var (source, _) = await gen.GenerateAsync(first.View, W, H, new Trellis2Options { TextureSize = 1024 });
        using var _s = source;
        Te_Expect(source.Edit != null && source.Edit.SsLatent.Length > 0 && source.Edit.SsRes == 16, "a generation keeps its structure latent");
        using var second = RgbaUpload.FromManaged(accelerator, Blob(0.14, 0.44, 60, 170, 220), W, H);      // a much thinner, taller subject
        var (edited, _) = await gen.EditShapeRegionAsync(source, second.View, W, H, (x, y, z) => x > 0f, seed: 3);
        using var _e = edited;

        var ca = await source.Coords.View.SubView(0, (long)source.VoxelCount * 3).CopyToHostAsync();
        var aa = await source.Attributes.View.SubView(0, (long)source.VoxelCount * 6).CopyToHostAsync();
        var cb = await edited.Coords.View.SubView(0, (long)edited.VoxelCount * 3).CopyToHostAsync();
        var ab = await edited.Attributes.View.SubView(0, (long)edited.VoxelCount * 6).CopyToHostAsync();
        long Key(int[] c, int i) => ((long)c[i * 3] * source.Grid + c[i * 3 + 1]) * source.Grid + c[i * 3 + 2];
        var before = new Dictionary<long, int>(); for (int i = 0; i < source.VoxelCount; i++) before[Key(ca, i)] = i;
        var after = new Dictionary<long, int>(); for (int i = 0; i < edited.VoxelCount; i++) after[Key(cb, i)] = i;
        int farOut = 0, farOutGone = 0, farOutNew = 0, inBefore = 0, inChanged = 0; double farOutDiff = 0, farOutSum = 0; int farOutBoth = 0;
        foreach (var (k, i) in before)
        {
            float x = (ca[i * 3] + 0.5f) / source.Grid - 0.5f;
            bool has = after.TryGetValue(k, out int j);
            if (x < -0.12f) { farOut++; if (!has) farOutGone++; else { double d = 0; for (int c = 0; c < 3; c++) d += Math.Abs(aa[i * 6 + c] - ab[j * 6 + c]) / 3; farOutDiff = Math.Max(farOutDiff, d); farOutSum += d; farOutBoth++; } }
            else if (x > 0.12f) { inBefore++; if (!has) inChanged++; }
        }
        foreach (var (k, j) in after) { float x = (cb[j * 3] + 0.5f) / edited.Grid - 0.5f; if (!before.ContainsKey(k)) { if (x < -0.12f) farOutNew++; else if (x > 0.12f) inChanged++; } }
        Te_Expect(farOut > 100 && inBefore > 100, $"voxels on both sides ({farOut} outside, {inBefore} inside)");
        Console.WriteLine($"[RegionEdit] far outside: {farOut} voxels, {farOutGone} gone, {farOutNew} new, colour change mean {farOutSum / Math.Max(1, farOutBoth):F5}, worst {farOutDiff:F4}");
        Te_Expect(farOutGone + farOutNew < farOut / 100, $"the geometry far outside the region is untouched but for the decoder's reach at the edges ({farOutGone} voxels gone, {farOutNew} new of {farOut})");
        Te_Expect(farOutSum / Math.Max(1, farOutBoth) < 0.003 && farOutDiff < 0.1, $"and so is its colour (mean change {farOutSum / Math.Max(1, farOutBoth):F5}, worst {farOutDiff:F4}: the decoder reads a few tokens around each voxel)");
        Te_Expect(inChanged > inBefore / 20, $"the geometry inside the region changed ({inChanged} voxels added or removed of {inBefore})");
        Console.WriteLine($"[RegionEdit] shape: {source.Edit!.Count} -> {edited.Edit!.Count} tokens; outside {farOut} voxels kept (mean colour change {farOutSum / Math.Max(1, farOutBoth):F5}); inside {inChanged} of {inBefore} voxels changed");
        await Task.CompletedTask;
        return $"{source.Edit.Count} -> {edited.Edit.Count} tokens, {inChanged} of {inBefore} region voxels changed";
    });
}
