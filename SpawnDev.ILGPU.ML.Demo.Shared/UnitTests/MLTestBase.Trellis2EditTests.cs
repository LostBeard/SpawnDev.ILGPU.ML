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
}
