using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="DeviceRemesher"/> (the whole narrow-band remesh in device kernels) against <see cref="NarrowBandRemesher"/> (the host version it replaces) on the same synthetic meshes: the same band voxels and corners, the same vertices
/// (as a sorted multiset), the same face count, the same topology (open edges, non-manifold edges, winding conflicts, pieces) and the same signed volume.
/// </summary>
public abstract partial class MLTestBase
{
    private static void Dr_Expect(bool ok, string what) { if (!ok) throw new Exception(what); }

    private async Task Dr_Compare(global::ILGPU.Runtime.Accelerator accelerator, float[] p, int[] f, int resolution, bool dropInnerSkins, string what)
    {
        using var host = await NarrowBandRemesher.RemeshAsync(accelerator, p, f, resolution, 1f, dropInnerSkins);
        using var pb = accelerator.Allocate1D(p); using var fb = accelerator.Allocate1D(f);
        using var sk = new DeviceRemeshHolder(accelerator);
        using var dev = await sk.Remesher.RemeshAsync(pb.View, p.Length / 3, fb.View, f.Length / 3, resolution, 1f, dropInnerSkins);
        Dr_Expect(dev.ActiveVoxels == host.ActiveVoxels, $"{what}: band voxels {dev.ActiveVoxels} vs host {host.ActiveVoxels}");
        Dr_Expect(dev.GridVertices == host.GridVertices, $"{what}: corners {dev.GridVertices} vs host {host.GridVertices}");
        Dr_Expect(dev.VertexCount == host.Positions.Length / 3, $"{what}: vertices {dev.VertexCount} vs host {host.Positions.Length / 3}");
        Dr_Expect(dev.FaceCount == host.Faces.Length / 3, $"{what}: faces {dev.FaceCount} vs host {host.Faces.Length / 3}");
        var dp = dev.VertexCount == 0 ? Array.Empty<float>() : await dev.Positions.View.SubView(0, (long)dev.VertexCount * 3).CopyToHostAsync();
        var df = dev.FaceCount == 0 ? Array.Empty<int>() : await dev.Faces.View.SubView(0, (long)dev.FaceCount * 3).CopyToHostAsync();

        // the vertex sets are the same points (the order differs): every device vertex must have a host vertex within 1e-4 (found through a grid of 1e-3 cells, so float noise cannot break a pairing the way a sorted comparison does)
        const float Cell = 1e-3f;
        var grid = new Dictionary<(int, int, int), List<int>>();
        for (int i = 0; i < host.Positions.Length / 3; i++)
        {
            var key = ((int)MathF.Floor(host.Positions[i * 3] / Cell), (int)MathF.Floor(host.Positions[i * 3 + 1] / Cell), (int)MathF.Floor(host.Positions[i * 3 + 2] / Cell));
            if (!grid.TryGetValue(key, out var l)) grid[key] = l = new List<int>();
            l.Add(i);
        }
        double worst = 0; int unmatched = 0;
        for (int i = 0; i < dev.VertexCount; i++)
        {
            int cx = (int)MathF.Floor(dp[i * 3] / Cell), cy = (int)MathF.Floor(dp[i * 3 + 1] / Cell), cz = (int)MathF.Floor(dp[i * 3 + 2] / Cell);
            double best = double.MaxValue;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++)
                if (grid.TryGetValue((cx + dx, cy + dy, cz + dz), out var l))
                    foreach (int j in l)
                        best = Math.Min(best, Math.Max(Math.Abs(dp[i * 3] - host.Positions[j * 3]), Math.Max(Math.Abs(dp[i * 3 + 1] - host.Positions[j * 3 + 1]), Math.Abs(dp[i * 3 + 2] - host.Positions[j * 3 + 2]))));
            if (best > 1e-4) unmatched++; else worst = Math.Max(worst, best);
        }
        Dr_Expect(unmatched == 0, $"{what}: {unmatched} of {dev.VertexCount} device vertices have no host vertex within 1e-4");

        var th = MeshRepair.Measure(host.Positions, host.Faces); var td = MeshRepair.Measure(dp, df);
        Dr_Expect(td.BoundaryEdges == th.BoundaryEdges && td.NonManifoldEdges == th.NonManifoldEdges && td.OrientationConflicts == th.OrientationConflicts && td.Components == th.Components,
            $"{what}: topology {td.BoundaryEdges}/{td.NonManifoldEdges}/{td.OrientationConflicts}/{td.Components} (open/non-manifold/conflicts/pieces) vs host {th.BoundaryEdges}/{th.NonManifoldEdges}/{th.OrientationConflicts}/{th.Components}");
        double vh = Mr_Volume(host.Positions, host.Faces), vd = Mr_Volume(dp, df);
        Dr_Expect(Math.Abs(vh - vd) <= 1e-3 * Math.Abs(vh) + 1e-7, $"{what}: signed volume {vd:G6} vs host {vh:G6}");
        Console.WriteLine($"[DeviceRemesh] {what}: {dev.ActiveVoxels} band voxels, {dev.GridVertices} corners, {dev.VertexCount} vertices, {dev.FaceCount} faces: same as the host (vertices within {worst:G2}, volume {vd:G5})");
    }

    private sealed class DeviceRemeshHolder : IDisposable
    {
        public readonly DeviceRemesher Remesher;
        public DeviceRemeshHolder(global::ILGPU.Runtime.Accelerator acc) => Remesher = new DeviceRemesher(acc, new SparseKernels(acc));
        public void Dispose() { }
    }

    [TestMethod]
    public async Task DeviceRemesh_Sphere_BothSkins_MatchesTheHost() => await RunTest(async accelerator =>
    {
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        await Dr_Compare(accelerator, p, f, 64, false, "sphere, both skins");
    });

    [TestMethod]
    public async Task DeviceRemesh_Sphere_DropsTheInnerSkin_LikeTheHost() => await RunTest(async accelerator =>
    {
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        await Dr_Compare(accelerator, p, f, 64, true, "sphere, outer skin only");
    });

    [TestMethod]
    public async Task DeviceRemesh_PinholeAndThinSheet_MatchTheHost() => await RunTest(async accelerator =>
    {
        var (p, f) = Rm_ScaledSphere(3, 0.3f);
        var open = new List<int>(f); open.RemoveRange(0, 3);
        await Dr_Compare(accelerator, p, open.ToArray(), 64, false, "sphere with a pinhole");
        var pp = new List<float>(); var ff = new List<int>();
        void Quad(float z)
        {
            int b = pp.Count / 3;
            pp.AddRange(new[] { -0.2f, -0.2f, z, 0.2f, -0.2f, z, 0.2f, 0.2f, z, -0.2f, 0.2f, z });
            ff.AddRange(new[] { b, b + 1, b + 2, b, b + 2, b + 3 });
        }
        Quad(0f); Quad(1f / 64f);
        await Dr_Compare(accelerator, pp.ToArray(), ff.ToArray(), 64, true, "thin double sheet");
    });
}
