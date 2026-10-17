using System.Diagnostics;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Run the UV unwrap on a real generated mesh (file written by reference/glb2mesh.py: int nv, int nf, float[nv*3], int[nf*3]) and report
/// charts / utilisation / time. Env: ATLAS_CONE (degrees), ATLAS_PAD (texels), ATLAS_RES.
/// </summary>
internal static class AtlasMeshRun
{
    public static int Run(string path)
    {
        var b = File.ReadAllBytes(path);
        int nv = BitConverter.ToInt32(b, 0), nf = BitConverter.ToInt32(b, 4);
        var v = new float[nv * 3]; var f = new int[nf * 3];
        Buffer.BlockCopy(b, 8, v, 0, v.Length * 4); Buffer.BlockCopy(b, 8 + v.Length * 4, f, 0, f.Length * 4);
        var o = new UvAtlas.Options();
        string? s;
        var opts = new UvAtlas.Options
        {
            MaxNormalDeviationDegrees = (s = Environment.GetEnvironmentVariable("ATLAS_CONE")) != null ? double.Parse(s) : o.MaxNormalDeviationDegrees,
            PaddingTexels = (s = Environment.GetEnvironmentVariable("ATLAS_PAD")) != null ? int.Parse(s) : o.PaddingTexels,
            Resolution = (s = Environment.GetEnvironmentVariable("ATLAS_RES")) != null ? int.Parse(s) : o.Resolution,
        };
        var sw = Stopwatch.StartNew();
        var r = UvAtlas.Unwrap(v, f, opts);
        Console.WriteLine($"{nf} faces -> {r.ChartCount} charts, utilisation {r.Utilization:P1}, {r.TexelsPerUnit:F0} texels/unit, {r.Positions.Length / 3} verts, {sw.Elapsed.TotalSeconds:F2}s");
        // effective texel density in "per voxel": the mesh spans ~1 unit = 512 voxels
        Console.WriteLine($"texels per voxel edge: {r.TexelsPerUnit / 512:F2}");
        return 0;
    }
}
