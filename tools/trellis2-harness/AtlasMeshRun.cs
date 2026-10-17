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
        if (Environment.GetEnvironmentVariable("ATLAS_DIAG") == "1") Diagnose(v, f, nf);
        var sw = Stopwatch.StartNew();
        var r = UvAtlas.Unwrap(v, f, opts);
        Console.WriteLine($"{nf} faces -> {r.ChartCount} charts, utilisation {r.Utilization:P1}, {r.TexelsPerUnit:F0} texels/unit, {r.Positions.Length / 3} verts, {sw.Elapsed.TotalSeconds:F2}s");
        // effective texel density in "per voxel": the mesh spans ~1 unit = 512 voxels
        Console.WriteLine($"texels per voxel edge: {r.TexelsPerUnit / 512:F2}");
        return 0;
    }

    /// <summary>Edge census: how many face edges have exactly one partner (manifold), none (boundary) or several (non-manifold), and the dihedral angles of the manifold ones.</summary>
    private static void Diagnose(float[] v, int[] f, int nf)
    {
        var edges = new Dictionary<long, List<int>>();
        for (int t = 0; t < nf; t++)
            for (int e = 0; e < 3; e++)
            {
                int a = f[t * 3 + e], b = f[t * 3 + (e + 1) % 3];
                long key = a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
                if (!edges.TryGetValue(key, out var l)) edges[key] = l = new List<int>(2);
                l.Add(t);
            }
        var n = new double[nf * 3];
        for (int t = 0; t < nf; t++)
        {
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            double ux = v[b * 3] - v[a * 3], uy = v[b * 3 + 1] - v[a * 3 + 1], uz = v[b * 3 + 2] - v[a * 3 + 2];
            double wx = v[c * 3] - v[a * 3], wy = v[c * 3 + 1] - v[a * 3 + 1], wz = v[c * 3 + 2] - v[a * 3 + 2];
            double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx, l = Math.Sqrt(nx * nx + ny * ny + nz * nz) + 1e-30;
            n[t * 3] = nx / l; n[t * 3 + 1] = ny / l; n[t * 3 + 2] = nz / l;
        }
        int boundary = 0, manifold = 0, nonManifold = 0;
        var ang = new List<double>();
        foreach (var l in edges.Values)
        {
            if (l.Count == 1) boundary++;
            else if (l.Count == 2) { manifold++; int p = l[0], q = l[1]; double d = n[p * 3] * n[q * 3] + n[p * 3 + 1] * n[q * 3 + 1] + n[p * 3 + 2] * n[q * 3 + 2]; ang.Add(Math.Acos(Math.Clamp(d, -1, 1)) * 180 / Math.PI); }
            else nonManifold++;
        }
        ang.Sort();
        string P(double q) => $"p{q * 100:F0}={ang[(int)(q * (ang.Count - 1))]:F1}";
        Console.WriteLine($"edges: {manifold} manifold, {boundary} boundary, {nonManifold} non-manifold; dihedral {P(0.25)} {P(0.5)} {P(0.75)} {P(0.9)} {P(0.99)} deg");
    }
}
