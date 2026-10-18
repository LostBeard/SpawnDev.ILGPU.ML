using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.UniRig;

namespace SpawnDev.ILGPU.ML.SkinTokens;

/// <summary>
/// Auto-rigging of a generated character with SkinTokens: a binary glTF in, the same file with a skeleton, a skin and learned per-vertex weights out.
/// <code>
/// mesh      = positions + triangles of the GLB, vertices that coincide (UV seams) welded for the analysis
/// rig       = <see cref="SkinTokensGenerator"/>: skeleton and per-sample skin weights from one pass over 54000 surface samples
/// weights   = every mesh vertex takes the inverse-distance blend of its 8 nearest samples (what the reference's Asset.from_data does), mapped back to every GLB vertex
/// output    = <see cref="RiggedGlb"/> (joint nodes, skin, JOINTS_0 / WEIGHTS_0: the strongest 4, renormalised); textures and material untouched
/// </code>
/// </summary>
public sealed class SkinTokensRigger : IDisposable
{
    public const int NumSamples = 54000, VertexSamples = 16384, NeighbourCount = 8;

    private readonly Accelerator _accelerator;
    private readonly SkinTokensGenerator _generator;

    public sealed record Result(byte[] Glb, UniRigTokenizer.Skeleton Skeleton);

    /// <param name="weights">The SkinTokens export (export_weights.py skintokens).</param>
    public SkinTokensRigger(Accelerator accelerator, GpuWeights weights)
    {
        _accelerator = accelerator;
        _generator = new SkinTokensGenerator(accelerator, weights);
    }

    public async Task<Result> RigGlbAsync(byte[] glb, int seed = 0, int maxSkeletonTokens = 1400)
    {
        var mesh = RiggedGlb.ReadMesh(glb);
        int V = mesh.Positions.Length / 3;

        // weld exact duplicates (a GLB repeats a vertex along every UV seam; the rig only cares about the surface)
        var ids = new Dictionary<(int, int, int), int>(); var map = new int[V]; var welded = new List<float>();
        for (int v = 0; v < V; v++)
        {
            var key = (BitConverter.SingleToInt32Bits(mesh.Positions[v * 3]), BitConverter.SingleToInt32Bits(mesh.Positions[v * 3 + 1]), BitConverter.SingleToInt32Bits(mesh.Positions[v * 3 + 2]));
            if (!ids.TryGetValue(key, out int id)) { ids[key] = id = welded.Count / 3; welded.Add(mesh.Positions[v * 3]); welded.Add(mesh.Positions[v * 3 + 1]); welded.Add(mesh.Positions[v * 3 + 2]); }
            map[v] = id;
        }
        var faces = new List<int>();
        for (int t = 0; t < mesh.Indices.Length; t += 3)
        {
            int a = map[mesh.Indices[t]], b = map[mesh.Indices[t + 1]], c = map[mesh.Indices[t + 2]];
            if (a != b && b != c && c != a) { faces.Add(a); faces.Add(b); faces.Add(c); }
        }
        var wp = welded.ToArray(); var wf = faces.ToArray();

        var s = UniRigMeshSampler.Sample(wp, wf, seed, NumSamples, VertexSamples, new[] { SkinTokensMeshEncoder.Tokens, SkinTokensVae.CondTokens });
        using var pts = _accelerator.Allocate1D(s.Points); using var nrm = _accelerator.Allocate1D(s.Normals);
        var r = await _generator.GenerateAsync(pts.View, nrm.View, NumSamples, s.QuerySets[0], s.QuerySets[1], maxSkeletonTokens);
        int J = r.Skeleton.Count;

        // the samples live in the mesh scaled into [-1,1]; so do the welded vertices for the neighbour search
        var vn = new float[wp.Length];
        for (int v = 0; v < wp.Length / 3; v++) for (int k = 0; k < 3; k++) vn[v * 3 + k] = (wp[v * 3 + k] - s.Center[k]) / s.Scale;
        var skin = Interpolate(s.Points, r.Skin, NumSamples, J, vn, NeighbourCount);

        var weights = new float[V * J];
        for (int v = 0; v < V; v++) Array.Copy(skin, map[v] * J, weights, v * J, J);
        var joints = new float[r.Skeleton.Joints.Length];
        s.ToMeshSpace(r.Skeleton.Joints, joints);
        var skeleton = new UniRigTokenizer.Skeleton(joints, r.Skeleton.Parents);
        return new Result(RiggedGlb.Add(glb, skeleton, weights), skeleton);
    }

    /// <summary>
    /// For each query point the inverse-distance (1 / (d + 1e-8)) blend of the rows of its <paramref name="k"/> nearest sample points; each result row is then
    /// scaled to sum 1. A uniform grid finds the neighbours.
    /// </summary>
    public static float[] Interpolate(float[] samplePoints, float[] sampleRows, int n, int J, float[] queries, int k)
    {
        int q = queries.Length / 3;
        float lo0 = float.MaxValue, lo1 = float.MaxValue, lo2 = float.MaxValue, hi0 = float.MinValue, hi1 = float.MinValue, hi2 = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            lo0 = Math.Min(lo0, samplePoints[i * 3]); hi0 = Math.Max(hi0, samplePoints[i * 3]);
            lo1 = Math.Min(lo1, samplePoints[i * 3 + 1]); hi1 = Math.Max(hi1, samplePoints[i * 3 + 1]);
            lo2 = Math.Min(lo2, samplePoints[i * 3 + 2]); hi2 = Math.Max(hi2, samplePoints[i * 3 + 2]);
        }
        // about 2 samples per cell on the occupied surface: the cube side over (n/2)^(1/3) is generous for a surface and keeps cells small
        float side = Math.Max(hi0 - lo0, Math.Max(hi1 - lo1, hi2 - lo2));
        int g = Math.Clamp((int)MathF.Cbrt(n / 2f), 4, 128); float cell = side / g + 1e-9f;
        int Cell(float v, float lo) => Math.Clamp((int)((v - lo) / cell), 0, g - 1);
        var head = new int[g * g * g]; Array.Fill(head, -1); var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            int c = (Cell(samplePoints[i * 3], lo0) * g + Cell(samplePoints[i * 3 + 1], lo1)) * g + Cell(samplePoints[i * 3 + 2], lo2);
            next[i] = head[c]; head[c] = i;
        }
        var result = new float[q * J];
        Parallel.For(0, q, qi =>
        {
            var found = new (float D, int I)[k]; int count = 0;
            float px = queries[qi * 3], py = queries[qi * 3 + 1], pz = queries[qi * 3 + 2];
            int cx = Cell(px, lo0), cy = Cell(py, lo1), cz = Cell(pz, lo2);
            for (int ring = 0; ring <= g; ring++)
            {
                for (int x = Math.Max(0, cx - ring); x <= Math.Min(g - 1, cx + ring); x++)
                    for (int y = Math.Max(0, cy - ring); y <= Math.Min(g - 1, cy + ring); y++)
                        for (int z = Math.Max(0, cz - ring); z <= Math.Min(g - 1, cz + ring); z++)
                        {
                            if (Math.Max(Math.Abs(x - cx), Math.Max(Math.Abs(y - cy), Math.Abs(z - cz))) != ring) continue;     // only the new shell
                            for (int i = head[(x * g + y) * g + z]; i >= 0; i = next[i])
                            {
                                float dx = samplePoints[i * 3] - px, dy = samplePoints[i * 3 + 1] - py, dz = samplePoints[i * 3 + 2] - pz, d = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                                if (count < k) { found[count++] = (d, i); }
                                else
                                {
                                    int w = 0; for (int t = 1; t < k; t++) if (found[t].D > found[w].D) w = t;
                                    if (d < found[w].D) found[w] = (d, i);
                                }
                            }
                        }
                // every sample within ring*cell of the query is already seen: stop when the k-th best is closer than that
                if (count == k)
                {
                    float far = 0; for (int t = 0; t < k; t++) far = Math.Max(far, found[t].D);
                    if (far <= ring * cell) break;
                }
            }
            float sumW = 0; var row = new float[J];
            for (int t = 0; t < count; t++)
            {
                float w = 1f / (found[t].D + 1e-8f); sumW += w;
                for (int j = 0; j < J; j++) row[j] += w * sampleRows[found[t].I * J + j];
            }
            float total = 0; for (int j = 0; j < J; j++) { row[j] /= sumW; total += row[j]; }
            float inv = 1f / MathF.Max(total, 1e-8f);
            for (int j = 0; j < J; j++) result[qi * J + j] = row[j] * inv;
        });
        return result;
    }

    public void Dispose() => _generator.Dispose();
}
