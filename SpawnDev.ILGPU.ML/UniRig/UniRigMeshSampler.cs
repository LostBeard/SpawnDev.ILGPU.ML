namespace SpawnDev.ILGPU.ML.UniRig;

/// <summary>
/// Turns a triangle mesh into UniRig's input: a point cloud with normals scaled into [-1,1], and the 1024 encoder query points. The recipe is the reference's
/// (tools/unirig-harness golden_skeleton.py sample_points / the encoder's eval path): centre the mesh and scale its longest side to 2; take 8192 random vertices with
/// their area-weighted vertex normals plus 57344 surface samples (a face is picked in proportion to the SQUARE of its cross-product length, as the reference does,
/// then a uniform point on it) with the face normal, 65536 in all; pre-sample 4096 of them at random and keep 1024 by farthest-point sampling from index 0 as queries.
/// Host-side and small (a few tens of milliseconds); the cloud is a terminal input of the device pipeline.
/// </summary>
public sealed class UniRigMeshSampler
{
    public const int NumSamples = 65536, VertexSamples = 8192, PreSamples = 4096;

    /// <summary>[n,3] points in [-1,1].</summary>
    public float[] Points { get; }
    /// <summary>[n,3] unit normals.</summary>
    public float[] Normals { get; }
    /// <summary>The 1024 point indices that become the encoder's query tokens.</summary>
    public int[] QueryRows => QuerySets[0];
    /// <summary>One farthest-point-sampled query set per requested size (<see cref="Sample(float[], int[], int, int, int, int[])"/>); the first is <see cref="QueryRows"/>.</summary>
    public int[][] QuerySets { get; }
    /// <summary>Mesh centre (bounding box middle): mesh position = normalised * <see cref="Scale"/> + this.</summary>
    public float[] Center { get; }
    /// <summary>Half the longest bounding-box side.</summary>
    public float Scale { get; }

    private UniRigMeshSampler(float[] points, float[] normals, int[][] rows, float[] center, float scale)
    { Points = points; Normals = normals; QuerySets = rows; Center = center; Scale = scale; }

    /// <summary>Map a joint position from the normalised space back into the mesh's own space.</summary>
    public void ToMeshSpace(ReadOnlySpan<float> joints, Span<float> o)
    {
        for (int i = 0; i < joints.Length; i += 3)
            for (int k = 0; k < 3; k++) o[i + k] = joints[i + k] * Scale + Center[k];
    }

    public static UniRigMeshSampler Sample(float[] positions, int[] faces, int seed = 0) => Sample(positions, faces, seed, NumSamples, VertexSamples, new[] { PreSamples / 4 });

    /// <summary>
    /// The same recipe for another network: <paramref name="numSamples"/> points (<paramref name="vertexSamples"/> of them vertex samples) and, for each entry of
    /// <paramref name="queryCounts"/>, 4x that many pre-sampled points reduced by farthest-point sampling to the query set (SkinTokens: 54000 / 16384 and 512 + 384 queries).
    /// </summary>
    public static UniRigMeshSampler Sample(float[] positions, int[] faces, int seed, int numSamples, int vertexSamples, int[] queryCounts)
    {
        int nv = positions.Length / 3, nf = faces.Length / 3;
        if (nv == 0 || nf == 0) throw new ArgumentException("empty mesh");
        float[] lo = { float.MaxValue, float.MaxValue, float.MaxValue }, hi = { float.MinValue, float.MinValue, float.MinValue };
        for (int v = 0; v < nv; v++)
            for (int k = 0; k < 3; k++) { float c = positions[v * 3 + k]; if (c < lo[k]) lo[k] = c; if (c > hi[k]) hi[k] = c; }
        float scale = Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2])) / 2;
        if (scale <= 0) throw new ArgumentException("degenerate mesh");
        var center = new[] { (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2 };
        var p = new float[positions.Length];
        for (int v = 0; v < nv; v++) for (int k = 0; k < 3; k++) p[v * 3 + k] = (positions[v * 3 + k] - center[k]) / scale;

        // face normals (unit), squared cross-product lengths (the sampling weights), area-weighted vertex normals
        var fn = new float[nf * 3]; var weight = new double[nf]; var vn = new double[nv * 3];
        for (int t = 0; t < nf; t++)
        {
            int a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2];
            double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
            double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, len2 = nx * nx + ny * ny + nz * nz, len = Math.Sqrt(len2);
            weight[t] = len2;
            if (len > 0) { fn[t * 3] = (float)(nx / len); fn[t * 3 + 1] = (float)(ny / len); fn[t * 3 + 2] = (float)(nz / len); }
            foreach (int v in new[] { a, b, c }) { vn[v * 3] += nx; vn[v * 3 + 1] += ny; vn[v * 3 + 2] += nz; }
        }
        for (int v = 0; v < nv; v++)
        {
            double l = Math.Sqrt(vn[v * 3] * vn[v * 3] + vn[v * 3 + 1] * vn[v * 3 + 1] + vn[v * 3 + 2] * vn[v * 3 + 2]);
            if (l > 0) { vn[v * 3] /= l; vn[v * 3 + 1] /= l; vn[v * 3 + 2] /= l; }
        }
        var cum = new double[nf]; double run = 0;
        for (int t = 0; t < nf; t++) { run += weight[t]; cum[t] = run; }

        var rng = new Random(seed);
        int nVertexSamples = Math.Min(vertexSamples, nv), nSurface = numSamples - nVertexSamples;
        var points = new float[numSamples * 3]; var normals = new float[numSamples * 3];
        var pick = Shuffled(nv, nVertexSamples, rng);
        for (int i = 0; i < nVertexSamples; i++)
            for (int k = 0; k < 3; k++) { points[i * 3 + k] = p[pick[i] * 3 + k]; normals[i * 3 + k] = (float)vn[pick[i] * 3 + k]; }
        for (int i = 0; i < nSurface; i++)
        {
            double target = rng.NextDouble() * run;
            int lo2 = 0, hi2 = nf - 1;                                           // first face whose cumulative weight reaches the target
            while (lo2 < hi2) { int mid = (lo2 + hi2) >> 1; if (cum[mid] < target) lo2 = mid + 1; else hi2 = mid; }
            int t = lo2;
            double r1 = rng.NextDouble(), r2 = rng.NextDouble();
            if (r1 + r2 > 1) { r1 -= 1; r2 -= 1; r1 = Math.Abs(r1); r2 = Math.Abs(r2); }   // the reference's fold of the unit square onto the triangle
            int a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2], o = (nVertexSamples + i) * 3;
            for (int k = 0; k < 3; k++)
            {
                points[o + k] = (float)(p[a * 3 + k] + (p[b * 3 + k] - p[a * 3 + k]) * r1 + (p[c * 3 + k] - p[a * 3 + k]) * r2);
                normals[o + k] = fn[t * 3 + k];
            }
        }
        // the encoder's eval-mode query points: 4096 at random, then farthest-point sampling keeps a quarter, starting from the first
        var sets = new int[queryCounts.Length][];
        for (int q = 0; q < queryCounts.Length; q++)
        {
            var pre = Shuffled(numSamples, queryCounts[q] * 4, rng);
            var fps = FarthestPoints(points, pre, queryCounts[q]);
            sets[q] = fps.Select(i => pre[i]).ToArray();
        }
        return new UniRigMeshSampler(points, normals, sets, center, scale);
    }

    /// <summary>The first <paramref name="count"/> of a random permutation of 0..n-1 (partial Fisher-Yates).</summary>
    private static int[] Shuffled(int n, int count, Random rng)
    {
        var a = new int[n]; for (int i = 0; i < n; i++) a[i] = i;
        for (int i = 0; i < count; i++) { int j = i + rng.Next(n - i); (a[i], a[j]) = (a[j], a[i]); }
        return a.AsSpan(0, count).ToArray();
    }

    /// <summary>Farthest-point sampling over points[rows]: the first chosen is rows[0], then repeatedly the point farthest from everything chosen. Returns indices into <paramref name="rows"/>.</summary>
    private static int[] FarthestPoints(float[] points, int[] rows, int count)
    {
        var chosen = new int[count]; var dist = new float[rows.Length];
        Array.Fill(dist, float.MaxValue);
        int cur = 0;
        for (int s = 0; s < count; s++)
        {
            chosen[s] = cur;
            int cr = rows[cur] * 3; float cx = points[cr], cy = points[cr + 1], cz = points[cr + 2];
            int far = 0; float best = -1f;
            for (int i = 0; i < rows.Length; i++)
            {
                int r = rows[i] * 3; float dx = points[r] - cx, dy = points[r + 1] - cy, dz = points[r + 2] - cz, d = dx * dx + dy * dy + dz * dz;
                if (d < dist[i]) dist[i] = d;
                if (dist[i] > best) { best = dist[i]; far = i; }
            }
            cur = far;
        }
        return chosen;
    }
}
