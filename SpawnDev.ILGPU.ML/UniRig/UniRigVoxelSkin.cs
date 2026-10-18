namespace SpawnDev.ILGPU.ML.UniRig;

/// <summary>
/// The skinning PRIOR of UniRig ("voxel skin"; geodesic voxel binding, Dionne and de Lasa 2013): a weight of every mesh vertex for every bone, from distances measured INSIDE
/// the shape. The learned skin network takes it as an input and as a mask, and on its own it is a sound skinning method. Port of the reference's <c>voxelization</c> (the
/// visual hull of six axis-aligned depth views) and <c>voxel_skin</c> (shortest paths over a graph of the mesh vertices and the interior voxels).
/// <code>
/// voxels   = cells inside along at least two axes, an axis counting a cell as inside between the nearest surface hits from both sides      (grid 196 over [-1,1]^3)
/// graph    = mesh edges (length) + very close vertices + voxel face neighbours (length x 3) + each voxel to its nearest vertex within one voxel
/// distance = Dijkstra from the graph node nearest each joint
/// weight   = 1 / ((1 - a) d + a d^2)^2, normalised over joints                                                                                   (a = 0.5)
/// </code>
/// Everything is in the mesh scaled into [-1,1] (<see cref="UniRigMeshSampler"/> conventions). Host-side: the graph is irregular and the work is a few seconds of Dijkstra,
/// parallel over joints.
/// </summary>
public static class UniRigVoxelSkin
{
    /// <summary>Interior voxels of an n x n x n grid over [-1,1]^3, x-major order, three ints (i, j, k) each. Voxel centre = (i + 0.5) * 2 / n - 1.</summary>
    public sealed record Voxels(int Grid, int[] Cells)
    {
        public int Count => Cells.Length / 3;
        public double Coord(int voxel, int axis) => (Cells[voxel * 3 + axis] + 0.5) * 2.0 / Grid - 1.0;
    }

    private const double JitterU = 0.0127, JitterV = 0.0093;   // fractions of a cell: a ray essentially never runs along a shared edge

    /// <summary>Voxels from float centres written by the reference's golden (or any (i + 0.5) * 2/n - 1 centres).</summary>
    public static Voxels FromCoords(float[] coords, int grid)
    {
        int m = coords.Length / 3; var cells = new int[m * 3];
        for (int v = 0; v < m; v++)
            for (int a = 0; a < 3; a++) cells[v * 3 + a] = (int)Math.Round((coords[v * 3 + a] + 1.0) * grid / 2.0 - 0.5);
        return new Voxels(grid, cells);
    }

    /// <summary>The visual hull of <paramref name="v"/> (xyz in [-1,1], <paramref name="f"/> triangles) at <paramref name="grid"/>^3, a voxel inside when two of three axes say so.</summary>
    public static Voxels Voxelize(float[] v, int[] f, int grid)
    {
        var lo = new float[3][]; var hi = new float[3][];
        for (int a = 0; a < 3; a++)
        {
            lo[a] = new float[grid * grid]; hi[a] = new float[grid * grid];
            Array.Fill(lo[a], float.PositiveInfinity); Array.Fill(hi[a], float.NegativeInfinity);
            int u = (a + 1) % 3, w = (a + 2) % 3;
            double cell = 2.0 / grid;
            for (int t = 0; t < f.Length; t += 3)
            {
                int i0 = f[t], i1 = f[t + 1], i2 = f[t + 2];
                // lattice coordinates with the rays at integers: ((x + 1) / cell) - 0.5 - jitter
                double x0 = (v[i0 * 3 + u] + 1) / cell - 0.5 - JitterU, y0 = (v[i0 * 3 + w] + 1) / cell - 0.5 - JitterV;
                double x1 = (v[i1 * 3 + u] + 1) / cell - 0.5 - JitterU, y1 = (v[i1 * 3 + w] + 1) / cell - 0.5 - JitterV;
                double x2 = (v[i2 * 3 + u] + 1) / cell - 0.5 - JitterU, y2 = (v[i2 * 3 + w] + 1) / cell - 0.5 - JitterV;
                double det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
                if (Math.Abs(det) < 1e-18) continue;
                int ia = Math.Max(0, (int)Math.Ceiling(Math.Min(x0, Math.Min(x1, x2)))), ib = Math.Min(grid - 1, (int)Math.Floor(Math.Max(x0, Math.Max(x1, x2))));
                int ja = Math.Max(0, (int)Math.Ceiling(Math.Min(y0, Math.Min(y1, y2)))), jb = Math.Min(grid - 1, (int)Math.Floor(Math.Max(y0, Math.Max(y1, y2))));
                double z0 = v[i0 * 3 + a], z1 = v[i1 * 3 + a], z2 = v[i2 * 3 + a];
                for (int j = ja; j <= jb; j++)
                    for (int i = ia; i <= ib; i++)
                    {
                        double w0 = ((y1 - y2) * (i - x2) + (x2 - x1) * (j - y2)) / det, w1 = ((y2 - y0) * (i - x2) + (x0 - x2) * (j - y2)) / det, w2 = 1 - w0 - w1;
                        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                        float z = (float)(w0 * z0 + w1 * z1 + w2 * z2); int c = i * grid + j;
                        if (z < lo[a][c]) lo[a][c] = z; if (z > hi[a][c]) hi[a][c] = z;
                    }
            }
        }
        var cells = new List<int>();
        for (int i = 0; i < grid; i++)
            for (int j = 0; j < grid; j++)
                for (int k = 0; k < grid; k++)
                {
                    float cx = (float)((i + 0.5) * 2.0 / grid - 1.0), cy = (float)((j + 0.5) * 2.0 / grid - 1.0), cz = (float)((k + 0.5) * 2.0 / grid - 1.0);
                    // axis 0: rays over (y, z) at cell (j, k); axis 1: over (z, x) at (k, i); axis 2: over (x, y) at (i, j)
                    int votes = 0;
                    if (lo[0][j * grid + k] <= cx && cx <= hi[0][j * grid + k]) votes++;
                    if (lo[1][k * grid + i] <= cy && cy <= hi[1][k * grid + i]) votes++;
                    if (lo[2][i * grid + j] <= cz && cz <= hi[2][i * grid + j]) votes++;
                    if (votes >= 2) { cells.Add(i); cells.Add(j); cells.Add(k); }
                }
        return new Voxels(grid, cells.ToArray());
    }

    /// <summary>
    /// <paramref name="v"/> mesh vertices xyz and <paramref name="joints"/> J joint positions xyz, both in [-1,1]; returns [N * J] weights, row n the weights of vertex n over the joints.
    /// </summary>
    public static float[] Compute(float[] v, int[] f, float[] joints, Voxels vox, double alpha = 0.5, double linkDis = 1e-5, double gridWeight = 3.0)
    {
        int n = v.Length / 3, J = joints.Length / 3, m = vox.Count, grid = vox.Grid, total = n + m;
        double reach = 2.0 / grid * 1.001;                                           // one voxel, with the reference's slack

        // ---- directed arcs, duplicates SUMMED (scipy's csr_matrix does that; a directed edge listed twice, inconsistent winding or a repeated face, counts double)
        var arcs = new Dictionary<long, double>();
        void Arc(int s, int d, double w) { long key = (long)s * total + d; arcs[key] = arcs.TryGetValue(key, out var old) ? old + w : w; }
        for (int t = 0; t < f.Length; t += 3)
            for (int e = 0; e < 3; e++)
            {
                int a = f[t + e], b = f[t + (e + 1) % 3];
                double dx = (double)v[a * 3] - v[b * 3], dy = (double)v[a * 3 + 1] - v[b * 3 + 1], dz = (double)v[a * 3 + 2] - v[b * 3 + 2];
                Arc(a, b, Math.Sqrt(dx * dx + dy * dy + dz * dz));
            }

        // ---- very close vertices (0 < distance <= linkDis): up to three nearest others
        var hash = new Dictionary<(int, int, int), List<int>>();
        double cellV = Math.Max(reach, linkDis * 1.001);
        (int, int, int) Cell(double x, double y, double z) => ((int)Math.Floor(x / cellV), (int)Math.Floor(y / cellV), (int)Math.Floor(z / cellV));
        for (int i = 0; i < n; i++)
        {
            var c = Cell(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
            if (!hash.TryGetValue(c, out var l)) hash[c] = l = new List<int>();
            l.Add(i);
        }
        var near = new List<(double D, int J)>();
        for (int i = 0; i < n; i++)
        {
            near.Clear();
            var (cx, cy, cz) = Cell(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++)
                if (hash.TryGetValue((cx + dx, cy + dy, cz + dz), out var l))
                    foreach (int j in l)
                    {
                        if (j == i) continue;
                        double ex = (double)v[i * 3] - v[j * 3], ey = (double)v[i * 3 + 1] - v[j * 3 + 1], ez = (double)v[i * 3 + 2] - v[j * 3 + 2], d = Math.Sqrt(ex * ex + ey * ey + ez * ez);
                        if (d > 0 && d < linkDis * 1.001) near.Add((d, j));
                    }
            near.Sort((a, b) => a.D.CompareTo(b.D));
            for (int k = 0; k < Math.Min(3, near.Count); k++) Arc(i, near[k].J, near[k].D);
        }

        // ---- voxels: face neighbours (weight length x gridWeight), and each voxel to its nearest mesh vertex when within one voxel
        var cellIndex = new int[(long)grid * grid * grid]; Array.Fill(cellIndex, -1);
        for (int q = 0; q < m; q++) cellIndex[((long)vox.Cells[q * 3] * grid + vox.Cells[q * 3 + 1]) * grid + vox.Cells[q * 3 + 2]] = q;
        double step = 2.0 / grid;
        for (int q = 0; q < m; q++)
        {
            int ci = vox.Cells[q * 3], cj = vox.Cells[q * 3 + 1], ck = vox.Cells[q * 3 + 2];
            for (int axis = 0; axis < 3; axis++)
                for (int s = -1; s <= 1; s += 2)
                {
                    int ni = ci + (axis == 0 ? s : 0), nj = cj + (axis == 1 ? s : 0), nk = ck + (axis == 2 ? s : 0);
                    if (ni < 0 || nj < 0 || nk < 0 || ni >= grid || nj >= grid || nk >= grid) continue;
                    int other = cellIndex[((long)ni * grid + nj) * grid + nk];
                    if (other >= 0) Arc(n + q, n + other, step * gridWeight);
                }
            double px = vox.Coord(q, 0), py = vox.Coord(q, 1), pz = vox.Coord(q, 2);
            var (hx, hy, hz) = Cell(px, py, pz);
            int best = -1; double bd = double.MaxValue;
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++)
                if (hash.TryGetValue((hx + dx, hy + dy, hz + dz), out var l))
                    foreach (int j in l)
                    {
                        double ex = px - v[j * 3], ey = py - v[j * 3 + 1], ez = pz - v[j * 3 + 2], d = Math.Sqrt(ex * ex + ey * ey + ez * ez);
                        if (d < bd || (d == bd && j < best)) { bd = d; best = j; }
                    }
            if (best >= 0 && bd > 0 && bd < reach) Arc(n + q, best, bd);
        }

        // ---- undirected adjacency in CSR form (an arc is usable in both directions, each with its own weight)
        var deg = new int[total + 1];
        foreach (var key in arcs.Keys) { deg[key / total + 1]++; deg[key % total + 1]++; }
        for (int i = 0; i < total; i++) deg[i + 1] += deg[i];
        var adj = new int[deg[total]]; var wt = new double[deg[total]]; var fill = new int[total];
        foreach (var (key, w) in arcs)
        {
            int s = (int)(key / total), d = (int)(key % total);
            int p = deg[s] + fill[s]++; adj[p] = d; wt[p] = w;
            int r = deg[d] + fill[d]++; adj[r] = s; wt[r] = w;
        }

        // ---- each joint starts at the graph node nearest to it (mesh vertices and voxel centres alike; the lowest index wins a tie)
        var start = new int[J];
        for (int j = 0; j < J; j++)
        {
            double jx = joints[j * 3], jy = joints[j * 3 + 1], jz = joints[j * 3 + 2], bd = double.MaxValue; int best = 0;
            for (int i = 0; i < n; i++)
            {
                double ex = jx - v[i * 3], ey = jy - v[i * 3 + 1], ez = jz - v[i * 3 + 2], d = ex * ex + ey * ey + ez * ez;
                if (d < bd) { bd = d; best = i; }
            }
            for (int q = 0; q < m; q++)
            {
                double ex = jx - vox.Coord(q, 0), ey = jy - vox.Coord(q, 1), ez = jz - vox.Coord(q, 2), d = ex * ex + ey * ey + ez * ez;
                if (d < bd) { bd = d; best = n + q; }
            }
            start[j] = best;
        }

        // ---- shortest paths, one Dijkstra per joint
        var dist = new double[J][];
        Parallel.For(0, J, j =>
        {
            var d = new double[total]; Array.Fill(d, double.PositiveInfinity);
            var queue = new PriorityQueue<int, double>();
            d[start[j]] = 0; queue.Enqueue(start[j], 0);
            while (queue.TryDequeue(out int u, out double du))
            {
                if (du > d[u]) continue;
                for (int p = deg[u]; p < deg[u + 1]; p++)
                {
                    double nd = du + wt[p];
                    if (nd < d[adj[p]]) { d[adj[p]] = nd; queue.Enqueue(adj[p], nd); }
                }
            }
            dist[j] = d;
        });

        // ---- vertices no joint reaches get the straight distance to their three nearest joints (every vertex must have a finite distance to somebody)
        var vd = new double[J * n];                                                  // [joint, vertex]
        for (int j = 0; j < J; j++) for (int i = 0; i < n; i++) vd[j * n + i] = dist[j][i];
        for (int i = 0; i < n; i++)
        {
            bool any = false; for (int j = 0; j < J && !any; j++) any = !double.IsPositiveInfinity(vd[j * n + i]);
            if (any) continue;
            var order = Enumerable.Range(0, J).Select(j =>
            {
                double ex = v[i * 3] - joints[j * 3], ey = v[i * 3 + 1] - joints[j * 3 + 1], ez = v[i * 3 + 2] - joints[j * 3 + 2];
                return (D: Math.Sqrt(ex * ex + ey * ey + ez * ez), J: j);
            }).OrderBy(x => x.D).Take(Math.Min(J, 3));
            foreach (var (d, j) in order) vd[j * n + i] = d;
        }
        double maxDis = 0;
        foreach (double d in vd) if (!double.IsInfinity(d) && !double.IsNaN(d) && d > maxDis) maxDis = d;
        var skin = new float[n * J];
        for (int i = 0; i < n; i++)
        {
            double sum = 0; var w = new double[J];
            for (int j = 0; j < J; j++)
            {
                double d = vd[j * n + i];
                if (double.IsInfinity(d) || double.IsNaN(d)) d = maxDis;            // pairs still unreachable: the largest finite distance
                d = Math.Max(d, 1e-6);
                double q = 1.0 / ((1 - alpha) * d + alpha * d * d);
                w[j] = q * q; sum += w[j];
            }
            for (int j = 0; j < J; j++) skin[i * J + j] = (float)(w[j] / sum);
        }
        return skin;
    }
}
