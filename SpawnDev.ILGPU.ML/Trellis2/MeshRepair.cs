namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Makes a generated triangle mesh well formed: no degenerate or duplicate faces, no non-manifold edges or pinched vertices, no floating
/// fragments, small holes closed, every face wound the same way and the surface facing outward. The reference's <c>to_glb</c> runs
/// <c>remove_duplicate_faces</c>, <c>repair_non_manifold_edges</c>, <c>remove_small_connected_components</c>, <c>fill_holes</c> and
/// <c>unify_face_orientations</c> (CuMesh) on the dual-grid mesh before it simplifies, unwraps and bakes it; the dual-grid extraction itself is
/// not manifold (MEASURED on the crown, 3,003,556 faces: 4.7% open edges, 227,427 non-manifold edges, 730,482 orientation conflicts).
/// <para>
/// Host-side, sequential, O(n log n): the topology work that follows (edge collapse, UV charts) is host-side too, and the mesh is about to
/// become a file. <b>No geometry is deleted to fix a non-manifold edge</b>: vertices are split so each fan of faces that is connected
/// through manifold edges owns its vertex (two sheets that touch along an edge stay two sheets). Only degenerate and duplicate faces and
/// fragments smaller than <see cref="Options.MinComponentArea"/> are removed.
/// </para>
/// </summary>
public static class MeshRepair
{
    public sealed class Options
    {
        /// <summary>Holes with a boundary loop up to this long are closed, as a fraction of the longest bounding-box side (reference: 0.03 on a unit box).</summary>
        public double MaxHolePerimeter { get; init; } = 0.03;
        /// <summary>Connected pieces with less area than this are dropped, as a fraction of the longest bounding-box side squared (reference: 1e-5).</summary>
        public double MinComponentArea { get; init; } = 1e-5;
        public bool FillHoles { get; init; } = true;
        /// <summary>Flip a consistently wound piece whose signed volume is negative (inside out).</summary>
        public bool OrientOutward { get; init; } = true;
    }

    /// <summary>What a mesh looks like to a rigger, a physics engine or a subdivision step.</summary>
    /// <param name="BoundaryEdges">Edges used by exactly one face (open boundary).</param>
    /// <param name="NonManifoldEdges">Edges used by more than two faces.</param>
    /// <param name="OrientationConflicts">Edges shared by exactly two faces that traverse it in the SAME direction.</param>
    /// <param name="Components">Pieces connected through shared edges.</param>
    public readonly record struct Topology(int Vertices, int Faces, int BoundaryEdges, int NonManifoldEdges, int OrientationConflicts, int Components);

    public readonly record struct Report(Topology Before, Topology After, int DegenerateFaces, int DuplicateFaces, int VerticesSplit,
        int FragmentComponents, int FragmentFaces, int HolesFilled, int HoleFacesAdded, int FacesFlipped, int PiecesTurnedOutward, int ExcessFacesDropped = 0);

    // ──────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Repair <paramref name="positions"/> (xyz) / <paramref name="faces"/> (triangle indices); returns new compacted arrays and a report.</summary>
    public static (float[] Positions, int[] Faces, Report Report) Repair(float[] positions, int[] faces, Options? options = null)
    {
        options ??= new Options();
        var before = Measure(positions, faces);
        double scale = LongestSide(positions);
        if (scale <= 0 || faces.Length < 3) return (positions, faces, new Report(before, before, 0, 0, 0, 0, 0, 0, 0, 0, 0));

        var f = DropDegenerateAndDuplicates(positions, faces, 1e-12 * scale * scale, out int degenerate, out int duplicate);
        var (p, f2, split) = SplitNonManifold(positions, f);
        int droppedExcess = 0;
        for (int pass = 0; pass < 4 && Measure(p, f2).NonManifoldEdges > 0; pass++)
        {
            f2 = DropExcessFaces(p, f2, out int d); droppedExcess += d;
            var (p2, f3, s2) = SplitNonManifold(p, f2); (p, f2) = (p2, f3); split += s2;
        }
        f2 = RemoveFragments(p, f2, options.MinComponentArea * scale * scale, out int fragComponents, out int fragFaces);
        (p, f2) = Compact(p, f2);
        Orient(p, f2, options.OrientOutward, out int flipped, out int turned);
        int holes = 0, added = 0;
        if (options.FillHoles) (p, f2, holes, added) = FillHoles(p, f2, options.MaxHolePerimeter * scale);
        if (holes > 0) { Orient(p, f2, options.OrientOutward, out int flipped2, out int turned2); flipped += flipped2; turned += turned2; }   // the fans join the one winding
        (p, f2) = Compact(p, f2);
        var after = Measure(p, f2);
        return (p, f2, new Report(before, after, degenerate, duplicate, split, fragComponents, fragFaces, holes, added, flipped, turned, droppedExcess));
    }

    /// <summary>Topology of a mesh, exactly as <c>reference/mesh_audit.py</c> counts it (that script is the independent check on this one).</summary>
    public static Topology Measure(float[] positions, int[] faces)
    {
        int nv = positions.Length / 3, nf = faces.Length / 3;
        if (nf == 0) return new Topology(nv, 0, 0, 0, 0, 0);
        var e = BuildEdges(faces, nv);
        int boundary = 0, nonManifold = 0, conflicts = 0;
        var dsu = new int[nf]; for (int i = 0; i < nf; i++) dsu[i] = i;
        for (int k = 0; k < e.Edges; k++)
        {
            int c = e.Count[k], s = e.Start[k];
            if (c == 1) boundary++;
            else if (c > 2) nonManifold++;
            else if (faces[e.Order[s]] == faces[e.Order[s + 1]]) conflicts++;
            for (int i = 1; i < c; i++) Union(dsu, e.Order[s] / 3, e.Order[s + i] / 3);
        }
        int comps = 0; for (int i = 0; i < nf; i++) if (Find(dsu, i) == i) comps++;
        return new Topology(nv, nf, boundary, nonManifold, conflicts, comps);
    }

    // ── edge table ────────────────────────────────────────────────────────────────────────────────

    /// <summary>Half-edge h = face * 3 + k runs from faces[h] to faces[Next(h)].</summary>
    private static int Next(int h) => h % 3 == 2 ? h - 2 : h + 1;

    private sealed class EdgeTable
    {
        public int Edges;
        public int[] EdgeOfHalf = null!;   // half-edge -> unique edge
        public int[] Start = null!;        // unique edge -> first slot in Order
        public int[] Count = null!;        // unique edge -> number of half-edges
        public int[] Order = null!;        // half-edges sorted by edge
    }

    private static EdgeTable BuildEdges(int[] f, int nv)
    {
        int nh = f.Length;
        var key = new long[nh]; var order = new int[nh];
        for (int h = 0; h < nh; h++)
        {
            int a = f[h], b = f[Next(h)];
            key[h] = a < b ? (long)a * nv + b : (long)b * nv + a;
            order[h] = h;
        }
        Array.Sort(key, order);
        int unique = 0;
        for (int i = 0; i < nh; i++) if (i == 0 || key[i] != key[i - 1]) unique++;
        var t = new EdgeTable { Edges = unique, EdgeOfHalf = new int[nh], Start = new int[unique], Count = new int[unique], Order = order };
        int e = -1;
        for (int i = 0; i < nh; i++)
        {
            if (i == 0 || key[i] != key[i - 1]) { e++; t.Start[e] = i; }
            t.Count[e]++;
            t.EdgeOfHalf[order[i]] = e;
        }
        return t;
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }

    private static void Union(int[] parent, int a, int b)
    {
        a = Find(parent, a); b = Find(parent, b);
        if (a != b) parent[Math.Max(a, b)] = Math.Min(a, b);
    }

    private static double LongestSide(float[] p)
    {
        if (p.Length < 3) return 0;
        double[] lo = { double.MaxValue, double.MaxValue, double.MaxValue }, hi = { double.MinValue, double.MinValue, double.MinValue };
        for (int i = 0; i < p.Length; i += 3)
            for (int k = 0; k < 3; k++) { if (p[i + k] < lo[k]) lo[k] = p[i + k]; if (p[i + k] > hi[k]) hi[k] = p[i + k]; }
        return Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2]));
    }

    private static double Area2(float[] p, int a, int b, int c)
    {
        double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
        double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        return Math.Sqrt(nx * nx + ny * ny + nz * nz);   // twice the area
    }

    // ── 1. degenerate and duplicate faces ─────────────────────────────────────────────────────────

    private static int[] DropDegenerateAndDuplicates(float[] p, int[] f, double minArea2, out int degenerate, out int duplicate)
    {
        degenerate = 0; duplicate = 0;
        var seen = new HashSet<(int, int, int)>(f.Length / 3);
        var kept = new List<int>(f.Length);
        for (int i = 0; i < f.Length; i += 3)
        {
            int a = f[i], b = f[i + 1], c = f[i + 2];
            if (a == b || b == c || c == a || Area2(p, a, b, c) < minArea2) { degenerate++; continue; }
            int lo = Math.Min(a, Math.Min(b, c)), hi = Math.Max(a, Math.Max(b, c)), mid = a + b + c - lo - hi;
            if (!seen.Add((lo, mid, hi))) { duplicate++; continue; }
            kept.Add(a); kept.Add(b); kept.Add(c);
        }
        return kept.ToArray();
    }

    // ── 2. non-manifold edges and pinched vertices: split vertices, keep every face ───────────────

    /// <summary>
    /// Corners (a face's vertex slot) are united when the two faces share a MANIFOLD edge (exactly two faces) at that vertex. Each group of
    /// corners at one vertex is a fan; every fan gets its own vertex (the first keeps the original id). Edges with three or more faces and
    /// bow-tie vertices stop connecting faces, so the pieces they joined become separate, geometry untouched.
    /// </summary>
    private static (float[] P, int[] F, int Split) SplitNonManifold(float[] p, int[] f)
    {
        int nv = p.Length / 3, nc = f.Length;
        var edges = BuildEdges(f, nv);
        var parent = new int[nc]; for (int i = 0; i < nc; i++) parent[i] = i;
        void Join(int h1, int h2)
        {
            int v1 = f[h1], v2 = f[Next(h1)];
            Union(parent, h1, f[h2] == v1 ? h2 : Next(h2));            // the corner of h2's face sitting on v1
            Union(parent, Next(h1), f[h2] == v2 ? h2 : Next(h2));      // ... and on v2
        }
        for (int e = 0; e < edges.Edges; e++)
        {
            int c = edges.Count[e];
            if (c == 2) Join(edges.Order[edges.Start[e]], edges.Order[edges.Start[e] + 1]);
            else if (c > 2) foreach (var (a, b) in PairAtEdge(p, f, edges, e)) Join(a, b);   // only the faces that continue each other stay joined
        }
        var idOfRoot = new int[nc]; Array.Fill(idOfRoot, -1);
        var original = new bool[nv];
        var extra = new List<int>();          // original vertex of every new copy
        var extraRoot = new List<int>();      // fan (corner root) of every new copy
        var nf = new int[nc];
        for (int c = 0; c < nc; c++)
        {
            int r = Find(parent, c);
            if (idOfRoot[r] < 0)
            {
                int v = f[c];
                if (!original[v]) { original[v] = true; idOfRoot[r] = v; }
                else { idOfRoot[r] = nv + extra.Count; extra.Add(v); extraRoot.Add(r); }
            }
            nf[c] = idOfRoot[r];
        }
        if (extra.Count == 0) return (p, nf, 0);
        var np = new float[(nv + extra.Count) * 3];
        Array.Copy(p, np, p.Length);
        for (int i = 0; i < extra.Count; i++) Array.Copy(p, extra[i] * 3, np, (nv + i) * 3, 3);
        // A copy that sits exactly on its original is undone by anything that merges vertices by position (Blender, trimesh, a rigger's loader),
        // and the surface is non-manifold again. Move each copy a hair toward its own fan (0.01% of the model: 0.05 voxel at 512) so the repair
        // survives a weld. The first fan keeps the original position.
        double nudge = SplitNudge * LongestSide(p);
        var sum = new Dictionary<int, double[]>();
        for (int c = 0; c < nc; c++)
        {
            int r = Find(parent, c);
            if (idOfRoot[r] < nv) continue;
            if (!sum.TryGetValue(r, out var s)) sum[r] = s = new double[4];
            foreach (int other in new[] { f[Next(c)], f[Next(Next(c))] }) { s[0] += p[other * 3]; s[1] += p[other * 3 + 1]; s[2] += p[other * 3 + 2]; s[3]++; }
        }
        for (int i = 0; i < extra.Count; i++)
        {
            if (!sum.TryGetValue(extraRoot[i], out var s) || s[3] == 0) continue;
            int v = extra[i], id = nv + i;
            double dx = s[0] / s[3] - p[v * 3], dy = s[1] / s[3] - p[v * 3 + 1], dz = s[2] / s[3] - p[v * 3 + 2];
            double len = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            if (len == 0) continue;
            np[id * 3] += (float)(dx / len * nudge); np[id * 3 + 1] += (float)(dy / len * nudge); np[id * 3 + 2] += (float)(dz / len * nudge);
        }
        return (np, nf, extra.Count);
    }

    /// <summary>How far a split vertex copy moves toward its own fan, as a fraction of the longest bounding-box side.</summary>
    private const double SplitNudge = 1e-4;

    /// <summary>Faces on one edge that are less than this far apart (angle round the edge) lie on top of each other, not one after the other.</summary>
    private const double MinSheetAngle = Math.PI / 4;

    /// <summary>
    /// Which faces on a non-manifold edge continue each other. Each face is reduced to the direction it leaves the edge (in-plane, perpendicular to
    /// the edge); two faces form a sheet through the edge when those directions are far apart (180 degrees = a flat continuation). Pairs are chosen
    /// greedily by that angle, never below <see cref="MinSheetAngle"/>, so a fin or a duplicate layer stays unpaired and is separated from the sheet.
    /// </summary>
    private static List<(int, int)> PairAtEdge(float[] p, int[] f, EdgeTable t, int e)
    {
        int s = t.Start[e], c = t.Count[e];
        var dir = new double[c][];
        for (int i = 0; i < c; i++)
        {
            int h = t.Order[s + i], a = f[h], b = f[Next(h)], third = f[Next(Next(h))];
            double ex = p[b * 3] - p[a * 3], ey = p[b * 3 + 1] - p[a * 3 + 1], ez = p[b * 3 + 2] - p[a * 3 + 2];
            double el = Math.Sqrt(ex * ex + ey * ey + ez * ez);
            if (el == 0) continue;
            ex /= el; ey /= el; ez /= el;
            double wx = p[third * 3] - p[a * 3], wy = p[third * 3 + 1] - p[a * 3 + 1], wz = p[third * 3 + 2] - p[a * 3 + 2];
            double along = wx * ex + wy * ey + wz * ez;
            wx -= along * ex; wy -= along * ey; wz -= along * ez;
            double wl = Math.Sqrt(wx * wx + wy * wy + wz * wz);
            if (wl > 0) dir[i] = new[] { wx / wl, wy / wl, wz / wl };
        }
        var cand = new List<(double Angle, int I, int J)>();
        for (int i = 0; i < c; i++)
            for (int j = i + 1; j < c; j++)
            {
                if (dir[i] == null || dir[j] == null) continue;
                double dot = dir[i][0] * dir[j][0] + dir[i][1] * dir[j][1] + dir[i][2] * dir[j][2];
                double angle = Math.Acos(Math.Clamp(dot, -1, 1));
                if (angle >= MinSheetAngle) cand.Add((angle, i, j));
            }
        cand.Sort((x, y) => y.Angle.CompareTo(x.Angle));
        var used = new bool[c]; var pairs = new List<(int, int)>();
        foreach (var (_, i, j) in cand)
            if (!used[i] && !used[j]) { used[i] = used[j] = true; pairs.Add((t.Order[s + i], t.Order[s + j])); }
        return pairs;
    }

    /// <summary>
    /// What splitting could not separate (faces still sharing both vertex copies of a non-manifold edge): keep the best sheet through each such edge
    /// and drop the rest. Hole filling closes what this opens when it is small; a bigger gap stays a visible gap rather than a hidden defect.
    /// </summary>
    private static int[] DropExcessFaces(float[] p, int[] f, out int dropped)
    {
        dropped = 0;
        var edges = BuildEdges(f, p.Length / 3);
        var drop = new bool[f.Length / 3];
        for (int e = 0; e < edges.Edges; e++)
        {
            int c = edges.Count[e];
            if (c <= 2) continue;
            var pairs = PairAtEdge(p, f, edges, e);
            var keep = new HashSet<int>();
            if (pairs.Count > 0) { keep.Add(pairs[0].Item1 / 3); keep.Add(pairs[0].Item2 / 3); }
            else
            {
                var byArea = new List<int>();
                for (int i = 0; i < c; i++) byArea.Add(edges.Order[edges.Start[e] + i] / 3);
                byArea.Sort((x, y) => Area2(p, f[y * 3], f[y * 3 + 1], f[y * 3 + 2]).CompareTo(Area2(p, f[x * 3], f[x * 3 + 1], f[x * 3 + 2])));
                keep.Add(byArea[0]); keep.Add(byArea[1]);
            }
            for (int i = 0; i < c; i++) { int face = edges.Order[edges.Start[e] + i] / 3; if (!keep.Contains(face)) drop[face] = true; }
        }
        var kept = new List<int>(f.Length);
        for (int t = 0; t < drop.Length; t++)
        {
            if (drop[t]) { dropped++; continue; }
            kept.Add(f[t * 3]); kept.Add(f[t * 3 + 1]); kept.Add(f[t * 3 + 2]);
        }
        return kept.ToArray();
    }

    // ── 3. floating fragments ─────────────────────────────────────────────────────────────────────

    private static int[] RemoveFragments(float[] p, int[] f, double minArea, out int components, out int removedFaces)
    {
        components = 0; removedFaces = 0;
        int nf = f.Length / 3;
        if (nf == 0) return f;
        var edges = BuildEdges(f, p.Length / 3);
        var dsu = new int[nf]; for (int i = 0; i < nf; i++) dsu[i] = i;
        for (int e = 0; e < edges.Edges; e++)
            for (int i = 1; i < edges.Count[e]; i++) Union(dsu, edges.Order[edges.Start[e]] / 3, edges.Order[edges.Start[e] + i] / 3);
        var area = new double[nf];
        for (int t = 0; t < nf; t++) area[Find(dsu, t)] += Area2(p, f[t * 3], f[t * 3 + 1], f[t * 3 + 2]) / 2;
        var kept = new List<int>(f.Length);
        for (int t = 0; t < nf; t++)
        {
            int r = Find(dsu, t);
            if (area[r] < minArea) { removedFaces++; continue; }
            kept.Add(f[t * 3]); kept.Add(f[t * 3 + 1]); kept.Add(f[t * 3 + 2]);
        }
        for (int t = 0; t < nf; t++) if (Find(dsu, t) == t && area[t] < minArea) components++;
        return kept.ToArray();
    }

    private static (float[] P, int[] F) Compact(float[] p, int[] f)
    {
        int nv = p.Length / 3;
        var map = new int[nv]; Array.Fill(map, -1);
        int used = 0;
        for (int i = 0; i < f.Length; i++) if (map[f[i]] < 0) map[f[i]] = used++;
        if (used == nv) return (p, f);
        var np = new float[used * 3];
        for (int v = 0; v < nv; v++) if (map[v] >= 0) Array.Copy(p, v * 3, np, map[v] * 3, 3);
        var nf = new int[f.Length];
        for (int i = 0; i < f.Length; i++) nf[i] = map[f[i]];
        return (np, nf);
    }

    // ── 4. one winding, facing outward ────────────────────────────────────────────────────────────

    /// <summary>
    /// Propagates a winding over faces joined by manifold edges: an edge whose two faces traverse it the same way means the neighbour is wound the
    /// other way and gets flipped. Each connected piece is then turned outward by the sign of its volume (meaningful for a closed piece; an open
    /// sheet has no inside, and a piece whose volume is ~0 is left as it is).
    /// </summary>
    private static void Orient(float[] p, int[] f, bool outward, out int flipped, out int turned)
    {
        flipped = 0; turned = 0;
        int nf = f.Length / 3;
        if (nf == 0) return;
        var edges = BuildEdges(f, p.Length / 3);
        var flip = new bool[nf]; var seen = new bool[nf];
        var comp = new List<int>();
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < p.Length; i += 3) { cx += p[i]; cy += p[i + 1]; cz += p[i + 2]; }
        int nv = p.Length / 3; cx /= nv; cy /= nv; cz /= nv;
        // Unit face normals: the confidence of a shared edge is how parallel its two faces are (a smooth patch cannot be wound two ways; a fold can).
        var nrm = new float[nf * 3];
        for (int t = 0; t < nf; t++)
        {
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
            double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
            double x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx, l = Math.Sqrt(x * x + y * y + z * z);
            if (l > 0) { nrm[t * 3] = (float)(x / l); nrm[t * 3 + 1] = (float)(y / l); nrm[t * 3 + 2] = (float)(z / l); }
        }
        // Grow each piece from a seed across its MOST confident edge first (Prim's order), so the places where winding cannot be made consistent
        // land on folds and thin double layers instead of cutting through smooth surface.
        var queue = new PriorityQueue<int, float>();
        int Neighbour(int h)
        {
            int e = edges.EdgeOfHalf[h];
            if (edges.Count[e] != 2) return -1;
            int s = edges.Start[e];
            return edges.Order[s] == h ? edges.Order[s + 1] : edges.Order[s];
        }
        void Push(int a)
        {
            for (int k = 0; k < 3; k++)
            {
                int h = a * 3 + k, h2 = Neighbour(h);
                if (h2 < 0 || seen[h2 / 3]) continue;
                int b = h2 / 3;
                float cos = Math.Abs(nrm[a * 3] * nrm[b * 3] + nrm[a * 3 + 1] * nrm[b * 3 + 1] + nrm[a * 3 + 2] * nrm[b * 3 + 2]);
                queue.Enqueue(h, 1f - cos);
            }
        }
        for (int seed = 0; seed < nf; seed++)
        {
            if (seen[seed]) continue;
            comp.Clear(); seen[seed] = true; comp.Add(seed); Push(seed);
            while (queue.Count > 0)
            {
                int h = queue.Dequeue(), a = h / 3, h2 = Neighbour(h), b = h2 / 3;
                if (seen[b]) continue;
                bool conflict = f[h] == f[h2];                 // both run the same way: b must be wound opposite to a
                seen[b] = true; flip[b] = flip[a] ^ conflict; comp.Add(b); Push(b);
            }
            if (!outward) continue;
            double vol = 0;
            foreach (int t in comp)
            {
                int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
                if (flip[t]) (b, c) = (c, b);
                double ax = p[a * 3] - cx, ay = p[a * 3 + 1] - cy, az = p[a * 3 + 2] - cz;
                double bx = p[b * 3] - cx, by = p[b * 3 + 1] - cy, bz = p[b * 3 + 2] - cz;
                double qx = p[c * 3] - cx, qy = p[c * 3 + 1] - cy, qz = p[c * 3 + 2] - cz;
                vol += ax * (by * qz - bz * qy) + ay * (bz * qx - bx * qz) + az * (bx * qy - by * qx);
            }
            if (vol < 0) { foreach (int t in comp) flip[t] = !flip[t]; turned++; }
        }
        for (int t = 0; t < nf; t++)
            if (flip[t]) { (f[t * 3 + 1], f[t * 3 + 2]) = (f[t * 3 + 2], f[t * 3 + 1]); flipped++; }
    }

    // ── 5. small holes ────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Boundary loops no longer than <paramref name="maxPerimeter"/> are closed with a fan around the loop's centroid (a triangle for a
    /// three-edge loop). The loop is walked as an UNDIRECTED chain of boundary edges (a vertex with exactly two boundary edges continues it): the
    /// directions of boundary half-edges are only trustworthy where the winding is consistent, and a chain that follows them breaks at every
    /// leftover conflict. Each loop's fan is wound to agree with the majority of the faces it closes, and the caller re-runs
    /// <see cref="Orient"/> so the new faces join the one winding. A vertex with more than two boundary edges ends the walk (that loop is left alone).
    /// </summary>
    private static (float[] P, int[] F, int Filled, int Added) FillHoles(float[] p, int[] f, double maxPerimeter)
    {
        int nv = p.Length / 3, nh = f.Length;
        if (nh == 0) return (p, f, 0, 0);
        var edges = BuildEdges(f, nv);
        var first = new int[nv]; var second = new int[nv]; var degree = new int[nv];   // boundary half-edges at each vertex (a vertex is an end of an edge)
        Array.Fill(first, -1); Array.Fill(second, -1);
        for (int h = 0; h < nh; h++)
        {
            if (edges.Count[edges.EdgeOfHalf[h]] != 1) continue;
            foreach (int v in new[] { f[h], f[Next(h)] })
            {
                if (degree[v]++ == 0) first[v] = h; else if (degree[v] == 2) second[v] = h;
            }
        }
        var done = new bool[nh];
        var centres = new List<float>(); var newFaces = new List<int>();
        int filled = 0;
        var loopV = new List<int>(); var loopH = new List<int>();
        for (int h0 = 0; h0 < nh; h0++)
        {
            if (done[h0] || edges.Count[edges.EdgeOfHalf[h0]] != 1) continue;
            loopV.Clear(); loopH.Clear();
            int start = f[h0], cur = f[Next(h0)], h = h0; bool closed = false;
            loopV.Add(start); loopH.Add(h0); done[h0] = true;
            while (true)
            {
                if (cur == start) { closed = true; break; }
                if (degree[cur] != 2) break;
                int nextH = first[cur] == h ? second[cur] : first[cur];
                if (done[nextH]) break;
                done[nextH] = true; loopH.Add(nextH); loopV.Add(cur);
                cur = f[nextH] == cur ? f[Next(nextH)] : f[nextH];
                h = nextH;
            }
            int n = loopV.Count;
            if (!closed || n < 3) continue;
            double per = 0, mx = 0, my = 0, mz = 0;
            int agree = 0;
            for (int i = 0; i < n; i++)
            {
                int a = loopV[i], b = loopV[(i + 1) % n];
                double dx = p[a * 3] - p[b * 3], dy = p[a * 3 + 1] - p[b * 3 + 1], dz = p[a * 3 + 2] - p[b * 3 + 2];
                per += Math.Sqrt(dx * dx + dy * dy + dz * dz);
                mx += p[a * 3]; my += p[a * 3 + 1]; mz += p[a * 3 + 2];
                if (f[loopH[i]] == a) agree++;          // the face on this edge runs a->b like the walk: the closing face must run b->a
            }
            if (per > maxPerimeter) continue;
            if (agree * 2 < n) loopV.Reverse();          // majority of the surrounding faces run the other way: walk the loop backwards
            filled++;
            if (n == 3)
            {
                newFaces.Add(loopV[0]); newFaces.Add(loopV[2]); newFaces.Add(loopV[1]);   // reverse of the walk
                continue;
            }
            int centre = nv + centres.Count / 3;
            centres.Add((float)(mx / n)); centres.Add((float)(my / n)); centres.Add((float)(mz / n));
            for (int i = 0; i < n; i++) { newFaces.Add(loopV[(i + 1) % n]); newFaces.Add(loopV[i]); newFaces.Add(centre); }
        }
        if (newFaces.Count == 0) return (p, f, 0, 0);
        var np = new float[p.Length + centres.Count];
        Array.Copy(p, np, p.Length); centres.CopyTo(np, p.Length);
        var nf = new int[f.Length + newFaces.Count];
        Array.Copy(f, nf, f.Length); newFaces.CopyTo(nf, f.Length);
        return (np, nf, filled, newFaces.Count / 3);
    }
}
