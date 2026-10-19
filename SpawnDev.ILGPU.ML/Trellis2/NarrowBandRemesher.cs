using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>
/// Rebuilds a generated mesh as a clean closed surface: the narrow-band dual-contouring remesh of CuMesh (<c>cumesh.remeshing.remesh_narrow_band_dc</c>), which the reference's
/// <c>to_glb</c> runs by default (<c>remesh=True</c>, band 1, no project-back) before it simplifies, unwraps and bakes.
/// <para>
/// The unsigned distance field of the input mesh is shifted by <c>band</c> voxels, so its zero level set is a shell one band outside every surface: pinholes and cracks narrower than the band
/// close, a thin plate (two sides, nearly touching) becomes one watertight skin, and the winding comes out consistent and outward. The shell is extracted by dual contouring on the voxels the
/// band crosses: a dual vertex is the mean of the zero crossings on its voxel's 12 edges, and every grid edge that crosses owns a quad of the 4 voxels around it.
/// </para>
/// <para>
/// The distance queries (the heavy part) run on the device through <see cref="MeshDistanceGrid"/>; the topology is host-side, O(voxels) with integer hashes.
/// </para>
/// </summary>
public static class NarrowBandRemesher
{
    /// <summary>What <see cref="RemeshAsync"/> returns.</summary>
    public sealed class Result : IDisposable
    {
        public float[] Positions = Array.Empty<float>();
        public int[] Faces = Array.Empty<int>();
        /// <summary>The distance grid of the INPUT mesh (kept for the texture bake, which projects every texel back onto it).</summary>
        public MeshDistanceGrid Source = null!;
        public int ActiveVoxels, GridVertices;
        public void Dispose() => Source?.Dispose();
    }

    private sealed class IntHash
    {
        private readonly int[] _keys; private readonly int[] _vals; private readonly int _mask, _shift;
        public IntHash(int expected)
        {
            int cap = 16, bits = 4; while (cap < expected * 2L) { cap <<= 1; bits++; }
            _keys = new int[cap]; _vals = new int[cap]; _mask = cap - 1; _shift = 32 - bits;
            Array.Fill(_keys, -1);
        }
        private int Slot(int k) => (int)(((uint)k * 2654435761u) >> _shift);
        /// <summary>The index stored for the key, inserting <paramref name="next"/> when it is new.</summary>
        public int GetOrAdd(int key, int next, out bool added)
        {
            int h = Slot(key);
            while (true)
            {
                int k = _keys[h];
                if (k == key) { added = false; return _vals[h]; }
                if (k < 0) { _keys[h] = key; _vals[h] = next; added = true; return next; }
                h = (h + 1) & _mask;
            }
        }
        public int Find(int key)
        {
            int h = Slot(key);
            while (true)
            {
                int k = _keys[h];
                if (k == key) return _vals[h];
                if (k < 0) return -1;
                h = (h + 1) & _mask;
            }
        }
    }

    private static readonly int[] SplitN1 = { 0, 1, 2, 0, 2, 3 }, SplitP1 = { 0, 2, 1, 0, 3, 2 }, SplitN2 = { 0, 1, 3, 3, 1, 2 }, SplitP2 = { 0, 3, 1, 3, 2, 1 };

    /// <param name="acc">device for the distance queries</param>
    /// <param name="positions">input mesh, xyz in world units (the volume is [-0.5, 0.5]^3)</param>
    /// <param name="faces">input triangles</param>
    /// <param name="resolution">the voxel grid of the volume (the reference passes grid_size.max())</param>
    /// <param name="band">shell offset in voxels (reference 1)</param>
    /// <param name="dropInnerSkins">
    /// The band around a closed surface has two skins: one outside it and one inside it, wound the other way. The inner one is hidden inside the model, costs face budget and confuses
    /// anything that samples the mesh (rigging); a piece whose signed volume is negative is dropped. (The reference keeps them; thin plates, whose two skins merge into one, are unaffected.)
    /// </param>
    public static async Task<Result> RemeshAsync(Accelerator acc, float[] positions, int[] faces, int resolution, float band = 1f, bool dropInnerSkins = true)
    {
        // the reference's domain: scale = (R + 3 band) / R times the unit box, R cells across it
        double scale = (resolution + 3.0 * band) / resolution;
        float toGrid = (float)(resolution / scale), offset = resolution * 0.5f;         // g = p * toGrid + R/2
        var source = MeshDistanceGrid.Build(acc, positions, faces, resolution, toGrid, offset);
        var result = new Result { Source = source };

        // ---- the narrow band: voxels whose centre lies within 0.87 voxel of the shell ----------------------------------------------------------
        var vox = await source.BandVoxelsAsync(band, 0.87f);
        int nv = vox.Length / 3;
        result.ActiveVoxels = nv;
        if (nv == 0) return result;

        // ---- unique corners of the active voxels, and the distance at each -----------------------------------------------------------------------
        int V1 = resolution + 1;
        var voxHash = new IntHash(nv);
        for (int i = 0; i < nv; i++) voxHash.GetOrAdd((vox[i * 3] * resolution + vox[i * 3 + 1]) * resolution + vox[i * 3 + 2], i, out _);
        var cornerHash = new IntHash(nv * 2);
        var corner = new int[nv * 8];                                    // per voxel: index of its 8 corners (bit 0 = +x, 1 = +y, 2 = +z)
        var cornerCoords = new List<int>(nv * 6);
        int nc = 0;
        for (int i = 0; i < nv; i++)
        {
            int x = vox[i * 3], y = vox[i * 3 + 1], z = vox[i * 3 + 2];
            for (int o = 0; o < 8; o++)
            {
                int cx = x + (o & 1), cy = y + ((o >> 1) & 1), cz = z + ((o >> 2) & 1);
                int idx = cornerHash.GetOrAdd((cx * V1 + cy) * V1 + cz, nc, out bool added);
                if (added) { cornerCoords.Add(cx); cornerCoords.Add(cy); cornerCoords.Add(cz); nc++; }
                corner[i * 8 + o] = idx;
            }
        }
        result.GridVertices = nc;
        var pts = new float[nc * 3];
        for (int i = 0; i < nc * 3; i++) pts[i] = cornerCoords[i];
        var dist = await source.DistancesAsync(pts);
        for (int i = 0; i < nc; i++) dist[i] -= band;                    // < 0 inside the band, >= 0 outside

        // ---- dual vertices and the edges that cross ----------------------------------------------------------------------------------------------
        var dual = new float[nv * 3];
        var dir = new sbyte[nv * 3];                                     // per voxel and axis: +1 / -1 when the edge (1,1) along that axis crosses, 0 otherwise
        for (int i = 0; i < nv; i++)
        {
            float sx = 0, sy = 0, sz = 0; int cnt = 0;
            int x = vox[i * 3], y = vox[i * 3 + 1], z = vox[i * 3 + 2];
            for (int axis = 0; axis < 3; axis++)
                for (int u = 0; u <= 1; u++)
                    for (int v = 0; v <= 1; v++)
                    {
                        // corner offsets of the edge's two ends
                        int o1, o2;
                        if (axis == 0) { o1 = (u << 1) | (v << 2); o2 = o1 | 1; }
                        else if (axis == 1) { o1 = u | (v << 2); o2 = o1 | 2; }
                        else { o1 = u | (v << 1); o2 = o1 | 4; }
                        float v1 = dist[corner[i * 8 + o1]], v2 = dist[corner[i * 8 + o2]];
                        bool cross = (v1 < 0 && v2 >= 0) || (v1 >= 0 && v2 < 0);
                        if (cross)
                        {
                            float t = -v1 / (v2 - v1);
                            if (axis == 0) { sx += x + t; sy += y + u; sz += z + v; }
                            else if (axis == 1) { sx += x + u; sy += y + t; sz += z + v; }
                            else { sx += x + u; sy += y + v; sz += z + t; }
                            cnt++;
                        }
                        if (u == 1 && v == 1) dir[i * 3 + axis] = (sbyte)(v1 < 0 && v2 >= 0 ? 1 : (v1 >= 0 && v2 < 0 ? -1 : 0));
                    }
            if (cnt > 0) { dual[i * 3] = sx / cnt; dual[i * 3 + 1] = sy / cnt; dual[i * 3 + 2] = sz / cnt; }
            else { dual[i * 3] = x + 0.5f; dual[i * 3 + 1] = y + 0.5f; dual[i * 3 + 2] = z + 0.5f; }
        }

        // ---- quads -> triangles ---------------------------------------------------------------------------------------------------------------------
        var quadIdx = new List<int>(nv * 4); var quadDir = new List<sbyte>(nv);
        for (int i = 0; i < nv; i++)
        {
            int x = vox[i * 3], y = vox[i * 3 + 1], z = vox[i * 3 + 2];
            for (int axis = 0; axis < 3; axis++)
            {
                if (dir[i * 3 + axis] == 0) continue;
                // the four voxels around the edge, in the reference's order
                int d1x = 0, d1y = 0, d1z = 0, d2x = 0, d2y = 0, d2z = 0, d3x = 0, d3y = 0, d3z = 0;
                if (axis == 0) { d1z = 1; d2y = 1; d2z = 1; d3y = 1; }
                else if (axis == 1) { d1x = 1; d2x = 1; d2z = 1; d3z = 1; }
                else { d1y = 1; d2x = 1; d2y = 1; d3x = 1; }
                int q1 = Find(x + d1x, y + d1y, z + d1z), q2 = Find(x + d2x, y + d2y, z + d2z), q3 = Find(x + d3x, y + d3y, z + d3z);
                if (q1 < 0 || q2 < 0 || q3 < 0) continue;
                quadIdx.Add(i); quadIdx.Add(q1); quadIdx.Add(q2); quadIdx.Add(q3); quadDir.Add(dir[i * 3 + axis]);
            }
        }
        int Find(int x, int y, int z) => (x < 0 || y < 0 || z < 0 || x >= resolution || y >= resolution || z >= resolution) ? -1 : voxHash.Find((x * resolution + y) * resolution + z);

        // vertices that some quad uses, in order of first use is not required: keep voxel order like the reference's unique()
        int nq = quadDir.Count;
        var used = new bool[nv];
        for (int k = 0; k < nq * 4; k++) used[quadIdx[k]] = true;
        var map = new int[nv]; int nOut = 0;
        for (int i = 0; i < nv; i++) if (used[i]) map[i] = nOut++;
        var outPos = new float[nOut * 3];
        for (int i = 0; i < nv; i++)
        {
            if (!used[i]) continue;
            int o = map[i] * 3;
            // grid -> world: p = (g / R - 0.5) * scale
            outPos[o] = (float)((dual[i * 3] / resolution - 0.5) * scale);
            outPos[o + 1] = (float)((dual[i * 3 + 1] / resolution - 0.5) * scale);
            outPos[o + 2] = (float)((dual[i * 3 + 2] / resolution - 0.5) * scale);
        }
        var tris = new int[nq * 6];
        var q = new int[4];
        for (int k = 0; k < nq; k++)
        {
            for (int c = 0; c < 4; c++) q[c] = map[quadIdx[k * 4 + c]];
            bool plus = quadDir[k] == 1;
            var s1 = plus ? SplitP1 : SplitN1; var s2 = plus ? SplitP2 : SplitN2;
            double a0 = Align(outPos, q, s1), a1 = Align(outPos, q, s2);
            var s = a0 > a1 ? s1 : s2;
            for (int c = 0; c < 6; c++) tris[k * 6 + c] = q[s[c]];
        }
        if (dropInnerSkins) (outPos, tris) = DropInwardPieces(outPos, tris);
        result.Positions = outPos; result.Faces = tris;
        return result;
    }

    /// <summary>Remove every connected piece (connected through shared vertices) whose signed volume is negative, i.e. that faces into the model; compacts the vertices.</summary>
    private static (float[] P, int[] F) DropInwardPieces(float[] p, int[] f)
    {
        int nv = p.Length / 3, nf = f.Length / 3;
        var parent = new int[nv]; for (int i = 0; i < nv; i++) parent[i] = i;
        int Root(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
        for (int t = 0; t < nf; t++)
        {
            int a = Root(f[t * 3]), b = Root(f[t * 3 + 1]), c = Root(f[t * 3 + 2]);
            if (a != b) parent[b] = a;
            b = Root(f[t * 3 + 1]); c = Root(f[t * 3 + 2]);
            if (a != c) parent[c] = Root(a);
        }
        var volume = new double[nv];
        for (int t = 0; t < nf; t++)
        {
            int a = f[t * 3] * 3, b = f[t * 3 + 1] * 3, c = f[t * 3 + 2] * 3;
            volume[Root(f[t * 3])] += ((double)p[a] * ((double)p[b + 1] * p[c + 2] - (double)p[b + 2] * p[c + 1])
                                     - (double)p[a + 1] * ((double)p[b] * p[c + 2] - (double)p[b + 2] * p[c])
                                     + (double)p[a + 2] * ((double)p[b] * p[c + 1] - (double)p[b + 1] * p[c])) / 6.0;
        }
        bool anyInward = false;
        for (int i = 0; i < nv; i++) if (parent[i] == i && volume[i] < 0) { anyInward = true; break; }
        if (!anyInward) return (p, f);
        var keepVertex = new int[nv]; Array.Fill(keepVertex, -1);
        var faces = new List<int>(f.Length);
        for (int t = 0; t < nf; t++)
            if (volume[Root(f[t * 3])] >= 0) { faces.Add(f[t * 3]); faces.Add(f[t * 3 + 1]); faces.Add(f[t * 3 + 2]); }
        int n = 0;
        foreach (int v in faces) if (keepVertex[v] < 0) keepVertex[v] = n++;
        var np = new float[n * 3];
        for (int v = 0; v < nv; v++) if (keepVertex[v] >= 0) { np[keepVertex[v] * 3] = p[v * 3]; np[keepVertex[v] * 3 + 1] = p[v * 3 + 1]; np[keepVertex[v] * 3 + 2] = p[v * 3 + 2]; }
        var nf2 = new int[faces.Count];
        for (int i = 0; i < nf2.Length; i++) nf2[i] = keepVertex[faces[i]];
        return (np, nf2);
    }

    /// <summary>|n0 . n1| exactly as the reference computes it: n0 = (t1 - t0) x (t2 - t0), n1 = (t2 - t1) x (t3 - t1) on the first four vertices of the attempted triangle list.</summary>
    private static double Align(float[] p, int[] q, int[] split)
    {
        int a = q[split[0]] * 3, b = q[split[1]] * 3, c = q[split[2]] * 3, d = q[split[3]] * 3;
        double ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
        double vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
        double n0x = uy * vz - uz * vy, n0y = uz * vx - ux * vz, n0z = ux * vy - uy * vx;
        double wx = p[c] - p[b], wy = p[c + 1] - p[b + 1], wz = p[c + 2] - p[b + 2];
        double zx = p[d] - p[b], zy = p[d + 1] - p[b + 1], zz = p[d + 2] - p[b + 2];
        double n1x = wy * zz - wz * zy, n1y = wz * zx - wx * zz, n1z = wx * zy - wy * zx;
        return Math.Abs(n0x * n1x + n0y * n1y + n0z * n1z);
    }
}
