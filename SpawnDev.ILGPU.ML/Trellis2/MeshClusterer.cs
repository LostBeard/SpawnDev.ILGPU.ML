using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Adaptive octree vertex clustering on the device (quadric clustering, Lindstrom 2000, with the bottom-up merge test of octree
/// dual-contouring simplification): every vertex gets the quadric of its faces; level L groups vertices by voxel &gt;&gt; L, and a cell
/// merges into one vertex (its quadric minimiser) only when all of its children merged and the minimiser's RMS distance to the cell's
/// faces is within <see cref="Options.MaxError"/> voxels (optionally also an area-weighted normal cone, <see cref="Options.MinCone"/>).
/// Flat regions coarsen to large cells, detail stays at voxel size. O(n), no sequential topology work:
/// it takes the 3M-face dual-grid mesh to a size that the host's quadric edge collapse (<see cref="MeshSimplifier"/>) finishes quickly,
/// which matters most in the browser, where the host code runs interpreted (the edge collapse alone took 292 s for the full mesh there).
/// Float atomics are CAS loops on WebGPU; quadrics live in cell-local voxel units so float32 resolves sub-voxel error.
/// </summary>
public sealed class MeshClusterer
{
    private readonly Accelerator _acc;
    private readonly SparseKernels _sk;
    public MeshClusterer(Accelerator accelerator, SparseKernels sparse) { _acc = accelerator; _sk = sparse; }

    public sealed class Options
    {
        /// <summary>Coarsest level: cells of 2^MaxLevel voxels.</summary>
        public int MaxLevel { get; set; } = 5;
        /// <summary>
        /// Largest RMS distance (voxels) from a merged vertex to the planes of the faces it replaces. This also keeps the two sides of a thin
        /// sheet apart (fusing them costs half the thickness): MEASURED on the TRELLIS.2 example plant, 0.5 opened gaps in the thinnest leaves
        /// that 0.35 does not; crown 2.79M -> 420K faces, plant 807K -> 244K.
        /// </summary>
        public float MaxError { get; set; } = 0.35f;
        /// <summary>
        /// Smallest |sum of area-weighted normals| / area for a merge (1 = flat; 0 = off). Off by default: the dual-grid surface is a
        /// staircase whose normals disagree even where it is smooth (MEASURED median 0.66 in 2-voxel cells of the crown), so a cone test
        /// blocked most merges (2.79M -> 1.89M faces at 0.75) while the error test already guards thin sheets.
        /// </summary>
        public float MinCone { get; set; } = 0f;
    }

    private const int VS = 14;   // per vertex: quadric (aa ab ac ad bb bc bd cc cd dd), area-weighted normal, area
    private const int CS = 18;   // per cell: the same + position sum (cell-local) + vertex count

    private static void Voxel(FV verts, int v, int grid, out int qx, out int qy, out int qz, out float px, out float py, out float pz)
    {
        px = (verts[v * 3] + 0.5f) * grid; py = (verts[v * 3 + 1] + 0.5f) * grid; pz = (verts[v * 3 + 2] + 0.5f) * grid;
        qx = Math.Min(Math.Max((int)MathF.Floor(px), 0), grid - 1);
        qy = Math.Min(Math.Max((int)MathF.Floor(py), 0), grid - 1);
        qz = Math.Min(Math.Max((int)MathF.Floor(pz), 0), grid - 1);
    }

    // ---- level 0: per-vertex quadrics, each in the frame of its own voxel corner -------------------------------------------------
    private Action<Index1D, FV, IV, FV, int>? _vertexQuadric;
    private static void VertexQuadricImpl(Index1D t, FV verts, IV faces, FV vq, int grid)
    {
        int a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2];
        float ax = (verts[a * 3] + 0.5f) * grid, ay = (verts[a * 3 + 1] + 0.5f) * grid, az = (verts[a * 3 + 2] + 0.5f) * grid;
        float ux = (verts[b * 3] + 0.5f) * grid - ax, uy = (verts[b * 3 + 1] + 0.5f) * grid - ay, uz = (verts[b * 3 + 2] + 0.5f) * grid - az;
        float wx = (verts[c * 3] + 0.5f) * grid - ax, wy = (verts[c * 3 + 1] + 0.5f) * grid - ay, wz = (verts[c * 3 + 2] + 0.5f) * grid - az;
        float nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
        if (len <= 1e-12f) return;
        nx /= len; ny /= len; nz /= len;
        float w = len * 0.5f;
        for (int k = 0; k < 3; k++)
        {
            int v = faces[t * 3 + k];
            Voxel(verts, v, grid, out int qx, out int qy, out int qz, out _, out _, out _);
            float d = -(nx * (ax - qx) + ny * (ay - qy) + nz * (az - qz));
            int o = v * VS;
            Atomic.Add(ref vq[o], w * nx * nx); Atomic.Add(ref vq[o + 1], w * nx * ny); Atomic.Add(ref vq[o + 2], w * nx * nz); Atomic.Add(ref vq[o + 3], w * nx * d);
            Atomic.Add(ref vq[o + 4], w * ny * ny); Atomic.Add(ref vq[o + 5], w * ny * nz); Atomic.Add(ref vq[o + 6], w * ny * d);
            Atomic.Add(ref vq[o + 7], w * nz * nz); Atomic.Add(ref vq[o + 8], w * nz * d); Atomic.Add(ref vq[o + 9], w * d * d);
            Atomic.Add(ref vq[o + 10], w * nx); Atomic.Add(ref vq[o + 11], w * ny); Atomic.Add(ref vq[o + 12], w * nz); Atomic.Add(ref vq[o + 13], w);
        }
    }

    // ---- per level: cell ids through a device hash ---------------------------------------------------------------------------------
    private Action<Index1D, FV, IV, int, int>? _cellKey;
    private static void CellKeyImpl(Index1D v, FV verts, IV vkey, int grid, int level)
    {
        Voxel(verts, v, grid, out int qx, out int qy, out int qz, out _, out _, out _);
        int g = (grid + (1 << level) - 1) >> level;
        vkey[v] = ((qx >> level) * g + (qy >> level)) * g + (qz >> level);
    }

    private Action<Index1D, IV, IV, IV, IV, IV, int, int>? _insert;
    private static void InsertImpl(Index1D v, IV vkey, IV keys, IV vals, IV cellKey, IV counter, int mask, int shift)
    {
        int key = vkey[v];
        int slot = (int)(((uint)key * 2654435761u) >> shift);
        while (true)
        {
            int old = Atomic.CompareExchange(ref keys[slot], -1, key);
            if (old == -1) { int id = Atomic.Add(ref counter[0], 1); vals[slot] = id; cellKey[id] = key; return; }
            if (old == key) return;
            slot = (slot + 1) & mask;
        }
    }

    private Action<Index1D, IV, IV, IV, IV, int, int>? _lookup;
    private static void LookupImpl(Index1D v, IV vkey, IV keys, IV vals, IV cellOf, int mask, int shift)
    {
        int key = vkey[v];
        int slot = (int)(((uint)key * 2654435761u) >> shift);
        while (keys[slot] != key) slot = (slot + 1) & mask;   // inserted by the previous dispatch: always found
        cellOf[v] = vals[slot];
    }

    // ---- per level: vertex quadrics translated into their cell's frame and summed --------------------------------------------------
    private Action<Index1D, FV, FV, IV, FV, IV, IV, int, int>? _accumulate;
    private static void AccumulateImpl(Index1D v, FV verts, FV vq, IV cellOf, FV acc, IV childOk, IV alive, int grid, int level)
    {
        int c = cellOf[v];
        Voxel(verts, v, grid, out int qx, out int qy, out int qz, out float px, out float py, out float pz);
        int cx = (qx >> level) << level, cy = (qy >> level) << level, cz = (qz >> level) << level;
        float tx = qx - cx, ty = qy - cy, tz = qz - cz;           // vertex frame origin in the cell frame
        int o = v * VS, s = c * CS;
        float a00 = vq[o], a01 = vq[o + 1], a02 = vq[o + 2], b0 = vq[o + 3], a11 = vq[o + 4], a12 = vq[o + 5], b1 = vq[o + 6], a22 = vq[o + 7], b2 = vq[o + 8], cc = vq[o + 9];
        // x_vertex = x_cell - t:  b' = b - A t,  c' = c - 2 b.t + t.A.t
        float at0 = a00 * tx + a01 * ty + a02 * tz, at1 = a01 * tx + a11 * ty + a12 * tz, at2 = a02 * tx + a12 * ty + a22 * tz;
        float nb0 = b0 - at0, nb1 = b1 - at1, nb2 = b2 - at2;
        float nc = cc - 2f * (b0 * tx + b1 * ty + b2 * tz) + (tx * at0 + ty * at1 + tz * at2);
        Atomic.Add(ref acc[s], a00); Atomic.Add(ref acc[s + 1], a01); Atomic.Add(ref acc[s + 2], a02); Atomic.Add(ref acc[s + 3], nb0);
        Atomic.Add(ref acc[s + 4], a11); Atomic.Add(ref acc[s + 5], a12); Atomic.Add(ref acc[s + 6], nb1);
        Atomic.Add(ref acc[s + 7], a22); Atomic.Add(ref acc[s + 8], nb2); Atomic.Add(ref acc[s + 9], nc);
        Atomic.Add(ref acc[s + 10], vq[o + 10]); Atomic.Add(ref acc[s + 11], vq[o + 11]); Atomic.Add(ref acc[s + 12], vq[o + 12]); Atomic.Add(ref acc[s + 13], vq[o + 13]);
        Atomic.Add(ref acc[s + 14], px - cx); Atomic.Add(ref acc[s + 15], py - cy); Atomic.Add(ref acc[s + 16], pz - cz); Atomic.Add(ref acc[s + 17], 1f);
        if (alive[v] == 0) childOk[c] = 0;                        // a child that did not merge keeps its parent apart
    }

    private Action<Index1D, FV, IV, IV, IV, FV, int, int, float, float>? _solve;
    private static void SolveImpl(Index1D c, FV acc, IV cellKey, IV childOk, IV merged, FV cellPos, int grid, int level, float tau2, float minCone)
    {
        int s = c * CS;
        float area = acc[s + 13], cnt = acc[s + 17];
        float x = acc[s + 14] / cnt, y = acc[s + 15] / cnt, z = acc[s + 16] / cnt;
        int ok = childOk[c];
        if (ok != 0 && area > 0f && cnt > 1.5f)
        {
            float q0 = acc[s], q1 = acc[s + 1], q2 = acc[s + 2], q3 = acc[s + 3], q4 = acc[s + 4], q5 = acc[s + 5], q6 = acc[s + 6], q7 = acc[s + 7], q8 = acc[s + 8], q9 = acc[s + 9];
            // minimise x'Qx + lambda |x - mass point|^2: flat directions stay at the mass point
            float lam = 1e-3f * area;
            float m00 = q0 + lam, m01 = q1, m02 = q2, m11 = q4 + lam, m12 = q5, m22 = q7 + lam;
            float r0 = lam * x - q3, r1 = lam * y - q6, r2 = lam * z - q8;
            float c00 = m11 * m22 - m12 * m12, c01 = m02 * m12 - m01 * m22, c02 = m01 * m12 - m02 * m11;
            float det = m00 * c00 + m01 * c01 + m02 * c02;
            if (MathF.Abs(det) > 1e-30f)
            {
                float c11 = m00 * m22 - m02 * m02, c12 = m01 * m02 - m00 * m12, c22 = m00 * m11 - m01 * m01;
                x = (c00 * r0 + c01 * r1 + c02 * r2) / det;
                y = (c01 * r0 + c11 * r1 + c12 * r2) / det;
                z = (c02 * r0 + c12 * r1 + c22 * r2) / det;
            }
            float hi = (1 << level) + 0.5f;
            x = MathF.Min(MathF.Max(x, -0.5f), hi); y = MathF.Min(MathF.Max(y, -0.5f), hi); z = MathF.Min(MathF.Max(z, -0.5f), hi);
            float err = q0 * x * x + 2f * q1 * x * y + 2f * q2 * x * z + 2f * q3 * x + q4 * y * y + 2f * q5 * y * z + 2f * q6 * y + q7 * z * z + 2f * q8 * z + q9;
            float nx = acc[s + 10], ny = acc[s + 11], nz = acc[s + 12];
            float cone = MathF.Sqrt(nx * nx + ny * ny + nz * nz) / area;
            ok = err <= tau2 * area && cone >= minCone ? 1 : 0;
        }
        merged[c] = ok;
        int g = (grid + (1 << level) - 1) >> level, key = cellKey[c];
        int kx = key / (g * g), ky = (key / g) % g, kz = key % g;
        cellPos[c * 3] = ((kx << level) + x) / grid - 0.5f;
        cellPos[c * 3 + 1] = ((ky << level) + y) / grid - 0.5f;
        cellPos[c * 3 + 2] = ((kz << level) + z) / grid - 0.5f;
    }

    private Action<Index1D, IV, IV, IV, IV, int>? _assign;
    private static void AssignImpl(Index1D v, IV cellOf, IV merged, IV alive, IV rep, int repBase)
    {
        if (alive[v] == 0) return;
        int c = cellOf[v];
        if (merged[c] != 0) rep[v] = repBase + c; else alive[v] = 0;
    }

    // ---- output ---------------------------------------------------------------------------------------------------------------------
    private Action<Index1D, IV, IV, IV, IV, IV>? _remap;
    private static void RemapImpl(Index1D t, IV faces, IV rep, IV mapped, IV keep, IV used)
    {
        int r0 = rep[faces[t * 3]], r1 = rep[faces[t * 3 + 1]], r2 = rep[faces[t * 3 + 2]];
        mapped[t * 3] = r0; mapped[t * 3 + 1] = r1; mapped[t * 3 + 2] = r2;
        int k = r0 != r1 && r1 != r2 && r0 != r2 ? 1 : 0;
        keep[t] = k;
        if (k != 0) { used[r0] = 1; used[r1] = 1; used[r2] = 1; }
    }

    private Action<Index1D, FV, IV, IV, FV, int>? _positionsOut;
    private static void PositionsOutImpl(Index1D i, FV src, IV used, IV index, FV dst, int repBase)
    {
        int r = repBase + i;
        if (used[r] == 0) return;
        int o = index[r];
        dst[o * 3] = src[i * 3]; dst[o * 3 + 1] = src[i * 3 + 1]; dst[o * 3 + 2] = src[i * 3 + 2];
    }

    private Action<Index1D, IV, IV, IV, IV, IV>? _facesOut;
    private static void FacesOutImpl(Index1D t, IV mapped, IV keep, IV faceOffsets, IV index, IV dst)
    {
        if (keep[t] == 0) return;
        int o = faceOffsets[t];
        dst[o * 3] = index[mapped[t * 3]]; dst[o * 3 + 1] = index[mapped[t * 3 + 1]]; dst[o * 3 + 2] = index[mapped[t * 3 + 2]];
    }

    private Action<Index1D, IV, int>? _fill;
    private static void FillImpl(Index1D i, IV v, int value) => v[i] = value;
    private Action<Index1D, IV>? _iota;
    private static void IotaImpl(Index1D i, IV v) => v[i] = i;
    private Action<Index1D, FV>? _zero;
    private static void ZeroImpl(Index1D i, FV v) => v[i] = 0f;

    private void Load()
    {
        _vertexQuadric ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, FV, int>(VertexQuadricImpl);
        _cellKey ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, int, int>(CellKeyImpl);
        _insert ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV, int, int>(InsertImpl);
        _lookup ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int, int>(LookupImpl);
        _accumulate ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, FV, IV, IV, int, int>(AccumulateImpl);
        _solve ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, FV, int, int, float, float>(SolveImpl);
        _assign ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int>(AssignImpl);
        _remap ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV>(RemapImpl);
        _positionsOut ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, FV, int>(PositionsOutImpl);
        _facesOut ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, IV>(FacesOutImpl);
        _fill ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, int>(FillImpl);
        _iota ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV>(IotaImpl);
        _zero ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV>(ZeroImpl);
    }

    /// <summary>
    /// Cluster the device mesh (<paramref name="verts"/> [nv,3] in [-0.5,0.5]^3, <paramref name="faces"/> [nf,3]) on a
    /// <paramref name="grid"/>^3 voxel lattice and read the result back: compacted positions and faces (unreferenced vertices dropped,
    /// degenerate faces removed). <see cref="LastLevelCells"/> holds the cells / merged cells per level afterwards.
    /// </summary>
    public async Task<(float[] Positions, int[] Faces)> ClusterAsync(FV verts, int nv, IV faces, int nf, int grid, Options? options = null)
    {
        var o = options ?? new Options();
        Load();
        var cellBuffers = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        var cellCounts = new List<int>();
        try
        {
            using var vq = _acc.Allocate1D<float>((long)nv * VS);
            _zero((int)vq.Length, vq.View);
            _vertexQuadric(nf, verts, faces, vq.View, grid);

            using var rep = _acc.Allocate1D<int>(nv);
            using var alive = _acc.Allocate1D<int>(nv);
            using var vkey = _acc.Allocate1D<int>(nv);
            using var cellOf = _acc.Allocate1D<int>(nv);
            using var cellKey = _acc.Allocate1D<int>(nv);
            using var counter = _acc.Allocate1D<int>(1);
            _iota(nv, rep.View);
            _fill(nv, alive.View, 1);
            int cap = 64, bits = 6;
            while (cap < 2L * nv) { cap <<= 1; bits++; }
            using var keys = _acc.Allocate1D<int>(cap);
            using var vals = _acc.Allocate1D<int>(cap);
            var scratch = new List<IDisposable>();
            var mergedBuffers = new List<MemoryBuffer1D<int, Stride1D.Dense>>();
            int repBase = nv;
            LastLevelCells = new List<int>();
            try
            {
                for (int level = 1; level <= o.MaxLevel && (grid >> level) >= 1; level++)
                {
                    _fill(cap, keys.View, -1);
                    _fill(1, counter.View, 0);
                    _cellKey(nv, verts, vkey.View, grid, level);
                    _insert(nv, vkey.View, keys.View, vals.View, cellKey.View, counter.View, cap - 1, 32 - bits);
                    _lookup(nv, vkey.View, keys.View, vals.View, cellOf.View, cap - 1, 32 - bits);
                    int m = (await counter.View.CopyToHostAsync())[0];
                    var acc = _acc.Allocate1D<float>((long)m * CS); scratch.Add(acc);
                    var childOk = _acc.Allocate1D<int>(m); scratch.Add(childOk);
                    var merged = _acc.Allocate1D<int>(m); scratch.Add(merged);
                    var pos = _acc.Allocate1D<float>((long)m * 3); cellBuffers.Add(pos);
                    _zero((int)acc.Length, acc.View);
                    _fill(m, childOk.View, 1);
                    _accumulate(nv, verts, vq.View, cellOf.View, acc.View, childOk.View, alive.View, grid, level);
                    _solve(m, acc.View, cellKey.View, childOk.View, merged.View, pos.View, grid, level, o.MaxError * o.MaxError, o.MinCone);
                    _assign(nv, cellOf.View, merged.View, alive.View, rep.View, repBase);
                    cellCounts.Add(m);
                    LastLevelCells.Add(m);
                    repBase += m;
                    mergedBuffers.Add(merged);
                }
                if (Environment.GetEnvironmentVariable("TRELLIS2_MESHPROFILE") == "1")
                    for (int l = 0; l < mergedBuffers.Count; l++)
                        Console.WriteLine($"  [cluster] level {l + 1}: {cellCounts[l]} cells, {(await mergedBuffers[l].View.CopyToHostAsync()).Sum()} merged");

                int reps = repBase;
                using var mapped = _acc.Allocate1D<int>((long)nf * 3);
                using var keep = _acc.Allocate1D<int>(nf);
                using var used = _acc.Allocate1D<int>(reps);
                using var faceOffsets = _acc.Allocate1D<int>((long)nf + 1);
                using var index = _acc.Allocate1D<int>((long)reps + 1);
                _fill(reps, used.View, 0);
                _remap(nf, faces, rep.View, mapped.View, keep.View, used.View);
                int outFaces = await _sk.ExclusiveScanAsync(keep.View, nf, faceOffsets.View);
                int outVerts = await _sk.ExclusiveScanAsync(used.View, reps, index.View);
                using var posOut = _acc.Allocate1D<float>(Math.Max(3L, (long)outVerts * 3));
                using var facesOut = _acc.Allocate1D<int>(Math.Max(3L, (long)outFaces * 3));
                _positionsOut(nv, verts, used.View, index.View, posOut.View, 0);
                int b = nv;
                for (int l = 0; l < cellBuffers.Count; l++) { _positionsOut(cellCounts[l], cellBuffers[l].View, used.View, index.View, posOut.View, b); b += cellCounts[l]; }
                _facesOut(nf, mapped.View, keep.View, faceOffsets.View, index.View, facesOut.View);
                var p = await posOut.View.SubView(0, (long)outVerts * 3).CopyToHostAsync();
                var f = await facesOut.View.SubView(0, (long)outFaces * 3).CopyToHostAsync();
                return (p, f);
            }
            finally { foreach (var d in scratch) d.Dispose(); }   // after the awaited readbacks: nothing pending references them
        }
        finally { foreach (var d in cellBuffers) d.Dispose(); }
    }

    /// <summary>Cells per level of the last <see cref="ClusterAsync"/> (level 1 first).</summary>
    public List<int>? LastLevelCells { get; private set; }
}
