using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// Unsigned distance to a triangle mesh, on the device, for points close to the surface (the role of cubvh's <c>unsigned_distance</c> in the reference's remesh and texture bake).
/// The mesh lives in a lattice of <see cref="Resolution"/>^3 voxels (grid units). The lattice is cut into cells of <see cref="CellSize"/> voxels; every triangle is listed in every cell its bounding box,
/// grown by <see cref="Reach"/> voxels, touches. A query reads only its own cell's list, so it finds the nearest triangle whenever that lies within <see cref="Reach"/> voxels, and reports "none" beyond.
/// </summary>
public sealed partial class MeshDistanceGrid : IDisposable
{
    /// <summary>Edge of a cell in voxels.</summary>
    public const int CellSize = 8;
    /// <summary>How far, in voxels, a query sees. The narrow band of the remesh needs corner distances up to 1.87 + 0.87 voxels, so 3 is exact for everything the remesh reads.</summary>
    public const float Reach = 3f;

    private readonly Accelerator _acc;
    internal MemoryBuffer1D<float, Stride1D.Dense> Positions = null!;      // vertex xyz, grid units
    internal MemoryBuffer1D<int, Stride1D.Dense> Faces = null!;
    internal MemoryBuffer1D<int, Stride1D.Dense> CellStart = null!, CellTris = null!;
    /// <summary>Cells that list at least one triangle (their linear index).</summary>
    internal MemoryBuffer1D<int, Stride1D.Dense> Occupied = null!;
    /// <summary>Cells per axis.</summary>
    public int Dim { get; private set; }
    public int OccupiedCount { get; private set; }
    /// <summary>The lattice the mesh was binned into: voxels per axis.</summary>
    public int Resolution { get; private set; }
    /// <summary>World -> grid units: g = p * Scale + Offset (per axis).</summary>
    public float Scale { get; private set; }
    public float Offset { get; private set; }

    private MeshDistanceGrid(Accelerator acc) => _acc = acc;

    /// <param name="worldPositions">xyz of the mesh in world units</param>
    /// <param name="faces">triangle indices</param>
    /// <param name="resolution">lattice voxels per axis</param>
    /// <param name="scale">world -> grid: g = p * scale + offset</param>
    public static MeshDistanceGrid Build(Accelerator acc, float[] worldPositions, int[] faces, int resolution, float scale, float offset)
    {
        var g = new MeshDistanceGrid(acc) { Resolution = resolution, Scale = scale, Offset = offset };
        int nv = worldPositions.Length / 3, nf = faces.Length / 3;
        var pos = new float[worldPositions.Length];
        for (int i = 0; i < nv; i++)
            for (int a = 0; a < 3; a++) pos[i * 3 + a] = worldPositions[i * 3 + a] * scale + offset;
        int dim = resolution / CellSize + 3;
        g.Dim = dim;
        long cells = (long)dim * dim * dim;
        if (cells >= int.MaxValue) throw new ArgumentOutOfRangeException(nameof(resolution));
        var count = new int[cells + 1];
        // pass 1: how many cells each triangle touches; pass 2: fill
        void Range(int t, out int x0, out int x1, out int y0, out int y1, out int z0, out int z1)
        {
            int a = faces[t * 3], b = faces[t * 3 + 1], c = faces[t * 3 + 2];
            float mnx = MathF.Min(pos[a * 3], MathF.Min(pos[b * 3], pos[c * 3])), mxx = MathF.Max(pos[a * 3], MathF.Max(pos[b * 3], pos[c * 3]));
            float mny = MathF.Min(pos[a * 3 + 1], MathF.Min(pos[b * 3 + 1], pos[c * 3 + 1])), mxy = MathF.Max(pos[a * 3 + 1], MathF.Max(pos[b * 3 + 1], pos[c * 3 + 1]));
            float mnz = MathF.Min(pos[a * 3 + 2], MathF.Min(pos[b * 3 + 2], pos[c * 3 + 2])), mxz = MathF.Max(pos[a * 3 + 2], MathF.Max(pos[b * 3 + 2], pos[c * 3 + 2]));
            int Lo(float v) => Math.Clamp((int)MathF.Floor((v - Reach) / CellSize) + 1, 0, dim - 1);
            int Hi(float v) => Math.Clamp((int)MathF.Floor((v + Reach) / CellSize) + 1, 0, dim - 1);
            x0 = Lo(mnx); x1 = Hi(mxx); y0 = Lo(mny); y1 = Hi(mxy); z0 = Lo(mnz); z1 = Hi(mxz);
        }
        for (int t = 0; t < nf; t++)
        {
            Range(t, out int x0, out int x1, out int y0, out int y1, out int z0, out int z1);
            for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) for (int z = z0; z <= z1; z++) count[((long)x * dim + y) * dim + z + 1]++;
        }
        for (long c = 0; c < cells; c++) count[c + 1] += count[c];
        var fill = new int[cells];
        var list = new int[count[cells]];
        for (int t = 0; t < nf; t++)
        {
            Range(t, out int x0, out int x1, out int y0, out int y1, out int z0, out int z1);
            for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) for (int z = z0; z <= z1; z++)
            {
                long c = ((long)x * dim + y) * dim + z;
                list[count[c] + fill[c]++] = t;
            }
        }
        var occupied = new List<int>();
        for (long c = 0; c < cells; c++) if (count[c + 1] > count[c]) occupied.Add((int)c);
        g.OccupiedCount = occupied.Count;
        g.Positions = acc.Allocate1D(pos);
        g.Faces = acc.Allocate1D(faces.Length > 0 ? faces : new int[3]);
        g.CellStart = acc.Allocate1D(count);
        g.CellTris = acc.Allocate1D(list.Length > 0 ? list : new int[1]);
        g.Occupied = acc.Allocate1D(occupied.Count > 0 ? occupied.ToArray() : new int[1]);
        return g;
    }

    public void Dispose() { Positions?.Dispose(); Faces?.Dispose(); CellStart?.Dispose(); CellTris?.Dispose(); Occupied?.Dispose(); }

    // ---------------------------------------------------------------------------------------------------
    // nearest point on the triangles of a cell (Ericson, Real-Time Collision Detection 5.1.5)
    // ---------------------------------------------------------------------------------------------------

    /// <summary>Squared distance from (px,py,pz) to the nearest triangle listed in its cell (1e30 when the cell is empty); the nearest point goes to <paramref name="q"/>[qi*3..] when qi >= 0.</summary>
    internal static float NearestD2(FV pos, IV faces, IV cellStart, IV cellTris, int dim, float px, float py, float pz, FV q, int qi)
    {
        int cx = (int)MathF.Floor(px / CellSize) + 1, cy = (int)MathF.Floor(py / CellSize) + 1, cz = (int)MathF.Floor(pz / CellSize) + 1;
        if (cx < 0 || cx >= dim || cy < 0 || cy >= dim || cz < 0 || cz >= dim) return 1e30f;
        int cell = (cx * dim + cy) * dim + cz;
        int s = cellStart[cell], e = cellStart[cell + 1];
        float best = 1e30f, bx = 0f, by = 0f, bz = 0f;
        for (int k = s; k < e; k++)
        {
            int t = cellTris[k];
            int ia = faces[t * 3] * 3, ib = faces[t * 3 + 1] * 3, ic = faces[t * 3 + 2] * 3;
            float ax = pos[ia], ay = pos[ia + 1], az = pos[ia + 2];
            float bxx = pos[ib], byy = pos[ib + 1], bzz = pos[ib + 2];
            float cxx = pos[ic], cyy = pos[ic + 1], czz = pos[ic + 2];
            float abx = bxx - ax, aby = byy - ay, abz = bzz - az;
            float acx = cxx - ax, acy = cyy - ay, acz = czz - az;
            float apx = px - ax, apy = py - ay, apz = pz - az;
            float d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
            float rx, ry, rz;
            if (d1 <= 0f && d2 <= 0f) { rx = ax; ry = ay; rz = az; }
            else
            {
                float bpx = px - bxx, bpy = py - byy, bpz = pz - bzz;
                float d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
                if (d3 >= 0f && d4 <= d3) { rx = bxx; ry = byy; rz = bzz; }
                else
                {
                    float vc = d1 * d4 - d3 * d2;
                    if (vc <= 0f && d1 >= 0f && d3 <= 0f)
                    {
                        float v = d1 / (d1 - d3);
                        rx = ax + v * abx; ry = ay + v * aby; rz = az + v * abz;
                    }
                    else
                    {
                        float cpx = px - cxx, cpy = py - cyy, cpz = pz - czz;
                        float d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
                        if (d6 >= 0f && d5 <= d6) { rx = cxx; ry = cyy; rz = czz; }
                        else
                        {
                            float vb = d5 * d2 - d1 * d6;
                            if (vb <= 0f && d2 >= 0f && d6 <= 0f)
                            {
                                float w = d2 / (d2 - d6);
                                rx = ax + w * acx; ry = ay + w * acy; rz = az + w * acz;
                            }
                            else
                            {
                                float va = d3 * d6 - d5 * d4;
                                if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
                                {
                                    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                                    rx = bxx + w * (cxx - bxx); ry = byy + w * (cyy - byy); rz = bzz + w * (czz - bzz);
                                }
                                else
                                {
                                    float denom = 1f / (va + vb + vc);
                                    float v = vb * denom, w = vc * denom;
                                    rx = ax + abx * v + acx * w; ry = ay + aby * v + acy * w; rz = az + abz * v + acz * w;
                                }
                            }
                        }
                    }
                }
            }
            float dx = px - rx, dy = py - ry, dz = pz - rz;
            float dd = dx * dx + dy * dy + dz * dz;
            if (dd < best) { best = dd; bx = rx; by = ry; bz = rz; }
        }
        if (qi >= 0 && best < 1e29f) { q[qi * 3] = bx; q[qi * 3 + 1] = by; q[qi * 3 + 2] = bz; }
        return best;
    }

    // ---------------------------------------------------------------------------------------------------
    // kernels
    // ---------------------------------------------------------------------------------------------------

    private Action<Index1D, FV, FV, IV, IV, IV, int, float, FV>? _udf;
    private static void UdfImpl(Index1D i, FV pts, FV pos, IV faces, IV cellStart, IV cellTris, int dim, float cap, FV outD)
    {
        float d2 = NearestD2(pos, faces, cellStart, cellTris, dim, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], pts, -1);
        float d = d2 > Reach * Reach ? cap : MathF.Min(MathF.Sqrt(d2), cap);
        outD[i] = d;
    }

    /// <summary>Distance, in voxels, from each point (grid units, xyz) to the mesh, capped at <paramref name="cap"/> (<see cref="Reach"/> when the nearest triangle lies beyond it).</summary>
    public async Task<float[]> DistancesAsync(float[] gridPoints, float cap = Reach)
    {
        int n = gridPoints.Length / 3;
        if (n == 0) return Array.Empty<float>();
        _udf ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, IV, IV, int, float, FV>(UdfImpl);
        using var p = _acc.Allocate1D(gridPoints);
        using var d = _acc.Allocate1D<float>(n);
        _udf(n, p.View, Positions.View, Faces.View, CellStart.View, CellTris.View, Dim, cap, d.View);
        return await d.View.CopyToHostAsync();
    }

    private Action<Index1D, FV, IV, IV, IV, IV, int, int, float, float, IV>? _bandVoxels;
    private static void BandVoxelsImpl(Index1D i, FV pos, IV faces, IV cellStart, IV cellTris, IV occupied, int dim, int resolution, float band, float halfWidth, IV flags)
    {
        int local = i & 511, cell = occupied[i >> 9];
        int cz = cell % dim, cy = (cell / dim) % dim, cx = cell / (dim * dim);
        int x = (cx - 1) * CellSize + (local >> 6), y = (cy - 1) * CellSize + ((local >> 3) & 7), z = (cz - 1) * CellSize + (local & 7);
        flags[i] = 0;
        if (x < 0 || y < 0 || z < 0 || x >= resolution || y >= resolution || z >= resolution) return;
        float d2 = NearestD2(pos, faces, cellStart, cellTris, dim, x + 0.5f, y + 0.5f, z + 0.5f, pos, -1);
        if (d2 > Reach * Reach) return;
        float d = MathF.Sqrt(d2);
        if (MathF.Abs(d - band) < halfWidth) flags[i] = 1;
    }

    /// <summary>
    /// The voxels of the narrow band: voxel centres (x+0.5, y+0.5, z+0.5) whose distance to the mesh differs from <paramref name="band"/> by less than <paramref name="halfWidth"/> voxels
    /// (the reference keeps |udf - eps| &lt; 0.87 voxel at the finest level; the coarser levels it descends through only prune what cannot qualify). Returns xyz per voxel.
    /// </summary>
    public async Task<int[]> BandVoxelsAsync(float band, float halfWidth)
    {
        if (OccupiedCount == 0) return Array.Empty<int>();
        _bandVoxels ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV, int, int, float, float, IV>(BandVoxelsImpl);
        long total = (long)OccupiedCount * 512;
        if (total >= int.MaxValue) throw new InvalidOperationException("too many candidate voxels");
        using var flags = _acc.Allocate1D<int>(total);
        _bandVoxels((int)total, Positions.View, Faces.View, CellStart.View, CellTris.View, Occupied.View, Dim, Resolution, band, halfWidth, flags.View);
        var f = await flags.View.CopyToHostAsync();
        var cells = await Occupied.View.SubView(0, OccupiedCount).CopyToHostAsync();
        var coords = new List<int>(1 << 20);
        for (long i = 0; i < total; i++)
        {
            if (f[i] == 0) continue;
            int local = (int)(i & 511), cell = cells[(int)(i >> 9)];
            int dim = Dim, cz = cell % dim, cy = (cell / dim) % dim, cx = cell / (dim * dim);
            coords.Add((cx - 1) * CellSize + (local >> 6)); coords.Add((cy - 1) * CellSize + ((local >> 3) & 7)); coords.Add((cz - 1) * CellSize + (local & 7));
        }
        return coords.ToArray();
    }

    private Action<Index1D, FV, FV, IV, IV, IV, int, FV>? _project;
    private static void ProjectImpl(Index1D i, FV pts, FV pos, IV faces, IV cellStart, IV cellTris, int dim, FV outQ)
    {
        float d2 = NearestD2(pos, faces, cellStart, cellTris, dim, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], outQ, i);
        if (d2 > Reach * Reach) { outQ[i * 3] = pts[i * 3]; outQ[i * 3 + 1] = pts[i * 3 + 1]; outQ[i * 3 + 2] = pts[i * 3 + 2]; }
    }

    /// <summary>The nearest point of the mesh for each point (grid units, xyz); a point with no triangle within <see cref="Reach"/> keeps its place.</summary>
    public async Task<float[]> ProjectAsync(float[] gridPoints)
    {
        int n = gridPoints.Length / 3;
        if (n == 0) return Array.Empty<float>();
        _project ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, IV, IV, int, FV>(ProjectImpl);
        using var p = _acc.Allocate1D(gridPoints);
        using var q = _acc.Allocate1D<float>(gridPoints.Length);
        _project(n, p.View, Positions.View, Faces.View, CellStart.View, CellTris.View, Dim, q.View);
        return await q.View.CopyToHostAsync();
    }

    // ---- device-side projection for the texture bake ------------------------------------------------------------

    internal Action<Index1D, IV, FV, IV, FV, FV, IV, IV, IV, int, float, float, FV>? _bakeProject;
    internal static void BakeProjectImpl(Index1D texel, IV idMap, FV bary, IV faces, FV positions, FV pos, IV tfaces, IV cellStart, IV cellTris, int dim, float scale, float offset, FV outP)
    {
        int id = idMap[texel];
        if (id == 0) return;
        int tri = id - 1;
        float w0 = bary[texel * 2], w1 = bary[texel * 2 + 1], w2 = 1f - w0 - w1;
        int i0 = faces[tri * 3], i1 = faces[tri * 3 + 1], i2 = faces[tri * 3 + 2];
        float wx = positions[i0 * 3] * w0 + positions[i1 * 3] * w1 + positions[i2 * 3] * w2;
        float wy = positions[i0 * 3 + 1] * w0 + positions[i1 * 3 + 1] * w1 + positions[i2 * 3 + 1] * w2;
        float wz = positions[i0 * 3 + 2] * w0 + positions[i1 * 3 + 2] * w1 + positions[i2 * 3 + 2] * w2;
        outP[texel * 3] = wx; outP[texel * 3 + 1] = wy; outP[texel * 3 + 2] = wz;
        float d2 = NearestD2(pos, tfaces, cellStart, cellTris, dim, wx * scale + offset, wy * scale + offset, wz * scale + offset, outP, texel);
        if (d2 > Reach * Reach) { outP[texel * 3] = wx; outP[texel * 3 + 1] = wy; outP[texel * 3 + 2] = wz; return; }   // too far from the original surface to say anything: keep the mesh's own position
        outP[texel * 3] = (outP[texel * 3] - offset) / scale; outP[texel * 3 + 1] = (outP[texel * 3 + 1] - offset) / scale; outP[texel * 3 + 2] = (outP[texel * 3 + 2] - offset) / scale;
    }

    /// <summary>
    /// For every covered texel (<paramref name="idMap"/> != 0) write to <paramref name="outP"/> [texels, 3] the 3D position the texel's colour should be read at: the texel's position on the
    /// (remeshed, simplified) surface moved onto the nearest point of the ORIGINAL mesh this grid holds - the reference's <c>bvh.unsigned_distance(valid_pos)</c> before <c>grid_sample_3d</c>.
    /// </summary>
    public void ProjectTexels(int texels, IV idMap, FV bary, IV faces, FV positions, FV outP)
    {
        _bakeProject ??= _acc.LoadAutoGroupedStreamKernel<Index1D, IV, FV, IV, FV, FV, IV, IV, IV, int, float, float, FV>(BakeProjectImpl);
        _bakeProject(texels, idMap, bary, faces, positions, Positions.View, Faces.View, CellStart.View, CellTris.View, Dim, Scale, Offset, outP);
    }
}
