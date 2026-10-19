using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.ML.Trellis2;

using FV = ArrayView1D<float, Stride1D.Dense>;
using IV = ArrayView1D<int, Stride1D.Dense>;

/// <summary>
/// The device-side construction of <see cref="MeshDistanceGrid"/>: the mesh never leaves the GPU. The host builder (<see cref="MeshDistanceGrid.Build"/>) loops over every triangle and every cell it touches in C#, which on the single-threaded
/// browser runtime is seconds of main-thread time for a 2M-triangle mesh and tens of MB of managed arrays. Here the same cell lists are made with three kernels (count the triangles per cell with atomics, scan, fill with atomics) and the
/// narrow-band voxels leave the device as a compacted coordinate buffer instead of a flag array that the host loops over. Within a cell the triangle order depends on the atomics' order; the nearest-point queries do not.
/// </summary>
public sealed partial class MeshDistanceGrid
{
    private static void ToGridImpl(Index1D i, FV world, FV grid, float scale, float offset) { grid[i] = world[i] * scale + offset; }

    private static int CellLo(float v, int dim) => Math.Min(Math.Max((int)MathF.Floor((v - Reach) / CellSize) + 1, 0), dim - 1);
    private static int CellHi(float v, int dim) => Math.Min(Math.Max((int)MathF.Floor((v + Reach) / CellSize) + 1, 0), dim - 1);

    private static void CountImpl(Index1D t, FV pos, IV faces, IV count, int dim)
    {
        int a = faces[t * 3] * 3, b = faces[t * 3 + 1] * 3, c = faces[t * 3 + 2] * 3;
        int x0 = CellLo(MathF.Min(pos[a], MathF.Min(pos[b], pos[c])), dim), x1 = CellHi(MathF.Max(pos[a], MathF.Max(pos[b], pos[c])), dim);
        int y0 = CellLo(MathF.Min(pos[a + 1], MathF.Min(pos[b + 1], pos[c + 1])), dim), y1 = CellHi(MathF.Max(pos[a + 1], MathF.Max(pos[b + 1], pos[c + 1])), dim);
        int z0 = CellLo(MathF.Min(pos[a + 2], MathF.Min(pos[b + 2], pos[c + 2])), dim), z1 = CellHi(MathF.Max(pos[a + 2], MathF.Max(pos[b + 2], pos[c + 2])), dim);
        for (int x = x0; x <= x1; x++)
            for (int y = y0; y <= y1; y++)
                for (int z = z0; z <= z1; z++)
                    Atomic.Add(ref count[(x * dim + y) * dim + z], 1);
    }

    private static void FillImpl(Index1D t, FV pos, IV faces, IV cellStart, IV fill, IV list, int dim)
    {
        int a = faces[t * 3] * 3, b = faces[t * 3 + 1] * 3, c = faces[t * 3 + 2] * 3;
        int x0 = CellLo(MathF.Min(pos[a], MathF.Min(pos[b], pos[c])), dim), x1 = CellHi(MathF.Max(pos[a], MathF.Max(pos[b], pos[c])), dim);
        int y0 = CellLo(MathF.Min(pos[a + 1], MathF.Min(pos[b + 1], pos[c + 1])), dim), y1 = CellHi(MathF.Max(pos[a + 1], MathF.Max(pos[b + 1], pos[c + 1])), dim);
        int z0 = CellLo(MathF.Min(pos[a + 2], MathF.Min(pos[b + 2], pos[c + 2])), dim), z1 = CellHi(MathF.Max(pos[a + 2], MathF.Max(pos[b + 2], pos[c + 2])), dim);
        for (int x = x0; x <= x1; x++)
            for (int y = y0; y <= y1; y++)
                for (int z = z0; z <= z1; z++)
                {
                    int cell = (x * dim + y) * dim + z;
                    int slot = Atomic.Add(ref fill[cell], 1);
                    list[cellStart[cell] + slot] = t;
                }
    }

    private static void CellFlagImpl(Index1D c, IV cellStart, IV flag) { flag[c] = cellStart[c + 1] > cellStart[c] ? 1 : 0; }
    private static void CellCompactImpl(Index1D c, IV flag, IV offsets, IV occupied) { if (flag[c] != 0) occupied[offsets[c]] = c; }

    /// <summary>
    /// Build the grid from a mesh that is ALREADY on the device (<paramref name="worldPositions"/> xyz for <paramref name="vertexCount"/> vertices, <paramref name="faces"/> for <paramref name="faceCount"/> triangles; vertices no face uses
    /// are harmless): nothing crosses to the host but the two counts that size the buffers. <c>g = p * scale + offset</c> as in <see cref="Build"/>.
    /// </summary>
    public static async Task<MeshDistanceGrid> BuildAsync(Accelerator acc, SparseKernels sk, FV worldPositions, int vertexCount, IV faces, int faceCount, int resolution, float scale, float offset)
    {
        var g = new MeshDistanceGrid(acc) { Resolution = resolution, Scale = scale, Offset = offset };
        int dim = resolution / CellSize + 3;
        g.Dim = dim;
        long cells = (long)dim * dim * dim;
        if (cells >= int.MaxValue) throw new ArgumentOutOfRangeException(nameof(resolution));
        var toGrid = acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, float, float>(ToGridImpl);
        var count = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, int>(CountImpl);
        var fillK = acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV, int>(FillImpl);
        var flagK = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV>(CellFlagImpl);
        var compactK = acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV>(CellCompactImpl);

        g.Positions = acc.Allocate1D<float>(Math.Max(1L, (long)vertexCount * 3));
        if (vertexCount > 0)
        {
            toGrid(vertexCount * 3, worldPositions, g.Positions.View, scale, offset);
        }
        g.Faces = acc.Allocate1D<int>(Math.Max(3L, (long)faceCount * 3));
        if (faceCount > 0) g.Faces.View.SubView(0, (long)faceCount * 3).CopyFrom(faces.SubView(0, (long)faceCount * 3));

        using var counts = acc.Allocate1D<int>(cells); counts.MemSetToZero();
        if (faceCount > 0) count(faceCount, g.Positions.View, g.Faces.View, counts.View, dim);
        g.CellStart = acc.Allocate1D<int>(cells + 1);
        int total = await sk.ExclusiveScanAsync(counts.View, (int)cells, g.CellStart.View);
        g.CellTris = acc.Allocate1D<int>(Math.Max(1, total));
        using (var fill = acc.Allocate1D<int>(cells))
        {
            fill.MemSetToZero();
            if (faceCount > 0) fillK(faceCount, g.Positions.View, g.Faces.View, g.CellStart.View, fill.View, g.CellTris.View, dim);
            using var flag = acc.Allocate1D<int>(cells);
            using var offsets = acc.Allocate1D<int>(cells + 1);
            flagK((int)cells, g.CellStart.View, flag.View);
            int occupied = await sk.ExclusiveScanAsync(flag.View, (int)cells, offsets.View);
            g.Occupied = acc.Allocate1D<int>(Math.Max(1, occupied));
            if (occupied > 0) compactK((int)cells, flag.View, offsets.View, g.Occupied.View);
            g.OccupiedCount = occupied;
            await acc.SynchronizeAsync();                                                // the temporaries are freed on return
        }
        return g;
    }

    private static void BandCompactImpl(Index1D i, IV flags, IV offsets, IV occupied, IV coords, int dim)
    {
        if (flags[i] == 0) return;
        int local = i & 511, cell = occupied[i >> 9];
        int cz = cell % dim, cy = (cell / dim) % dim, cx = cell / (dim * dim);
        int o = offsets[i] * 3;
        coords[o] = (cx - 1) * CellSize + (local >> 6); coords[o + 1] = (cy - 1) * CellSize + ((local >> 3) & 7); coords[o + 2] = (cz - 1) * CellSize + (local & 7);
    }

    /// <summary>
    /// <see cref="BandVoxelsAsync"/> without the read-back: the voxel coordinates (xyz per voxel) stay on the device, compacted by a scan of the flags. The caller disposes the buffer.
    /// </summary>
    public async Task<(MemoryBuffer1D<int, Stride1D.Dense> Coords, int Count)> BandVoxelsDeviceAsync(SparseKernels sk, float band, float halfWidth)
    {
        if (OccupiedCount == 0) return (_acc.Allocate1D<int>(3), 0);
        _bandVoxels ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, IV, IV, IV, IV, int, int, float, float, IV>(BandVoxelsImpl);
        var compact = _acc.LoadAutoGroupedStreamKernel<Index1D, IV, IV, IV, IV, int>(BandCompactImpl);
        long total = (long)OccupiedCount * 512;
        if (total >= int.MaxValue) throw new InvalidOperationException("too many candidate voxels");
        using var flags = _acc.Allocate1D<int>(total);
        _bandVoxels((int)total, Positions.View, Faces.View, CellStart.View, CellTris.View, Occupied.View, Dim, Resolution, band, halfWidth, flags.View);
        using var offsets = _acc.Allocate1D<int>(total + 1);
        int n = await sk.ExclusiveScanAsync(flags.View, (int)total, offsets.View);
        var coords = _acc.Allocate1D<int>(Math.Max(1L, (long)n * 3));
        if (n > 0) compact((int)total, flags.View, offsets.View, Occupied.View, coords.View, Dim);
        await _acc.SynchronizeAsync();
        return (coords, n);
    }

    private Action<Index1D, FV, FV, IV, IV, IV, int, float, float, FV>? _udfDevice;
    private static void UdfCornersImpl(Index1D i, FV pts, FV pos, IV faces, IV cellStart, IV cellTris, int dim, float cap, float shift, FV outD)
    {
        float d2 = NearestD2(pos, faces, cellStart, cellTris, dim, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], pts, -1);
        outD[i] = (d2 > Reach * Reach ? cap : MathF.Min(MathF.Sqrt(d2), cap)) - shift;
    }

    /// <summary><see cref="DistancesAsync"/> for points that are on the device: <paramref name="gridPoints"/> (grid units, xyz) in, distance minus <paramref name="shift"/> out, no host copy.</summary>
    public void DistancesDevice(FV gridPoints, int count, FV result, float shift, float cap = Reach)
    {
        if (count == 0) return;
        _udfDevice ??= _acc.LoadAutoGroupedStreamKernel<Index1D, FV, FV, IV, IV, IV, int, float, float, FV>(UdfCornersImpl);
        _udfDevice(count, gridPoints, Positions.View, Faces.View, CellStart.View, CellTris.View, Dim, cap, shift, result);
    }
}
