using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Trellis2;

namespace Trellis2Harness;

/// <summary>
/// Texture bake against an analytic attribute volume: the voxels of the golden shape-decoder mesh carry their own world position
/// (colour = world xyz + 0.5, metallic 0.5, roughness 0.25, alpha 1). A covered texel must then read back its own 3D position, whatever
/// the atlas layout, so the rasteriser, barycentric interpolation, trilinear volume sampling and the 8-bit packing are all checked at once.
/// </summary>
internal static class BakeGate
{
    public static int Run(string? writeGlb)
    {
        string dg = Path.Combine(Gpu.WeightDir, "golden", "dualgrid"), dec = Path.Combine(Gpu.WeightDir, "golden", "decoders");
        if (!File.Exists(Path.Combine(dg, "faces.i32"))) { Console.WriteLine("run reference/golden_dualgrid.py and golden_decoders.py first"); return 2; }
        var verts = Golden.ReadF32(Path.Combine(dg, "vertices.f32"));
        var faces0 = Golden.ReadI32(Path.Combine(dg, "faces.i32"));
        var coords = Golden.ReadI32(Path.Combine(dec, "shape_out_coords.i32"));
        int n = coords.Length / 3, grid = 512;
        var attrs = new float[n * 6];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++) attrs[i * 6 + c] = (coords[i * 3 + c] + 0.5f) / grid;     // = world position of the voxel centre + 0.5
            attrs[i * 6 + 3] = 0.5f; attrs[i * 6 + 4] = 0.25f; attrs[i * 6 + 5] = 1f;
        }
        var (ctx, accel) = Gpu.Create();
        using var _c = ctx; using var _a = accel;
        var sk = new SparseKernels(accel);
        var baker = new TextureBaker(accel, sk);

        var remap = new int[verts.Length / 3]; Array.Fill(remap, -1);
        int used = 0; foreach (int v in faces0) if (remap[v] < 0) remap[v] = used++;
        var positions = new float[used * 3];
        for (int v = 0; v < remap.Length; v++) if (remap[v] >= 0) Array.Copy(verts, v * 3, positions, remap[v] * 3, 3);
        var faces = faces0.Select(v => remap[v]).ToArray();
        const int size = 1024;
        var atlas = UvAtlas.Unwrap(positions, faces, new UvAtlas.Options { Resolution = size });

        using var pos = accel.Allocate1D(atlas.Positions); using var uv = accel.Allocate1D(atlas.Uvs); using var fb = accel.Allocate1D(atlas.Faces);
        using var cb = accel.Allocate1D(coords); using var ab = accel.Allocate1D(attrs);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tex = baker.Bake(pos.View, uv.View, fb.View, atlas.Faces.Length / 3, cb.View, ab.View, n, grid, size, dilation: 8);
        Console.WriteLine($"bake {sw.Elapsed.TotalSeconds:F2}s, {tex.CoveredTexels} covered texels of {size * size}");

        // expected colour of every strictly-covered texel from a CPU rasterisation of the same UV triangles
        double maxErr = 0; long checkedTexels = 0, bad = 0, badMr = 0;
        var seen = new byte[size * size];
        int F = atlas.Faces.Length / 3;
        for (int t = 0; t < F; t++)
        {
            double[] ux = new double[3], uy = new double[3]; int[] vi = new int[3];
            for (int k = 0; k < 3; k++) { vi[k] = atlas.Faces[t * 3 + k]; ux[k] = atlas.Uvs[vi[k] * 2] * size; uy[k] = atlas.Uvs[vi[k] * 2 + 1] * size; }
            double det = (ux[1] - ux[0]) * (uy[2] - uy[0]) - (ux[2] - ux[0]) * (uy[1] - uy[0]);
            if (Math.Abs(det) < 1e-9) continue;
            int x0 = Math.Max(0, (int)Math.Floor(Math.Min(ux[0], Math.Min(ux[1], ux[2])))), x1 = Math.Min(size - 1, (int)Math.Ceiling(Math.Max(ux[0], Math.Max(ux[1], ux[2]))));
            int y0 = Math.Max(0, (int)Math.Floor(Math.Min(uy[0], Math.Min(uy[1], uy[2])))), y1 = Math.Min(size - 1, (int)Math.Ceiling(Math.Max(uy[0], Math.Max(uy[1], uy[2]))));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    double px = x + 0.5, py = y + 0.5;
                    double w0 = ((ux[1] - px) * (uy[2] - py) - (ux[2] - px) * (uy[1] - py)) / det;
                    double w1 = ((ux[2] - px) * (uy[0] - py) - (ux[0] - px) * (uy[2] - py)) / det;
                    double w2 = 1 - w0 - w1;
                    if (w0 < 0.02 || w1 < 0.02 || w2 < 0.02) continue;          // well inside: avoid shared-edge ties
                    if (seen[y * size + x] != 0) continue; seen[y * size + x] = 1;
                    uint p = tex.BaseColorAlpha[y * size + x], q = tex.MetallicRoughness[y * size + x];
                    double err = 0;
                    for (int c = 0; c < 3; c++)
                    {
                        double want = atlas.Positions[vi[0] * 3 + c] * w0 + atlas.Positions[vi[1] * 3 + c] * w1 + atlas.Positions[vi[2] * 3 + c] * w2 + 0.5;
                        double got = ((p >> (8 * c)) & 255) / 255.0;
                        err = Math.Max(err, Math.Abs(got - want));
                    }
                    checkedTexels++; maxErr = Math.Max(maxErr, err);
                    if (err > 0.006) bad++;
                    int rough = (int)((q >> 8) & 255), metal = (int)((q >> 16) & 255), alpha = (int)(p >> 24);
                    if (Math.Abs(rough - 64) > 1 || Math.Abs(metal - 128) > 1 || alpha != 255) badMr++;
                }
        }
        Console.WriteLine($"checked {checkedTexels} texels: max colour error {maxErr:F4} (tolerance 0.006), {bad} over, {badMr} wrong metallic/roughness/alpha");
        // dilation: every texel within 3 of a covered texel must be non-empty (alpha 255 baked into covered AND dilated texels)
        long holes = 0;
        for (int y = 3; y < size - 3; y++)
            for (int x = 3; x < size - 3; x++)
                if (tex.BaseColorAlpha[y * size + x] >> 24 == 0)
                    for (int dy = -1; dy <= 1 && true; dy++) for (int dx = -1; dx <= 1; dx++)
                        if (seen[(y + dy) * size + x + dx] != 0) { holes++; dy = 2; break; }
        Console.WriteLine($"{holes} empty texels next to covered ones");
        bool ok = checkedTexels > 1000 && bad == 0 && badMr == 0 && holes == 0;

        if (writeGlb != null)
        {
            var normals = PbrGlbWriter.VertexNormals(positions, faces);
            var split = new float[atlas.Positions.Length];
            for (int i = 0; i < atlas.SourceVertex.Length; i++) Array.Copy(normals, atlas.SourceVertex[i] * 3, split, i * 3, 3);
            var glb = PbrGlbWriter.Write(atlas.Positions, split, atlas.Uvs, atlas.Faces, PngEncoder.EncodeRgba8(tex.BaseColorAlpha, size, size), PngEncoder.EncodeRgba8(tex.MetallicRoughness, size, size));
            File.WriteAllBytes(writeGlb, glb);
            File.WriteAllBytes(Path.ChangeExtension(writeGlb, ".basecolor.png"), PngEncoder.EncodeRgba8(tex.BaseColorAlpha, size, size));
            Console.WriteLine($"wrote {writeGlb} ({glb.Length / 1024} KiB)");
        }
        Console.WriteLine(ok ? "BAKE PASS" : "BAKE FAIL");
        return ok ? 0 : 1;
    }
}
