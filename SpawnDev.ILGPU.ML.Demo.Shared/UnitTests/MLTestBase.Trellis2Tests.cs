using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// TRELLIS.2 image-to-3D: the host-side mesh stages (simplify, UV atlas, GLB) as pure tests on a torus whose surface is known, and the whole
/// photo-to-GLB run on CUDA when the checkpoints are on the machine. The networks themselves are gated against the real reference classes in
/// <c>tools/trellis2-harness</c> (goldens from CPU torch); those gates need the Python reference and are not part of this suite.
/// </summary>
public abstract partial class MLTestBase
{
    private static (float[] V, int[] F) TrellisTorus(int major, int minor)
    {
        var v = new float[major * minor * 3]; var f = new int[major * minor * 6];
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                double u = 2 * Math.PI * i / major, w = 2 * Math.PI * j / minor, r = 0.5 + 0.2 * Math.Cos(w);
                int o = (i * minor + j) * 3;
                v[o] = (float)(r * Math.Cos(u)); v[o + 1] = (float)(0.2 * Math.Sin(w)); v[o + 2] = (float)(r * Math.Sin(u));
            }
        int k = 0;
        for (int i = 0; i < major; i++)
            for (int j = 0; j < minor; j++)
            {
                int a = i * minor + j, b = i * minor + (j + 1) % minor, c = ((i + 1) % major) * minor + j, d = ((i + 1) % major) * minor + (j + 1) % minor;
                f[k++] = a; f[k++] = b; f[k++] = c; f[k++] = b; f[k++] = d; f[k++] = c;
            }
        return (v, f);
    }

    /// <summary>
    /// Simplification must reach the target while keeping the surface a closed 2-manifold of the same genus, on the torus, with the winding intact.
    /// A simplifier that merely deletes faces fails the closed-manifold and Euler checks; one that ignores the quadric fails the surface check.
    /// </summary>
    [TestMethod]
    public async Task Trellis2_MeshSimplifier_KeepsTheTorusClosedAndOnSurface() => await RunPureTest(async () =>
    {
        TorusSimplifyChecks(TrellisTorus(120, 60), 3000, parallel: false);
        // the slab-parallel path (frozen seam vertices, stitched, one global pass) must hold the same properties
        TorusSimplifyChecks(TrellisTorus(400, 200), 12000, parallel: true);
        await Task.CompletedTask;
    });

    private static void TorusSimplifyChecks((float[] V, int[] F) torus, int target, bool parallel)
    {
        var (v, f) = torus;
        var (sv, sf) = parallel ? MeshSimplifier.SimplifyParallel(v, f, target) : MeshSimplifier.Simplify(v, f, target);
        int nf = sf.Length / 3, nv = sv.Length / 3;
        if (nf > target + 8 || nf < target * 0.9) throw new Exception($"face count {nf} for target {target}");
        var edges = new Dictionary<long, int>();
        for (int t = 0; t < nf; t++)
            for (int k = 0; k < 3; k++)
            {
                int a = sf[t * 3 + k], b = sf[t * 3 + (k + 1) % 3];
                long key = (long)Math.Min(a, b) * nv + Math.Max(a, b);
                edges[key] = edges.GetValueOrDefault(key) + 1;
            }
        int bad = edges.Values.Count(c => c != 2);
        if (bad != 0) throw new Exception($"{bad} edges are not in exactly two faces");
        int euler = nv - edges.Count + nf;
        if (euler != 0) throw new Exception($"Euler characteristic {euler}, a torus has 0");
        double maxDev = 0;
        for (int i = 0; i < nv; i++)
        {
            double x = sv[i * 3], y = sv[i * 3 + 1], z = sv[i * 3 + 2];
            maxDev = Math.Max(maxDev, Math.Abs(Math.Sqrt(Math.Pow(Math.Sqrt(x * x + z * z) - 0.5, 2) + y * y) - 0.2));
        }
        if (maxDev > 0.01) throw new Exception($"vertices drifted {maxDev:F4} off the torus");
    }

    /// <summary>The atlas must keep every face, put every UV in [0,1], and never cover one texel with two triangles.</summary>
    [TestMethod]
    public async Task Trellis2_UvAtlas_NoOverlapAndAllFacesKept() => await RunPureTest(async () =>
    {
        var (v, f) = TrellisTorus(64, 32);
        var atlas = UvAtlas.Unwrap(v, f, new UvAtlas.Options { Resolution = 1024 });
        if (atlas.Faces.Length != f.Length) throw new Exception("faces lost");
        foreach (var c in atlas.Uvs) if (c < 0 || c > 1) throw new Exception($"uv {c} outside [0,1]");
        int R = 1024; var owner = new int[R * R]; int overlap = 0, covered = 0;
        int nf = atlas.Faces.Length / 3;
        for (int t = 0; t < nf; t++)
        {
            double ax = atlas.Uvs[atlas.Faces[t * 3] * 2] * R, ay = atlas.Uvs[atlas.Faces[t * 3] * 2 + 1] * R;
            double bx = atlas.Uvs[atlas.Faces[t * 3 + 1] * 2] * R, by = atlas.Uvs[atlas.Faces[t * 3 + 1] * 2 + 1] * R;
            double cx = atlas.Uvs[atlas.Faces[t * 3 + 2] * 2] * R, cy = atlas.Uvs[atlas.Faces[t * 3 + 2] * 2 + 1] * R;
            double det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            if (Math.Abs(det) < 1e-12) continue;
            for (int y = (int)Math.Max(Math.Min(ay, Math.Min(by, cy)), 0); y <= Math.Min(Math.Max(ay, Math.Max(by, cy)), R - 1); y++)
                for (int x = (int)Math.Max(Math.Min(ax, Math.Min(bx, cx)), 0); x <= Math.Min(Math.Max(ax, Math.Max(bx, cx)), R - 1); x++)
                {
                    double px = x + 0.5, py = y + 0.5;
                    double w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / det, w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / det, w2 = 1 - w0 - w1;
                    if (w0 <= 1e-6 || w1 <= 1e-6 || w2 <= 1e-6) continue;
                    if (owner[y * R + x] != 0 && owner[y * R + x] != t + 1) overlap++; else { owner[y * R + x] = t + 1; covered++; }
                }
        }
        if (overlap > covered / 500) throw new Exception($"{overlap} texels covered twice of {covered}");
        await Task.CompletedTask;
    });

    /// <summary>The GLB writer: a valid binary glTF 2.0 container with the PBR material, two embedded PNGs and the vertex streams.</summary>
    [TestMethod]
    public async Task Trellis2_PbrGlb_IsAWellFormedGltf() => await RunPureTest(async () =>
    {
        var (v, f) = TrellisTorus(16, 8);
        var atlas = UvAtlas.Unwrap(v, f, new UvAtlas.Options { Resolution = 256 });
        var normals = PbrGlbWriter.VertexNormals(atlas.Positions, atlas.Faces);
        var px = new uint[64 * 64]; Array.Fill(px, 0xFF336699u);
        var png = PngEncoder.EncodeRgba8(px, 64, 64);
        var glb = PbrGlbWriter.Write(atlas.Positions, normals, atlas.Uvs, atlas.Faces, png, png);
        if (BitConverter.ToUInt32(glb, 0) != 0x46546C67u) throw new Exception("not a glTF binary (magic)");
        if (BitConverter.ToUInt32(glb, 4) != 2) throw new Exception("version is not 2");
        if (BitConverter.ToUInt32(glb, 8) != (uint)glb.Length) throw new Exception("container length mismatch");
        int jsonLen = (int)BitConverter.ToUInt32(glb, 12);
        using var doc = JsonDocument.Parse(glb.AsMemory(20, jsonLen));
        var root = doc.RootElement;
        if (root.GetProperty("images").GetArrayLength() != 2) throw new Exception("expected two embedded images");
        var prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0];
        foreach (var attr in new[] { "POSITION", "NORMAL", "TEXCOORD_0" })
            if (!prim.GetProperty("attributes").TryGetProperty(attr, out _)) throw new Exception($"missing {attr}");
        var pbr = root.GetProperty("materials")[0].GetProperty("pbrMetallicRoughness");
        if (!pbr.TryGetProperty("baseColorTexture", out _) || !pbr.TryGetProperty("metallicRoughnessTexture", out _)) throw new Exception("material lacks a PBR texture");
        await Task.CompletedTask;
    });

    /// <summary>
    /// The whole thing on CUDA: a synthetic cut-out (RGBA, alpha = foreground) through DINOv3 features, the three flows, the decoders, the mesh
    /// extraction, simplify, atlas, bake and GLB. Skipped without the checkpoints (TRELLIS2_DIR, default the Hugging Face cache folder
    /// <c>spawndev-trellis2</c>) or off CUDA (the tensor-core kernels are CUDA inline PTX; other backends take the naive path and are not a target yet).
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "HeavyModel")]
    public async Task Trellis2_ImageToGlb_EndToEnd() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda) throw new UnsupportedTestException("Trellis2 end to end runs on CUDA");
        var dir = Environment.GetEnvironmentVariable("TRELLIS2_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "spawndev-trellis2");
        var paths = new Trellis2Paths
        {
            ModelDir = Path.Combine(dir, "TRELLIS.2-4B"),
            SparseStructureDecoder = Path.Combine(dir, "TRELLIS-image-large", "ckpts", "ss_dec_conv3d_16l8_fp16"),
            DinoWeights = Path.Combine(dir, "dinov3-timm-qkvb", "model.safetensors"),
        };
        if (!File.Exists(paths.DinoWeights) || !File.Exists(paths.SparseStructureDecoder + ".safetensors") || !Directory.Exists(Path.Combine(paths.ModelDir, "ckpts")))
            throw new UnsupportedTestException("TRELLIS.2 checkpoints not found (set TRELLIS2_DIR)");

        const int W = 384, H = 320;
        var rgba = new int[W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                double dx = (x - W / 2.0) / (W * 0.30), dy = (y - H / 2.0) / (H * 0.36), d = dx * dx + dy * dy;
                int a = d < 1 ? 255 : 0;
                int r = 120 + (int)(100 * Math.Sin(x * 0.05)), g = 80 + (int)(90 * Math.Cos(y * 0.07)), b = 200 - y / 3;
                rgba[y * W + x] = r | (g << 8) | (b << 16) | (a << 24);
            }
        var stages = new List<string>();
        using var gen = new Trellis2ImageTo3D(accelerator, paths) { StageDone = (n, s) => stages.Add($"{n} {s:F1}s") };
        var glb = gen.GenerateGlb(rgba, W, H, new Trellis2Options { TextureSize = 1024 });
        if (glb.Length < 100_000 || BitConverter.ToUInt32(glb, 0) != 0x46546C67u) throw new Exception($"implausible GLB ({glb.Length} bytes)");
        int jsonLen = (int)BitConverter.ToUInt32(glb, 12);
        using var doc = JsonDocument.Parse(glb.AsMemory(20, jsonLen));
        int triangles = doc.RootElement.GetProperty("accessors").EnumerateArray()
            .Where(a => a.GetProperty("type").GetString() == "SCALAR").Select(a => a.GetProperty("count").GetInt32()).Max() / 3;
        if (triangles < 1000) throw new Exception($"only {triangles} triangles");
        await Task.CompletedTask;
        return $"{glb.Length / 1024} KiB GLB, {triangles} triangles; " + string.Join(", ", stages.TakeLast(6));
    });
}
