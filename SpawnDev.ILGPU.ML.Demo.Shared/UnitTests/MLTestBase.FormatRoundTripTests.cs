using SpawnDev.ILGPU.ML.Formats;
using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Round-trip tests for 3D format export→import: SPZ, PLY, glTF, OBJ.
/// Verifies data survives the encode→decode cycle.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Format_PLY_RoundTrip_VerticesPreserved()
    {
        await Task.CompletedTask;

        // Create a simple cloud
        var cloud = CreateTestCloud(5);

        // Export to PLY
        var plyBytes = SPZExporter.ExportPLY(cloud);
        if (plyBytes.Length < 100)
            throw new Exception($"PLY export too small: {plyBytes.Length} bytes");

        // Import back
        var parsed = PLYParser.Parse(plyBytes);
        if (parsed.VertexCount != 5)
            throw new Exception($"PLY round-trip: vertexCount={parsed.VertexCount}, expected 5");

        // Verify positions survive
        if (parsed.Gaussians == null)
            throw new Exception("PLY round-trip: no Gaussian data extracted");

        for (int i = 0; i < 5; i++)
        {
            float dx = MathF.Abs(parsed.Gaussians.Positions[i * 3] - cloud.Positions[i * 3]);
            float dy = MathF.Abs(parsed.Gaussians.Positions[i * 3 + 1] - cloud.Positions[i * 3 + 1]);
            float dz = MathF.Abs(parsed.Gaussians.Positions[i * 3 + 2] - cloud.Positions[i * 3 + 2]);
            if (dx > 0.01f || dy > 0.01f || dz > 0.01f)
                throw new Exception($"PLY round-trip: position[{i}] error ({dx:F4},{dy:F4},{dz:F4})");
        }

        Console.WriteLine($"[Format] PLY round-trip: {plyBytes.Length} bytes, 5 vertices preserved");
    }

    [TestMethod]
    public async Task Format_OBJ_RoundTrip_MeshPreserved()
    {
        await Task.CompletedTask;

        // Create a simple triangle mesh
        var mesh = new MeshData
        {
            Vertices = new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 },
            Indices = new int[] { 0, 1, 2, 1, 3, 2 },
            VertexCount = 4,
            TriangleCount = 2,
        };

        var objBytes = OBJExporter.Export(mesh);
        var loaded = OBJExporter.Load(objBytes);

        if (loaded.VertexCount != 4)
            throw new Exception($"OBJ round-trip: vertexCount={loaded.VertexCount}, expected 4");
        if (loaded.TriangleCount != 2)
            throw new Exception($"OBJ round-trip: triangleCount={loaded.TriangleCount}, expected 2");

        // Verify vertex positions
        for (int i = 0; i < 12; i++)
        {
            if (MathF.Abs(loaded.Vertices[i] - mesh.Vertices[i]) > 0.001f)
                throw new Exception($"OBJ round-trip: vertex[{i}]={loaded.Vertices[i]:F4}, expected {mesh.Vertices[i]:F4}");
        }

        Console.WriteLine($"[Format] OBJ round-trip: {objBytes.Length} bytes, 4 vertices, 2 triangles preserved");
    }

    [TestMethod]
    public async Task Format_GLTF_RoundTrip_MeshPreserved()
    {
        await Task.CompletedTask;

        var mesh = new MeshData
        {
            Vertices = new float[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 },
            Indices = new int[] { 0, 1, 2 },
            VertexCount = 3,
            TriangleCount = 1,
        };

        var glbBytes = GLTFExporter.ExportGLB(mesh);

        // Verify magic
        if (!GLTFExporter.IsValidGLB(glbBytes))
            throw new Exception("GLB export: invalid magic bytes");

        // Load back
        var loaded = GLTFLoader.LoadGLB(glbBytes);
        if (loaded.VertexCount != 3)
            throw new Exception($"GLB round-trip: vertexCount={loaded.VertexCount}, expected 3");
        if (loaded.TriangleCount != 1)
            throw new Exception($"GLB round-trip: triangleCount={loaded.TriangleCount}, expected 1");

        Console.WriteLine($"[Format] glTF round-trip: {glbBytes.Length} bytes, 3 vertices, 1 triangle preserved");
    }

    [TestMethod]
    public async Task Format_SPZ_CompressionRatio()
    {
        await Task.CompletedTask;

        var cloud = CreateTestCloud(100);

        var spzBytes = SPZExporter.ExportSPZ(cloud);
        var plyBytes = SPZExporter.ExportPLY(cloud);

        float ratio = (float)plyBytes.Length / spzBytes.Length;

        Console.WriteLine($"[Format] SPZ compression: PLY={plyBytes.Length} bytes, SPZ={spzBytes.Length} bytes, ratio={ratio:F1}x");

        // SPZ should be significantly smaller than PLY
        if (ratio < 2f)
            throw new Exception($"SPZ compression ratio {ratio:F1}x too low — expected ≥ 2x");
    }

    private static GaussianCloud CreateTestCloud(int n)
    {
        var rng = new Random(42);
        var cloud = new GaussianCloud
        {
            NumPoints = n,
            Positions = new float[n * 3],
            Alphas = new float[n],
            Colors = new float[n * 3],
            Scales = new float[n * 3],
            Rotations = new float[n * 4],
        };

        for (int i = 0; i < n; i++)
        {
            cloud.Positions[i * 3] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Positions[i * 3 + 1] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Positions[i * 3 + 2] = (float)(rng.NextDouble() * 10 - 5);
            cloud.Alphas[i] = (float)(rng.NextDouble() * 0.8 + 0.1);
            cloud.Colors[i * 3] = (float)rng.NextDouble();
            cloud.Colors[i * 3 + 1] = (float)rng.NextDouble();
            cloud.Colors[i * 3 + 2] = (float)rng.NextDouble();
            cloud.Scales[i * 3] = -2f + (float)rng.NextDouble() * 2;
            cloud.Scales[i * 3 + 1] = -2f + (float)rng.NextDouble() * 2;
            cloud.Scales[i * 3 + 2] = -2f + (float)rng.NextDouble() * 2;
            cloud.Rotations[i * 4] = 1; // w (identity quaternion)
            cloud.Rotations[i * 4 + 1] = 0;
            cloud.Rotations[i * 4 + 2] = 0;
            cloud.Rotations[i * 4 + 3] = 0;
        }

        return cloud;
    }
}
