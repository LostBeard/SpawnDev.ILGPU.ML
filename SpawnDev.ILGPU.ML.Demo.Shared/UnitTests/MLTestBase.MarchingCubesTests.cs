using SpawnDev.ILGPU.ML.Kernels;
using SpawnDev.ILGPU.ML.Formats;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for MarchingCubes isosurface extraction and 3D export pipeline.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task MarchingCubes_Sphere_ProducesVertices()
    {
        await Task.CompletedTask;
        int dim = 16;
        var field = new float[dim * dim * dim];

        // Sphere: distance from center minus radius
        float cx = dim / 2f, cy = dim / 2f, cz = dim / 2f, r = 5f;
        for (int z = 0; z < dim; z++)
            for (int y = 0; y < dim; y++)
                for (int x = 0; x < dim; x++)
                {
                    float dx = x - cx, dy = y - cy, dz = z - cz;
                    field[x + y * dim + z * dim * dim] = MathF.Sqrt(dx * dx + dy * dy + dz * dz) - r;
                }

        var mesh = MarchingCubesKernel.ExtractSurface(field, dim, dim, dim, isoLevel: 0f);

        if (mesh.VertexCount == 0)
            throw new Exception("Sphere should produce vertices");
        if (mesh.TriangleCount == 0)
            throw new Exception("Sphere should produce triangles");

        Console.WriteLine($"[MarchingCubes] Sphere 16^3: {mesh.VertexCount} vertices, {mesh.TriangleCount} triangles");
    }

    [TestMethod]
    public async Task MarchingCubes_Empty_NoOutput()
    {
        await Task.CompletedTask;
        // All positive (no surface crossing)
        int dim = 4;
        var field = new float[dim * dim * dim];
        Array.Fill(field, 1f);

        var mesh = MarchingCubesKernel.ExtractSurface(field, dim, dim, dim, isoLevel: 0f);

        if (mesh.VertexCount != 0)
            throw new Exception($"All-positive field should have 0 vertices, got {mesh.VertexCount}");

        Console.WriteLine("[MarchingCubes] All-positive field: 0 vertices, 0 triangles — correct");
    }

    [TestMethod]
    public async Task MarchingCubes_ToGLTF_ValidFile()
    {
        await Task.CompletedTask;
        // Generate a small sphere mesh
        int dim = 8;
        var field = new float[dim * dim * dim];
        float cx = 4, cy = 4, cz = 4, r = 2.5f;
        for (int z = 0; z < dim; z++)
            for (int y = 0; y < dim; y++)
                for (int x = 0; x < dim; x++)
                    field[x + y * dim + z * dim * dim] = MathF.Sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz)) - r;

        var mesh = MarchingCubesKernel.ExtractSurface(field, dim, dim, dim);
        if (mesh.VertexCount == 0)
            throw new Exception("No vertices from sphere");

        // Export to GLB
        var glb = GLTFExporter.ExportGLB(mesh);
        if (!GLTFExporter.IsValidGLB(glb))
            throw new Exception("Exported GLB has invalid magic");

        // Load back
        var loaded = GLTFLoader.LoadGLB(glb);
        if (loaded.VertexCount != mesh.VertexCount)
            throw new Exception($"GLB round-trip: {loaded.VertexCount} vs {mesh.VertexCount} vertices");

        Console.WriteLine($"[MarchingCubes] Sphere → GLB: {glb.Length} bytes, {mesh.VertexCount} vertices round-tripped");
    }

    [TestMethod]
    public async Task MarchingCubes_ToOBJ_ValidFile()
    {
        await Task.CompletedTask;
        int dim = 8;
        var field = new float[dim * dim * dim];
        float cx = 4, cy = 4, cz = 4, r = 2.5f;
        for (int z = 0; z < dim; z++)
            for (int y = 0; y < dim; y++)
                for (int x = 0; x < dim; x++)
                    field[x + y * dim + z * dim * dim] = MathF.Sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz)) - r;

        var mesh = MarchingCubesKernel.ExtractSurface(field, dim, dim, dim);

        // Export to OBJ
        var obj = OBJExporter.Export(mesh);
        var text = System.Text.Encoding.UTF8.GetString(obj);
        int vCount = text.Split('\n').Count(l => l.StartsWith("v "));
        int fCount = text.Split('\n').Count(l => l.StartsWith("f "));

        if (vCount != mesh.VertexCount)
            throw new Exception($"OBJ vertex count {vCount} vs mesh {mesh.VertexCount}");
        if (fCount != mesh.TriangleCount)
            throw new Exception($"OBJ face count {fCount} vs mesh {mesh.TriangleCount}");

        // Round-trip
        var loaded = OBJExporter.Load(obj);
        if (loaded.VertexCount != mesh.VertexCount)
            throw new Exception($"OBJ round-trip: {loaded.VertexCount} vs {mesh.VertexCount}");

        Console.WriteLine($"[MarchingCubes] Sphere → OBJ: {obj.Length} bytes, v={vCount} f={fCount} round-tripped");
    }
}
