using System.Globalization;
using System.Text;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// Wavefront OBJ exporter and loader for triangle meshes.
/// The simplest 3D format — plain text, universally supported.
/// Used by Blender, Unity, game engines, 3D printing software.
/// </summary>
public static class OBJExporter
{
    /// <summary>
    /// Export MeshData to OBJ format.
    /// </summary>
    public static byte[] Export(MeshData mesh, string objectName = "mesh")
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# SpawnDev.ILGPU.ML — {mesh.VertexCount} vertices, {mesh.TriangleCount} triangles");
        sb.AppendLine($"o {objectName}");

        // Vertices
        for (int i = 0; i < mesh.VertexCount; i++)
        {
            float x = mesh.Vertices[i * 3];
            float y = mesh.Vertices[i * 3 + 1];
            float z = mesh.Vertices[i * 3 + 2];
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "v {0:F6} {1:F6} {2:F6}", x, y, z));
        }

        // Faces (OBJ uses 1-based indices)
        for (int i = 0; i < mesh.TriangleCount; i++)
        {
            int a = mesh.Indices[i * 3] + 1;
            int b = mesh.Indices[i * 3 + 1] + 1;
            int c = mesh.Indices[i * 3 + 2] + 1;
            sb.AppendLine($"f {a} {b} {c}");
        }

        return Encoding.UTF8.GetBytes(sb.ToString());
    }

    /// <summary>
    /// Load an OBJ file into MeshData.
    /// Handles v (vertex) and f (face) lines. Ignores normals, texcoords, materials.
    /// </summary>
    public static MeshData Load(byte[] objBytes)
    {
        var text = Encoding.UTF8.GetString(objBytes);
        var lines = text.Split('\n');

        var vertices = new List<float>();
        var indices = new List<int>();

        foreach (var rawLine in lines)
        {
            var line = rawLine.TrimEnd('\r').Trim();
            if (line.StartsWith("v "))
            {
                var parts = line.Substring(2).Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3)
                {
                    vertices.Add(float.Parse(parts[0], CultureInfo.InvariantCulture));
                    vertices.Add(float.Parse(parts[1], CultureInfo.InvariantCulture));
                    vertices.Add(float.Parse(parts[2], CultureInfo.InvariantCulture));
                }
            }
            else if (line.StartsWith("f "))
            {
                var parts = line.Substring(2).Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3)
                {
                    // OBJ faces can be "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                    int ParseIndex(string s) => int.Parse(s.Split('/')[0]) - 1; // 1-based → 0-based
                    indices.Add(ParseIndex(parts[0]));
                    indices.Add(ParseIndex(parts[1]));
                    indices.Add(ParseIndex(parts[2]));

                    // Handle quads by triangulating
                    if (parts.Length == 4)
                    {
                        indices.Add(ParseIndex(parts[0]));
                        indices.Add(ParseIndex(parts[2]));
                        indices.Add(ParseIndex(parts[3]));
                    }
                }
            }
        }

        int vertexCount = vertices.Count / 3;
        return new MeshData
        {
            Vertices = vertices.ToArray(),
            Indices = indices.ToArray(),
            VertexCount = vertexCount,
            TriangleCount = indices.Count / 3,
        };
    }
}
