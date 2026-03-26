using System.Text;
using System.Text.Json;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// Minimal glTF 2.0 binary (.glb) exporter for triangle meshes.
/// Exports MeshData from MarchingCubes or any triangle list as a
/// self-contained .glb file viewable in any 3D viewer.
///
/// glTF spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
/// </summary>
public static class GLTFExporter
{
    private const uint GLTF_MAGIC = 0x46546C67; // "glTF"
    private const uint GLTF_VERSION = 2;
    private const uint JSON_CHUNK_TYPE = 0x4E4F534A; // "JSON"
    private const uint BIN_CHUNK_TYPE = 0x004E4942; // "BIN\0"

    /// <summary>
    /// Export a triangle mesh to glTF binary (.glb) format.
    /// Returns the complete .glb file as byte array.
    /// </summary>
    public static byte[] ExportGLB(MeshData mesh, float[]? vertexColors = null)
    {
        int vertexCount = mesh.VertexCount;
        int triangleCount = mesh.TriangleCount;

        // Build binary buffer: [positions][indices][colors?]
        int positionBytes = vertexCount * 3 * 4; // float32 × 3 × N
        int indexBytes = triangleCount * 3 * 4;    // uint32 × 3 × N
        int colorBytes = vertexColors != null ? vertexCount * 3 * 4 : 0;
        int totalBinBytes = positionBytes + indexBytes + colorBytes;
        // Align to 4 bytes
        int paddedBinBytes = (totalBinBytes + 3) & ~3;

        var binData = new byte[paddedBinBytes];
        int offset = 0;

        // Positions
        Buffer.BlockCopy(mesh.Vertices, 0, binData, offset, positionBytes);
        int positionOffset = offset;
        offset += positionBytes;

        // Indices
        int indexOffset = offset;
        Buffer.BlockCopy(mesh.Indices, 0, binData, offset, indexBytes);
        offset += indexBytes;

        // Vertex colors (optional)
        int colorOffset = offset;
        if (vertexColors != null)
        {
            Buffer.BlockCopy(vertexColors, 0, binData, offset, colorBytes);
            offset += colorBytes;
        }

        // Compute bounding box for positions
        float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
        float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
        for (int i = 0; i < vertexCount; i++)
        {
            float x = mesh.Vertices[i * 3], y = mesh.Vertices[i * 3 + 1], z = mesh.Vertices[i * 3 + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }

        // Build JSON chunk
        var accessors = new List<object>();
        var bufferViews = new List<object>();
        var attributes = new Dictionary<string, int>();

        // BufferView 0: positions
        bufferViews.Add(new { buffer = 0, byteOffset = positionOffset, byteLength = positionBytes, target = 34962 });
        // Accessor 0: positions
        accessors.Add(new
        {
            bufferView = 0, componentType = 5126, count = vertexCount, type = "VEC3",
            max = new[] { maxX, maxY, maxZ }, min = new[] { minX, minY, minZ }
        });
        attributes["POSITION"] = 0;

        // BufferView 1: indices
        bufferViews.Add(new { buffer = 0, byteOffset = indexOffset, byteLength = indexBytes, target = 34963 });
        // Accessor 1: indices
        accessors.Add(new { bufferView = 1, componentType = 5125, count = triangleCount * 3, type = "SCALAR" });

        int colorAccessorIdx = -1;
        if (vertexColors != null)
        {
            // BufferView 2: colors
            bufferViews.Add(new { buffer = 0, byteOffset = colorOffset, byteLength = colorBytes, target = 34962 });
            // Accessor 2: colors
            colorAccessorIdx = accessors.Count;
            accessors.Add(new { bufferView = 2, componentType = 5126, count = vertexCount, type = "VEC3" });
            attributes["COLOR_0"] = colorAccessorIdx;
        }

        var gltf = new
        {
            asset = new { version = "2.0", generator = "SpawnDev.ILGPU.ML" },
            scene = 0,
            scenes = new[] { new { nodes = new[] { 0 } } },
            nodes = new[] { new { mesh = 0 } },
            meshes = new[]
            {
                new
                {
                    primitives = new[]
                    {
                        new { attributes, indices = 1, mode = 4 } // TRIANGLES
                    }
                }
            },
            accessors,
            bufferViews,
            buffers = new[] { new { byteLength = paddedBinBytes } },
        };

        var jsonStr = JsonSerializer.Serialize(gltf);
        var jsonBytes = Encoding.UTF8.GetBytes(jsonStr);
        // Pad JSON to 4-byte alignment with spaces
        int paddedJsonLen = (jsonBytes.Length + 3) & ~3;
        var paddedJson = new byte[paddedJsonLen];
        Array.Copy(jsonBytes, paddedJson, jsonBytes.Length);
        for (int i = jsonBytes.Length; i < paddedJsonLen; i++)
            paddedJson[i] = 0x20; // space

        // Build GLB file
        int totalLength = 12 + 8 + paddedJsonLen + 8 + paddedBinBytes; // header + json chunk + bin chunk
        var glb = new byte[totalLength];
        int pos = 0;

        // GLB header (12 bytes)
        WriteUInt32(glb, ref pos, GLTF_MAGIC);
        WriteUInt32(glb, ref pos, GLTF_VERSION);
        WriteUInt32(glb, ref pos, (uint)totalLength);

        // JSON chunk
        WriteUInt32(glb, ref pos, (uint)paddedJsonLen);
        WriteUInt32(glb, ref pos, JSON_CHUNK_TYPE);
        Array.Copy(paddedJson, 0, glb, pos, paddedJsonLen);
        pos += paddedJsonLen;

        // BIN chunk
        WriteUInt32(glb, ref pos, (uint)paddedBinBytes);
        WriteUInt32(glb, ref pos, BIN_CHUNK_TYPE);
        Array.Copy(binData, 0, glb, pos, paddedBinBytes);

        return glb;
    }

    /// <summary>Validate that bytes start with glTF magic.</summary>
    public static bool IsValidGLB(byte[] data)
    {
        if (data.Length < 12) return false;
        return BitConverter.ToUInt32(data, 0) == GLTF_MAGIC;
    }

    private static void WriteUInt32(byte[] buffer, ref int offset, uint value)
    {
        buffer[offset] = (byte)(value & 0xFF);
        buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
        buffer[offset + 2] = (byte)((value >> 16) & 0xFF);
        buffer[offset + 3] = (byte)((value >> 24) & 0xFF);
        offset += 4;
    }
}
