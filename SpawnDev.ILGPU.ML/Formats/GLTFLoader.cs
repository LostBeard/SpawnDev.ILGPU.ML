using System.Text;
using System.Text.Json;
using SpawnDev.ILGPU.ML.Kernels;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// Minimal glTF 2.0 binary (.glb) loader for triangle meshes.
/// Loads vertex positions, indices, and optional vertex colors.
/// </summary>
public static class GLTFLoader
{
    private const uint GLTF_MAGIC = 0x46546C67;

    /// <summary>
    /// Load a .glb file into MeshData.
    /// </summary>
    public static MeshData LoadGLB(byte[] glbBytes)
    {
        if (glbBytes.Length < 12)
            throw new InvalidDataException("GLB file too small");

        uint magic = BitConverter.ToUInt32(glbBytes, 0);
        if (magic != GLTF_MAGIC)
            throw new InvalidDataException($"Invalid GLB magic: 0x{magic:X8}");

        uint version = BitConverter.ToUInt32(glbBytes, 4);
        uint totalLength = BitConverter.ToUInt32(glbBytes, 8);

        // Parse chunks
        int pos = 12;
        byte[]? jsonBytes = null;
        byte[]? binBytes = null;

        while (pos < glbBytes.Length - 8)
        {
            uint chunkLength = BitConverter.ToUInt32(glbBytes, pos);
            uint chunkType = BitConverter.ToUInt32(glbBytes, pos + 4);
            pos += 8;

            if (chunkType == 0x4E4F534A) // JSON
            {
                jsonBytes = new byte[chunkLength];
                Array.Copy(glbBytes, pos, jsonBytes, 0, (int)chunkLength);
            }
            else if (chunkType == 0x004E4942) // BIN
            {
                binBytes = new byte[chunkLength];
                Array.Copy(glbBytes, pos, binBytes, 0, (int)chunkLength);
            }

            pos += (int)chunkLength;
        }

        if (jsonBytes == null || binBytes == null)
            throw new InvalidDataException("GLB missing JSON or BIN chunk");

        var jsonStr = Encoding.UTF8.GetString(jsonBytes);
        using var doc = JsonDocument.Parse(jsonStr);
        var root = doc.RootElement;

        // Find the first mesh primitive
        var meshes = root.GetProperty("meshes");
        var primitive = meshes[0].GetProperty("primitives")[0];
        var attributes = primitive.GetProperty("attributes");
        int positionAccessorIdx = attributes.GetProperty("POSITION").GetInt32();
        int indicesAccessorIdx = primitive.GetProperty("indices").GetInt32();

        var accessors = root.GetProperty("accessors");
        var bufferViews = root.GetProperty("bufferViews");

        // Read positions
        var posAccessor = accessors[positionAccessorIdx];
        int posCount = posAccessor.GetProperty("count").GetInt32();
        int posBVIdx = posAccessor.GetProperty("bufferView").GetInt32();
        var posBV = bufferViews[posBVIdx];
        int posOffset = posBV.GetProperty("byteOffset").GetInt32();
        int posLength = posBV.GetProperty("byteLength").GetInt32();

        var vertices = new float[posCount * 3];
        Buffer.BlockCopy(binBytes, posOffset, vertices, 0, posCount * 3 * 4);

        // Read indices
        var idxAccessor = accessors[indicesAccessorIdx];
        int idxCount = idxAccessor.GetProperty("count").GetInt32();
        int idxBVIdx = idxAccessor.GetProperty("bufferView").GetInt32();
        var idxBV = bufferViews[idxBVIdx];
        int idxOffset = idxBV.GetProperty("byteOffset").GetInt32();

        int componentType = idxAccessor.GetProperty("componentType").GetInt32();
        var indices = new int[idxCount];
        if (componentType == 5125) // UNSIGNED_INT
        {
            Buffer.BlockCopy(binBytes, idxOffset, indices, 0, idxCount * 4);
        }
        else if (componentType == 5123) // UNSIGNED_SHORT
        {
            for (int i = 0; i < idxCount; i++)
                indices[i] = BitConverter.ToUInt16(binBytes, idxOffset + i * 2);
        }

        return new MeshData
        {
            Vertices = vertices,
            Indices = indices,
            VertexCount = posCount,
            TriangleCount = idxCount / 3,
        };
    }
}
