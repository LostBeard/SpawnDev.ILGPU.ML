using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.ML.Trellis2;

/// <summary>Minimal PNG writer for 8-bit RGBA (the textures of a GLB are PNG images embedded in the binary chunk).</summary>
public static class PngEncoder
{
    private static readonly uint[] CrcTable = BuildCrcTable();
    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++) { uint c = n; for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1; t[n] = c; }
        return t;
    }
    private static uint Crc(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (var b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc;
    }

    private static void Chunk(Stream s, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> len = stackalloc byte[4]; BinaryPrimitives.WriteUInt32BigEndian(len, (uint)data.Length); s.Write(len);
        var t = Encoding.ASCII.GetBytes(type); s.Write(t); s.Write(data);
        uint crc = Crc(Crc(0xFFFFFFFFu, t), data) ^ 0xFFFFFFFFu;
        Span<byte> c = stackalloc byte[4]; BinaryPrimitives.WriteUInt32BigEndian(c, crc); s.Write(c);
    }

    /// <summary>Encode packed pixels (R in the low byte ... A in the high byte), row 0 first.</summary>
    public static byte[] EncodeRgba8(uint[] pixels, int width, int height, CompressionLevel level = CompressionLevel.Fastest)
    {
        using var ms = new MemoryStream();
        ms.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
        Span<byte> ihdr = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)width); BinaryPrimitives.WriteUInt32BigEndian(ihdr[4..], (uint)height);
        ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
        Chunk(ms, "IHDR", ihdr);
        var raw = new byte[(width * 4 + 1) * height];
        int o = 0;
        for (int y = 0; y < height; y++)
        {
            raw[o++] = 0;                                   // filter: none
            for (int x = 0; x < width; x++)
            {
                uint p = pixels[y * width + x];
                raw[o++] = (byte)p; raw[o++] = (byte)(p >> 8); raw[o++] = (byte)(p >> 16); raw[o++] = (byte)(p >> 24);
            }
        }
        using var z = new MemoryStream();
        using (var zs = new ZLibStream(z, level, leaveOpen: true)) zs.Write(raw);
        Chunk(ms, "IDAT", z.GetBuffer().AsSpan(0, (int)z.Length));
        Chunk(ms, "IEND", ReadOnlySpan<byte>.Empty);
        return ms.ToArray();
    }
}

/// <summary>Binary glTF 2.0 writer for one PBR mesh: positions, normals, UVs, 32-bit indices, base colour and metallic-roughness PNGs.</summary>
public static class PbrGlbWriter
{
    /// <summary>
    /// Inputs are in the pipeline's frame. The reference converts for glTF with (x, y, z) -> (x, z, -y) on positions and normals
    /// (a proper rotation, winding is kept); UVs are used as-is (image row 0 = v 0).
    /// </summary>
    public static byte[] Write(float[] positions, float[] normals, float[] uvs, int[] indices, byte[] baseColorPng, byte[] metallicRoughnessPng, bool doubleSided = true)
    {
        int V = positions.Length / 3;
        var pos = new float[V * 3]; var nrm = new float[V * 3];
        float[] min = { float.MaxValue, float.MaxValue, float.MaxValue }, max = { float.MinValue, float.MinValue, float.MinValue };
        for (int i = 0; i < V; i++)
        {
            pos[i * 3] = positions[i * 3]; pos[i * 3 + 1] = positions[i * 3 + 2]; pos[i * 3 + 2] = -positions[i * 3 + 1];
            nrm[i * 3] = normals[i * 3]; nrm[i * 3 + 1] = normals[i * 3 + 2]; nrm[i * 3 + 2] = -normals[i * 3 + 1];
            for (int c = 0; c < 3; c++) { min[c] = Math.Min(min[c], pos[i * 3 + c]); max[c] = Math.Max(max[c], pos[i * 3 + c]); }
        }
        var bin = new MemoryStream();
        var views = new List<(int offset, int length, int? target)>();
        void AddView(ReadOnlySpan<byte> data, int? target)
        {
            while (bin.Length % 4 != 0) bin.WriteByte(0);
            views.Add(((int)bin.Length, data.Length, target));
            bin.Write(data);
        }
        AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pos.AsSpan()), 34962);
        AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(nrm.AsSpan()), 34962);
        AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(uvs.AsSpan()), 34962);
        AddView(System.Runtime.InteropServices.MemoryMarshal.AsBytes(indices.AsSpan()), 34963);
        AddView(baseColorPng, null);
        AddView(metallicRoughnessPng, null);
        while (bin.Length % 4 != 0) bin.WriteByte(0);

        var json = new MemoryStream();
        using (var w = new Utf8JsonWriter(json))
        {
            w.WriteStartObject();
            w.WriteStartObject("asset"); w.WriteString("version", "2.0"); w.WriteString("generator", "SpawnDev.ILGPU.ML TRELLIS.2"); w.WriteEndObject();
            w.WriteNumber("scene", 0);
            w.WriteStartArray("scenes"); w.WriteStartObject(); w.WriteStartArray("nodes"); w.WriteNumberValue(0); w.WriteEndArray(); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("nodes"); w.WriteStartObject(); w.WriteNumber("mesh", 0); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("meshes"); w.WriteStartObject(); w.WriteStartArray("primitives"); w.WriteStartObject();
            w.WriteStartObject("attributes"); w.WriteNumber("POSITION", 0); w.WriteNumber("NORMAL", 1); w.WriteNumber("TEXCOORD_0", 2); w.WriteEndObject();
            w.WriteNumber("indices", 3); w.WriteNumber("material", 0); w.WriteNumber("mode", 4);
            w.WriteEndObject(); w.WriteEndArray(); w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("materials"); w.WriteStartObject();
            w.WriteStartObject("pbrMetallicRoughness");
            w.WriteStartObject("baseColorTexture"); w.WriteNumber("index", 0); w.WriteEndObject();
            w.WriteStartObject("metallicRoughnessTexture"); w.WriteNumber("index", 1); w.WriteEndObject();
            w.WriteStartArray("baseColorFactor"); for (int i = 0; i < 4; i++) w.WriteNumberValue(1.0); w.WriteEndArray();
            w.WriteNumber("metallicFactor", 1.0); w.WriteNumber("roughnessFactor", 1.0);
            w.WriteEndObject();
            w.WriteString("alphaMode", "OPAQUE"); w.WriteBoolean("doubleSided", doubleSided);
            w.WriteEndObject(); w.WriteEndArray();
            w.WriteStartArray("textures"); for (int i = 0; i < 2; i++) { w.WriteStartObject(); w.WriteNumber("source", i); w.WriteEndObject(); } w.WriteEndArray();
            w.WriteStartArray("images"); for (int i = 0; i < 2; i++) { w.WriteStartObject(); w.WriteNumber("bufferView", 4 + i); w.WriteString("mimeType", "image/png"); w.WriteEndObject(); } w.WriteEndArray();
            w.WriteStartArray("accessors");
            w.WriteStartObject(); w.WriteNumber("bufferView", 0); w.WriteNumber("componentType", 5126); w.WriteNumber("count", V); w.WriteString("type", "VEC3");
            w.WriteStartArray("min"); foreach (var m in min) w.WriteNumberValue(m); w.WriteEndArray();
            w.WriteStartArray("max"); foreach (var m in max) w.WriteNumberValue(m); w.WriteEndArray(); w.WriteEndObject();
            w.WriteStartObject(); w.WriteNumber("bufferView", 1); w.WriteNumber("componentType", 5126); w.WriteNumber("count", V); w.WriteString("type", "VEC3"); w.WriteEndObject();
            w.WriteStartObject(); w.WriteNumber("bufferView", 2); w.WriteNumber("componentType", 5126); w.WriteNumber("count", V); w.WriteString("type", "VEC2"); w.WriteEndObject();
            w.WriteStartObject(); w.WriteNumber("bufferView", 3); w.WriteNumber("componentType", 5125); w.WriteNumber("count", indices.Length); w.WriteString("type", "SCALAR"); w.WriteEndObject();
            w.WriteEndArray();
            w.WriteStartArray("bufferViews");
            foreach (var (offset, length, target) in views)
            {
                w.WriteStartObject(); w.WriteNumber("buffer", 0); w.WriteNumber("byteOffset", offset); w.WriteNumber("byteLength", length);
                if (target.HasValue) w.WriteNumber("target", target.Value);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("buffers"); w.WriteStartObject(); w.WriteNumber("byteLength", (int)bin.Length); w.WriteEndObject(); w.WriteEndArray();
            w.WriteEndObject();
        }
        while (json.Length % 4 != 0) json.WriteByte(0x20);
        int total = 12 + 8 + (int)json.Length + 8 + (int)bin.Length;
        var glb = new MemoryStream(total);
        var u = new byte[4];
        void U32(uint v) { BinaryPrimitives.WriteUInt32LittleEndian(u, v); glb.Write(u); }
        U32(0x46546C67); U32(2); U32((uint)total);                  // 'glTF', version 2
        U32((uint)json.Length); U32(0x4E4F534A); glb.Write(json.GetBuffer().AsSpan(0, (int)json.Length));   // JSON chunk
        U32((uint)bin.Length); U32(0x004E4942); glb.Write(bin.GetBuffer().AsSpan(0, (int)bin.Length));      // BIN chunk
        return glb.ToArray();
    }

    /// <summary>
    /// Vertex normals from BILATERALLY filtered face normals (Zheng et al. 2011, "Bilateral normal filtering for mesh denoising"): each pass sets a face's normal to the area-weighted sum of the normals of the faces
    /// that share a vertex with it, each weighted by exp(-(1 - cos angle) / sigma^2) against the face's own normal, so faces on the same smooth surface average out (the voxel-scale bumps of a decimated mesh,
    /// which a mirror-like surface shows as facets) while a face across a sharp edge counts for nothing (90 degrees at sigma 20 degrees: weight 3e-4) and flat sides keep their exact normal.
    /// The vertex normals are then the area-weighted sums of their faces' filtered normals; 0 iterations = <see cref="VertexNormals"/>. The device version is <c>DeviceUvAtlas</c>'s (same arithmetic).
    /// </summary>
    public static float[] BilateralVertexNormals(float[] positions, int[] faces, int iterations, double sigmaDegrees)
    {
        if (iterations <= 0) return VertexNormals(positions, faces);
        int nv = positions.Length / 3, nf = faces.Length / 3;
        var start = new int[nv + 1];
        for (int i = 0; i < faces.Length; i++) start[faces[i] + 1]++;
        for (int v = 0; v < nv; v++) start[v + 1] += start[v];
        var fill = new int[nv]; var list = new int[faces.Length];
        for (int f = 0; f < nf; f++) for (int k = 0; k < 3; k++) { int v = faces[f * 3 + k]; list[start[v] + fill[v]++] = f; }
        var src = new float[nf * 3]; var area = new float[nf]; var dst = new float[nf * 3];
        for (int f = 0; f < nf; f++)
        {
            int a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
            float ax = positions[b * 3] - positions[a * 3], ay = positions[b * 3 + 1] - positions[a * 3 + 1], az = positions[b * 3 + 2] - positions[a * 3 + 2];
            float bx = positions[c * 3] - positions[a * 3], by = positions[c * 3 + 1] - positions[a * 3 + 1], bz = positions[c * 3 + 2] - positions[a * 3 + 2];
            float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, l = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            area[f] = l * 0.5f;
            if (l > 1e-20f) { src[f * 3] = nx / l; src[f * 3 + 1] = ny / l; src[f * 3 + 2] = nz / l; } else { src[f * 3 + 2] = 1f; }
        }
        float sigma = (float)(sigmaDegrees * Math.PI / 180), invSigma2 = 1f / (sigma * sigma);
        for (int it = 0; it < iterations; it++)
        {
            for (int f = 0; f < nf; f++)
            {
                float nx = src[f * 3], ny = src[f * 3 + 1], nz = src[f * 3 + 2], sx = 0, sy = 0, sz = 0;
                for (int k = 0; k < 3; k++)
                {
                    int v = faces[f * 3 + k];
                    for (int i = start[v]; i < start[v + 1]; i++)
                    {
                        int g = list[i];
                        float gx = src[g * 3], gy = src[g * 3 + 1], gz = src[g * 3 + 2], w = area[g] * MathF.Exp(-(1f - (gx * nx + gy * ny + gz * nz)) * invSigma2);
                        sx += w * gx; sy += w * gy; sz += w * gz;
                    }
                }
                float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
                if (l < 1e-20f) { dst[f * 3] = nx; dst[f * 3 + 1] = ny; dst[f * 3 + 2] = nz; } else { dst[f * 3] = sx / l; dst[f * 3 + 1] = sy / l; dst[f * 3 + 2] = sz / l; }
            }
            (src, dst) = (dst, src);
        }
        var r = new float[positions.Length];
        for (int v = 0; v < nv; v++)
        {
            float sx = 0, sy = 0, sz = 0;
            for (int i = start[v]; i < start[v + 1]; i++) { int g = list[i]; sx += area[g] * src[g * 3]; sy += area[g] * src[g * 3 + 1]; sz += area[g] * src[g * 3 + 2]; }
            float l = MathF.Sqrt(sx * sx + sy * sy + sz * sz);
            if (l < 1e-20f) { r[v * 3 + 2] = 1f; continue; }
            r[v * 3] = sx / l; r[v * 3 + 1] = sy / l; r[v * 3 + 2] = sz / l;
        }
        return r;
    }

    /// <summary>Area-weighted vertex normals of an indexed triangle mesh.</summary>
    public static float[] VertexNormals(float[] positions, int[] faces)
    {
        var n = new double[positions.Length];
        for (int f = 0; f < faces.Length / 3; f++)
        {
            int a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
            double ax = positions[b * 3] - positions[a * 3], ay = positions[b * 3 + 1] - positions[a * 3 + 1], az = positions[b * 3 + 2] - positions[a * 3 + 2];
            double bx = positions[c * 3] - positions[a * 3], by = positions[c * 3 + 1] - positions[a * 3 + 1], bz = positions[c * 3 + 2] - positions[a * 3 + 2];
            double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;      // area-weighted (length = 2 * area)
            foreach (int v in new[] { a, b, c }) { n[v * 3] += nx; n[v * 3 + 1] += ny; n[v * 3 + 2] += nz; }
        }
        var r = new float[positions.Length];
        for (int v = 0; v < r.Length / 3; v++)
        {
            double l = Math.Sqrt(n[v * 3] * n[v * 3] + n[v * 3 + 1] * n[v * 3 + 1] + n[v * 3 + 2] * n[v * 3 + 2]);
            if (l < 1e-20) { r[v * 3 + 2] = 1; continue; }
            r[v * 3] = (float)(n[v * 3] / l); r[v * 3 + 1] = (float)(n[v * 3 + 1] / l); r[v * 3 + 2] = (float)(n[v * 3 + 2] / l);
        }
        return r;
    }
}
