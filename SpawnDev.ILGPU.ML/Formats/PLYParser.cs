using System.Globalization;
using System.Text;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// PLY (Polygon File Format) parser for Gaussian splat and mesh data.
/// Supports both ASCII and binary_little_endian formats.
/// Standard format for 3D Gaussian Splatting output (3DGS, LGM, etc.).
///
/// Gaussian splat PLY files contain per-vertex properties:
/// x,y,z (position), nx,ny,nz (normal), f_dc_0..2 (color DC),
/// f_rest_0..N (SH coefficients), opacity, scale_0..2, rot_0..3.
/// </summary>
public static class PLYParser
{
    /// <summary>
    /// Parse a PLY file into a GaussianCloud (if Gaussian splat properties found)
    /// or raw vertex/face data.
    /// </summary>
    public static PLYData Parse(byte[] plyBytes)
    {
        var text = Encoding.ASCII.GetString(plyBytes);
        var lines = text.Split('\n');

        if (!lines[0].TrimEnd('\r').StartsWith("ply"))
            throw new InvalidDataException("Not a PLY file — missing 'ply' header");

        // Parse header
        string format = "ascii";
        int vertexCount = 0, faceCount = 0;
        var properties = new List<(string Name, string Type)>();
        int headerEndLine = 0;
        bool inVertexElement = false;

        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i].TrimEnd('\r').Trim();
            if (line == "end_header")
            {
                headerEndLine = i;
                break;
            }

            if (line.StartsWith("format "))
                format = line.Substring(7).Trim().Split(' ')[0];
            else if (line.StartsWith("element vertex "))
            {
                vertexCount = int.Parse(line.Substring(15).Trim());
                inVertexElement = true;
            }
            else if (line.StartsWith("element face "))
            {
                faceCount = int.Parse(line.Substring(13).Trim());
                inVertexElement = false;
            }
            else if (line.StartsWith("element "))
                inVertexElement = false;
            else if (line.StartsWith("property ") && inVertexElement)
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3)
                    properties.Add((parts[2], parts[1]));
            }
        }

        var result = new PLYData
        {
            Format = format,
            VertexCount = vertexCount,
            FaceCount = faceCount,
            Properties = properties.Select(p => p.Name).ToArray(),
        };

        // Parse vertex data
        if (format == "ascii")
        {
            result.VertexData = ParseASCII(lines, headerEndLine + 1, vertexCount, properties.Count);
        }
        else if (format.StartsWith("binary_little_endian"))
        {
            // Find header end in bytes
            int headerBytes = 0;
            for (int i = 0; i <= headerEndLine; i++)
                headerBytes += Encoding.ASCII.GetByteCount(lines[i]) + 1; // +1 for newline

            result.VertexData = ParseBinaryLE(plyBytes, headerBytes, vertexCount, properties);
        }

        // Try to extract Gaussian splat data if properties match
        result.Gaussians = TryExtractGaussians(result, properties);

        return result;
    }

    private static float[,] ParseASCII(string[] lines, int startLine, int vertexCount, int propCount)
    {
        var data = new float[vertexCount, propCount];
        for (int v = 0; v < vertexCount && startLine + v < lines.Length; v++)
        {
            var parts = lines[startLine + v].TrimEnd('\r').Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int p = 0; p < propCount && p < parts.Length; p++)
            {
                if (float.TryParse(parts[p], NumberStyles.Float, CultureInfo.InvariantCulture, out float val))
                    data[v, p] = val;
            }
        }
        return data;
    }

    private static float[,] ParseBinaryLE(byte[] bytes, int offset, int vertexCount,
        List<(string Name, string Type)> properties)
    {
        var data = new float[vertexCount, properties.Count];
        int pos = offset;

        for (int v = 0; v < vertexCount; v++)
        {
            for (int p = 0; p < properties.Count; p++)
            {
                if (pos >= bytes.Length) break;
                var type = properties[p].Type;
                switch (type)
                {
                    case "float":
                    case "float32":
                        data[v, p] = BitConverter.ToSingle(bytes, pos);
                        pos += 4;
                        break;
                    case "double":
                    case "float64":
                        data[v, p] = (float)BitConverter.ToDouble(bytes, pos);
                        pos += 8;
                        break;
                    case "uchar":
                    case "uint8":
                        data[v, p] = bytes[pos] / 255f;
                        pos += 1;
                        break;
                    case "int":
                    case "int32":
                        data[v, p] = BitConverter.ToInt32(bytes, pos);
                        pos += 4;
                        break;
                    case "short":
                    case "int16":
                        data[v, p] = BitConverter.ToInt16(bytes, pos);
                        pos += 2;
                        break;
                    default:
                        pos += 4; // assume float
                        break;
                }
            }
        }

        return data;
    }

    private static GaussianCloud? TryExtractGaussians(PLYData data, List<(string Name, string Type)> properties)
    {
        // Check for Gaussian splat properties
        var propNames = properties.Select(p => p.Name).ToList();
        int xIdx = propNames.IndexOf("x");
        int yIdx = propNames.IndexOf("y");
        int zIdx = propNames.IndexOf("z");
        int opacityIdx = propNames.IndexOf("opacity");

        if (xIdx < 0 || yIdx < 0 || zIdx < 0 || data.VertexData == null) return null;

        int n = data.VertexCount;
        var cloud = new GaussianCloud
        {
            NumPoints = n,
            Positions = new float[n * 3],
            Alphas = new float[n],
            Colors = new float[n * 3],
            Scales = new float[n * 3],
            Rotations = new float[n * 4],
        };

        // Extract positions
        for (int i = 0; i < n; i++)
        {
            cloud.Positions[i * 3 + 0] = data.VertexData[i, xIdx];
            cloud.Positions[i * 3 + 1] = data.VertexData[i, yIdx];
            cloud.Positions[i * 3 + 2] = data.VertexData[i, zIdx];
        }

        // Extract opacity
        if (opacityIdx >= 0)
            for (int i = 0; i < n; i++)
                cloud.Alphas[i] = 1f / (1f + MathF.Exp(-data.VertexData[i, opacityIdx])); // sigmoid

        // Extract colors (f_dc_0, f_dc_1, f_dc_2)
        int dc0 = propNames.IndexOf("f_dc_0");
        int dc1 = propNames.IndexOf("f_dc_1");
        int dc2 = propNames.IndexOf("f_dc_2");
        if (dc0 >= 0 && dc1 >= 0 && dc2 >= 0)
        {
            for (int i = 0; i < n; i++)
            {
                // SH DC to RGB: color = 0.5 + C0 * SH_C0 where SH_C0 = 0.28209479
                cloud.Colors[i * 3 + 0] = 0.5f + data.VertexData[i, dc0] * 0.28209479f;
                cloud.Colors[i * 3 + 1] = 0.5f + data.VertexData[i, dc1] * 0.28209479f;
                cloud.Colors[i * 3 + 2] = 0.5f + data.VertexData[i, dc2] * 0.28209479f;
            }
        }

        // Extract scales (scale_0, scale_1, scale_2)
        int s0 = propNames.IndexOf("scale_0");
        int s1 = propNames.IndexOf("scale_1");
        int s2 = propNames.IndexOf("scale_2");
        if (s0 >= 0 && s1 >= 0 && s2 >= 0)
            for (int i = 0; i < n; i++)
            {
                cloud.Scales[i * 3 + 0] = data.VertexData[i, s0];
                cloud.Scales[i * 3 + 1] = data.VertexData[i, s1];
                cloud.Scales[i * 3 + 2] = data.VertexData[i, s2];
            }

        // Extract rotations (rot_0, rot_1, rot_2, rot_3)
        int r0 = propNames.IndexOf("rot_0");
        int r1 = propNames.IndexOf("rot_1");
        int r2 = propNames.IndexOf("rot_2");
        int r3 = propNames.IndexOf("rot_3");
        if (r0 >= 0 && r1 >= 0 && r2 >= 0 && r3 >= 0)
            for (int i = 0; i < n; i++)
            {
                cloud.Rotations[i * 4 + 0] = data.VertexData[i, r0]; // w
                cloud.Rotations[i * 4 + 1] = data.VertexData[i, r1]; // x
                cloud.Rotations[i * 4 + 2] = data.VertexData[i, r2]; // y
                cloud.Rotations[i * 4 + 3] = data.VertexData[i, r3]; // z
            }

        return cloud;
    }
}

/// <summary>Raw PLY file data.</summary>
public class PLYData
{
    public string Format { get; set; } = "";
    public int VertexCount { get; set; }
    public int FaceCount { get; set; }
    public string[] Properties { get; set; } = Array.Empty<string>();
    public float[,]? VertexData { get; set; }
    public GaussianCloud? Gaussians { get; set; }
}
