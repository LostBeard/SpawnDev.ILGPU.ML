using System.IO.Compression;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// Parser for SPZ (Spatially ordered Packed Gaussians) format.
/// 15-20x compression over PLY for Gaussian Splatting scenes.
/// Format by PlayCanvas + Niantic, becoming Khronos glTF standard.
///
/// SPZ files are gzip-compressed binary with column-major attribute storage.
/// After decompression: [header][positions][alphas][colors][scales][rotations][SH]
/// </summary>
public static class SPZParser
{
    // SPZ magic: "SPGS" in ASCII = 0x53504753
    private const uint MAGIC_SPGS = 0x53504753;

    /// <summary>
    /// Parse an SPZ file into a GaussianCloud.
    /// </summary>
    public static GaussianCloud Parse(byte[] spzBytes)
    {
        // Decompress gzip
        byte[] data;
        using (var compressed = new MemoryStream(spzBytes))
        using (var gzip = new GZipStream(compressed, CompressionMode.Decompress))
        using (var decompressed = new MemoryStream())
        {
            gzip.CopyTo(decompressed);
            data = decompressed.ToArray();
        }

        return ParseDecompressed(data);
    }

    /// <summary>
    /// Parse decompressed SPZ data.
    /// </summary>
    public static GaussianCloud ParseDecompressed(byte[] data)
    {
        if (data.Length < 16)
            throw new InvalidDataException($"SPZ data too small: {data.Length} bytes");

        // Parse header
        int pos = 0;
        uint magic = BitConverter.ToUInt32(data, pos); pos += 4;

        // Accept multiple magic conventions
        if (magic != MAGIC_SPGS && magic != 0x4E475350 && magic != 0x5053474E)
            throw new InvalidDataException($"Invalid SPZ magic: 0x{magic:X8} (expected SPGS/NGSP/PSGN)");

        int version = BitConverter.ToInt32(data, pos); pos += 4;
        int numPoints = BitConverter.ToInt32(data, pos); pos += 4;
        int shDegree = data[pos]; pos += 1;
        int fractionalBits = data[pos]; pos += 1;
        int flags = data[pos]; pos += 1;
        pos += 1; // reserved byte

        if (numPoints < 0 || numPoints > 100_000_000)
            throw new InvalidDataException($"Invalid numPoints: {numPoints}");

        var cloud = new GaussianCloud
        {
            Version = version,
            NumPoints = numPoints,
            SHDegree = shDegree,
            FractionalBits = fractionalBits,
            Flags = flags,
            Positions = new float[numPoints * 3],
            Alphas = new float[numPoints],
            Colors = new float[numPoints * 3],
            Scales = new float[numPoints * 3],
            Rotations = new float[numPoints * 4],
        };

        // Column-major attribute storage
        // Positions: 3 × int24 (24-bit fixed-point per coordinate)
        if (pos + numPoints * 9 <= data.Length) // 3 bytes × 3 coords = 9 bytes per point
        {
            for (int i = 0; i < numPoints; i++)
            {
                for (int c = 0; c < 3; c++)
                {
                    // 24-bit signed fixed-point with fractionalBits
                    int raw = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16);
                    // Sign extend from 24-bit
                    if ((raw & 0x800000) != 0) raw |= unchecked((int)0xFF000000);
                    cloud.Positions[i * 3 + c] = raw / (float)(1 << fractionalBits);
                    pos += 3;
                }
            }
        }

        // Alphas: 1 × uint8 per point (inverse sigmoid encoded)
        if (pos + numPoints <= data.Length)
        {
            for (int i = 0; i < numPoints; i++)
            {
                // Decode: uint8 → sigmoid(value/255 * range - offset)
                float raw = data[pos++] / 255f;
                // Inverse sigmoid: alpha = sigmoid(raw * 10 - 5) for typical range
                cloud.Alphas[i] = 1f / (1f + MathF.Exp(-(raw * 10f - 5f)));
            }
        }

        // Colors: 3 × uint8 per point (scaled)
        if (pos + numPoints * 3 <= data.Length)
        {
            float colorScale = 1f / 255f;
            for (int i = 0; i < numPoints; i++)
            {
                cloud.Colors[i * 3 + 0] = data[pos++] * colorScale;
                cloud.Colors[i * 3 + 1] = data[pos++] * colorScale;
                cloud.Colors[i * 3 + 2] = data[pos++] * colorScale;
            }
        }

        // Scales: 3 × uint8 per point (log-space encoded)
        if (pos + numPoints * 3 <= data.Length)
        {
            for (int i = 0; i < numPoints; i++)
            {
                for (int c = 0; c < 3; c++)
                {
                    // Log-space: scale = exp(uint8/255 * range + offset)
                    float raw = data[pos++] / 255f;
                    cloud.Scales[i * 3 + c] = raw * 15.9f - 10f; // log-space value
                }
            }
        }

        // Rotations: depends on version
        if (version >= 3)
        {
            // v3+: smallest-three encoding, 3 × uint8 per point
            if (pos + numPoints * 3 <= data.Length)
            {
                for (int i = 0; i < numPoints; i++)
                {
                    int b0 = data[pos++], b1 = data[pos++], b2 = data[pos++];
                    // Decode smallest-three quaternion
                    // For now, store as identity — full decode needs the missing component index
                    cloud.Rotations[i * 4 + 0] = 1f; // w
                    cloud.Rotations[i * 4 + 1] = (b0 / 255f) * 2f - 1f; // x
                    cloud.Rotations[i * 4 + 2] = (b1 / 255f) * 2f - 1f; // y
                    cloud.Rotations[i * 4 + 3] = (b2 / 255f) * 2f - 1f; // z
                    // Normalize to unit quaternion
                    float len = MathF.Sqrt(
                        cloud.Rotations[i * 4 + 0] * cloud.Rotations[i * 4 + 0] +
                        cloud.Rotations[i * 4 + 1] * cloud.Rotations[i * 4 + 1] +
                        cloud.Rotations[i * 4 + 2] * cloud.Rotations[i * 4 + 2] +
                        cloud.Rotations[i * 4 + 3] * cloud.Rotations[i * 4 + 3]);
                    if (len > 0)
                    {
                        cloud.Rotations[i * 4 + 0] /= len;
                        cloud.Rotations[i * 4 + 1] /= len;
                        cloud.Rotations[i * 4 + 2] /= len;
                        cloud.Rotations[i * 4 + 3] /= len;
                    }
                }
            }
        }

        cloud.CompressedSize = data.Length;

        return cloud;
    }

    /// <summary>
    /// Validate that data starts with valid SPZ magic bytes (after gzip decompression).
    /// </summary>
    public static bool IsValidSPZ(byte[] spzBytes)
    {
        try
        {
            using var compressed = new MemoryStream(spzBytes);
            using var gzip = new GZipStream(compressed, CompressionMode.Decompress);
            var header = new byte[4];
            int read = gzip.Read(header, 0, 4);
            if (read < 4) return false;
            uint magic = BitConverter.ToUInt32(header, 0);
            return magic == MAGIC_SPGS || magic == 0x4E475350 || magic == 0x5053474E;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// A cloud of Gaussian splats parsed from SPZ or PLY format.
/// </summary>
public class GaussianCloud
{
    public int Version { get; set; }
    public int NumPoints { get; set; }
    public int SHDegree { get; set; }
    public int FractionalBits { get; set; }
    public int Flags { get; set; }
    public int CompressedSize { get; set; }

    /// <summary>Positions [numPoints × 3] (x, y, z)</summary>
    public float[] Positions { get; set; } = Array.Empty<float>();
    /// <summary>Opacities [numPoints] in [0, 1]</summary>
    public float[] Alphas { get; set; } = Array.Empty<float>();
    /// <summary>Colors [numPoints × 3] (r, g, b) in [0, 1]</summary>
    public float[] Colors { get; set; } = Array.Empty<float>();
    /// <summary>Scales [numPoints × 3] (log-space)</summary>
    public float[] Scales { get; set; } = Array.Empty<float>();
    /// <summary>Rotations [numPoints × 4] (w, x, y, z quaternion)</summary>
    public float[] Rotations { get; set; } = Array.Empty<float>();
    /// <summary>Spherical harmonics [numPoints × shCoeffs × 3] (optional)</summary>
    public float[]? SHCoefficients { get; set; }
}
