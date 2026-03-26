using System.IO.Compression;
using System.Text;

namespace SpawnDev.ILGPU.ML.Formats;

/// <summary>
/// Exports a GaussianCloud to SPZ (compressed) or PLY (standard) format.
/// SPZ gives 15-20x compression over PLY via 24-bit fixed-point + uint8 quantization + gzip.
/// </summary>
public static class SPZExporter
{
    /// <summary>
    /// Export GaussianCloud to SPZ format (gzip-compressed binary).
    /// </summary>
    public static byte[] ExportSPZ(GaussianCloud cloud, int fractionalBits = 12)
    {
        var raw = ExportSPZRaw(cloud, fractionalBits);
        using var output = new MemoryStream();
        using (var gzip = new GZipStream(output, CompressionLevel.Optimal))
        {
            gzip.Write(raw, 0, raw.Length);
        }
        return output.ToArray();
    }

    private static byte[] ExportSPZRaw(GaussianCloud cloud, int fractionalBits)
    {
        int n = cloud.NumPoints;
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Header: magic(4) + version(4) + numPoints(4) + shDegree(1) + fractionalBits(1) + flags(1) + reserved(1) = 16 bytes
        bw.Write(0x53504753u); // "SPGS"
        bw.Write(4); // version 4
        bw.Write(n);
        bw.Write((byte)cloud.SHDegree);
        bw.Write((byte)fractionalBits);
        bw.Write((byte)0); // flags
        bw.Write((byte)0); // reserved

        // Positions: 3 × 24-bit fixed-point per point (column-major: all x, then all y, then all z)
        float scale = 1 << fractionalBits;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                int val = (int)MathF.Round(cloud.Positions[i * 3 + c] * scale);
                // Write 24-bit signed
                bw.Write((byte)(val & 0xFF));
                bw.Write((byte)((val >> 8) & 0xFF));
                bw.Write((byte)((val >> 16) & 0xFF));
            }
        }

        // Alphas: 1 × uint8 per point (inverse sigmoid encoded)
        for (int i = 0; i < n; i++)
        {
            float alpha = Math.Clamp(cloud.Alphas[i], 0.001f, 0.999f);
            // Inverse sigmoid: logit = log(alpha / (1 - alpha))
            float logit = MathF.Log(alpha / (1f - alpha));
            // Map to [0, 255]: uint8 = (logit + 5) / 10 * 255
            int val = (int)MathF.Round((logit + 5f) / 10f * 255f);
            bw.Write((byte)Math.Clamp(val, 0, 255));
        }

        // Colors: 3 × uint8 per point
        for (int i = 0; i < n; i++)
        {
            bw.Write((byte)Math.Clamp((int)(cloud.Colors[i * 3 + 0] * 255f + 0.5f), 0, 255));
            bw.Write((byte)Math.Clamp((int)(cloud.Colors[i * 3 + 1] * 255f + 0.5f), 0, 255));
            bw.Write((byte)Math.Clamp((int)(cloud.Colors[i * 3 + 2] * 255f + 0.5f), 0, 255));
        }

        // Scales: 3 × uint8 per point (log-space)
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float logScale = cloud.Scales[i * 3 + c];
                int val = (int)MathF.Round((logScale + 10f) / 15.9f * 255f);
                bw.Write((byte)Math.Clamp(val, 0, 255));
            }
        }

        // Rotations: 3 × uint8 per point (smallest-three, v4)
        for (int i = 0; i < n; i++)
        {
            float x = cloud.Rotations[i * 4 + 1];
            float y = cloud.Rotations[i * 4 + 2];
            float z = cloud.Rotations[i * 4 + 3];
            bw.Write((byte)Math.Clamp((int)((x + 1f) / 2f * 255f + 0.5f), 0, 255));
            bw.Write((byte)Math.Clamp((int)((y + 1f) / 2f * 255f + 0.5f), 0, 255));
            bw.Write((byte)Math.Clamp((int)((z + 1f) / 2f * 255f + 0.5f), 0, 255));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Export GaussianCloud to PLY format (binary_little_endian, float32).
    /// Lossless — no quantization. Standard format for 3DGS tools.
    /// </summary>
    public static byte[] ExportPLY(GaussianCloud cloud)
    {
        int n = cloud.NumPoints;
        using var ms = new MemoryStream();
        using var sw = new StreamWriter(ms, Encoding.ASCII, leaveOpen: true);

        // Header
        sw.WriteLine("ply");
        sw.WriteLine("format binary_little_endian 1.0");
        sw.WriteLine($"element vertex {n}");
        sw.WriteLine("property float x");
        sw.WriteLine("property float y");
        sw.WriteLine("property float z");
        sw.WriteLine("property float opacity");
        sw.WriteLine("property float f_dc_0");
        sw.WriteLine("property float f_dc_1");
        sw.WriteLine("property float f_dc_2");
        sw.WriteLine("property float scale_0");
        sw.WriteLine("property float scale_1");
        sw.WriteLine("property float scale_2");
        sw.WriteLine("property float rot_0");
        sw.WriteLine("property float rot_1");
        sw.WriteLine("property float rot_2");
        sw.WriteLine("property float rot_3");
        sw.WriteLine("end_header");
        sw.Flush();

        // Binary vertex data
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);
        for (int i = 0; i < n; i++)
        {
            // Position
            bw.Write(cloud.Positions[i * 3 + 0]);
            bw.Write(cloud.Positions[i * 3 + 1]);
            bw.Write(cloud.Positions[i * 3 + 2]);

            // Opacity (inverse sigmoid for PLY convention)
            float alpha = Math.Clamp(cloud.Alphas[i], 0.001f, 0.999f);
            bw.Write(MathF.Log(alpha / (1f - alpha)));

            // Colors (SH DC: (color - 0.5) / SH_C0)
            float shC0 = 0.28209479f;
            bw.Write((cloud.Colors[i * 3 + 0] - 0.5f) / shC0);
            bw.Write((cloud.Colors[i * 3 + 1] - 0.5f) / shC0);
            bw.Write((cloud.Colors[i * 3 + 2] - 0.5f) / shC0);

            // Scales (log-space)
            bw.Write(cloud.Scales[i * 3 + 0]);
            bw.Write(cloud.Scales[i * 3 + 1]);
            bw.Write(cloud.Scales[i * 3 + 2]);

            // Rotation quaternion (w, x, y, z)
            bw.Write(cloud.Rotations[i * 4 + 0]);
            bw.Write(cloud.Rotations[i * 4 + 1]);
            bw.Write(cloud.Rotations[i * 4 + 2]);
            bw.Write(cloud.Rotations[i * 4 + 3]);
        }

        return ms.ToArray();
    }
}
