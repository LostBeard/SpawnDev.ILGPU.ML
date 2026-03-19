using System.IO.Compression;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Minimal PNG encoder. Writes RGBA pixel data to PNG format.
/// Pure C# — no external dependencies.
/// Useful for saving inference results (depth maps, segmentation masks, style transfer output).
/// </summary>
public static class PngEncoder
{
    /// <summary>
    /// Encode RGBA pixel data as a PNG file.
    /// </summary>
    /// <param name="rgba">RGBA pixel data, row-major, 4 bytes per pixel</param>
    /// <param name="width">Image width</param>
    /// <param name="height">Image height</param>
    /// <returns>Complete PNG file as byte array</returns>
    public static byte[] Encode(byte[] rgba, int width, int height)
    {
        using var output = new MemoryStream();

        // PNG signature
        output.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        // IHDR chunk
        WriteChunk(output, "IHDR", writer =>
        {
            WriteBE32(writer, width);
            WriteBE32(writer, height);
            writer.WriteByte(8);  // Bit depth: 8
            writer.WriteByte(6);  // Color type: RGBA
            writer.WriteByte(0);  // Compression: deflate
            writer.WriteByte(0);  // Filter: adaptive
            writer.WriteByte(0);  // Interlace: none
        });

        // IDAT chunk(s)
        var rawData = new byte[height * (1 + width * 4)]; // Filter byte + pixel data per row
        int pos = 0;
        for (int y = 0; y < height; y++)
        {
            rawData[pos++] = 0; // Filter type: None (simplest)
            int rowStart = y * width * 4;
            System.Array.Copy(rgba, rowStart, rawData, pos, width * 4);
            pos += width * 4;
        }

        // Compress with zlib (2-byte header + DEFLATE + 4-byte Adler32)
        var compressedData = CompressZlib(rawData);
        WriteChunk(output, "IDAT", writer => writer.Write(compressedData));

        // IEND chunk
        WriteChunk(output, "IEND", _ => { });

        return output.ToArray();
    }

    /// <summary>
    /// Encode RGB pixel data (no alpha) as a PNG file.
    /// </summary>
    public static byte[] EncodeRGB(byte[] rgb, int width, int height)
    {
        using var output = new MemoryStream();

        output.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        WriteChunk(output, "IHDR", writer =>
        {
            WriteBE32(writer, width);
            WriteBE32(writer, height);
            writer.WriteByte(8);  // Bit depth
            writer.WriteByte(2);  // Color type: RGB
            writer.WriteByte(0);
            writer.WriteByte(0);
            writer.WriteByte(0);
        });

        var rawData = new byte[height * (1 + width * 3)];
        int pos = 0;
        for (int y = 0; y < height; y++)
        {
            rawData[pos++] = 0;
            System.Array.Copy(rgb, y * width * 3, rawData, pos, width * 3);
            pos += width * 3;
        }

        var compressedData = CompressZlib(rawData);
        WriteChunk(output, "IDAT", writer => writer.Write(compressedData));
        WriteChunk(output, "IEND", _ => { });

        return output.ToArray();
    }

    /// <summary>
    /// Encode a grayscale float array [0,1] as a PNG with an optional colormap.
    /// Useful for saving depth maps.
    /// </summary>
    public static byte[] EncodeGrayscale(float[] data, int width, int height, string? colormap = null)
    {
        if (colormap != null)
        {
            var rgba = DepthColorMaps.ApplyColorMap(data, width, height, colormap);
            return Encode(rgba, width, height);
        }

        // Simple grayscale
        var gray = new byte[width * height];
        for (int i = 0; i < data.Length && i < gray.Length; i++)
        {
            gray[i] = (byte)Math.Clamp((int)(data[i] * 255 + 0.5f), 0, 255);
        }

        using var output = new MemoryStream();
        output.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        WriteChunk(output, "IHDR", writer =>
        {
            WriteBE32(writer, width);
            WriteBE32(writer, height);
            writer.WriteByte(8);
            writer.WriteByte(0); // Grayscale
            writer.WriteByte(0);
            writer.WriteByte(0);
            writer.WriteByte(0);
        });

        var rawData = new byte[height * (1 + width)];
        int pos = 0;
        for (int y = 0; y < height; y++)
        {
            rawData[pos++] = 0;
            System.Array.Copy(gray, y * width, rawData, pos, width);
            pos += width;
        }

        var compressedData = CompressZlib(rawData);
        WriteChunk(output, "IDAT", writer => writer.Write(compressedData));
        WriteChunk(output, "IEND", _ => { });

        return output.ToArray();
    }

    // ── Internal helpers ──

    private static void WriteChunk(Stream output, string type, Action<MemoryStream> writeData)
    {
        using var dataStream = new MemoryStream();
        writeData(dataStream);
        var data = dataStream.ToArray();

        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);

        // Length (4 bytes, big-endian)
        WriteBE32(output, data.Length);

        // Type (4 bytes)
        output.Write(typeBytes, 0, 4);

        // Data
        if (data.Length > 0)
            output.Write(data, 0, data.Length);

        // CRC32 over type + data
        uint crc = Crc32(typeBytes, data);
        WriteBE32(output, (int)crc);
    }

    private static byte[] CompressZlib(byte[] data)
    {
        using var output = new MemoryStream();

        // Zlib header: CMF=0x78 (deflate, window 32K), FLG=0x01 (no dict, check bits)
        output.WriteByte(0x78);
        output.WriteByte(0x01);

        // DEFLATE compressed data
        using (var deflate = new DeflateStream(output, CompressionLevel.Fastest, leaveOpen: true))
        {
            deflate.Write(data, 0, data.Length);
        }

        // Adler32 checksum
        uint adler = Adler32(data);
        output.WriteByte((byte)(adler >> 24));
        output.WriteByte((byte)(adler >> 16));
        output.WriteByte((byte)(adler >> 8));
        output.WriteByte((byte)adler);

        return output.ToArray();
    }

    private static void WriteBE32(Stream s, int value)
    {
        s.WriteByte((byte)(value >> 24));
        s.WriteByte((byte)(value >> 16));
        s.WriteByte((byte)(value >> 8));
        s.WriteByte((byte)value);
    }

    private static void WriteBE32(MemoryStream s, int value)
    {
        s.WriteByte((byte)(value >> 24));
        s.WriteByte((byte)(value >> 16));
        s.WriteByte((byte)(value >> 8));
        s.WriteByte((byte)value);
    }

    // ── CRC32 (PNG uses ISO 3309 / ITU-T V.42 polynomial) ──

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    private static uint Crc32(byte[] type, byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in type)
            crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        foreach (byte b in data)
            crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }

    private static uint Adler32(byte[] data)
    {
        uint a = 1, b = 0;
        foreach (byte d in data)
        {
            a = (a + d) % 65521;
            b = (b + a) % 65521;
        }
        return (b << 16) | a;
    }
}
