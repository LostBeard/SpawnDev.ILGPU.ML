using System.IO.Compression;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// PNG decoder utilities. Extracts dimensions, metadata, and chunks.
/// Can decode pixels for simple cases (8-bit RGBA/RGB, no interlacing).
/// For complex cases or browser usage, use canvas via MediaInterop.
/// </summary>
public static class PngDecoder
{
    /// <summary>Get PNG image dimensions from the IHDR chunk.</summary>
    public static (int Width, int Height) GetDimensions(byte[] data)
    {
        if (data.Length < 24) return (0, 0);
        // PNG signature: 89 50 4E 47 0D 0A 1A 0A
        if (data[0] != 0x89 || data[1] != 0x50) return (0, 0);

        // IHDR is always the first chunk (at offset 8)
        // Chunk: [4 bytes length][4 bytes type][data][4 bytes CRC]
        int width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
        int height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];
        return (width, height);
    }

    /// <summary>Get detailed PNG info from header and metadata chunks.</summary>
    public static PngInfo GetInfo(byte[] data)
    {
        var info = new PngInfo();
        var (w, h) = GetDimensions(data);
        info.Width = w;
        info.Height = h;

        if (data.Length < 29) return info;

        // IHDR data starts at offset 16
        info.BitDepth = data[24];
        info.ColorType = data[25];
        info.CompressionMethod = data[26];
        info.FilterMethod = data[27];
        info.InterlaceMethod = data[28];

        // Parse text chunks for metadata
        EnumerateChunks(data, (type, chunkData) =>
        {
            switch (type)
            {
                case "tEXt":
                    var (key, val) = ParseTextChunk(chunkData);
                    info.TextMetadata[key] = val;
                    break;
                case "iTXt":
                    var (iKey, iVal) = ParseInternationalTextChunk(chunkData);
                    info.TextMetadata[iKey] = iVal;
                    break;
                case "eXIf":
                    info.Exif = ExifReader.Read(PrependTiffHeader(chunkData));
                    break;
                case "pHYs":
                    if (chunkData.Length >= 9)
                    {
                        info.PixelsPerUnitX = ReadBE32(chunkData, 0);
                        info.PixelsPerUnitY = ReadBE32(chunkData, 4);
                        info.PhysUnitIsMeters = chunkData[8] == 1;
                    }
                    break;
            }
        });

        return info;
    }

    /// <summary>
    /// Decode PNG pixels to RGBA byte array.
    /// Supports 8-bit RGB and RGBA, non-interlaced only.
    /// For interlaced PNGs or other formats, use canvas via MediaInterop.
    /// </summary>
    public static byte[]? DecodePixels(byte[] data)
    {
        var info = GetInfo(data);
        if (info.InterlaceMethod != 0) return null; // Interlaced not supported
        if (info.BitDepth != 8) return null; // Only 8-bit supported
        if (info.ColorType != 2 && info.ColorType != 6) return null; // RGB or RGBA only

        int channels = info.ColorType == 6 ? 4 : 3;
        int stride = info.Width * channels + 1; // +1 for filter byte per row

        // Collect all IDAT chunks
        using var compressedStream = new MemoryStream();
        EnumerateChunks(data, (type, chunkData) =>
        {
            if (type == "IDAT")
                compressedStream.Write(chunkData, 0, chunkData.Length);
        });

        // Decompress (zlib = 2-byte header + DEFLATE)
        compressedStream.Position = 2; // Skip zlib header
        using var deflateStream = new DeflateStream(compressedStream, CompressionMode.Decompress);
        using var decompressed = new MemoryStream();
        deflateStream.CopyTo(decompressed);
        var rawData = decompressed.ToArray();

        if (rawData.Length < stride * info.Height) return null;

        // Unfilter and convert to RGBA
        var rgba = new byte[info.Width * info.Height * 4];
        var prevRow = new byte[info.Width * channels];
        var currentRow = new byte[info.Width * channels];

        for (int y = 0; y < info.Height; y++)
        {
            int rowStart = y * stride;
            byte filterType = rawData[rowStart];

            // Extract row data (without filter byte)
            System.Array.Copy(rawData, rowStart + 1, currentRow, 0, info.Width * channels);

            // Apply PNG filter
            ApplyFilter(filterType, currentRow, prevRow, channels);

            // Write to RGBA output
            for (int x = 0; x < info.Width; x++)
            {
                int srcIdx = x * channels;
                int dstIdx = (y * info.Width + x) * 4;
                rgba[dstIdx + 0] = currentRow[srcIdx + 0]; // R
                rgba[dstIdx + 1] = currentRow[srcIdx + 1]; // G
                rgba[dstIdx + 2] = currentRow[srcIdx + 2]; // B
                rgba[dstIdx + 3] = channels == 4 ? currentRow[srcIdx + 3] : (byte)255; // A
            }

            // Swap prev/current
            (prevRow, currentRow) = (currentRow, prevRow);
        }

        return rgba;
    }

    // ── PNG filter types ──

    private static void ApplyFilter(byte filterType, byte[] row, byte[] prev, int bpp)
    {
        switch (filterType)
        {
            case 0: break; // None
            case 1: // Sub
                for (int i = bpp; i < row.Length; i++)
                    row[i] = (byte)(row[i] + row[i - bpp]);
                break;
            case 2: // Up
                for (int i = 0; i < row.Length; i++)
                    row[i] = (byte)(row[i] + prev[i]);
                break;
            case 3: // Average
                for (int i = 0; i < row.Length; i++)
                {
                    int a = i >= bpp ? row[i - bpp] : 0;
                    row[i] = (byte)(row[i] + (a + prev[i]) / 2);
                }
                break;
            case 4: // Paeth
                for (int i = 0; i < row.Length; i++)
                {
                    int a = i >= bpp ? row[i - bpp] : 0;
                    int b = prev[i];
                    int c = i >= bpp ? prev[i - bpp] : 0;
                    row[i] = (byte)(row[i] + PaethPredictor(a, b, c));
                }
                break;
        }
    }

    private static int PaethPredictor(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a);
        int pb = Math.Abs(p - b);
        int pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        if (pb <= pc) return b;
        return c;
    }

    // ── Chunk enumeration ──

    private static void EnumerateChunks(byte[] data, Action<string, byte[]> handler)
    {
        int pos = 8; // Skip PNG signature
        while (pos + 12 <= data.Length)
        {
            int length = ReadBE32(data, pos);
            string type = System.Text.Encoding.ASCII.GetString(data, pos + 4, 4);
            pos += 8;

            if (pos + length > data.Length) break;
            var chunkData = new byte[length];
            System.Array.Copy(data, pos, chunkData, 0, length);
            handler(type, chunkData);

            pos += length + 4; // Skip data + CRC
            if (type == "IEND") break;
        }
    }

    private static (string Key, string Value) ParseTextChunk(byte[] data)
    {
        int nullPos = System.Array.IndexOf(data, (byte)0);
        if (nullPos < 0) return ("", "");
        string key = System.Text.Encoding.ASCII.GetString(data, 0, nullPos);
        string value = System.Text.Encoding.Latin1.GetString(data, nullPos + 1, data.Length - nullPos - 1);
        return (key, value);
    }

    private static (string Key, string Value) ParseInternationalTextChunk(byte[] data)
    {
        int nullPos = System.Array.IndexOf(data, (byte)0);
        if (nullPos < 0) return ("", "");
        string key = System.Text.Encoding.ASCII.GetString(data, 0, nullPos);
        // Skip compression flag, compression method, language tag, translated keyword
        int textStart = nullPos + 1;
        for (int nulls = 0; nulls < 3 && textStart < data.Length; textStart++)
            if (data[textStart] == 0) nulls++;
        string value = System.Text.Encoding.UTF8.GetString(data, textStart, data.Length - textStart);
        return (key, value);
    }

    private static byte[] PrependTiffHeader(byte[] exifData)
    {
        // For eXIf chunks that contain raw TIFF data without "Exif\0\0" prefix
        if (exifData.Length >= 2 && (exifData[0] == 'I' || exifData[0] == 'M'))
            return exifData; // Already TIFF format
        return exifData;
    }

    private static int ReadBE32(byte[] data, int offset)
    {
        return (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3];
    }
}

public class PngInfo
{
    public int Width { get; set; }
    public int Height { get; set; }
    public int BitDepth { get; set; }
    public int ColorType { get; set; } // 0=Gray, 2=RGB, 3=Indexed, 4=GrayAlpha, 6=RGBA
    public int CompressionMethod { get; set; }
    public int FilterMethod { get; set; }
    public int InterlaceMethod { get; set; } // 0=None, 1=Adam7
    public Dictionary<string, string> TextMetadata { get; set; } = new();
    public ExifData? Exif { get; set; }
    public int PixelsPerUnitX { get; set; }
    public int PixelsPerUnitY { get; set; }
    public bool PhysUnitIsMeters { get; set; }
    public int Channels => ColorType switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => 0 };
    public string ColorTypeDescription => ColorType switch { 0 => "Grayscale", 2 => "RGB", 3 => "Indexed", 4 => "GrayAlpha", 6 => "RGBA", _ => "Unknown" };
}
