namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// WebP decoder utilities. Extracts dimensions and metadata from WebP files.
/// Supports VP8 (lossy), VP8L (lossless), and extended (VP8X) formats.
/// For pixel decoding, use canvas via MediaInterop (WebP decoding is complex).
/// </summary>
public static class WebPDecoder
{
    /// <summary>Get WebP image dimensions without decoding pixels.</summary>
    public static (int Width, int Height) GetDimensions(byte[] data)
    {
        if (data.Length < 16) return (0, 0);
        // RIFF header: "RIFF" + size + "WEBP"
        if (data[0] != 'R' || data[8] != 'W') return (0, 0);

        // Check chunk type at offset 12
        string chunk = System.Text.Encoding.ASCII.GetString(data, 12, 4);

        return chunk switch
        {
            "VP8 " => GetVP8Dimensions(data, 12),    // Lossy
            "VP8L" => GetVP8LDimensions(data, 12),   // Lossless
            "VP8X" => GetVP8XDimensions(data, 12),   // Extended
            _ => (0, 0),
        };
    }

    /// <summary>Get detailed WebP info including format variant and metadata.</summary>
    public static WebPInfo GetInfo(byte[] data)
    {
        var info = new WebPInfo();
        var (w, h) = GetDimensions(data);
        info.Width = w;
        info.Height = h;

        if (data.Length < 16) return info;

        string chunk = System.Text.Encoding.ASCII.GetString(data, 12, 4);
        info.Format = chunk.TrimEnd(' ') switch
        {
            "VP8" => WebPFormat.Lossy,
            "VP8L" => WebPFormat.Lossless,
            "VP8X" => WebPFormat.Extended,
            _ => WebPFormat.Unknown,
        };

        // For extended format, check feature flags
        if (info.Format == WebPFormat.Extended && data.Length >= 30)
        {
            byte flags = data[20];
            info.HasAnimation = (flags & 0x02) != 0;
            info.HasAlpha = (flags & 0x10) != 0;
            info.HasExif = (flags & 0x08) != 0;
            info.HasXMP = (flags & 0x04) != 0;
            info.HasICC = (flags & 0x20) != 0;
        }

        // Try to read EXIF
        info.Exif = ExifReader.Read(data);

        return info;
    }

    // VP8 (lossy): dimensions in frame header
    private static (int, int) GetVP8Dimensions(byte[] data, int chunkOffset)
    {
        // Chunk header: [4 type][4 size]
        int dataStart = chunkOffset + 8;
        if (dataStart + 10 > data.Length) return (0, 0);

        // VP8 bitstream starts with 3-byte frame tag
        // Check for keyframe: bit 0 of first byte == 0
        if ((data[dataStart] & 1) != 0) return (0, 0); // Not a keyframe

        // Skip 3-byte frame tag + 3-byte start code (9D 01 2A)
        int sizeOffset = dataStart + 6;
        if (sizeOffset + 4 > data.Length) return (0, 0);
        if (data[dataStart + 3] != 0x9D || data[dataStart + 4] != 0x01 || data[dataStart + 5] != 0x2A)
            return (0, 0);

        int width = (data[sizeOffset] | (data[sizeOffset + 1] << 8)) & 0x3FFF;
        int height = (data[sizeOffset + 2] | (data[sizeOffset + 3] << 8)) & 0x3FFF;
        return (width, height);
    }

    // VP8L (lossless): dimensions in bitstream header
    private static (int, int) GetVP8LDimensions(byte[] data, int chunkOffset)
    {
        int dataStart = chunkOffset + 8;
        if (dataStart + 5 > data.Length) return (0, 0);

        // VP8L signature byte: 0x2F
        if (data[dataStart] != 0x2F) return (0, 0);

        // Width and height packed in 4 bytes starting at dataStart+1
        uint bits = (uint)(data[dataStart + 1] | (data[dataStart + 2] << 8) |
                          (data[dataStart + 3] << 16) | (data[dataStart + 4] << 24));
        int width = (int)(bits & 0x3FFF) + 1;
        int height = (int)((bits >> 14) & 0x3FFF) + 1;
        return (width, height);
    }

    // VP8X (extended): dimensions in VP8X chunk
    private static (int, int) GetVP8XDimensions(byte[] data, int chunkOffset)
    {
        int dataStart = chunkOffset + 8;
        if (dataStart + 10 > data.Length) return (0, 0);

        // Canvas size at bytes 4-9 of VP8X data (24-bit width-1, 24-bit height-1)
        int width = (data[dataStart + 4] | (data[dataStart + 5] << 8) | (data[dataStart + 6] << 16)) + 1;
        int height = (data[dataStart + 7] | (data[dataStart + 8] << 8) | (data[dataStart + 9] << 16)) + 1;
        return (width, height);
    }
}

public class WebPInfo
{
    public int Width { get; set; }
    public int Height { get; set; }
    public WebPFormat Format { get; set; }
    public bool HasAnimation { get; set; }
    public bool HasAlpha { get; set; }
    public bool HasExif { get; set; }
    public bool HasXMP { get; set; }
    public bool HasICC { get; set; }
    public ExifData? Exif { get; set; }
}

public enum WebPFormat
{
    Unknown, Lossy, Lossless, Extended
}
