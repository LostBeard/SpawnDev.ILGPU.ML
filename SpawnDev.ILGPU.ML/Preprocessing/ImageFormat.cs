namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Image format detection from magic bytes.
/// Identifies PNG, JPEG, WebP, GIF, BMP, and TIFF without full decoding.
/// </summary>
public static class ImageFormat
{
    public enum Format
    {
        Unknown, JPEG, PNG, WebP, GIF, BMP, TIFF, ICO, AVIF
    }

    /// <summary>Detect image format from the first bytes of a file.</summary>
    public static Format Detect(byte[] data)
    {
        if (data.Length < 4) return Format.Unknown;
        return Detect(data.AsSpan());
    }

    /// <summary>Detect image format from span.</summary>
    public static Format Detect(ReadOnlySpan<byte> data)
    {
        if (data.Length < 4) return Format.Unknown;

        // JPEG: FF D8 FF
        if (data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF)
            return Format.JPEG;

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47)
            return Format.PNG;

        // WebP: RIFF....WEBP
        if (data.Length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F'
            && data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P')
            return Format.WebP;

        // GIF: GIF87a or GIF89a
        if (data[0] == 'G' && data[1] == 'I' && data[2] == 'F')
            return Format.GIF;

        // BMP: BM
        if (data[0] == 'B' && data[1] == 'M')
            return Format.BMP;

        // TIFF: II (little-endian) or MM (big-endian)
        if ((data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0) ||
            (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == 42))
            return Format.TIFF;

        // AVIF/HEIF: ....ftypavif or ....ftypheic
        if (data.Length >= 12 && data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p')
            return Format.AVIF;

        return Format.Unknown;
    }

    /// <summary>Get the MIME type for a format.</summary>
    public static string GetMimeType(Format format) => format switch
    {
        Format.JPEG => "image/jpeg",
        Format.PNG => "image/png",
        Format.WebP => "image/webp",
        Format.GIF => "image/gif",
        Format.BMP => "image/bmp",
        Format.TIFF => "image/tiff",
        Format.AVIF => "image/avif",
        _ => "application/octet-stream",
    };

    /// <summary>Get the file extension for a format.</summary>
    public static string GetExtension(Format format) => format switch
    {
        Format.JPEG => ".jpg",
        Format.PNG => ".png",
        Format.WebP => ".webp",
        Format.GIF => ".gif",
        Format.BMP => ".bmp",
        Format.TIFF => ".tiff",
        Format.AVIF => ".avif",
        _ => ".bin",
    };

    /// <summary>
    /// Get image dimensions without full decoding.
    /// Returns (width, height) or (0, 0) if unable to determine.
    /// </summary>
    public static (int Width, int Height) GetDimensions(byte[] data)
    {
        var format = Detect(data);
        return format switch
        {
            Format.PNG => PngDecoder.GetDimensions(data),
            Format.JPEG => JpegDecoder.GetDimensions(data),
            Format.WebP => WebPDecoder.GetDimensions(data),
            _ => (0, 0),
        };
    }
}
