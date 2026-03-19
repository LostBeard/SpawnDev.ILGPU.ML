namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// JPEG decoder utilities. Extracts dimensions and metadata without full pixel decoding.
/// For pixel decoding in browser, use canvas via MediaInterop.
/// </summary>
public static class JpegDecoder
{
    /// <summary>Get JPEG image dimensions from file bytes without decoding pixels.</summary>
    public static (int Width, int Height) GetDimensions(byte[] data)
    {
        if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8) return (0, 0);

        int pos = 2;
        while (pos < data.Length - 4)
        {
            if (data[pos] != 0xFF) { pos++; continue; }
            byte marker = data[pos + 1];
            pos += 2;

            // SOF markers (Start Of Frame) contain dimensions
            if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC)
            {
                if (pos + 7 > data.Length) return (0, 0);
                // Skip length (2 bytes) and precision (1 byte)
                int height = (data[pos + 3] << 8) | data[pos + 4];
                int width = (data[pos + 5] << 8) | data[pos + 6];
                return (width, height);
            }

            if (marker == 0xD9 || marker == 0xDA) break; // EOI or SOS

            // Skip this marker's data
            if (pos + 2 > data.Length) break;
            int length = (data[pos] << 8) | data[pos + 1];
            pos += length;
        }

        return (0, 0);
    }

    /// <summary>
    /// Get all JPEG metadata including EXIF orientation.
    /// Returns dimensions from SOF marker + EXIF data from APP1.
    /// </summary>
    public static JpegInfo GetInfo(byte[] data)
    {
        var info = new JpegInfo();
        var (w, h) = GetDimensions(data);
        info.Width = w;
        info.Height = h;

        var exif = ExifReader.Read(data);
        info.Exif = exif;

        if (exif != null && exif.NeedsOrientationCorrection)
        {
            var (ow, oh) = exif.OrientedDimensions;
            if (ow > 0 && oh > 0)
            {
                info.OrientedWidth = ow;
                info.OrientedHeight = oh;
            }
        }
        else
        {
            info.OrientedWidth = w;
            info.OrientedHeight = h;
        }

        return info;
    }
}

public class JpegInfo
{
    public int Width { get; set; }
    public int Height { get; set; }
    public int OrientedWidth { get; set; }
    public int OrientedHeight { get; set; }
    public ExifData? Exif { get; set; }
}
