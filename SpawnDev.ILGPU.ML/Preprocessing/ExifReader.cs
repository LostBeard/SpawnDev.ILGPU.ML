namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// EXIF metadata reader for JPEG, WebP, and TIFF images.
/// Extracts orientation, dimensions, camera info, GPS coordinates, and timestamps.
/// Pure C# — no external dependencies.
/// </summary>
public static class ExifReader
{
    /// <summary>
    /// Read EXIF metadata from image file bytes.
    /// Supports JPEG (APP1 marker), WebP (EXIF chunk), and TIFF (header).
    /// </summary>
    public static ExifData? Read(byte[] data)
    {
        var format = ImageFormat.Detect(data);
        return format switch
        {
            ImageFormat.Format.JPEG => ReadFromJpeg(data),
            ImageFormat.Format.WebP => ReadFromWebP(data),
            ImageFormat.Format.TIFF => ReadFromTiff(data, 0),
            _ => null,
        };
    }

    /// <summary>
    /// Get just the EXIF orientation (1-8). Returns 1 (normal) if not found.
    /// This is the most commonly needed EXIF value for ML preprocessing.
    /// </summary>
    public static int GetOrientation(byte[] data)
    {
        var exif = Read(data);
        return exif?.Orientation ?? 1;
    }

    /// <summary>
    /// Check if the image needs rotation/flip based on EXIF orientation.
    /// </summary>
    public static bool NeedsOrientationCorrection(byte[] data) => GetOrientation(data) != 1;

    // ──────────────────────────────────────────────
    //  JPEG EXIF extraction
    // ──────────────────────────────────────────────

    private static ExifData? ReadFromJpeg(byte[] data)
    {
        if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8) return null;

        int pos = 2;
        while (pos < data.Length - 4)
        {
            if (data[pos] != 0xFF) break;
            byte marker = data[pos + 1];
            pos += 2;

            if (marker == 0xD9) break; // EOI
            if (marker == 0xDA) break; // SOS — start of scan data

            int length = (data[pos] << 8) | data[pos + 1];
            if (length < 2) break;

            // APP1 marker (0xE1) with "Exif\0\0" header
            if (marker == 0xE1 && length > 8)
            {
                if (data[pos + 2] == 'E' && data[pos + 3] == 'x' && data[pos + 4] == 'i' &&
                    data[pos + 5] == 'f' && data[pos + 6] == 0 && data[pos + 7] == 0)
                {
                    int tiffStart = pos + 8;
                    return ReadFromTiff(data, tiffStart);
                }
            }

            pos += length;
        }

        return null;
    }

    // ──────────────────────────────────────────────
    //  WebP EXIF extraction
    // ──────────────────────────────────────────────

    private static ExifData? ReadFromWebP(byte[] data)
    {
        if (data.Length < 12) return null;

        int pos = 12; // Skip RIFF header + "WEBP"
        while (pos < data.Length - 8)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            int chunkSize = BitConverter.ToInt32(data, pos + 4); // Little-endian
            pos += 8;

            if (chunkId == "EXIF" && chunkSize > 6)
            {
                // WebP EXIF chunk contains TIFF data (may have "Exif\0\0" prefix)
                int tiffStart = pos;
                if (data[pos] == 'E' && data[pos + 1] == 'x')
                    tiffStart += 6; // Skip "Exif\0\0"
                return ReadFromTiff(data, tiffStart);
            }

            pos += chunkSize;
            if (chunkSize % 2 == 1) pos++; // Chunks are padded to even size
        }

        return null;
    }

    // ──────────────────────────────────────────────
    //  TIFF/EXIF IFD parser (shared by JPEG and WebP)
    // ──────────────────────────────────────────────

    private static ExifData? ReadFromTiff(byte[] data, int tiffStart)
    {
        if (tiffStart + 8 > data.Length) return null;

        // Byte order: II = little-endian, MM = big-endian
        bool littleEndian = data[tiffStart] == 'I' && data[tiffStart + 1] == 'I';
        bool bigEndian = data[tiffStart] == 'M' && data[tiffStart + 1] == 'M';
        if (!littleEndian && !bigEndian) return null;

        // Verify magic number 42
        int magic = ReadUInt16(data, tiffStart + 2, littleEndian);
        if (magic != 42) return null;

        // Offset to first IFD
        int ifdOffset = (int)ReadUInt32(data, tiffStart + 4, littleEndian);

        var exif = new ExifData();
        ReadIFD(data, tiffStart, tiffStart + ifdOffset, littleEndian, exif, false);

        return exif;
    }

    private static void ReadIFD(byte[] data, int tiffStart, int ifdPos, bool le, ExifData exif, bool isExifIfd)
    {
        if (ifdPos + 2 > data.Length) return;

        int entryCount = ReadUInt16(data, ifdPos, le);
        int pos = ifdPos + 2;

        for (int i = 0; i < entryCount && pos + 12 <= data.Length; i++)
        {
            int tag = ReadUInt16(data, pos, le);
            int type = ReadUInt16(data, pos + 2, le);
            int count = (int)ReadUInt32(data, pos + 4, le);
            int valueOffset = pos + 8;

            // If value is larger than 4 bytes, it's stored at an offset
            int valueSize = GetTypeSize(type) * count;
            if (valueSize > 4)
            {
                valueOffset = tiffStart + (int)ReadUInt32(data, pos + 8, le);
            }

            if (valueOffset >= 0 && valueOffset < data.Length)
            {
                ProcessTag(data, tiffStart, tag, type, count, valueOffset, le, exif, isExifIfd);
            }

            pos += 12;
        }
    }

    private static void ProcessTag(byte[] data, int tiffStart, int tag, int type, int count, int valueOffset, bool le, ExifData exif, bool isExifIfd)
    {
        switch (tag)
        {
            // IFD0 tags
            case 0x0112: exif.Orientation = ReadUInt16(data, valueOffset, le); break; // Orientation
            case 0x010E: exif.ImageDescription = ReadAscii(data, valueOffset, count); break;
            case 0x010F: exif.Make = ReadAscii(data, valueOffset, count); break; // Camera make
            case 0x0110: exif.Model = ReadAscii(data, valueOffset, count); break; // Camera model
            case 0x0131: exif.Software = ReadAscii(data, valueOffset, count); break;
            case 0x0132: exif.DateTime = ReadAscii(data, valueOffset, count); break;
            case 0x013B: exif.Artist = ReadAscii(data, valueOffset, count); break;
            case 0x8298: exif.Copyright = ReadAscii(data, valueOffset, count); break;
            case 0x0100: exif.ImageWidth = (int)ReadUInt32OrShort(data, valueOffset, type, le); break;
            case 0x0101: exif.ImageHeight = (int)ReadUInt32OrShort(data, valueOffset, type, le); break;

            // ExifIFD pointer
            case 0x8769:
                int exifIfdOffset = (int)ReadUInt32(data, valueOffset, le);
                ReadIFD(data, tiffStart, tiffStart + exifIfdOffset, le, exif, true);
                break;

            // GPS IFD pointer
            case 0x8825:
                int gpsIfdOffset = (int)ReadUInt32(data, valueOffset, le);
                ReadGpsIFD(data, tiffStart, tiffStart + gpsIfdOffset, le, exif);
                break;

            // ExifIFD tags
            case 0x9003: exif.DateTimeOriginal = ReadAscii(data, valueOffset, count); break;
            case 0x9004: exif.DateTimeDigitized = ReadAscii(data, valueOffset, count); break;
            case 0x920A: exif.FocalLength = ReadRational(data, valueOffset, le); break;
            case 0xA405: exif.FocalLengthIn35mm = ReadUInt16(data, valueOffset, le); break;
            case 0x829A: exif.ExposureTime = ReadRational(data, valueOffset, le); break;
            case 0x829D: exif.FNumber = ReadRational(data, valueOffset, le); break;
            case 0x8827: exif.ISO = ReadUInt16(data, valueOffset, le); break;
            case 0xA002: exif.PixelXDimension = (int)ReadUInt32OrShort(data, valueOffset, type, le); break;
            case 0xA003: exif.PixelYDimension = (int)ReadUInt32OrShort(data, valueOffset, type, le); break;
            case 0xA001: exif.ColorSpace = ReadUInt16(data, valueOffset, le); break;
        }
    }

    private static void ReadGpsIFD(byte[] data, int tiffStart, int ifdPos, bool le, ExifData exif)
    {
        if (ifdPos + 2 > data.Length) return;
        int entryCount = ReadUInt16(data, ifdPos, le);
        int pos = ifdPos + 2;

        string? latRef = null, lonRef = null;
        double? lat = null, lon = null, alt = null;

        for (int i = 0; i < entryCount && pos + 12 <= data.Length; i++)
        {
            int tag = ReadUInt16(data, pos, le);
            int type = ReadUInt16(data, pos + 2, le);
            int count = (int)ReadUInt32(data, pos + 4, le);
            int valueOffset = pos + 8;
            int valueSize = GetTypeSize(type) * count;
            if (valueSize > 4)
                valueOffset = tiffStart + (int)ReadUInt32(data, pos + 8, le);

            if (valueOffset >= 0 && valueOffset < data.Length)
            {
                switch (tag)
                {
                    case 1: latRef = ReadAscii(data, valueOffset, count); break;
                    case 2: lat = ReadGpsCoordinate(data, valueOffset, le); break;
                    case 3: lonRef = ReadAscii(data, valueOffset, count); break;
                    case 4: lon = ReadGpsCoordinate(data, valueOffset, le); break;
                    case 6: alt = ReadRational(data, valueOffset, le); break;
                }
            }
            pos += 12;
        }

        if (lat.HasValue && lon.HasValue)
        {
            exif.GpsLatitude = latRef == "S" ? -lat.Value : lat.Value;
            exif.GpsLongitude = lonRef == "W" ? -lon.Value : lon.Value;
            exif.GpsAltitude = alt;
        }
    }

    // ──────────────────────────────────────────────
    //  Primitive readers
    // ──────────────────────────────────────────────

    private static int ReadUInt16(byte[] data, int offset, bool le)
    {
        if (offset + 2 > data.Length) return 0;
        return le ? (data[offset] | (data[offset + 1] << 8))
                  : ((data[offset] << 8) | data[offset + 1]);
    }

    private static uint ReadUInt32(byte[] data, int offset, bool le)
    {
        if (offset + 4 > data.Length) return 0;
        return le ? (uint)(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24))
                  : (uint)((data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]);
    }

    private static uint ReadUInt32OrShort(byte[] data, int offset, int type, bool le)
    {
        return type == 3 ? (uint)ReadUInt16(data, offset, le) : ReadUInt32(data, offset, le);
    }

    private static double ReadRational(byte[] data, int offset, bool le)
    {
        uint num = ReadUInt32(data, offset, le);
        uint den = ReadUInt32(data, offset + 4, le);
        return den == 0 ? 0 : (double)num / den;
    }

    private static double ReadGpsCoordinate(byte[] data, int offset, bool le)
    {
        double degrees = ReadRational(data, offset, le);
        double minutes = ReadRational(data, offset + 8, le);
        double seconds = ReadRational(data, offset + 16, le);
        return degrees + minutes / 60.0 + seconds / 3600.0;
    }

    private static string ReadAscii(byte[] data, int offset, int count)
    {
        if (offset + count > data.Length) count = data.Length - offset;
        if (count <= 0) return "";
        // Trim null terminator
        while (count > 0 && data[offset + count - 1] == 0) count--;
        return System.Text.Encoding.ASCII.GetString(data, offset, count);
    }

    private static int GetTypeSize(int type) => type switch
    {
        1 => 1,  // BYTE
        2 => 1,  // ASCII
        3 => 2,  // SHORT
        4 => 4,  // LONG
        5 => 8,  // RATIONAL
        6 => 1,  // SBYTE
        7 => 1,  // UNDEFINED
        8 => 2,  // SSHORT
        9 => 4,  // SLONG
        10 => 8, // SRATIONAL
        11 => 4, // FLOAT
        12 => 8, // DOUBLE
        _ => 1,
    };
}

/// <summary>
/// EXIF metadata from an image.
/// </summary>
public class ExifData
{
    // Orientation (1-8)
    // 1=Normal, 2=FlipH, 3=Rotate180, 4=FlipV
    // 5=Transpose, 6=Rotate90CW, 7=Transverse, 8=Rotate90CCW
    public int Orientation { get; set; } = 1;

    // Dimensions
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
    public int PixelXDimension { get; set; }
    public int PixelYDimension { get; set; }
    public int ColorSpace { get; set; }

    // Camera
    public string? Make { get; set; }
    public string? Model { get; set; }
    public string? Software { get; set; }
    public string? Artist { get; set; }
    public string? Copyright { get; set; }
    public string? ImageDescription { get; set; }

    // Exposure
    public double ExposureTime { get; set; }
    public double FNumber { get; set; }
    public int ISO { get; set; }
    public double FocalLength { get; set; }
    public int FocalLengthIn35mm { get; set; }

    // Timestamps
    public string? DateTime { get; set; }
    public string? DateTimeOriginal { get; set; }
    public string? DateTimeDigitized { get; set; }

    // GPS
    public double? GpsLatitude { get; set; }
    public double? GpsLongitude { get; set; }
    public double? GpsAltitude { get; set; }

    /// <summary>Whether the image needs rotation/flip based on orientation.</summary>
    public bool NeedsOrientationCorrection => Orientation != 1;

    /// <summary>Effective dimensions after applying EXIF orientation.</summary>
    public (int Width, int Height) OrientedDimensions
    {
        get
        {
            int w = PixelXDimension > 0 ? PixelXDimension : ImageWidth;
            int h = PixelYDimension > 0 ? PixelYDimension : ImageHeight;
            return Orientation >= 5 && Orientation <= 8 ? (h, w) : (w, h);
        }
    }
}
