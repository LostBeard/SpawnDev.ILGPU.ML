using System.Text.RegularExpressions;

namespace UniRigHarness;

/// <summary>Minimal reader for the .npy files tools/unirig-harness/reference writes (little-endian f4 / i4 / i8, C order).</summary>
internal static class Npy
{
    public static (T[] Data, int[] Shape) Load<T>(string path) where T : unmanaged
    {
        using var f = File.OpenRead(path);
        var magic = new byte[8]; f.ReadExactly(magic);
        if (magic[0] != 0x93 || magic[1] != (byte)'N') throw new InvalidDataException($"{path} is not an .npy file");
        int hlen = magic[6] == 1 ? BitConverter.ToUInt16(Read(f, 2), 0) : BitConverter.ToInt32(Read(f, 4), 0);
        var header = System.Text.Encoding.ASCII.GetString(Read(f, hlen));
        var descr = Regex.Match(header, @"'descr':\s*'([^']+)'").Groups[1].Value;
        if (header.Contains("'fortran_order': True")) throw new NotSupportedException("fortran order");
        var shape = Regex.Match(header, @"'shape':\s*\(([^)]*)\)").Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        long count = shape.Aggregate(1L, (a, b) => a * b);
        string want = typeof(T) == typeof(float) ? "<f4" : typeof(T) == typeof(int) ? "<i4" : typeof(T) == typeof(long) ? "<i8" : throw new NotSupportedException(typeof(T).Name);
        if (descr != want) throw new InvalidDataException($"{path}: dtype {descr}, expected {want}");
        var data = new T[count];
        f.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()));
        return (data, shape);
    }

    private static byte[] Read(Stream s, int n) { var b = new byte[n]; s.ReadExactly(b); return b; }
}
