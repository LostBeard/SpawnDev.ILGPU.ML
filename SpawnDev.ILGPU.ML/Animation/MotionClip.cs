using System.IO.Compression;
using System.Text.RegularExpressions;

namespace SpawnDev.ILGPU.ML.Animation;

/// <summary>
/// A motion clip for the 77-joint <see cref="SomaSkeleton"/>, as NVIDIA Kimodo writes it (and as image-to-3dlab packs its preset moves): per frame the world rotation of
/// every joint away from SOMA's T-pose, the T-pose-relative frame-0 joint positions and the hips track.
/// <code>
/// quats    [T, 77, 4] float16   (w, x, y, z)
/// joints0  [77, 3]    float32   posed joint positions of frame 0
/// hips     [T, 3]     float32   hips position per frame
/// fps      scalar     float32
/// </code>
/// </summary>
public sealed class MotionClip
{
    public string Name { get; init; } = "";
    public double Fps { get; }
    public int Frames { get; }
    /// <summary>[Frames * 77] world rotations away from the T-pose.</summary>
    public Mat3[] Rotations { get; }
    /// <summary>[77] frame-0 joint positions (SOMA space, Y up).</summary>
    public Vec3[] Joints0 { get; }
    /// <summary>[Frames] hips position (SOMA space).</summary>
    public Vec3[] Hips { get; }

    public MotionClip(double fps, Mat3[] rotations, Vec3[] joints0, Vec3[] hips)
    {
        if (joints0.Length != SomaSkeleton.Count) throw new ArgumentException($"expected {SomaSkeleton.Count} joints");
        Frames = hips.Length;
        if (rotations.Length != Frames * SomaSkeleton.Count) throw new ArgumentException("rotations must be frames x 77");
        Fps = fps; Rotations = rotations; Joints0 = joints0; Hips = hips;
    }

    public Mat3 Rotation(int frame, int joint) => Rotations[frame * SomaSkeleton.Count + joint];
    public double Seconds => Frames / Fps;

    /// <summary>Read a packed clip (<c>.npz</c> with quats, joints0, hips, fps).</summary>
    public static MotionClip FromNpz(byte[] bytes, string name = "") => FromNpz(new MemoryStream(bytes), name);

    public static MotionClip FromNpz(Stream stream, string name = "")
    {
        using var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
        (float[] Data, int[] Shape) Array(string entry)
        {
            var e = zip.GetEntry(entry + ".npy") ?? throw new InvalidDataException($"the clip has no '{entry}' array");
            using var s = e.Open(); using var ms = new MemoryStream(); s.CopyTo(ms);
            return ReadNpy(ms.ToArray());
        }
        var (q, qs) = Array("quats"); var (j0, js) = Array("joints0"); var (hp, hs) = Array("hips"); var (fp, _) = Array("fps");
        if (qs.Length != 3 || qs[1] != SomaSkeleton.Count || qs[2] != 4) throw new InvalidDataException($"quats has shape ({string.Join(",", qs)}), expected (T, 77, 4)");
        int T = qs[0];
        var rot = new Mat3[T * SomaSkeleton.Count];
        for (int i = 0; i < rot.Length; i++) rot[i] = Mat3.FromQuaternion(q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3], q[i * 4]);       // stored (w, x, y, z)
        var joints0 = new Vec3[SomaSkeleton.Count]; for (int i = 0; i < joints0.Length; i++) joints0[i] = new Vec3(j0[i * 3], j0[i * 3 + 1], j0[i * 3 + 2]);
        var hips = new Vec3[T]; for (int i = 0; i < T; i++) hips[i] = new Vec3(hp[i * 3], hp[i * 3 + 1], hp[i * 3 + 2]);
        return new MotionClip(fp[0], rot, joints0, hips) { Name = name };
    }

    /// <summary>A little-endian float16 / float32 / float64 .npy array as floats, with its shape.</summary>
    internal static (float[] Data, int[] Shape) ReadNpy(byte[] b)
    {
        if (b.Length < 10 || b[0] != 0x93 || b[1] != (byte)'N') throw new InvalidDataException("not an .npy array");
        int hlen = b[6] == 1 ? BitConverter.ToUInt16(b, 8) : BitConverter.ToInt32(b, 8), off = (b[6] == 1 ? 10 : 12) + hlen;
        var header = System.Text.Encoding.ASCII.GetString(b, off - hlen, hlen);
        var descr = Regex.Match(header, @"'descr':\s*'([^']+)'").Groups[1].Value;
        if (header.Contains("'fortran_order': True")) throw new InvalidDataException("fortran-ordered arrays are not supported");
        var shape = Regex.Match(header, @"'shape':\s*\(([^)]*)\)").Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        long n = shape.Aggregate(1L, (a, c) => a * c);
        var data = new float[n];
        switch (descr)
        {
            case "<f2": for (int i = 0; i < n; i++) data[i] = (float)BitConverter.ToHalf(b, off + i * 2); break;
            case "<f4": for (int i = 0; i < n; i++) data[i] = BitConverter.ToSingle(b, off + i * 4); break;
            case "<f8": for (int i = 0; i < n; i++) data[i] = (float)BitConverter.ToDouble(b, off + i * 8); break;
            default: throw new InvalidDataException($"unsupported dtype {descr}");
        }
        return (data, shape);
    }
}
