using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.ILGPU.ML.Preprocessing;

/// <summary>
/// Shared RGBA → device upload for vision pipelines. Keeps the three entry shapes
/// (<c>int[]</c>, JS <see cref="TypedArray"/>, already-on-device view) in one place so every
/// pipeline cannot drift into a managed copy on the browser path.
/// </summary>
public static class RgbaUpload
{
    /// <summary>
    /// Allocates a packed-RGBA <c>int</c> buffer and fills it from managed pixels (desktop / oracle path).
    /// Caller owns the buffer.
    /// </summary>
    public static MemoryBuffer1D<int, Stride1D.Dense> FromManaged(
        Accelerator accelerator, int[] rgbaPixels, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgbaPixels);
        int expected = checked(width * height);
        if (rgbaPixels.Length < expected)
            throw new ArgumentException(
                $"RGBA int[] length {rgbaPixels.Length} is shorter than width*height ({expected}).",
                nameof(rgbaPixels));
        return accelerator.Allocate1D(rgbaPixels);
    }

    /// <summary>
    /// Allocates a packed-RGBA <c>int</c> buffer and fills it from a JS typed array via
    /// <see cref="MediaInterop.UploadToDevice{T}"/> — bytes never enter the .NET managed heap.
    /// Caller owns the buffer. Throws on non-browser accelerators.
    /// </summary>
    public static MemoryBuffer1D<int, Stride1D.Dense> FromTypedArray(
        Accelerator accelerator, TypedArray rgbaBytes, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgbaBytes);
        int expected = checked(width * height);
        long byteLen = rgbaBytes.ByteLength;
        long expectedBytes = (long)expected * 4;
        if (byteLen < expectedBytes)
            throw new ArgumentException(
                $"RGBA TypedArray byteLength {byteLen} is shorter than width*height*4 ({expectedBytes}).",
                nameof(rgbaBytes));

        var buf = accelerator.Allocate1D<int>(expected);
        try
        {
            MediaInterop.UploadToDevice(rgbaBytes, buf);
            return buf;
        }
        catch
        {
            buf.Dispose();
            throw;
        }
    }
}
