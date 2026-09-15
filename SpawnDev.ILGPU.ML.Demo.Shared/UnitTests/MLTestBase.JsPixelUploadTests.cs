using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Prices the two ways browser pixels can reach a GPU buffer, and proves they produce the SAME bytes.
/// </summary>
/// <remarks>
/// 🔴 WHY THIS EXISTS (TJ, 2026-09-15, reading MediaInterop): every public method on <c>MediaInterop</c>
/// returns <c>byte[]</c> or <c>float[]</c>, and <c>FromImageData</c> - whose docstring calls it *"the fastest
/// path ... zero JavaScript overhead beyond the typed array read"* - is <c>data.ReadBytes()</c>, which lands
/// the whole pixel buffer on the .NET WASM managed heap. The read IS the cost. TJ: *"anything that uses those
/// should be keeping the data JS side and handing it to the accelerators as JS data... that would certainly
/// cause ML to be much slower in the browser than desktop."*
///
/// The zero-copy target already exists one layer down and says so in its own summary:
/// <c>IBrowserMemoryBuffer.CopyFromJS(TypedArray, long)</c> - *"Copies data from a JS TypedArray directly into
/// the GPU buffer without crossing into .NET managed memory. This is the zero-copy path for browser
/// backends."* Nothing in the media path uses it. Today's chain for a classified frame is:
///
///     canvas -> Uint8Array -> ReadBytes() -> byte[] -> repack to int[] -> Allocate1D -> GPU
///                             ^^^ crossing 1        ^^^ copy 2         ^^^ crossing 3
///
/// versus what it should be:   canvas -> Uint8Array -> CopyFromJS -> GPU
///
/// ⚠️ MEASURED, NOT ASSUMED. Twice on 2026-09-15 I promoted a real inefficiency to "the cause" without
/// measuring the backend that mattered. This reports ms on whatever lane it runs and asserts only what must
/// never differ: the resulting GPU bytes.
///
/// Read the numbers with <c>PMT_CONSOLE_LOG=PixelUp</c> - PMT summarises browser console lines away, which is
/// how a diagnostic runs every sweep and has its verdict discarded.
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>Writes 1f where the two packed-RGBA buffers differ, 0f where they agree. Compared as INTs:
    /// reinterpreting arbitrary pixel bytes as floats can produce NaN bit patterns, and NaN != NaN would fail
    /// a run that is actually correct. The 0/1 flags then reduce through the existing finite-check, so the
    /// readback stays at 3 floats instead of the whole frame.</summary>
    private static void PixelMismatchFlagImpl(Index1D i,
        ArrayView1D<int, Stride1D.Dense> a,
        ArrayView1D<int, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> flags)
        => flags[i] = a[i] == b[i] ? 0f : 1f;

    [TestMethod(Timeout = 180000)]
    public async Task JsPixelUpload_ZeroCopyMatchesManaged() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("browser-only: there is no JS heap to cross on a desktop lane");

        // 1280x720 RGBA - a webcam frame, the case the media path exists for. 3.69 MB.
        const int w = 1280, h = 720;
        const int pixels = w * h;
        const int bytes = pixels * 4;

        // Fixture: a JS-side Uint8Array standing in for what getImageData/VideoFrame hands back. Seeded from
        // .NET ONCE, outside both timed regions - the point under test is how the pixels LEAVE JS, not how
        // they got there. Every byte varies so a reordering or stride bug cannot hide in zeros.
        var seed = new byte[bytes];
        for (int i = 0; i < bytes; i++) seed[i] = (byte)((i * 7 + (i >> 9)) & 0xFF);
        using var src = new Uint8Array(bytes);
        src.Set(seed);

        // ── Path A: today's managed route (ReadBytes -> byte[] -> int[] -> Allocate1D) ──
        var swA = System.Diagnostics.Stopwatch.StartNew();
        var managedBytes = src.ReadBytes();                       // bulk crossing onto the WASM managed heap
        var packed = new int[pixels];
        for (int i = 0; i < pixels; i++)
            packed[i] = managedBytes[i * 4]
                      | (managedBytes[i * 4 + 1] << 8)
                      | (managedBytes[i * 4 + 2] << 16)
                      | (managedBytes[i * 4 + 3] << 24);
        using var bufA = accelerator.Allocate1D(packed);           // second bulk crossing, host -> device
        await accelerator.SynchronizeAsync();
        swA.Stop();

        // ── Path B: the zero-copy route the library already provides ──
        var swB = System.Diagnostics.Stopwatch.StartNew();
        using var bufB = accelerator.Allocate1D<int>(pixels);
        // ⚠️ .Buffer - the UNDERLYING MemoryBuffer implements IBrowserMemoryBuffer, not the
        // MemoryBuffer1D<T,TStride> wrapper. Testing the wrapper compiles and is ALWAYS false, so the first
        // run of this test skipped on every lane including WebGPU and reported nothing at all. A guard that
        // silently converts "I wired this wrong" into "unsupported here" is worse than no guard.
        Preprocessing.MediaInterop.UploadRgbaToDevice(src, bufB);  // JS -> GPU, .NET never sees a pixel
        await accelerator.SynchronizeAsync();
        swB.Stop();

        double msA = swA.Elapsed.TotalMilliseconds, msB = swB.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[PixelUp] {BackendName,-8} {w}x{h} RGBA ({bytes / 1024.0 / 1024.0:F2} MB)  "
                        + $"managed(ReadBytes+repack+upload)={msA:F1} ms   zero-copy(CopyFromJS)={msB:F1} ms   "
                        + $"{(msB > 0 ? msA / msB : 0):F1}x");

        // ── The bytes MUST be identical. A faster path that differs is not a faster path. ──
        var flagKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(PixelMismatchFlagImpl);
        using var flags = accelerator.Allocate1D<float>(pixels);
        flagKernel(pixels, bufA.View, bufB.View, flags.View);
        await accelerator.SynchronizeAsync();

        var ew = GetOrCreateEW(accelerator);
        var (nan, mismatches, _) = await ew.FiniteCheckOnGpuAsync(flags.View, pixels);
        if (nan != 0 || mismatches != 0f)
            throw new Exception(
                $"{BackendName}: zero-copy upload does not match the managed upload - {mismatches} of {pixels} "
              + $"pixels differ (nan={nan}). The bytes must be bit-identical; a mismatch means CopyFromJS "
              + "interprets the typed array differently, not merely that it is faster.");
    });
}
