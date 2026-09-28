using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <c>MediaInterop.DecodeToDeviceAsync</c>: an ENCODED image (a Blob / File) decoded and resized by the browser and
/// uploaded to the GPU without the pixels entering the .NET heap.
/// </summary>
/// <remarks>
/// Written 2026-09-28 after a 35-photo SpawnScene project ran the WASM heap out of memory on gh-pages: the only
/// encoded-image entry point, <c>MediaInterop.FromBlobAsync</c>, returned a managed <c>byte[]</c>, so importers
/// copied every full-size photo into .NET. TJ: "fucking around with data in .net wasm that does not need to be there".
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>An encoded PNG with four solid quadrants (PNG is lossless, so decoded colours are exact).</summary>
    private static async Task<Blob> QuadrantPngAsync(int w, int h)
    {
        using var canvas = new OffscreenCanvas(w, h);
        using var ctx = canvas.Get2DContext();
        ctx.FillStyle = "rgb(255,0,0)"; ctx.FillRect(0, 0, w / 2, h / 2);
        ctx.FillStyle = "rgb(0,255,0)"; ctx.FillRect(w / 2, 0, w - w / 2, h / 2);
        ctx.FillStyle = "rgb(0,0,255)"; ctx.FillRect(0, h / 2, w / 2, h - h / 2);
        ctx.FillStyle = "rgb(250,200,40)"; ctx.FillRect(w / 2, h / 2, w - w / 2, h - h / 2);
        return await canvas.ConvertToBlob(new ConvertToBlobOptions { Type = "image/png" });
    }

    private static readonly int[] QuadrantRgba =
    {
        255 | (0 << 8) | (0 << 16) | (255 << 24),
        0 | (255 << 8) | (0 << 16) | (255 << 24),
        0 | (0 << 8) | (255 << 16) | (255 << 24),
        250 | (200 << 8) | (40 << 16) | (255 << 24),
    };

    [TestMethod(Timeout = 120000)]
    public async Task BlobDecode_ToDevice_ExactPixelsAndResize() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("browser-only: decodes through createImageBitmap + OffscreenCanvas");
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.Cuda or AcceleratorType.OpenCL)
            throw new UnsupportedTestException("browser accelerators only (UploadToDevice)");

        using var png = await QuadrantPngAsync(200, 100);
        foreach (var (maxEdge, wantW, wantH) in new[] { (0, 200, 100), (1024, 200, 100), (50, 50, 25) })
        {
            var (rgba, w, h, sw, sh) = await Preprocessing.MediaInterop.DecodeToDeviceAsync(png, accelerator, maxEdge);
            using (rgba)
            {
                if (w != wantW || h != wantH || sw != 200 || sh != 100)
                    throw new Exception($"maxLongEdge {maxEdge}: decoded {w}x{h} from {sw}x{sh}, expected {wantW}x{wantH} from 200x100");
                var got = await rgba.CopyToHostAsync<int>();   // test oracle only
                var centres = new[] { (w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4), (3 * w / 4, 3 * h / 4) };
                for (int q = 0; q < 4; q++)
                {
                    var (x, y) = centres[q];
                    int px = got[y * w + x];
                    if (px != QuadrantRgba[q])
                        throw new Exception($"maxLongEdge {maxEdge}: quadrant {q} centre ({x},{y}) = 0x{px:X8}, expected 0x{QuadrantRgba[q]:X8} (packed RGBA, R low byte)");
                }
            }
        }
    });

    [TestMethod(Timeout = 180000)]
    public async Task BlobDecode_ToDevice_PixelsNeverEnterManagedHeap() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("browser-only: there is no JS heap to keep pixels in on a desktop lane");
        if (accelerator.AcceleratorType is AcceleratorType.CPU or AcceleratorType.Cuda or AcceleratorType.OpenCL)
            throw new UnsupportedTestException("browser accelerators only (UploadToDevice)");

        // A 12 MP phone-photo-sized image: 4000x3000 RGBA is 48 MB.
        const int w = 4000, h = 3000;
        const long frameBytes = (long)w * h * 4;
        using var png = await QuadrantPngAsync(w, h);

        long before = GC.GetTotalMemory(forceFullCollection: true);
        long peakNew = 0;
        var (rgba, dw, dh, _, _) = await Preprocessing.MediaInterop.DecodeToDeviceAsync(png, accelerator, 0);
        using (rgba)
        {
            await accelerator.SynchronizeAsync();
            peakNew = GC.GetTotalMemory(forceFullCollection: false) - before;
            if (dw != w || dh != h) throw new Exception($"decoded {dw}x{dh}, expected {w}x{h}");
        }

        // Red check on the same image: the managed path this replaces copies the whole frame into .NET.
        var interop = new Preprocessing.MediaInterop(js);
        long beforeOld = GC.GetTotalMemory(forceFullCollection: true);
        var (managed, _, _) = await interop.FromBlobAsync(png);
        long peakOld = GC.GetTotalMemory(forceFullCollection: false) - beforeOld;
        GC.KeepAlive(managed);

        Console.WriteLine($"[BlobDecode] {BackendName} {w}x{h} ({frameBytes / (1024 * 1024)} MB RGBA): managed heap grew " +
                          $"{peakNew / 1024.0 / 1024.0:F2} MB with DecodeToDeviceAsync, {peakOld / 1024.0 / 1024.0:F2} MB with FromBlobAsync");
        if (peakNew > frameBytes / 16)
            throw new Exception($"DecodeToDeviceAsync grew the managed heap by {peakNew:N0} bytes for a {frameBytes:N0}-byte frame - pixels crossed into .NET");
        if (peakOld < frameBytes / 2)
            throw new Exception($"red check broken: FromBlobAsync grew the heap by only {peakOld:N0} bytes, so this test cannot see a copy");
    });
}
