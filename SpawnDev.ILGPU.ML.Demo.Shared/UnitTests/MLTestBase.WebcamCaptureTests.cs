using System;
using System.Threading.Tasks;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

public abstract partial class MLTestBase
{
    /// <summary>
    /// <see cref="MediaStreamCapture.StartWebcamAsync"/> opens the camera (PMT's Chromium runs with Chromium's fake capture device), waits for the video to PLAY
    /// (it awaits the play() promise; a fixed 100 ms delay used to stand in for it, so a rejected play was an unobserved exception) and delivers frames of the
    /// requested size with picture content. Browser lanes only.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Webcam_StartWebcam_DeliversFrames()
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser) throw new UnsupportedTestException("the camera is a browser API (not a browser lane)");
        using var capture = new MediaStreamCapture(js);
        var first = new TaskCompletionSource<(int W, int H, int NonZero)>(TaskCreationOptions.RunContinuationsAsynchronously);
        capture.OnFrameReadyJs += (rgba, w, h) =>
        {
            if (first.Task.IsCompleted) return;
            // count non-zero bytes of the first row only (a tiny read: the frame itself stays in JS)
            using var row = rgba.ReCast<SpawnDev.SpawnJS.JSObjects.Uint8Array>(0, Math.Min(w * 4L, rgba.ByteLength));
            var bytes = row.ReadBytes();
            int nz = 0; foreach (var b in bytes) if (b != 0) nz++;
            first.TrySetResult((w, h, nz));
        };
        bool started = await capture.StartWebcamAsync(320, 240);
        if (!started) throw new Exception($"StartWebcamAsync returned false with Chromium's fake camera: {capture.LastVideoError?.GetType().Name}: {capture.LastVideoError?.Message}");
        var done = await Task.WhenAny(first.Task, Task.Delay(10000));
        capture.Stop();
        if (done != first.Task) throw new Exception("no frame within 10 s of a successful start");
        var (w, h, nonZero) = first.Task.Result;
        if (w != 320 || h != 240 || nonZero == 0) throw new Exception($"frame {w}x{h}, {nonZero} non-zero bytes in the first row (want 320x240 with picture content)");
    }
}
