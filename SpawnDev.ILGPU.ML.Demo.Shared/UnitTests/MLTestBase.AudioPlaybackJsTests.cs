using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// <see cref="AudioPlayback.PlayAsync(Float32Array, int)"/> must accept JS PCM without a managed lap.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 60000)]
    public async Task AudioPlayback_Float32Array_PlayDoesNotThrow() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("browser-only: AudioPlayback needs a JS AudioContext");

        // Short click (~50 ms @ 24 kHz) — fixture only, not heard in CI meaningfully.
        int n = 1200;
        var managed = new float[n];
        for (int i = 0; i < n; i++)
            managed[i] = 0.1f * MathF.Sin(2f * MathF.PI * 440f * i / 24000f);

        using var jsPcm = new Float32Array(n);
        jsPcm.Set(managed);

        using var player = new AudioPlayback(js);
        double seconds = await player.PlayAsync(jsPcm, 24000);
        if (seconds <= 0 || double.IsNaN(seconds))
            throw new Exception($"PlayAsync(Float32Array) returned non-positive duration {seconds}");
        player.Stop();
        Console.WriteLine($"[AudioPlay] {BackendName}: PlayAsync(Float32Array) ok, {seconds:F3}s");
        await Task.CompletedTask;
    });
}
