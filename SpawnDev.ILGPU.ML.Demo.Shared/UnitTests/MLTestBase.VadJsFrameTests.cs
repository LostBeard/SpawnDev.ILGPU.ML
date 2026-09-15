using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// The VAD's JS-typed-array frame path must give the SAME speech probability as the managed one.
/// </summary>
/// <remarks>
/// 🔴 WHY. A microphone frame arrives as a WebCodecs <c>AudioData</c> - already a JS typed array. The only
/// entry point the VAD had was <c>ProcessFrameAsync(float[])</c>, so every frame went
///
///     AudioData -> Float32Array -> .ToArray() -> float[] -> CopyFromCPU -> GPU
///
/// out of JS and straight back to the GPU, with nothing in between that needed .NET. TJ, 2026-09-15:
/// *"it always matters when taking the less performant path for no reason."* 512 samples is well past the
/// ~64-element metadata exemption, and "it is only 2 KB" is not a reason - I tried that reasoning and it
/// was wrong.
///
/// ⚠️ THE ASSERTION IS THE PROBABILITY, NOT THE TIMING. A faster frame path that changes what the detector
/// answers is not a faster frame path, it is a broken one - and the VAD is stateful (h/c), so a wrong
/// upload would drift rather than fail loudly. Both frames run from the SAME reset state and must agree
/// exactly; the samples are identical bytes, so there is no tolerance to spend.
///
/// Timings print for information (<c>PMT_CONSOLE_LOG=VadFrame</c>) and are not gated - one frame on a shared
/// browser device is far too noisy to fail a build on.
/// </remarks>
public abstract partial class MLTestBase
{
    [TestMethod(Timeout = 300000)]
    public async Task Vad_JsFramePath_MatchesManagedFrame() => await RunTest(async accelerator =>
    {
        var js = SpawnJSRuntime.Instance;
        if (js == null || !js.IsBrowser)
            throw new UnsupportedTestException("browser-only: there is no JS heap to cross on a desktop lane");

        var assets = GetHttpClient();
        if (assets == null) throw new UnsupportedTestException("HttpClient not available");
        var modelBytes = await assets.GetByteArrayAsync("references/vad/silero_vad.onnx");

        using var vad = SileroVad.Create(accelerator, modelBytes);

        // A frame with real structure - a silent or constant frame would let a broken upload agree by
        // accident. Speech-like: a couple of harmonics plus noise, in [-1, 1].
        int n = SileroVad.WindowSize;
        var managed = new float[n];
        var rng = new Random(9152);
        for (int i = 0; i < n; i++)
            managed[i] = 0.45f * MathF.Sin(2f * MathF.PI * 140f * i / 16000f)
                       + 0.25f * MathF.Sin(2f * MathF.PI * 310f * i / 16000f)
                       + 0.05f * (float)(rng.NextDouble() * 2 - 1);

        using var jsFrame = new Float32Array(n);
        jsFrame.Set(managed);   // fixture only - outside both timed regions

        // 🔴 WARM UP BOTH PATHS BEFORE TIMING EITHER. The first run of this test timed one frame per path
        // in order and printed managed=2437 ms vs js=7.36 ms - a 331x that is pure fiction: the FIRST call
        // carries the model load, the kernel compilation and graph capture's warm/probe/record passes, and
        // whichever path runs first eats all of it. A number that flatters the change while measuring
        // something else is worse than no number. Same reason INORMBENCH discards its first rep.
        const int warm = 12, reps = 20;
        for (int i = 0; i < warm; i++)
        {
            await vad.ProcessFrameAsync(managed);
            await vad.ProcessFrameAsync(jsFrame);
        }

        // Managed path, from a known state, averaged.
        vad.Reset();
        float pManaged = await vad.ProcessFrameAsync(managed);
        var swManaged = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) await vad.ProcessFrameAsync(managed);
        swManaged.Stop();

        // JS path, from the SAME state - Reset makes the two probabilities comparable, which matters because
        // the model is stateful through h/c and the second frame would otherwise see the first one's state.
        vad.Reset();
        float pJs = await vad.ProcessFrameAsync(jsFrame);
        var swJs = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) await vad.ProcessFrameAsync(jsFrame);
        swJs.Stop();

        double msManaged = swManaged.Elapsed.TotalMilliseconds / reps;
        double msJs = swJs.Elapsed.TotalMilliseconds / reps;
        Console.WriteLine($"[VadFrame] {BackendName,-8} p(managed)={pManaged:F6} p(js)={pJs:F6}  "
                        + $"managed={msManaged:F2} ms/frame  js={msJs:F2} ms/frame  "
                        + $"(warm={warm}, reps={reps}, load+compile excluded)");

        // Same bytes, same state, same graph - so the answers must be equal, not merely close.
        if (pManaged != pJs)
            throw new Exception(
                $"{BackendName}: the JS frame path gave a different speech probability than the managed one "
              + $"({pJs:F8} vs {pManaged:F8}). Identical samples uploaded two ways from the same reset state "
              + "must produce identical output; a difference means CopyFromJS landed the samples differently, "
              + "which on a stateful detector drifts quietly instead of failing.");

        // And the detector must actually be responding to this input - comparing two flat answers would
        // prove nothing about either path.
        if (pManaged <= 0f || pManaged >= 1f || float.IsNaN(pManaged))
            throw new Exception(
                $"{BackendName}: speech probability {pManaged} is not a usable value, so the equality check "
              + "above compared two degenerate outputs and proved nothing.");
    });
}
