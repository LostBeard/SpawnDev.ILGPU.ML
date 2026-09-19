using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// GPU Whisper log-mel must match <see cref="AudioPreprocessor.ComputeLogMelSpectrogram"/>.
/// </summary>
/// <remarks>
/// ⚠️ This is the gate for zero-copy PCM: until GPU mel agrees with the CPU oracle, the pipeline
/// cannot drop <c>ToArray()</c> on the browser path. Fast backends first (CUDA / OpenCL / WebGPU /
/// CPU); WebGL and Wasm are skipped for the full 80×3000 direct-DFT — same standing rule as other
/// heavy preprocess oracles (do not gate ML progress on the slow lanes).
/// <para>
/// Tolerance is absolute on the final Whisper-normalized mel. Direct DFT vs the CPU's Cooley-Tukey
/// (400 = 16×25) is mathematically the same transform but not bit-identical; documented ULP drift
/// lives in the assert message if it ever grows.
/// </para>
/// </remarks>
public abstract partial class MLTestBase
{
    /// <summary>Absolute tolerance on Whisper-normalized log-mel (roughly [-1, 1] after norm).</summary>
    /// <remarks>
    /// GPU uses a direct real DFT; the shipping CPU path uses Cooley-Tukey (400 = 16×25). Measured on
    /// CUDA: silence is ~2e-7; speech ~3e-4; pure tones peak ~1.9e-3. STFT power vs a CPU direct-DFT
    /// reference is ~1.5e-6 relative — the gap to shipping mel is algorithm association, not a framing bug.
    /// </remarks>
    private const float WhisperMelGpuTol = 2.5e-3f;

    [TestMethod(Timeout = 600000)]
    public async Task WhisperMel_Gpu_MatchesCpuOracle() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType is AcceleratorType.WebGL or AcceleratorType.Wasm)
            throw new UnsupportedTestException(
                "full 80×3000 direct-DFT Whisper mel is gated to fast backends (CUDA/OpenCL/WebGPU/CPU); "
                + "WebGL/Wasm stay on the CPU mel path until a faster STFT lands");

        using var melGpu = new WhisperMelPreprocessor(accelerator);

        // ── Fixtures: silence, pure tone, speech-like broadband ──
        await AssertMelFixture(accelerator, melGpu, Silence(16000), "silence");
        await AssertMelFixture(accelerator, melGpu, Sine(16000, 440f), "sine_440");
        await AssertMelFixture(accelerator, melGpu, SpeechLike(19200), "speech_1_2s");

        Console.WriteLine($"[WhisperMel] GPU matches CPU oracle (tol={WhisperMelGpuTol:E1}) on {accelerator.AcceleratorType}");
    });

    private static async Task AssertMelFixture(
        Accelerator accelerator,
        WhisperMelPreprocessor melGpu,
        float[] pcm,
        string name)
    {
        var expected = AudioPreprocessor.ComputeLogMelSpectrogram(pcm);
        if (expected.Length != WhisperMelPreprocessor.MelLength)
            throw new Exception($"{name}: CPU mel length {expected.Length}, expected {WhisperMelPreprocessor.MelLength}");

        using var pcmBuf = accelerator.Allocate1D(pcm.Length == 0 ? new float[1] : pcm);
        using var melBuf = accelerator.Allocate1D<float>(WhisperMelPreprocessor.MelLength);
        melGpu.ComputeLogMelSpectrogram(pcmBuf.View, pcm.Length, melBuf.View);

        // Drain before compare — WebGPU batches; AssertCloseGpu uploads expected and compares on device.
        await accelerator.SynchronizeAsync();
        await AssertCloseGpu(accelerator, melBuf.View, expected, WhisperMelGpuTol, $"WhisperMel[{name}] ");
    }

    private static float[] Silence(int n) => new float[n];

    private static float[] Sine(int n, float hz, int rate = 16000)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = 0.5f * MathF.Sin(2f * MathF.PI * hz * i / rate);
        return a;
    }

    private static float[] SpeechLike(int n, int rate = 16000)
    {
        var a = new float[n];
        var rng = new Random(4242);
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)rate;
            double v = 0;
            for (int h = 1; h <= 12; h++)
                v += Math.Sin(2 * Math.PI * 110 * h * t) / h;
            a[i] = (float)(0.3 * v + 0.02 * (rng.NextDouble() - 0.5));
        }
        return a;
    }
}
