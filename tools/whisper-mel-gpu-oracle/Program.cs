// Quick host oracle: CPU AudioPreprocessor.ComputeLogMelSpectrogram vs WhisperMelPreprocessor on CUDA/CPU.
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using SpawnDev.ILGPU.ML.Preprocessing;

static float[] Sine(int n, float hz)
{
    var a = new float[n];
    for (int i = 0; i < n; i++)
        a[i] = 0.5f * MathF.Sin(2f * MathF.PI * hz * i / 16000f);
    return a;
}

static float[] Speech(int n)
{
    var a = new float[n];
    var rng = new Random(4242);
    for (int i = 0; i < n; i++)
    {
        double t = i / 16000.0;
        double v = 0;
        for (int h = 1; h <= 12; h++) v += Math.Sin(2 * Math.PI * 110 * h * t) / h;
        a[i] = (float)(0.3 * v + 0.02 * (rng.NextDouble() - 0.5));
    }
    return a;
}

static void Compare(Accelerator acc, WhisperMelPreprocessor gpu, string name, float[] pcm, float tol)
{
    var expected = AudioPreprocessor.ComputeLogMelSpectrogram(pcm);
    using var pcmBuf = acc.Allocate1D(pcm.Length == 0 ? new float[1] : pcm);
    using var melBuf = acc.Allocate1D<float>(WhisperMelPreprocessor.MelLength);
    gpu.ComputeLogMelSpectrogram(pcmBuf.View, pcm.Length, melBuf.View);
    acc.Synchronize();
    var got = melBuf.GetAsArray1D();

    float maxAbs = 0, sumAbs = 0;
    int worst = -1, over = 0;
    for (int i = 0; i < expected.Length; i++)
    {
        float d = MathF.Abs(expected[i] - got[i]);
        sumAbs += d;
        if (d > maxAbs) { maxAbs = d; worst = i; }
        if (d > tol) over++;
    }
    bool ok = maxAbs <= tol;
    Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {name}: maxAbs={maxAbs:E3} meanAbs={sumAbs / expected.Length:E3} "
        + $"overTol={over} at[{worst}] cpu={expected[worst]:F6} gpu={got[worst]:F6} (tol={tol:E1})");
    if (!ok) Environment.ExitCode = 1;
}

using var context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
Accelerator acc;
try { acc = context.CreateCudaAccelerator(0); Console.WriteLine($"device: CUDA {acc.Name}"); }
catch { acc = context.CreateCPUAccelerator(0); Console.WriteLine("device: CPU (no CUDA)"); }

using (acc)
using (var gpu = new WhisperMelPreprocessor(acc))
{
    // Shipping CPU uses Cooley-Tukey; GPU uses direct DFT. Documented gap peaks ~1.9e-3 on pure tones.
    const float shippingTol = 2.5e-3f;
    Compare(acc, gpu, "silence", new float[16000], shippingTol);
    Compare(acc, gpu, "sine_440", Sine(16000, 440f), shippingTol);
    Compare(acc, gpu, "speech_1_2s", Speech(19200), shippingTol);
}
