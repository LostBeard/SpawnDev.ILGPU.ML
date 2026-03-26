using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.ML.Demo.Shared.UnitTests;

/// <summary>
/// Tests for DDPM diffusion pipeline components.
/// </summary>
public abstract partial class MLTestBase
{
    [TestMethod]
    public async Task Diffusion_BetaSchedule_LinearRange()
    {
        await Task.CompletedTask;

        // Verify linear beta schedule produces correct range
        int numSteps = 50;
        float betaStart = 0.0001f, betaEnd = 0.02f;
        var betas = new float[numSteps];
        for (int t = 0; t < numSteps; t++)
            betas[t] = betaStart + (betaEnd - betaStart) * t / (numSteps - 1);

        if (MathF.Abs(betas[0] - betaStart) > 1e-6f)
            throw new Exception($"Beta[0]={betas[0]}, expected {betaStart}");
        if (MathF.Abs(betas[numSteps - 1] - betaEnd) > 1e-6f)
            throw new Exception($"Beta[last]={betas[numSteps - 1]}, expected {betaEnd}");

        // All betas should be positive and increasing
        for (int i = 1; i < numSteps; i++)
        {
            if (betas[i] <= betas[i - 1])
                throw new Exception($"Betas not increasing: [{i - 1}]={betas[i - 1]} >= [{i}]={betas[i]}");
        }

        // Alpha cumprod should be decreasing from ~1 to ~0
        var alphasCumprod = new float[numSteps];
        alphasCumprod[0] = 1f - betas[0];
        for (int t = 1; t < numSteps; t++)
            alphasCumprod[t] = alphasCumprod[t - 1] * (1f - betas[t]);

        if (alphasCumprod[0] < 0.99f)
            throw new Exception($"AlphasCumprod[0]={alphasCumprod[0]}, expected ~1.0");
        if (alphasCumprod[numSteps - 1] > 0.5f)
            throw new Exception($"AlphasCumprod[last]={alphasCumprod[numSteps - 1]}, expected < 0.5");

        Console.WriteLine($"[Diffusion] Beta schedule: [{betaStart}..{betaEnd}], alphasCumprod [{alphasCumprod[0]:F4}..{alphasCumprod[numSteps - 1]:F4}]");
    }

    [TestMethod]
    public async Task Diffusion_GaussianNoise_Statistics()
    {
        await Task.CompletedTask;

        // Verify Box-Muller generates proper Gaussian noise
        var rng = new Random(42);
        int n = 10000;
        var samples = new float[n];
        for (int i = 0; i < n; i++)
        {
            float u1 = (float)rng.NextDouble();
            float u2 = (float)rng.NextDouble();
            samples[i] = MathF.Sqrt(-2f * MathF.Log(u1 + 1e-10f)) * MathF.Cos(2f * MathF.PI * u2);
        }

        float mean = samples.Average();
        float variance = samples.Select(s => (s - mean) * (s - mean)).Average();

        // Mean should be ~0, variance should be ~1
        if (MathF.Abs(mean) > 0.1f)
            throw new Exception($"Gaussian mean={mean:F4}, expected ~0");
        if (MathF.Abs(variance - 1f) > 0.2f)
            throw new Exception($"Gaussian variance={variance:F4}, expected ~1");

        Console.WriteLine($"[Diffusion] Gaussian noise: mean={mean:F4}, variance={variance:F4}");
    }
}
