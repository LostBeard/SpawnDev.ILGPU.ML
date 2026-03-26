using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Tensors;
using System.Diagnostics;

namespace SpawnDev.ILGPU.ML.Pipelines;

/// <summary>
/// Unconditional image generation via DDPM (Denoising Diffusion Probabilistic Models).
/// Generates images from pure noise through iterative denoising.
///
/// Usage:
///   var pipeline = new DiffusionPipeline(unetSession, accelerator, channels: 1, size: 28);
///   var result = await pipeline.GenerateAsync(numSteps: 50);
///   // result.ImagePixels is the generated RGBA image
/// </summary>
public class DiffusionPipeline : IDisposable
{
    private readonly InferenceSession _unetSession;
    private readonly Accelerator _accelerator;
    private readonly int _channels;
    private readonly int _size;

    public DiffusionPipeline(InferenceSession unetSession, Accelerator accelerator,
        int channels = 1, int size = 28)
    {
        _unetSession = unetSession;
        _accelerator = accelerator;
        _channels = channels;
        _size = size;
    }

    /// <summary>
    /// Generate an image from noise via iterative DDPM denoising.
    /// </summary>
    public async Task<DiffusionResult> GenerateAsync(
        int numSteps = 50, int? seed = null,
        Action<int, int>? onStep = null)
    {
        var sw = Stopwatch.StartNew();
        var rng = new Random(seed ?? Environment.TickCount);
        int pixels = _channels * _size * _size;

        // Start with random Gaussian noise
        var sample = new float[pixels];
        for (int i = 0; i < pixels; i++)
        {
            // Box-Muller transform for Gaussian noise
            float u1 = (float)rng.NextDouble();
            float u2 = (float)rng.NextDouble();
            sample[i] = MathF.Sqrt(-2f * MathF.Log(u1 + 1e-10f)) * MathF.Cos(2f * MathF.PI * u2);
        }

        // Linear beta schedule
        float betaStart = 0.0001f, betaEnd = 0.02f;
        var betas = new float[numSteps];
        for (int t = 0; t < numSteps; t++)
            betas[t] = betaStart + (betaEnd - betaStart) * t / (numSteps - 1);

        // Precompute alphas
        var alphas = new float[numSteps];
        var alphasCumprod = new float[numSteps];
        alphas[0] = 1f - betas[0];
        alphasCumprod[0] = alphas[0];
        for (int t = 1; t < numSteps; t++)
        {
            alphas[t] = 1f - betas[t];
            alphasCumprod[t] = alphasCumprod[t - 1] * alphas[t];
        }

        // Reverse diffusion: denoise from t=T-1 to t=0
        for (int step = numSteps - 1; step >= 0; step--)
        {
            onStep?.Invoke(numSteps - 1 - step, numSteps);

            // Upload sample to GPU
            using var sampleBuf = _accelerator.Allocate1D(sample);
            var sampleTensor = new Tensor(sampleBuf.View, new[] { 1, _channels, _size, _size });

            // Timestep as float (our engine uses float tensors)
            using var tBuf = _accelerator.Allocate1D(new float[] { step });
            var tTensor = new Tensor(tBuf.View, new[] { 1 });

            // Predict noise
            var inputs = new Dictionary<string, Tensor>
            {
                [_unetSession.InputNames[0]] = sampleTensor,
                [_unetSession.InputNames[1]] = tTensor,
            };

            var outputs = await _unetSession.RunAsync(inputs);
            var predicted = outputs[_unetSession.OutputNames[0]];

            // Read predicted noise
            using var readBuf = _accelerator.Allocate1D<float>(pixels);
            new ElementWiseKernels(_accelerator).Scale(
                predicted.Data.SubView(0, pixels), readBuf.View, pixels, 1f);
            await _accelerator.SynchronizeAsync();
            var noise = await readBuf.CopyToHostAsync<float>(0, pixels);

            // DDPM reverse step
            float alpha = alphas[step];
            float alphaCumprod = alphasCumprod[step];
            float alphaCumprodPrev = step > 0 ? alphasCumprod[step - 1] : 1f;
            float beta = betas[step];

            float coeff1 = 1f / MathF.Sqrt(alpha);
            float coeff2 = beta / MathF.Sqrt(1f - alphaCumprod);

            for (int i = 0; i < pixels; i++)
                sample[i] = coeff1 * (sample[i] - coeff2 * noise[i]);

            // Add noise (except at t=0)
            if (step > 0)
            {
                float sigma = MathF.Sqrt(beta);
                for (int i = 0; i < pixels; i++)
                {
                    float u1 = (float)rng.NextDouble();
                    float u2 = (float)rng.NextDouble();
                    float z = MathF.Sqrt(-2f * MathF.Log(u1 + 1e-10f)) * MathF.Cos(2f * MathF.PI * u2);
                    sample[i] += sigma * z;
                }
            }
        }

        sw.Stop();

        // Convert to RGBA pixels
        var rgbaPixels = new int[_size * _size];
        for (int i = 0; i < _size * _size; i++)
        {
            // Normalize from model output range to [0, 255]
            float v = (sample[i] + 1f) / 2f * 255f; // [-1,1] → [0,255]
            int gray = Math.Clamp((int)(v + 0.5f), 0, 255);
            rgbaPixels[i] = gray | (gray << 8) | (gray << 16) | (0xFF << 24);
        }

        return new DiffusionResult
        {
            ImagePixels = rgbaPixels,
            Width = _size,
            Height = _size,
            NumSteps = numSteps,
            Seed = seed ?? -1,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
        };
    }

    public void Dispose() => _unetSession?.Dispose();
}

/// <summary>Result from diffusion image generation.</summary>
public class DiffusionResult
{
    public int[] ImagePixels { get; init; } = Array.Empty<int>();
    public int Width { get; init; }
    public int Height { get; init; }
    public int NumSteps { get; init; }
    public int Seed { get; init; }
    public double InferenceTimeMs { get; init; }
}
